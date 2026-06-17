"""Phase 2 -- group queries by issue type, on the frozen Phase 1 embeddings
(BAAI/bge-m3, normalized). Three algorithm families, deliberately different
in kind, not just parameterization:

  - KMeans          -- centroid-based, needs k up front, fast, assumes
                        roughly spherical/equal-size clusters
  - Agglomerative    -- hierarchical (Ward linkage), needs k up front, builds
                        a full merge tree so you get every coarser/finer
                        grouping "for free" from one run
  - HDBSCAN          -- density-based, does NOT take k -- it decides how many
                        clusters exist and can leave outliers unassigned
                        (label -1, "noise"). The contrast point: what does an
                        algorithm with no k knob think the natural structure
                        looks like?

bge-m3 embeddings are L2-normalized (sentence_transformers'
normalize_embeddings=True), so Euclidean distance is a monotonic function of
cosine similarity here -- KMeans/Ward/HDBSCAN's default Euclidean metric is
therefore consistent with the cosine similarity Phase 1 was built on, not a
different notion of "close."

Never touches eval_holdout/ -- that's cluster_eval.py's job, and only after a
k/algorithm is frozen.
"""
import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def kmeans_cluster(embeddings: np.ndarray, k: int, seed: int = 42) -> tuple[np.ndarray, float]:
    """Returns (labels, inertia). n_init=10 -- KMeans is sensitive to centroid
    init, so this re-runs from 10 random starts and keeps the best (lowest
    inertia) one rather than reporting a single lucky/unlucky draw."""
    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = model.fit_predict(embeddings)
    return labels, float(model.inertia_)


def agglomerative_cluster(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Ward linkage: merges the pair of clusters that minimizes the increase
    in total within-cluster variance at each step -- same objective KMeans
    optimizes, but built bottom-up and deterministic (no random init)."""
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    return model.fit_predict(embeddings)


def umap_reduce(embeddings: np.ndarray, n_components: int, n_neighbors: int = 15,
                 seed: int = 42) -> np.ndarray:
    """Project embeddings down to n_components before density-based clustering.
    metric="cosine" matches what Phase 1 built similarity on. Density-based
    methods (HDBSCAN) are the ones expected to benefit -- in high dimensions,
    pairwise distances concentrate (the gap between near and far neighbors
    shrinks relative to the average distance), which erodes the local-density
    estimate HDBSCAN relies on. KMeans/Ward aren't run on the reduced space
    here since they only need relative centroid distances, not density.
    """
    import umap
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                         min_dist=0.0, metric="cosine", random_state=seed)
    return reducer.fit_transform(embeddings)


def bertopic_cluster(texts: list[str], embeddings: np.ndarray, min_cluster_size: int = 5,
                      n_components: int = 5, seed: int = 42):
    """BERTopic = the same UMAP-then-HDBSCAN pipeline as umap_reduce +
    hdbscan_cluster, packaged with c-TF-IDF auto-labelling on top. Runs on our
    already-frozen Phase 1 embeddings (embedding_model=None, embeddings passed
    in directly) rather than re-embedding with BERTopic's own default model --
    keeps this an apples-to-apples clustering comparison, not a different
    embedding comparison. Returns (labels, topic_model) -- topic_model exposes
    .get_topic_info() / .get_topic(id) for a free keyword-based label preview.
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN as HDBSCANStandalone
    import umap

    umap_model = umap.UMAP(n_components=n_components, n_neighbors=15, min_dist=0.0,
                            metric="cosine", random_state=seed)
    hdbscan_model = HDBSCANStandalone(min_cluster_size=min_cluster_size, metric="euclidean",
                                       cluster_selection_method="eom", prediction_data=True)
    topic_model = BERTopic(embedding_model=None, umap_model=umap_model,
                            hdbscan_model=hdbscan_model, calculate_probabilities=False,
                            verbose=False)
    topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)
    return np.asarray(topics), topic_model


def hdbscan_cluster(embeddings: np.ndarray, min_cluster_size: int = 5) -> np.ndarray:
    """min_cluster_size is HDBSCAN's main knob (not k): the smallest group of
    points it's willing to call a cluster rather than noise. Smaller values
    -> more, finer clusters and less noise; larger values -> fewer, coarser
    clusters and more points discarded as noise. Swept across a few values
    below rather than picked once, same spirit as the KMeans/Ward k-sweep.
    """
    model = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    return model.fit_predict(embeddings)


def internal_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """Silhouette / Davies-Bouldin / Calinski-Harabasz -- all label-free
    (no gold involved), all computed on the embeddings + cluster assignment
    only. HDBSCAN's noise points (label -1) are excluded: they're explicitly
    "not in any cluster," so including them would corrupt cohesion/separation
    math that assumes every point belongs to exactly one cluster.

    Returns None for metrics that need >= 2 real clusters if there aren't any
    (e.g. HDBSCAN found 0 or 1 cluster at this min_cluster_size).
    """
    mask = labels != -1
    n_clusters = len(np.unique(labels[mask])) if mask.any() else 0
    out = {"n_clusters": n_clusters, "n_noise": int((~mask).sum())}
    if n_clusters < 2 or mask.sum() < 2:
        out.update(silhouette=None, davies_bouldin=None, calinski_harabasz=None)
        return out
    X, y = embeddings[mask], labels[mask]
    out.update(
        silhouette=float(silhouette_score(X, y)),
        davies_bouldin=float(davies_bouldin_score(X, y)),
        calinski_harabasz=float(calinski_harabasz_score(X, y)),
    )
    return out
