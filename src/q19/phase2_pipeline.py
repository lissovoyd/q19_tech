"""Phase 2 -- clustering: the single file that runs the chosen path end-to-end
(KMeans, k=17) AND keeps every alternative we built and benchmarked sitting
right next to it, wired up but switched off, with a comment on *why* it's off
and *when* flipping it back on would be the right call. Same convention as
phase1_pipeline.py. Nothing below was deleted because it "didn't work" --
every function here was actually run and measured (see outputs/PHASE2_CLUSTERING.md for quality + the UMAP+KMeans k-sweep,
outputs/PHASE2_SCALING.md for time and memory) and the choice was made on
that evidence, not on omission.

Frozen path: KMeans, k=17. Chosen because, in evidence gathered before any
freeze:
  - silhouette ties with every other family in the corpus's ~0.17-0.18
    plateau (0.171 @ k=17, best 0.179 @ k=15) -- PHASE2_CLUSTERING.md
  - full coverage by construction: every query gets a real cluster, never
    "noise" -- unlike the HDBSCAN-based families
  - cheapest of every family tested, and the only one with NO observed
    scaling wall through n=32,000 (27s / 707MB) -- PHASE2_SCALING.md
  - Agglomerative/Ward was the *original* frozen choice (tied on quality,
    deterministic, gives a free merge hierarchy) until PHASE2_SCALING.md
    showed its O(n^2) memory growth is a real production wall: 2.3GB at
    n=16,000, outright timeout at n=32,000. Quality parity wasn't enough
    once cost entered the picture.

Run: python -m q19.phase2_pipeline
"""
from __future__ import annotations

import numpy as np

from q19.cluster import (
    agglomerative_cluster, bertopic_cluster, hdbscan_cluster, kmeans_cluster, umap_reduce,
)

K = 17
UMAP_N_COMPONENTS = 10
MIN_CLUSTER_SIZE = 8  # the real-data config that lands on 17 clusters/topics, see PHASE2_CLUSTERING.md


# ============================================================================
# CLUSTERING -- how queries get grouped, and whether every query gets a label
# ============================================================================

def cluster_kmeans(embeddings: np.ndarray, texts: list[str] | None = None) -> np.ndarray:
    """ACTIVE. Centroid-based, k fixed up front. O(n*d) memory -- never
    materializes a pairwise distance matrix, just n points against k
    centroids each iteration -- which is exactly why it has no scaling wall
    (PHASE2_SCALING.md: 707MB at n=32,000 vs Agglomerative's 2.3GB at
    n=16,000 and timeout at n=32,000). Forces every query into its nearest
    centroid, even a genuine outlier -- the tradeoff for never leaving a
    query unlabeled.
    """
    labels, _ = kmeans_cluster(embeddings, K)
    return labels


def cluster_agglomerative(embeddings: np.ndarray, texts: list[str] | None = None) -> np.ndarray:
    """DISABLED in the frozen path. Ward linkage -- deterministic (no
    random-init lottery, unlike KMeans) and produces a full merge hierarchy
    for free, which is genuinely useful if you want coarser/finer cuts
    in one run. Was the original frozen choice; ties KMeans on every
    quality metric (PHASE2_CLUSTERING.md).

    Turn this on when: n is bounded well under ~10,000 (PHASE2_SCALING.md
    shows its memory tripling from n=8,000 to n=16,000) AND you actually
    want the merge hierarchy -- e.g. interactively picking cut granularity
    rather than committing to one k.
    """
    return agglomerative_cluster(embeddings, K)


def cluster_umap_hdbscan(embeddings: np.ndarray, texts: list[str] | None = None,
                          allow_other_bucket: bool = False) -> np.ndarray:
    """DISABLED in the frozen path. UMAP-reduce, then density-based HDBSCAN --
    discovers its own cluster count and can refuse to assign a point at all
    (label -1, "noise"). Ties KMeans/Agglomerative on quality when scored
    fairly (PHASE2_CLUSTERING.md: 0.174-0.181 at high-coverage configs) and
    has the same scaling profile as KMeans, no wall through n=32,000
    (PHASE2_SCALING.md: 1,001MB / 65.6s). By default raises rather than
    silently returning unlabeled (-1) points -- pass allow_other_bucket=True
    to fold them into an explicit extra cluster instead (see
    `cluster_bertopic` below for the full reasoning on that tradeoff; this
    function and that one share the same UMAP+HDBSCAN core, just without
    BERTopic's c-TF-IDF auto-labels on top).

    Turn this on when: see `cluster_bertopic`'s docstring -- same call to
    make, since they're the same underlying clustering decision.
    """
    reduced = umap_reduce(embeddings, n_components=UMAP_N_COMPONENTS)
    labels = hdbscan_cluster(reduced, min_cluster_size=MIN_CLUSTER_SIZE)
    if (labels == -1).any() and not allow_other_bucket:
        raise ValueError(
            f"{int((labels == -1).sum())} queries left unassigned (label -1). "
            "Pass allow_other_bucket=True to fold them into an explicit 'Other' "
            "cluster, or use cluster_kmeans if every query must land in one of "
            "the K named clusters."
        )
    return with_other_bucket(labels) if allow_other_bucket else labels


def cluster_bertopic(embeddings: np.ndarray, texts: list[str],
                      allow_other_bucket: bool = True) -> np.ndarray:
    """DISABLED in the frozen path. Functionally the same UMAP+HDBSCAN
    pipeline as `cluster_umap_hdbscan` above, just packaged by a library
    (BERTopic) with c-TF-IDF auto-labelling on top, which makes it the
    easier of the two to operate day-to-day (keyword labels come free
    instead of needing a separate LLM call per cluster, though this project's
    Phase 3 already does LLM labelling regardless, so that convenience is
    redundant here specifically). Ties KMeans on quality once its noise
    points are scored honestly instead of excluded (PHASE2_CLUSTERING.md
    silhouette 0.176 excl. noise -> 0.170 with noise folded into an explicit
    'Other' bucket, 9/400 points, ~2.3%). Scaling was never benchmarked
    (PHASE2_SCALING.md only covers the bare UMAP+HDBSCAN core) -- treat that
    as unproven, not as "presumed fine."

    The real decision this function embodies, regardless of library: do you
    want every query *forced* into one of K existing classes (KMeans,
    Agglomerative), or do you want clustering allowed to say "this doesn't
    fit anything I know" and park it in its own bucket? Forcing assignment
    is simpler and avoids an awkward "what does Other mean" conversation
    with the business, but it also means a genuine outlier silently drags
    on whichever real cluster's centroid it lands nearest to, which can
    quietly corrupt that cluster's label/summary in Phase 3. Allowing an
    Other bucket surfaces that ambiguity explicitly (e.g. "6% of this
    week's queries didn't match a known issue type") instead of hiding it.

    Turn this on when: the ops report should highlight queries that don't
    fit any known issue type as their own signal, rather than always
    forcing every query into one of K named clusters -- and when BERTopic's
    keyword auto-labels are themselves useful (e.g. no LLM labelling step
    available). Otherwise prefer `cluster_umap_hdbscan(allow_other_bucket=True)`
    for the same Other-bucket behavior without BERTopic's untested scaling
    and extra dependency.
    """
    labels, _ = bertopic_cluster(texts, embeddings, min_cluster_size=MIN_CLUSTER_SIZE)
    if (labels == -1).any() and not allow_other_bucket:
        raise ValueError(
            f"{int((labels == -1).sum())} queries left unassigned (label -1). "
            "Pass allow_other_bucket=True to fold them into an explicit 'Other' "
            "cluster, or use cluster_kmeans if every query must land in one of "
            "the K named clusters."
        )
    return with_other_bucket(labels) if allow_other_bucket else labels


def with_other_bucket(labels: np.ndarray) -> np.ndarray:
    """Relabel HDBSCAN/BERTopic noise (-1) into one real extra cluster
    (max_label + 1) instead of leaving those queries unassigned. Restores
    full coverage at zero extra clustering cost (it's a relabel, not a
    recompute) -- but be honest that this bucket is a grab-bag, not a real
    topic: internal_metrics scored on it will be lower than the
    noise-excluded numbers in PHASE2_CLUSTERING.md, because that heterogeneity
    is now being scored instead of given a free pass.
    """
    labels = labels.copy()
    other_id = int(labels.max()) + 1
    labels[labels == -1] = other_id
    return labels


# ============================================================================
# THE FROZEN PATH
# ============================================================================

def cluster_queries(embeddings: np.ndarray, texts: list[str], cluster_fn=cluster_kmeans) -> np.ndarray:
    """End-to-end: embeddings (+ texts, for the BERTopic path) -> cluster
    labels. `cluster_fn` defaults to the frozen, evaluated-once Phase 2
    choice -- every alternative above is a parameter swap, not a rewrite,
    same as phase1_pipeline.detect_duplicates's retrieve_fn/threshold_fn.
    """
    return cluster_fn(embeddings, texts)


def main():
    from q19.data import load_queries
    from q19.methods import embed_texts

    q = load_queries()
    texts = q["text"].tolist()
    embeddings = np.asarray(embed_texts(texts), dtype=np.float32)
    labels = cluster_queries(embeddings, texts)  # frozen path only -- see module docstring for why
    n_clusters = len(set(labels.tolist()))
    print(f"{n_clusters} clusters over {len(texts)} queries (KMeans, k={K})")


if __name__ == "__main__":
    main()
