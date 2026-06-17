"""Approximate nearest-neighbor candidate generation (the "scale path").

At n=400, brute force is O(n^2) and cheap -- that's what methods.py uses.
This module is the alternative: an HNSW index (via faiss) that finds each
query's top-k neighbors in roughly O(n log n) instead of O(n^2), so the same
candidate-generation step can be swapped in without touching anything
downstream once n grows past the point where brute force stops being free.
Never touches eval_holdout/.
"""
import numpy as np
import pandas as pd
import faiss


def exact_topk_pairs(embeddings: np.ndarray, ids: list[str], k: int = 10) -> pd.DataFrame:
    """Brute-force top-k neighbors per query via faiss IndexFlatIP.
    Exhaustive (same O(n^2 * d) cost as methods.py's full matrix) but only
    materializes O(n*k) results instead of the full O(n^2) matrix -- the
    "ground truth" this experiment measures ANN recall against.
    """
    n, d = embeddings.shape
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))
    scores, idx = index.search(embeddings.astype(np.float32), k + 1)  # +1: query matches itself

    rows = []
    for i in range(n):
        for rank in range(k + 1):
            j = idx[i, rank]
            if j == i:
                continue
            rows.append((ids[i], ids[j], float(scores[i, rank])))
    return pd.DataFrame(rows, columns=["id_1", "id_2", "score"])


def build_hnsw_index(embeddings: np.ndarray, M: int = 16, ef_construction: int = 100) -> faiss.IndexHNSWFlat:
    d = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(embeddings.astype(np.float32))
    return index


def ann_candidate_pairs(
    embeddings: np.ndarray, ids: list[str], k: int = 10, ef_search: int = 64, M: int = 16
) -> pd.DataFrame:
    """Approximate top-k neighbors per query via an HNSW graph index.
    Same shape of output as exact_topk_pairs, but found via greedy graph
    search instead of exhaustively scoring every other point.
    """
    index = build_hnsw_index(embeddings, M=M)
    index.hnsw.efSearch = ef_search
    n = embeddings.shape[0]
    scores, idx = index.search(embeddings.astype(np.float32), k + 1)

    rows = []
    for i in range(n):
        for rank in range(k + 1):
            j = idx[i, rank]
            if j == i or j == -1:
                continue
            rows.append((ids[i], ids[j], float(scores[i, rank])))
    return pd.DataFrame(rows, columns=["id_1", "id_2", "score"])


def ivf_candidate_pairs(
    embeddings: np.ndarray, ids: list[str], k: int = 10, nlist: int = 16, nprobe: int = 8
) -> pd.DataFrame:
    """Approximate top-k via an IVF (inverted-file) index: vectors are clustered
    into `nlist` cells, and a query only searches its `nprobe` nearest cells
    instead of all points -- a different ANN family than HNSW (cluster-based vs
    graph-based). At n=400 nlist is tiny, so this is near-exact; it's the SAME
    interface, included to compare ANN families fairly.
    """
    d = embeddings.shape[1]
    emb = embeddings.astype(np.float32)
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(emb)
    index.add(emb)
    index.nprobe = nprobe
    n = embeddings.shape[0]
    scores, idx = index.search(emb, k + 1)

    rows = []
    for i in range(n):
        for rank in range(k + 1):
            j = idx[i, rank]
            if j == i or j == -1:
                continue
            rows.append((ids[i], ids[j], float(scores[i, rank])))
    return pd.DataFrame(rows, columns=["id_1", "id_2", "score"])


def recall_at_k(exact_pairs: pd.DataFrame, ann_pairs: pd.DataFrame) -> float:
    """Of all (query -> neighbor) edges found by exact search, what fraction
    does ANN also find? Direction-sensitive (id_1 is the query)."""
    exact_set = set(zip(exact_pairs.id_1, exact_pairs.id_2))
    ann_set = set(zip(ann_pairs.id_1, ann_pairs.id_2))
    if not exact_set:
        return 1.0
    return len(exact_set & ann_set) / len(exact_set)
