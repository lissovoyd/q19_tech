"""Cross-encoder reranking -- the second stage of two-stage duplicate detection.

A bi-encoder (sentence embeddings) encodes each query independently, so scoring
a pair is just a cheap dot product -- great for scanning all pairs, but it never
sees the two texts *together*. A cross-encoder feeds both texts into the model at
once, so it can judge "do these describe the same issue?" far more accurately --
but at one forward pass per pair, far too expensive for all 79,800 pairs.

So we use it as a *reranker*: the bi-encoder retrieves a small candidate set
(each query's top-k neighbors), and the cross-encoder re-scores only those. This
is the standard retrieve-then-rerank pattern. Never touches eval_holdout/.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from q19.methods import embed_texts

# Local snapshot so it loads offline (same cache as bge-m3).
_RERANKER_PATH = (
    "D:/_PROJECTS_/bankdoc-ai/models_cache/models--BAAI--bge-reranker-v2-m3/"
    "snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
)

CACHE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "cache"


def retrieve_candidates(embeddings: np.ndarray, ids: list[str], k: int = 10) -> list[tuple[int, int]]:
    """Each query's top-k neighbors (by cosine), as a deduped set of (i, j) index
    pairs with i < j. This is the bi-encoder's cheap candidate-generation step."""
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -1.0)
    n = len(ids)
    cand = set()
    for i in range(n):
        topk = np.argpartition(sims[i], -k)[-k:]
        for j in topk:
            a, b = (i, int(j)) if i < j else (int(j), i)
            if a != b:
                cand.add((a, b))
    return sorted(cand)


def method_cross_encoder(
    texts: list[str],
    ids: list[str],
    retrieve_k: int = 10,
    model_name: str = _RERANKER_PATH,
    embed_model: str = "BAAI/bge-m3",
    use_cache: bool = True,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Two-stage: bi-encoder retrieves top-k candidates, cross-encoder re-scores
    them. Returns a pairs_df (id_1, id_2, score) sorted desc -- same shape as
    matrix_to_pairs, so it drops straight into eval.py. Pairs that were never
    retrieved are simply absent (treated as non-duplicates downstream)."""
    emb = embed_texts(texts, model_name=embed_model, use_cache=use_cache)
    candidates = retrieve_candidates(np.asarray(emb, dtype=np.float32), ids, k=retrieve_k)

    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, max_length=512)
    text_pairs = [(texts[i], texts[j]) for i, j in candidates]
    # CrossEncoder.default_activation_function is Sigmoid() for this model, so
    # predict() already returns scores in (0, 1) -- do NOT squash again.
    scores = np.asarray(model.predict(text_pairs, batch_size=batch_size, show_progress_bar=True))

    df = pd.DataFrame({
        "id_1": [ids[i] for i, _ in candidates],
        "id_2": [ids[j] for _, j in candidates],
        "score": scores,
    })
    return df.sort_values("score", ascending=False).reset_index(drop=True)
