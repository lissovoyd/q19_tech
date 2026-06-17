"""Phase 1 -- duplicate detection: the single file that runs the chosen path
end-to-end (embed -> cosine -> sort -> fixed cutoff) AND keeps every
alternative we built and benchmarked sitting right next to it, wired up but
switched off, with a comment on *why* it's off and *when* flipping it back on
would be the right call. Nothing below was deleted because it "didn't work" --
every function here was actually run and measured (see
outputs/PHASE1_GRAND.md for quality, outputs/PHASE1_COST_MATRIX.md for time
and memory) and the choice was made on that evidence, not on omission.

Frozen path: BAAI/bge-m3 embeddings -> exact all-pairs cosine -> fixed cutoff
0.75 -> undirected duplicate pairs. Chosen because, in a 13-configuration
capstone comparison built *before* a single look at the gold set and
evaluated once at the end:
  - best recall on the 60 gold pairs (48/60) of any method tried
  - cheapest of the embedding-based methods (~15-18s / ~3.5GB, one model load,
    the all-pairs cosine itself is a few milliseconds)
  - no approximation risk at this scale (n=400): ANN backends match it on
    recall but buy nothing on cost yet; cross-encoder reranking buys
    precision/hard-pair efficiency but at 5-13x the cost for *lower* overall
    recall here.

Run: python -m q19.phase1_pipeline
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from q19.ann import ann_candidate_pairs, exact_topk_pairs, ivf_candidate_pairs
from q19.data import load_queries
from q19.methods import embed_texts
from q19.pairs import matrix_to_pairs
from q19.rerank import method_cross_encoder
from q19.threshold import otsu_threshold

EMBED_MODEL = "BAAI/bge-m3"
FIXED_COSINE_CUTOFF = 0.75


# ============================================================================
# RETRIEVAL / CANDIDATE GENERATION -- how pairs get *found*
# ============================================================================

def retrieve_exact(embeddings: np.ndarray, ids: list[str]) -> pd.DataFrame:
    """ACTIVE. Score every one of the n*(n-1)/2 pairs (O(n^2)) and return them
    all, sorted by score desc. At n=400 this is ~79.8k pairs, scored in a few
    milliseconds once embeddings exist (PHASE1_COST_MATRIX.md: 0.005s) -- no
    reason to approximate at this scale, and exact has zero index-build
    overhead, so it's also the *fastest* retrieval here (see the
    retrieval-time column, not total time which is dominated by embedding).
    """
    return matrix_to_pairs(embeddings @ embeddings.T, ids)


def retrieve_ann_hnsw(embeddings: np.ndarray, ids: list[str], k: int = 50) -> pd.DataFrame:
    """DISABLED in the frozen path. Graph-based ANN (faiss HNSW): retrieves
    each query's top-k neighbors via greedy graph search instead of scoring
    every pair. At n=400 it gives the same recall as exact (cand-rec ~0.97,
    PHASE1_GRAND.md) but isn't measurably faster -- embedding cost dominates,
    and retrieval itself is sub-millisecond either way -- so it trades
    exactness for nothing here.

    Turn this on when: n grows past the empirical compute crossover on this
    hardware (~32,000 queries, see ANN_SCALING.md) where O(n^2) brute-force
    cosine starts to cost more wall-clock time than building+querying the
    graph, OR when the query set keeps growing and you need incremental
    inserts (HNSW tolerates those better than IVF).
    """
    return ann_candidate_pairs(embeddings, ids, k=k)


def retrieve_ann_ivf(embeddings: np.ndarray, ids: list[str], k: int = 50) -> pd.DataFrame:
    """DISABLED in the frozen path. Cluster-based ANN (faiss IVF): partitions
    vectors into nlist cells, a query only searches its nprobe nearest cells.
    A different ANN family than HNSW (clustering vs graph traversal) --
    included so the choice was actually "exact vs ANN family A vs ANN family
    B," not just "exact vs ANN." Same recall as HNSW at this n
    (PHASE1_GRAND.md): no benefit yet either.

    Turn this on instead of HNSW when: the corpus is large but mostly static
    -- IVF's clustering/training step is a one-time cost best amortized over
    many queries against a fixed index, e.g. a nightly-rebuilt search index.
    """
    return ivf_candidate_pairs(embeddings, ids, k=k)


def retrieve_exact_topk(embeddings: np.ndarray, ids: list[str], k: int = 50) -> pd.DataFrame:
    """DISABLED. faiss IndexFlatIP top-k per query -- still exact, but only
    materializes O(n*k) results instead of the full O(n^2) matrix. Used as the
    ground truth for measuring ANN recall (see ann.py + PHASE1_GRAND's
    cand-rec column); also a memory-saving exact alternative once n^2
    *storage*, not n^2 compute, is the bottleneck.

    Turn this on when: n is large enough that the O(n^2) similarity matrix
    won't fit in memory, but exact (not approximate) top-k is still cheap
    enough to compute -- i.e. memory, not time, is the binding constraint.
    """
    return exact_topk_pairs(embeddings, ids, k=k)


def _dedup_undirected(pairs: pd.DataFrame) -> pd.DataFrame:
    """ANN/top-k retrieval returns directed candidates (i->j and j->i found
    independently from each side); collapse to one row per undirected pair,
    keeping the max score, so cutoff counts compare fairly against the exact
    all-pairs path (which is undirected by construction)."""
    pairs = pairs.copy()
    pairs["_key"] = [tuple(sorted((a, b))) for a, b in zip(pairs.id_1, pairs.id_2)]
    pairs = pairs.sort_values("score", ascending=False).drop_duplicates("_key")
    return pairs.drop(columns="_key").reset_index(drop=True)


# ============================================================================
# THRESHOLDING -- how a score becomes a duplicate/not-duplicate decision
# ============================================================================

def threshold_fixed_cosine(scores: np.ndarray) -> float:
    """ACTIVE. A fixed cosine cutoff (0.75), picked by inspecting the score
    distribution and a handful of manually-checked example pairs *before* any
    gold lookup -- not fit to the data. Necessary because raw cosine
    similarity over this corpus is NOT bimodal (no clean valley between
    dup / non-dup, just a smooth decay), so a shape-based unsupervised method
    has nothing reliable to grab onto -- see threshold_otsu below for what
    happens if you try anyway.
    """
    return FIXED_COSINE_CUTOFF


def threshold_otsu(scores: np.ndarray) -> float:
    """DISABLED for raw cosine -- this is what we tried FIRST. Otsu assumes
    the scores are a mixture of two classes and picks the cut that maximizes
    between-class variance from the histogram shape alone, no labels needed.
    It works well on the cross-encoder's output (genuinely bimodal: the
    reranker pushes non-duplicates toward 0 and duplicates toward 1), but on
    raw cosine similarity the distribution is unimodal, so Otsu finds a cut
    near the bulk of the mass and floods the result with false positives.

    Turn this on when: scoring with a method whose output is empirically
    bimodal (cross-encoder rerank scores, or any calibrated classifier
    output) -- check the histogram shape first, don't apply it blindly.
    """
    return otsu_threshold(scores)


# ============================================================================
# RERANKING -- an optional second stage that re-scores retrieved candidates
# ============================================================================

def rerank_cross_encoder(texts: list[str], ids: list[str], retrieve_k: int = 20) -> pd.DataFrame:
    """DISABLED in the frozen path (the default end-to-end run never calls
    this). A cross-encoder (bge-reranker-v2-m3) reads both texts of a
    candidate pair *together* and re-scores it -- much better judgment than
    cosine similarity, but one forward pass per pair, so it only scores the
    bi-encoder's retrieved candidates (k per query), never all n^2 pairs.

    Why it's off: the capstone comparison (PHASE1_GRAND.md) showed two-stage
    tops out at 44/60 gold recall vs one-stage's 48/60 -- the reranker itself
    rejects some hard pairs the raw embeddings caught -- while costing
    ~5-13x as much wall-clock time (PHASE1_COST_MATRIX.md: 70-194s vs ~15s,
    almost entirely the cross-encoder forward passes) and loading a second
    model (~3.85GB peak vs ~3.5GB). It is not a strictly worse method,
    though: at *equal candidate budget* it is the best performer specifically
    on hard pairs, and its output is bimodal enough for Otsu to threshold
    cleanly (pair it with threshold_otsu, not threshold_fixed_cosine).

    Turn this on when: the cost of a false positive is high enough to justify
    5-13x more compute for better precision and hard-pair handling -- e.g. a
    compliance/escalation workflow where a human reviews every flagged pair
    and each review is expensive -- not a high-recall sweep like this task.
    """
    return method_cross_encoder(texts, ids, retrieve_k=retrieve_k)


# ============================================================================
# THE FROZEN PATH
# ============================================================================

def detect_duplicates(
    texts: list[str],
    ids: list[str],
    retrieve_fn=retrieve_exact,
    threshold_fn=threshold_fixed_cosine,
    use_cross_encoder: bool = False,
    cross_encoder_k: int = 20,
) -> pd.DataFrame:
    """End-to-end: embed -> retrieve candidates -> (optional rerank) ->
    threshold -> duplicate pairs.

    Defaults are the frozen, evaluated-once Phase 1 choice. The retrieve_fn /
    threshold_fn / use_cross_encoder knobs exist so every alternative above is
    a parameter swap, not a rewrite -- each one was actually run and measured,
    none was guessed or left untested.
    """
    embeddings = np.asarray(embed_texts(texts, model_name=EMBED_MODEL), dtype=np.float32)
    pairs = retrieve_fn(embeddings, ids)
    if retrieve_fn is not retrieve_exact:
        pairs = _dedup_undirected(pairs)

    if use_cross_encoder:
        pairs = rerank_cross_encoder(texts, ids, retrieve_k=cross_encoder_k)

    cutoff = threshold_fn(pairs["score"].to_numpy())
    return pairs[pairs["score"] >= cutoff].reset_index(drop=True)


def main():
    q = load_queries()
    texts, ids = q["text"].tolist(), q["query_id"].tolist()
    dups = detect_duplicates(texts, ids)  # frozen path only -- see module docstring for why
    print(f"{len(dups)} duplicate pairs flagged at cutoff={FIXED_COSINE_CUTOFF}")
    print(dups.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
