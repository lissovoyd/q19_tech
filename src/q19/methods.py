"""Similarity methods (the 'rungs'). Each takes a list of texts and returns
an (n, n) similarity matrix (symmetric, diagonal ignored downstream).
"""
import hashlib
import re
from pathlib import Path

import numpy as np
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CACHE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalize(text: str) -> str:
    return text.lower().strip()


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


_stemmer = PorterStemmer()


def _stem_text(text: str) -> str:
    """'transactions failing' -> 'transact fail' -- collapses inflected forms
    (failed/failing/fails, card/cards) to a shared root so lexical methods
    match on root words instead of exact surface forms."""
    return " ".join(_stemmer.stem(tok) for tok in _tokenize(text))


def _stem_texts(texts: list[str]) -> list[str]:
    return [_stem_text(t) for t in texts]


# ---- Rung 0: exact / normalized match ----------------------------------
def method_exact_match(texts: list[str]) -> np.ndarray:
    norm = [_normalize(t) for t in texts]
    n = len(texts)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            m[i, j] = 1.0 if norm[i] == norm[j] else 0.0
    return m


# ---- Rung 1: classical lexical methods ----------------------------------
def method_tfidf_word(texts: list[str], ngram_range=(1, 2)) -> np.ndarray:
    vec = TfidfVectorizer(ngram_range=ngram_range, min_df=1)
    X = vec.fit_transform(texts)
    return cosine_similarity(X)


def method_tfidf_char(texts: list[str], ngram_range=(3, 5)) -> np.ndarray:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, min_df=1)
    X = vec.fit_transform(texts)
    return cosine_similarity(X)


def method_bm25(texts: list[str]) -> np.ndarray:
    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    n = len(texts)
    m = np.zeros((n, n))
    for i in range(n):
        scores = bm25.get_scores(tokenized[i])  # score of query i against every doc
        m[i, :] = scores
    # BM25(i -> j) != BM25(j -> i); symmetrize, then min-max normalize to [0, 1]
    m = (m + m.T) / 2.0
    if m.max() > 0:
        m = m / m.max()
    return m


def method_jaccard(texts: list[str]) -> np.ndarray:
    token_sets = [set(_tokenize(t)) for t in texts]
    n = len(texts)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            a, b = token_sets[i], token_sets[j]
            union = len(a | b)
            score = len(a & b) / union if union else 0.0
            m[i, j] = m[j, i] = score
    return m


def method_fuzzy(texts: list[str]) -> np.ndarray:
    n = len(texts)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            score = fuzz.token_sort_ratio(texts[i], texts[j]) / 100.0
            m[i, j] = m[j, i] = score
    return m


# ---- Rung 1b: lexical methods + stemming preprocessing ------------------
# Same scoring as Rung 1, but texts are stemmed first so e.g. "failed",
# "fails", "failing" all collapse to "fail" before comparison. Tests whether
# closing the inflection gap recovers some of the paraphrase recall that
# pure word-matching misses, without paying for embeddings.
def method_tfidf_word_stemmed(texts: list[str], ngram_range=(1, 2)) -> np.ndarray:
    return method_tfidf_word(_stem_texts(texts), ngram_range=ngram_range)


def method_bm25_stemmed(texts: list[str]) -> np.ndarray:
    return method_bm25(_stem_texts(texts))


def method_jaccard_stemmed(texts: list[str]) -> np.ndarray:
    return method_jaccard(_stem_texts(texts))


# ---- Rung 2: sentence embeddings ----------------------------------------
def _cache_key(texts: list[str], model_name: str) -> str:
    h = hashlib.sha256(("\n".join(texts) + model_name).encode()).hexdigest()[:16]
    safe_model = model_name.replace("/", "_")
    return f"emb_{safe_model}_{h}.npy"


def embed_texts(
    texts: list[str],
    model_name: str = "BAAI/bge-m3",
    cache_folder: str | None = "D:/_PROJECTS_/bankdoc-ai/models_cache",
    use_cache: bool = True,
) -> np.ndarray:
    """Returns (n, dim) embeddings, cached to disk by content hash.
    use_cache=False forces a fresh recompute (model load + encode), even if a
    cached .npy already exists -- used to measure a method's true standalone cost.
    """
    cache_path = CACHE_DIR / _cache_key(texts, model_name)
    if use_cache and cache_path.exists():
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, cache_folder=cache_folder)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    np.save(cache_path, embeddings)
    return embeddings


def method_embeddings(
    texts: list[str], model_name: str = "BAAI/bge-m3", use_cache: bool = True
) -> np.ndarray:
    emb = embed_texts(texts, model_name=model_name, use_cache=use_cache)
    return cosine_similarity(emb)


# ---- Rung 3: hybrid (rank fusion of lexical + embeddings) ---------------
def _rank_normalize(matrix: np.ndarray) -> np.ndarray:
    """Convert scores to percentile ranks in [0, 1] so different scales mix fairly."""
    n = matrix.shape[0]
    flat = matrix[np.triu_indices(n, k=1)]
    order = flat.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(flat))
    ranks = ranks / max(len(flat) - 1, 1)
    out = np.zeros_like(matrix)
    iu = np.triu_indices(n, k=1)
    out[iu] = ranks
    out = out + out.T
    np.fill_diagonal(out, 1.0)
    return out


def method_hybrid_bm25_embeddings(
    texts: list[str],
    model_name: str = "BAAI/bge-m3",
    weight_embeddings: float = 0.7,
    use_cache: bool = True,
) -> np.ndarray:
    bm25 = _rank_normalize(method_bm25(texts))
    emb = _rank_normalize(method_embeddings(texts, model_name=model_name, use_cache=use_cache))
    return weight_embeddings * emb + (1 - weight_embeddings) * bm25


# ---- Candidate-generation variant (the scale path) ---------------------
# Same embeddings + cosine as embeddings_bge_m3, but instead of scoring all
# n^2 pairs it retrieves only each query's top-k neighbors via an HNSW index
# (q19/ann.py). Returns a SPARSE matrix: retrieved pairs carry their true
# cosine, every other entry is 0. So its score distribution is 0 for the vast
# majority of pairs *by design* -- it's answering "who are the few neighbors
# worth looking at," not "score every pair." See outputs/ANN_SCALING.md.
def method_ann_embeddings(
    texts: list[str], model_name: str = "BAAI/bge-m3", k: int = 10, use_cache: bool = True
) -> np.ndarray:
    from q19.ann import ann_candidate_pairs

    emb = embed_texts(texts, model_name=model_name, use_cache=use_cache).astype(np.float32)
    ids = list(range(len(texts)))
    pairs = ann_candidate_pairs(emb, ids, k=k)
    n = len(texts)
    m = np.zeros((n, n))
    for r in pairs.itertuples():
        i, j = int(r.id_1), int(r.id_2)
        m[i, j] = m[j, i] = max(m[i, j], r.score)
    return m


METHODS = {
    "exact_match": method_exact_match,
    "tfidf_word": method_tfidf_word,
    "tfidf_char": method_tfidf_char,
    "bm25": method_bm25,
    "jaccard": method_jaccard,
    "fuzzy": method_fuzzy,
    "tfidf_word_stemmed": method_tfidf_word_stemmed,
    "bm25_stemmed": method_bm25_stemmed,
    "jaccard_stemmed": method_jaccard_stemmed,
    "embeddings_bge_m3": method_embeddings,
    "ann_embeddings": method_ann_embeddings,
}
# Dropped `hybrid_bm25_embeddings`: the fair equal-budget comparison showed it is
# the *worst* method on hard pairs -- fusing BM25 penalizes low-lexical-overlap
# pairs, which is exactly where semantic help is needed. Its earlier "high recall"
# was a flooding artifact. Kept the function above for reference, out of the registry.

# n = number of queries (400); L = tokens per query (~avg 15);
# V = vocabulary size (distinct words/n-grams, V << n*L); d = embedding dim (1024, bge-m3).
# Every method also pays O(n^2) to store the full pairwise similarity matrix --
# that term is shared and omitted from "extra" complexity below except where it dominates.
METHOD_INFO = {
    "exact_match": {
        "rung": "0",
        "description": (
            "Lowercase + strip, then exact string equality. No similarity score, "
            "just 1.0/0.0. Establishes the floor: catches zero paraphrases."
        ),
        "time_complexity": "O(n^2 * L) -- string compare per pair",
        "memory_complexity": "O(n^2) matrix + O(n*L) normalized strings",
    },
    "tfidf_word": {
        "rung": "1",
        "description": (
            "Vectorize each query into word/bigram TF-IDF weights, then cosine "
            "similarity between vectors. Rewards shared *specific* vocabulary."
        ),
        "time_complexity": "O(n*L) to vectorize + O(n^2 * V_avg) for pairwise cosine (sparse)",
        "memory_complexity": "O(n*V) sparse TF-IDF matrix + O(n^2) dense similarity matrix",
    },
    "tfidf_char": {
        "rung": "1",
        "description": (
            "Same as tfidf_word but on overlapping 3-5 character chunks instead of "
            "whole words -- tolerant of typos and word-form variants."
        ),
        "time_complexity": "O(n*L) to vectorize + O(n^2 * V_avg) for pairwise cosine (sparse)",
        "memory_complexity": "O(n*V) sparse TF-IDF matrix (V larger than word-level) + O(n^2)",
    },
    "bm25": {
        "rung": "1",
        "description": (
            "Search-engine ranking formula: rewards shared rare words, dampens "
            "reward for repeated words. Asymmetric by construction (query vs doc), "
            "so scored both directions and averaged, then rescaled to [0,1]."
        ),
        "time_complexity": "O(n^2 * L) -- one get_scores() call per query against all docs",
        "memory_complexity": "O(n*L) tokenized corpus + O(n^2) similarity matrix",
    },
    "jaccard": {
        "rung": "1",
        "description": (
            "Token-set overlap: |shared words| / |union of words|. No weighting "
            "at all -- purest, crudest lexical baseline."
        ),
        "time_complexity": "O(n^2 * L) -- set intersection/union per pair",
        "memory_complexity": "O(n*L) token sets + O(n^2) similarity matrix",
    },
    "fuzzy": {
        "rung": "1",
        "description": (
            "rapidfuzz token_sort_ratio: sort words alphabetically (so word order "
            "doesn't matter), then edit-distance ratio between the two strings."
        ),
        "time_complexity": "O(n^2 * L log L) -- sort per text + O(L^2)-ish edit distance per pair",
        "memory_complexity": "O(n*L) sorted strings + O(n^2) similarity matrix",
    },
    "tfidf_word_stemmed": {
        "rung": "1b",
        "description": (
            "tfidf_word, but every token is Porter-stemmed first (failing/failed/"
            "fails -> fail) so inflected forms match on their shared root."
        ),
        "time_complexity": "O(n*L) stemming + same as tfidf_word",
        "memory_complexity": "same as tfidf_word",
    },
    "bm25_stemmed": {
        "rung": "1b",
        "description": "bm25 on stemmed tokens -- same rationale as tfidf_word_stemmed.",
        "time_complexity": "O(n*L) stemming + same as bm25",
        "memory_complexity": "same as bm25",
    },
    "jaccard_stemmed": {
        "rung": "1b",
        "description": "jaccard on stemmed tokens -- closes the inflection gap for the crudest baseline.",
        "time_complexity": "O(n*L) stemming + same as jaccard",
        "memory_complexity": "same as jaccard",
    },
    "embeddings_bge_m3": {
        "rung": "2",
        "description": (
            "Encode each query with a transformer sentence-embedding model "
            "(BAAI/bge-m3, 1024-dim) so similar *meaning* -- not matching words -- "
            "produces a nearby vector; cosine similarity between vectors. "
            "Embeddings are computed once and cached to disk by content hash."
        ),
        "time_complexity": "O(n * f(L)) to encode (f ~ transformer forward cost) + O(n^2 * d) for pairwise cosine",
        "memory_complexity": "O(n*d) embedding matrix (cached .npy) + O(n^2) similarity matrix",
    },
    "ann_embeddings": {
        "rung": "2-scale",
        "description": (
            "Same embeddings + cosine as embeddings_bge_m3, but retrieves only each "
            "query's top-k neighbors via an HNSW index (faiss) instead of scoring all "
            "n^2 pairs. Sparse by design: most pairs score 0 because they're never "
            "retrieved. This is the candidate-generation / scale path -- at n=400 it's "
            "no faster than brute force (see ANN_SCALING.md), but its cost grows ~n log n "
            "instead of n^2. Same accuracy on the pairs that matter (recall@10 = 1.0 vs exact here)."
        ),
        "time_complexity": "O(n * f(L)) to encode + O(n log n) HNSW build/query (vs O(n^2) brute force)",
        "memory_complexity": "O(n*d) embeddings + O(n*M) graph index + O(n*k) candidate pairs (no n^2 matrix needed at scale)",
    },
}
