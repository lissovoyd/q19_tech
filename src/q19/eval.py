"""Scoring against the gold pairs. Only call this once you've frozen your
method and threshold -- this is the one module allowed to read eval_holdout/.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = ROOT / "eval_holdout" / "duplicate_pairs.csv"


def load_gold(path: Path | str = GOLD_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def evaluate(pairs_df: pd.DataFrame, gold_df: pd.DataFrame, threshold: float) -> dict:
    """pairs_df: output of matrix_to_pairs (all pairs, with 'score'), sorted desc.
    Returns precision/recall/F1 overall + by difficulty, plus threshold-free stats.
    """
    gold_keys = {
        _pair_key(r.query_id_1, r.query_id_2): r.difficulty for r in gold_df.itertuples()
    }
    pairs_df = pairs_df.copy()
    pairs_df["key"] = [_pair_key(a, b) for a, b in zip(pairs_df.id_1, pairs_df.id_2)]
    pairs_df["is_gold"] = pairs_df["key"].isin(gold_keys)
    pairs_df["predicted"] = pairs_df["score"] >= threshold

    tp = int((pairs_df["predicted"] & pairs_df["is_gold"]).sum())
    fp = int((pairs_df["predicted"] & ~pairs_df["is_gold"]).sum())
    fn = int((~pairs_df["predicted"] & pairs_df["is_gold"]).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    by_difficulty = {}
    for diff in ("easy", "medium", "hard"):
        diff_keys = {k for k, d in gold_keys.items() if d == diff}
        sub = pairs_df[pairs_df["key"].isin(diff_keys)]
        caught = int(sub["predicted"].sum())
        by_difficulty[diff] = {
            "total_gold": len(diff_keys),
            "caught": caught,
            "recall": caught / len(diff_keys) if diff_keys else 0.0,
        }

    # threshold-free: rank-based recall@K and average precision
    n_pairs = len(pairs_df)
    pairs_df["rank"] = np.arange(1, n_pairs + 1)  # already sorted desc by score
    gold_ranks = pairs_df.loc[pairs_df["is_gold"], "rank"]
    recall_at_k = {
        k: float((gold_ranks <= k).mean()) for k in (60, 100, 200, 500, 1000)
    }
    ap = average_precision_score(
        pairs_df["is_gold"].astype(int), pairs_df["score"]
    )

    return {
        "threshold": threshold,
        "overall": {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn},
        "by_difficulty": by_difficulty,
        "rank_based": {
            "median_gold_rank": float(gold_ranks.median()),
            "recall_at_k": recall_at_k,
            "total_pairs": n_pairs,
        },
        "average_precision": float(ap),
    }


def missed_pairs(pairs_df: pd.DataFrame, gold_df: pd.DataFrame, threshold: float, queries_df) -> pd.DataFrame:
    """Gold pairs that scored below threshold -- for the required failure analysis."""
    gold_keys = {
        _pair_key(r.query_id_1, r.query_id_2): r.difficulty for r in gold_df.itertuples()
    }
    pairs_df = pairs_df.copy()
    pairs_df["key"] = [_pair_key(a, b) for a, b in zip(pairs_df.id_1, pairs_df.id_2)]
    pairs_df["is_gold"] = pairs_df["key"].isin(gold_keys)
    missed = pairs_df[pairs_df["is_gold"] & (pairs_df["score"] < threshold)].copy()
    missed["difficulty"] = missed["key"].map(gold_keys)
    text_map = dict(zip(queries_df.query_id, queries_df.text))
    missed["text_1"] = missed["id_1"].map(text_map)
    missed["text_2"] = missed["id_2"].map(text_map)
    return missed.sort_values("score", ascending=False)
