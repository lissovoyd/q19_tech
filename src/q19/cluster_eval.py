"""Phase 2 gold checks -- the only module allowed to read
eval_holdout/duplicate_pairs.csv for clustering. Mirrors eval.py's role in
Phase 1: call this ONLY after k and the algorithm are already frozen from
unsupervised internal metrics (cluster.py + the sweep in
scripts/phase2_clustering.py). Gold never influences which k gets chosen.
"""
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = ROOT / "eval_holdout" / "duplicate_pairs.csv"


def load_gold(path: Path | str = GOLD_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def bridge_metric(labels: list, ids: list[str], gold_df: pd.DataFrame) -> dict:
    """% of the 60 gold duplicate pairs whose two queries land in the same
    cluster -- recall-shaped (every gold pair is a *known* true duplicate, so
    "did clustering keep them together" has no false-positive ambiguity, same
    reasoning that made Phase 1's recall trustworthy despite non-exhaustive
    gold). Also split by difficulty, same as Phase 1's by-difficulty recall.
    """
    id_to_label = dict(zip(ids, labels))
    by_difficulty = {d: [0, 0] for d in ("easy", "medium", "hard")}
    co_clustered = 0
    for r in gold_df.itertuples():
        same = id_to_label[r.query_id_1] == id_to_label[r.query_id_2]
        co_clustered += int(same)
        by_difficulty[r.difficulty][0] += int(same)
        by_difficulty[r.difficulty][1] += 1
    n_total = len(gold_df)
    return {
        "co_clustered": co_clustered,
        "total": n_total,
        "pct": co_clustered / n_total if n_total else 0.0,
        "by_difficulty": {
            d: {"caught": c, "total": t, "pct": c / t if t else 0.0}
            for d, (c, t) in by_difficulty.items()
        },
    }


def root_issue_alignment(labels: list, ids: list[str], gold_df: pd.DataFrame) -> dict:
    """Bonus sanity check, NOT required and NOT used to pick k: on the subset
    of queries that appear in a gold pair (~120 of 400 -- the rest have no
    root_issue label at all), how well does our cluster assignment line up
    with the gold root_issue category? ARI/NMI are invariant to cluster *ID*
    numbering -- they ask "do same-root_issue queries end up in the same
    cluster," not "did we guess the right cluster index" -- so they're a fair
    comparison between two completely independent labelings (ours has no
    names yet; gold's root_issue are human-written category names).
    """
    id_to_label = dict(zip(ids, labels))
    q_to_issue: dict[str, str] = {}
    for r in gold_df.itertuples():
        q_to_issue[r.query_id_1] = r.root_issue
        q_to_issue[r.query_id_2] = r.root_issue
    qs = list(q_to_issue)
    y_true = [q_to_issue[q] for q in qs]
    y_pred = [id_to_label[q] for q in qs]
    return {
        "n_labelled_queries": len(qs),
        "n_root_issues": len(set(y_true)),
        "adjusted_rand_index": float(adjusted_rand_score(y_true, y_pred)),
        "normalized_mutual_info": float(normalized_mutual_info_score(y_true, y_pred)),
    }
