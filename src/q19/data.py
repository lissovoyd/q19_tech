"""Load the 400 support queries. Never touches eval_holdout/."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
QUERIES_PATH = ROOT / "queries.csv"


def load_queries(path: Path | str = QUERIES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str).str.strip()
    return df.reset_index(drop=True)
