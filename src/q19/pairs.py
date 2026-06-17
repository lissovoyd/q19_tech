"""Turn a similarity matrix into a ranked list of pairs."""
import numpy as np
import pandas as pd


def matrix_to_pairs(matrix: np.ndarray, query_ids: list[str]) -> pd.DataFrame:
    """All n*(n-1)/2 unordered pairs, sorted by score descending."""
    n = matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    df = pd.DataFrame(
        {
            "id_1": [query_ids[i] for i in iu[0]],
            "id_2": [query_ids[i] for i in iu[1]],
            "score": matrix[iu],
        }
    )
    return df.sort_values("score", ascending=False).reset_index(drop=True)
