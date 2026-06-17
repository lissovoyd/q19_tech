"""Turn (pairs + threshold) into duplicate-cluster structure via union-find.

Same machinery serves two purposes at two thresholds:
- high threshold -> near-duplicate clusters (Phase 1)
- low threshold  -> topic/issue clusters (Phase 2)
Never touches eval_holdout/.
"""
import pandas as pd


class _UnionFind:
    def __init__(self, items: list[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def pairs_to_components(pairs_df: pd.DataFrame, query_ids: list[str], threshold: float) -> pd.DataFrame:
    """One row per query_id, with its component_id and component_size.
    A query with no edge above threshold is its own singleton component.
    """
    uf = _UnionFind(query_ids)
    edges = pairs_df[pairs_df["score"] > threshold]
    for r in edges.itertuples():
        uf.union(r.id_1, r.id_2)

    roots = {qid: uf.find(qid) for qid in query_ids}
    sizes: dict[str, int] = {}
    for root in roots.values():
        sizes[root] = sizes.get(root, 0) + 1

    out = pd.DataFrame({
        "query_id": query_ids,
        "component_id": [roots[qid] for qid in query_ids],
    })
    out["component_size"] = out["component_id"].map(sizes)
    return out
