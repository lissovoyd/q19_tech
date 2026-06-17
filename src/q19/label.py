"""Phase 3 cluster labelling — medoid selection, representative picking,
LLM prompting (one call per cluster), JSON parsing with retry.
"""
from __future__ import annotations

import json

import numpy as np

from q19.llm import chat

_MAX_PARSE_RETRIES = 2


def find_medoid(embeddings: np.ndarray, mask: np.ndarray) -> int:
    """Index (in the original array) of the point closest to the cluster mean."""
    cluster_embs = embeddings[mask]
    centroid = cluster_embs.mean(axis=0)
    dists = np.linalg.norm(cluster_embs - centroid, axis=1)
    local_idx = int(dists.argmin())
    return int(np.where(mask)[0][local_idx])


def pick_representatives(
    embeddings: np.ndarray,
    mask: np.ndarray,
    n: int = 5,
) -> list[int]:
    """Medoid first, then the n-1 nearest neighbours to the medoid within
    the cluster. All indices are in the original (full-dataset) array."""
    medoid_idx = find_medoid(embeddings, mask)
    cluster_indices = np.where(mask)[0]

    if len(cluster_indices) <= n:
        return cluster_indices.tolist()

    medoid_emb = embeddings[medoid_idx]
    dists = np.linalg.norm(embeddings[cluster_indices] - medoid_emb, axis=1)
    # sort by distance; skip the medoid itself (dist=0)
    order = np.argsort(dists)
    selected = [medoid_idx]
    for i in order:
        idx = int(cluster_indices[i])
        if idx != medoid_idx:
            selected.append(idx)
        if len(selected) == n:
            break
    return selected


def _build_prompt(texts: list[str], rep_indices: list[int], cluster_id: int) -> list[dict]:
    examples = "\n".join(
        f'{i+1}. "{texts[idx]}"' for i, idx in enumerate(rep_indices)
    )
    return [
        {
            "role": "system",
            "content": (
                "You label customer support query clusters. "
                "Respond with valid JSON only — no markdown, no extra text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Below are example queries from cluster {cluster_id}.\n\n"
                f"{examples}\n\n"
                'Return exactly this JSON:\n'
                '{"label": "<3-6 word title-case name>", '
                '"description": "<one sentence explaining what this cluster is about>"}'
            ),
        },
    ]


def _parse_label(text: str) -> dict:
    """Extract the first JSON object from the response."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in: {text!r}")
    parsed = json.loads(text[start:end])
    parsed = {k.lower(): v for k, v in parsed.items()}
    if "label" not in parsed or "description" not in parsed:
        raise ValueError(f"Missing required fields in: {parsed}")
    return {"label": str(parsed["label"]), "description": str(parsed["description"])}


def label_cluster(
    texts: list[str],
    embeddings: np.ndarray,
    mask: np.ndarray,
    cluster_id: int,
    n_representatives: int = 5,
) -> dict:
    """Label one cluster. Returns {"cluster_id", "label", "description",
    "n_queries", "representative_indices", "cached"}."""
    rep_indices = pick_representatives(embeddings, mask, n=n_representatives)
    messages = _build_prompt(texts, rep_indices, cluster_id)

    parsed = None
    cached = False
    for attempt in range(1 + _MAX_PARSE_RETRIES):
        result = chat(messages, temperature=0.0, max_tokens=128)
        cached = result["cached"]
        try:
            parsed = _parse_label(result["text"])
            break
        except (ValueError, json.JSONDecodeError):
            if attempt == _MAX_PARSE_RETRIES:
                parsed = {"label": f"Cluster {cluster_id}", "description": result["text"]}

    return {
        "cluster_id": cluster_id,
        "label": parsed["label"],
        "description": parsed["description"],
        "n_queries": int(mask.sum()),
        "representative_indices": rep_indices,
        "cached": cached,
    }


def label_all_clusters(
    texts: list[str],
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_representatives: int = 5,
) -> list[dict]:
    """Label every cluster in order. Returns a list sorted by cluster_id."""
    cluster_ids = sorted(set(labels.tolist()))
    results = []
    for cid in cluster_ids:
        mask = labels == cid
        r = label_cluster(texts, embeddings, mask, cid, n_representatives)
        status = "cached" if r["cached"] else "called"
        print(f"  cluster {cid:2d} ({r['n_queries']:3d} queries) [{status}]: {r['label']}")
        results.append(r)
    return results
