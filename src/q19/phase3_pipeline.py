"""Phase 3 — cluster labelling and ops report.

Loads the frozen Phase 2 config (KMeans k=17, BAAI/bge-m3), labels every
cluster with one LLM call each, then makes one more call for a structured
ops report (executive summary, escalation clusters, recommendations).

Saves outputs/cluster_labels.json. All LLM calls are cached in
outputs/llm_cache/ — re-runs are free and deterministic.

Run: python -m q19.phase3_pipeline
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from q19.cluster import kmeans_cluster
from q19.data import load_queries
from q19.label import label_all_clusters
from q19.llm import chat, reset_usage, usage_summary
from q19.methods import embed_texts

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
EMBED_MODEL = "BAAI/bge-m3"
K = 17
_MAX_PARSE_RETRIES = 2


def build_volume_table(cluster_labels: list[dict], labels: np.ndarray) -> list[dict]:
    id_to_meta = {r["cluster_id"]: r for r in cluster_labels}
    total = len(labels)
    rows = []
    for cid in sorted(set(labels.tolist())):
        meta = id_to_meta[cid]
        n = int((labels == cid).sum())
        rows.append({
            "cluster_id": cid,
            "label": meta["label"],
            "description": meta["description"],
            "n_queries": n,
            "pct": n / total * 100,
        })
    rows.sort(key=lambda r: r["n_queries"], reverse=True)
    return rows


def ops_report(rows: list[dict], total: int) -> dict:
    """One LLM call → structured JSON: summary, escalation clusters, recommendations."""
    all_clusters = "\n".join(
        f"  {i+1}. {r['label']} ({r['n_queries']} queries, {r['pct']:.1f}%): {r['description']}"
        for i, r in enumerate(rows)
    )
    prompt = (
        f"You are writing a weekly support operations report for a banking app. "
        f"This week covers {total} customer support queries across {K} issue clusters.\n\n"
        f"All clusters sorted by volume:\n{all_clusters}\n\n"
        "Return valid JSON only — no markdown fences, no extra text:\n"
        "{\n"
        '  "executive_summary": "<2-3 sentences describing the overall support landscape; '
        'mention the most significant clusters by their exact name>",\n'
        '  "escalation_clusters": [\n'
        '    {"label": "<exact cluster name>", '
        '"reason": "<why this needs urgent attention regardless of volume>"}\n'
        "  ],\n"
        '  "recommendations": ["<actionable recommendation 1>", "<actionable recommendation 2>"]\n'
        "}"
    )
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(1 + _MAX_PARSE_RETRIES):
        result = chat(messages, temperature=0.3, max_tokens=600)
        text = result["text"]
        try:
            start, end = text.find("{"), text.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("no JSON object found")
            data = json.loads(text[start:end])
            for key in ("executive_summary", "escalation_clusters", "recommendations"):
                if key not in data:
                    raise ValueError(f"missing key: {key}")
            return data
        except (ValueError, json.JSONDecodeError):
            if attempt == _MAX_PARSE_RETRIES:
                return {"executive_summary": text, "escalation_clusters": [], "recommendations": []}


def main() -> None:
    print("Loading queries and embeddings (cached)...")
    q = load_queries()
    texts = q["text"].tolist()
    embeddings = np.asarray(embed_texts(texts, model_name=EMBED_MODEL), dtype=np.float32)

    print(f"Clustering (KMeans k={K})...")
    labels, _ = kmeans_cluster(embeddings, K)

    # --- Step 1: label clusters ---
    print(f"\nLabelling {K} clusters (1 LLM call each)...")
    reset_usage()
    cluster_labels = label_all_clusters(texts, embeddings, labels)

    OUT.mkdir(exist_ok=True)
    labels_path = OUT / "cluster_labels.json"
    labels_path.write_text(json.dumps(cluster_labels, indent=2, ensure_ascii=False), encoding="utf-8")

    u = usage_summary()
    cost_label = "Actual" if u["cost_is_real"] else "Hypothetical"
    print(f"\nLabelling — {u['calls']} calls ({u['cache_hits']} cached), "
          f"{u['total_tokens']} tokens, {cost_label} cost ${u['cost_usd']:.4f}")
    print(f"\n{'ID':>3}  {'Size':>5}  {'Label':<35}  Description")
    print("-" * 100)
    for r in cluster_labels:
        print(f"{r['cluster_id']:>3}  {r['n_queries']:>5}  {r['label']:<35}  {r['description']}")

    # --- Step 2: ops report ---
    print("\nGenerating ops report (1 LLM call)...")
    reset_usage()
    rows = build_volume_table(cluster_labels, labels)
    report = ops_report(rows, total=len(labels))

    u = usage_summary()
    cost_label = "Actual" if u["cost_is_real"] else "Hypothetical"
    print(f"Report — {u['calls']} call(s), {u['total_tokens']} tokens, "
          f"{cost_label} cost ${u['cost_usd']:.4f}")

    print("\n--- Executive Summary ---")
    print(report["executive_summary"])
    print("\n--- Escalation Clusters ---")
    for e in report["escalation_clusters"]:
        print(f"  {e['label']}: {e['reason']}")
    print("\n--- Recommendations ---")
    for i, r in enumerate(report["recommendations"], 1):
        print(f"  {i}. {r}")


if __name__ == "__main__":
    main()
