# Q19 — Banking Support Query Pipeline

A three-phase NLP pipeline over 400 real banking support queries: **duplicate detection**, **intent clustering**, and **LLM-powered ops reporting**. Every design choice is justified by measured evidence, not assertion.

---

## Repository Layout

```
src/q19/
  phase1_pipeline.py  # entry point — embed, deduplicate, evaluate
  phase2_pipeline.py  # entry point — cluster, internal metrics
  phase3_pipeline.py  # entry point — LLM labelling + ops report
  data.py             # load queries
  preprocess.py       # PII / numeric scrubbing (<CARD>, <AMOUNT>, <DATE>)
  methods.py          # similarity methods: exact → TF-IDF → BM25 → embeddings
  ann.py              # ANN drop-in (faiss HNSW + IVF) for scale
  pairs.py            # similarity matrix → sorted pair dataframe
  threshold.py        # Otsu threshold from score histogram
  graph.py            # connected-component duplicate grouping
  rerank.py           # cross-encoder reranker (bge-reranker-v2-m3)
  eval.py             # gold evaluation — only file that reads eval_holdout/
  cluster.py          # KMeans / Agglomerative / HDBSCAN / UMAP / BERTopic
  cluster_eval.py     # silhouette, Davies-Bouldin, Calinski-Harabasz, bridge metric
  llm.py              # LLM client: disk cache, retry, token + cost tracking
  label.py            # representative selection, cluster labelling

eval_holdout/
  duplicate_pairs.csv # 60 labelled pairs — evaluation only, never used during building

outputs/
  PHASE1_REPORT.md    # method ladder, ANN scaling, 13-config comparison, gold evaluation
  PHASE2_REPORT.md    # clustering sweeps, k selection, scaling experiment, bridge metric
  PHASE3_OPS_REPORT.md  # Haiku vs Qwen: labels, narrative, escalation, cost comparison
```

---

<!--
scripts/
  compare_methods.py         # Phase 1 — runs every method isolated, writes method_runs/
  ann_experiment.py          # empirical ANN vs brute-force crossover
  evaluate_phase1.py         # one-time gold evaluation → PHASE1_EVAL.md
  phase1_grand_comparison.py # 13 configs (method × backend × k) → PHASE1_GRAND.md
  compare_focus.py           # with vs without cross-encoder at equal budget → PHASE1_FOCUS.md
  cost_matrix.py             # time + memory for every config → PHASE1_COST_MATRIX.md
  phase2_clustering.py       # all clustering sweeps + bridge metric → PHASE2_CLUSTERING.md
  cluster_scaling.py         # n=400…32,000 scaling experiment → PHASE2_SCALING.md
  bertopic_gold_check.py     # curiosity-only BERTopic gold check (post-freeze)
  phase3_label_clusters.py   # label all 17 clusters via LLM → cluster_labels.json
  phase3_ops_report.py       # aggregate volumes + 1 LLM call → PHASE3_OPS_REPORT.md
  phase3_compare.py          # side-by-side Haiku vs Qwen comparison → PHASE3_OPS_REPORT.md

outputs/
  PHASE1_EVAL.md         # gold evaluation: P/R/F1 by difficulty + missed-pair taxonomy
  PHASE1_EXPERIMENTS.md  # method ladder, ANN scaling, 13-config recall+cost, equal-budget comparison
  PHASE2_CLUSTERING.md   # full sweep (KMeans/Agglo/HDBSCAN/UMAP/BERTopic) + bridge metric
  PHASE2_SCALING.md      # clustering time + memory at n=400…32,000
  PHASE3_OPS_REPORT.md   # Haiku vs Qwen side-by-side: labels, narrative, escalation, cost
  cluster_labels.json    # 17 cluster labels + descriptions (active model: Qwen2.5/Ollama)
  llm_cache/             # SHA-256-keyed JSON cache of every LLM call
```
-->

---

## Quick Start

```bash
pip install -r requirements.txt

# Phase 1 — embed + deduplicate (frozen path)
python -m q19.phase1_pipeline

# Phase 2 — cluster (frozen path)
python -m q19.phase2_pipeline

# Phase 3 — label clusters + ops report
python -m q19.phase3_pipeline
```

All heavy steps are cached: embeddings on disk (content-hash keyed `.npy`), LLM responses in `outputs/llm_cache/` (SHA-256 keyed JSON). Re-runs are free and deterministic.

**LLM backends:** `src/q19/llm.py` uses an ACTIVE/DISABLED comment block to select between two backends. Default: local Ollama (`qwen2.5:7b-instruct-q4_K_M`, no cost). Alternative: Anthropic SDK (`claude-haiku-4-5-20251001`, requires `ANTHROPIC_API_KEY` in `.env`). Both were run and compared — see `outputs/PHASE3_OPS_REPORT.md`.

---

## Evaluation Discipline

`eval_holdout/duplicate_pairs.csv` is read **only** by `eval.py`, **only** at evaluation time. Every method choice and threshold was made *before and independently of* peeking at these labels:

- **Phase 1 threshold** — chosen from the *shape* of the score distribution (Otsu on bimodal cross-encoder output; fixed 0.75 cosine cutoff picked by manually inspecting a handful of pairs, never from gold recall).
- **Phase 2 k** — chosen from internal metrics (silhouette, Davies-Bouldin, Calinski-Harabasz) and convergence across three independent k-free methods. Gold was read once, after the freeze, to compute the bridge metric.
- **Method selection** — ranked by recall (the trustworthy metric here — see below), not by tuning against labels.

---

## Phase 1 — Duplicate Detection

### The Method Ladder

All methods ran in **cold-isolated subprocesses** (separate interpreter, no shared cache, fresh model load) so every number is each method's true standalone cost, not an artifact of run order.

| Rung | Method | What it captures |
|---|---|---|
| 0 | `exact_match` | Identical text after lowercasing — the floor |
| 1 | `tfidf_word` | Shared specific vocabulary (word/bigram TF-IDF cosine) |
| 1 | `tfidf_char` | Character n-gram overlap — typo-tolerant |
| 1 | `bm25` | Search-engine ranking: rewards rare shared words, dampens repeated ones |
| 1 | `jaccard` | Token set overlap — no weighting |
| 1 | `fuzzy` | Edit-distance ratio (RapidFuzz token_sort_ratio) after alphabetising words |
| 1b | `*_stemmed` | All Rung 1 methods re-run on Porter-stemmed text |
| 2 | `embeddings_bge_m3` | Semantic similarity via BAAI/bge-m3 (1024-dim transformer) |
| 3 | `hybrid_bm25_embeddings` | Rank-fusion: 0.7×embeddings + 0.3×BM25 (scores converted to percentile ranks first) |
| 4 | `cross_encoder_k30` | Two-stage: embeddings retrieve top-30 → bge-reranker-v2-m3 scores each candidate pair jointly |

**Speed/memory summary** (cold-isolated, n=400, 79,800 pairs):

| Method | Time (s) | Mem Δ (MB) |
|---|---|---|
| `exact_match` | 0.03 | 1 |
| `tfidf_word` | 0.01 | 2 |
| `embeddings_bge_m3` | 39.3 | 1,955 |
| `cross_encoder_k30` (retrieve+rerank) | 714.8 | 2,511 |

Full table: `outputs/PHASE1_EXPERIMENTS.md`.

### Threshold Selection

- **Embeddings (cosine):** smooth decay over 79,800 pairs, no clean valley. **Fixed cutoff 0.75**, chosen by inspecting a handful of pairs before any gold lookup.
- **Cross-encoder:** output is bimodal (reranker collapses non-duplicates toward 0, duplicates toward 1). **Otsu threshold (0.423)** — purely shape-based, gold-free.

### Results: 13-Configuration Grand Comparison

| Config | Backend | k | Recall | easy/med/hard | Flagged |
|---|---|---|---|---|---|
| `tfidf_word` | exact | — | 40/60 | 20/20/0 | 4,918 |
| **one-stage embeddings** | exact | — | **48/60** | 20/20/8 | 6,363 |
| one-stage embeddings | faiss-IVF | — | 48/60 | 20/20/8 | 6,149 |
| one-stage embeddings | faiss-HNSW | — | 48/60 | 20/20/8 | 6,154 |
| two-stage (CE) | exact | k=10 | 42/60 | 20/19/3 | 1,798 |
| two-stage (CE) | exact | k=20 | 44/60 | 20/20/4 | 2,379 |
| two-stage (CE) | exact | k=30 | 44/60 | 20/20/4 | 2,675 |
| two-stage (CE) | faiss-IVF | k=30 | 44/60 | 20/20/4 | 2,675 |
| two-stage (CE) | faiss-HNSW | k=30 | 44/60 | 20/20/4 | 2,675 |

Key findings:
- **ANN backends give identical recall** to exact. The choice is purely scale/cost, not quality.
- **One-stage embeddings wins on recall** (48/60) and is 5–13× faster than two-stage depending on mode.
- **Two-stage wins on precision** — flags only 2,675 pairs vs 6,363 — but tops out at 44/60 because the reranker genuinely judges some hard pairs as non-duplicates.
- **k barely moves two-stage recall** (42→44→44 for k=10→20→30). Bottleneck is the reranker's judgment, not retrieval.

**Frozen path:** one-stage embeddings (`BAAI/bge-m3`, cosine ≥ 0.75).

### Required Metric: P/R/F1 by Difficulty

Evaluated at the frozen operating threshold:

| Metric | embeddings (0.75) | cross-encoder (Otsu 0.423) |
|---|---|---|
| Recall | **0.80** (48/60) | 0.73 (44/60) |
| Precision | 0.008 | 0.016 |
| F1 | 0.015 | 0.031 |
| Flagged pairs | 6,363 | 2,675 |

**Precision and F1 are not trustworthy here.** The gold set is a non-exhaustive sample — many of the pairs flagged but not in the gold set are real duplicates. A better method finds more of them, accruing phantom false positives and *looking worse* on precision. We rank by **recall** (unaffected by unlabelled duplicates) and treat precision as context only.

Cross-encoder by difficulty (Otsu 0.423):

| Difficulty | Gold pairs | Caught | Recall |
|---|---|---|---|
| easy | 20 | 20 | 1.00 |
| medium | 20 | 20 | 1.00 |
| hard | 20 | 4 | **0.20** |

Hard pairs fail because the reranker retrieves most of them (cand-recall ≈ 0.93) but then genuinely scores them below threshold — the semantics are insufficient even for a cross-encoder.

### ANN vs Brute Force — Measured, Not Asserted

| n | Exact (s) | ANN (s) | Speedup |
|---|---|---|---|
| 400 | 0.007 | 0.012 | 0.58× |
| 2,000 | 0.056 | 0.124 | 0.45× |
| 8,000 | 0.448 | 0.864 | 0.52× |
| 16,000 | 1.642 | 2.027 | 0.81× |
| 32,000 | 6.029 | 4.404 | **1.37×** |

**Crossover: ~n=32,000.** Below that, brute force is faster. `ann.py` provides a drop-in `ann_candidate_pairs()` — swapping at scale is a one-parameter change in `phase1_pipeline.py`. Full report: `outputs/PHASE1_EXPERIMENTS.md`.

---

## Phase 2 — Clustering

### Algorithms

Five families compared on the same `BAAI/bge-m3` embeddings:

| Family | k control | Coverage |
|---|---|---|
| KMeans | fixed k | 100% |
| Agglomerative/Ward | fixed k | 100%, free merge hierarchy |
| HDBSCAN (raw) | auto — density | Partial (noise allowed) |
| UMAP + HDBSCAN | auto — density | Partial |
| BERTopic | auto — density | Partial |

All internal metrics scored in the **original embedding space**, including UMAP-based families — scoring in reduced space inflates separation artificially.

### K Selection

**KMeans sweep (k = 4 to 30, selected rows):**

| k | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|
| 4 | 0.102 | 2.761 | 31.2 |
| 8 | 0.137 | 2.180 | 26.9 |
| **15** | **0.179** | **1.858** | **23.5** |
| **17** | 0.171 | 1.921 | 22.4 |
| 25 | 0.171 | 1.860 | 18.4 |

Silhouette is *lowest* at small k and climbs as k grows — the opposite of the textbook pattern. Banking queries don't form 4–6 mega-topics; they form many narrow, partially-overlapping issue types. Gains flatten past k≈15 into a noisy plateau.

Three independent k-free methods all converge on 17:

| Method | Clusters found | Noise pts |
|---|---|---|
| HDBSCAN (raw, mcs=5) | **17** | 195/400 |
| UMAP+HDBSCAN (n_comp=5, mcs=8) | **17** | 9/400 |
| UMAP+HDBSCAN (n_comp=10, mcs=8) | **17** | 17/400 |
| BERTopic (mcs=8) | **17** | 9/400 |

**Frozen choice: KMeans, k=17.** Sits in the plateau; confirmed by count convergence from three algorithm families that were never told k; full coverage (every query assigned).

### Why KMeans

All methods were quality-tied at n=400. KMeans was chosen because it scales best on both time and memory across all methods tested:

| n | KMeans | Agglomerative | UMAP+HDBSCAN | UMAP+KMeans |
|---|---|---|---|---|
| 400 | 5.2s / 191 MB | 0.02s / 187 MB | 20.9s / 418 MB | 25.0s / 418 MB |
| 8,000 | 10.2s / 313 MB | 13.6s / 728 MB | 44.1s / 624 MB | 47.1s / 620 MB |
| 16,000 | 15.8s / 444 MB | 58.2s / **2,297 MB** | 45.5s / 750 MB | 48.6s / 751 MB |
| 32,000 | **27.3s / 707 MB** | timeout (>240s) | 65.6s / 1,001 MB | 61.0s / 1,001 MB |

At n=32,000 KMeans is the only method that completes within budget — 2.4× faster than UMAP-based methods and 30% cheaper on memory. Agglomerative's O(n²) memory growth makes it a non-starter at scale. All alternatives remain implemented in `phase2_pipeline.py`. Full experiment: `outputs/PHASE2_SCALING.md`.


---

## Phase 3 — Cluster Labelling and Ops Report

### Representative Selection

For each cluster: **query closest to centroid** + **4 nearest neighbours** within the cluster. Gives the LLM a focused view of the cluster's core intent without fringe cases. Implemented in `src/q19/label.py`.

### LLM Client Design

`src/q19/llm.py` supports two backends via an ACTIVE/DISABLED comment block:

- **Prompt-hash disk cache** (`outputs/llm_cache/<sha256>.json`): cache key = SHA-256 of full request JSON. Cache hit never touches the network. Doubles as an audit trail.
- **Retry + exponential backoff**: 3 attempts, backoff base 2.0s.
- **Token accumulation**: `usage_summary()` returns total tokens + cost (real for Anthropic, hypothetical at GPT-4o-mini rates for Ollama).

### Labelling: 17 LLM Calls, One Per Cluster

Prompt: system message requesting JSON only, user message with 5 representative queries and schema `{"label": "<3-6 word title-case name>", "description": "<one sentence>"}`. Temperature 0 for determinism. Up to 2 parse retries on malformed JSON.

**Results (sorted by volume):**

| Cluster | Label | Queries |
|---|---|---|
| 2 | Pending Payment Inquiry | 48 |
| 3 | Lost Card Assistance | 38 |
| 10 | Transfer Issues | 37 |
| 1 | Card Payment Issues | 27 |
| 7 | Card Delivery Issues | 22 |
| 12 | PIN Change Procedures | 21 |
| 13 | Unrecognized Card Payments | 21 |
| 9 | Update Personal Details | 20 |
| 11 | Identity Verification Issues | 20 |
| 15 | Extra Fee Inquiry | 20 |
| 16 | Card Activation | 20 |
| 4 | Top-Up Failure Reasons | 19 |
| 5 | Refund Assistance | 19 |
| 6 | Cancel Transfer | 18 |
| 0 | Duplicate Charges | 17 |
| 14 | PIN Unblock Queries | 17 |
| 8 | Contactless Troubleshooting | 16 |

Full labels + descriptions: `outputs/cluster_labels.json`.

### Ops Report: 1 LLM Call

`scripts/phase3_ops_report.py` aggregates cluster volumes and sends one LLM call with all 17 clusters as context. The model returns structured JSON: a 2–3 sentence executive summary naming significant clusters by volume and theme, a list of escalation clusters with reasoning (LLM-identified based on content severity, not just volume — e.g. fraud and identity clusters flagged even if mid-volume), and actionable recommendations. Python assembles this into `outputs/PHASE3_OPS_REPORT.md`.

Division of work: the code built the structure (tables, rankings, cost footnote); the LLM wrote the narrative and identified which clusters warrant priority attention. `scripts/phase3_compare.py` ran both Ollama and Haiku through the same pipeline and produced a side-by-side comparison document.

### Required Metric: LLM Cost

Both backends were run and compared (see `outputs/PHASE3_OPS_REPORT.md`):

| Step | Calls | Qwen2.5 7B / Ollama (default) | Claude Haiku 4.5 (comparison) |
|---|---|---|---|
| Cluster labelling (17 clusters) | 17 | hypothetical at GPT-4o-mini rates | actual Haiku pricing |
| Ops report (single call) | 1 | 817 tokens — hypothetical $0.0002 | 1,282 tokens — actual $0.0029 |
| **Total** | **18** | **$0.00 actual (local)** | **actual ~$0.003** |

The Ollama hypothetical is computed at GPT-4o-mini rates ($0.15/$0.60 per 1M tokens) as a reference point. Full per-call token breakdown in `outputs/llm_cache/`.

---

## Architecture Decisions and Tradeoffs

**Gold quarantine.** `eval_holdout/` is physically separated and read by exactly one file (`eval.py`). Thresholds were picked from score distributions; k from internal metrics. Both frozen before gold was read. On a 60-pair sample any method tuned on the labels would overfit to those specific phrasings.

**Subprocess isolation for fair benchmarking.** All Phase 1 method benchmarks and clustering scaling runs use fresh Python interpreters per config. Eliminates warm-cache effects (OS page cache, loaded model weights) that would make method B look cheaper because method A already paid the setup cost.

**ACTIVE/DISABLED docstring convention.** `phase1_pipeline.py` and `phase2_pipeline.py` keep every alternative that was built and measured — not just the frozen winner. Each function is tagged `ACTIVE` or `DISABLED` with a sentence on why it's off and when to flip it. Nothing was deleted because it "didn't win."


---

## What I Would Improve With More Time

**Phase 1:**
- **Cross-encoder fine-tuning on banking pairs.** bge-reranker-v2-m3 is general-purpose. Fine-tuning on manually-labelled banking pairs would directly address the 20% hard-pair recall gap.
- **De-bias precision via manual audit.** A sample of 50–100 predicted-positive pairs, manually labelled, would give a trustworthy estimate of true precision.
- **Pair-level connected components.** Currently the pipeline returns flagged pairs; `src/q19/graph.py` is already implemented for transitive grouping — just not wired to the output.

**Phase 2:**
- **Dedup before clustering.** Dense duplicate pockets pull centroids toward themselves and can create "duplicate artifact" clusters. Clustering the unique set would produce cleaner intent boundaries.
- **Cluster-drift monitoring.** A weekly cadence should compare new assignments to the previous week's — which clusters grew, which shrank, which new issue types emerged.

**Phase 3:**
- **Escalation classifier.** The current escalation signal is LLM-driven — prompt-sensitive, as the comparison study showed (Ollama defaulted to volume-based; Haiku correctly prioritised fraud/security regardless of volume). A lightweight classifier trained on historical escalation decisions would fix this.
- **Larger hosted model.** Tested with Claude Haiku 4.5 (actual ~$0.003/run). A larger model would improve escalation judgment and recommendation specificity. The ACTIVE/DISABLED block in `llm.py` makes this a four-line swap.

**Engineering:**
- **FAISS/HNSW at scale.** `ann.py` drop-in is already written; wiring at the measured n=32,000 crossover makes the pipeline production-ready.
- **Vector DB for persistence.** CSV → embedding at ingest, stored in Milvus/Qdrant/Pinecone. Weekly runs query the DB rather than re-encoding history.
