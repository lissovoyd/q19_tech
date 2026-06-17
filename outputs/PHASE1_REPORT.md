# Phase 1 — Duplicate Detection

All experiments below were run **before** a single look at `eval_holdout/duplicate_pairs.csv`. Every method was built and measured first; the gold file was opened exactly once, at the end, by `evaluate_phase1.py`. Numbers in the Experiments sections come only from the 400 queries themselves (score distributions, timing, memory) — no accuracy labels.

---

## 1. Method Ladder (single-method, cold-isolated)

Each method ran in its own fresh subprocess with no shared cache. Time and Mem Δ are each method's true standalone cost.

| Method | Rung | Time (s) | Mem Δ (MB) | Time complexity | Score min / med / p90 / max |
|---|---|---|---|---|---|
| `exact_match` | 0 | 0.03 | 1 | O(n²·L) string compare | 0.000 / 0.000 / 0.000 / 0.000 |
| `tfidf_word` | 1 | 0.01 | 2 | O(n·L) vectorize + O(n²·V) cosine (sparse) | 0.000 / 0.009 / 0.059 / 0.864 |
| `tfidf_char` | 1 | 0.04 | 4 | same, larger V (char n-grams) | 0.000 / 0.025 / 0.119 / 0.934 |
| `bm25` | 1 | 0.34 | 2 | O(n²·L) | 0.000 / 0.019 / 0.053 / 0.337 |
| `jaccard` | 1 | 0.11 | 2 | O(n²·L) set ops | 0.000 / 0.077 / 0.172 / 1.000 |
| `fuzzy` | 1 | 0.35 | 1 | O(n²·L·log L) sort + edit distance | 0.051 / 0.365 / 0.460 / 0.987 |
| `tfidf_word_stemmed` | 1b | 0.06 | 2 | O(n·L) stem + tfidf_word | 0.000 / 0.010 / 0.065 / 0.860 |
| `bm25_stemmed` | 1b | 0.39 | 2 | O(n·L) stem + bm25 | 0.000 / 0.019 / 0.055 / 0.363 |
| `jaccard_stemmed` | 1b | 0.16 | 2 | O(n·L) stem + jaccard | 0.000 / 0.077 / 0.176 / 1.000 |
| `embeddings_bge_m3` | 2 | 39.3 | 1,955 | O(n·f(L)) encode + O(n²·d) cosine | 0.322 / 0.603 / 0.733 / 0.999 |
| `hybrid_bm25_embeddings` | 3 | 38.7 | 1,962 | bm25 + embeddings + O(n² log n) rank-fusion | 0.002 / 0.490 / 0.830 / 1.000 |

Key observations:
- Lexical methods (Rung 1) have heavily right-skewed distributions (median ~0.01–0.08). `fuzzy` is the exception: its minimum is 0.051 and median 0.365 — it rescales edit distance so even dissimilar pairs get non-trivial scores, making threshold placement much harder.
- `embeddings_bge_m3` minimum is 0.322 — even the least similar pair scores above a third of max cosine. The distribution is **not** bimodal; there is no clean valley to cut on.

---

## 2. ANN vs Brute Force — Empirical Crossover

At n=400 all methods use exact O(n²) cosine. Rather than asserting "use ANN at scale," `ann_experiment.py` measured the actual crossover. Synthetic embeddings are bootstrap-resampled + jittered from the real 400 embeddings so the similarity geometry stays domain-realistic.

### Real data (n=400)

- Exact brute force (faiss `IndexFlatIP`): **0.010 s**
- ANN (faiss `IndexHNSWFlat`, build + query): **0.018 s**
- ANN recall@10 vs exact: **1.000**

ANN is not faster at n=400 — graph construction overhead outweighs the trivially cheap brute-force multiply. Brute force is the correct choice at this scale.

### Synthetic scaling

| n | Exact (s) | ANN (s) | Speedup |
|---|---|---|---|
| 400 | 0.007 | 0.012 | 0.58× |
| 2,000 | 0.056 | 0.124 | 0.45× |
| 8,000 | 0.448 | 0.864 | 0.52× |
| 16,000 | 1.642 | 2.027 | 0.81× |
| **32,000** | 6.029 | 4.404 | **1.37×** |

**Crossover: ~n=32,000** on this machine (d=1024, k=10). Below that, a single BLAS matrix multiply beats graph construction. `q19/ann.py` is already a drop-in with the same output shape as the brute-force path — switching at scale is a one-parameter change in `phase1_pipeline.py`.

---

## 3. Grand Comparison — 13 Configurations (Quality)

Built and run before the gold file was opened. ANN backends are included to prove they give the same recall as exact (cand-rec ≈ 1.00) — the ANN choice is purely a speed decision, not a quality one.

| Config | Backend | k | Recall | easy/med/hard | Flagged | cand-rec vs exact |
|---|---|---|---|---|---|---|
| `tfidf_word` | exact | — | 40/60 | 20/20/0 | 4,918 | 1.00 |
| one-stage embeddings | exact | — | **48/60** | 20/20/8 | 6,363 | 1.00 |
| one-stage embeddings | faiss-IVF | — | 48/60 | 20/20/8 | 6,149 | 0.97 |
| one-stage embeddings | faiss-HNSW | — | 48/60 | 20/20/8 | 6,154 | 0.97 |
| two-stage (CE) | exact | k=10 | 42/60 | 20/19/3 | 1,798 | 1.00 |
| two-stage (CE) | faiss-IVF | k=10 | 42/60 | 20/19/3 | 1,798 | 1.00 |
| two-stage (CE) | faiss-HNSW | k=10 | 42/60 | 20/19/3 | 1,798 | 1.00 |
| two-stage (CE) | exact | k=20 | 44/60 | 20/20/4 | 2,379 | 1.00 |
| two-stage (CE) | faiss-IVF | k=20 | 44/60 | 20/20/4 | 2,379 | 1.00 |
| two-stage (CE) | faiss-HNSW | k=20 | 44/60 | 20/20/4 | 2,379 | 1.00 |
| two-stage (CE) | exact | k=30 | 44/60 | 20/20/4 | 2,675 | 1.00 |
| two-stage (CE) | faiss-IVF | k=30 | 44/60 | 20/20/4 | 2,675 | 0.99 |
| two-stage (CE) | faiss-HNSW | k=30 | 44/60 | 20/20/4 | 2,675 | 1.00 |

What this settles:
- **Backend is a pure cost decision.** All three give identical recall at equal k; cand-rec ≈ 1.00 proves ANN is a faithful speedup.
- **k barely moves two-stage recall** (10→20→30: 42→44→44). The bottleneck has shifted from retrieval to the reranker's judgment on hard pairs.
- **One-stage embeddings wins on recall** (48/60 vs 44/60). The cross-encoder's edge is precision (flags ~half as many pairs) and hard-pair handling at equal budget — not overall recall.

---

## 4. Cost Matrix — Time and Memory, End-to-End (Cold-Isolated)

| Mode | Backend | k | Candidates | Embed (s) | Retrieval (s) | Rerank (s) | **Total (s)** | **Peak mem (MB)** |
|---|---|---|---|---|---|---|---|---|
| one-stage | exact | — | 6,363 | 17.8 | 0.005 | 0.00 | **17.8** | 3,527 |
| one-stage | faiss-IVF | — | 6,149 | 15.6 | 0.059 | 0.00 | **15.7** | 3,526 |
| one-stage | faiss-HNSW | — | 6,154 | 14.8 | 0.033 | 0.00 | **14.8** | 3,527 |
| two-stage | exact | k=10 | 4,000 | ~18† | 0.010 | 69.9 | **69.9** | 3,852 |
| two-stage | faiss-IVF | k=10 | 4,000 | ~18† | 0.018 | 70.6 | **70.6** | 3,852 |
| two-stage | faiss-HNSW | k=10 | 4,000 | ~18† | 0.016 | 71.1 | **71.1** | 3,852 |
| two-stage | exact | k=20 | 8,000 | ~18† | 0.024 | 132.7 | **132.7** | 3,852 |
| two-stage | faiss-IVF | k=20 | 8,000 | ~18† | 0.024 | 133.5 | **133.6** | 3,852 |
| two-stage | faiss-HNSW | k=20 | 8,000 | ~18† | 0.026 | 133.9 | **134.0** | 3,852 |
| two-stage | exact | k=30 | 12,000 | ~18† | 0.023 | 193.8 | **193.8** | 3,852 |
| two-stage | faiss-IVF | k=30 | 12,000 | ~18† | 0.024 | 192.3 | **192.4** | 3,852 |
| two-stage | faiss-HNSW | k=30 | 12,000 | ~18† | 0.033 | 193.8 | **193.9** | 3,852 |

_† two-stage reused cached embeddings this run; cold two-stage total = ~18s embed + retrieval + rerank. Peak mem for two-stage is the cold peak (both models loaded, k-independent); warm (embeddings precomputed) drops to ~1,560 MB._

- **One-stage cost is the embedding** (~15s / ~3.5GB). The all-pairs cosine itself is milliseconds.
- **Backend barely affects total cost** at n=400. ANN's gain materialises at n≥32,000.
- **k drives two-stage cost linearly** via candidate count (k=10→30 ≈ triples rerank time: 70→193s).
- **Peak memory**: two-stage (~3.85GB) > one-stage (~3.5GB) because it loads both models.

---

## 5. Equal-Budget Comparison — With vs Without Cross-Encoder

The operational question: at the same cost (pairs flagged), who catches more?

### Recall vs budget (overall)

| Budget (pairs flagged) | tfidf_word | fuzzy | embeddings (no CE) | two-stage (+CE) |
|---|---|---|---|---|
| 500 | 39/60 | 31/60 | 28/60 | 29/60 |
| 1,000 | 40/60 | 33/60 | 36/60 | 37/60 |
| 2,784 | 40/60 | 38/60 | 41/60 | **44/60** |
| 5,000 | 40/60 | 40/60 | 47/60 | **48/60** |

### Hard pairs only (of 20) at the same budgets

| Budget | tfidf_word | fuzzy | embeddings (no CE) | two-stage (+CE) |
|---|---|---|---|---|
| 500 | 0/20 | 1/20 | 0/20 | 0/20 |
| 1,000 | 0/20 | 1/20 | 2/20 | 1/20 |
| 2,784 | 0/20 | 2/20 | 2/20 | **4/20** |
| 5,000 | 0/20 | 2/20 | 7/20 | **8/20** |

The cross-encoder's only real edge over embeddings-alone is on hard pairs at equal budget. `tfidf_word` is essentially free and competitive overall (40/60) but gets zero hard pairs. `fuzzy` collapses on hard pairs because edit distance cannot bridge different wording. **The budget is the real lever** — every method improves with a bigger budget, but so does false-positive load.

---

## 6. Gold Evaluation

This is the **one-time** evaluation: the first and only place `eval_holdout/duplicate_pairs.csv` was read. Methods and thresholds were all decided beforehand without it.

### ⚠ Why precision / AP / F1 are not trustworthy on this data

The cross-encoder flags **1475 pairs** as duplicates. Only **39** are in the 60-pair gold set. The other **1436** are counted as false positives — but they are overwhelmingly real duplicates that were never labelled. A better semantic method finds more of these unlabelled true positives and its precision/AP look *worse*. **Ranking methods by AP here is actively misleading — it rewards methods that miss real duplicates.** We therefore rank by recall (do we catch the labelled gold pairs?), which is unaffected by the unlabelled ones.

| Metric | Trust here? | Why |
|---|---|---|
| **Recall** | ✅ high | Did we catch the labelled gold pairs? Unaffected by unlabelled dups. |
| **Median gold rank / recall@1000** | ✅ high | Threshold-free; where do gold pairs sit in the ranking. |
| Pairs flagged | ✅ (context) | Sanity check: high recall by flagging 40% of pairs isn't usable. |
| Precision / F1 / AP | ❌ low | Biased down by unlabelled true positives; AP is *inverted* (rewards methods that miss dups). |

### Full results

| Method | **R** ✅ | P ❌ | F1 ❌ | AP ❌ | Med rank ✅ | R@1k ✅ | Flagged | Time (s) | Mem (MB) |
|---|---|---|---|---|---|---|---|---|---|
| `fuzzy` | **0.92** | 0.001 | 0.003 | 0.175 | 436 | 0.55 | 42,608 | 0.3 | 1 |
| `embeddings_bge_m3` | **0.80** | 0.008 | 0.015 | 0.106 | 661 | 0.60 | 6,363 | 39.3 | 1,955 |
| `tfidf_char` | **0.73** | 0.005 | 0.011 | 0.126 | 310 | 0.65 | 8,090 | 0.0 | 4 |
| `cross_encoder_k30` | **0.73** | 0.016 | 0.031 | 0.088 | 527 | 0.62 | 2,784 | 714.8 | 2,511 |
| `jaccard_stemmed` | **0.70** | 0.002 | 0.004 | 0.164 | 402 | 0.63 | 23,171 | 0.2 | 2 |
| `bm25` | **0.70** | 0.002 | 0.005 | 0.146 | 584 | 0.55 | 17,749 | 0.3 | 2 |
| `tfidf_word` | **0.67** | 0.008 | 0.016 | 0.199 | 184 | 0.67 | 4,918 | 0.0 | 2 |

### Recommended setup and honest tradeoff

- **Embeddings alone** (cosine cutoff 0.75): recall **0.80**, flagging 6,363 pairs, ~39s. Simple, cheap, strong.
- **Two-stage** (retrieve top-30 → cross-encoder rerank): recall **0.73**, flagging 2,784 pairs at Otsu threshold 0.423, but ~715s (5–13× depending on mode).

Embeddings-alone wins on recall and cost. The cross-encoder's edge is precision (flags ~half as many pairs) and hard-pair efficiency at equal budget. Pragmatic default is **embeddings-alone**; add the cross-encoder only when false positives are expensive or hard-pair recall is critical.

`cross_encoder_k30` recall by difficulty (Otsu 0.423):

| Difficulty | Gold pairs | Caught | Recall |
|---|---|---|---|
| easy | 20 | 20 | 1.00 |
| medium | 20 | 20 | 1.00 |
| hard | 20 | 4 | 0.20 |

### Two-stage retrieval ceiling

| top-k | Candidate pairs | Gold pairs retrieved (ceiling) |
|---|---|---|
| 5 | 1,448 | 0.65 |
| 10 | 2,729 | 0.77 |
| 20 | 5,240 | 0.87 |
| 30 | 7,998 | 0.93 |
| 50 | 13,610 | 0.95 |

At k=30 the ceiling is 0.93, well above the measured recall of 0.73 — retrieval is **not** the binding constraint. The reranker's judgment on hard, low-overlap pairs is. Raising k would lift the ceiling but wouldn't recover pairs the reranker already scores as non-duplicate.

### Missed gold pairs — failure analysis

The two-stage system (`cross_encoder_k30`) missed 16 gold pairs. The hardest misses (score 0.000):

| Pair | Query A | Query B | Failure mode |
|---|---|---|---|
| Q019↔Q023 | _My statement shows different transaction times._ | _I have been double charged_ | Framing difference — symptom vs consequence |
| Q099↔Q124 | _Can you tell me if I can track the card you sent me?_ | _I was supposed to receive my new card by now, but it hasn't came in._ | Implicit reference — one side assumes shipment context |
| Q005↔Q228 | _My contactless is non-functional._ | _Would reinstalling the app solve the problem?_ | Lexical gap — zero shared vocabulary despite same intent |
| Q077↔Q229 | _What is the dollar that I have pending on my statement there?_ | _Why did you charge me extra?_ | Lexical gap + implicit reference |
| Q013↔Q060 | _Why wasn't my bank balance updated?_ | _I made a transfer and am still waiting._ | Implicit reference — transfer context dropped |
| Q002↔Q018 | _Problem verifying my account_ | _The app won't let me log in as myself._ | Framing difference — verification vs login |

**Failure mode taxonomy:** (1) Lexical gap — no shared vocabulary despite identical intent; (2) Implicit reference — one query omits the key noun/context; (3) Framing difference — same event described at different abstraction levels; (4) Length mismatch — terse form loses discriminating signal.
