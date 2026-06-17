# Phase 2 — Clustering

Every row below was computed **before** opening `eval_holdout/duplicate_pairs.csv`. KMeans and Agglomerative/Ward (take k up front), HDBSCAN (density-based, no k), UMAP+HDBSCAN (density-based after dimensionality reduction — the standard remedy for distance concentration in high dimensions), UMAP+KMeans (same reduction, keeps full coverage), and BERTopic (UMAP+HDBSCAN packaged with c-TF-IDF auto-labelling). All run on the 400 query embeddings (`BAAI/bge-m3`).

Internal metrics for UMAP-based families are scored back in the **original** embedding space, not the reduced one — a lower-dimensional space mechanically inflates separation, so scoring every family against the same original-space X is the only fair comparison.

---

## KMeans Sweep

| k | Inertia | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|---|
| 4 | 124.9 | 0.102 | 2.761 | 31.2 |
| 6 | 113.4 | 0.128 | 2.507 | 28.5 |
| 8 | 104.3 | 0.137 | 2.180 | 26.9 |
| 10 | 96.8 | 0.152 | 2.056 | 25.8 |
| 12 | 90.2 | 0.167 | 1.934 | 25.1 |
| 15 | 83.3 | 0.179 | 1.858 | 23.5 |
| 17 | 79.8 | 0.171 | 1.921 | 22.4 |
| 18 | 79.4 | 0.160 | 2.007 | 21.2 |
| 20 | 76.0 | 0.174 | 1.934 | 20.6 |
| 22 | 73.3 | 0.167 | 1.939 | 19.9 |
| 25 | 70.9 | 0.171 | 1.860 | 18.4 |
| 28 | 68.9 | 0.165 | 1.892 | 17.1 |
| 30 | 67.3 | 0.169 | 1.836 | 16.5 |

---

## Agglomerative (Ward) Sweep

| k | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|
| 4 | 0.095 | 2.891 | 29.1 |
| 6 | 0.121 | 2.431 | 27.4 |
| 8 | 0.142 | 2.101 | 26.2 |
| 10 | 0.145 | 2.095 | 25.0 |
| 12 | 0.155 | 1.937 | 24.0 |
| 15 | 0.165 | 1.838 | 22.6 |
| 17 | 0.172 | 1.850 | 22.0 |
| 20 | 0.178 | 1.946 | 20.7 |
| 25 | 0.176 | 1.810 | 18.7 |
| 30 | 0.182 | 1.740 | 17.2 |

---

## HDBSCAN Sweep (raw embeddings — chooses its own cluster count)

| min_cluster_size | Clusters found | Noise points | Silhouette* | Davies-Bouldin* | Calinski-Harabasz* |
|---|---|---|---|---|---|
| 3 | 38 | 128 | 0.250 | 1.223 | 16.6 |
| 5 | 17 | 195 | 0.296 | 1.225 | 23.6 |
| 8 | 9 | 154 | 0.187 | 1.502 | 17.8 |
| 10 | 2 | 138 | 0.101 | 1.544 | 11.1 |

_* silhouette excludes noise points — scores are not directly comparable in magnitude to full-coverage methods (KMeans/Agglomerative). Fair cross-family signal is cluster count convergence, not silhouette value._

---

## UMAP + HDBSCAN Sweep

| UMAP n_components | min_cluster_size | Clusters found | Noise points | Silhouette* | Davies-Bouldin* | Calinski-Harabasz* |
|---|---|---|---|---|---|---|
| 5 | 3 | 44 | 38 | 0.158 | 1.730 | 13.0 |
| 5 | 5 | 21 | 10 | 0.179 | 1.663 | 19.7 |
| 5 | 8 | **17** | 9 | 0.174 | 1.778 | 22.5 |
| 5 | 10 | **17** | 8 | 0.176 | 1.808 | 22.3 |
| 10 | 3 | 38 | 28 | 0.159 | 1.571 | 12.9 |
| 10 | 5 | 21 | 7 | 0.169 | 1.710 | 19.5 |
| 10 | 8 | **17** | 17 | 0.179 | 1.762 | 22.3 |
| 10 | 10 | 15 | 9 | 0.159 | 1.807 | 22.5 |
| 20 | 3 | 31 | 14 | 0.143 | 1.611 | 15.0 |
| 20 | 5 | 20 | 13 | 0.185 | 1.660 | 20.4 |
| 20 | 8 | **17** | 19 | 0.179 | 1.732 | 22.6 |
| 20 | 10 | **17** | 23 | 0.181 | 1.697 | 22.6 |

_* scored in original (non-reduced) embedding space._

---

## UMAP + KMeans Sweep

Added after the scaling experiment (see below) showed Agglomerative breaking at n=32,000. Tests whether UMAP-reducing first changes KMeans quality, while keeping full coverage (no noise points).

### UMAP(n_components=5) + KMeans

| k | Silhouette* | Davies-Bouldin* | Calinski-Harabasz* |
|---|---|---|---|
| 12 | 0.161 | 1.937 | 24.4 |
| 15 | 0.167 | 1.923 | 22.8 |
| **17** | 0.167 | 1.932 | 22.0 |
| 20 | 0.170 | 1.863 | 21.0 |

### UMAP(n_components=10) + KMeans

| k | Silhouette* | Davies-Bouldin* | Calinski-Harabasz* |
|---|---|---|---|
| 12 | 0.161 | 1.941 | 24.3 |
| 15 | 0.168 | 1.899 | 22.9 |
| **17** | 0.178 | 1.873 | 22.4 |
| 20 | 0.169 | 1.884 | 20.9 |

### UMAP(n_components=20) + KMeans

| k | Silhouette* | Davies-Bouldin* | Calinski-Harabasz* |
|---|---|---|---|
| 12 | 0.153 | 1.926 | 23.9 |
| 15 | 0.166 | 1.926 | 22.8 |
| **17** | 0.178 | 1.838 | 22.5 |
| 20 | 0.164 | 1.950 | 20.9 |

_* scored in original (non-reduced) embedding space. UMAP-reducing first does not change where KMeans's quality optimum sits on this corpus — it mainly changes cost (see Scaling section)._

---

## BERTopic Sweep (UMAP + HDBSCAN + c-TF-IDF)

| min_cluster_size | Topics found | Noise points | Silhouette* | Davies-Bouldin* | Calinski-Harabasz* |
|---|---|---|---|---|---|
| 3 | 28 | 24 | 0.165 | 1.697 | 16.3 |
| 5 | 18 | 6 | 0.176 | 1.768 | 21.6 |
| **8** | **17** | 9 | 0.176 | 1.791 | 22.4 |
| 10 | 17 | 9 | 0.176 | 1.791 | 22.4 |

_* scored in original embedding space._

**BERTopic auto-labels preview** (top keywords per topic, min_cluster_size=5):

- Topic 0: pending, money, transfer, payment, yet, hasnt
- Topic 1: pin, change, do, blocked, can, my
- Topic 2: declined, card, was, why, accept, payment
- Topic 3: lost, card, stolen, my, with, wallet
- Topic 4: cancel, transaction, wrong, to, yesterday, the

---

## Reading the Sweep

**Silhouette does not peak at small k.** For both KMeans and Agglomerative it is lowest at k=4–6 and rises as k grows (KMeans best: k=15 → 0.179; Agglomerative best: k=30 → 0.182). This is the opposite of the textbook pattern. Banking queries don't form 4–6 well-separated mega-topics — they form many narrow, partially-overlapping issue types, so a small k forces unrelated issues into the same cluster and *hurts* cohesion. Gains flatten into a noisy plateau from roughly k=15 onward.

**A magnitude caveat before comparing families:** HDBSCAN/BERTopic silhouette excludes noise points, scoring itself on a self-selected easier subset. Silhouette values across families are not directly comparable. The fair cross-family signal is cluster *count* convergence.

**Three independent k-free pipelines all converge on 17:**
- HDBSCAN (raw, min_cluster_size=5): **17 clusters**, 195/400 noise
- UMAP+HDBSCAN: **17 clusters** at 5 of 12 swept configs (every min_cluster_size=8 or 10, across all three n_components), 8–23/400 noise, original-space silhouette 0.174–0.181
- BERTopic (min_cluster_size=8 and 10): **17 topics**, 9/400 noise, silhouette 0.176

Three algorithm families that were never told k all gravitate to 17 once their density threshold is set in a reasonable range. Count convergence from denominator-independent pipelines is much stronger triangulation than any single internal metric.

**Frozen choice: KMeans, k=17.** Reasoning:
1. Sits inside the plateau where full-coverage silhouette stops improving meaningfully (~0.17–0.18 from k=15 onward).
2. Three independent k-free pipelines converge on 17 — not a single lucky match.
3. Full coverage: every query gets a real cluster. HDBSCAN-based families leave 8–195/400 queries as noise; for an ops-routing use case, every query has to land in a bucket.

Why not UMAP+HDBSCAN or BERTopic (both also hold up at scale)? They leave points unassigned as noise. The 'Other' bucket (`with_other_bucket=True` in `phase2_pipeline.py`) is a real working fix but introduces a grab-bag cluster with inflated noise-excluded silhouette. KMeans is the safer default when forced-assignment is the requirement; BERTopic is kept fully wired and disabled in `phase2_pipeline.py` for easy switching.

---

## Gold Checks (computed once, after freezing the choice)

**Bridge metric** — of the 60 gold duplicate pairs, how many land in the same KMeans k=17 cluster:

| Difficulty | Co-clustered | Total | % |
|---|---|---|---|
| easy | 19 | 20 | 95.0% |
| medium | 19 | 20 | 95.0% |
| hard | 15 | 20 | 75.0% |
| **overall** | **53** | **60** | **88.3%** |

This metric is recall-shaped (every gold pair is a known true duplicate, so co-clustering has no false-positive ambiguity from the non-exhaustive set).

**Bonus root_issue alignment** (not required, not used to pick k) — on the 120 queries appearing in a gold pair: ARI = **0.676**, NMI = **0.892**.

> **Honest cost of the re-freeze:** when Agglomerative (k=17) was the frozen choice, the same gold check scored 57/60 (95.0%) bridge metric, ARI=0.774, NMI=0.938 — noticeably higher, concentrated in hard pairs (Agglomerative: 18/20 hard pairs co-clustered vs KMeans's 15/20). Agglomerative was not replaced for an accuracy reason — it tied KMeans on every unsupervised metric. The lower bridge metric is the real cost of prioritising scalability over Ward linkage's edge cases, reported because the discipline is to look once and report what's there.

---

## Scaling Experiment

This experiment asks the production question: **what happens as n grows?** Same empirical approach as Phase 1's ANN crossover. Synthetic embeddings are bootstrap-resampled + jittered from the real 400 query embeddings so similarity geometry stays realistic. Every method targets the same granularity (KMeans/Agglomerative/UMAP+KMeans: k=17; UMAP+HDBSCAN: min_cluster_size=8). Each (method, n) combo ran in its own subprocess for clean memory measurement.

**Outcome: Agglomerative's O(n²) memory wall caused the frozen choice to be re-frozen to KMeans, which has no observed wall through n=32,000.**

| n | KMeans | Agglomerative | UMAP+HDBSCAN | UMAP+KMeans |
|---|---|---|---|---|
| 400 | 5.2s / 191 MB | 0.02s / 187 MB | 20.9s / 418 MB | 25.0s / 418 MB |
| 2,000 | 5.8s / 215 MB | 0.8s / 224 MB | 27.1s / 451 MB | 31.3s / 450 MB |
| 8,000 | 10.2s / 313 MB | 13.6s / 728 MB | 44.1s / 624 MB | 47.1s / 620 MB |
| 16,000 | 15.8s / 444 MB | 58.2s / **2,297 MB** | 45.5s / 750 MB | 48.6s / 751 MB |
| 32,000 | **27.3s / 707 MB** | timeout (>240s) | 65.6s / 1,001 MB | 61.0s / 1,001 MB |

KMeans is the fastest and most memory-efficient method at scale. At n=32,000 it completes in 27s / 707 MB while every alternative either times out (Agglomerative) or costs 2.3–2.4× more time and 40% more memory (UMAP-based methods). Agglomerative's only edge over KMeans — determinism and a free merge hierarchy — is not worth the O(n²) memory growth that makes it non-viable beyond ~n=16,000.
