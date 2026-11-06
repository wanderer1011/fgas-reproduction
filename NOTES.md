# NOTES — Reproducing FGAS (Phase 0, revision 2)

**Paper:** C. Liu, Z. Yuan, B. Chen, H. Chen, D. Peng. *Fuzzy granular anomaly detection using Markov random walk.* Information Sciences 646 (2023) 119400. DOI 10.1016/j.ins.2023.119400
**Official code:** https://github.com/BELLoney/FGAS
- `Code/FGAS.m` and `Code/FGAS.py`
- `Datasets/*.mat` for all 15 datasets, ~2 MB in total

Nothing has been downloaded yet; I read the code on GitHub.

**How I read the paper:** ScienceDirect blocked automated access with a CAPTCHA. Instead I rendered every PDF page to an image with PyMuPDF and read the equations, Algorithm 1, Tables 1–2 and Figs. 1–4 visually. Figs. 2 and 4 were re-rendered at 300 dpi. This revision replaces the earlier text-only notes. Every formula below is checked against the rendered page.

**Source tags:** **[P]** = stated in the paper · **[C]** = only in the official code · **[F]** = read off a figure · **[A]** = my assumption

**Notation:**
- 𝒪 = set of objects (samples), n = |𝒪|
- 𝒜 = set of attributes, m = |𝒜|
- **σ** = the granule threshold. It is FGAS's only tuned parameter, and the code also calls it `sigma`.
- d = damping factor
- ν = anomaly threshold (never used in the experiments)

---

## 1. Core method (in my own words)

FGAS gives every sample in a numeric table an anomaly score without using any labels.

1. **Similarity per attribute.** For each attribute a, compare every pair of samples with the fuzzy similarity `R_a(o,p) = 1 − |a(o) − a(p)|`, which needs attributes scaled to [0,1]. Set every similarity below σ to 0.
2. **Typicality per attribute.** For a sample o, the thresholded row `[o]_a` is its *fuzzy granule* on that attribute. Its relative cardinality `|[o]_a|/|𝒪|`, the average thresholded similarity of o to every sample, measures how typical o's value is on attribute a.
3. **Distance.** The *fuzzy granular distance* FGD between two samples adds up the squared differences of these typicality values over all attributes.
4. **Random walk.** Row-normalising the FGD matrix gives a Markov transition matrix. Transition probability grows with *distance*, so the walker drifts toward atypical points.
5. **Scores.** A damped power iteration `π ← d + (1−d)·πP` with d = 0.1 runs from a uniform start until it converges. The stationary vector π is min-max normalised into the anomaly score; higher means more anomalous.

## 2. Algorithm details stated in the paper

| Item | Exact statement | Src |
|---|---|---|
| Data table | `D = ⟨𝒪, 𝒜⟩`; a(o) is the value of o on attribute a. There is no decision attribute. | P |
| Fuzzy relation (Eq. 1) | `R_B : 𝒪×𝒪 → [0,1]`, reflexive and symmetric, with `R_B(o,p) = min_{a∈B} R_a(o,p)` | P |
| Similarity used | `R_a(o,p) = 1 − |a(o) − a(p)|` | P §4.1 |
| Granular structure (Eq. 2) | `G(R_B) = {[o_1]_B, …, [o_n]_B}`, where `[o_i]_B = (R_B(o_i,o_1), …, R_B(o_i,o_n))` and `|[o_i]_B| = Σ_j R_B(o_i,o_j)` | P |
| Parameterised granule (Eq. 3) | `[o]_B(p) = 0 if R_B(o,p) < σ; R_B(o,p) if R_B(o,p) ≥ σ`, with σ ∈ [0,1) | P |
| **FGD (Eq. 7)** | `FGD(o,p) = Σ_{a∈𝒜} ( |[o]_a|/|𝒪| − |[p]_a|/|𝒪| )²`. **Squared, with no square root.** | P |
| Distance matrix | `D = [FGD(o,p)]_{n×n}`. Note that the paper reuses the symbol D for the data table. | P |
| Transition matrix (Eq. 8) | `P = E⁻¹D`, with `E = diag(e_1…e_n)`, `e_i = Σ_j FGD(o_i,o_j)`. Equivalently `p_ij = FGD(o_i,o_j)/Σ_j FGD(o_i,o_j)`. | P |
| Damped walk (Eq. 9) | `π^(t+1) = d + (1−d)·π^(t)·P`, with **d = 0.10** | P |
| Initialisation | `π^(0) = [1/|𝒪|, …, 1/|𝒪|]` | P |
| Stopping rule | `until ‖π^(t+1) − π^(t)‖₁ ≤ 10⁻³` (Algorithm 1, line 17) | P |
| Score (Eq. 10) | `AS(o_i) = (π_i − min π)/(max π − min π)` | P |
| Binary decision | o_i is an anomaly if `AS(o_i) > ν`. ν is never set: evaluation uses no threshold. | P |
| Complexity | `O(|𝒜||𝒪|²)` | P |
| σ grid | "We adjust σ from 0 to 1 in steps of 0.05 to find the optimal result" | P §4.1 |
| Missing values | "maximum frequency method", i.e. fill with the most frequent value (mode) | P §4.1 |

## 3. Experiments

All experiments use 15 numeric benchmark datasets. There is **no train/test split**: every method scores the full dataset, and AUC is computed against the ground-truth outlier labels. Hardware, runtime and software versions are **not reported**. The figures are clearly MATLAB output [F].

### Datasets (Table 1)

| # | Name / repo file | Abbr | #attr | #obj | #outl |
|---|---|---|---|---|---|
| 1 | Cardio / cardio | Card | 21 | 1831 | 176 |
| 2 | Diabetes_tested_positive_26_variant1 | Diab | 8 | 526 | 26 |
| 3 | Ecoli | Ecoli | 7 | 336 | 9 |
| 4 | Ionosphere_b_24_variant1 | Ionos | 34 | 249 | 24 |
| 5 | Iris_Irisvirginica_11_variant1 | Iris | 4 | 111 | 11 |
| 6 | Musk | Musk | 166 | 3062 | 97 |
| 7 | Pageblocks_1_258_variant1 | Page | 10 | 5171 | 258 |
| 8 | Pima_TRUE_55_variant1 | Pima | 9 | 555 | 55 |
| 9 | Satellite | Sate | 36 | 6435 | 2036 |
| 10 | Sonar_M_10_variant1 | Sonar | 60 | 107 | 10 |
| 11 | Thyroid | Thyro | 6 | 3772 | 93 |
| 12 | Wbc_malignant_39_variant1 | Wbc | 9 | 483 | 39 |
| 13 | Wdbc_M_39_variant1 | Wdbc | 31 | 396 | 39 |
| 14 | Wine | Wine | 13 | 129 | 10 |
| 15 | Wpbc_variant1 | Wpbc | 33 | 198 | 47 |

The paper cites two sources: github.com/BElloney/Outlier-detection and ODDS (odds.cs.stonybrook.edu). The FGAS repo bundles all 15 as `.mat` files. Sample sizes range from 107 to 6435, and dimensions from 4 to 166.

### E1 — AUC comparison → **Table 2** (main result)
- **Methods:** FGAS plus 13 baselines: DIS [14], COF [29], FastABOD [16], INFLO [13], kNN [27], LDOF [38], LoOP [15], ODIN [9], DCROD [18], ECOD [19], IForest [20], OutRanka [24], WNINOD [31].
- **Hyperparameters [P]:**
  - FGAS: σ from 0 to 1 in steps of 0.05, best result reported; d = 0.1.
  - COF, FastABOD, INFLO, kNN, LDOF, LoOP, ODIN, DCROD, OutRanka: k = 1…60 in steps of 1, best result reported.
  - IForest: 100 base estimators.
  - WNINOD: its parameter in [1, 10] in steps of 1; outlier score = reciprocal of its inlier score.
  - DIS: the authors "define the distance AS" instead of binary distance-based outliers. No formula is given.
  - ECOD: no parameters.
- **Metric:** ROC AUC.
  - Eq. 13: `AUC = (1/(|OS_t||OS_t^c|)) Σ_{o_i∈OS_t, o_j∈OS_t^c} P_ij`, with P_ij = 1 / 0.5 / 0 when AS(o_i) is >, = or < AS(o_j).
  - Reported to 3 decimals.
  - The table shows the best value per row in bold, plus a "1st order" row counting wins.
- **FGAS targets:**

| Card | Diab | Ecoli | Ionos | Iris | Musk | Page | Pima | Sate | Sonar | Thyro | Wbc | Wdbc | Wine | Wpbc | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| .953 | .989 | .955 | 1.000 | 1.000 | 1.000 | .960 | .971 | .734 | .995 | .985 | .997 | .998 | .885 | .594 | **.934** |

FGAS ranks first on 12 of 15 datasets.

### E2 — ROC curves → **Fig. 1**
- TPR and FPR follow Eqs. 11–12, ranking samples by descending score:
  - `TPR(ord) = |OS_d(ord) ∩ OS_t| / |OS_t|`
  - `FPR(ord) = |OS_d(ord) − OS_t| / |OS_t^c|`
- **Layout [F]:**
  - 5 rows × 3 columns of panels, (a) Card through (o) Wpbc.
  - Axes are FPR (%) vs TPR (%) on 0–100 scales. The x-limit varies per panel (MATLAB auto-range, e.g. 0–80 on Ionos, 0–90 on Diab and Sonar, 0–70 on Wbc).
  - All 14 methods share one panel, with a legend in the lower right. FGAS is black; the other colours are MATLAB defaults.
- Paper's claim: FGAS's curve on Iris and Musk hugs the top-left corner.

### E3 — Attribute-noise robustness → **Fig. 2**
- **Procedure [P]:** following [43], "the sample ⌈xn⌉ with a specific attribute value is replaced by a random value". A numeric attribute is replaced by a random value between that attribute's min and max, giving noise level x·100%. "⌈xn⌉ refers to n samples of the same attribute."
- **Layout [F]:**
  - Noise level x ∈ {0, 0.05, …, 0.50}, 11 points.
  - Two panels: (a) datasets 1–8 and (b) datasets 9–15.
  - FGAS AUC only, one line per dataset with markers. No error bars, so each point looks like a single run.
- **Readings [F]:**
  - Most curves are flat.
  - Ecoli drops from 0.955 to about 0.905 at x = 0.4, then recovers to about 0.923.
  - Page drops from 0.960 to about 0.933, then rises to about 0.946.
  - Diab slides from 0.989 to 0.982.
  - Pima slides from 0.971 to 0.960.

### E4 — Statistical tests → **Fig. 3**
- Friedman test, then Nemenyi test, at α = 0.1 over k = 14 algorithms and N = 15 datasets (following [35]).
- Paper reports: τ_F = **15.1347** with F(13,182); critical value **1.5610**; CD₀.₁ = **4.7659**.
- **Layout [F]:**
  - The CD diagram's axis runs from **14 on the left to 1 on the right**. FGAS sits at about 13.4 on the left and FastABOD at about 2.3 on the right.
  - So the paper ranks **higher AUC = higher rank number** (equivalent to the usual ranking mirrored: 15 − 1.63 = 13.37).
  - The CD bar is drawn at the top.
- **Pre-check done on the paper's own Table 2 numbers** (ties share the average rank):
  - τ_F = **15.071**. This is 0.4 % off; likely caused by unrounded AUCs or tie handling.
  - Critical value **1.5610** (exact match).
  - CD **4.7659** with q₀.₁₀ = 3.12 (exact match).
  - FGAS average rank 1.63 (13.37 mirrored).
  - Column averages and the "1st order" row match.

### E5 — σ sensitivity → **Fig. 4**
- FGAS AUC vs σ, one line per dataset, in two panels (datasets 1–8 and 9–15).
- **Plotted σ grid [F]: 0, 0.05, …, 0.95 (20 points). σ = 1 is not plotted.**
- Readings: the best σ is often high, about 0.8–0.9:
  - Diab ≈ 0.989 at 0.85–0.9
  - Pima ≈ 0.971 at 0.8
  - Sate ≈ 0.734 at 0.8
  - Wine ≈ 0.885 at 0.8
  - Thyro ≈ 0.985 at 0.95
  - Iris = 1.0 at 0.85
- Several curves collapse at σ = 0.95: Iris, Pima, Page, Ecoli and Wine.

## 4. Evaluation protocol
- Each method outputs one score per sample. AUC is computed over all samples against the binary labels.
- **The best AUC over each method's grid is reported, which is oracle tuning that uses the labels.** That's an optimistic upper bound for unsupervised detection. We will follow the same protocol for comparability and say so in the results.
- FGAS is deterministic. IForest and the noise injection are random.

---

## 5. Ambiguities & Missing Details

Each item lists the gap, then my **proposed default (flagged ASSUMPTION)**. ⚠ marks items where I'd like your decision.

1. **⚠ FGD: paper and official code disagree.**
   - The paper's Eq. 7, confirmed visually, is the **sum of squared** per-attribute differences.
   - The official code (`Dis = Dis + pdist2(temp,temp)` on a single column) uses the **sum of absolute** differences.
   - These give different transition matrices and different rankings.
   - The paper also says FGD is "a qualified distance measure". The squared form isn't a metric (it breaks the triangle inequality); the L1 form is.
   - ~~ASSUMPTION: implement both as a config switch (`fgd: squared | abs`). Report squared (paper = spec) as the primary result and code-L1 as a variant.~~
   - **RESOLVED (M3, 2026-11-06, user decision):** with abs FGD, the Iris and Wine σ-curves match Fig. 4 point for point and match Table 2 exactly; squared FGD does not. **Headline = abs (official code).** Squared (Eq. 7 as printed) is always run and reported as a variant. See `results/M3_smoke_and_benchmark.md`.
2. **Stopping tolerance.** The paper uses 1e-3; the code iterates while the change is > 1e-4.
   - **ASSUMPTION:** 1e-3, as the spec says, plus a check that 1e-4 gives the same AUCs to 3 decimals. Also a max-iteration cap (e.g. 10,000) as a safeguard, which neither the paper nor the code has.
3. **Damping term d.**
   - The paper calls it both a vector and a scalar ("π = [π₁…π_n] indicates a probability distribution. d is a damping factor"); the code adds the scalar 0.1 to every entry.
   - Its fixed point `0.1·1ᵀ(I − 0.9P)⁻¹` is a positive multiple of the version with a uniform restart vector. Eq. 10's min-max scaling cancels the multiple, so **the scores are identical**. Only the absolute stopping threshold changes meaning.
   - **ASSUMPTION:** scalar d as written in Eq. 9 and the code.
4. **σ grid endpoint.**
   - Eq. 3 says σ ∈ [0,1). The text says "0 to 1 in steps of 0.05". Fig. 4 plots 0…0.95.
   - ~~ASSUMPTION: search 0…0.95 (20 values), matching Eq. 3 and Fig. 4. Log σ = 1.0 separately for information only.~~
   - **RESOLVED (M4):** the assumption was wrong. Table 2 is the max over **σ ∈ {0, 0.05, …, 1.0} (21 values)**, as the text says. Ecoli (0.9555) and Wpbc (0.5941) peak at σ = 1.0. Fig. 4 simply doesn't plot σ = 1.0. The config now uses 0…1. `results/table2_fgas.md` shows 15/15 matches.
5. **⚠ Fig. 4 doesn't fully agree with Table 2.**
   - **Ecoli:** Table 2 and Fig. 2 (at noise 0) both give **0.955**, but the Ecoli curve in Fig. 4 peaks around **0.912** (σ = 0.6). The legend hides σ = 0.85, but a point at 0.955 would sit above the legend box and isn't visible.
   - **Wpbc:** the visible maximum in Fig. 4 is about 0.51 (σ ≤ 0.8). Its σ = 0.85–0.95 points are hidden behind the legend, so 0.594 could be there; this can't be checked.
   - **Card:** Fig. 4 peaks around 0.952 vs 0.953 in the table. That's within rounding.
   - **Proposal:** treat Table 2 as the target, record our full σ curves, and report the gap for each dataset. I won't try to explain the Ecoli gap beyond noting it.
   - **RESOLVED (M4):** no inconsistency. Fig. 4 is consistent with Table 2 once σ = 1.0 is included (see #4). Our σ = 0…0.95 curves reproduce Fig. 4, including Ecoli's 0.912 peak at σ = 0.6 and Wpbc's ≈0.51–0.54 hidden behind the legend.
6. **⚠ Fig. 2 (noise level 0) doesn't fully agree with Table 2.**
   - Thyroid at noise 0 is about **0.959** in Fig. 2 vs **0.985** in Table 2.
   - Every other dataset at noise 0 matches Table 2. That suggests the noise experiment used per-dataset best σ but a different setting for Thyroid, or some other inconsistency.
   - ~~ASSUMPTION: re-tune σ at every noise level, matching the Table 2 protocol, and also report the variant with σ fixed at the clean best value. Thyroid's mismatch will be flagged, not forced.~~
   - **RESOLVED (M4):** Fig. 2's noise-0 values equal the best AUC over a **coarser grid, σ ∈ {0, 0.1, …, 1.0}**, for all 15 datasets. Thyroid's best on that grid is 0.9591 at σ = 0.9 (0.95 isn't on it); Sonar is 0.9938 at 0.7. So Fig. 2 used a step-0.1 σ grid.
   - **Still unknown:** whether σ was re-tuned at each noise level or fixed. **New proposal for M7:** σ grid {0, 0.1, …, 1.0}, re-tuned at each noise level (main), plus σ fixed at the clean step-0.1 best (secondary). Needs your OK before M7.
7. **Noise-injection details are missing:**
   - ⌈xn⌉ corrupted values *per attribute* (each attribute independently) or for one chosen attribute
   - whether outliers can be corrupted
   - number of repeats and seeds (the figure looks like one run)
   - whether noise is added before or after min-max scaling

   **ASSUMPTION:** for **each** attribute independently, pick ⌈x·n⌉ random samples and replace their value with U[min_a, max_a] of the raw attribute, then min-max scale. All samples are eligible. Use 5 seeds and plot the mean (with a band), plus seed 0 alone in paper format.
8. **Preprocessing / scaling.** The paper never says to scale, but `1 − |a(o) − a(p)|` only works on [0,1]. The code says "All attributes should be normalized into [0,1]", and its demo uses `normalize(…,'range')`.
   - **ASSUMPTION:** per-attribute min-max scaling to [0,1] on the whole dataset. Constant attributes become 0.
9. **Missing values.** The paper fills them with the mode, but doesn't say before or after scaling. The repo `.mat` files may already be complete.
   - **ASSUMPTION:** check for NaNs first. If any exist, fill with the per-column mode on raw values, then scale.
10. **Attribute counts in Table 1 may include a label or ID column.** Pima shows 9 (the original has 8), Wdbc 31 (original 30), Wpbc 33.
    - **ASSUMPTION:** in Phase 2, inspect each `.mat` file and identify the label column or field. Report the actual feature counts vs Table 1, and stop and ask if anything is ambiguous.
11. **Data source.** Two repos are cited, and the FGAS repo bundles its own copies.
    - **ASSUMPTION:** use the FGAS-repo `.mat` files as the canonical data.
12. **Rows with zero total distance.** If every FGD in a row is 0 (possible with duplicates when σ is high), `E⁻¹` is undefined. Neither the paper nor the code handles this.
    - **ASSUMPTION:** detect it and log it. That σ gets AUC = NaN and is excluded from the best-σ search.
13. **AUC ties.** Eq. 13 counts ties as 0.5, which is the same as `sklearn.metrics.roc_auc_score`.
    - **ASSUMPTION:** use sklearn, with a unit test checking it against a direct implementation of Eq. 13.
14. **⚠ Baselines are unspecified.**
    - No implementations, libraries or versions are named, and DIS has no formula.
    - Seven methods (DIS, INFLO, LDOF, ODIN, DCROD, OutRanka, WNINOD) have no standard Python package.
    - kNN's variant (k-th distance vs mean distance) and the distance metric aren't stated.
    - **PROPOSAL:**
      - Fully reproduce the FGAS column.
      - Re-run the baselines PyOD provides (kNN, COF, FastABOD via `ABOD(method='fast')`, ECOD, IForest) and LoOP (via PyNomaly) on the paper's grids.
      - Copy the remaining columns from Table 2, labelled "paper-reported, not re-run".
      - Re-implementing the rest from their original papers would be a separate, larger job.
15. **IForest randomness.** No seed or repeat count is given.
    - **ASSUMPTION:** 5 seeds; report mean ± std.
16. **Friedman/Nemenyi details.** The q_α source isn't stated, and Demšar's table stops at k = 10. q₀.₁₀(14) = 3.12 reproduces CD = 4.7659 exactly.
    - **ASSUMPTION:** average ranks with ties shared, q₀.₁₀ = 3.12. Draw the CD diagram in the paper's mirrored orientation (14 on the left, 1 on the right).
17. **Fig. 1 operating point.** It isn't said which k or σ each ROC curve uses.
    - **ASSUMPTION:** each method's best AUC setting from Table 2.
18. **Compute.** Not reported.
    - Satellite needs about 6435² × 36 ≈ 1.5×10⁹ operations per σ, and each n×n float64 matrix takes ~330 MB.
    - That's fine for vectorised numpy on a laptop, taking minutes for the full σ sweep. The noise experiment multiplies this by 11 levels × seeds.

### Remaining unknowns that can't be resolved from the paper or code
- Why Ecoli's Fig. 4 curve and Thyroid's Fig. 2 noise-0 point disagree with Table 2 (#5 and #6).
- Which FGD form (squared vs abs) produced the published numbers (#1). Our runs will provide evidence on this.
