# SVI 2011 — Methodology

## Purpose
A Socioeconomic Vulnerability Index (0-100) for each of the 150 historical
GHMC wards, derived from Census 2011 using Principal Component Analysis.

This is a **cross-sectional baseline**, not a forecast. Downstream modules
project this baseline forward using WorldPop and UDISE pressure signals.

## Why PCA
OECD Handbook on Constructing Composite Indicators (2008) recommends PCA
when the goal is to combine correlated indicators without arbitrary weights.
PCA picks the linear combination of standardized inputs that captures the
most variance — the data determines the weights.

## Variables (literature-guided selection)
Each variable is standard in Indian socioeconomic deprivation literature
(NITI MPI framework, Census 2011 deprivation studies, TCPO SVI work).

| Variable | Definition | Direction | Rationale |
|---|---|---|---|
| Illiteracy rate | 1 - literacy_rate | higher = more vulnerable | Universal education-deprivation indicator |
| Non-worker rate | non_worker_rate | higher = more vulnerable | Labour-force exclusion |
| SC+ST share | sc_share + st_share | higher = more vulnerable | Historically marginalised groups |
| Child (0-6) share | P_06 / population | higher = more vulnerable | Dependency burden |
| Household crowding | population / No_HH | higher = more vulnerable | Housing-quality proxy |
| Marginal worker share | MARGWORK_P / TOT_WORK_P | higher = more vulnerable | Precarious employment |

## Variables deliberately excluded
- Sex ratio (TOT_F / TOT_M): direction contested in the literature.
- Female share: not a deprivation signal by itself.
- Household count: a scale variable, not a deprivation variable.

## Method
1. Compute the six variables per ward.
2. Standardize (z-score) each variable across the 150 wards.
3. Run PCA. Retain PC1 (explained variance = 0.5253).
4. Fix sign so PC1 correlates positively with the mean standardized
   vulnerability vector.
5. Min-max rescale PC1 to 0-100.
6. Two band systems are provided in the output:
   - `svi_band_absolute`: fixed thresholds 0-30 green, 31-60 yellow, 61-100 red.
   - `svi_band_percentile`: terciles of the empirical SVI distribution
     (bottom third green, middle third yellow, top third red). Use this for
     the map; it is informative even when the absolute distribution is
     skewed.

## PC1 loadings

         feature  pc1_loading  abs_loading  pc1_variance_share_pct
      v_crowding     0.505116     0.505116               25.514190
   v_child_share     0.467257     0.467257               21.832955
     v_nonworker     0.451158     0.451158               20.354376
    v_illiteracy     0.348599     0.348599               12.152124
   v_sc_st_share    -0.330021     0.330021               10.891388
v_marginal_share     0.304220     0.304220                9.254966

## Note on SC/ST share (negative loading)

SC/ST share loads at -0.330 on PC1. In the standard all-India SVI
framework, SC/ST share loads positively because of entrenched
caste-linked rural deprivation. In Census 2011 Hyderabad, this pattern
inverts: wards with higher SC/ST share tend to score lower on the other
five vulnerability dimensions.

Investigation: SC/ST populations in Hyderabad are concentrated in
peripheral and industrial wards (e.g. Ward 3 Cherlapally, Ward 132
Jeedimetla, Ward 134 Alwal) that are newer, more literate, and less
crowded than the central old-city belt that drives the top of the
vulnerability distribution. The negative loading is therefore a
Hyderabad-specific spatial fact, not a data error.

We retain SC/ST share for framework completeness — it is a standard input
in the NITI MPI and TCPO SVI literature — and document the anomaly rather
than silently dropping it. A sensitivity check (PCA without SC/ST) yields
substantively similar rank ordering across wards.

## Per-ward decomposition
Each ward's SVI is decomposed into per-variable contributions:

    contribution_i = loading_i * z_i

This is the PCA-equivalent of a SHAP explanation: it says how much each
variable pushed this ward's score up or down relative to the mean ward.

## Validation
See `svi_validation_report.txt`. Two checks:

1. **Split-half loading stability.** PCA fitted on two random halves of the
   data (100 repeats) yields PC1 loadings that correlate at mean r = 0.97.
   High stability = the score is not an artifact of the specific sample.

2. **Leave-one-out score stability.** Refitting PCA with each ward held out
   and re-projecting that ward changes its SVI by mean |delta| = 0.25
   points. The maximum deviation (12.35 points) occurs for boundary wards
   whose score sits near 0 or 100 — a known property of min-max scaling,
   not a data problem.

## What this is NOT
- Not a poverty measure. We have no income or consumption data.
- Not a forecast. It is a 2011 snapshot.
- Not causal. It measures co-occurrence of known deprivation indicators.
