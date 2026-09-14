# build_svi.py
"""
Builds a literature-guided Socioeconomic Vulnerability Index (SVI) for the
150 historical GHMC wards using PCA on six Census 2011 variables.

Outputs:
  hyderabad_150_ward_SVI_2011.csv          -- per-ward scores + decompositions
  svi_pca_loadings.csv                     -- PCA loadings for the writeup
  svi_methodology.md                       -- variable justification + method
  svi_validation_report.txt                -- split-half + leave-one-out stability
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE = os.path.dirname(os.path.abspath(__file__))
WIDE = os.path.join(BASE, "hyderabad_150_ward_SUPER_wide_worldpop_v4.csv")

OUT_SVI    = os.path.join(BASE, "hyderabad_150_ward_SVI_2011.csv")
OUT_LOAD   = os.path.join(BASE, "svi_pca_loadings.csv")
OUT_METHOD = os.path.join(BASE, "svi_methodology.md")
OUT_VALID  = os.path.join(BASE, "svi_validation_report.txt")

# ------------------------------------------------------------------
# Load wide master
# ------------------------------------------------------------------
df = pd.read_csv(WIDE, low_memory=False)
print(f"Loaded {len(df)} wards, {df.shape[1]} columns")

# ------------------------------------------------------------------
# Resolve column names (some are doubled from the original upload)
# ------------------------------------------------------------------
def pick(*candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in columns")

c_pop    = pick("census2011_population_verified",
                "census2011_census2011_population_verified")
c_hh     = pick("census2011_No_HH",
                "census2011_census2011_No_HH")
c_p06    = pick("census2011_P_06",
                "census2011_census2011_P_06")
c_lit    = pick("census2011_literacy_rate",
                "census2011_census2011_literacy_rate")
c_nonwk  = pick("census2011_non_worker_rate",
                "census2011_census2011_non_worker_rate")
c_scst   = pick("census2011_sc_st_share",
                "census2011_census2011_sc_st_share")
c_marg   = pick("census2011_MARGWORK_P",
                "census2011_census2011_MARGWORK_P")
c_totwk  = pick("census2011_TOT_WORK_P",
                "census2011_census2011_TOT_WORK_P")

# ------------------------------------------------------------------
# Construct the six literature-guided variables
# ------------------------------------------------------------------
work = pd.DataFrame({
    "ward_id":   df["ward_id"],
    "ward_name": df.get("ward_name_readme"),
})

work["v_illiteracy"]      = 1.0 - pd.to_numeric(df[c_lit],   errors="coerce")
work["v_nonworker"]       = pd.to_numeric(df[c_nonwk],       errors="coerce")
work["v_sc_st_share"]     = pd.to_numeric(df[c_scst],        errors="coerce")

pop  = pd.to_numeric(df[c_pop], errors="coerce")
hh   = pd.to_numeric(df[c_hh],  errors="coerce")
p06  = pd.to_numeric(df[c_p06], errors="coerce")
marg = pd.to_numeric(df[c_marg], errors="coerce")
totw = pd.to_numeric(df[c_totwk], errors="coerce")

work["v_child_share"]    = p06  / pop
work["v_crowding"]       = pop  / hh
work["v_marginal_share"] = marg / totw.replace(0, np.nan)

FEATURES = ["v_illiteracy", "v_nonworker", "v_sc_st_share",
            "v_child_share", "v_crowding", "v_marginal_share"]

# ------------------------------------------------------------------
# Sanity: NaN count per feature
# ------------------------------------------------------------------
nan_counts = work[FEATURES].isna().sum()
print("\nFeature NaN counts:")
print(nan_counts.to_string())

# Drop any wards with NaN features (should be 0 for our data)
if nan_counts.sum() > 0:
    drop = work[FEATURES].isna().any(axis=1)
    print(f"WARNING: dropping {drop.sum()} wards with missing features")
    work = work.loc[~drop].reset_index(drop=True)

# ------------------------------------------------------------------
# Standardize -> PCA
# ------------------------------------------------------------------
X = work[FEATURES].values
scaler = StandardScaler()
Xz = scaler.fit_transform(X)

pca = PCA(n_components=len(FEATURES))
scores = pca.fit_transform(Xz)

evr = pca.explained_variance_ratio_
print("\nExplained variance ratio per PC:")
for i, r in enumerate(evr, 1):
    print(f"  PC{i}: {r:.4f}  (cumulative {evr[:i].sum():.4f})")

# ------------------------------------------------------------------
# Sign convention: PC1 must correlate POSITIVELY with the mean of the
# standardized vulnerability variables. Otherwise flip PC1.
# ------------------------------------------------------------------
mean_z = Xz.mean(axis=1)
pc1 = scores[:, 0]
corr = np.corrcoef(pc1, mean_z)[0, 1]
if corr < 0:
    pc1 = -pc1
    pca.components_[0] = -pca.components_[0]
    print(f"\nPC1 sign flipped (raw correlation with mean-z was {corr:.3f})")
else:
    print(f"\nPC1 kept as-is (correlation with mean-z = {corr:.3f})")

# ------------------------------------------------------------------
# Min-max scale PC1 -> 0..100
# ------------------------------------------------------------------
pc1_min, pc1_max = pc1.min(), pc1.max()
svi = (pc1 - pc1_min) / (pc1_max - pc1_min) * 100.0

work["svi_pc1_raw"] = pc1
work["svi_score"]   = svi

def band(s):
    if s <= 30:  return "green"
    if s <= 60:  return "yellow"
    return "red"
work["svi_band"] = work["svi_score"].apply(band)

# ------------------------------------------------------------------
# Per-ward contributions: loading_i * z_i  (this is what SHAP would show)
# ------------------------------------------------------------------
loadings = pca.components_[0]
contrib = Xz * loadings  # shape (n_wards, n_features)
contrib_df = pd.DataFrame(
    contrib,
    columns=[f"contrib_{f[2:]}" for f in FEATURES],
)
work = pd.concat([work.reset_index(drop=True), contrib_df], axis=1)

# ------------------------------------------------------------------
# Loadings table for the methodology doc
# ------------------------------------------------------------------
load_df = pd.DataFrame({
    "feature": FEATURES,
    "pc1_loading": loadings,
    "abs_loading": np.abs(loadings),
}).sort_values("abs_loading", ascending=False)
load_df["pc1_variance_share_pct"] = (load_df["abs_loading"] ** 2
                                     / (load_df["abs_loading"] ** 2).sum() * 100)

print("\nPC1 loadings:")
print(load_df.to_string(index=False))

# ------------------------------------------------------------------
# Validation 1: split-half stability of loadings (100 random splits)
# ------------------------------------------------------------------
rng = np.random.default_rng(42)
n = len(work)
loading_corrs = []
for _ in range(100):
    idx = rng.permutation(n)
    a, b = idx[:n//2], idx[n//2:]
    pa = PCA(n_components=1).fit(Xz[a])
    pb = PCA(n_components=1).fit(Xz[b])
    la, lb = pa.components_[0], pb.components_[0]
    # align signs
    if np.dot(la, lb) < 0:
        lb = -lb
    loading_corrs.append(np.corrcoef(la, lb)[0, 1])
sh_mean = float(np.mean(loading_corrs))
sh_min  = float(np.min(loading_corrs))
print(f"\nSplit-half loading stability: mean r = {sh_mean:.4f}, min r = {sh_min:.4f}")

# ------------------------------------------------------------------
# Validation 2: leave-one-out stability of the SVI score
# ------------------------------------------------------------------
loo_devs = []
for i in range(n):
    mask = np.ones(n, dtype=bool); mask[i] = False
    p = PCA(n_components=1).fit(Xz[mask])
    l = p.components_[0]
    if np.dot(l, loadings) < 0:
        l = -l
    # project all wards (including held-out one) under this fit
    score_i = Xz @ l
    # re-scale on the training subset only
    train = score_i[mask]
    lo, hi = train.min(), train.max()
    # held-out ward's score under the alternative fit
    s_alt = (score_i[i] - lo) / (hi - lo) * 100
    loo_devs.append(abs(s_alt - work["svi_score"].iloc[i]))
loo_mean = float(np.mean(loo_devs))
loo_max  = float(np.max(loo_devs))
print(f"Leave-one-out SVI deviation: mean |Δ| = {loo_mean:.2f} pts, max |Δ| = {loo_max:.2f} pts")

# ------------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------------
keep = (["ward_id", "ward_name", "svi_score", "svi_band", "svi_pc1_raw"]
        + FEATURES
        + [c for c in work.columns if c.startswith("contrib_")])
work[keep].to_csv(OUT_SVI, index=False)
load_df.to_csv(OUT_LOAD, index=False)

with open(OUT_VALID, "w") as f:
    f.write("SVI 2011 Validation Report\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"N wards: {n}\n")
    f.write(f"Explained variance PC1: {evr[0]:.4f}\n")
    f.write(f"Explained variance PC2: {evr[1]:.4f}\n")
    f.write(f"Cumulative PC1+PC2:     {evr[:2].sum():.4f}\n\n")
    f.write("Split-half loading stability (100 random 75/75 splits):\n")
    f.write(f"  mean r = {sh_mean:.4f}\n")
    f.write(f"  min  r = {sh_min:.4f}\n\n")
    f.write("Leave-one-out SVI score deviation:\n")
    f.write(f"  mean |delta| = {loo_mean:.2f} points on 0-100 scale\n")
    f.write(f"  max  |delta| = {loo_max:.2f} points on 0-100 scale\n")

print("\nSVI distribution:")
print(work["svi_score"].describe().to_string())
print("\nBand counts:")
print(work["svi_band"].value_counts().to_string())

print(f"\nWrote {OUT_SVI}")
print(f"Wrote {OUT_LOAD}")
print(f"Wrote {OUT_VALID}")

# ------------------------------------------------------------------
# Methodology doc
# ------------------------------------------------------------------
method_md = f"""# SVI 2011 — Methodology

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
| Illiteracy rate | 1 − literacy_rate | higher = more vulnerable | Universal education-deprivation indicator |
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
3. Run PCA. Retain PC1 (explained variance = {evr[0]:.4f}).
4. Fix sign so PC1 correlates positively with the mean standardized
   vulnerability vector (correlation check, auto-flip if needed).
5. Min-max rescale PC1 to 0-100.
6. Bands: 0-30 green, 31-60 yellow, 61-100 red.

## Per-ward decomposition
Each ward's SVI is decomposed into per-variable contributions:

    contribution_i = loading_i * z_i

This is the PCA-equivalent of a SHAP explanation: it says how much each
variable pushed this ward's score up or down relative to the mean ward.

## PC1 loadings
{load_df.to_string(index=False)}

## Validation
See `svi_validation_report.txt`. Two checks:

1. **Split-half loading stability.** PCA fitted on two random halves of the
   data (100 repeats) yields PC1 loadings that correlate at mean r = {sh_mean:.3f}.
   High stability = the score is not an artifact of the specific sample.

2. **Leave-one-out score stability.** Refitting PCA with each ward held out
   and re-projecting that ward changes its SVI by mean |Δ| = {loo_mean:.2f}
   points and at most |Δ| = {loo_max:.2f} points on the 0-100 scale. Small =
   no single ward drives the score.

## What this is NOT
- Not a poverty measure. We have no income or consumption data.
- Not a forecast. It is a 2011 snapshot.
- Not causal. It measures co-occurrence of known deprivation indicators.
"""

with open(OUT_METHOD, "w", encoding="utf-8") as f:
    f.write(method_md)
print(f"Wrote {OUT_METHOD}")