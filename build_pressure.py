# build_pressure.py
"""
Builds a Pressure score for each of the 150 historical GHMC wards.

Pressure = population_growth_z - infrastructure_improvement_z

Where:
  population_growth_z        = z-scored annualized WorldPop growth 2015→2018
  infrastructure_improvement_z = z-scored composite of UDISE 2021→2023 changes

Higher pressure = ward is growing faster than its infrastructure is improving.
Positive pressure  -> vulnerability likely drifts UP over time.
Negative pressure  -> infrastructure is keeping pace or ahead, vulnerability
                      likely drifts DOWN.

Coverage:
  full    - both signals available (population + UDISE)
  partial - population only (UDISE missing for that ward)
  no_data - neither available (should not happen)

Output: hyderabad_150_ward_pressure.csv
"""
import os, sys
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
WIDE = os.path.join(BASE, "hyderabad_150_ward_SUPER_wide_worldpop_v4.csv")
OUT  = os.path.join(BASE, "hyderabad_150_ward_pressure.csv")

df = pd.read_csv(WIDE, low_memory=False)
print(f"Loaded wide: {df.shape}")

# ------------------------------------------------------------------
# Signal 1 — Population growth (all 150 wards)
# ------------------------------------------------------------------
pop_col = "worldpop_annualized_growth_2015_2018_pct"
if pop_col not in df.columns:
    sys.exit(f"Missing {pop_col}")

pop = pd.to_numeric(df[pop_col], errors="coerce")
print(f"\nPopulation signal: {pop.notna().sum()}/150 wards have values")

pop_z = pd.Series(np.nan, index=df.index)
mask = pop.notna()
mu, sd = pop[mask].mean(), pop[mask].std()
pop_z[mask] = (pop[mask] - mu) / sd
print(f"  pop_growth_z  mean={pop_z.mean():.3f}  std={pop_z.std():.3f}")

# ------------------------------------------------------------------
# Signal 2 — Infrastructure improvement (subset of wards)
# Convention: positive direction = improvement
# ------------------------------------------------------------------
udise_signals = {
    "udise_2021_2023_student_teacher_ratio_pct_change": -1,  # lower ratio = improvement
    "udise_2021_2023_internet_share_change":            +1,
    "udise_2021_2023_library_share_change":             +1,
    "udise_2021_2023_teacher_count_pct_change":         +1,
    "udise_2021_2023_classroom_count_pct_change":       +1,
}

missing = [c for c in udise_signals if c not in df.columns]
if missing:
    print(f"\nWARNING missing UDISE change columns: {missing}")
    for c in missing:
        udise_signals.pop(c)

if not udise_signals:
    sys.exit("No UDISE change columns available; cannot build full pressure.")

print(f"\nUsing {len(udise_signals)} UDISE signals:")
for c, d in udise_signals.items():
    print(f"  {'+' if d>0 else '-'}  {c}")

# Raw directional matrix
infra_raw = pd.DataFrame(index=df.index)
for col, direction in udise_signals.items():
    infra_raw[col] = pd.to_numeric(df[col], errors="coerce") * direction

n_signals = infra_raw.notna().sum(axis=1)
print(f"\nUDISE signal availability per ward:")
print(f"  all {len(udise_signals)} signals : {(n_signals == len(udise_signals)).sum()}")
print(f"  1 to {len(udise_signals)-1} signals   : {((n_signals > 0) & (n_signals < len(udise_signals))).sum()}")
print(f"  0 signals               : {(n_signals == 0).sum()}")

# Z-score each signal, then average what's available
z_cols = []
for col in infra_raw.columns:
    v = infra_raw[col]
    m = v.notna()
    if m.sum() < 5:
        continue
    mu_, sd_ = v[m].mean(), v[m].std()
    z = pd.Series(np.nan, index=df.index)
    if sd_ and sd_ > 0:
        z[m] = (v[m] - mu_) / sd_
    else:
        z[m] = 0.0
    z_cols.append(z)

infra_z = pd.concat(z_cols, axis=1).mean(axis=1, skipna=True)
print(f"\ninfra_improvement_z  mean={infra_z.mean():.3f}  std={infra_z.std():.3f}  "
      f"NaN={infra_z.isna().sum()}")

# ------------------------------------------------------------------
# Combine into pressure
# ------------------------------------------------------------------
full_mask    = pop_z.notna() & infra_z.notna()
partial_mask = pop_z.notna() & infra_z.isna()

raw_pressure = pd.Series(np.nan, index=df.index)
raw_pressure[full_mask]    = pop_z[full_mask] - infra_z[full_mask]
raw_pressure[partial_mask] = pop_z[partial_mask]

# Standardize the final score using ONLY the full-coverage distribution,
# then apply the same (mu, sd) to the partial wards so they share one scale.
if full_mask.sum() > 1:
    mu_p = raw_pressure[full_mask].mean()
    sd_p = raw_pressure[full_mask].std()
    pressure_z = (raw_pressure - mu_p) / sd_p
else:
    pressure_z = raw_pressure

coverage = pd.Series("no_data", index=df.index)
coverage[full_mask]    = "full"
coverage[partial_mask] = "partial_population_only"

print(f"\nPressure coverage:")
print(f"  full (pop + infra):  {full_mask.sum()}")
print(f"  partial (pop only):  {partial_mask.sum()}")
print(f"  no data:             {(~pop_z.notna()).sum()}")

# ------------------------------------------------------------------
# Output table
# ------------------------------------------------------------------
out = pd.DataFrame({
    "ward_id": df["ward_id"],
    "ward_name": df.get("ward_name_readme"),
    "pop_annualized_growth_pct": pop,
    "pop_growth_z": pop_z,
    "infra_improvement_z": infra_z,
    "pressure_zscore": pressure_z,
    "pressure_data_coverage": coverage,
    "n_udise_signals_used": n_signals,
})

out["pressure_rank"] = out["pressure_zscore"].rank(ascending=False, method="min")
out = out.sort_values("pressure_rank").reset_index(drop=True)
out.to_csv(OUT, index=False)
print(f"\nWrote {OUT}  shape={out.shape}")

# ------------------------------------------------------------------
# Reports
# ------------------------------------------------------------------
print("\nPressure distribution:")
print(out["pressure_zscore"].describe().to_string())

print("\nTop 10 HIGHEST pressure wards (population outpacing infrastructure):")
print(out.nlargest(10, "pressure_zscore")[
    ["ward_id", "ward_name", "pop_annualized_growth_pct",
     "infra_improvement_z", "pressure_zscore", "pressure_data_coverage"]
].to_string(index=False))

print("\nTop 10 LOWEST pressure wards (infrastructure keeping pace or ahead):")
print(out.nsmallest(10, "pressure_zscore")[
    ["ward_id", "ward_name", "pop_annualized_growth_pct",
     "infra_improvement_z", "pressure_zscore", "pressure_data_coverage"]
].to_string(index=False))

print("\nCoverage breakdown:")
print(out["pressure_data_coverage"].value_counts().to_string())