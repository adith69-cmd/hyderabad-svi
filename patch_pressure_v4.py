# patch_pressure_v4.py — self-contained, replaces the whole patch chain
"""
Produces the final pressure CSV from build_pressure.py's output.

Adds: udise_imputed flag, neutral imputation for missing UDISE,
      pressure_confidence, pressure_uncertainty.

For 87 wards with UDISE: udise_z is used directly.
For 63 wards without: udise_z_used = 0 (neutral).
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PRES = os.path.join(BASE, "hyderabad_150_ward_pressure.csv")
AMEN = os.path.join(BASE, "ghmc_amenities_ward_counts.csv")

df = pd.read_csv(PRES)
amen = pd.read_csv(AMEN)

# Confirm build_pressure.py's output columns exist
required = ["ward_id", "ward_name", "pop_growth_z", "infra_improvement_z"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns from build_pressure.py: {missing}. "
                   f"Run build_pressure.py first.")

# Rename infra_improvement_z to udise_improvement_z for clarity
if "udise_improvement_z" not in df.columns:
    df["udise_improvement_z"] = df["infra_improvement_z"]

# Flag imputed rows
df["udise_imputed"] = df["udise_improvement_z"].isna()

# Neutral imputation: set udise_used = 0 where missing
df["udise_improvement_z_used"] = df["udise_improvement_z"].fillna(0.0)

# Add amenity count (descriptive only — not used in pressure)
df["amenity_count"] = df["ward_id"].map(
    dict(zip(amen["ward_id"], amen["amenity_count"]))
).fillna(0).astype(int)

# Recompute pressure with neutral imputation
raw = df["pop_growth_z"] - df["udise_improvement_z_used"]
mu, sd = raw.mean(), raw.std()
df["pressure_zscore"] = (raw - mu) / sd

# Confidence and coverage
df["pressure_data_coverage"] = np.where(df["udise_imputed"],
                                        "population_only", "full")
df["pressure_confidence"] = np.where(df["udise_imputed"], "medium", "high")
df["pressure_uncertainty"] = np.where(df["udise_imputed"], 0.8, 0.5)

# Rank
df["pressure_rank"] = df["pressure_zscore"].rank(ascending=False, method="min")
df = df.sort_values("pressure_rank").reset_index(drop=True)

df.to_csv(PRES, index=False)

print("Wrote final pressure CSV")
print(f"\nCoverage:   {df['pressure_data_coverage'].value_counts().to_dict()}")
print(f"Confidence: {df['pressure_confidence'].value_counts().to_dict()}")
print(f"\nPressure distribution:")
print(df["pressure_zscore"].describe().to_string())

print("\nTop 5 highest pressure:")
print(df.nlargest(5, "pressure_zscore")[
    ["ward_id", "ward_name", "pop_growth_z",
     "udise_improvement_z_used", "pressure_zscore",
     "pressure_confidence"]
].to_string(index=False))

print("\nTop 5 lowest pressure:")
print(df.nsmallest(5, "pressure_zscore")[
    ["ward_id", "ward_name", "pop_growth_z",
     "udise_improvement_z_used", "pressure_zscore",
     "pressure_confidence"]
].to_string(index=False))