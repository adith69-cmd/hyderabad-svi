# project_svi.py
"""
Projects SVI 2011 forward using the pressure score from Step B.

Model:
    SVI(year) = clamp( SVI_2011 * (1 + beta * pressure_z)^(year - 2011), 0, 100 )

Uncertainty per ward:
    pressure_uncertainty is 0.5 for high-confidence (full UDISE coverage)
    wards and 1.0 for low-confidence (population-only) wards. Upper and
    lower projections use (pressure + uncertainty) and (pressure - uncertainty).

beta sweep: [0.001, 0.003, 0.005, 0.007, 0.010]
    beta=0.005 (central case) means a ward with +1 std pressure drifts
    +0.5% per year relative to its 2011 baseline. Over 19 years that
    compounds to about +10%. The sweep spans 0.1% to 1.0% per year per
    unit of pressure, which covers the plausible range.

Outputs:
    hyderabad_150_ward_SVI_projection.csv          -- long, full sweep
    hyderabad_150_ward_SVI_projection_central.csv  -- wide, beta=0.005
"""
import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SVI_IN  = os.path.join(BASE, "hyderabad_150_ward_SVI_2011.csv")
PRES_IN = os.path.join(BASE, "hyderabad_150_ward_pressure.csv")
LONG_OUT    = os.path.join(BASE, "hyderabad_150_ward_SVI_projection.csv")
CENTRAL_OUT = os.path.join(BASE, "hyderabad_150_ward_SVI_projection_central.csv")

BASELINE_YEAR = 2011
TARGET_YEARS  = [2025, 2027, 2030]
BETA_SWEEP    = [0.001, 0.003, 0.005, 0.007, 0.010]
BETA_CENTRAL  = 0.005

# ---- Load ----
svi  = pd.read_csv(SVI_IN)
pres = pd.read_csv(PRES_IN)
df = svi.merge(
    pres[["ward_id", "pressure_zscore", "pressure_uncertainty",
          "pressure_confidence", "pressure_data_coverage"]],
    on="ward_id", how="left",
)
print(f"Merged: {df.shape}")
assert df["pressure_zscore"].notna().all(), "Some wards missing pressure"

# ---- Formula ----
def project(base, pressure, beta, year):
    yrs = year - BASELINE_YEAR
    factor = np.power(1.0 + beta * pressure, yrs)
    return np.clip(base * factor, 0.0, 100.0)

def band(s):
    if s <= 30:  return "green"
    if s <= 60:  return "yellow"
    return "red"

# ---- Build long-format output ----
rows = []
for beta in BETA_SWEEP:
    for year in TARGET_YEARS:
        # central projection
        p = project(df["svi_score"].values,
                    df["pressure_zscore"].values,
                    beta, year)
        # upper/lower using pressure +- uncertainty
        p_hi = project(df["svi_score"].values,
                       df["pressure_zscore"].values + df["pressure_uncertainty"].values,
                       beta, year)
        p_lo = project(df["svi_score"].values,
                       df["pressure_zscore"].values - df["pressure_uncertainty"].values,
                       beta, year)
        for i, row in df.iterrows():
            rows.append({
                "ward_id": row["ward_id"],
                "ward_name": row.get("ward_name"),
                "beta": beta,
                "year": year,
                "svi_2011": row["svi_score"],
                "pressure_zscore": row["pressure_zscore"],
                "pressure_confidence": row["pressure_confidence"],
                "projected_svi_central": round(p[i], 3),
                "projected_svi_lower":  round(p_lo[i], 3),
                "projected_svi_upper":  round(p_hi[i], 3),
                "band_central": band(p[i]),
                "band_2011":    row.get("svi_band_percentile"),
            })

long_df = pd.DataFrame(rows)
long_df.to_csv(LONG_OUT, index=False)
print(f"\nWrote {LONG_OUT}  shape={long_df.shape}")

# ---- Wide format for the central case ----
central = df[["ward_id", "ward_name", "svi_score",
              "svi_band_percentile", "pressure_zscore",
              "pressure_confidence", "pressure_data_coverage"]].copy()
central = central.rename(columns={
    "svi_score": "svi_2011",
    "svi_band_percentile": "band_2011",
})

for year in TARGET_YEARS:
    p    = project(df["svi_score"].values, df["pressure_zscore"].values,
                   BETA_CENTRAL, year)
    p_hi = project(df["svi_score"].values,
                   df["pressure_zscore"].values + df["pressure_uncertainty"].values,
                   BETA_CENTRAL, year)
    p_lo = project(df["svi_score"].values,
                   df["pressure_zscore"].values - df["pressure_uncertainty"].values,
                   BETA_CENTRAL, year)
    central[f"svi_{year}_central"] = np.round(p,    2)
    central[f"svi_{year}_lower"]   = np.round(p_lo, 2)
    central[f"svi_{year}_upper"]   = np.round(p_hi, 2)
    central[f"band_{year}"]        = [band(x) for x in p]

# Drift magnitude
central["drift_2011_to_2030"] = central["svi_2030_central"] - central["svi_2011"]
central["drift_band_change"]  = (
    central["band_2011"] != central["band_2030"]
)

central = central.sort_values("drift_2011_to_2030", ascending=False).reset_index(drop=True)
central.to_csv(CENTRAL_OUT, index=False)
print(f"Wrote {CENTRAL_OUT}  shape={central.shape}")

# ---- Reports ----
print("\n=== Central case (beta=0.005) ===")
print("\nProjected SVI distribution by year:")
for year in TARGET_YEARS:
    col = f"svi_{year}_central"
    print(f"  {year}:  mean={central[col].mean():.1f}  "
          f"median={central[col].median():.1f}  "
          f"min={central[col].min():.1f}  max={central[col].max():.1f}")

print("\nBand counts (2011 vs projected):")
print(f"  2011 : {central['band_2011'].value_counts().to_dict()}")
for year in TARGET_YEARS:
    print(f"  {year}: {central[f'band_{year}'].value_counts().to_dict()}")

print(f"\nWards whose band changes between 2011 and 2030: "
      f"{central['drift_band_change'].sum()} / 150")

print("\nTop 10 wards with LARGEST upward drift (SVI increase 2011->2030):")
top = central.nlargest(10, "drift_2011_to_2030")[
    ["ward_id", "ward_name", "svi_2011", "svi_2030_central",
     "svi_2030_lower", "svi_2030_upper", "band_2011", "band_2030",
     "pressure_confidence"]
]
print(top.to_string(index=False))

print("\nTop 10 wards with LARGEST downward drift:")
bot = central.nsmallest(10, "drift_2011_to_2030")[
    ["ward_id", "ward_name", "svi_2011", "svi_2030_central",
     "svi_2030_lower", "svi_2030_upper", "band_2011", "band_2030",
     "pressure_confidence"]
]
print(bot.to_string(index=False))

print("\nSensitivity sweep — mean projected SVI 2030 across beta values:")
for beta in BETA_SWEEP:
    sub = long_df[(long_df["beta"] == beta) & (long_df["year"] == 2030)]
    print(f"  beta={beta:.3f}:  mean={sub['projected_svi_central'].mean():.1f}  "
          f"max={sub['projected_svi_central'].max():.1f}  "
          f"min={sub['projected_svi_central'].min():.1f}")