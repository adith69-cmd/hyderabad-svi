# patch_projection.py
import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
CENT = os.path.join(BASE, "hyderabad_150_ward_SVI_projection_central.csv")
LONG = os.path.join(BASE, "hyderabad_150_ward_SVI_projection.csv")

def band_absolute(s):
    if s <= 30:  return "green"
    if s <= 60:  return "yellow"
    return "red"

# ---- Central case ----
c = pd.read_csv(CENT)

# Keep percentile band for reference, add absolute 2011 band
c["band_2011_percentile"] = c["band_2011"]
c["band_2011_absolute"]   = c["svi_2011"].apply(band_absolute)

# Recompute "did band change" using ABSOLUTE bands
c["drift_band_change_absolute"] = (
    c["band_2011_absolute"] != c["band_2030"]
)

# Flag saturation
c["svi_2030_saturated"] = c["svi_2030_central"] >= 99.99

# Flag low-confidence
c["low_confidence_flag"] = c["pressure_confidence"] == "low"

c.to_csv(CENT, index=False)

# ---- Reports ----
print("=== Absolute band counts ===")
print(f"2011: {c['band_2011_absolute'].value_counts().to_dict()}")
for yr in [2025, 2027, 2030]:
    print(f"{yr}: {c[f'band_{yr}'].value_counts().to_dict()}")

print(f"\nWards changing absolute band 2011 -> 2030: "
      f"{c['drift_band_change_absolute'].sum()} / 150")

print("\nTransition matrix (2011 absolute -> 2030):")
print(pd.crosstab(c["band_2011_absolute"], c["band_2030"]))

print(f"\nWards clamped at 100 in 2030: {c['svi_2030_saturated'].sum()}")
print(c[c["svi_2030_saturated"]][["ward_id", "ward_name",
                                  "svi_2011", "svi_2030_central"]].to_string(index=False))

print(f"\nLow-confidence wards: {c['low_confidence_flag'].sum()}")

# ---- Long-format file: add absolute 2011 band too ----
ldf = pd.read_csv(LONG)
ldf["band_2011_absolute"] = ldf["svi_2011"].apply(band_absolute)
ldf["low_confidence_flag"] = ldf["pressure_confidence"] == "low"
ldf.to_csv(LONG, index=False)
print(f"\nUpdated {LONG}")

print(f"\nUpdated {CENT}")