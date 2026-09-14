# build_amenities_spatial.py
"""
Counts GHMC amenities per historical ward by spatial join.

Reads:
  ghmc_amenities/geolocations_amenities_ghmc.csv   (88K rows, lat/lon + category)
  ghmc_wards_full_150.geojson                       (150 ward polygons)

Outputs:
  ghmc_amenities_ward_counts.csv         -- amenity count per ward (150 rows)
  ghmc_amenities_by_category.csv         -- count per ward per category
  ghmc_infrastructure_2024.csv           -- normalized 0-1 infrastructure score
"""
import os, json
import numpy as np
import pandas as pd
from shapely.geometry import shape, Point
from shapely.strtree import STRtree

BASE = os.path.dirname(os.path.abspath(__file__))
CSV       = os.path.join(BASE, "ghmc_amenities", "geolocations_amenities_ghmc.csv")
GEOJSON   = os.path.join(BASE, "ghmc_wards_full_150.geojson")
OUT_COUNT = os.path.join(BASE, "ghmc_amenities_ward_counts.csv")
OUT_CAT   = os.path.join(BASE, "ghmc_amenities_by_category.csv")
OUT_INFRA = os.path.join(BASE, "ghmc_infrastructure_2024.csv")

# ------------------------------------------------------------------
# 1. Load amenities
# ------------------------------------------------------------------
amen = pd.read_csv(CSV, low_memory=False)
print(f"Amenities loaded: {amen.shape}")

amen = amen.dropna(subset=["Latitude", "Longitude"]).copy()
print(f"  after dropping missing coords: {len(amen)}")

# Quick sanity: bbox of all points
print(f"  lat range: {amen['Latitude'].min():.4f} .. {amen['Latitude'].max():.4f}")
print(f"  lon range: {amen['Longitude'].min():.4f} .. {amen['Longitude'].max():.4f}")

# ------------------------------------------------------------------
# 2. Load ward polygons
# ------------------------------------------------------------------
with open(GEOJSON, "r", encoding="utf-8") as f:
    gj = json.load(f)

wards = []
for feat in gj["features"]:
    wn = feat["properties"].get("ward_number")
    if wn is None:
        continue
    geom = shape(feat["geometry"])
    if not geom.is_valid:
        geom = geom.buffer(0)
    wards.append((int(wn), geom))

wards.sort(key=lambda x: x[0])
print(f"\nWards loaded: {len(wards)}")

ward_ids = [w[0] for w in wards]
ward_geoms = [w[1] for w in wards]
tree = STRtree(ward_geoms)

# ------------------------------------------------------------------
# 3. Point-in-polygon join
# ------------------------------------------------------------------
ward_counts   = np.zeros(len(wards), dtype=int)
category_map  = {}  # (ward_id, category) -> count

print("\nJoining 88K points to 150 polygons ...")
lat_arr = amen["Latitude"].values
lon_arr = amen["Longitude"].values
cat_arr = amen["Category"].fillna("UNKNOWN").values

hits = 0
for i in range(len(amen)):
    p = Point(lon_arr[i], lat_arr[i])
    candidates = tree.query(p)
    for idx in candidates:
        if ward_geoms[idx].contains(p):
            wid = ward_ids[idx]
            ward_counts[idx] += 1
            key = (wid, cat_arr[i])
            category_map[key] = category_map.get(key, 0) + 1
            hits += 1
            break

print(f"Points matched to a ward: {hits} / {len(amen)} "
      f"({hits/len(amen)*100:.1f}%)")
print(f"Points outside all wards: {len(amen) - hits}")

# ------------------------------------------------------------------
# 4. Per-ward table
# ------------------------------------------------------------------
counts_df = pd.DataFrame({
    "ward_id": ward_ids,
    "amenity_count": ward_counts,
})
counts_df = counts_df.sort_values("ward_id").reset_index(drop=True)
counts_df.to_csv(OUT_COUNT, index=False)
print(f"\nWrote {OUT_COUNT}")

# ------------------------------------------------------------------
# 5. Category breakdown
# ------------------------------------------------------------------
if category_map:
    cat_df = pd.DataFrame(
        [(w, c, n) for (w, c), n in category_map.items()],
        columns=["ward_id", "category", "count"],
    )
    cat_pivot = cat_df.pivot_table(
        index="ward_id", columns="category", values="count",
        fill_value=0, aggfunc="sum",
    ).reset_index()
    cat_pivot.columns = ["ward_id"] + [
        f"amenity_{str(c).lower().replace(' ', '_').replace('/', '_')}"
        for c in cat_pivot.columns[1:]
    ]
    # Ensure all 150 wards present
    all_w = pd.DataFrame({"ward_id": ward_ids})
    cat_pivot = all_w.merge(cat_pivot, on="ward_id", how="left").fillna(0)
    cat_pivot.to_csv(OUT_CAT, index=False)
    print(f"Wrote {OUT_CAT}  shape={cat_pivot.shape}")

# ------------------------------------------------------------------
# 6. Normalized infrastructure score (0-1, log-scaled)
# ------------------------------------------------------------------
raw = counts_df["amenity_count"].astype(float)
log_v = np.log1p(raw)
lo, hi = log_v.min(), log_v.max()
counts_df["infrastructure_score_2024"] = (
    (log_v - lo) / (hi - lo) if hi > lo else 0.0
)

counts_df[["ward_id", "amenity_count", "infrastructure_score_2024"]].to_csv(
    OUT_INFRA, index=False)
print(f"Wrote {OUT_INFRA}")

# ------------------------------------------------------------------
# 7. Report
# ------------------------------------------------------------------
print(f"\n=== Amenity distribution across wards ===")
print(f"Total amenities matched: {int(raw.sum()):,}")
print(f"Wards with 0 amenities: {(raw == 0).sum()}")
print(f"Wards with >0 amenities: {(raw > 0).sum()}")
print(f"\nAmenity count quantiles:")
print(raw.describe().to_string())

print(f"\nTop 10 wards by amenity count:")
print(counts_df.nlargest(10, "amenity_count")[
    ["ward_id", "amenity_count", "infrastructure_score_2024"]
].to_string(index=False))

zero_wards = counts_df[counts_df["amenity_count"] == 0]["ward_id"].tolist()
if zero_wards:
    print(f"\nWards with ZERO amenities: {zero_wards}")
    print("(These may fall outside GHMC boundary or have sparse coverage)")

# ------------------------------------------------------------------
# 8. Which of our 63 UDISE-missing wards now have amenity data?
# ------------------------------------------------------------------
try:
    wide = pd.read_csv(os.path.join(BASE,
        "hyderabad_150_ward_SUPER_wide_worldpop_v4.csv"), low_memory=False)
    udise_col = "udise2023_24_school_count"
    if udise_col in wide.columns:
        udise_missing = wide.loc[wide[udise_col].isna(), "ward_id"].tolist()
        amen_map = dict(zip(counts_df["ward_id"], counts_df["amenity_count"]))
        filled = [w for w in udise_missing if amen_map.get(w, 0) > 0]
        print(f"\n63-ward UDISE gap check:")
        print(f"  UDISE missing wards: {len(udise_missing)}")
        print(f"  Of those, now have amenity data: {len(filled)}")
        still_empty = [w for w in udise_missing if amen_map.get(w, 0) == 0]
        if still_empty:
            print(f"  Still no amenity data: {still_empty}")
except Exception as e:
    print(f"\n(UDISE gap check skipped: {e})")