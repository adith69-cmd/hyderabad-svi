# reclip_missing5.py  — aggregate WorldPop only for the 5 newly-added wards
import os, json, re, sys
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape

BASE = os.path.dirname(os.path.abspath(__file__))
WARDS_GEOJSON = os.path.join(BASE, "ghmc_wards_full_150.geojson")
WIDE = os.path.join(BASE, "hyderabad_150_ward_SUPER_wide_worldpop_v3.csv")

RASTERS = {
    2015: os.path.join(BASE, "hyderabad_clipped_corrected",
                       "ind_pop_2015_CN_100m_R2025A_v1_HYDERABAD_CORRECTED.tif"),
    2018: os.path.join(BASE, "hyderabad_clipped_corrected",
                       "ind_pop_2018_CN_100m_R2025A_v1_HYDERABAD_CORRECTED.tif"),
}

MISSING = [3, 4, 11, 13, 113]

for y, p in RASTERS.items():
    if not os.path.exists(p):
        sys.exit(f"Missing raster: {p}")

# Load ward geometries
with open(WARDS_GEOJSON, "r", encoding="utf-8") as f:
    gj = json.load(f)

def ward_poly(feat):
    """Return shapely geometry, or None if ward number unknown."""
    wn = feat["properties"].get("ward_number")
    if wn is None:
        return None, None
    geom = shape(feat["geometry"])
    if not geom.is_valid:
        geom = geom.buffer(0)
    return int(wn), geom

missing_geoms = []
missing_nums = []
for f in gj["features"]:
    wn, g = ward_poly(f)
    if wn in MISSING and g is not None:
        missing_geoms.append(g)
        missing_nums.append(wn)

print(f"Loaded {len(missing_geoms)} polygons for wards {missing_nums}")
if len(missing_geoms) != 5:
    sys.exit(f"Expected 5 geometries, got {len(missing_geoms)}")

# Load wide master
wide = pd.read_csv(WIDE, low_memory=False)

for year, path in RASTERS.items():
    print(f"\nProcessing year {year} …")
    with rasterio.open(path) as src:
        print(f"  raster: {src.width}x{src.height}  CRS={src.crs}  nodata={src.nodata}")
        pop = src.read(1).astype("float64")
        if src.nodata is not None:
            pop = np.where(np.isclose(pop, src.nodata), 0.0, pop)
        pop = np.where(np.isfinite(pop) & (pop > 0), pop, 0.0)

        labels = rasterize(
            ((g, i + 1) for i, g in enumerate(missing_geoms)),
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0, dtype="int32", all_touched=False,
        )

    n = len(missing_geoms)
    sums = np.bincount(labels.ravel(), weights=pop.ravel(), minlength=n + 1)[1:]
    counts = np.bincount(labels.ravel(), minlength=n + 1)[1:]

    for i, wn in enumerate(missing_nums):
        print(f"  Ward {wn:>3}: pop={sums[i]:,.0f}   pixels={counts[i]}")
        wide.loc[wide["ward_id"] == wn, f"worldpop_population_{year}"] = sums[i]

# Recompute derived growth for the touched wards
mask = wide["ward_id"].isin(MISSING)

def _g(row):
    a, b = row["worldpop_population_2015"], row["worldpop_population_2018"]
    if pd.isna(a) or pd.isna(b) or a == 0:
        return pd.Series([np.nan, np.nan, np.nan])
    abs_g = b - a
    pct_g = abs_g / a * 100
    ann_g = ((b / a) ** (1 / 3) - 1) * 100
    return pd.Series([abs_g, pct_g, ann_g])

wide.loc[mask, ["worldpop_abs_growth_2015_2018",
                "worldpop_pct_growth_2015_2018",
                "worldpop_annualized_growth_2015_2018_pct"]] = \
    wide.loc[mask].apply(_g, axis=1)

# Update status fields
wide.loc[mask, "worldpop_status"] = "observed_raster_aggregation"
wide.loc[mask, "geometry_status"] = "150_WARD_ID_FRAMEWORK_LOCKED; POLYGON_FILE_150_FEATURES_VERIFIED"

# Report
print("\nFinal check on the 5 touched wards:")
print(wide.loc[mask, ["ward_id", "worldpop_population_2015", "worldpop_population_2018",
                      "worldpop_pct_growth_2015_2018"]].to_string(index=False))

still_nan = wide["worldpop_population_2015"].isna().sum()
print(f"\nTotal wards with WorldPop 2015 = NaN: {still_nan}")

wide.to_csv(WIDE, index=False)
print(f"\nUpdated {WIDE}")