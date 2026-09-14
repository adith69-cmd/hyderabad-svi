# compute_attributions.py
"""
Local permutation attribution — SHAP-style but without the shap library.

For each ward and each feature:
  - Replace the feature with its population median
  - Re-predict
  - The prediction delta is the feature's local contribution

Output columns are named shap_* so downstream code (dashboard) can use them
unchanged. Method is documented in attributions_methodology.txt.
"""
import os, json, pickle
import numpy as np
import pandas as pd
from shapely.geometry import shape

BASE = os.path.dirname(os.path.abspath(__file__))
WIDE  = os.path.join(BASE, "hyderabad_150_ward_SUPER_wide_worldpop_v4.csv")
SVI   = os.path.join(BASE, "hyderabad_150_ward_SVI_2011.csv")
PROJ  = os.path.join(BASE, "hyderabad_150_ward_SVI_projection_central.csv")
AMEN  = os.path.join(BASE, "ghmc_amenities_ward_counts.csv")
PRES  = os.path.join(BASE, "hyderabad_150_ward_pressure.csv")
GEO   = os.path.join(BASE, "ghmc_wards_full_150.geojson")
MODELS = os.path.join(BASE, "surrogate_models.pkl")

# ---- Rebuild feature matrix (same as surrogate_model.py) ----
wide = pd.read_csv(WIDE, low_memory=False)
svi  = pd.read_csv(SVI)
pres = pd.read_csv(PRES)
proj = pd.read_csv(PROJ)
amen = pd.read_csv(AMEN)

with open(GEO, "r", encoding="utf-8") as f:
    gj = json.load(f)

spatial = []
for feat in gj["features"]:
    wn = feat["properties"].get("ward_number")
    if wn is None: continue
    g = shape(feat["geometry"])
    c = g.centroid
    spatial.append({
        "ward_id": int(wn),
        "ward_area_km2": g.area * 106 * 111,
        "centroid_lon": c.x, "centroid_lat": c.y,
    })
sp = pd.DataFrame(spatial)
cx, cy = 78.4738, 17.4239
sp["dist_center_km"] = np.sqrt(
    ((sp["centroid_lon"] - cx) * 106)**2 +
    ((sp["centroid_lat"] - cy) * 111)**2)

wide = wide.merge(sp, on="ward_id", how="left")
wide = wide.merge(svi[["ward_id", "svi_score"]],
                  on="ward_id", how="left", suffixes=("", "_svi"))
wide = wide.merge(pres[["ward_id", "pressure_zscore"]], on="ward_id", how="left")
wide = wide.merge(proj[["ward_id", "svi_2030_central"]], on="ward_id", how="left")

wide["amenity_count"] = wide["ward_id"].map(
    dict(zip(amen["ward_id"], amen["amenity_count"]))).fillna(0)
wide["amenity_log"] = np.log1p(wide["amenity_count"])
wide["pop_density_log"] = np.log1p(
    wide["census2011_population_verified"] / wide["ward_area_km2"])

FEATURES = [
    "census2011_literacy_rate",
    "census2011_non_worker_rate",
    "census2011_sc_st_share",
    "census2011_child_share",
    "census2011_population_verified",
    "census2011_marginal_worker_share",
    "worldpop_annualized_growth_2015_2018_pct",
    "pop_density_log",
    "ward_area_km2",
    "dist_center_km",
    "centroid_lat",
    "centroid_lon",
    "amenity_log",
    "pressure_zscore",
]

X_raw = wide[FEATURES].fillna(wide[FEATURES].median())
with open(MODELS, "rb") as f:
    bundle = pickle.load(f)

scaler = bundle["scaler"]
X_scaled = scaler.transform(X_raw)

# ---- Median baseline in scaled space ----
median_raw = X_raw.median()
median_scaled = scaler.transform(median_raw.values.reshape(1, -1))[0]

report = []
def log(m):
    print(m); report.append(m)

log("=" * 70)
log("Local permutation attribution (no SHAP library)")
log("=" * 70)

for name in ["svi_2011", "svi_2030"]:
    model = bundle["models"][name]["model"]

    base_preds = model.predict(X_scaled)  # (150,)

    attribs = np.zeros_like(X_scaled)
    for i in range(X_scaled.shape[0]):
        for j in range(X_scaled.shape[1]):
            # swap feature j to median for ward i
            modified = X_scaled[i].copy()
            modified[j] = median_scaled[j]
            pred_modified = model.predict(modified.reshape(1, -1))[0]
            # attribution = how much feature j contributes to ward i's
            # deviation from the median baseline
            # sign convention: positive = pushes score UP
            attribs[i, j] = base_preds[i] - pred_modified

    df_attr = pd.DataFrame(
        attribs,
        columns=[f"shap_{f}" for f in FEATURES],
    )
    df_attr.insert(0, "ward_id", wide["ward_id"].values)
    df_attr.insert(1, "ward_name",
                   wide.get("ward_name_readme", "").values)
    df_attr.insert(2, "predicted_svi", base_preds)
    df_attr.insert(3, "svi_actual",
                   wide["svi_score"].values if name == "svi_2011"
                   else wide["svi_2030_central"].values)

    out_csv = os.path.join(BASE, f"shap_values_{name}.csv")
    df_attr.to_csv(out_csv, index=False)

    # Global ranking by mean absolute attribution
    mean_abs = np.abs(attribs).mean(axis=0)
    ranking = pd.DataFrame({
        "feature": FEATURES,
        "mean_abs_attribution": mean_abs,
    }).sort_values("mean_abs_attribution", ascending=False).reset_index(drop=True)
    ranking.to_csv(os.path.join(BASE, f"shap_ranking_{name}.csv"), index=False)

    log(f"\n--- {name} ---")
    log(f"Base predictions: mean={base_preds.mean():.2f}  "
        f"std={base_preds.std():.2f}")
    log(f"Saved {out_csv}")
    log(f"Top 8 features by mean |attribution|:")
    for _, row in ranking.head(8).iterrows():
        log(f"  {row['feature']:<45} {row['mean_abs_attribution']:.3f}")

# ---- Show example attributions for top-3 most vulnerable wards ----
log("\n" + "=" * 70)
log("Example: attribution for 3 most vulnerable wards")
log("=" * 70)

attr_2011 = pd.read_csv(os.path.join(BASE, "shap_values_svi_2011.csv"))
top3 = wide.nlargest(3, "svi_score")[
    ["ward_id", "ward_name_readme", "svi_score"]]

for _, row in top3.iterrows():
    sub = attr_2011[attr_2011["ward_id"] == row["ward_id"]].iloc[0]
    att_cols = [c for c in attr_2011.columns if c.startswith("shap_")]
    contrib = [(c.replace("shap_", ""), sub[c]) for c in att_cols]
    contrib.sort(key=lambda x: abs(x[1]), reverse=True)
    log(f"\nWard {row['ward_id']} — {row['ward_name_readme']}  "
        f"(SVI={row['svi_score']:.1f})")
    for feat, val in contrib[:6]:
        sign = "+" if val > 0 else ""
        log(f"  {sign}{val:>6.2f}   {feat}")

with open(os.path.join(BASE, "shap_summary.txt"), "w") as f:
    f.write("\n".join(report))

# ---- Methodology note ----
with open(os.path.join(BASE, "attributions_methodology.txt"), "w") as f:
    f.write(
        "Local permutation attribution\n"
        "=============================\n\n"
        "For each ward and each feature:\n"
        "  1. Take the ward's standardized feature vector\n"
        "  2. Replace feature j with the population median (also standardized)\n"
        "  3. Re-predict with the trained surrogate GBM\n"
        "  4. Attribution = base_prediction - perturbed_prediction\n\n"
        "Sign convention: positive attribution = feature pushes ward's SVI up.\n\n"
        "This is a local attribution method functionally similar to SHAP's\n"
        "single-feature baseline approach, computed without the shap library\n"
        "because the Windows Application Control policy on the build machine\n"
        "blocks scipy.cluster's compiled DLL (a dependency of shap).\n\n"
        "The method is transparent, reproducible, and identical in output\n"
        "format to SHAP values. For TreeExplainer-equivalent exact SHAP\n"
        "values, install shap on a machine without the DLL policy.\n"
    )

log(f"\nWrote shap_summary.txt, attributions_methodology.txt")