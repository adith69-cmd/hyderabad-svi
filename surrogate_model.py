# surrogate_model.py
"""
Trains Gradient Boosting surrogates to predict SVI 2011 and SVI 2030,
validated with 5-fold CV.

Purpose:
  1. Give us a supervised model to run SHAP on
  2. Test how reconstructible SVI is from raw features
  3. Provide the model backing the dashboard's scenario simulator

Outputs:
  surrogate_svi_2011.pkl
  surrogate_svi_2030.pkl
  surrogate_report.txt
"""
import os, json, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
from shapely.geometry import shape

BASE = os.path.dirname(os.path.abspath(__file__))
WIDE = os.path.join(BASE, "hyderabad_150_ward_SUPER_wide_worldpop_v4.csv")
PROJ = os.path.join(BASE, "hyderabad_150_ward_SVI_projection_central.csv")
AMEN = os.path.join(BASE, "ghmc_amenities_ward_counts.csv")
GEO  = os.path.join(BASE, "ghmc_wards_full_150.geojson")
SVI  = os.path.join(BASE, "hyderabad_150_ward_SVI_2011.csv")
PRES = os.path.join(BASE, "hyderabad_150_ward_pressure.csv")

# ============================================================
# Load everything
# ============================================================
wide = pd.read_csv(WIDE, low_memory=False)
proj = pd.read_csv(PROJ)
amen = pd.read_csv(AMEN)
svi  = pd.read_csv(SVI)
pres = pd.read_csv(PRES)

# ============================================================
# Spatial features from polygons
# ============================================================
with open(GEO, "r", encoding="utf-8") as f:
    gj = json.load(f)

spatial = []
for feat in gj["features"]:
    wn = feat["properties"].get("ward_number")
    if wn is None:
        continue
    geom = shape(feat["geometry"])
    c = geom.centroid
    spatial.append({
        "ward_id": int(wn),
        "ward_area_km2": geom.area * 106 * 111,  # approx at Hyderabad lat
        "centroid_lon": c.x,
        "centroid_lat": c.y,
    })
spatial_df = pd.DataFrame(spatial)

cx, cy = 78.4738, 17.4239  # Hussain Sagar
spatial_df["dist_center_km"] = np.sqrt(
    ((spatial_df["centroid_lon"] - cx) * 106)**2 +
    ((spatial_df["centroid_lat"] - cy) * 111)**2
)

# ============================================================
# Merge everything into one dataframe keyed on ward_id
# ============================================================
wide = wide.merge(
    spatial_df[["ward_id", "ward_area_km2", "centroid_lon",
                "centroid_lat", "dist_center_km"]],
    on="ward_id", how="left",
)

wide = wide.merge(
    svi[["ward_id", "svi_score", "svi_band_percentile", "svi_rank"]],
    on="ward_id", how="left", suffixes=("", "_svi"),
)

wide = wide.merge(
    pres[["ward_id", "pressure_zscore", "pressure_confidence"]],
    on="ward_id", how="left",
)

wide = wide.merge(
    proj[["ward_id", "svi_2030_central", "svi_2025_central", "svi_2027_central"]],
    on="ward_id", how="left",
)

# ============================================================
# Derived features
# ============================================================
amen_map = dict(zip(amen["ward_id"], amen["amenity_count"]))
wide["amenity_count"] = wide["ward_id"].map(amen_map).fillna(0)
wide["amenity_log"] = np.log1p(wide["amenity_count"])
wide["pop_density_log"] = np.log1p(
    wide["census2011_population_verified"] / wide["ward_area_km2"]
)

# ============================================================
# Feature matrix
# ============================================================
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

existing = [f for f in FEATURES if f in wide.columns]
missing_feats = [f for f in FEATURES if f not in wide.columns]
if missing_feats:
    print(f"WARNING: dropping missing features: {missing_feats}")
FEATURES = existing
print(f"Using {len(FEATURES)} features: {FEATURES}")

X_raw = wide[FEATURES].copy()
if X_raw.isna().any().any():
    print("\nNaN in features:")
    print(X_raw.isna().sum()[X_raw.isna().sum() > 0])
    X_raw = X_raw.fillna(X_raw.median())

X = StandardScaler().fit_transform(X_raw)
# ---------- Targets ----------
targets = {
    "svi_2011": wide["svi_score"].values,
    "svi_2030": proj.set_index("ward_id").loc[
        wide["ward_id"], "svi_2030_central"].values,
}

# ---------- Train + CV ----------
report = []
def log(m):
    print(m); report.append(m)

log("=" * 70)
log("Surrogate GBM — 5-fold cross-validated")
log("=" * 70)
log(f"Features: {len(FEATURES)}")
log(f"Wards:    {len(X)}")
log("")

models = {}
for name, y in targets.items():
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=cv)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    log(f"{name:<12}  CV R2 = {r2:>6.3f}   CV MAE = {mae:>6.2f}")

    # Fit final model on all data (for SHAP + dashboard)
    model.fit(X, y)
    models[name] = {"model": model, "features": FEATURES,
                    "cv_r2": r2, "cv_mae": mae}

# ---------- Save ----------
with open(os.path.join(BASE, "surrogate_models.pkl"), "wb") as f:
    pickle.dump({"models": models,
                 "feature_names": FEATURES,
                 "scaler": StandardScaler().fit(X_raw)},
                f)

log("")
log(f"Saved surrogate_models.pkl")
log(f"CV R2 range: {min(m['cv_r2'] for m in models.values()):.3f} "
    f"to {max(m['cv_r2'] for m in models.values()):.3f}")

with open(os.path.join(BASE, "surrogate_report.txt"), "w") as f:
    f.write("\n".join(report))

# ---------- Feature importance sanity ----------
for name, m in models.items():
    imp = pd.Series(m["model"].feature_importances_, index=FEATURES)
    log(f"\nTop 6 features for {name}:")
    for feat, val in imp.nlargest(6).items():
        log(f"  {feat:<45} {val:.4f}")