# train_mlp.py
"""
Trains a small MLP to predict SVI 2011 and SVI 2030 from raw features.
Compares against the Gradient Boosting surrogate.

Purpose: demonstrate whether a neural network can match tree-based
performance on this tabular dataset, and produce a deep learning
artefact for the portfolio.

Outputs:
  mlp_models.pkl
  mlp_vs_gbm_report.txt
"""
import os, json, pickle
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
from shapely.geometry import shape

BASE = os.path.dirname(os.path.abspath(__file__))
WIDE = os.path.join(BASE, "hyderabad_150_ward_SUPER_wide_worldpop_v4.csv")
SVI  = os.path.join(BASE, "hyderabad_150_ward_SVI_2011.csv")
PROJ = os.path.join(BASE, "hyderabad_150_ward_SVI_projection_central.csv")
AMEN = os.path.join(BASE, "ghmc_amenities_ward_counts.csv")
PRES = os.path.join(BASE, "hyderabad_150_ward_pressure.csv")
GEO  = os.path.join(BASE, "ghmc_wards_full_150.geojson")

# ---- Rebuild the exact feature matrix ----
wide = pd.read_csv(WIDE, low_memory=False)
svi  = pd.read_csv(SVI)
pres = pd.read_csv(PRES)
proj = pd.read_csv(PROJ)
amen = pd.read_csv(AMEN)

with open(GEO, "r", encoding="utf-8") as f:
    gj = json.load(f)

sp = []
for feat in gj["features"]:
    wn = feat["properties"].get("ward_number")
    if wn is None: continue
    g = shape(feat["geometry"])
    c = g.centroid
    sp.append({"ward_id": int(wn),
               "ward_area_km2": g.area * 106 * 111,
               "centroid_lon": c.x, "centroid_lat": c.y})
sp = pd.DataFrame(sp)
cx, cy = 78.4738, 17.4239
sp["dist_center_km"] = np.sqrt(
    ((sp["centroid_lon"] - cx) * 106)**2 +
    ((sp["centroid_lat"] - cy) * 111)**2)

wide = wide.merge(sp, on="ward_id", how="left")
wide = wide.merge(svi[["ward_id", "svi_score"]], on="ward_id",
                  how="left", suffixes=("", "_svi"))
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
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

targets = {
    "svi_2011": wide["svi_score"].values,
    "svi_2030": wide["svi_2030_central"].values,
}

# ---- Train MLP with 5-fold CV ----
report = []
def log(m):
    print(m); report.append(m)

log("=" * 70)
log("MLP vs Gradient Boosting — SVI prediction")
log("=" * 70)

log("\nMLP architecture: 2 hidden layers (32, 16), ReLU, adam, 500 max iter")
log("")

mlp_results = {}
for name, y in targets.items():
    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=0.001,                 # L2 regularisation
        learning_rate_init=0.005,
        max_iter=2000,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=42,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(mlp, X, y, cv=cv)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mlp_results[name] = {"r2": r2, "mae": mae}

    log(f"{name:<12}  MLP CV R2 = {r2:>6.3f}   CV MAE = {mae:>6.2f}")

    # Refit on all data for saving
    mlp.fit(X, y)

# ---- Load GBM results for comparison ----
gbm_models = {}
with open(os.path.join(BASE, "surrogate_models.pkl"), "rb") as f:
    bundle = pickle.load(f)
for name in ["svi_2011", "svi_2030"]:
    gbm_models[name] = bundle["models"][name]

log("\n" + "-" * 70)
log("Comparison:")
log(f"{'target':<12} {'GBM R2':>10} {'MLP R2':>10} {'GBM MAE':>10} {'MLP MAE':>10}")
log("-" * 70)

for name in ["svi_2011", "svi_2030"]:
    g = gbm_models[name]
    m = mlp_results[name]
    log(f"{name:<12} {g['cv_r2']:>10.3f} {m['r2']:>10.3f} "
        f"{g['cv_mae']:>10.2f} {m['mae']:>10.2f}")

# ---- Save MLP ----
with open(os.path.join(BASE, "mlp_models.pkl"), "wb") as f:
    pickle.dump({"models": {k: MLPRegressor(
                    hidden_layer_sizes=(32, 16), activation="relu",
                    solver="adam", alpha=0.001, max_iter=2000,
                    early_stopping=True, random_state=42,
                ).fit(X, targets[k]) for k in targets},
                 "scaler": scaler,
                 "feature_names": FEATURES}, f)

log(f"\nSaved mlp_models.pkl")

# ---- Conclusion ----
log("\n" + "=" * 70)
log("Conclusion:")
if all(mlp_results[t]["r2"] < gbm_models[t]["cv_r2"] for t in targets):
    log("GBM outperforms MLP on both targets. This is consistent with the")
    log("known pattern that tree ensembles dominate neural networks on")
    log("small tabular datasets (Grinsztajn et al. 2022). Documented as")
    log("a deliberate comparison, not a failure.")
elif all(mlp_results[t]["r2"] > gbm_models[t]["cv_r2"] for t in targets):
    log("MLP outperforms GBM on both targets. Surprising for n=150 tabular")
    log("data but worth reporting.")
else:
    log("Mixed result — one target favours GBM, the other MLP.")

with open(os.path.join(BASE, "mlp_vs_gbm_report.txt"), "w") as f:
    f.write("\n".join(report))
log(f"Wrote mlp_vs_gbm_report.txt")