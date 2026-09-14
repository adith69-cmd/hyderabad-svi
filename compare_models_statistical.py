# compare_models_statistical.py — complete, runnable
import os, json, pickle
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy import stats
from shapely.geometry import shape

BASE = os.path.dirname(os.path.abspath(__file__))
WIDE = os.path.join(BASE, "hyderabad_150_ward_SUPER_wide_worldpop_v4.csv")
SVI  = os.path.join(BASE, "hyderabad_150_ward_SVI_2011.csv")
PROJ = os.path.join(BASE, "hyderabad_150_ward_SVI_projection_central.csv")
AMEN = os.path.join(BASE, "ghmc_amenities_ward_counts.csv")
PRES = os.path.join(BASE, "hyderabad_150_ward_pressure.csv")
GEO  = os.path.join(BASE, "ghmc_wards_full_150.geojson")

# ============================================================
# Build the exact feature matrix used elsewhere
# ============================================================
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

# ============================================================
# Fold-by-fold R² for both models, then paired t-test
# ============================================================
def fold_scores(model_ctor, X, y):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr, te in cv.split(X):
        m = model_ctor()
        m.fit(X[tr], y[tr])
        scores.append(r2_score(y[te], m.predict(X[te])))
    return np.array(scores)

print("=" * 70)
print("Paired t-test — GBM vs MLP, fold-by-fold")
print("=" * 70)

summary_rows = []

for name, y in targets.items():
    gbm_folds = fold_scores(
        lambda: GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42), X, y)
    mlp_folds = fold_scores(
        lambda: MLPRegressor(
            hidden_layer_sizes=(32, 16), activation="relu",
            solver="adam", alpha=0.001, learning_rate_init=0.005,
            max_iter=2000, early_stopping=True,
            n_iter_no_change=30, random_state=42), X, y)

    t, p = stats.ttest_rel(mlp_folds, gbm_folds)
    mean_gbm = gbm_folds.mean()
    mean_mlp = mlp_folds.mean()

    print(f"\n--- {name} ---")
    print(f"GBM folds: {np.round(gbm_folds, 3)}  mean={mean_gbm:.3f}")
    print(f"MLP folds: {np.round(mlp_folds, 3)}  mean={mean_mlp:.3f}")
    print(f"Mean delta (MLP − GBM): {mean_mlp - mean_gbm:+.4f}")
    print(f"Paired t-test: t = {t:.3f},  p = {p:.4f}")
    verdict = "SIGNIFICANT" if p < 0.05 else "not significant"
    print(f"Verdict: {verdict}")

    summary_rows.append({
        "target": name,
        "gbm_mean_r2": mean_gbm,
        "mlp_mean_r2": mean_mlp,
        "delta": mean_mlp - mean_gbm,
        "t": t,
        "p": p,
        "significant_at_0.05": p < 0.05,
    })

# ============================================================
# Save report
# ============================================================
out = pd.DataFrame(summary_rows)
out.to_csv(os.path.join(BASE, "model_comparison_stats.csv"), index=False)

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(out.to_string(index=False))

print("\nInterpretation:")
for _, row in out.iterrows():
    if row["significant_at_0.05"]:
        print(f"  {row['target']}: MLP significantly better (p={row['p']:.4f})")
    elif row["delta"] > 0:
        print(f"  {row['target']}: MLP nominally better but "
              f"NOT significant (p={row['p']:.4f}) — treat as tied")
    else:
        print(f"  {row['target']}: GBM nominally better but "
              f"NOT significant (p={row['p']:.4f}) — treat as tied")

print(f"\nWrote model_comparison_stats.csv")