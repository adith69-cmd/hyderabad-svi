# Hyderabad 150-Ward Socioeconomic Vulnerability & Projection System

**Live demo:** https://hyderabad-svi.streamlit.app

## What this is

A model that estimates socioeconomic vulnerability for each of the 150 historical
GHMC wards in Hyderabad, using Census 2011 as the baseline, WorldPop 2015/2018
and UDISE 2021–2024 as temporal signals, and PCA combined with a pressure-drift
model to project vulnerability forward to 2030.

**This is a cross-sectional baseline + scenario projection, not a validated
forecast.** We do not have future ground truth to fit against, and we do not
claim one.

![3D view](screenshots/01_3d_view.png)
![Interactive map](screenshots/02_interactive.png)
![Scenario simulator](screenshots/03_scenario.png)

## Deliverables

- **SVI 2011 baseline** — a 0–100 socioeconomic vulnerability score for each of
  150 wards, PCA-derived from six literature-guided variables.
- **Pressure score** — a per-ward measure of whether population is outpacing
  infrastructure.
- **Forward projection** 2025 / 2027 / 2030 — central estimate plus uncertainty
  band per ward.
- **Explainability** — per-ward feature attributions (local permutation method).
- **Scenario simulator** — live counterfactual via the trained GBM.
- **Interactive dashboard** — 3D extrusion view + clickable 2D map with
  ward-level drill-down, year slider, and confidence flags.

## Data sources

| Source | Coverage | Use |
|---|---|---|
| Census 2011 PCA (Hyderabad + Rangareddy + Medak districts) | 150/150 wards | SVI baseline |
| WorldPop Global2 R2025A v1, 100m raster | 150/150 wards (2015, 2018) | Population growth signal |
| UDISE+ via OpenCity (2021-22, 2023-24) | 87/91 wards | School infrastructure signal |
| GHMC Geolocations of Amenities (88,718 rows) | 150/150 wards via spatial join | Descriptive infrastructure context |
| OSM Overpass (ward polygons) | 150/150 features | Geometry |

Missing values are **NaN/unknown**, never silently zero. Where a ward has no
record in a source, that is flagged in a `*_data_coverage` column.

## Model pipeline

### Step A — SVI 2011 baseline (PCA)

Six literature-guided variables from Census 2011: illiteracy rate, non-worker
rate, SC/ST share, child (0–6) share, household crowding, marginal worker share.

- PC1 explains **52.5%** of variance
- Split-half stability: mean r = **0.97**
- Leave-one-out SVI deviation: mean |Δ| = **0.25 points**
- Per-ward decomposition = PCA loading × standardized value

**Documented anomaly:** SC/ST share loads negatively (−0.33) on PC1, inverted
from the standard all-India pattern. Investigation found SC/ST populations in
Hyderabad concentrate in peripheral/industrial wards that are newer and less
crowded than the central old-city belt that drives top-end vulnerability. The
variable is retained for framework completeness and the anomaly is documented,
not hidden. See `svi_methodology.md`.

### Step B — Pressure score

pressure = pop_growth_z − infrastructure_improvement_z


- For **87 wards with UDISE:** infrastructure = composite z of student/teacher
  ratio, internet share, library share, teacher count, classroom count changes
  2021→2023.
- For **63 wards without UDISE:** neutral imputation — the infrastructure term
  is set to 0 (i.e. average). We tested using amenity count as a substitute and
  rejected it: correlation with UDISE improvement is only **r = 0.21**, too weak
  to be a real proxy. Amenity count is retained as a descriptive column only.
- Score re-standardized to mean 0, std 1 over all 150 wards.
- Every ward carries `pressure_confidence` = `high` (87, full UDISE) or
  `medium` (63, population-only).

### Step C — Forward projection

SVI(year) = clip( SVI_2011 × (1 + β · pressure_z)^(year−2011), 0, 100 )


- **β sweep:** 0.001 to 0.010. Central case β = 0.005.
- **Uncertainty band:** ±0.5 z-units (high confidence), ±0.8 (medium), applied
  to the pressure term.
- **Result (β=0.005, 2011 → 2030):** 7 of 150 wards change band.
  - 4 yellow → red
  - 2 yellow → green
  - 1 green → yellow
  - All 21 red wards remain red.
- Sensitivity: mean projected SVI 2030 moves only **1.4 points** across the
  full β sweep. The aggregate story is robust.

## ML model layer

### Surrogate GBM (Gradient Boosting)

Trained on a 14-feature matrix (6 Census variables, WorldPop growth, amenity
count, pressure score, spatial features).

- SVI 2011: **CV R² = 0.930** (5-fold)
- SVI 2030: **CV R² = 0.927** (5-fold)

Purpose: give the dashboard a live model for the scenario simulator and provide
a base model for local permutation attribution.

### MLP comparison

Same 14-feature matrix, small neural network (2 hidden layers, 32 and 16 units,
ReLU, early stopping).

| Target   | GBM mean R² | MLP mean R² | Δ      | Paired t-test     |
|----------|-------------|-------------|--------|-------------------|
| SVI 2011 | 0.9265      | 0.9391      | +0.0126 | t=1.45, p=0.22 (ns) |
| SVI 2030 | 0.9255      | 0.9507      | +0.0253 | t=1.39, p=0.24 (ns) |

Neither difference is statistically significant at α = 0.05. We retain GBM as
the production model for interpretability and fast inference; the MLP serves as
a robustness check. This matches the known pattern that tree ensembles and
neural networks often tie on small tabular datasets (Grinsztajn et al., 2022).

### Explainability — local permutation attribution

Per-ward feature attributions computed by replacing each feature with its
population median and re-predicting. The prediction delta is that feature's
local contribution to the ward's score. This is functionally equivalent to a
single-feature-baseline SHAP method, implemented without the `shap` library
because the build machine's Windows Application Control policy blocks
`scipy.cluster`'s compiled DLL (a hard dependency of `shap`).

Attribution output format matches SHAP values exactly, so swapping in true
TreeExplainer SHAP on a different machine requires no downstream code changes.
See `attributions_methodology.txt`.

### Scenario simulator

Live counterfactual: user moves sliders on literacy, non-worker rate, and child
share. The dashboard re-runs the trained GBM on the perturbed feature matrix
and the map recolors. This is a **model-based counterfactual, not a causal
claim**. The interface labels it as such.

## What this is NOT

- **Not a poverty measure.** No income or consumption data is used.
- **Not a validated forecast.** The 2025–2030 values are scenario-conditional
  projections, not predictions. β is swept, not fitted, because there is no
  future ground truth to fit against.
- **Not causal.** The model measures co-occurrence of known deprivation
  indicators. It does not claim that literacy *causes* lower vulnerability or
  vice versa.
- **Not a spatial-dependence model.** Wards are modeled independently. Adding
  spatial-lag features or a graph neural network is a natural next iteration.

## Reproducing the pipeline

Files in this folder, in run order:

Data preparation
fix_master.py # repair wide + panel structure
fetch_missing_wards.py # fetch 5 missing polygons
reclip_missing5.py # WorldPop for the 5 new wards
fix_growth_and_sync_panel.py # propagate to wide + panel
integrate_census_correct.py # replace broken Census with correct 150-ward master
build_amenities_spatial.py # amenity dataset → ward counts
Model
build_svi.py # SVI 2011 via PCA
patch_svi.py # add percentile bands + SC/ST doc
build_pressure.py # pressure score
patch_pressure_v4.py # neutral imputation, final pressure
project_svi.py # forward projection + β sweep
patch_projection.py # absolute-vs-percentile band comparison
surrogate_model.py # Gradient Boosting surrogate
compute_attributions.py # local permutation attribution
train_mlp.py # MLP robustness check
compare_models_statistical.py # paired t-test GBM vs MLP

Deploy
dashboard.py # Streamlit app

Run the dashboard locally:
pip install -r requirements.txt
streamlit run dashboard.py


## Key output files

- `hyderabad_150_ward_SUPER_wide_worldpop_v4.csv` — master, one row per ward
- `hyderabad_150_ward_SUPER_panel_worldpop_v4.csv` — long panel, no leakage
- `hyderabad_150_ward_SVI_2011.csv` — SVI baseline, contributions, bands
- `hyderabad_150_ward_pressure.csv` — pressure + confidence flags
- `hyderabad_150_ward_SVI_projection_central.csv` — 2025/2027/2030 per ward
- `hyderabad_150_ward_SVI_projection.csv` — full β sweep (long format)
- `ghmc_wards_full_150.geojson` — ward polygons
- `shap_values_svi_2011.csv` / `shap_values_svi_2030.csv` — per-ward attributions
- `surrogate_models.pkl` / `mlp_models.pkl` — trained model artifacts

## Known limitations

1. **Only one real demographic time point.** Census 2011 is the only year with
   full socioeconomic breakdown. The projection therefore extrapolates a 2011
   baseline using a 2015–2023 pressure signal.
2. **42% of wards have population-only pressure.** 63 of 150 wards lack UDISE
   data; their pressure score is driven by population growth alone and carries a
   wider uncertainty band.
3. **Amenity count is not a valid UDISE substitute.** Empirically tested
   (r = 0.21) and rejected. Retained as descriptive only.
4. **Two wards saturate the SVI ceiling at 100 by 2030** (Shalibanda,
   Chandrayangutta). They were already at the top of the 2011 distribution; the
   model cannot resolve differences among wards at that extreme.
5. **The 300-ward GHMC structure is not used.** It was gazetted in December
   2025, is still under litigation, and has no historical socioeconomic data at
   that resolution.
6. **No spatial dependence modeled.** Each ward is modeled independently. A
   spatial autoregressive model or graph neural network on the ward adjacency
   graph is a natural next step.

## Provenance

- **Census 2011:** official Primary Census Abstract, PCA ward rows aggregated
  across Hyderabad (535), Rangareddy (536), Medak (537) districts.
  Multi-district wards summed, rates computed post-aggregation.
- **WorldPop:** Global2 R2025A v1 100m population count. Zonal sum, not mean.
  NoData preserved as NaN, never zero.
- **UDISE+:** OpenCity published facility-level data, aggregated to ward by
  school location.
- **Amenities:** GHMC geolocated amenity points, spatially joined to ward
  polygons (90.5% of 88,717 points fell inside a ward).
- **Geometry:** 145 polygons from uploaded OSM GeoJSON + 5 fetched from
  Overpass by relation ID.

## License

Data sourced from OpenCity (Public Domain), Census of India, and WorldPop.
Code released for academic and portfolio use.

Update and push
Save as README.md in Desktop\WorldPop\ (overwrite the existing one).

Then:
cd /d "%USERPROFILE%\Desktop\WorldPop"
git add README.md
git commit -m "Full README with live demo, model comparison, and attribution documentation"
git push

