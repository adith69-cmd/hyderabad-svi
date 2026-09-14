# dashboard.py — tabbed: 3D view + clickable 2D map
import os, json, pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import folium
import pydeck as pdk
from streamlit_folium import st_folium
from shapely.geometry import shape

BASE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Hyderabad SVI",
                   page_icon="🗺️", layout="wide",
                   initial_sidebar_state="expanded")

FEATURE_LABELS = {
    "census2011_non_worker_rate":               "High non-worker rate",
    "census2011_child_share":                   "High child (0-6) share",
    "census2011_marginal_worker_share":         "High marginal employment",
    "census2011_sc_st_share":                   "High SC/ST share",
    "census2011_literacy_rate":                 "Low literacy",
    "census2011_population_verified":           "Population size",
    "worldpop_annualized_growth_2015_2018_pct": "Population growth",
    "pop_density_log":                          "Population density",
    "ward_area_km2":                            "Ward area",
    "dist_center_km":                           "Distance from centre",
    "centroid_lat":                             "Location (south-central belt)",
    "centroid_lon":                             "Location (east-west)",
    "amenity_log":                              "Amenity availability",
    "pressure_zscore":                          "Service pressure",
}

# =========================================================
# Load
# =========================================================
@st.cache_data
def load_data():
    with open(os.path.join(BASE, "ghmc_wards_full_150.geojson"),
              "r", encoding="utf-8") as f:
        geo = json.load(f)

    svi   = pd.read_csv(os.path.join(BASE, "hyderabad_150_ward_SVI_2011.csv"))
    pres  = pd.read_csv(os.path.join(BASE, "hyderabad_150_ward_pressure.csv"))
    proj  = pd.read_csv(os.path.join(BASE,
        "hyderabad_150_ward_SVI_projection_central.csv"))
    attr11 = pd.read_csv(os.path.join(BASE, "shap_values_svi_2011.csv"))
    attr30 = pd.read_csv(os.path.join(BASE, "shap_values_svi_2030.csv"))

    with open(os.path.join(BASE, "surrogate_models.pkl"), "rb") as f:
        bundle = pickle.load(f)

    proj_cols = ["ward_id"]
    for y in (2025, 2027, 2030):
        proj_cols += [f"svi_{y}_central", f"svi_{y}_lower", f"svi_{y}_upper"]
    proj_cols = [c for c in proj_cols if c in proj.columns]

    df = svi.merge(pres[["ward_id", "pressure_zscore",
                         "pressure_confidence"]],
                   on="ward_id", how="left")
    df = df.merge(proj[proj_cols], on="ward_id", how="left")

    attr11 = attr11.rename(columns={
        c: c.replace("shap_", "shap11_") for c in attr11.columns
        if c.startswith("shap_")})
    attr30 = attr30.rename(columns={
        c: c.replace("shap_", "shap30_") for c in attr30.columns
        if c.startswith("shap_")})
    df = df.merge(attr11[["ward_id"] + [c for c in attr11.columns
                                         if c.startswith("shap11_")]],
                  on="ward_id", how="left")
    df = df.merge(attr30[["ward_id"] + [c for c in attr30.columns
                                         if c.startswith("shap30_")]],
                  on="ward_id", how="left")
    return df, geo, bundle

df, GEO, BUNDLE = load_data()
NAME_COL = next((c for c in ("ward_name", "ward_name_readme", "name")
                 if c in df.columns), None)

FEATURES = BUNDLE["feature_names"]
scaler   = BUNDLE["scaler"]

@st.cache_data
def build_feature_matrix():
    w = pd.read_csv(os.path.join(BASE,
        "hyderabad_150_ward_SUPER_wide_worldpop_v4.csv"), low_memory=False)
    with open(os.path.join(BASE, "ghmc_wards_full_150.geojson"),
              "r", encoding="utf-8") as f:
        g = json.load(f)
    sp = []
    for feat in g["features"]:
        wn = feat["properties"].get("ward_number")
        if wn is None: continue
        shp = shape(feat["geometry"])
        c = shp.centroid
        sp.append({"ward_id": int(wn),
                   "ward_area_km2": shp.area * 106 * 111,
                   "centroid_lon": c.x, "centroid_lat": c.y})
    sp = pd.DataFrame(sp)
    cx, cy = 78.4738, 17.4239
    sp["dist_center_km"] = np.sqrt(
        ((sp["centroid_lon"] - cx) * 106)**2 +
        ((sp["centroid_lat"] - cy) * 111)**2)
    w = w.merge(sp, on="ward_id", how="left")
    pres = pd.read_csv(os.path.join(BASE,
        "hyderabad_150_ward_pressure.csv"))
    w = w.merge(pres[["ward_id", "pressure_zscore"]], on="ward_id", how="left")
    am = pd.read_csv(os.path.join(BASE, "ghmc_amenities_ward_counts.csv"))
    w["amenity_count"] = w["ward_id"].map(
        dict(zip(am["ward_id"], am["amenity_count"]))).fillna(0)
    w["amenity_log"] = np.log1p(w["amenity_count"])
    w["pop_density_log"] = np.log1p(
        w["census2011_population_verified"] / w["ward_area_km2"])
    return w[["ward_id"] + FEATURES].copy()

WFEAT = build_feature_matrix().set_index("ward_id").reindex(df["ward_id"])
X_base_raw = WFEAT.fillna(WFEAT.median())
X_base_scaled = scaler.transform(X_base_raw)

gbm_2011 = BUNDLE["models"]["svi_2011"]["model"]
gbm_2030 = BUNDLE["models"]["svi_2030"]["model"]

# =========================================================
# Sidebar
# =========================================================
st.sidebar.markdown("## 🗺️ Hyderabad SVI")
st.sidebar.caption("Socioeconomic Vulnerability · 150 GHMC wards")

year = st.sidebar.select_slider("Year", options=[2011, 2025, 2027, 2030],
                                value=2011)
elev = st.sidebar.slider("3D bar height", 20, 120, 60, step=5,
                         help="Only affects the 3D View tab.")
show_low = st.sidebar.checkbox("Show population-only wards", value=True)

st.sidebar.divider()
st.sidebar.markdown("🟢 0–30 · low\n\n🟡 31–60 · moderate\n\n🔴 61–100 · high")

# =========================================================
# Scenario simulator
# =========================================================
st.sidebar.divider()
st.sidebar.markdown("### Scenario Simulator")
st.sidebar.caption("Model-based counterfactual. Not a causal claim.")

if "scen_lit"   not in st.session_state: st.session_state.scen_lit   = 0
if "scen_nw"    not in st.session_state: st.session_state.scen_nw    = 0
if "scen_child" not in st.session_state: st.session_state.scen_child = 0

scen_lit_pp = st.sidebar.slider("Literacy change (pp)", -20, 20,
                                st.session_state.scen_lit,
                                step=1, key="scen_lit")
scen_nw_pp = st.sidebar.slider("Non-worker rate change (pp)", -20, 20,
                               st.session_state.scen_nw,
                               step=1, key="scen_nw")
scen_child_pp = st.sidebar.slider("Child (0-6) share change (pp)", -10, 10,
                                  st.session_state.scen_child,
                                  step=1, key="scen_child")

if st.sidebar.button("Reset scenario"):
    st.session_state.scen_lit = 0
    st.session_state.scen_nw = 0
    st.session_state.scen_child = 0
    st.rerun()

scenario_active = (scen_lit_pp != 0) or (scen_nw_pp != 0) or (scen_child_pp != 0)

# =========================================================
# Counterfactual via GBM
# =========================================================
SCENARIO_FEATURES = {
    "census2011_literacy_rate":   scen_lit_pp   / 100.0,
    "census2011_non_worker_rate": scen_nw_pp    / 100.0,
    "census2011_child_share":     scen_child_pp / 100.0,
}

X_pert_raw = X_base_raw.copy()
for f, delta in SCENARIO_FEATURES.items():
    if f in X_pert_raw.columns:
        X_pert_raw[f] = (X_pert_raw[f] + delta).clip(0, 1)
X_pert_scaled = scaler.transform(X_pert_raw)

pred_base_2011 = gbm_2011.predict(X_base_scaled)
pred_scen_2011 = gbm_2011.predict(X_pert_scaled)
pred_base_2030 = gbm_2030.predict(X_base_scaled)
pred_scen_2030 = gbm_2030.predict(X_pert_scaled)

scen_map = pd.DataFrame({
    "ward_id": df["ward_id"].values,
    "delta_2011": pred_scen_2011 - pred_base_2011,
    "delta_2030": pred_scen_2030 - pred_base_2030,
})
df = df.merge(scen_map, on="ward_id", how="left")

if year == 2011:
    df["svi_baseline"] = df["svi_score"]
    df["scenario_delta"] = df["delta_2011"]
else:
    df["svi_baseline"] = df[f"svi_{year}_central"]
    df["scenario_delta"] = df["delta_2030"]

df["svi_scenario"] = np.clip(df["svi_baseline"] + df["scenario_delta"], 0, 100)
df["svi_current"] = df["svi_scenario"] if scenario_active else df["svi_baseline"]

# =========================================================
# Enrich GeoJSON
# =========================================================
def band_color_hex(v):
    if pd.isna(v): return "#787878"
    if v <= 30:  return "#2ecc71"
    if v <= 60:  return "#f1c40f"
    return "#e74c3c"

def band_color_rgb(v):
    if pd.isna(v): return [120, 120, 120]
    if v <= 30:  return [46, 204, 113]
    if v <= 60:  return [241, 196, 15]
    return [231, 76, 60]

lookup = df.set_index("ward_id").to_dict(orient="index")

enriched = []
for feat in GEO["features"]:
    wn = feat["properties"].get("ward_number")
    if wn is None: continue
    wn = int(wn)
    p = lookup.get(wn, {})
    vb = p.get("svi_baseline"); vs = p.get("svi_current")
    vb = float(vb) if vb is not None and not pd.isna(vb) else 0.0
    vs = float(vs) if vs is not None and not pd.isna(vs) else 0.0
    conf = str(p.get("pressure_confidence", ""))

    if not show_low and conf in ("low", "medium"):
        fill_hex = "#282828"
        fill_rgb = [40, 40, 40]
        elev_val = 0.0
        display_val = None
    else:
        fill_hex = band_color_hex(vs)
        fill_rgb = band_color_rgb(vs)
        elev_val = vs
        display_val = vs

    name = str(feat["properties"].get("name", ""))
    enriched.append({
        "type": "Feature",
        "geometry": feat["geometry"],
        "properties": {
            "ward_id": wn,
            "ward_name": name,
            "svi_show": round(vs, 1),
            "svi_base": round(vb, 1),
            "delta": round(vs - vb, 1),
            "confidence": conf,
            "fill_hex": fill_hex,
            "fill_rgb": fill_rgb,
            "elev": elev_val,
            "display_val": display_val,
        },
    })

enriched_geo = {"type": "FeatureCollection", "features": enriched}

# =========================================================
# Tabs
# =========================================================
tab_3d, tab_click = st.tabs(["🎨 3D View (visual)", "🖱️ Interactive Map (click)"])

# =========================================================
# TAB 1 — PyDeck 3D
# =========================================================
with tab_3d:
    st.caption("3D extrusion visualisation. Use the sidebar dropdown below "
               "to select a ward, or switch to the Interactive Map tab to click.")

    if scenario_active:
        st.info(f"**Scenario active.** Literacy {scen_lit_pp:+d}pp · "
                f"Non-worker {scen_nw_pp:+d}pp · Child share {scen_child_pp:+d}pp.")

    layer = pdk.Layer(
        "GeoJsonLayer",
        data=enriched_geo,
        extruded=True,
        wireframe=True,
        get_elevation="properties.elev",
        elevation_scale=elev,
        get_fill_color="properties.fill_rgb",
        get_line_color=[255, 255, 255, 180],
        line_width_min_pixels=0.5,
        pickable=False,
    )
    view = pdk.ViewState(latitude=17.42, longitude=78.45,
                         zoom=10.4, pitch=52, bearing=-17)

    # Use Mapbox-free basemap: Carto via visualisation (no API key needed for
    # the vector style used by deck.gl's visualisation mode)
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json",
        tooltip=False,
    )
    st.pydeck_chart(deck, width="stretch")

# =========================================================
# TAB 2 — Folium interactive (click works)
# =========================================================
with tab_click:
    st.caption("Click a ward on the map, or use the sidebar dropdown.")

    if scenario_active:
        st.info(f"**Scenario active.** Literacy {scen_lit_pp:+d}pp · "
                f"Non-worker {scen_nw_pp:+d}pp · Child share {scen_child_pp:+d}pp. "
                f"Map shows scenario-adjusted SVI.")

    m = folium.Map(
        location=[17.42, 78.45],
        zoom_start=11,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        control_scale=True,
    )

    for feat in enriched:
        props = feat["properties"]
        wn = props["ward_id"]
        v  = props["svi_show"]

        if props["display_val"] is None:
            tooltip = f"<b>Ward {wn}</b> — {props['ward_name']}<br/>hidden (low confidence)"
        else:
            tooltip = f"<b>Ward {wn}</b> — {props['ward_name']}<br/>SVI: {v:.1f}"
            if scenario_active:
                tooltip += (f"<br/>Baseline: {props['svi_base']:.1f}"
                            f"<br/>Δ: {props['delta']:+.1f}")

        folium.GeoJson(
            data={"type": "Feature", "geometry": feat["geometry"],
                  "properties": props},
            style_function=lambda x, f=props["fill_hex"]: {
                "fillColor": f,
                "color": "#ffffff",
                "weight": 0.4,
                "fillOpacity": 0.82,
            },
            highlight_function=lambda x: {
                "weight": 2.5,
                "color": "#ffffff",
                "fillOpacity": 0.95,
            },
            tooltip=folium.Tooltip(tooltip, sticky=True),
        ).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background-color: rgba(20,20,20,0.85); padding: 10px 14px;
                border-radius: 6px; font-family: sans-serif; font-size: 13px;
                color: #fafafa; line-height: 1.6;">
    <b>SVI score</b><br>
    <span style="color:#2ecc71">●</span> 0–30 &nbsp; low<br>
    <span style="color:#f1c40f">●</span> 31–60 moderate<br>
    <span style="color:#e74c3c">●</span> 61–100 high
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    map_state = st_folium(
        m,
        width="100%",
        height=560,
        returned_objects=["last_active_drawing"],
        key=f"ward_map_{year}_{scenario_active}_{scen_lit_pp}_{scen_nw_pp}_{scen_child_pp}",
    )

    def get_clicked(state):
        if not state:
            return None
        drawing = state.get("last_active_drawing")
        if not drawing:
            return None
        props = drawing.get("properties", {})
        if isinstance(props, dict):
            return props.get("ward_id")
        return None

    clicked = get_clicked(map_state)
    st.session_state["last_clicked_ward"] = clicked

# =========================================================
# Ward selector (works regardless of tab)
# =========================================================
st.sidebar.divider()
manual = st.sidebar.selectbox(
    "Or jump to a ward",
    ["(use map click)"] + sorted(df["ward_id"].tolist())
)

selected = None
if manual != "(use map click)":
    hit = df[df["ward_id"] == manual]
    if len(hit): selected = hit.iloc[0]
elif st.session_state.get("last_clicked_ward") is not None:
    hit = df[df["ward_id"] == st.session_state["last_clicked_ward"]]
    if len(hit): selected = hit.iloc[0]

# =========================================================
# Detail panel
# =========================================================
st.divider()
st.markdown("### Ward detail")

if selected is None:
    st.info("Click a ward on the map, or pick one from the sidebar dropdown.")
else:
    r = selected
    base_svi = float(r["svi_baseline"])
    scen_svi = float(r["svi_scenario"])
    scen_delta = scen_svi - base_svi

    st.markdown(f"**Ward {r['ward_id']} — {r.get(NAME_COL, '—')}**")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"SVI {year}", f"{base_svi:.1f}")
    c2.metric("Band", str(r.get("svi_band_percentile", "—")))
    c3.metric("Pressure", f"{r['pressure_zscore']:+.2f}")
    c4.metric("Confidence", r.get("pressure_confidence", "—"))
    if scenario_active:
        c5.metric("Scenario SVI", f"{scen_svi:.1f}",
                  delta=f"{scen_delta:+.1f}",
                  delta_color="inverse")
    else:
        c5.metric("Scenario SVI", "—",
                  help="Move the sidebar sliders to activate.")

    cc1, cc2 = st.columns([1, 1])

    with cc1:
        st.markdown("**Why this score? (feature attributions)**")
        st.caption("Positive = pushes vulnerability up. "
                   "Local permutation method.")
        attr_prefix = "shap11_" if year == 2011 else "shap30_"
        attr_cols = [c for c in df.columns if c.startswith(attr_prefix)]
        if attr_cols:
            att = [(c.replace(attr_prefix, ""), float(r[c]))
                   for c in attr_cols if pd.notna(r[c])]
            att.sort(key=lambda x: abs(x[1]), reverse=True)
            top = att[:8]
            labels = [FEATURE_LABELS.get(f, f) for f, _ in top][::-1]
            vals = [v for _, v in top][::-1]
            bar = go.Figure(go.Bar(
                x=vals, y=labels, orientation="h",
                marker_color=["#e74c3c" if v > 0 else "#4a90e2" for v in vals],
            ))
            bar.update_layout(
                height=340, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="attribution (SVI points)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fafafa"),
            )
            st.plotly_chart(bar, width="stretch")

    with cc2:
        st.markdown("**SVI trajectory 2011 → 2030**")
        traj = pd.DataFrame({
            "year": [2011, 2025, 2027, 2030],
            "central": [r["svi_score"], r.get("svi_2025_central"),
                        r.get("svi_2027_central"), r.get("svi_2030_central")],
            "lower": [r["svi_score"], r.get("svi_2025_lower"),
                      r.get("svi_2027_lower"), r.get("svi_2030_lower")],
            "upper": [r["svi_score"], r.get("svi_2025_upper"),
                      r.get("svi_2027_upper"), r.get("svi_2030_upper")],
        }).dropna()
        line = go.Figure()
        line.add_trace(go.Scatter(x=traj["year"], y=traj["upper"],
                                  mode="lines", line=dict(width=0),
                                  hoverinfo="skip", showlegend=False))
        line.add_trace(go.Scatter(x=traj["year"], y=traj["lower"],
                                  mode="lines", line=dict(width=0),
                                  fill="tonexty",
                                  fillcolor="rgba(120,120,120,0.25)",
                                  name="uncertainty"))
        line.add_trace(go.Scatter(x=traj["year"], y=traj["central"],
                                  mode="lines+markers",
                                  line=dict(width=3, color="#4a90e2"),
                                  name="central"))
        if scenario_active:
            line.add_trace(go.Scatter(
                x=[year], y=[scen_svi], mode="markers",
                marker=dict(size=14, color="#f39c12", symbol="diamond"),
                name="scenario"))
        line.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0),
                           yaxis_title="SVI", xaxis_title="year",
                           plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)",
                           font=dict(color="#fafafa"),
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(line, width="stretch")

# =========================================================
# Footer
# =========================================================
st.divider()
f1, f2, f3, f4 = st.columns(4)
f1.metric("Wards", len(df))
f2.metric("Full UDISE pressure",
          int((df["pressure_confidence"] == "high").sum()))
f3.metric("Population-only pressure",
          int((df["pressure_confidence"] == "medium").sum()))
if scenario_active:
    n_improved = int((df["scenario_delta"] < -1).sum())
    n_worsened = int((df["scenario_delta"] > 1).sum())
    f4.metric("Scenario impact",
              f"{n_improved} better · {n_worsened} worse")
else:
    shifted = int(((df["svi_score"] > 60) != (df["svi_2030_central"] > 60)).sum())
    f4.metric("Band shifts by 2030", shifted)