# integrate_census_correct.py
import os
import pandas as pd

BASE      = os.path.dirname(os.path.abspath(__file__))
WIDE_IN   = os.path.join(BASE, "hyderabad_150_ward_SUPER_wide_worldpop_v3.csv")
PANEL_IN  = os.path.join(BASE, "hyderabad_150_ward_SUPER_panel_worldpop_v3.csv")
CENSUS_IN = os.path.join(BASE, "hyderabad_2011_census_150_ward_master.csv")
WIDE_OUT  = os.path.join(BASE, "hyderabad_150_ward_SUPER_wide_worldpop_v4.csv")
PANEL_OUT = os.path.join(BASE, "hyderabad_150_ward_SUPER_panel_worldpop_v4.csv")

# ---- Load ----
wide  = pd.read_csv(WIDE_IN, low_memory=False)
cens  = pd.read_csv(CENSUS_IN, low_memory=False)
panel = pd.read_csv(PANEL_IN, low_memory=False)

assert len(cens) == 150, f"Census master must have 150 rows, got {len(cens)}"
assert cens["ward_id"].nunique() == 150, "Census master must have 150 unique wards"
print(f"Wide v3:   {wide.shape}")
print(f"Census:    {cens.shape}")
print(f"Panel v3:  {panel.shape}")

# ---- Column mapping (census master -> wide naming convention) ----
rename_map = {
    "TOT_P":                   "census2011_population_verified",
    "TOT_M":                   "census2011_TOT_M",
    "TOT_F":                   "census2011_TOT_F",
    "No_HH":                   "census2011_No_HH",
    "P_06":                    "census2011_P_06",
    "P_SC":                    "census2011_P_SC",
    "P_ST":                    "census2011_P_ST",
    "P_LIT":                   "census2011_P_LIT",
    "P_ILL":                   "census2011_P_ILL",
    "TOT_WORK_P":              "census2011_TOT_WORK_P",
    "MAINWORK_P":              "census2011_MAINWORK_P",
    "MARGWORK_P":              "census2011_MARGWORK_P",
    "NON_WORK_P":              "census2011_NON_WORK_P",
    "literacy_rate":           "census2011_literacy_rate",
    "work_participation_rate": "census2011_work_participation_rate",
    "non_worker_rate":         "census2011_non_worker_rate",
    "sc_share":                "census2011_sc_share",
    "st_share":                "census2011_st_share",
    "female_share":            "census2011_female_share",
    "child_share":             "census2011_child_share",
    "main_worker_share":       "census2011_main_worker_share",
    "marginal_worker_share":   "census2011_marginal_worker_share",
    "source_districts":        "census2011_source_district",
    "source_rows":             "census2011_source_rows",
}

census_block = cens[["ward_id"] + list(rename_map.keys())].rename(columns=rename_map)

# ---- Drop every existing census column from wide, then merge the corrected block ----
old_census_cols = [c for c in wide.columns if c.startswith("census2011_")]
print(f"\nDropping {len(old_census_cols)} stale census columns from wide")
wide = wide.drop(columns=old_census_cols)

wide = wide.merge(census_block, on="ward_id", how="left")

# ---- Derived combos ----
wide["census2011_sc_st_share"] = wide["census2011_sc_share"] + wide["census2011_st_share"]
wide["census2011_literacy_risk"] = 1.0 - wide["census2011_literacy_rate"]

# ---- Sanity ----
nan_pop  = wide["census2011_population_verified"].isna().sum()
nan_lit  = wide["census2011_literacy_rate"].isna().sum()
nan_nowk = wide["census2011_non_worker_rate"].isna().sum()
nan_scst = wide["census2011_sc_st_share"].isna().sum()

print(f"\nCensus NaN counts after merge:")
print(f"  population : {nan_pop}")
print(f"  literacy   : {nan_lit}")
print(f"  non-worker : {nan_nowk}")
print(f"  sc_st      : {nan_scst}")

assert nan_pop == 0, "Some wards still missing Census population"
assert nan_lit == 0, "Some wards still missing Census literacy"

print(f"\nTotal Census population: {wide['census2011_population_verified'].sum():,.0f}")

wide.to_csv(WIDE_OUT, index=False)
print(f"\nWrote {WIDE_OUT}   shape={wide.shape}")

# ---- Refresh the panel's 2011 rows ----
panel_non2011 = panel[panel["year"] != 2011].copy()
panel_non2011 = panel_non2011.drop(
    columns=[c for c in panel_non2011.columns if c.startswith("census2011_")],
    errors="ignore",
)

c2011_cols = [c for c in wide.columns if c.startswith("census2011_")]
panel_2011 = wide[["ward_id", "ward_name_readme"] + c2011_cols].copy()
panel_2011 = panel_2011.rename(columns={"ward_name_readme": "ward_name"})
panel_2011["year"] = 2011

panel_out = pd.concat([panel_non2011, panel_2011], ignore_index=True, sort=False)
panel_out = panel_out.sort_values(["ward_id", "year"]).reset_index(drop=True)

panel_out.to_csv(PANEL_OUT, index=False)
print(f"Wrote {PANEL_OUT}   shape={panel_out.shape}")

# ---- Spot check ----
print("\nSpot check — 5 boundary wards that were wrong in v3:")
sample = wide[wide["ward_id"].isin([1, 2, 3, 4, 5])][
    ["ward_id", "census2011_population_verified",
     "census2011_literacy_rate", "census2011_non_worker_rate",
     "census2011_sc_st_share"]
]
print(sample.to_string(index=False))