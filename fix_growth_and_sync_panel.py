# fix_growth_and_sync_panel.py
import os
import pandas as pd

BASE   = os.path.dirname(os.path.abspath(__file__))
WIDE   = os.path.join(BASE, "hyderabad_150_ward_SUPER_wide_worldpop_v3.csv")
PANEL  = os.path.join(BASE, "hyderabad_150_ward_SUPER_panel_worldpop_v3.csv")
MISSING = [3, 4, 11, 13, 113]

# --- 1) Fix WIDE growth columns for the 5 touched wards ---
w = pd.read_csv(WIDE)
m = w["ward_id"].isin(MISSING)

a = w.loc[m, "worldpop_population_2015"]
b = w.loc[m, "worldpop_population_2018"]

w.loc[m, "worldpop_abs_growth_2015_2018"] = b - a
w.loc[m, "worldpop_pct_growth_2015_2018"] = (b - a) / a * 100
w.loc[m, "worldpop_annualized_growth_2015_2018_pct"] = ((b / a) ** (1 / 3) - 1) * 100

w.to_csv(WIDE, index=False)
print("=== WIDE: 5 touched wards ===")
print(w.loc[m, ["ward_id", "worldpop_population_2015", "worldpop_population_2018",
                "worldpop_abs_growth_2015_2018",
                "worldpop_pct_growth_2015_2018",
                "worldpop_annualized_growth_2015_2018_pct"]].to_string(index=False))

# --- 2) Propagate into the PANEL (2015 and 2018 rows) ---
p = pd.read_csv(PANEL)

pop15 = dict(zip(w["ward_id"], w["worldpop_population_2015"]))
pop18 = dict(zip(w["ward_id"], w["worldpop_population_2018"]))

mask15 = (p["year"] == 2015) & (p["ward_id"].isin(MISSING))
mask18 = (p["year"] == 2018) & (p["ward_id"].isin(MISSING))

p.loc[mask15, "worldpop_population"] = p.loc[mask15, "ward_id"].map(pop15)
p.loc[mask18, "worldpop_population"] = p.loc[mask18, "ward_id"].map(pop18)

p.to_csv(PANEL, index=False)

print("\n=== PANEL: verify the 5 wards now have values ===")
print(p[p["ward_id"].isin(MISSING) & p["year"].isin([2015, 2018])]
      [["ward_id", "year", "worldpop_population"]]
      .sort_values(["ward_id", "year"]).to_string(index=False))

# --- 3) Global NaN check ---
nan15 = w["worldpop_population_2015"].isna().sum()
nan18 = w["worldpop_population_2018"].isna().sum()
print(f"\nWIDE NaN counts:  2015={nan15}   2018={nan18}")