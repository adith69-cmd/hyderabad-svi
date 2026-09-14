# fetch_missing_wards.py  — one relation per request, GDAL-free
import os, sys, time, json, re
import requests

BASE     = os.path.dirname(os.path.abspath(__file__))
EXISTING = os.path.join(BASE, "ghmc_wards_uploaded_corrected_145.geojson")
OUT      = os.path.join(BASE, "ghmc_wards_full_150.geojson")

MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
HEADERS = {
    "User-Agent": "ghmc-ward-fetch/1.0 (research)",
    "Accept": "application/json",
}

MISSING = {3: 7848845, 4: 7852319, 11: 7849316, 13: 7848886, 113: 7848829}

def fetch_one(rid):
    q = f"[out:json][timeout:90]; relation({rid}); out geom;"
    for url in MIRRORS:
        try:
            print(f"  trying {url.split('//')[1].split('/')[0]} …", flush=True)
            r = requests.post(url, data={"data": q}, headers=HEADERS, timeout=120)
            print(f"    HTTP {r.status_code}", flush=True)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            print(f"    exception: {e}", flush=True)
        time.sleep(3)
    return None

def overpass_to_polygon(el):
    outer_rings, inner_rings = [], []
    for m in el.get("members", []):
        if m.get("type") != "way":
            continue
        geom = m.get("geometry")
        if not geom:
            continue
        ring = [[pt["lon"], pt["lat"]] for pt in geom]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        (inner_rings if m.get("role") == "inner" else outer_rings).append(ring)
    if not outer_rings:
        return None
    if len(outer_rings) == 1 and not inner_rings:
        return {"type": "Polygon", "coordinates": [outer_rings[0]]}
    return {"type": "MultiPolygon",
            "coordinates": [[r] for r in outer_rings] + [[r] for r in inner_rings]}

new_features = []
for wnum, rid in MISSING.items():
    print(f"\nWard {wnum}  relation/{rid}", flush=True)
    data = fetch_one(rid)
    if not data or not data.get("elements"):
        print(f"  FAILED to fetch Ward {wnum}", flush=True)
        continue
    el = data["elements"][0]
    poly = overpass_to_polygon(el)
    if not poly:
        print(f"  no polygon for Ward {wnum}", flush=True)
        continue
    new_features.append({
        "type": "Feature",
        "properties": {
            "name": el.get("tags", {}).get("name", ""),
            "ward_number": wnum,
            "@id": f"relation/{rid}",
            "source": "OSM Overpass (OpenCity-derived)",
        },
        "geometry": poly,
    })
    print(f"  OK", flush=True)
    time.sleep(4)

print(f"\nFetched {len(new_features)} of {len(MISSING)} polygons", flush=True)

with open(EXISTING, "r", encoding="utf-8") as f:
    existing = json.load(f)

existing_feats = existing.get("features", [])
print(f"Existing GeoJSON: {len(existing_feats)} features", flush=True)

def extract_ward_num(props):
    wn = props.get("ward_number")
    if wn is not None:
        try:
            return int(wn)
        except Exception:
            pass
    for fld in ("name", "ward_name_readme"):
        v = props.get(fld)
        if isinstance(v, str):
            m = re.search(r"Ward\s+(\d+)", v)
            if m:
                return int(m.group(1))
    return None

for f in existing_feats:
    if "ward_number" not in f["properties"]:
        wn = extract_ward_num(f["properties"])
        if wn is not None:
            f["properties"]["ward_number"] = wn

seen = set()
merged_feats = []
for f in existing_feats + new_features:
    fid = f["properties"].get("@id")
    if fid and fid in seen:
        continue
    if fid:
        seen.add(fid)
    merged_feats.append(f)

print(f"After merge + dedupe: {len(merged_feats)} features", flush=True)

nums = set()
unresolved = 0
for f in merged_feats:
    wn = f["properties"].get("ward_number")
    if wn is None:
        unresolved += 1
    else:
        nums.add(int(wn))

still_missing = sorted(set(range(1, 151)) - nums)
print(f"Unique ward numbers: {len(nums)}", flush=True)
if unresolved:
    print(f"WARNING: {unresolved} features have no ward_number", flush=True)
if still_missing:
    print(f"STILL MISSING wards: {still_missing}", flush=True)
else:
    print("All 150 ward numbers present OK", flush=True)

out = {
    "type": "FeatureCollection",
    "generator": "merge_145_plus_5",
    "crs": {"type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
    "features": merged_feats,
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)

print(f"\nWrote {OUT}", flush=True)