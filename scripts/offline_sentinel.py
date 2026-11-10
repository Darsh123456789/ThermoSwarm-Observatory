"""Offline end-to-end run of the Sentinel layer on an archive of element sets (no network).
Builds the catalogue as it stood on DAY, runs 72-h screening, storm-aware Pc, avoidance options and
anomaly detection, and writes docs/feed/sentinel.json into a temporary root.
usage: python scripts/offline_sentinel.py <tles.parquet> [YYYY-MM-DD] [hours]"""
import os, sys, tempfile, shutil, json, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
tmp = Path(tempfile.mkdtemp(prefix="sentinel-e2e-"))
os.environ["OBSERVATORY_ROOT"] = str(tmp)
sys.path.insert(0, str(REPO))
import numpy as np, pandas as pd
from observatory import config as C
for f in ("errmodel.json", "reentry_windows.json"):
    if (REPO / "store" / f).exists():
        shutil.copy(REPO / "store" / f, C.STORE / f)
from observatory.sentinel import run

day = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2025-11-12")
hours = float(sys.argv[3]) if len(sys.argv) > 3 else 72
t = pd.read_parquet(sys.argv[1])
jd = day.to_julian_date()
h = t[(t.jd < jd) & (t.jd > jd - 45)].copy()
h["perigee_km"] = h.a_km * (1 - h.ecc) - C.RE_KM
h["apogee_km"] = h.a_km * (1 + h.ecc) - C.RE_KM
rng = np.random.default_rng(0)
cat = h.sort_values("jd").groupby("norad").tail(1).copy()
cat = cat[cat.jd > jd - 5]
cat["name"] = "OBJ " + cat.norad.astype(str)
cat["type"] = rng.choice(["PAYLOAD", "DEBRIS", "ROCKET BODY"], len(cat), p=[0.6, 0.3, 0.1])   # archive has no SATCAT types
cat["rcs"] = rng.choice(["SMALL", "MEDIUM", "LARGE"], len(cat))
cat["intl"] = ""
bt = {k: dict(bound72_km=b, shift72_km=0.0) for k, b in enumerate([380, 189, 57, 19, 8.3, 5.2, 3.4, 0.9, 1.4, 4.1, 4, 4])}
t0 = time.time()
out = run(day, band_terms=bt, regime="storm", cat=cat, source=None, hours=hours, hist=h)
print(json.dumps(dict(counts=out["counts"], log=out["log"]), indent=1, default=str))
print("top events:")
for e in out["events"][:5]:
    print({k: e[k] for k in ("i", "j", "tca", "miss_km", "vrel_kms", "pc_nominal", "pc_storm", "pc_max", "cat_nominal", "cat_max")})
for p in out["plans"][:3]:
    print("plan:", {k: p.get(k) for k in ("i", "j", "manoeuvring", "pc_before")}, p.get("recommended") or p.get("note"))
print("feed:", C.DOCS / "feed" / "sentinel.json", f"{(C.DOCS / 'feed' / 'sentinel.json').stat().st_size/1e3:.0f} kB", f"total {time.time()-t0:.0f}s")
