"""Offline end-to-end test of the LIVE pipeline.

Replaces the two network calls (element sets, space-weather drivers) with real archived
TLEs and synthetic quiet drivers, then runs the daily pipeline for N consecutive days in a
temporary root. Checks that densities, forecasts, registry commits, verification, scoring,
bulletins and the website are all produced.

    python scripts/offline_e2e.py <tles_clean.parquet> [n_objects] [n_days]
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
tmp = Path(tempfile.mkdtemp(prefix="obs-e2e-"))
for sub in ("store", "docs/feed"):
    (tmp / sub).mkdir(parents=True, exist_ok=True)
os.environ["OBSERVATORY_ROOT"] = str(tmp)
sys.path.insert(0, str(REPO))

from observatory import pipeline  # noqa: E402
from observatory import config as C  # noqa: E402
from observatory.referee import Registry  # noqa: E402

tle_path = sys.argv[1]
n_obj = int(sys.argv[2]) if len(sys.argv) > 2 else 60
n_days = int(sys.argv[3]) if len(sys.argv) > 3 else 20

t = pd.read_parquet(tle_path, columns=["norad", "jd", "n", "ecc", "l1", "l2"])
t["n"] = t.n * 1440 / (2 * np.pi)                      # archive stores rad/min; feeds give rev/day
rng = np.random.default_rng(3)
objs = rng.choice(t.norad.unique(), n_obj, replace=False)
t = t[t.norad.isin(objs)]
pd.DataFrame({"norad": sorted(objs)}).to_csv(C.STORE / "network.csv", index=False)
for f in ("thermoswarm-gb.json", "conformal_state.json"):
    if (REPO / "store" / f).exists():
        shutil.copy(REPO / "store" / f, C.STORE / f)

# synthetic quiet drivers spanning the archive
idx = pd.date_range("2025-06-01", "2026-12-31", freq="3h")
three = pd.DataFrame({"kp": 2.0, "ap": 7.0}, index=idx)
days = pd.date_range("2025-06-01", "2026-12-31", freq="D")
daily = pd.DataFrame({"Ap": 7.0, "f107": 140.0, "f107a": 140.0}, index=days)

start = pd.Timestamp("2025-10-25 06:17")
for k in range(n_days):
    now = start + pd.Timedelta(days=k)
    jd_now = now.to_julian_date()
    pipeline.fetch_elements = lambda norads, jd_now=jd_now: ("archive", t[(t.jd <= jd_now) & (t.jd > jd_now - 5)])
    pipeline.fetch_drivers = lambda: (three, daily, None)
    st = pipeline.run(now)
    log = st["log"]
    print(f"{now:%Y-%m-%d}: sets={log['element_sets']} new_iv={log['new_intervals']} new_win={log['new_windows']} "
          f"verified={log['verified']} issued={log.get('issued', 0)} | {st['registry']}")

reg = Registry(C.REGISTRY)
ok, msg = reg.verify()
sc = reg.scores()
print("\nregistry:", msg)
print("scored pairs:", len(sc), "models:", sorted(sc.model.unique()) if len(sc) else [])
print("site:", (C.DOCS / "index.html").exists(), "feed:", (C.DOCS / "feed" / "latest.json").exists())
assert ok and len(sc) > 0 and (C.DOCS / "index.html").exists()
print("E2E OK  root:", tmp)
