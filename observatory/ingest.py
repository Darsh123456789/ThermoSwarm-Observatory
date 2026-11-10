"""Live data collection. Raw element sets go to .cache/ only (never committed, never
published: Space-Track's user agreement forbids redistribution). Only derived densities
are stored in the repository.

Element-set providers, tried in order:
  spacetrack - needs SPACETRACK_USER / SPACETRACK_PASS (GitHub secrets); queries gp_history
               for the network objects over the last few days, so no history cache is needed.
  tracss     - TraCSS public API (no account). Set TRACSS_OMM_URL to its OMM endpoint
               (CCSDS OMM JSON); snapshots accumulate in the cache.
  celestrak  - CelesTrak GP groups (no account), current elements only; snapshots accumulate.
"""
import io
import json
import os
import time

import numpy as np
import pandas as pd
import requests

from . import config as C
from .drivers import GFZ_NOWCAST, SWPC_3DAY, parse_gfz, parse_swpc_3day

UA = {"User-Agent": "ThermoSwarm-Observatory/0.1 (+https://github.com/Darsh123456789/ThermoSwarm-Observatory)"}
CELESTRAK_GROUPS = ["last-30-days", "active", "cosmos-1408-debris", "fengyun-1c-debris",
                    "iridium-33-debris", "cosmos-2251-debris"]


def _omm_to_rows(records):
    """CCSDS OMM (JSON) records -> element rows with TLE lines (synthesised if absent)."""
    from sgp4.api import Satrec, WGS72
    from sgp4 import omm, exporter
    rows = []
    for r in records:
        try:
            norad = int(r["NORAD_CAT_ID"])
            l1, l2 = r.get("TLE_LINE1"), r.get("TLE_LINE2")
            if not l1 or not l2:
                sat = Satrec()
                omm.initialize(sat, r)
                l1, l2 = exporter.export_tle(sat)
            sat = Satrec.twoline2rv(l1, l2, WGS72)
            rows.append(dict(norad=norad, jd=sat.jdsatepoch + sat.jdsatepochF, l1=l1, l2=l2,
                             n=float(r.get("MEAN_MOTION", sat.no_kozai * 1440 / (2 * np.pi))),
                             ecc=float(r.get("ECCENTRICITY", sat.ecco))))
        except Exception:
            continue
    return pd.DataFrame(rows)


def fetch_spacetrack(norads, days=5):
    user, pw = os.environ.get("SPACETRACK_USER"), os.environ.get("SPACETRACK_PASS")
    if not user or not pw:
        return None
    s = requests.Session()
    s.headers.update(UA)
    s.post("https://www.space-track.org/ajaxauth/login", data={"identity": user, "password": pw},
           timeout=60).raise_for_status()
    out = []
    ids = sorted(set(int(x) for x in norads))
    for k in range(0, len(ids), 400):                      # stay well inside rate limits
        chunk = ",".join(map(str, ids[k:k + 400]))
        url = ("https://www.space-track.org/basicspacedata/query/class/gp_history/"
               f"NORAD_CAT_ID/{chunk}/EPOCH/>now-{days}/orderby/NORAD_CAT_ID,EPOCH/format/json")
        r = s.get(url, timeout=180)
        r.raise_for_status()
        out.extend(r.json())
        time.sleep(3)
    s.get("https://www.space-track.org/ajaxauth/logout", timeout=30)
    return _omm_to_rows(out)


def fetch_tracss():
    url = os.environ.get("TRACSS_OMM_URL")
    if not url:
        return None
    r = requests.get(url, headers=UA, timeout=300)
    r.raise_for_status()
    js = r.json()
    records = js if isinstance(js, list) else js.get("data", js.get("omm", []))
    return _omm_to_rows(records)


def fetch_celestrak():
    recs = []
    for g in CELESTRAK_GROUPS:
        r = requests.get(f"https://celestrak.org/NORAD/elements/gp.php?GROUP={g}&FORMAT=json",
                         headers=UA, timeout=180)
        if r.ok:
            recs.extend(r.json())
        time.sleep(2)
    return _omm_to_rows(recs)


def fetch_elements(norads):
    """Return (provider_name, elements) from the first provider that is configured and works."""
    errors = []
    for name, fn in (("spacetrack", lambda: fetch_spacetrack(norads)),
                     ("tracss", fetch_tracss), ("celestrak", fetch_celestrak)):
        try:
            e = fn()
            if e is not None and len(e):
                e = e[e.norad.isin(set(int(x) for x in norads))]
                return name, e
        except Exception as ex:
            errors.append(f"{name}: {ex}")
    raise RuntimeError("no element-set provider worked: " + "; ".join(errors))


def update_cache(new, keep_days=45):
    path = C.CACHE / "elements.parquet"
    old = pd.read_parquet(path) if path.exists() else new.iloc[:0]
    e = pd.concat([old, new], ignore_index=True).drop_duplicates(["norad", "jd"])
    e = e[e.jd >= e.jd.max() - keep_days]
    e.to_parquet(path)
    return e


def fetch_drivers():
    r = requests.get(GFZ_NOWCAST, headers=UA, timeout=120)
    r.raise_for_status()
    three, daily = parse_gfz(r.text)
    kp_fc = None
    try:
        s = requests.get(SWPC_3DAY, headers=UA, timeout=60)
        s.raise_for_status()
        kp_fc = parse_swpc_3day(s.text)
    except Exception as ex:
        print("SWPC 3-day forecast unavailable:", ex)
    three.to_csv(C.STORE / "drivers_3h.csv")
    daily.to_csv(C.STORE / "drivers_daily.csv")
    if kp_fc is not None:
        kp_fc.rename("kp").to_csv(C.STORE / "swpc_kp_forecast.csv")
    return three, daily, kp_fc
