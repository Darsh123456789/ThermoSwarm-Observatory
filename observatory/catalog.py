"""Whole-catalogue ingest for screening, anomaly detection and the 3D map.

Sources, tried in order:
  tracss     TraCSS public TODD API (no account). The DoD public element-set catalogue and the
             satellite catalogue are TraCSS Open Data released under CC0 1.0, so they may be
             republished; the 3D map is built only from this source (or CelesTrak).
  spacetrack full 'gp' class (needs SPACETRACK_USER/PASS). Used for screening if TraCSS is down,
             but never republished (Space-Track user agreement).
  celestrak  public GP groups (partial catalogue) as a last resort.

Rows: norad, name, intl, type (PAYLOAD / ROCKET BODY / DEBRIS / UNKNOWN), rcs (SMALL/MEDIUM/
LARGE/''), jd (epoch), l1, l2, n (rev/day), ecc, incl, perigee_km, apogee_km, source.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import requests

from . import config as C

UA = {"User-Agent": "ThermoSwarm-Observatory/0.2 (+https://github.com/Darsh123456789/ThermoSwarm-Observatory)"}
TODD = os.environ.get("TRACSS_TODD", "https://api.tracss.gov/todd")
OMM_URL = os.environ.get("TRACSS_OMM_URL") or f"{TODD}/omm/v1/stream?correlationId=most_recent"
CAT_URL = f"{TODD}/tracsscat/v1/stream"
REPUBLISHABLE = {"tracss", "celestrak"}


def _stream(url, timeout=600):
    """Newline-delimited JSON (TODD streams) or a JSON array."""
    r = requests.get(url, headers=UA, timeout=timeout, stream=True)
    r.raise_for_status()
    text = r.content.decode("utf-8", "replace").strip()
    if text.startswith("["):
        return json.loads(text)
    out = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if line and line not in ("[", "]"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def parse_omm(records, source):
    """OMM / GP JSON (optionally wrapped as {"omm": {...}}) -> catalogue rows."""
    from sgp4.api import Satrec, WGS72
    from sgp4 import omm as omm_mod, exporter
    rows = []
    for rec in records:
        r = rec.get("omm", rec) if isinstance(rec, dict) else None
        if not r:
            continue
        try:
            norad = int(r["NORAD_CAT_ID"])
            l1 = r.get("TLE_LINE1") or r.get("USER_DEFINED_TLE_LINE_1")
            l2 = r.get("TLE_LINE2") or r.get("USER_DEFINED_TLE_LINE_2")
            if not l1 or not l2:
                sat = Satrec()
                omm_mod.initialize(sat, r)
                l1, l2 = exporter.export_tle(sat)
            sat = Satrec.twoline2rv(l1, l2, WGS72)
            if sat.error:
                continue
            n = sat.no_kozai * 1440 / (2 * np.pi)
            a = (C.MU / (n * 2 * np.pi / 86400) ** 2) ** (1 / 3)
            rows.append(dict(norad=norad, name=str(r.get("OBJECT_NAME", "")).strip(),
                             intl=str(r.get("OBJECT_ID", "")).strip(),
                             type=str(r.get("OBJECT_TYPE", "")).strip().upper(),
                             rcs=str(r.get("RCS_SIZE", "") or "").strip().upper(),
                             jd=sat.jdsatepoch + sat.jdsatepochF, l1=l1, l2=l2, n=n, ecc=sat.ecco,
                             incl=np.degrees(sat.inclo), perigee_km=a * (1 - sat.ecco) - C.RE_KM,
                             apogee_km=a * (1 + sat.ecco) - C.RE_KM, source=source))
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("jd").drop_duplicates("norad", keep="last").reset_index(drop=True)
    return df


def satcat_tracss():
    """TraCSS CAT -> norad, name, intl, type, rcs, decay (date or NaT)."""
    recs = _stream(CAT_URL, timeout=300)
    rows = []
    for r in recs:
        g = lambda *k: next((r[x] for x in k if x in r and r[x] not in (None, "")), "")
        try:
            rows.append(dict(norad=int(g("noradId", "NORAD_CAT_ID")), name=str(g("satelliteName", "OBJECT_NAME")),
                             intl=str(g("internationalDesignator", "OBJECT_ID")),
                             type=str(g("objectType", "OBJECT_TYPE")).upper(),
                             rcs=str(g("rcsSize", "RCS_SIZE")).upper(),
                             decay=pd.to_datetime(g("decayDate", "DECAY_DATE") or None, errors="coerce")))
        except (TypeError, ValueError):
            continue
    return pd.DataFrame(rows)


def fetch_tracss():
    cat = parse_omm(_stream(OMM_URL), "tracss")
    try:
        sc = satcat_tracss()
        if len(sc) and len(cat):
            sc = sc.drop_duplicates("norad").set_index("norad")
            for col in ("type", "rcs", "intl"):
                fill = cat.norad.map(sc[col])
                cat[col] = np.where(cat[col].astype(str).str.len() > 0, cat[col], fill.fillna(""))
    except Exception:
        pass
    return cat


def fetch_spacetrack_full():
    user, pw = os.environ.get("SPACETRACK_USER"), os.environ.get("SPACETRACK_PASS")
    if not user or not pw:
        return None
    s = requests.Session()
    s.headers.update(UA)
    s.post("https://www.space-track.org/ajaxauth/login", data={"identity": user, "password": pw},
           timeout=60).raise_for_status()
    url = ("https://www.space-track.org/basicspacedata/query/class/gp/decay_date/null-val/"
           "epoch/>now-10/orderby/norad_cat_id/format/json")
    r = s.get(url, timeout=600)
    r.raise_for_status()
    s.get("https://www.space-track.org/ajaxauth/logout", timeout=30)
    return parse_omm(r.json(), "spacetrack")


def fetch_celestrak():
    groups = ["active", "last-30-days", "analyst", "cosmos-1408-debris", "fengyun-1c-debris",
              "iridium-33-debris", "cosmos-2251-debris"]
    recs = []
    for g in groups:
        try:
            r = requests.get(f"https://celestrak.org/NORAD/elements/gp.php?GROUP={g}&FORMAT=json",
                             headers=UA, timeout=180)
            if r.ok:
                recs.extend(r.json())
        except Exception:
            pass
        time.sleep(2)
    return parse_omm(recs, "celestrak")


def fetch_catalog():
    """(source, catalogue) from the first provider that works."""
    errors = []
    for name, fn in (("tracss", fetch_tracss), ("spacetrack", fetch_spacetrack_full),
                     ("celestrak", fetch_celestrak)):
        try:
            cat = fn()
            if cat is not None and len(cat) > 1000:
                return name, cat
            errors.append(f"{name}: {0 if cat is None else len(cat)} objects")
        except Exception as ex:
            errors.append(f"{name}: {ex}")
    raise RuntimeError("no catalogue provider worked: " + "; ".join(errors))


# ---------------------------------------------------------------- history (for anomaly detection)
HIST = C.CACHE / "catalog_history.parquet"


def update_history(cat, keep_days=21):
    """Element history of the whole catalogue, kept in .cache only (raw elements, never committed)."""
    cols = ["norad", "jd", "l1", "l2", "n", "ecc", "incl", "perigee_km", "apogee_km", "source"]
    old = pd.read_parquet(HIST) if HIST.exists() else pd.DataFrame(columns=cols)
    h = pd.concat([old, cat[cols]], ignore_index=True).drop_duplicates(["norad", "jd"])
    h = h[h.jd >= h.jd.max() - keep_days].reset_index(drop=True)
    h.to_parquet(HIST)
    return h


def export_for_map(cat, source, path):
    """Compact catalogue for the 3D map: only from republishable (CC0 / public) sources."""
    if source not in REPUBLISHABLE:
        return False
    keep = cat[cat.perigee_km < 50000]
    tcode = {"PAYLOAD": "P", "ROCKET BODY": "R", "DEBRIS": "D"}
    rows = [[int(r.norad), r.name[:24], tcode.get(r.type, "U"), r.l1, r.l2] for r in keep.itertuples()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(source=source, licence="CC0-1.0 (TraCSS Open Data)" if source == "tracss"
                                    else "CelesTrak public GP data", generated=pd.Timestamp.now("UTC").isoformat(),
                                    n=len(rows), objects=rows), separators=(",", ":")))
    return True
