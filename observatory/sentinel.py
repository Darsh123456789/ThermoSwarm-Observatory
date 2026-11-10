"""Sentinel: the daily whole-catalogue layer of the Observatory.

  catalogue -> 72-h all-vs-all screening -> storm-aware Pc (empirical element-set covariance +
  today's drag interval) -> avoidance options -> anomaly alerts -> map data and feed.

Everything published is derived (event summaries, alerts) except the map catalogue, which is
published only when it came from a CC0 / public source (TraCSS Open Data, CelesTrak).
"""
import json
import time

import numpy as np
import pandas as pd

from . import config as C
from .anomalies import (ReentryWindows, detect_fragmentation, detect_manoeuvres, drag_signature_changes,
                        predict_reentries, track_gaps, breakup_time, launch_of)
from .avoid import plan
from .catalog import HIST, export_for_map, fetch_catalog, update_history
from .collision import Conjunction, drag_terms
from .errmodel import ErrorModel
from .screening import assess, hbr_m, screen

OUT = C.DOCS / "feed" / "sentinel.json"
MAP = C.ROOT / "map" / "catalog.json"     # published to the map-data branch, not committed to main
IDS = C.STORE / "catalog_ids.txt"
SUMMARY = C.STORE / "sentinel_daily.csv"
HOURS = 72.0
MAX_PLANS = 40


def _ts(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x) else str(pd.Timestamp(x))[:19]


def run(now, drivers=None, band_terms=None, iv_cal=None, bandmap=None, regime="quiet", cat=None,
        source=None, hours=HOURS, hist=None):
    t_start = time.time()
    now = pd.Timestamp(now)
    log = {}
    if cat is None:
        source, cat = fetch_catalog()
    log.update(source=source, objects=len(cat))
    if hist is None:
        hist = update_history(cat, keep_days=45) if source else cat
    published_map = export_for_map(cat, source, MAP)
    log["map_published"] = bool(published_map)

    # ---- screening
    t0 = time.time()
    ev, n_screened = screen(cat, now, hours=hours)
    log.update(screened=n_screened, screen_seconds=round(time.time() - t0, 1), approaches_5km=len(ev),
               coorbiting=len(ev.attrs.get("coorbiting", [])))
    em = ErrorModel.load()
    ev = assess(ev, em, band_terms or {}, regime=regime, now=now) if len(ev) else ev

    # ---- avoidance options for the riskiest encounters involving a payload
    plans = []
    if len(ev):
        risky = ev[(ev.cat_nominal != "low") | (ev.cat_max != "low") | (ev.cat_storm != "low")].copy()
        risky = risky.sort_values("pc_max", ascending=False).head(MAX_PLANS)
        for e in risky.itertuples():
            movable = [k for k, t in ((1, e.type_i), (2, e.type_j)) if str(t).upper() == "PAYLOAD"]
            if not movable:
                plans.append(dict(i=e.i, j=e.j, tca=_ts(e.tca), note="no manoeuvrable object (debris / rocket body)"))
                continue
            tca_jd = pd.Timestamp(e.tca).to_julian_date()
            c1 = em.cov_rtn_m2(e.alt_km, tca_jd - e.epoch_i, regime)
            c2 = em.cov_rtn_m2(e.alt_km, tca_jd - e.epoch_j, regime)
            r1, v1, r2, v2 = (np.array(x) * 1e3 for x in (e.r1, e.v1, e.r2, e.v2))
            if movable[0] == 2:                         # manoeuvring object goes first
                r1, v1, r2, v2, c1, c2 = r2, v2, r1, v1, c2, c1
            conj = Conjunction(r1, v1, c1, r2, v2, c2, hbr_m(e.rcs_i, e.rcs_j), lead_h=e.lead_h,
                               alt1=e.alt_km, alt2=e.alt_km)
            s1, sh1, _ = drag_terms(e.alt_km, e.lead_h, band_terms or {})
            p = plan(conj, (s1 ** 2, sh1), (s1 ** 2, sh1))
            rec = p["recommended"]
            plans.append(dict(i=e.i, j=e.j, tca=_ts(e.tca), manoeuvring=int(e.i if movable[0] == 1 else e.j),
                              pc_before=p["pc_before"], target=p["target"],
                              recommended=rec, options=p["options"][:6]))
    log["plans"] = len(plans)

    # ---- anomalies
    alerts = dict(manoeuvres=[], fragmentations=[], reentries=[], drag_changes=[], track_gaps=[])
    try:
        recent = hist[hist.jd >= now.to_julian_date() - 45]
        man, _ = detect_manoeuvres(recent)
        if len(man):
            man = man[man.t >= now - pd.Timedelta("3D")].merge(cat[["norad", "name", "type"]], on="norad", how="left")
            alerts["manoeuvres"] = man.sort_values("t", ascending=False).head(200).assign(
                t=lambda d: d.t.astype(str).str[:19]).to_dict("records")
    except Exception as ex:
        log["manoeuvre_error"] = str(ex)[:200]
    try:
        known = set(int(x) for x in IDS.read_text().split()) if IDS.exists() else set()
        if known:
            fr = detect_fragmentation(cat, known, now)
            for f in fr.to_dict("records") if len(fr) else []:
                par = cat[cat.norad == f["parent"]]
                pcs = cat[cat.norad.isin(f["piece_ids"][:40])]
                if len(par) and len(pcs):
                    bt = breakup_time((par.l1.iloc[0], par.l2.iloc[0]), list(zip(pcs.l1, pcs.l2)), now, days=10)
                    f.update(t_breakup=_ts(bt["t_breakup"]), median_sep_km=bt["median_sep_km"],
                             ejection_ms=bt["ejection_ms"])
                alerts["fragmentations"].append(f)
        IDS.write_text("\n".join(str(x) for x in sorted(set(cat.norad.astype(int)) | known)))
    except Exception as ex:
        log["fragmentation_error"] = str(ex)[:200]
    try:
        wpath = C.STORE / "reentry_windows.json"
        w = ReentryWindows.from_json(wpath.read_text()) if wpath.exists() else ReentryWindows()
        re_ = predict_reentries(hist, now, w)
        if len(re_):
            re_ = re_.merge(cat[["norad", "name", "type"]], on="norad", how="left")
            for c in ("predicted", "window_lo", "window_hi"):
                re_[c] = re_[c].map(_ts)
            alerts["reentries"] = re_.head(100).to_dict("records")
    except Exception as ex:
        log["reentry_error"] = str(ex)[:200]
    try:
        dc = drag_signature_changes(iv_cal, bandmap)
        if len(dc):
            alerts["drag_changes"] = dc.assign(since=dc.since.astype(str).str[:19]).to_dict("records")
    except Exception as ex:
        log["drag_change_error"] = str(ex)[:200]
    alerts["track_gaps"] = track_gaps(cat, now).head(100).to_dict("records")
    log["alerts"] = {k: len(v) for k, v in alerts.items()}

    # ---- publish
    keep = ["i", "j", "name_i", "name_j", "type_i", "type_j", "tca", "miss_km", "vrel_kms", "dR_km", "dT_km",
            "dN_km", "alt_km", "hbr_m", "sigT1_km", "sigT2_km", "pc_nominal", "pc_storm", "pc_max",
            "cat_nominal", "cat_storm", "cat_max", "escalated", "diluted", "lead_h"]
    events = []
    if len(ev):
        top = ev.sort_values(["pc_max", "miss_km"], ascending=[False, True]).head(500)
        for e in top.to_dict("records"):
            rec = {k: e.get(k) for k in keep}
            rec["tca"] = _ts(rec["tca"])
            if source in ("tracss", "celestrak", None):     # states only from republishable sources
                rec["r1"], rec["r2"] = [round(x, 3) for x in e["r1"]], [round(x, 3) for x in e["r2"]]
            events.append(rec)
    counts = dict(screened=n_screened, approaches_5km=int(len(ev)),
                  approaches_1km=int((ev.miss_km < 1).sum()) if len(ev) else 0,
                  high=int((ev.cat_max == "high").sum()) if len(ev) else 0,
                  medium=int((ev.cat_max == "medium").sum()) if len(ev) else 0,
                  riskier=int(ev.escalated.sum()) if len(ev) else 0,
                  diluted=int(ev.diluted.sum()) if len(ev) else 0)
    out = dict(generated=str(now)[:19], horizon_h=hours, source=source, regime=regime, counts=counts,
               events=events, coorbiting=ev.attrs.get("coorbiting", [])[:100] if len(ev) else [],
               plans=plans, alerts=alerts, log=log,
               notes=["Pc uses an empirical element-set error model calibrated on a year of public element sets "
                      "plus today's measured storm-drag interval; it is not an operational CDM.",
                      "Risk categories: high >= 1e-4, medium >= 1e-6 (illustrative thresholds).",
                      "Avoidance options assume an impulsive burn and linearised (Hill) relative motion; they are "
                      "decision aids, not commands, and ignore secondary conjunctions created by the burn.",
                      "With public element-set uncertainty (often several km along-track after a day), the burn needed "
                      "to reach the target Pc can be large; operationally this is the point at which an operator "
                      "requests better tracking or its own ephemeris before deciding."])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(C.clean_json(out), default=str, allow_nan=False, separators=(",", ":")))
    row = dict(day=str(now.date()), **counts, **{f"alerts_{k}": v for k, v in log["alerts"].items()},
               source=source, seconds=round(time.time() - t_start))
    old = pd.read_csv(SUMMARY) if SUMMARY.exists() else pd.DataFrame()
    pd.concat([old[old.day != row["day"]] if len(old) else old, pd.DataFrame([row])]).to_csv(SUMMARY, index=False)
    log["seconds"] = round(time.time() - t_start)
    return out
