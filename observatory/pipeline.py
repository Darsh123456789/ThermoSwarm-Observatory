"""Live daily run (GitHub Actions): collect -> invert -> calibrate -> verify -> forecast ->
commit to the registry -> bulletin -> collision impact -> website.

    python -m observatory.pipeline            # normal daily run
    python -m observatory.pipeline --verify   # only check the registry hash chain
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .biasmap import daily_band_map, nowcast
from .collision import classify, read_cdm, simulate, storm_aware
from .conformal import AdaptiveConformal, regime
from .drivers import Drivers, extend_with_forecast
from .ingest import fetch_drivers, fetch_elements, update_cache
from .models import ThermoSwarmGB, load_submissions, registry
from .referee import Registry, leaderboard
from .replay import band_terms
from .sensors import calibrate_causal, latest_state, make_intervals, make_windows, quality_control

IV = C.STORE / "intervals.parquet"
WIN = C.STORE / "windows.parquet"
ISSUED = C.STORE / "issued_state.parquet"
CONF = C.STORE / "conformal_state.json"
GB = C.STORE / "thermoswarm-gb.json"
WARM = C.STORE / "warm_start"     # state learned in the full-year backtest; used only until live state exists
BUL = C.STORE / "bulletins.csv"


def _read(p):
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def _csv(p):
    try:
        return pd.read_csv(p)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def storm_flags(bt_today, bmap_now, kp_fc, now):
    """Per-band storm-risk flag from the paper's finding that storm-time error follows the
    pre-storm baseline (r = 0.50): a forecast storm on top of a high baseline is the worst case."""
    kp_max = float(kp_fc[(kp_fc.index > now) & (kp_fc.index <= now + pd.Timedelta("72h"))].max()) \
        if kp_fc is not None and len(kp_fc) else np.nan
    storm = np.isfinite(kp_max) and kp_max >= 14 / 3
    flags = {}
    for _, r in bmap_now.iterrows():
        base = r.baseline if np.isfinite(r.baseline) else r.R
        if storm and base >= 1.05:
            f = "HIGH: storm forecast and the model is already too low here; expect more drag than predicted"
        elif storm and base <= 0.90:
            f = "MIXED: storm forecast, but the model is too high here; storm and bias partly cancel"
        elif storm:
            f = "ELEVATED: storm forecast"
        elif base >= 1.10:
            f = "WATCH: model running >10% low; any storm will add drag beyond prediction"
        else:
            f = "NORMAL"
        flags[int(r.band)] = f
    return flags, kp_max


def run(now=None):
    now = pd.Timestamp(now or pd.Timestamp.now("UTC").tz_localize(None))
    day = now.floor("D")
    log = {"run_utc": str(now)}

    # 1) collect
    network = pd.read_csv(C.STORE / "network.csv").norad.astype(int).tolist()
    provider, new = fetch_elements(network)
    els = update_cache(new)
    three, daily, kp_fc = fetch_drivers()
    drv = Drivers(three, daily)
    log.update(provider=provider, element_sets=len(new), objects=int(new.norad.nunique()))

    # 2) invert (new pairs only) and store derived data
    qc = quality_control(els)
    iv_old, win_old = _read(IV), _read(WIN)
    iv_new = make_intervals(qc, drv, iv_old)
    issued_prev = _read(ISSUED)
    win_new = make_windows(qc, drv, win_old, starts=issued_prev[["norad", "jd1"]] if len(issued_prev) else None)
    iv = pd.concat([iv_old, iv_new], ignore_index=True) if len(iv_new) else iv_old
    win = pd.concat([win_old, win_new], ignore_index=True) if len(win_new) else win_old
    if len(iv):
        iv.to_parquet(IV)
    if len(win):
        win.to_parquet(WIN)
    log.update(new_intervals=len(iv_new), new_windows=len(win_new))

    ivc = calibrate_causal(iv[iv.t2 >= now - pd.Timedelta("90D")]) if len(iv) else iv
    bmap = daily_band_map(ivc) if len(ivc) else pd.DataFrame()
    if len(bmap):
        bmap.to_csv(C.STORE / "biasmap.csv", index=False)

    # 3) verify earlier forecasts, update conformal calibration
    reg = Registry(C.REGISTRY, stream="live")
    src = CONF if CONF.exists() else WARM / CONF.name
    conf = AdaptiveConformal.from_json(src.read_text()) if src.exists() else AdaptiveConformal()
    fresh = reg.score(win, now) if len(win) else pd.DataFrame()
    if len(fresh):
        fresh = fresh.sort_values("verified_at")
        for mname, g in fresh.groupby("model", sort=False):
            conf.update(mname, g.R_pred.values, g.R72.values, g.regime.values, g.lo.values, g.hi.values)
    log["verified"] = len(fresh)

    # 4) weekly retraining of the boosting model on verified live windows (+ backtest warm start)
    models = registry()
    gb = next(m for m in models if isinstance(m, ThermoSwarmGB))
    issued = _read(ISSUED)
    if GB.exists():
        gb.load(GB)
    elif (WARM / GB.name).exists():
        gb.load(WARM / GB.name)
    if len(issued) and len(win) and day.dayofweek == 0:
        tr = issued.merge(win[["norad", "jd1", "f72"]], on=["norad", "jd1"])
        tr["R72"] = tr.f72 / tr.B
        if len(tr) >= 2000:
            gb.fit(tr)
            gb.save(GB)
            log["retrained"] = len(tr)

    # 5) issue forecasts for each object's latest element set (only if still fresh)
    latest = qc.sort_values("jd").groupby("norad").tail(1)
    latest = latest[pd.to_datetime(latest.jd - 2440587.5, unit="D") > now - pd.Timedelta("24h")]
    rows = []
    for _, e in latest.iterrows():
        t0 = pd.Timestamp(pd.to_datetime(e.jd - 2440587.5, unit="D"))
        st = latest_state(ivc[ivc.norad == e.norad], t0)
        if st.empty:
            continue
        r = st.iloc[0].to_dict()
        r.update(drv.features(t0))
        r.update(norad=int(e.norad), t0=t0, jd1=float(e.jd), a_km=float(C.MU / (e.n * 2 * np.pi / 86400) ** 2) ** (1 / 3))
        rows.append(r)
    state = pd.DataFrame(rows)
    committed = None
    if len(state):
        rg = regime(state.kp_max24.values, state.h_since_storm.values)
        out = []
        for m in models + load_submissions(C.ROOT / "submissions", day):
            rp = np.clip(m.predict(state), 0.2, 5)
            lo, hi, _ = conf.interval(m.name, rp, rg)
            out.append(pd.DataFrame(dict(model=m.name, norad=state.norad, t0=state.t0, jd1=state.jd1,
                                         alt=state.alt, band=[C.band_of(a) for a in state.alt], regime=rg,
                                         B=state.B, a_km=state.a_km, R_pred=rp, lo=lo, hi=hi)))
        fc = pd.concat(out, ignore_index=True)
        fc = fc[np.isfinite(fc.R_pred)]
        try:
            committed = reg.commit(day, fc)
        except FileExistsError:
            committed = {"note": "already committed today"}
        state.assign(issued_utc=now).pipe(lambda s: pd.concat([issued, s], ignore_index=True)
                                          .drop_duplicates(["norad", "jd1"])).to_parquet(ISSUED)
        log["issued"] = len(state)
    CONF.write_text(conf.to_json())

    # 6) bulletin
    bulletin, bt = {}, {}
    if len(state):
        J72 = (win.groupby(win.alt.map(C.band_of)).J.median().to_dict() if len(win) else {})
        fc_b = fc.assign(J=fc.band.map(J72).fillna(0.65))
        bt = band_terms(fc_b)
        hist = _csv(BUL)
        bn = nowcast(bmap, day) if len(bmap) else pd.DataFrame()
        flags, kp_max = storm_flags(bt, bn, kp_fc, now)
        recs = []
        for b, v in bt.items():
            past = hist[(hist.band == b)].tail(60).bound72_km if len(hist) else pd.Series(dtype=float)
            mult = v["bound72_km"] / past.median() if len(past) >= 10 and past.median() > 0 else np.nan
            row = bn[bn.band == b]
            recs.append(dict(day=str(day.date()), band=b, label=C.band_label(b),
                             nowcast_R=float(row.R.iloc[0]) if len(row) else np.nan,
                             baseline=float(row.baseline.iloc[0]) if len(row) else np.nan,
                             multiplier=mult, flag=flags.get(b, "NORMAL"), **v))
        if recs:
            pd.concat([hist, pd.DataFrame(recs)], ignore_index=True).to_csv(BUL, index=False)
        bulletin = dict(day=str(day.date()), kp_max_72h_forecast=kp_max, bands=recs)

        # 7) collision impact: operator-supplied CDMs if present, otherwise a simulated set
        cdms = sorted((C.ROOT / "conjunctions").glob("*.cdm")) if (C.ROOT / "conjunctions").exists() else []
        conjs = [read_cdm(p.read_text()) for p in cdms]
        source = "cdm" if conjs else "simulated"
        if not conjs:
            conjs = simulate(300, ivc.groupby("norad").alt.median().to_numpy(), seed=int(day.strftime("%Y%m%d")))
        res = classify(pd.DataFrame([dict(id=c.ids[0], alt=c.alt1, lead_h=c.lead_h, **storm_aware(c, bt, 9))
                                     for c in conjs]))
        bulletin["collision"] = dict(source=source, n=len(res),
                                     category_changed=int(res.storm_sensitive.sum()),
                                     riskier=int(res.escalated.sum()),     # higher category inside the drag interval
                                     diluted=int(res.diluted.sum()))       # lower category: too uncertain to reassure

    # 8) Sentinel: whole-catalogue screening, avoidance options and anomaly alerts
    sentinel_summary = None
    if os.environ.get("SENTINEL", "1") != "0":
        try:
            from .sentinel import run as sentinel_run
            f = drv.features(now)
            reg_now = str(regime([f.get("kp_max24", 0)], [f.get("h_since_storm", 240)])[0])
            snt = sentinel_run(now, drv, bt, ivc, bmap, regime=reg_now)
            sentinel_summary = dict(counts=snt["counts"], alerts=snt["log"].get("alerts"), log=snt["log"])
        except Exception as ex:                       # never let Sentinel break the core forecasts
            sentinel_summary = dict(error=f"{type(ex).__name__}: {ex}"[:500])
    log["sentinel"] = sentinel_summary

    lb = leaderboard(reg.scores()) if len(reg.scores()) else pd.DataFrame()
    status = dict(log=log, committed=committed, registry=reg.verify()[1], bulletin=bulletin,
                  leaderboard=lb.to_dict("records"))
    (C.DOCS / "feed" / "latest.json").write_text(json.dumps(C.clean_json(status), indent=1, default=str, allow_nan=False))
    from .site import build
    build()
    return status


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        ok, msg = Registry(C.REGISTRY).verify()
        print(msg)
        raise SystemExit(0 if ok else 1)
    print(json.dumps(run(), indent=1, default=str)[:4000])
