"""Backtest: run the observatory day by day over the ThermoSwarm 2025-26 record.

Uses the published ThermoSwarm dataset (Zenodo 10.5281/zenodo.23674144): `intervals.parquet`
(density inversions) and `forecast_results.parquet` (72-h windows with issue-time state).
Strictly causal: on each day D the system only uses windows verified by the end of D,
retrains the boosting model weekly on those, and issues forecasts for windows starting on D.
Everything produced here is labelled BACKTEST and kept apart from the live registry.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .biasmap import daily_band_map
from .collision import simulate, storm_aware
from .conformal import AdaptiveConformal, regime
from .models import STATE_COLUMNS, registry
from .referee import Registry, leaderboard, score_pairs
from .sensors import calibrate_causal

KEEP = list(dict.fromkeys(["norad", "t0", "jd1", "jd2", "alt", "a_km", "B", "J", "f72", "R72"] + STATE_COLUMNS))


def load(data_dir):
    d = Path(data_dir)
    fr = pd.read_parquet(d / "forecast_results.parquet", columns=KEEP + ["stormtime"]).reset_index(drop=True)
    fr = fr[(fr.B > 0) & np.isfinite(fr.J) & (fr.J > 0)].copy()
    fr["R_base7"] = fr.R_base7.where((fr.R_base7 > 0.2) & (fr.R_base7 < 5), fr.R_last)
    fr["verified_at"] = pd.to_datetime(fr.jd2 - 2440587.5, unit="D")
    iv = pd.read_parquet(d / "intervals.parquet",
                         columns=["norad", "jd1", "jd2", "t1", "t2", "tmid", "alt", "f", "quiet"])
    return fr, iv


def band_terms(fc):
    """Per-band 72-h drag terms from one day's thermoswarm-gb forecasts."""
    g = fc[(fc.model == "thermoswarm-gb") & np.isfinite(fc.R_pred)].copy()
    if g.empty:
        return {}                     # intervals may be NaN while conformal is warming up
    scale = g.a_km * g.B * g.J
    g["bound72_km"] = scale * np.maximum(g.hi - g.R_pred, g.R_pred - g.lo)
    g["shift72_km"] = scale * (g.R_pred - 1.0)
    out = {}
    for b, h in g.groupby("band"):
        out[int(b)] = dict(bound72_km=float(h.bound72_km.median()), shift72_km=float(h.shift72_km.median()),
                           R_pred=float(h.R_pred.median()), lo=float(h.lo.median()), hi=float(h.hi.median()),
                           n=len(h))
    return out


def run(data_dir, out_dir=C.REPLAY, n_conj=400, verbose=True):
    t_start = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fr, iv = load(data_dir)
    storms = pd.read_csv(Path(data_dir) / "storms.csv", parse_dates=["onset", "end", "peak_time"])

    # ---- bias map from causally calibrated intervals
    ivc = calibrate_causal(iv)
    bmap = daily_band_map(ivc)
    bmap.to_csv(out_dir / "biasmap.csv", index=False)

    reg_dir = out_dir / "registry"
    if reg_dir.exists():
        for p in reg_dir.rglob("*"):
            if p.is_file():
                p.unlink()
    reg = Registry(reg_dir, stream="backtest")
    models = registry()
    conf = AdaptiveConformal()
    pending = []
    scored = []
    bulletins, coverage_log, collision_rows = [], [], []
    altitudes = ivc.groupby("norad").alt.median().to_numpy()
    conjs = simulate(n_conj, altitudes, seed=1)
    storm_days = set(storms.peak_time.dt.floor("D"))

    day0 = fr.t0.min().floor("D")
    day1 = fr.verified_at.max().floor("D")
    last_train = None
    for day in pd.date_range(day0, day1, freq="D"):
        end = day + pd.Timedelta("1D")

        # 1) verify forecasts whose window ended today; update conformal in time order
        if pending:
            P = pd.concat(pending, ignore_index=True)
            done = P[P.verified_at < end].sort_values("verified_at")
            pending = [P[P.verified_at >= end]] if (P.verified_at >= end).any() else []
            if len(done):
                s = score_pairs(done)
                scored.append(s)
                for mname, g in s.groupby("model", sort=False):
                    conf.update(mname, g.R_pred.values, g.R72.values, g.regime.values, g.lo.values, g.hi.values)

        # 2) weekly causal retraining on windows verified before today
        if last_train is None or (day - last_train).days >= 7:
            train = fr[fr.verified_at < day]
            for m in models:
                if m.trainable:
                    m.fit(train)
            last_train = day

        # 3) issue forecasts for windows starting today
        today = fr[(fr.t0 >= day) & (fr.t0 < end)]
        if len(today):
            rg = regime(today.kp_max24.values, today.h_since_storm.values)
            rows = []
            for m in models:
                rp = np.clip(m.predict(today), 0.2, 5)
                lo, hi, _ = conf.interval(m.name, rp, rg)
                rows.append(pd.DataFrame(dict(model=m.name, norad=today.norad.values, t0=today.t0.values,
                                              jd1=today.jd1.values, alt=today.alt.values,
                                              band=[C.band_of(a) for a in today.alt], regime=rg,
                                              B=today.B.values, a_km=today.a_km.values, R_pred=rp,
                                              lo=lo, hi=hi)))
            fc = pd.concat(rows, ignore_index=True)
            reg.commit(day, fc, created_utc=f"{day:%Y-%m-%d}T23:59:59 (backtest)")
            fc = fc.merge(today[["norad", "jd1", "jd2", "f72", "J", "verified_at"]], on=["norad", "jd1"])
            pending.append(fc)

            # bulletin for today
            bt = band_terms(fc)
            for b, v in bt.items():
                bulletins.append(dict(day=day, band=b, **v, regime_storm_frac=float((rg == "storm").mean())))

            # 4) collision impact on storm days and weekly
            if bt and (day in storm_days or day.dayofweek == 0):
                for c in conjs:
                    r = storm_aware(c, bt, n_scan=9)
                    collision_rows.append(dict(day=day, id=c.ids[0], alt=c.alt1, lead_h=c.lead_h,
                                               miss_m=c.miss_m(), storm_day=day in storm_days, **r))

        # coverage log
        if scored:
            last = scored[-1]
            for mname, g in last.groupby("model"):
                ok = np.isfinite(g.interval_score)
                if ok.any():
                    coverage_log.append(dict(day=day, model=mname, n=int(ok.sum()),
                                             covered=float(g.covered[ok].mean()),
                                             alpha_storm=conf._s(mname, "storm")["alpha_t"],
                                             alpha_quiet=conf._s(mname, "quiet")["alpha_t"]))
        if verbose and day.day == 1:
            print(f"{day:%Y-%m}  scored={sum(len(s) for s in scored):,}  t={time.time() - t_start:.0f}s", flush=True)

    S = pd.concat(scored, ignore_index=True)
    S["k"] = S.norad.astype(int).astype(str) + "_" + S.jd1.round(5).astype(str)
    S.to_csv(out_dir / "scores.csv.gz", index=False, compression="gzip")
    pd.DataFrame(bulletins).to_csv(out_dir / "bulletins.csv", index=False)
    pd.DataFrame(coverage_log).to_csv(out_dir / "coverage.csv", index=False)
    pd.DataFrame(collision_rows).to_csv(out_dir / "collision.csv", index=False)
    (out_dir / "conformal_state.json").write_text(conf.to_json())
    for m in models:
        if m.trainable:
            m.save(out_dir / f"{m.name}.json")

    lb = {"all": leaderboard(S).to_dict("records"),
          "storm": leaderboard(S, regime="storm").to_dict("records"),
          "quiet": leaderboard(S, regime="quiet").to_dict("records")}
    summary = dict(stream="backtest", period=[str(day0.date()), str(day1.date())], scored=len(S),
                   windows=int(S.k.nunique()), objects=int(S.norad.nunique()),
                   registry=reg.verify()[1], runtime_s=round(time.time() - t_start), leaderboard=lb)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


if __name__ == "__main__":
    import sys
    print(json.dumps(run(sys.argv[1]), indent=1, default=float)[:3000])
