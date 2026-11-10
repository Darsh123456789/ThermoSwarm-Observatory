"""Altitude-resolved model-bias map: daily median density ratio R per 50-km band.

Nowcast for day D uses intervals whose second element set arrived by the end of D
(t2 <= D + 1 day), assigned to the day of their midpoint. The 'baseline' is the causal
15-day trailing median of the daily ratio: the slow error component (paper, Section 6.2).
"""
import numpy as np
import pandas as pd

from . import config as C


def _boot_ci(x, rng, n=200):
    if len(x) < 5:
        return np.nan, np.nan
    m = np.median(rng.choice(x, (n, len(x))), axis=1)
    return float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))


def daily_band_map(iv_cal, start=None, end=None, seed=0):
    rng = np.random.default_rng(seed)
    d = iv_cal.copy()
    d["day"] = d.tmid.dt.floor("D")
    d["band"] = [C.band_of(a) for a in d.alt]
    d = d[d.band >= 0]
    if start is not None:
        d = d[d.day >= pd.Timestamp(start)]
    if end is not None:
        d = d[d.day <= pd.Timestamp(end)]
    rows = []
    for (day, b), g in d.groupby(["day", "band"]):
        lo, hi = _boot_ci(g.R.to_numpy(), rng)
        rows.append(dict(day=day, band=int(b), R=float(g.R.median()), lo=lo, hi=hi,
                         n=len(g), objects=int(g.norad.nunique())))
    m = pd.DataFrame(rows)
    if m.empty:
        return m
    m = m.sort_values(["band", "day"])
    m["baseline"] = (m.groupby("band").R
                     .transform(lambda s: s.rolling(15, min_periods=5).median()))
    return m.reset_index(drop=True)


def nowcast(bandmap, day):
    """Bias map row per band for one day (latest available for each band, at most 2 days old)."""
    day = pd.Timestamp(day).floor("D")
    m = bandmap[(bandmap.day <= day) & (bandmap.day >= day - pd.Timedelta("2D"))]
    return m.sort_values("day").groupby("band").tail(1).sort_values("band")
