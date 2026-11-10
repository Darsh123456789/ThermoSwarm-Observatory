"""The sensor network: element sets -> quality-controlled orbit-averaged densities.

Element-set table columns: norad, jd (epoch), l1, l2, n (rev/day), ecc.
Interval table columns:    norad, jd1, jd2, t1, t2, tmid, dn, I, alt, alt_eff, f, ap_max, ap_max_lead, quiet
Window table columns:      norad, jd1, jd2, t0, a_km, alt, I, J, f72
"""
import numpy as np
import pandas as pd

from . import config as C
from .physics import drag_integrals


def semi_major_km(n_rev_day):
    n = np.asarray(n_rev_day, float) * 2 * np.pi / 86400
    return (C.MU / n ** 2) ** (1 / 3)


def quality_control(els):
    """Duplicate removal, orbit selection, manoeuvre rejection (paper, Section 4.1)."""
    e = els.drop_duplicates(["norad", "jd"]).sort_values(["norad", "jd"]).copy()
    e["a"] = semi_major_km(e.n)
    e["perigee"] = e.a * (1 - e.ecc) - C.RE_KM
    e["apogee"] = e.a * (1 + e.ecc) - C.RE_KM
    e = e[(e.ecc < C.ECC_MAX) & (e.perigee > C.PERIGEE_MIN_KM - 100) & (e.apogee < C.APOGEE_MAX_KM)]
    keep = []
    for norad, g in e.groupby("norad", sort=False):
        g = g.sort_values("jd")
        rise = g.a.diff() > 0.3                     # drag only lowers a: a rise is a manoeuvre
        if rise.sum() > 3:
            continue                                 # routinely manoeuvring object
        bad = np.zeros(len(g), bool)
        for t in g.jd.values[rise.values]:
            bad |= np.abs(g.jd.values - t) <= 2.0    # cut out +-2 days around each rise
        g = g[~bad]
        if len(g) >= 3:
            keep.append(g)
    return pd.concat(keep) if keep else e.iloc[:0]


def _ap_stats(three, t1, t2):
    w = three[(three.index >= t1 - pd.Timedelta("24h")) & (three.index <= t2)]
    core = w[w.index >= t1]
    return (float(core.ap.max()) if len(core) else np.nan,
            float(w.ap.max()) if len(w) else np.nan)


def make_intervals(els, drivers, existing=None):
    """Density intervals from TLE pairs PAIR_MIN..PAIR_MAX days apart. Skips pairs already in
    `existing` (keyed by norad, jd1, jd2)."""
    done = set()
    if existing is not None and len(existing):
        done = set(zip(existing.norad, existing.jd1.round(6), existing.jd2.round(6)))
    rows = []
    for norad, g in els.groupby("norad"):
        g = g.sort_values("jd").reset_index(drop=True)
        jd = g.jd.values
        i = 0
        while i < len(g) - 1:
            j = int(np.searchsorted(jd, jd[i] + C.PAIR_MIN_DAYS))
            if j >= len(g):
                break
            if jd[j] - jd[i] > C.PAIR_MAX_DAYS:
                i = j
                continue
            if (norad, round(jd[i], 6), round(jd[j], 6)) not in done:
                r = drag_integrals(g.l1[i], g.l2[i], jd[i], jd[j], drivers, C.NSAMP_INTERVAL)
                if r is not None:
                    dn = (g.n[j] - g.n[i]) * 2 * np.pi / 86400   # rad/s
                    rows.append(dict(norad=int(norad), jd1=jd[i], jd2=jd[j], dn=dn, **r))
            i = j
    iv = pd.DataFrame(rows)
    if iv.empty:
        return iv
    iv["t1"] = pd.to_datetime(iv.jd1 - 2440587.5, unit="D")
    iv["t2"] = pd.to_datetime(iv.jd2 - 2440587.5, unit="D")
    iv["tmid"] = iv.t1 + (iv.t2 - iv.t1) / 2
    iv["f"] = iv.dn / iv.I
    st = [_ap_stats(drivers.three, a, b) for a, b in zip(iv.t1, iv.t2)]
    iv["ap_max"] = [s[0] for s in st]
    iv["ap_max_lead"] = [s[1] for s in st]
    iv["quiet"] = iv.ap_max_lead <= C.QUIET_AP
    return iv


def make_windows(els, drivers, existing=None, starts=None):
    """72-hour verification windows: TLE pairs WINDOW_MIN..MAX hours apart, issue times >= 1 day
    apart. `starts` (DataFrame norad, jd1) forces windows to begin at the epochs forecasts were
    issued from, so every committed forecast can be verified."""
    done = set()
    if existing is not None and len(existing):
        done = set(zip(existing.norad, existing.jd1.round(6)))
    forced = {}
    if starts is not None and len(starts):
        for n, j in zip(starts.norad.astype(int), starts.jd1):
            forced.setdefault(n, set()).add(round(float(j), 6))
    rows = []
    for norad, g in els.groupby("norad"):
        g = g.sort_values("jd").reset_index(drop=True)
        jd, last = g.jd.values, -1e9
        want = forced.get(int(norad), set())
        for i in range(len(g)):
            if jd[i] < last + 1.0 and round(jd[i], 6) not in want:
                continue
            j = int(np.searchsorted(jd, jd[i] + C.WINDOW_MIN_H / 24))
            if j >= len(g) or jd[j] > jd[i] + C.WINDOW_MAX_H / 24:
                continue
            last = jd[i]
            if (norad, round(jd[i], 6)) in done:
                continue
            r = drag_integrals(g.l1[i], g.l2[i], jd[i], jd[j], drivers, C.NSAMP_WINDOW)
            if r is None:
                continue
            dn = (g.n[j] - g.n[i]) * 2 * np.pi / 86400
            rows.append(dict(norad=int(norad), jd1=jd[i], jd2=jd[j], a_km=r["a_km"], alt=r["alt"],
                             I=r["I"], J=r["J"], f72=dn / r["I"]))
    w = pd.DataFrame(rows)
    if len(w):
        w["t0"] = pd.to_datetime(w.jd1 - 2440587.5, unit="D")
    return w


def calibrate_causal(iv):
    """Causal quiet-time calibration: for each interval, B = median f of the same object's
    quiet intervals that ENDED before this one ended and within the previous CAL_WINDOW_DAYS
    (>= CAL_MIN_QUIET of them). R = f / B. sigR = 1.4826 MAD of those quiet ratios."""
    out = []
    for norad, g in iv.groupby("norad", sort=False):
        g = g.sort_values("jd2").copy()
        q = g[g.quiet & (g.f > 0)]
        qt, qf = q.jd2.to_numpy(), q.f.to_numpy()
        B = np.full(len(g), np.nan)
        S = np.full(len(g), np.nan)
        for k, t in enumerate(g.jd2.to_numpy()):
            m = (qt < t) & (qt >= t - C.CAL_WINDOW_DAYS)
            if m.sum() >= C.CAL_MIN_QUIET:
                b = np.median(qf[m])
                B[k] = b
                rq = qf[m] / b
                S[k] = 1.4826 * np.median(np.abs(rq - np.median(rq)))
        g["B"], g["sigR"] = B, S
        out.append(g)
    iv = pd.concat(out)
    iv["R"] = iv.f / iv.B
    ok = np.isfinite(iv.R) & (iv.B > 0) & (iv.sigR <= C.SIGR_MAX)
    return iv[ok]


def latest_state(iv_cal, t):
    """Per-object state known at time t: B, last R, 7-day median R, altitude, sigR."""
    known = iv_cal[iv_cal.t2 <= t]
    rows = []
    for norad, g in known.groupby("norad"):
        g = g.sort_values("t2")
        last = g.iloc[-1]
        if (t - last.t2) > pd.Timedelta("4D"):
            continue
        r7 = g[g.t2 > t - pd.Timedelta("7D")].R
        rows.append(dict(norad=norad, B=last.B, R_last=last.R, R_base7=float(r7.median()),
                         alt=last.alt, sigR=last.sigR, last_t2=last.t2))
    return pd.DataFrame(rows)
