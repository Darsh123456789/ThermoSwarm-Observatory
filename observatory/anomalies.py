"""Anomaly detection on the catalogue: what changed in orbit that nobody announced.

1. Storm-aware manoeuvre detection. Every object's orbital decay is compared with what the
   *swarm* says the atmosphere did that day. The swarm decay index S_b(t) is the median, over
   all objects in a 50-km band, of each object's decay rate divided by its own trailing 30-day
   median rate, on a 6-hour grid. Expected change in semi-major axis = (object's baseline rate) x S_b x dt.
   A rise beyond the expected decay, a drop several times larger than it, or an inclination
   jump, is flagged as a manoeuvre. A naive detector (no swarm index) mistakes storm-time decay
   for lowering burns; the comparison is part of the validation.
2. Fragmentation (break-up) detection. New catalogue entries carrying the international
   designator of an old launch are pieces of an existing object. Three or more new pieces of the
   same launch within 7 days raise a break-up alert; the break-up time is estimated by
   propagating parent and pieces backwards and finding when they were closest together.
3. Re-entry prediction with calibrated windows. The decay rate is integrated down to 120 km
   through an NRLMSIS-shaped density profile; the window is set by conformal quantiles of the
   relative error measured on past re-entries, so a "90% window" means what it says.
4. Drag-signature changes. A network object whose density ratio departs from its altitude
   band for 3+ consecutive intervals (attitude change, tumbling, deployed drag device, mis-tag).
5. Track gaps. LEO objects whose newest element set is more than 3 days old.
"""
import json

import numpy as np
import pandas as pd

from . import config as C

BAND_W = 50.0


def _band(alt):
    return int(alt // BAND_W)


# ---------------------------------------------------------------- 1. manoeuvres
def decay_rates(hist, min_dt=0.4, max_dt=4.0):
    """Consecutive element-set pairs -> rate of change of semi-major axis (km/day)."""
    h = hist.sort_values(["norad", "jd"]).copy()
    if "a_km" not in h:
        h["a_km"] = (C.MU / (h.n * 2 * np.pi / 86400) ** 2) ** (1 / 3)
    rows = []
    for norad, g in h.groupby("norad", sort=False):
        jd, a, inc = g.jd.to_numpy(), g.a_km.to_numpy(), g.incl.to_numpy()
        i = 0
        while i < len(g) - 1:
            j = int(np.searchsorted(jd, jd[i] + min_dt))
            if j >= len(g):
                break
            dt = jd[j] - jd[i]
            if dt <= max_dt:
                rows.append((int(norad), jd[i], jd[j], a[i], a[j], inc[i], inc[j]))
            i = j
    r = pd.DataFrame(rows, columns=["norad", "jd1", "jd2", "a1", "a2", "i1", "i2"])
    r["dt"] = r.jd2 - r.jd1
    r["da"] = r.a2 - r.a1
    r["rate"] = r.da / r.dt
    r["di"] = r.i2 - r.i1
    r["alt"] = 0.5 * (r.a1 + r.a2) - C.RE_KM
    r["band"] = (r.alt // BAND_W).astype(int)
    mid = pd.to_datetime(0.5 * (r.jd1 + r.jd2) - 2440587.5, unit="D")
    r["day"] = mid.dt.floor("D")
    r["slot"] = mid.dt.floor("6h")            # swarm index resolution: storms do not end at midnight
    return r


def _object_baselines(r, window=30.0, min_n=5):
    """Causal per-pair baseline: median decay rate of the same object's pairs that ended in the
    previous `window` days (rises excluded)."""
    r = r.sort_values(["norad", "jd2"]).reset_index(drop=True)
    t = pd.to_datetime(r.jd2 - 2440587.5, unit="D")
    neg = r.rate.where(r.rate < 0)
    base = (pd.Series(neg.to_numpy(), index=t).groupby(r.norad.to_numpy())
            .rolling(f"{int(window)}D", closed="left", min_periods=min_n).median())
    r["base"] = base.to_numpy()
    return r


def swarm_index(r):
    """S_b(t): median over objects of rate / own baseline, per 50-km band and 6-hour slot."""
    d = r[np.isfinite(r.base) & (r.base < 0)].copy()
    d["ratio"] = d.rate / d.base
    s = d.groupby(["band", "slot"]).ratio.agg(["median", "size"]).reset_index()
    s = s[s["size"] >= 5].rename(columns={"median": "S", "size": "n"})
    return s


def detect_manoeuvres(hist, use_swarm=True, k_sigma=6.0, floor_km=0.15, inc_floor_deg=0.01,
                      rates=None):
    """Flag manoeuvres. Returns (events, scored_pairs)."""
    r = rates if rates is not None else _object_baselines(decay_rates(hist))
    if use_swarm:
        s = swarm_index(r)
        r = r.merge(s[["band", "slot", "S"]], on=["band", "slot"], how="left")
        r["S"] = r.S.fillna(1.0)
    else:
        r = r.copy()
        r["S"] = 1.0
    r["expected"] = r.base * r.S * r.dt
    r["resid"] = r.da - r.expected
    out = []
    for norad, g in r.groupby("norad", sort=False):
        g = g[np.isfinite(g.expected)]
        if len(g) < 12:
            continue
        res = g.resid.to_numpy()
        di = g.di.to_numpy()
        # local, causal noise scale: robust spread of the previous 30 residuals (noise grows as an
        # object sinks and decays faster, so a single per-object scale is wrong)
        rs_ = pd.Series(res)
        med = rs_.rolling(30, min_periods=10, closed="left").median()
        sig = 1.4826 * (rs_ - med).abs().rolling(30, min_periods=10, closed="left").median()
        di_ = pd.Series(di)
        sig_i = 1.4826 * (di_ - di_.rolling(30, min_periods=10, closed="left").median()).abs() \
            .rolling(30, min_periods=10, closed="left").median()
        for k, row in enumerate(g.itertuples()):
            if not np.isfinite(sig.iloc[k]):
                continue
            z = (res[k] - med.iloc[k])
            # 25% of the expected decay: margin for uncertainty in the swarm index itself
            thr = max(k_sigma * (sig.iloc[k] + 1e-3), floor_km, 0.25 * abs(row.expected))
            kind = None
            if z > thr:
                kind = "raise"
            elif z < -thr and abs(row.da) > 3 * abs(row.expected):
                kind = "lower"
            if np.isfinite(sig_i.iloc[k]) and abs(di[k]) > max(8 * (sig_i.iloc[k] + 1e-4), inc_floor_deg):
                kind = (kind + "+plane") if kind else "plane"
            if kind:
                out.append(dict(norad=int(norad), jd1=row.jd1, jd2=row.jd2, kind=kind, da_km=row.da,
                                expected_km=row.expected, di_deg=di[k], sigma_km=float(sig.iloc[k]), S=row.S,
                                alt=row.alt, t=pd.to_datetime(row.jd2 - 2440587.5, unit="D")))
    return pd.DataFrame(out), r


# ---------------------------------------------------------------- 2. fragmentation
def launch_of(intl):
    """'1999-025DXJ' -> '1999-025'; TLE style '99025DXJ' -> '1999-025'."""
    s = str(intl).strip()
    if len(s) >= 8 and s[4] == "-":
        return s[:8]
    if len(s) >= 5 and s[:5].isdigit():
        yy = int(s[:2])
        return f"{1900 + yy if yy >= 57 else 2000 + yy}-{s[2:5]}"
    return ""


def detect_fragmentation(cat, known_ids, now, min_pieces=3, old_days=60):
    """New catalogue objects of an old launch, grouped by launch. Returns candidate events."""
    now = pd.Timestamp(now)
    new = cat[~cat.norad.isin(set(known_ids))].copy()
    if new.empty:
        return pd.DataFrame()
    new["launch"] = [launch_of(x) for x in new.intl]
    new = new[new.launch.str.len() == 8]
    new["launch_year"] = new.launch.str[:4].astype(int)
    new = new[new.launch_year < now.year + 1]
    rows = []
    for L, g in new.groupby("launch"):
        fam = cat[[launch_of(x) == L for x in cat.intl]]
        old = fam[fam.norad.isin(set(known_ids))]
        if len(g) < min_pieces or old.empty:
            continue
        parent = old.sort_values("intl").iloc[0]
        rows.append(dict(launch=L, parent=int(parent.norad), parent_name=parent.get("name", ""),
                         pieces=int(len(g)), piece_ids=g.norad.astype(int).tolist()[:200],
                         alt_km=float(parent.perigee_km + parent.apogee_km) / 2))
    return pd.DataFrame(rows)


def breakup_time(parent_tle, piece_tles, t_end, days=10.0, step_min=1.0):
    """Estimate when the pieces left the parent: time over the past `days` that minimises the
    median parent-to-piece distance (SGP4 backward propagation)."""
    from sgp4.api import Satrec, SatrecArray, WGS72
    sats = [Satrec.twoline2rv(*parent_tle, WGS72)] + [Satrec.twoline2rv(a, b, WGS72) for a, b in piece_tles]
    sa = SatrecArray(sats)
    t_end = pd.Timestamp(t_end)
    times = t_end - pd.to_timedelta(np.arange(0, days * 1440 + 1, step_min), unit="min")
    jd = times.to_julian_date().to_numpy()
    whole = np.floor(jd - 0.5) + 0.5
    e, r, v = sa.sgp4(whole, jd - whole)
    d = np.linalg.norm(r[1:] - r[0], axis=2)                 # pieces x times
    d[e[1:] != 0] = np.nan
    med = np.nanmedian(d, axis=0)
    k = int(np.nanargmin(med))
    # refine on a 5-s grid around the minimum
    tf = times[k] + pd.to_timedelta(np.arange(-120, 121, 5), unit="s")
    jd = tf.to_julian_date().to_numpy(); whole = np.floor(jd - 0.5) + 0.5
    e, r, v = sa.sgp4(whole, jd - whole)
    d = np.linalg.norm(r[1:] - r[0], axis=2)
    med = np.nanmedian(d, axis=0)
    kk = int(np.nanargmin(med))
    dv = None
    if len(piece_tles):
        # spread of pieces' velocity relative to parent at the estimated time = ejection speed
        dv = float(np.nanmedian(np.linalg.norm(v[1:, kk] - v[0, kk], axis=1)) * 1e3)
    return dict(t_breakup=tf[kk], median_sep_km=float(med[kk]), ejection_ms=dv)


# ---------------------------------------------------------------- 3. re-entry
_PROFILE = None


def density_profile():
    """Relative density vs altitude (NRLMSIS 2.1, moderate activity, global average)."""
    global _PROFILE
    if _PROFILE is None:
        import pymsis
        alts = np.arange(100, 701, 5.0)
        lats = np.array([-60, -30, 0, 30, 60.0])
        lons = np.array([0, 90, 180, 270.0])
        dates = np.array([np.datetime64("2025-03-20T00:00"), np.datetime64("2025-03-20T12:00")])
        out = pymsis.calculate(dates, lons, lats, alts, f107s=[150] * 2, f107as=[150] * 2,
                               aps=[[15] * 7] * 2, geomagnetic_activity=-1)
        rho = np.nanmean(out[..., 0], axis=(0, 1, 2))
        _PROFILE = (alts, rho)
    return _PROFILE


def remaining_days(alt_km, rate_km_day, h_end=120.0):
    """Days until the orbit decays to h_end, integrating da/dt ~ rho(h) * sqrt(a) from the
    current decay rate (rate < 0)."""
    alts, rho = density_profile()
    lr = np.interp
    h = float(alt_km)
    if not (rate_km_day < 0) or h <= h_end:
        return np.nan
    a0 = C.RE_KM + h
    rho0 = np.exp(lr(h, alts, np.log(rho)))
    hs = np.arange(h, h_end, -0.5)
    rates = rate_km_day * np.exp(lr(hs, alts, np.log(rho))) / rho0 * np.sqrt((C.RE_KM + hs) / a0)
    return float(np.sum(0.5 / np.abs(rates)))


def reentry_state(g):
    """Current altitude and decay rate from an object's latest element sets (last ~2 days)."""
    g = g.sort_values("jd")
    if "a_km" not in g:
        g = g.assign(a_km=(C.MU / (g.n * 2 * np.pi / 86400) ** 2) ** (1 / 3))
    last = g.jd.iloc[-1]
    w = g[g.jd >= last - 2.0]
    if len(w) < 3 or w.jd.iloc[-1] - w.jd.iloc[0] < 0.5:
        w = g.tail(4)
    if len(w) < 2:
        return None
    p = np.polyfit(w.jd - last, w.a_km, 1)
    perigee = (w.a_km * (1 - w.ecc)).iloc[-1] - C.RE_KM if "ecc" in w else w.a_km.iloc[-1] - C.RE_KM
    return dict(jd=last, alt=float(p[1] - C.RE_KM), perigee=float(perigee), rate=float(p[0]))


class ReentryWindows:
    """Conformal re-entry windows: quantiles of log(true remaining / predicted remaining)."""

    def __init__(self, q_lo=None, q_hi=None, n=0, coverage=0.90):
        self.q_lo, self.q_hi, self.n, self.coverage = q_lo, q_hi, n, coverage

    @classmethod
    def fit(cls, log_ratios, coverage=0.90):
        x = np.asarray(log_ratios, float)
        x = x[np.isfinite(x)]
        a = (1 - coverage) / 2
        k_lo = max(int(np.floor(a * (len(x) + 1))) - 1, 0)
        k_hi = min(int(np.ceil((1 - a) * (len(x) + 1))) - 1, len(x) - 1)
        s = np.sort(x)
        return cls(float(s[k_lo]), float(s[k_hi]), len(x), coverage)

    def window(self, jd_now, pred_days):
        if self.q_lo is None or not np.isfinite(pred_days):
            return np.nan, np.nan
        return jd_now + pred_days * np.exp(self.q_lo), jd_now + pred_days * np.exp(self.q_hi)

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, t):
        return cls(**json.loads(t))


def predict_reentries(hist, now, windows, max_perigee=300.0, max_days=60):
    out = []
    jd_now = pd.Timestamp(now).to_julian_date()
    for norad, g in hist.groupby("norad"):
        g = g.sort_values("jd")
        if g.perigee_km.iloc[-1] > max_perigee or g.jd.iloc[-1] < jd_now - 3:
            continue                                  # not low, or no longer tracked (already down)
        st = reentry_state(g)
        if st is None:
            continue
        d = remaining_days(st["alt"], st["rate"])
        if not np.isfinite(d) or d > max_days:
            continue
        if st["jd"] + d < jd_now:
            continue
        lo, hi = windows.window(st["jd"], d)
        out.append(dict(norad=int(norad), alt_km=st["alt"], rate_km_day=st["rate"],
                        predicted=pd.to_datetime(st["jd"] + d - 2440587.5, unit="D"),
                        window_lo=pd.to_datetime(lo - 2440587.5, unit="D") if np.isfinite(lo) else pd.NaT,
                        window_hi=pd.to_datetime(hi - 2440587.5, unit="D") if np.isfinite(hi) else pd.NaT,
                        days_left=d - (jd_now - st["jd"])))
    return pd.DataFrame(out).sort_values("predicted") if out else pd.DataFrame()


# ---------------------------------------------------------------- 4. drag signature
def drag_signature_changes(iv_cal, bandmap, min_run=3, z=4.0):
    """Network objects whose R departs from their band's daily median for >= min_run intervals."""
    if iv_cal is None or iv_cal.empty or bandmap is None or bandmap.empty:
        return pd.DataFrame()
    d = iv_cal.copy()
    d["day"] = d.tmid.dt.floor("D")
    d["band"] = [C.band_of(a) for a in d.alt]
    bm = bandmap.assign(day=pd.to_datetime(bandmap.day))[["day", "band", "R"]].rename(columns={"R": "Rb"})
    d = d.merge(bm, on=["day", "band"], how="inner")
    d["lr"] = np.log(d.R / d.Rb)
    out = []
    for norad, g in d.sort_values("t2").groupby("norad"):
        if len(g) < 10:
            continue
        hist = g.lr.iloc[:-min_run]
        sig = 1.4826 * np.median(np.abs(hist - hist.median())) + 0.02
        zz = (g.lr - hist.median()) / sig
        tail = zz.iloc[-min_run:]
        if (tail.abs() > z).all() and (np.sign(tail) == np.sign(tail.iloc[-1])).all():
            out.append(dict(norad=int(norad), since=g.t1.iloc[-min_run], z=float(tail.mean()),
                            ratio_change=float(np.exp(g.lr.iloc[-min_run:].mean() - hist.median())),
                            alt=float(g.alt.iloc[-1])))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 5. track gaps
def track_gaps(cat, now, max_age_days=3.0, leo_only=True):
    jd_now = pd.Timestamp(now).to_julian_date()
    c = cat[cat.perigee_km < 2000] if leo_only else cat
    age = jd_now - c.jd
    g = c[(age > max_age_days) & (age < 30)].assign(age_days=age)
    return g[["norad", "name", "type", "age_days", "perigee_km"]].sort_values("age_days", ascending=False)
