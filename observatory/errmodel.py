"""Empirical position-error model for public element sets, calibrated from the catalogue itself.

Public TLE/GP element sets carry no covariance, yet collision probability needs one. We measure it:
for many pairs of element sets of the same object, the older set is propagated to the epoch of
the newer one and the position difference is resolved into radial / along-track / cross-track
(R, T, N) components. For each altitude band and space-weather regime the spread grows with the
propagation time dt (days) as

    sigma_c(dt)^2 = a_c^2 + (b_c dt)^2 + (c_c dt^2)^2          (c = R, T, N)

fitted by non-negative least squares to spreads in dt bins, each scaled so that 1.645 sigma holds 90%
of the measured differences (the errors are heavy-tailed). The newer set
has its own error, so the model slightly overstates the error (conservative). Its calibration is
checked by the share of measured differences inside the model's 90% bounds (reported).
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from . import config as C

ERR_BANDS = [(150, 350), (350, 450), (450, 550), (550, 700), (700, 1000), (1000, 2000), (2000, 50000)]
LAGS = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0]
Z90 = 1.6448536


def err_band(alt):
    for k, (lo, hi) in enumerate(ERR_BANDS):
        if lo <= alt < hi:
            return k
    return len(ERR_BANDS) - 1 if alt >= ERR_BANDS[-1][1] else 0


def _rtn(r, v):
    R = r / np.linalg.norm(r, axis=-1, keepdims=True)
    N = np.cross(r, v)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    T = np.cross(N, R)
    return R, T, N


def measure(els, storm_windows=None, every=3, max_pairs=400_000, seed=0):
    """Position differences (km) between an older element set propagated forward and a newer one.
    els: norad, jd, l1, l2.  storm_windows: list of (start, end) Timestamps (storm regime)."""
    from sgp4.api import Satrec, WGS72
    rng = np.random.default_rng(seed)
    rows = []
    for norad, g in els.sort_values(["norad", "jd"]).groupby("norad"):
        jd = g.jd.to_numpy()
        if len(jd) < 4:
            continue
        L1, L2 = g.l1.to_numpy(), g.l2.to_numpy()
        sats = {}
        start = int(rng.integers(0, every))
        for i in range(start, len(jd) - 1, every):
            for lag in LAGS:
                j = int(np.searchsorted(jd, jd[i] + lag))
                cand = [k for k in (j - 1, j) if i < k < len(jd)]
                if not cand:
                    continue
                k = min(cand, key=lambda k: abs(jd[k] - jd[i] - lag))
                if abs(jd[k] - jd[i] - lag) > 0.3 * lag:
                    continue
                for q in (i, k):
                    if q not in sats:
                        sats[q] = Satrec.twoline2rv(L1[q], L2[q], WGS72)
                s_old, s_new = sats[i], sats[k]
                e1, r1, v1 = s_old.sgp4(s_new.jdsatepoch, s_new.jdsatepochF)
                e2, r2, v2 = s_new.sgp4(s_new.jdsatepoch, s_new.jdsatepochF)
                if e1 or e2:
                    continue
                r2, v2 = np.array(r2), np.array(v2)
                d = np.array(r1) - r2
                R, T, N = _rtn(r2, v2)
                rows.append((int(norad), jd[i], jd[k], float(np.linalg.norm(r2) - C.RE_KM),
                             float(d @ R), float(d @ T), float(d @ N)))
        if len(rows) > max_pairs:
            break
    m = pd.DataFrame(rows, columns=["norad", "jd0", "jd1", "alt", "dR", "dT", "dN"])
    m["dt"] = m.jd1 - m.jd0
    m["band"] = [err_band(a) for a in m.alt]
    m["regime"] = "quiet"
    if storm_windows:
        t0 = pd.to_datetime(m.jd0 - 2440587.5, unit="D")
        t1 = pd.to_datetime(m.jd1 - 2440587.5, unit="D")
        storm = np.zeros(len(m), bool)
        for a, b in storm_windows:
            storm |= (t0 <= b) & (t1 >= a)
        m.loc[storm, "regime"] = "storm"
    return m


def _robust_sigma(x):
    """Spread scaled so that +-1.645 sigma holds 90% of the measured differences. Element-set
    errors are heavy-tailed, so a MAD-based sigma would cover only ~70%; the 90% point is what
    the collision-probability and avoidance thresholds depend on."""
    x = np.asarray(x, float)
    return np.quantile(np.abs(x - np.median(x)), 0.90) / Z90


class ErrorModel:
    """sigma_R/T/N (km) as a function of band, regime and propagation time dt (days)."""

    def __init__(self, coef=None, checks=None):
        self.coef = coef or {}        # "band|regime" -> {"R": [a,b,c], "T": [...], "N": [...]}
        self.checks = checks or {}

    @classmethod
    def fit(cls, m, min_n=40):
        coef, checks = {}, {}
        for (b, reg), g in m.groupby(["band", "regime"]):
            fits = {}
            for comp in ("R", "T", "N"):
                xs, ys, ws = [], [], []
                for lag in LAGS:
                    s = g[(g.dt > lag * 0.7) & (g.dt < lag * 1.3)]
                    if len(s) < min_n:
                        continue
                    xs.append(s.dt.median())
                    ys.append(_robust_sigma(s["d" + comp]) ** 2)
                    ws.append(np.sqrt(len(s)))
                if len(xs) < 2:
                    continue
                x, y, w = map(np.asarray, (xs, ys, ws))
                A = np.vstack([np.ones_like(x), x ** 2, x ** 4]).T / y[:, None] * w[:, None]
                sol, _ = nnls(A, w * np.ones_like(y))      # relative least squares on variance
                fits[comp] = np.sqrt(sol).tolist()
            if len(fits) == 3:
                key = f"{b}|{reg}"
                coef[key] = fits
                em = cls({key: fits})
                sig = np.array([em.sigmas(ERR_BANDS[b][0] + 1, dt, reg) for dt in g.dt])
                inside = (np.abs(g[["dR", "dT", "dN"]].to_numpy()) <= Z90 * sig).mean(axis=0)
                checks[key] = dict(n=int(len(g)), objects=int(g.norad.nunique()),
                                   inside90_R=float(inside[0]), inside90_T=float(inside[1]),
                                   inside90_N=float(inside[2]))
        return cls(coef, checks)

    def sigmas(self, alt_km, dt_days, regime="quiet"):
        b = err_band(alt_km)
        key = f"{b}|{regime}"
        if key not in self.coef:
            key = f"{b}|quiet"
        if key not in self.coef:                       # nearest calibrated band
            keys = [k for k in self.coef if k.endswith("|quiet")] or list(self.coef)
            if not keys:
                return np.array([0.1, 1.0, 0.1]) * (1 + abs(dt_days))
            key = min(keys, key=lambda k: abs(int(k.split("|")[0]) - b))
        dt = abs(float(dt_days))
        out = []
        for comp in ("R", "T", "N"):
            a, bb, c = self.coef[key][comp]
            out.append(np.sqrt(a ** 2 + (bb * dt) ** 2 + (c * dt * dt) ** 2))
        return np.array(out)

    def cov_rtn_m2(self, alt_km, dt_days, regime="quiet"):
        s = self.sigmas(alt_km, dt_days, regime) * 1e3
        return np.diag(s ** 2)

    def to_json(self):
        return json.dumps(dict(bands=ERR_BANDS, coef=self.coef, checks=self.checks), indent=1)

    @classmethod
    def from_json(cls, text):
        d = json.loads(text)
        return cls(d["coef"], d.get("checks"))

    @classmethod
    def load(cls, path=None):
        path = path or (C.STORE / "errmodel.json")
        return cls.from_json(path.read_text()) if path.exists() else cls()
