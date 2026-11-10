"""Drag physics along SGP4 trajectories (Picone, Emmert & Lean 2005), as in the ThermoSwarm paper.

g(t) = (3/2) n rho_MSIS v_rel^2 / v   is the rate of change of mean motion per unit
ballistic coefficient.  Over an interval of length T:
    I = int_0^T g dt                       ->  f = dn_obs / I = B * (rho_true / rho_MSIS)
    J = int_0^T (T - t) g dt               ->  along-track error of a prediction using
                                               B*R_pred:  e = a * (B*R_pred - f) * J
"""
import numpy as np
import pymsis
from sgp4.api import Satrec
from sgp4.propagation import gstime

from .config import RE_KM, FLAT, OMEGA_E


def _geodetic(r):
    rn = np.linalg.norm(r, axis=1)
    lat = np.arctan(np.tan(np.arcsin(r[:, 2] / rn)) / (1 - FLAT) ** 2)
    alt = rn / 1e3 - RE_KM * (1 - FLAT * np.sin(lat) ** 2)
    return lat, alt


def drag_integrals(l1, l2, jd1, jd2, drivers, nsamp=96):
    """Return dict(I, J, alt, alt_eff, rho_mean, a_km) for the TLE (l1, l2) over [jd1, jd2],
    or None if SGP4 fails."""
    s = Satrec.twoline2rv(l1, l2)
    jd = jd1 + (np.arange(nsamp) + 0.5) * (jd2 - jd1) / nsamp
    e, r, v = s.sgp4_array(np.floor(jd), jd - np.floor(jd))
    if np.any(e):
        return None
    r, v = r * 1e3, v * 1e3
    gm = np.array([gstime(x) for x in jd])
    lon = (np.degrees(np.arctan2(r[:, 1], r[:, 0]) - gm) + 180) % 360 - 180
    lat, alt = _geodetic(r)
    vrel = v.copy()
    vrel[:, 0] += OMEGA_E * r[:, 1]
    vrel[:, 1] -= OMEGA_E * r[:, 0]
    unix = (jd - 2440587.5) * 86400
    f107, f107a, aps = drivers(unix)
    dates = (unix * 1e6).astype("int64").astype("datetime64[us]")
    rho = pymsis.calculate(dates, lon, np.degrees(lat), alt, f107s=f107, f107as=f107a,
                           aps=aps, version=2.1)[:, 0]
    n = s.no_kozai / 60.0                                   # rad/s
    g = 1.5 * n * rho * np.linalg.norm(vrel, axis=1) ** 2 / np.linalg.norm(v, axis=1)
    T = (jd2 - jd1) * 86400
    dt = T / nsamp
    tt = (np.arange(nsamp) + 0.5) * dt
    a_km = (398600.4418 / n ** 2) ** (1 / 3)
    return dict(I=float(g.sum() * dt), J=float(((T - tt) * g).sum() * dt),
                alt=float(alt.mean()), alt_eff=float(np.average(alt, weights=rho)),
                rho_mean=float(rho.mean()), a_km=float(a_km))


def along_track_error_km(a_km, B, R_pred, f_obs, J):
    """Drag-induced along-track error (km) of predicting with B*R_pred when truth was f_obs."""
    return a_km * (B * R_pred - f_obs) * J


def along_track_bound_km(a_km, B, R_lo, R_hi, R_pred, J):
    """Half-width (km) of the along-track uncertainty implied by an interval [R_lo, R_hi]
    around the prediction R_pred (the larger of the two sides)."""
    return a_km * B * J * np.maximum(np.abs(R_hi - R_pred), np.abs(R_pred - R_lo))
