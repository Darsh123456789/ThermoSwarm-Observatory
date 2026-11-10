"""Tests for the Sentinel layer: screening, avoidance, error model, anomalies, catalogue parsing."""
import json

import numpy as np
import pandas as pd
import pytest
from sgp4.api import Satrec, WGS72
from sgp4 import exporter

from observatory import config as C

MU = 398600.4418


def _jd(t):
    return pd.Timestamp(t).to_julian_date()


def make_tle(norad, epoch, a_km, ecc, inc, raan, argp, ma, bstar=0.0):
    s = Satrec()
    n = np.sqrt(MU / a_km ** 3) * 60                  # rad/min
    s.sgp4init(WGS72, "i", norad, _jd(epoch) - 2433281.5, bstar, 0.0, 0.0, ecc, np.radians(argp),
               np.radians(inc), np.radians(ma), n, np.radians(raan))
    return exporter.export_tle(s)


def rv2coe(r, v):
    """Two-body classical elements from an ECI state (km, km/s)."""
    r, v = np.asarray(r), np.asarray(v)
    h = np.cross(r, v); n = np.cross([0, 0, 1], h)
    e = ((np.dot(v, v) - MU / np.linalg.norm(r)) * r - np.dot(r, v) * v) / MU
    ecc = np.linalg.norm(e)
    a = 1 / (2 / np.linalg.norm(r) - np.dot(v, v) / MU)
    inc = np.degrees(np.arccos(h[2] / np.linalg.norm(h)))
    raan = np.degrees(np.arctan2(n[1], n[0])) % 360
    argp = np.degrees(np.arccos(np.clip(np.dot(n, e) / (np.linalg.norm(n) * ecc), -1, 1)))
    if e[2] < 0:
        argp = 360 - argp
    nu = np.degrees(np.arccos(np.clip(np.dot(e, r) / (ecc * np.linalg.norm(r)), -1, 1)))
    if np.dot(r, v) < 0:
        nu = 360 - nu
    E = 2 * np.arctan(np.sqrt((1 - ecc) / (1 + ecc)) * np.tan(np.radians(nu) / 2))
    M = np.degrees(E - ecc * np.sin(E)) % 360
    return a, ecc, inc, raan, argp, M


def random_catalog(n, epoch, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        a = C.RE_KM + rng.uniform(500, 560)
        l1, l2 = make_tle(10000 + k, epoch, a, rng.uniform(0.0001, 0.002), rng.uniform(50, 99),
                          rng.uniform(0, 360), rng.uniform(0, 360), rng.uniform(0, 360))
        rows.append(dict(norad=10000 + k, l1=l1, l2=l2, jd=_jd(epoch), perigee_km=a - C.RE_KM - 2,
                         apogee_km=a - C.RE_KM + 2, name=f"T{k}", type="PAYLOAD", rcs="MEDIUM"))
    return pd.DataFrame(rows)


def test_screening_finds_every_close_approach():
    from observatory.screening import screen, refine, _satrec_array, _jd_split
    epoch = pd.Timestamp("2026-01-01")
    cat = random_catalog(120, epoch)
    H, D = 2.0, 60.0
    ev, n = screen(cat, epoch, hours=H, d_km=D)
    sa, sats, keep = _satrec_array(cat)
    times = pd.DatetimeIndex(epoch + pd.to_timedelta(np.arange(0, H * 3600 + 1, 2.0), unit="s"))
    jd, fr = _jd_split(times)
    e, r, v = sa.sgp4(jd, fr)
    truth = []
    for a in range(len(cat)):
        d = np.linalg.norm(r[a + 1:] - r[a], axis=2)
        for k in np.where(d.min(axis=1) < D * 2)[0]:
            dd = d[k]
            loc = np.where((dd[1:-1] < dd[:-2]) & (dd[1:-1] <= dd[2:]))[0] + 1
            for l in loc:
                x = refine(sats[a], sats[a + 1 + k], times[l], 4.0)
                if x and x["miss_km"] < D and 0 < (x["tca"] - epoch).total_seconds() < H * 3600:
                    truth.append((min(cat.norad[a], cat.norad[a + 1 + k]), max(cat.norad[a], cat.norad[a + 1 + k]), x["tca"]))
    assert len(truth) > 3
    found = [((ev[["i", "j"]].min(axis=1) == i) & (ev[["i", "j"]].max(axis=1) == j) &
              ((ev.tca - t).abs() < pd.Timedelta("30s"))).any() for i, j, t in truth]
    assert all(found), f"missed {len(found) - sum(found)} of {len(found)}"


def test_hill_burn_displacement_matches_numerical_integration():
    from scipy.integrate import solve_ivp
    from observatory.avoid import cw_shift
    from observatory.collision import rtn_basis
    mu, J2, RE = 398600.4418e9, 1.08263e-3, 6378137.0

    def f(t, y):
        r = y[:3]; R = np.linalg.norm(r); z = r[2]
        k = 1.5 * J2 * mu * RE ** 2 / R ** 5
        acc = -mu * r / R ** 3 + k * np.array([r[0] * (5 * z * z / R ** 2 - 1), r[1] * (5 * z * z / R ** 2 - 1),
                                                r[2] * (5 * z * z / R ** 2 - 3)])
        return np.r_[y[3:], acc]
    a = RE + 550e3; v = np.sqrt(mu / a); inc = np.radians(53)
    y0 = np.array([a, 0, 0, 0, v * np.cos(inc), v * np.sin(inc)]); n = np.sqrt(mu / a ** 3); T = 2 * 2 * np.pi / n
    B = rtn_basis(y0[:3], y0[3:]); y1 = y0.copy(); y1[3:] += B[1] * 0.05
    s0 = solve_ivp(f, (0, T), y0, rtol=1e-11, atol=1e-6).y[:, -1]
    s1 = solve_ivp(f, (0, T), y1, rtol=1e-11, atol=1e-6).y[:, -1]
    d = rtn_basis(s0[:3], s0[3:]) @ (s1[:3] - s0[:3])
    cw = cw_shift(0.05, 0, T, n)
    assert abs(d[1] - cw[1]) / abs(cw[1]) < 0.01


def test_avoidance_plan_brings_pc_below_target():
    from observatory.avoid import plan
    from observatory.collision import Conjunction
    r = np.array([6928e3, 0, 0]); v1 = np.array([0, 7585.0, 0]); v2 = np.array([0, 0, 7585.0])
    cov = np.diag([30.0 ** 2, 200.0 ** 2, 30.0 ** 2])
    c = Conjunction(r, v1, cov, r + np.array([0, 20.0, 15.0]), v2, cov, hbr=10, lead_h=30)
    assert c.pc() > 1e-4
    p = plan(c)
    assert p["recommended"] is not None and p["recommended"]["pc_after"] <= 1e-6 * 1.001
    dvs = [abs(o["dv_mm_s"]) for o in p["options"] if o["kind"] == "along-track"]
    hs = [o["burn_before_tca_h"] for o in p["options"] if o["kind"] == "along-track"]
    assert dvs[np.argmax(hs)] <= dvs[np.argmin(hs)]          # earlier burns need less delta-v


def test_pc_upper_bound_holds():
    from observatory.collision import Conjunction
    rng = np.random.default_rng(1)
    for _ in range(20):
        r = np.array([6928e3, 0, 0]); v1 = np.array([0, 7585.0, 0]); v2 = np.array([0, 3000.0, 7000.0])
        cov = np.diag(rng.uniform(10, 500, 3) ** 2)
        c = Conjunction(r, v1, cov, r + rng.normal(0, 200, 3), v2, cov, hbr=rng.uniform(1, 20))
        m, S = c._projected()
        assert c.pc() <= c.hbr ** 2 / (2 * np.sqrt(np.linalg.det(S))) * (1 + 1e-9)


def test_error_model_recovers_known_growth():
    from observatory.errmodel import ErrorModel, LAGS, Z90
    rng = np.random.default_rng(0)
    rows = []
    true = dict(R=(0.05, 0.03, 0.0), T=(0.3, 0.4, 0.2), N=(0.05, 0.02, 0.0))
    for lag in LAGS:
        for _ in range(3000):
            dt = lag * rng.uniform(0.9, 1.1)
            s = {c: np.sqrt(a ** 2 + (b * dt) ** 2 + (cc * dt * dt) ** 2) for c, (a, b, cc) in true.items()}
            rows.append(dict(norad=1, alt=500, band=2, regime="quiet", dt=dt,
                             dR=rng.normal(0, s["R"]), dT=rng.normal(0, s["T"]), dN=rng.normal(0, s["N"])))
    em = ErrorModel.fit(pd.DataFrame(rows))
    for dt in (1.0, 3.0):
        got = em.sigmas(500, dt)
        want = [np.sqrt(a ** 2 + (b * dt) ** 2 + (cc * dt * dt) ** 2) for a, b, cc in true.values()]
        assert np.allclose(got, want, rtol=0.12)
    assert abs(em.checks["2|quiet"]["inside90_T"] - 0.90) < 0.03


def test_breakup_time_recovered_from_fragments():
    from observatory.anomalies import breakup_time
    tb = pd.Timestamp("2026-02-01 05:17:00")
    l1, l2 = make_tle(30000, tb, C.RE_KM + 780, 0.001, 98.6, 40, 90, 10)
    p = Satrec.twoline2rv(l1, l2, WGS72)
    jd = tb.to_julian_date(); _, r, v = p.sgp4(np.floor(jd - 0.5) + 0.5, jd - (np.floor(jd - 0.5) + 0.5))
    rng = np.random.default_rng(3)
    a, e, i, O, w, M = rv2coe(r, v)
    parent = make_tle(30000, tb, a, max(e, 1e-5), i, O, w, M)
    pieces = []
    for k in range(12):
        dv = rng.normal(0, 0.012, 3)                         # ~20 m/s ejection
        a, e, i, O, w, M = rv2coe(r, np.array(v) + dv)
        pieces.append(make_tle(30001 + k, tb, a, max(e, 1e-5), i, O, w, M))
    est = breakup_time(parent, pieces, tb + pd.Timedelta("3D"), days=5, step_min=1)
    assert abs((est["t_breakup"] - tb).total_seconds()) < 120
    assert 5 < est["ejection_ms"] < 60


def test_swarm_index_prevents_storm_false_alarms():
    from observatory.anomalies import detect_manoeuvres
    rng = np.random.default_rng(0)
    rows = []
    t0 = 2461000.5
    storm = (t0 + 30, t0 + 33)
    for k in range(40):
        a = C.RE_KM + 300 + rng.uniform(0, 40)
        for d in np.arange(0, 45, 0.5):
            jd = t0 + d
            rate = -0.3 * (4.0 if storm[0] <= jd < storm[1] else 1.0)   # storm: 4x faster decay
            a += rate * 0.5 + rng.normal(0, 0.004)
            rows.append(dict(norad=k, jd=jd, a_km=a, incl=51.6 + rng.normal(0, 0.0005), ecc=0.001))
    h = pd.DataFrame(rows)
    naive, _ = detect_manoeuvres(h, use_swarm=False)
    aware, _ = detect_manoeuvres(h, use_swarm=True)
    assert len(naive) > 20                    # the naive detector calls the storm a lowering campaign
    assert len(aware) <= 2


def test_reentry_windows_and_launch_parsing():
    from observatory.anomalies import ReentryWindows, launch_of, remaining_days
    w = ReentryWindows.fit(np.random.default_rng(0).normal(0, 0.2, 999))
    assert -0.36 < w.q_lo < -0.29 and 0.29 < w.q_hi < 0.36
    assert launch_of("1999-025DXJ") == "1999-025" and launch_of("99025A") == "1999-025" and launch_of("24001B") == "2024-001"
    assert remaining_days(220, -5.0) < remaining_days(300, -1.0)


def test_parse_tracss_omm_records():
    from observatory.catalog import parse_omm
    l1, l2 = make_tle(25544, pd.Timestamp("2026-01-01"), C.RE_KM + 420, 0.0005, 51.6, 10, 20, 30)
    recs = [{"omm": {"NORAD_CAT_ID": 25544, "OBJECT_NAME": "ISS (ZARYA)", "OBJECT_ID": "1998-067A",
                     "OBJECT_TYPE": "PAYLOAD", "USER_DEFINED_TLE_LINE_1": l1, "USER_DEFINED_TLE_LINE_2": l2}},
            {"NORAD_CAT_ID": "bad"}]
    df = parse_omm(recs, "tracss")
    assert len(df) == 1 and df.norad[0] == 25544 and abs(df.perigee_km[0] - 418) < 5 and df.type[0] == "PAYLOAD"


def test_small_disk_closed_form_matches_quadrature():
    import observatory.collision as col
    rng = np.random.default_rng(4)
    for _ in range(30):
        S = np.diag(rng.uniform(80, 2000, 2) ** 2); S[0, 1] = S[1, 0] = 0.3 * np.sqrt(S[0, 0] * S[1, 1])
        m = rng.normal(0, 1500, 2); hbr = 0.04 * np.sqrt(np.linalg.eigvalsh(S).min())
        fast = col.pc_2d(m, S, hbr)
        old = col.SMALL_DISK; col.SMALL_DISK = 0.0
        exact = col.pc_2d(m, S, hbr)
        col.SMALL_DISK = old
        assert abs(fast - exact) <= 0.005 * exact + 1e-300
