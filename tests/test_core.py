import numpy as np
import pandas as pd
import pytest

from observatory import config as C
from observatory.collision import Conjunction, pc_2d, read_cdm, storm_aware
from observatory.conformal import AdaptiveConformal
from observatory.drivers import kp_to_ap, parse_gfz, parse_swpc_3day
from observatory.referee import Registry, interval_score, score_pairs


# ------------------------------------------------------------------ collision probability
def test_pc_matches_monte_carlo():
    rng = np.random.default_rng(0)
    mu, S, R = np.array([40.0, 15.0]), np.array([[900.0, 200.0], [200.0, 2500.0]]), 20.0
    x = rng.multivariate_normal(mu, S, 2_000_000)
    mc = (np.hypot(x[:, 0], x[:, 1]) < R).mean()
    assert abs(pc_2d(mu, S, R) - mc) < 4 * np.sqrt(mc * (1 - mc) / len(x))


def test_pc_small_disk_limit():
    # disk much smaller than covariance: Pc ~ area * density at the origin
    mu, S, R = np.array([100.0, -50.0]), np.diag([400.0 ** 2, 300.0 ** 2]), 1.0
    dens = np.exp(-0.5 * mu @ np.linalg.inv(S) @ mu) / (2 * np.pi * np.sqrt(np.linalg.det(S)))
    assert pc_2d(mu, S, R) == pytest.approx(np.pi * R ** 2 * dens, rel=1e-3)


def _conj(miss=200.0, st=200.0):
    r = np.array([C.RE_KM * 1e3 + 450e3, 0, 0])
    v1, v2 = np.array([0, 7600.0, 0]), np.array([0, 0, 7600.0])
    cov = np.diag([30.0 ** 2, st ** 2, 50.0 ** 2])
    return Conjunction(r, v1, cov, r + np.array([0, 0, 0]) + np.array([0, miss, 0]) * 0 + np.array([0, -miss / np.sqrt(2), miss / np.sqrt(2)]),
                       v2, cov, hbr=15.0, lead_h=72, alt1=450, alt2=450)


def test_storm_inflation_never_hidden():
    c = _conj()
    terms = {C.band_of(450): dict(bound72_km=1.0, shift72_km=0.2)}
    r = storm_aware(c, terms, n_scan=7)
    assert r["pc_max"] >= r["pc_nominal"] * 0.999
    assert r["drag_bound1_m"] == pytest.approx(1000.0)


def test_cdm_roundtrip():
    cdm = """CCSDS_CDM_VERS = 1.0
CREATION_DATE = 2026-11-10T00:00:00
TCA = 2026-11-12T12:00:00
OBJECT = OBJECT1
OBJECT_DESIGNATOR = 11111
X = 6828.137 [km]
Y = 0 [km]
Z = 0 [km]
X_DOT = 0 [km/s]
Y_DOT = 7.6 [km/s]
Z_DOT = 0 [km/s]
CR_R = 900 [m**2]
CT_R = 0
CT_T = 40000
CN_R = 0
CN_T = 0
CN_N = 2500
OBJECT = OBJECT2
OBJECT_DESIGNATOR = 22222
X = 6828.137 [km]
Y = -0.1414 [km]
Z = 0.1414 [km]
X_DOT = 0 [km/s]
Y_DOT = 0 [km/s]
Z_DOT = 7.6 [km/s]
CR_R = 900
CT_R = 0
CT_T = 40000
CN_R = 0
CN_T = 0
CN_N = 2500
"""
    c = read_cdm(cdm)
    assert c.lead_h == pytest.approx(60.0)
    assert c.ids == ("11111", "22222")
    assert c.miss_m() == pytest.approx(200.0, rel=1e-3)
    assert 0 < c.pc() < 1


# ------------------------------------------------------------------ conformal coverage
def test_adaptive_conformal_reaches_target_under_drift():
    rng = np.random.default_rng(1)
    ac = AdaptiveConformal(coverage=0.9, gamma=0.01)
    hits = []
    for t in range(6000):
        scale = 0.1 if t < 3000 else 0.3                    # error level triples halfway
        truth = np.exp(rng.normal(0, scale))
        lo, hi, _ = ac.interval("m", [1.0], ["quiet"])
        if np.isfinite(lo[0]):
            hits.append(lo[0] <= truth <= hi[0])
        ac.update("m", [1.0], [truth], ["quiet"], lo, hi)
    late = np.mean(hits[-2000:])
    assert 0.86 <= late <= 0.94


def test_interval_score():
    s = interval_score(np.array([0.0]), np.array([1.0]), np.array([1.5]), 0.1)
    assert s[0] == pytest.approx(1.0 + 20 * 0.5)


# ------------------------------------------------------------------ registry integrity
def _fc():
    return pd.DataFrame(dict(model=["a", "b"], norad=[1, 1], t0=pd.Timestamp("2026-01-01"), jd1=2461041.5,
                             alt=400.0, band=4, regime="quiet", B=0.1, a_km=6778.0, R_pred=[1.0, 1.1],
                             lo=[0.9, 1.0], hi=[1.1, 1.2]))


def test_registry_chain_and_tamper_detection(tmp_path):
    reg = Registry(tmp_path, "test")
    reg.commit("2026-01-01", _fc(), created_utc="x")
    reg.commit("2026-01-02", _fc(), created_utc="y")
    assert reg.verify()[0]
    with pytest.raises(FileExistsError):
        reg.commit("2026-01-02", _fc())
    p = tmp_path / "forecasts" / "2026-01-01.csv.gz"
    p.write_bytes(p.read_bytes() + b"x")
    ok, msg = reg.verify()
    assert not ok and "modified" in msg


def test_scoring_reproduces_paper_definition():
    m = _fc().assign(f72=0.12, J=0.5)
    s = score_pairs(m)
    # e = a (B R_pred - f72) J
    assert s.e_km.iloc[0] == pytest.approx(6778.0 * (0.1 * 1.0 - 0.12) * 0.5)
    assert s.R72.iloc[0] == pytest.approx(1.2)
    assert not s.covered.iloc[0] and s.covered.iloc[1]


# ------------------------------------------------------------------ drivers
def test_kp_to_ap_table():
    assert kp_to_ap([0, 3, 14 / 3, 5, 9]).tolist() == [0, 15, 39, 48, 400]


def test_parse_gfz_and_swpc():
    line = ("2026 11 08 34640 34640.5 2621  1 2.333 2.667 3.000 2.000 1.667 1.333 2.000 2.333"
            "   9  12  15   7   6   5   7   9   9 101 145.2 150.1 1\n")
    three, daily = parse_gfz("# header\n" + line)
    assert len(three) == 8 and three.ap.iloc[2] == 15 and daily.f107.iloc[0] == pytest.approx(145.2)  # observed, as in the paper
    swpc = """:Product: 3-Day Forecast
NOAA Kp index breakdown Nov 10-Nov 12 2026

             Nov 10       Nov 11       Nov 12
00-03UT       2.67         5.33 (G1)    3.00
03-06UT       2.33         4.67         2.67
06-09UT       2.00         4.00         2.33
09-12UT       2.00         3.67         2.00
12-15UT       2.33         3.33         2.00
15-18UT       2.67         3.00         2.33
18-21UT       3.00         3.33         2.67
21-00UT       4.00         3.67         2.67
"""
    kp = parse_swpc_3day(swpc)
    assert len(kp) == 24
    assert kp[pd.Timestamp("2026-11-11 00:00")] == pytest.approx(5.33)


def test_classify_separates_riskier_from_diluted():
    import pandas as pd
    from observatory.collision import classify
    r = classify(pd.DataFrame(dict(cat_nominal=["medium", "high", "low", "medium"],
                                   cat_storm=["medium", "medium", "low", "low"],
                                   cat_max=["high", "high", "low", "medium"])))
    assert r.escalated.tolist() == [True, False, False, False]
    assert r.diluted.tolist() == [False, True, False, True]
