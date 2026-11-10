"""Collision-avoidance options: the smallest manoeuvre that brings Pc below a target.

For a near-circular orbit, an impulsive velocity change applied t seconds before closest
approach displaces the manoeuvring object at TCA by (Clohessy-Wiltshire / Hill equations,
x radial, y along-track, n mean motion):
  along-track burn dv_T:  x = 2 dv_T (1 - cos nt) / n         y = dv_T (4 sin nt / n - 3 t)
  radial burn dv_R:       x = dv_R sin nt / n                 y = 2 dv_R (cos nt - 1) / n
The displacement is applied to the primary's position at TCA and the Pc recomputed with the
same covariance (and the storm-aware drag terms), so the plan answers: "with what we know about
the atmosphere today, how big a burn, how early, makes this encounter safe?"

Burns earlier in time need far less delta-v: along-track displacement grows ~3 dv t. The
planner scans burn times on a half-orbit grid (radial effect peaks every half orbit) and finds the
minimum |dv| per burn time by bisection. The linearised displacement is verified against a
numerical two-body + J2 integration in the tests.
"""
import numpy as np

from . import config as C
from .collision import rtn_basis

PC_TARGET = 1e-6


def cw_shift(dv_t, dv_r, t, n):
    """RTN displacement (m) at time t (s) after an impulse (m/s), mean motion n (rad/s)."""
    x = 2 * dv_t * (1 - np.cos(n * t)) / n + dv_r * np.sin(n * t) / n
    y = dv_t * (4 * np.sin(n * t) / n - 3 * t) + 2 * dv_r * (np.cos(n * t) - 1) / n
    return np.array([x, y, 0.0])


def _pc_with_shift(conj, shift_rtn, extra1, extra2):
    """Pc after moving the primary (object 1) by shift_rtn (m, RTN at TCA)."""
    B1 = rtn_basis(conj.r1, conj.v1)
    r1_orig = conj.r1
    conj.r1 = r1_orig + B1.T @ shift_rtn
    try:
        return conj.pc(extra1, extra2)
    finally:
        conj.r1 = r1_orig


def plan(conj, extra1=(0.0, 0.0), extra2=(0.0, 0.0), target=PC_TARGET, max_dv=1.0,
         lead_max_h=None):
    """Return a list of options (burn lead time, direction, delta-v) that bring Pc <= target.
    extra = (added along-track variance m^2, shift m) for each object (the storm-aware terms)."""
    a = np.linalg.norm(conj.r1)
    n = np.sqrt(C.MU * 1e9 / a ** 3)
    period = 2 * np.pi / n
    lead_max = (lead_max_h or max(conj.lead_h - 1.0, 0.5)) * 3600
    pc0 = conj.pc(extra1, extra2)
    options = []
    halves = np.array([0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48]) * period
    for t in halves[halves <= lead_max]:
        for kind in ("along-track", "radial"):
            best = None
            for sign in (+1, -1):
                def pc_of(dv):
                    sh = cw_shift(sign * dv, 0, t, n) if kind == "along-track" else cw_shift(0, sign * dv, t, n)
                    return _pc_with_shift(conj, sh, extra1, extra2)
                if pc_of(max_dv) > target:
                    continue
                lo, hi = 0.0, max_dv
                for _ in range(28):
                    mid = 0.5 * (lo + hi)
                    if pc_of(mid) > target:
                        lo = mid
                    else:
                        hi = mid
                dv = hi
                if best is None or dv < best[0]:
                    best = (dv, sign, pc_of(dv))
            if best:
                sh = cw_shift(best[1] * best[0], 0, t, n) if kind == "along-track" else cw_shift(0, best[1] * best[0], t, n)
                options.append(dict(burn_before_tca_h=t / 3600, orbits_before=t / period, kind=kind,
                                    dv_mm_s=best[0] * best[1] * 1e3, pc_after=best[2],
                                    shift_R_m=float(sh[0]), shift_T_m=float(sh[1])))
    options.sort(key=lambda o: abs(o["dv_mm_s"]))
    return dict(pc_before=pc0, target=target, period_min=period / 60, options=options,
                recommended=options[0] if options else None)


def side_effects(option, primary_events, n):
    """Re-evaluate the primary's OTHER screened encounters after the burn (their Pc may rise).
    primary_events: list of (Conjunction, seconds from burn to that TCA, extra1, extra2)."""
    out = []
    dv = option["dv_mm_s"] / 1e3
    for conj, t_after, e1, e2 in primary_events:
        if t_after <= 0:
            out.append(dict(pc_before=conj.pc(e1, e2), pc_after=conj.pc(e1, e2), affected=False))
            continue
        sh = cw_shift(dv, 0, t_after, n) if option["kind"] == "along-track" else cw_shift(0, dv, t_after, n)
        out.append(dict(pc_before=conj.pc(e1, e2), pc_after=_pc_with_shift(conj, sh, e1, e2), affected=True))
    return out
