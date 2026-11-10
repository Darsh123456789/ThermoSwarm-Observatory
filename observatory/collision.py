"""Storm-aware collision probability.

Standard short-encounter probability of collision (Pc): the 2-D Gaussian of the relative
position, projected on the encounter plane (perpendicular to the relative velocity),
integrated over the hard-body disk.

ThermoSwarm adds the drag uncertainty that screening systems usually treat as fixed:
for each object, the 90% conformal interval of its density ratio over the lead time is
converted into an along-track position uncertainty (and, for the model's own expected
bias, an along-track shift). The result is a storm-aware Pc RANGE:
  Pc_nominal  - as screened (input covariance, no density correction)
  Pc_storm    - covariance inflated by drag uncertainty, mean shifted by the predicted bias
  Pc_max      - the highest Pc over along-track shifts inside the drag interval
A conjunction whose category changes between these is 'storm-sensitive'.
"""
import re
import numpy as np
import pandas as pd

from . import config as C

Z90 = 1.6448536


# ---------------------------------------------------------------- frames
def rtn_basis(r, v):
    """Rows = unit vectors R, T, N (radial, along-track/in-track, orbit normal) in ECI."""
    R = r / np.linalg.norm(r)
    N = np.cross(r, v)
    N /= np.linalg.norm(N)
    T = np.cross(N, R)
    return np.vstack([R, T, N])


def encounter_basis(dr, dv):
    """Rows = x (miss direction in the plane), y, z (relative velocity)."""
    z = dv / np.linalg.norm(dv)
    x = dr - np.dot(dr, z) * z
    if np.linalg.norm(x) < 1e-9:
        x = np.cross(z, [1.0, 0, 0]) if abs(z[0]) < 0.9 else np.cross(z, [0, 1.0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z])


# ---------------------------------------------------------------- Pc
_GL_R = np.polynomial.legendre.leggauss(64)
_GL_T = np.polynomial.legendre.leggauss(128)


SMALL_DISK = 0.05      # hbr / smallest sigma below which the closed form is used


def pc_2d(miss_xy, cov_xy, hbr):
    """Probability that the 2-D Gaussian N(miss_xy, cov_xy) falls inside a disk of radius hbr
    centred on the origin (all in metres). Gauss-Legendre quadrature in polar coordinates; when the
    disk is tiny compared with the uncertainty (hbr < 0.05 sigma_min) and the density changes little
    across it (hbr |S^-1 m| < 0.1), the usual case for public element sets, the density is constant across it to second order and the closed form
    pi hbr^2 N(0; miss, cov) is used (relative error < ~0.5%)."""
    mu = np.asarray(miss_xy, float)
    S = np.asarray(cov_xy, float)
    Si = np.linalg.inv(S)
    det = np.linalg.det(S)
    lam_min = 0.5 * (S[0, 0] + S[1, 1]) - np.sqrt(0.25 * (S[0, 0] - S[1, 1]) ** 2 + S[0, 1] ** 2)
    g = Si @ mu                                   # gradient of the log-density at the disk centre
    if lam_min > 0 and hbr < SMALL_DISK * np.sqrt(lam_min) and hbr * np.linalg.norm(g) < 0.1:
        q = mu @ g
        return float(hbr ** 2 / (2 * np.sqrt(det)) * np.exp(-0.5 * q))
    xr, wr = _GL_R
    xt, wt = _GL_T
    rr = 0.5 * hbr * (xr + 1)
    th = np.pi * (xt + 1)
    Rg, Tg = np.meshgrid(rr, th, indexing="ij")
    P = np.stack([Rg * np.cos(Tg), Rg * np.sin(Tg)], -1) - mu
    q = np.einsum("...i,ij,...j->...", P, Si, P)
    dens = np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(det))
    W = np.outer(wr * 0.5 * hbr, wt * np.pi)
    return float(np.sum(W * dens * Rg))


def category(pc):
    return "high" if pc >= C.PC_HIGH else "medium" if pc >= C.PC_MEDIUM else "low"


# ---------------------------------------------------------------- conjunction model
class Conjunction:
    """Two objects at TCA: ECI states (m, m/s), 3x3 RTN covariances (m^2), hard-body radius (m)."""

    def __init__(self, r1, v1, cov1_rtn, r2, v2, cov2_rtn, hbr, tca=None, lead_h=72.0,
                 alt1=None, alt2=None, ids=("1", "2")):
        self.r1, self.v1, self.r2, self.v2 = map(lambda a: np.asarray(a, float), (r1, v1, r2, v2))
        self.c1, self.c2 = np.asarray(cov1_rtn, float), np.asarray(cov2_rtn, float)
        self.hbr, self.tca, self.lead_h, self.ids = float(hbr), tca, float(lead_h), ids
        self.alt1 = alt1 if alt1 is not None else np.linalg.norm(self.r1) / 1e3 - C.RE_KM
        self.alt2 = alt2 if alt2 is not None else np.linalg.norm(self.r2) / 1e3 - C.RE_KM

    def _projected(self, extra1=(0.0, 0.0), extra2=(0.0, 0.0)):
        """extra = (added along-track variance m^2, along-track shift m) per object."""
        dr, dv = self.r2 - self.r1, self.v2 - self.v1
        E = encounter_basis(dr, dv)
        B1, B2 = rtn_basis(self.r1, self.v1), rtn_basis(self.r2, self.v2)
        c1 = self.c1.copy(); c1[1, 1] += extra1[0]
        c2 = self.c2.copy(); c2[1, 1] += extra2[0]
        cov = B1.T @ c1 @ B1 + B2.T @ c2 @ B2                    # ECI
        shift = extra2[1] * B2[1] - extra1[1] * B1[1]            # along-track shifts move each object
        miss = E @ (dr + shift)
        cxy = (E @ cov @ E.T)[:2, :2]
        return miss[:2], cxy

    def pc(self, extra1=(0.0, 0.0), extra2=(0.0, 0.0)):
        m, S = self._projected(extra1, extra2)
        return pc_2d(m, S, self.hbr)

    def miss_m(self):
        return float(np.linalg.norm(self.r2 - self.r1))


def drag_terms(alt_km, lead_h, band_terms):
    """(sigma_m, shift_m, bound_m) of drag-induced along-track position for an object at alt_km,
    scaled from the 72-h band values ~ (lead/72)^2 (error grows with the square of time)."""
    k = C.band_of(alt_km)
    if k < 0 or k not in band_terms:
        return 0.0, 0.0, 0.0
    b = band_terms[k]
    s = (lead_h / C.HORIZON_H) ** 2
    bound = b["bound72_km"] * 1e3 * s
    shift = b["shift72_km"] * 1e3 * s
    if not np.isfinite(bound):
        return 0.0, 0.0, 0.0          # no calibrated interval yet: do not invent uncertainty
    return bound / Z90, (shift if np.isfinite(shift) else 0.0), bound


def storm_aware(conj, band_terms, n_scan=21):
    """Pc nominal / storm / max over the drag interval, and the category of each."""
    s1, sh1, b1 = drag_terms(conj.alt1, conj.lead_h, band_terms)
    s2, sh2, b2 = drag_terms(conj.alt2, conj.lead_h, band_terms)
    pc0 = conj.pc()
    pcs = conj.pc((s1 ** 2, sh1), (s2 ** 2, sh2))
    best = pcs
    for u in np.linspace(-1, 1, n_scan):                    # same-sign shifts (same atmosphere)
        for w in (-1.0, 1.0):
            best = max(best, conj.pc((0.0, u * b1), (0.0, w * u * b2)))
    return dict(pc_nominal=pc0, pc_storm=pcs, pc_max=best, cat_nominal=category(pc0),
                cat_storm=category(pcs), cat_max=category(best),
                storm_sensitive=category(best) != category(pc0) or category(pcs) != category(pc0),
                drag_bound1_m=b1, drag_bound2_m=b2)


RANK = {"low": 0, "medium": 1, "high": 2}


def classify(res):
    """Split category changes into the two effects drag uncertainty has on Pc.
    escalated: the storm Pc or the worst case across the drag interval reaches a HIGHER category
               than screened (the screen understated the risk).
    diluted:   the storm Pc falls to a LOWER category because the position is too uncertain
               (probability dilution: a low Pc then means 'not known well enough', not 'safe')."""
    r = res.copy()
    nom = r.cat_nominal.map(RANK)
    up = np.maximum(r.cat_storm.map(RANK), r.cat_max.map(RANK))
    r["escalated"] = up > nom
    r["diluted"] = (r.cat_storm.map(RANK) < nom) & ~r.escalated
    return r


# ---------------------------------------------------------------- CCSDS CDM (KVN) reader
def read_cdm(text):
    """Minimal CCSDS 508.0 KVN conjunction data message reader -> Conjunction."""
    head, objs, cur = {}, [], None
    for line in text.splitlines():
        line = line.split("COMMENT")[0].strip()
        m = re.match(r"([A-Z0-9_]+)\s*=\s*([^\[]*)", line)
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip()
        if k == "OBJECT":
            cur = {}
            objs.append(cur)
            continue
        (cur if cur is not None else head)[k] = v
    if len(objs) != 2:
        raise ValueError("CDM must contain two OBJECT sections")

    def state(o):
        r = np.array([float(o[k]) for k in ("X", "Y", "Z")]) * 1e3
        v = np.array([float(o[k]) for k in ("X_DOT", "Y_DOT", "Z_DOT")]) * 1e3
        c = np.array([[float(o["CR_R"]), float(o["CT_R"]), float(o["CN_R"])],
                      [float(o["CT_R"]), float(o["CT_T"]), float(o["CN_T"])],
                      [float(o["CN_R"]), float(o["CN_T"]), float(o["CN_N"])]])
        return r, v, c

    r1, v1, c1 = state(objs[0])
    r2, v2, c2 = state(objs[1])
    tca = pd.Timestamp(head.get("TCA")) if "TCA" in head else None
    created = pd.Timestamp(head["CREATION_DATE"]) if "CREATION_DATE" in head else None
    lead = (tca - created).total_seconds() / 3600 if tca is not None and created is not None else 72.0
    hbr = float(head.get("HBR", objs[0].get("HBR", 10.0)) or 10.0)
    ids = (objs[0].get("OBJECT_DESIGNATOR", "1"), objs[1].get("OBJECT_DESIGNATOR", "2"))
    return Conjunction(r1, v1, c1, r2, v2, c2, hbr, tca=tca, lead_h=max(lead, 1.0), ids=ids)


# ---------------------------------------------------------------- simulated conjunctions
def simulate(n, altitudes_km, seed=0):
    """Synthetic LEO encounters with realistic geometry and covariances (clearly labelled
    'simulated' wherever used). Miss distances are log-uniform 30 m - 3 km."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        h = float(rng.choice(altitudes_km)) + rng.normal(0, 5)
        r = (C.RE_KM + h) * 1e3
        vc = np.sqrt(C.MU * 1e9 / r)
        inc1, inc2 = np.radians(rng.uniform(40, 100, 2))
        raan = rng.uniform(0, 2 * np.pi)
        pos = np.array([r, 0, 0])
        v1 = vc * np.array([0, np.cos(inc1), np.sin(inc1)])
        v2 = vc * np.array([0, np.cos(inc2 + raan * 0.3), np.sin(inc2 + raan * 0.3)])
        if np.linalg.norm(v2 - v1) < 200:
            v2 = vc * np.array([0, -np.cos(inc2), np.sin(inc2)])
        miss = 10 ** rng.uniform(np.log10(30), np.log10(3000))
        E = encounter_basis(np.array([1.0, 0, 0]), v2 - v1)
        ang = rng.uniform(0, 2 * np.pi)
        d = miss * (np.cos(ang) * E[0] + np.sin(ang) * E[1])
        lead = rng.uniform(24, 96)
        def cov():
            sr, st, sn = rng.uniform(10, 60), rng.uniform(60, 600) * lead / 72, rng.uniform(10, 120)
            return np.diag([sr ** 2, st ** 2, sn ** 2])
        out.append(Conjunction(pos, v1, cov(), pos + d, v2, cov(), hbr=rng.uniform(5, 20),
                               lead_h=lead, alt1=h, alt2=h, ids=(f"SIM-{k}a", f"SIM-{k}b")))
    return out
