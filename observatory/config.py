"""Shared settings for the ThermoSwarm Observatory."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("OBSERVATORY_ROOT", Path(__file__).resolve().parents[1]))
STORE = ROOT / "store"            # derived, shareable data (densities, drivers) - committed
CACHE = ROOT / ".cache"           # raw element sets - never committed (Space-Track terms)
REGISTRY = ROOT / "registry"      # hash-chained forecast registry - committed
DOCS = ROOT / "docs"              # GitHub Pages site and JSON feed
REPLAY = ROOT / "replay"          # backtest outputs (clearly labelled, never mixed with live)

# Object selection (same as the ThermoSwarm paper)
PERIGEE_MIN_KM, APOGEE_MAX_KM, ECC_MAX = 200.0, 800.0, 0.01

# Density inversion
PAIR_MIN_DAYS, PAIR_MAX_DAYS = 1.0, 3.0         # TLE pair spacing for orbit-averaged density
NSAMP_INTERVAL = 96                              # integration points per interval
QUIET_AP = 15                                    # quiet = max ap <= 15 over interval and 24 h before
CAL_WINDOW_DAYS = 30                             # trailing window for causal B calibration
CAL_MIN_QUIET = 8
SIGR_MAX = 0.5

# Forecasting
HORIZON_H = 72
WINDOW_MIN_H, WINDOW_MAX_H = 66, 78              # TLE pairs that verify a 72-h forecast
NSAMP_WINDOW = 144
BANDS = [(lo, lo + 50) for lo in range(200, 800, 50)]   # 50-km altitude bands
COVERAGE = 0.90                                  # nominal coverage of published intervals
ACI_GAMMA = 0.05                                 # adaptive-conformal step per daily batch

# Collision-risk categories (configurable; not official thresholds)
PC_HIGH, PC_MEDIUM = 1e-4, 1e-6

# Physical constants (WGS-84)
RE_KM, FLAT, OMEGA_E = 6378.137, 1 / 298.257223563, 7.2921159e-5
MU = 398600.4418

for p in (STORE, CACHE, REGISTRY, DOCS / "feed", REPLAY):
    p.mkdir(parents=True, exist_ok=True)


def band_of(alt_km):
    """Index of the 50-km band containing alt_km, or -1."""
    for k, (lo, hi) in enumerate(BANDS):
        if lo <= alt_km < hi:
            return k
    return -1


def band_label(k):
    lo, hi = BANDS[k]
    return f"{lo}-{hi} km"


def clean_json(o):
    """Replace NaN/inf by None so feeds are strict, standard JSON."""
    import math
    if isinstance(o, dict):
        return {k: clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            o = o.item()
        except (ValueError, TypeError):
            pass
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o
