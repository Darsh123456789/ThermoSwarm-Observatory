"""Public referee: forecasts are committed before the truth exists, then scored automatically.

Registry layout (per stream: 'live' in registry/, 'backtest' in replay/registry/):
  forecasts/YYYY-MM-DD.csv.gz   every model's forecasts issued that day (deterministic gzip)
  chain.jsonl                   one line per day: file SHA-256 and a hash chained to the
                                previous line, so no past forecast can be altered unnoticed.
  scores.csv.gz                 verified forecast-outcome pairs
A forecast for object k issued from its element set at epoch t0 targets the 72-h window
starting at t0; it is verified when an element set 66-78 h later arrives. Forecasts use
only information up to t0, and are written before that later element set exists.
"""
import gzip
import hashlib
import io
import json
import numpy as np
import pandas as pd

from . import config as C
from .physics import along_track_error_km

FORECAST_COLS = ["model", "norad", "t0", "jd1", "alt", "band", "regime", "B", "a_km",
                 "R_pred", "lo", "hi"]


def _gz_bytes(df):
    raw = df.to_csv(index=False, float_format="%.13g").encode()   # keeps epochs to ~0.1 s
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as g:
        g.write(raw)
    return buf.getvalue()


class Registry:
    def __init__(self, root, stream="live"):
        self.root = root
        self.stream = stream
        (root / "forecasts").mkdir(parents=True, exist_ok=True)
        self.chain_path = root / "chain.jsonl"
        self.scores_path = root / "scores.csv.gz"

    # ------------------------------------------------------------ commit
    def chain(self):
        if not self.chain_path.exists():
            return []
        return [json.loads(l) for l in self.chain_path.read_text().splitlines() if l.strip()]

    def commit(self, day, forecasts, created_utc=None):
        day = pd.Timestamp(day).strftime("%Y-%m-%d")
        f = forecasts[FORECAST_COLS].sort_values(["model", "norad"]).reset_index(drop=True)
        data = _gz_bytes(f)
        path = self.root / "forecasts" / f"{day}.csv.gz"
        if path.exists():
            raise FileExistsError(f"forecasts for {day} already committed; registry is append-only")
        path.write_bytes(data)
        chain = self.chain()
        prev = chain[-1]["hash"] if chain else "0" * 64
        fsha = hashlib.sha256(data).hexdigest()
        h = hashlib.sha256(f"{prev}|{day}|{fsha}|{self.stream}".encode()).hexdigest()
        rec = dict(day=day, file=f"forecasts/{day}.csv.gz", sha256=fsha, prev=prev, hash=h,
                   stream=self.stream, n=len(f), models=sorted(f.model.unique().tolist()),
                   created_utc=created_utc or pd.Timestamp.now("UTC").isoformat())
        with self.chain_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec

    def verify(self):
        """Recompute every file hash and the chain. Returns (ok, message)."""
        prev = "0" * 64
        for k, rec in enumerate(self.chain()):
            data = (self.root / rec["file"]).read_bytes()
            fsha = hashlib.sha256(data).hexdigest()
            if fsha != rec["sha256"]:
                return False, f"line {k + 1}: {rec['file']} was modified after commit"
            h = hashlib.sha256(f"{prev}|{rec['day']}|{fsha}|{rec['stream']}".encode()).hexdigest()
            if rec["prev"] != prev or rec["hash"] != h:
                return False, f"line {k + 1}: chain broken"
            prev = h
        return True, f"{len(self.chain())} daily commits verified"

    def forecasts(self, since=None):
        out = []
        for rec in self.chain():
            if since is not None and rec["day"] < str(since)[:10]:
                continue
            out.append(pd.read_csv(self.root / rec["file"], parse_dates=["t0"]))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=FORECAST_COLS)

    # ------------------------------------------------------------ scoring
    def scores(self):
        if self.scores_path.exists():
            return pd.read_csv(self.scores_path, parse_dates=["t0", "verified_at"])
        return pd.DataFrame()

    def score(self, windows, now):
        """Match committed forecasts to verification windows that ended by `now`.
        windows: norad, jd1, jd2, f72, J, a_km (and R72 is f72/B of the forecast)."""
        fc = self.forecasts()
        if fc.empty or windows.empty:
            return pd.DataFrame()
        old = self.scores()
        w = windows.copy()
        w["verified_at"] = pd.to_datetime(w.jd2 - 2440587.5, unit="D")
        w = w[w.verified_at <= pd.Timestamp(now)]
        fc["k"] = fc.norad.astype(int).astype(str) + "_" + fc.jd1.round(5).astype(str)
        w["k"] = w.norad.astype(int).astype(str) + "_" + w.jd1.round(5).astype(str)
        m = fc.merge(w[["k", "jd2", "f72", "J", "verified_at"]], on="k")
        if len(old):
            done = set(zip(old.model, old.k))
            m = m[[(a, b) not in done for a, b in zip(m.model, m.k)]]
        if m.empty:
            return m
        m = score_pairs(m)
        allm = pd.concat([old, m], ignore_index=True) if len(old) else m
        allm.to_csv(self.scores_path, index=False, compression="gzip")
        return m


def score_pairs(m):
    """Add outcome metrics to forecast rows that carry f72 and J of their verification window."""
    m = m.copy()
    m["R72"] = m.f72 / m.B
    m["e_km"] = along_track_error_km(m.a_km, m.B, m.R_pred, m.f72, m.J)
    m["lr"] = np.abs(np.log(m.R_pred) - np.log(m.R72))
    m["covered"] = (m.lo <= m.R72) & (m.R72 <= m.hi)
    with np.errstate(invalid="ignore", divide="ignore"):
        m["interval_score"] = interval_score(np.log(m.lo), np.log(m.hi), np.log(m.R72), 1 - C.COVERAGE)
    return m


def interval_score(lo, hi, y, alpha):
    """Winkler/Gneiting interval score (lower is better); NaN where no interval was issued."""
    lo, hi, y = map(lambda a: np.asarray(a, float), (lo, hi, y))
    return (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)


def leaderboard(scores, since=None, regime=None):
    s = scores.copy()
    if since is not None:
        s = s[s.verified_at >= pd.Timestamp(since)]
    if regime is not None:
        s = s[s.regime == regime]
    if s.empty:
        return pd.DataFrame()
    ref = s[s.model == "nrlmsis-2.1"].set_index("k").e_km.abs()
    rows = []
    for mname, g in s.groupby("model"):
        ae = g.e_km.abs()
        common = g.set_index("k").e_km.abs().reindex(ref.index).dropna()
        base = ref.reindex(common.index)
        rows.append(dict(model=mname, n=len(g), objects=int(g.norad.nunique()),
                         median_err_km=float(ae.median()), rms_err_km=float(np.sqrt((g.e_km ** 2).mean())),
                         p90_err_km=float(ae.quantile(0.9)), mae_logR=float(g.lr.mean()),
                         coverage=float(g.covered[np.isfinite(g.interval_score)].mean())
                         if np.isfinite(g.interval_score).any() else np.nan,
                         interval_score=float(np.nanmean(g.interval_score)),
                         skill_median=float(1 - common.median() / base.median()) if len(common) else np.nan,
                         skill_rms=float(1 - np.sqrt((common ** 2).mean() / (base ** 2).mean()))
                         if len(common) else np.nan))
    return pd.DataFrame(rows).sort_values("rms_err_km").reset_index(drop=True)
