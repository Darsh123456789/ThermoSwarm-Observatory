"""Validate re-entry predictions and calibrated windows on past re-entries.
usage: python scripts/validate_reentry.py <reentry_cases.parquet (norad, jd_dec)> <tles.parquet> [<tles2.parquet> ...]"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from observatory import config as C
from observatory.anomalies import reentry_state, remaining_days, ReentryWindows

cols = ["norad", "jd", "a_km", "ecc", "n"]
t = pd.concat([pd.read_parquet(f, columns=cols) for f in sys.argv[2:]]).drop_duplicates(["norad", "jd"])
t = t[t.ecc < 0.02].sort_values(["norad", "jd"])
truth = pd.read_parquet(sys.argv[1])[["norad", "jd_dec"]].drop_duplicates("norad").set_index("norad").jd_dec
rows = []
for norad, g in t.groupby("norad"):
    if norad not in truth.index:
        continue
    jd_dec = truth[norad]
    for lead in (1, 2, 3, 5, 7, 10, 14):
        h = g[g.jd <= jd_dec - lead]
        if len(h) < 3 or jd_dec - h.jd.max() > lead + 1.0:
            continue
        st = reentry_state(h)
        if st is None:
            continue
        d = remaining_days(st["alt"], st["rate"])
        true_left = jd_dec - st["jd"]
        if np.isfinite(d) and d > 0:
            rows.append(dict(norad=int(norad), lead=lead, alt=st["alt"], pred=d, true=true_left,
                             lr=np.log(true_left / d)))
v = pd.DataFrame(rows)
ids = np.array(sorted(v.norad.unique()))
rng = np.random.default_rng(2)
cover = []
for rep in range(50):                                     # repeated half splits by object
    fit = set(rng.choice(ids, len(ids) // 2, replace=False))
    w = ReentryWindows.fit(v[v.norad.isin(fit)].lr)
    te = v[~v.norad.isin(fit)]
    cover.append(((te.lr >= w.q_lo) & (te.lr <= w.q_hi)).mean())
w = ReentryWindows.fit(v.lr)
(C.STORE / "reentry_windows.json").write_text(w.to_json())
by = v.groupby("lead").apply(lambda g: pd.Series(dict(n=len(g), median_abs_err_days=float(np.median(np.abs(g.pred - g.true))),
                                                       median_abs_err_pct=float(100 * np.median(np.abs(g.pred - g.true) / g.true)))))
summary = dict(objects=int(v.norad.nunique()), predictions=int(len(v)),
               window90_log=[w.q_lo, w.q_hi], window90_factor=[float(np.exp(w.q_lo)), float(np.exp(w.q_hi))],
               out_of_sample_coverage_mean=float(np.mean(cover)), out_of_sample_coverage_range=[float(np.min(cover)), float(np.max(cover))],
               by_lead=by.reset_index().to_dict("records"))
(C.STORE / "reentry_validation.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
