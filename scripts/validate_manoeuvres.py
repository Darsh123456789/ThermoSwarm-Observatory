"""Validate storm-aware manoeuvre detection on an archive of element sets of objects that did not
manoeuvre (QC removed rises > 0.3 km): false-alarm rates in storm and quiet conditions, and recall of
synthetic burns injected into the element history. Compares against a naive detector (no swarm index).
usage: python scripts/validate_manoeuvres.py <tles.parquet> <storms.csv>"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from observatory import config as C
from observatory.anomalies import decay_rates, _object_baselines, detect_manoeuvres

t = pd.read_parquet(sys.argv[1], columns=["norad", "jd", "a_km", "incl", "ecc"])
st = pd.read_csv(sys.argv[2], parse_dates=["onset", "end"])
def stormy(day):
    m = np.zeros(len(day), bool)
    for a, b in zip(st.onset, st.end):
        m |= (day >= a.floor("D")) & (day <= (b + pd.Timedelta("1D")).floor("D"))
    return m
def rates(df):
    r = _object_baselines(decay_rates(df)); r.index = pd.RangeIndex(len(r)); return r
r = rates(t)
out = dict(data=dict(objects=int(t.norad.nunique()), element_sets=int(len(t)), pairs=int(len(r)),
                     period=[str(pd.to_datetime(t.jd.min() - 2440587.5, unit="D").date()),
                             str(pd.to_datetime(t.jd.max() - 2440587.5, unit="D").date())]),
           false_alarms={}, recall={})
for name, sw in (("swarm-aware", True), ("naive", False)):
    ev, sc = detect_manoeuvres(None, use_swarm=sw, rates=r)
    sc = sc[np.isfinite(sc.expected)]; ss = stormy(sc.day)
    evs = stormy(pd.to_datetime(0.5 * (ev.jd1 + ev.jd2) - 2440587.5, unit="D").dt.floor("D")) if len(ev) else np.array([], bool)
    out["false_alarms"][name] = dict(total=int(len(ev)), storm=int(evs.sum()), quiet=int((~evs).sum()),
                                     per_1000_checks_storm=round(1000 * evs.sum() / ss.sum(), 2),
                                     per_1000_checks_quiet=round(1000 * (~evs).sum() / (~ss).sum(), 2))
rng = np.random.default_rng(5)
objs = rng.choice(r.norad.unique(), 300, replace=False)
for size in (0.2, 0.5, 1.0, -0.5):
    tt = t.copy(); truth = {}
    for o in objs:
        g = tt[tt.norad == o]
        if len(g) < 40: continue
        jb = rng.uniform(g.jd.quantile(0.3), g.jd.quantile(0.9)); truth[o] = jb
        tt.loc[(tt.norad == o) & (tt.jd > jb), "a_km"] += size
    rr = rates(tt)
    for name, sw in (("swarm-aware", True), ("naive", False)):
        ev, _ = detect_manoeuvres(None, use_swarm=sw, rates=rr)
        hit = sum(((ev.norad == o) & (ev.jd1 <= jb + 0.01) & (ev.jd2 >= jb - 0.01)).any() for o, jb in truth.items())
        out["recall"].setdefault(f"{size:+.1f} km", {})[name] = round(hit / len(truth), 3)
(C.STORE / "validation_manoeuvres.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
