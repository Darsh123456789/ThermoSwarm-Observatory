"""Calibrate the element-set error model on an archive of element sets (fit on half of the objects,
check on the other half) and write store/errmodel.json.
usage: python scripts/calibrate_errmodel.py <tles.parquet> <storms.csv>"""
import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from observatory import config as C
from observatory.errmodel import ErrorModel, measure, Z90

els = pd.read_parquet(sys.argv[1], columns=["norad", "jd", "l1", "l2"])
st = pd.read_csv(sys.argv[2], parse_dates=["onset", "end"])
storms = [(a, b + pd.Timedelta("1D")) for a, b in zip(st.onset, st.end)]
t = time.time()
m = measure(els, storms, every=10, max_pairs=600_000)
print(f"{len(m)} pairs from {m.norad.nunique()} objects in {time.time() - t:.0f} s")
ids = np.array(sorted(m.norad.unique()))
rng = np.random.default_rng(1)
fit_ids = set(rng.choice(ids, len(ids) // 2, replace=False))
mf, mt = m[m.norad.isin(fit_ids)], m[~m.norad.isin(fit_ids)]
em = ErrorModel.fit(mf)
# out-of-sample check
oos = {}
for (b, reg), g in mt.groupby(["band", "regime"]):
    if f"{b}|{reg}" not in em.coef:
        continue
    sig = np.array([em.sigmas(a, dt, reg) for a, dt in zip(g.alt, g.dt)])
    ins = (np.abs(g[["dR", "dT", "dN"]].to_numpy()) <= Z90 * sig).mean(axis=0)
    oos[f"{b}|{reg}"] = dict(n=int(len(g)), inside90_R=round(float(ins[0]), 3),
                             inside90_T=round(float(ins[1]), 3), inside90_N=round(float(ins[2]), 3))
em_all = ErrorModel.fit(m)
em_all.checks = dict(in_sample=em_all.checks, out_of_sample_half_split=oos,
                     data=dict(pairs=int(len(m)), objects=int(m.norad.nunique()),
                               period=[str(pd.to_datetime(m.jd0.min() - 2440587.5, unit="D").date()),
                                       str(pd.to_datetime(m.jd1.max() - 2440587.5, unit="D").date())]))
(C.STORE / "errmodel.json").write_text(em_all.to_json())
print(json.dumps(oos, indent=1))
for k, v in em_all.coef.items():
    s1 = em_all.sigmas(int(k.split("|")[0]) * 0 + [200, 400, 500, 600, 800, 1500, 3000][int(k.split("|")[0])], 1, k.split("|")[1])
    s3 = em_all.sigmas([200, 400, 500, 600, 800, 1500, 3000][int(k.split("|")[0])], 3, k.split("|")[1])
    print(k, "1d sigma km R/T/N", np.round(s1, 3), " 3d", np.round(s3, 3))
