"""Model-agnostic, adaptive conformal intervals for 72-hour density-ratio forecasts.

For a point forecast R_pred the published interval is
    [R_pred * exp(-q), R_pred * exp(+q)],
where q is the (1 - alpha_t) empirical quantile of recent verified log-errors
|log R72 - log R_pred| of the SAME model in the SAME regime (Mondrian conformal:
'storm' when Kp >= 5- in the past 24 h or a storm began < 48 h ago, else 'quiet').

alpha_t is updated online by adaptive conformal inference (Gibbs & Candes, 2021):
    alpha_{t+1} = alpha_t + gamma * (alpha - miss rate of day t),
which keeps long-run coverage at 1 - alpha even when the error distribution drifts
(e.g., through the solar cycle). Coverage is therefore checked, not assumed.
"""
import json
import numpy as np

from . import config as C

BUFFER = 20000          # about two months of verified outcomes per model and regime


def regime(kp_max24, h_since_storm):
    kp = np.nan_to_num(np.asarray(kp_max24, float), nan=0.0)
    hs = np.nan_to_num(np.asarray(h_since_storm, float), nan=240.0)
    return np.where((kp >= 14 / 3) | (hs < 48), "storm", "quiet")


class AdaptiveConformal:
    def __init__(self, coverage=C.COVERAGE, gamma=C.ACI_GAMMA):
        self.alpha = 1 - coverage
        self.gamma = gamma
        self.state = {}            # key "model|regime" -> {"alpha_t": float, "res": [floats]}

    def _s(self, model, reg):
        k = f"{model}|{reg}"
        if k not in self.state:
            self.state[k] = {"alpha_t": self.alpha, "res": []}
        return self.state[k]

    def q(self, model, reg):
        s = self._s(model, reg)
        if len(s["res"]) < 50:                     # not calibrated yet
            return np.nan
        a = float(np.clip(s["alpha_t"], 0.005, 0.3))
        r = np.asarray(s["res"])
        k = int(np.ceil((len(r) + 1) * (1 - a))) - 1
        return float(np.sort(r)[min(max(k, 0), len(r) - 1)])

    def interval(self, model, R_pred, regimes):
        R_pred = np.asarray(R_pred, float)
        qs = np.array([self.q(model, r) for r in regimes])
        return R_pred * np.exp(-qs), R_pred * np.exp(qs), qs

    def update(self, model, R_pred, R_true, regimes, lo, hi):
        """Feed one batch of verified outcomes (one day's worth).

        Outcomes verified on the same day share one atmosphere, so their misses are strongly
        correlated. alpha_t therefore takes ONE step per batch and regime, driven by that
        batch's miss rate, instead of one step per outcome (which overreacts to a single
        unusual day)."""
        miss = {}
        for rp, rt, rg, l, h in zip(R_pred, R_true, regimes, lo, hi):
            if not (np.isfinite(rp) and np.isfinite(rt) and rp > 0 and rt > 0):
                continue
            s = self._s(model, rg)
            if np.isfinite(l) and np.isfinite(h):
                miss.setdefault(rg, []).append(float(not (l <= rt <= h)))
            s["res"].append(float(abs(np.log(rt) - np.log(rp))))
            if len(s["res"]) > BUFFER:
                s["res"] = s["res"][-BUFFER:]
        for rg, m in miss.items():
            s = self._s(model, rg)
            s["alpha_t"] = float(np.clip(s["alpha_t"] + self.gamma * (self.alpha - np.mean(m)), 0.005, 0.3))

    def to_json(self):
        return json.dumps({"alpha": self.alpha, "gamma": self.gamma, "state": self.state})

    @classmethod
    def from_json(cls, text):
        js = json.loads(text)
        c = cls(1 - js["alpha"], js["gamma"])
        c.state = js["state"]
        return c
