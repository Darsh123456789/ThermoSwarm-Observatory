"""Forecast models for the 72-hour density ratio R72 of each object.

Every model maps a table of issue-time state (one row per object) to a point prediction
R_pred. Uncertainty is added afterwards, identically for all models, by the conformal
layer (conformal.py), so models compete on equal terms.

To add a model: subclass Model, give it a unique `name`, and add it to `registry()`.
Only information available at issue time may be used (columns listed in STATE_COLUMNS).
"""
import json
import numpy as np
import pandas as pd

STATE_COLUMNS = ["alt", "R_last", "R_base7", "ap_now", "ap_max24", "kp_max24",
                 "f107", "f107_dev", "h_since_storm"]


class Model:
    name = "base"
    description = ""
    trainable = False

    def fit(self, train):          # train: verified windows with STATE_COLUMNS and R72
        return self

    def predict(self, state):
        raise NotImplementedError


class NRLMSIS(Model):
    name = "nrlmsis-2.1"
    description = "Empirical model as is (R = 1 relative to its own quiet-time level)."

    def predict(self, state):
        return np.ones(len(state))


class Persistence(Model):
    name = "persistence"
    description = "The object's most recent measured ratio persists."

    def predict(self, state):
        return state.R_last.to_numpy(float)


class Baseline7(Model):
    name = "baseline-7d"
    description = "The object's 7-day median ratio: the slow error component only."

    def predict(self, state):
        return state.R_base7.fillna(state.R_last).to_numpy(float)


class ThermoSwarmGB(Model):
    name = "thermoswarm-gb"
    description = ("Gradient boosting on log R72 from the object's recent ratios, altitude and "
                   "issue-time drivers (ap, Kp, F10.7, time since storm onset). Retrained causally.")
    trainable = True

    def __init__(self, params=None):
        self.params = params or dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, objective="reg:absoluteerror", random_state=7)
        self.model = None

    def fit(self, train):
        import xgboost as xgb
        d = train.dropna(subset=STATE_COLUMNS + ["R72"])
        d = d[(d.R72 > 0.2) & (d.R72 < 5)]
        if len(d) < 2000:
            self.model = None
            return self
        self.model = xgb.XGBRegressor(**self.params).fit(d[STATE_COLUMNS], np.log(d.R72))
        return self

    def predict(self, state):
        if self.model is None:                      # not enough history yet: fall back
            return Persistence().predict(state)
        return np.exp(self.model.predict(state[STATE_COLUMNS]))

    def save(self, path):
        if self.model is not None:
            self.model.save_model(str(path))

    def load(self, path):
        import xgboost as xgb
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(path))
        return self


class Submitted(Model):
    """A forecast submitted as a file by an outside team (see SUBMITTING.md)."""

    def __init__(self, name, table):
        self.name, self.table = name, table
        self.description = "External submission."

    def predict(self, state):
        m = state[["norad"]].merge(self.table[["norad", "R_pred"]], on="norad", how="left")
        return m.R_pred.to_numpy(float)


def registry():
    return [NRLMSIS(), Persistence(), Baseline7(), ThermoSwarmGB()]


def load_submissions(folder, day):
    """Read submissions/<team>/<YYYY-MM-DD>.json -> list of Submitted models."""
    out = []
    for p in sorted(folder.glob(f"*/{pd.Timestamp(day):%Y-%m-%d}.json")):
        try:
            js = json.loads(p.read_text())
            t = pd.DataFrame(js["forecasts"])[["norad", "R_pred"]]
            t = t[np.isfinite(t.R_pred) & (t.R_pred > 0)]
            out.append(Submitted(f"submitted:{p.parent.name}", t))
        except Exception as e:                      # malformed submissions are skipped, not fatal
            print(f"skipping {p}: {e}")
    return out
