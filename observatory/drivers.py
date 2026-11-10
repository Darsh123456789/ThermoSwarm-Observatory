"""Space-weather drivers: GFZ Kp/ap/F10.7 (observed) and NOAA SWPC 3-day Kp forecast.

Everything here is causal: F10.7A is a trailing 81-day mean (the paper used a centred
mean, which is not available in real time).
"""
import re
import numpy as np
import pandas as pd

GFZ_NOWCAST = "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_nowcast.txt"
SWPC_3DAY = "https://services.swpc.noaa.gov/text/3-day-forecast.txt"

# Kp (thirds) -> ap conversion (Bartels)
_KP_STEPS = np.arange(0, 28) / 3.0
_AP_STEPS = np.array([0, 2, 3, 4, 5, 6, 7, 9, 12, 15, 18, 22, 27, 32, 39, 48, 56, 67, 80, 94,
                      111, 132, 154, 179, 207, 236, 300, 400], float)


def kp_to_ap(kp):
    return np.interp(np.asarray(kp, float), _KP_STEPS, _AP_STEPS)


def parse_gfz(text):
    """Parse the GFZ 'Kp_ap_Ap_SN_F107' text format -> (three_hourly, daily) DataFrames."""
    rows = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if len(p) < 26:
            continue
        rows.append([int(p[0]), int(p[1]), int(p[2])] + [float(x) for x in p[7:15]]
                    + [float(x) for x in p[15:23]] + [float(p[23]), float(p[25])])
    cols = ["y", "m", "d"] + [f"kp{i}" for i in range(8)] + [f"ap{i}" for i in range(8)] + ["Ap", "f107"]
    d = pd.DataFrame(rows, columns=cols)
    d.index = pd.to_datetime(dict(year=d.y, month=d.m, day=d.d))
    d = d.replace(-1.0, np.nan)
    kp = d[[f"kp{i}" for i in range(8)]].to_numpy().ravel()
    ap = d[[f"ap{i}" for i in range(8)]].to_numpy().ravel()
    idx = pd.date_range(d.index[0], periods=len(kp), freq="3h")
    three = pd.DataFrame({"kp": kp, "ap": ap}, index=idx).dropna()
    daily = pd.DataFrame({"Ap": d.Ap, "f107": d.f107.where(d.f107 > 0)})
    daily["f107"] = daily.f107.interpolate().ffill()
    daily["f107a"] = daily.f107.rolling(81, min_periods=27).mean()          # trailing (causal)
    daily["Ap"] = daily.Ap.fillna(three.ap.resample("1D").mean())
    return three, daily.dropna(subset=["f107"])


def parse_swpc_3day(text, issued=None):
    """Parse NOAA SWPC '3-day forecast' -> 3-hourly predicted Kp as a Series (UTC index)."""
    lines = text.splitlines()
    k = next(i for i, l in enumerate(lines) if "Kp index breakdown" in l)
    hdr = lines[k + 2]
    m = re.search(r"Kp index breakdown\s+(\w{3})\s*(\d{1,2})-(\w{3})\s*(\d{1,2})\s+(\d{4})", lines[k])
    year = int(m.group(5))
    days = re.findall(r"([A-Z][a-z]{2})\s+(\d{1,2})", hdr)
    starts = [pd.Timestamp(f"{year}-{mon}-{int(day):02d}") for mon, day in days]
    # handle Dec -> Jan rollover
    for i in range(1, len(starts)):
        if starts[i] < starts[i - 1]:
            starts[i] = starts[i] + pd.DateOffset(years=1)
    out = {}
    for line in lines[k + 3:k + 11]:
        mm = re.match(r"\s*(\d{2})-(\d{2})UT\s+(.*)", line)
        if not mm:
            continue
        h0 = int(mm.group(1))
        vals = re.findall(r"(\d+(?:\.\d+)?)(?:\s*\(G\d\))?", mm.group(3))
        for s, v in zip(starts, vals):
            out[s + pd.Timedelta(hours=h0)] = float(v)
    return pd.Series(out).sort_index()


class Drivers:
    """NRLMSIS driver lookup (F10.7 of previous day, trailing F10.7A, 7-element ap array).

    `three` is 3-hourly ap (observed, optionally extended with forecast ap);
    `daily` has Ap, f107, f107a.
    """

    def __init__(self, three, daily):
        self.three, self.daily = three.sort_index(), daily.sort_index()
        self.t0 = self.three.index[0].value / 1e9
        self.ap3 = self.three.ap.to_numpy(float)
        self.d0 = self.daily.index[0].value / 1e9
        self.f107 = self.daily.f107.to_numpy(float)
        self.f107a = self.daily.f107a.bfill().to_numpy(float)
        self.Ap = self.daily.Ap.to_numpy(float)

    def __call__(self, unix):
        unix = np.asarray(unix, float)
        k = ((unix - self.t0) // 10800).astype(int)
        di = ((unix - self.d0) // 86400).astype(int)

        def A(j):
            return self.ap3[np.clip(j, 0, len(self.ap3) - 1)]

        aps = np.empty((len(unix), 7))
        aps[:, 0] = self.Ap[np.clip(di, 0, len(self.Ap) - 1)]
        for c, lag in zip(range(1, 5), range(0, 4)):
            aps[:, c] = A(k - lag)
        aps[:, 5] = np.mean([A(k - j) for j in range(4, 12)], axis=0)
        aps[:, 6] = np.mean([A(k - j) for j in range(12, 20)], axis=0)
        f107 = self.f107[np.clip(di - 1, 0, len(self.f107) - 1)]
        f107a = self.f107a[np.clip(di, 0, len(self.f107a) - 1)]
        return f107, f107a, aps

    def features(self, t):
        """Forecast drivers known at issue time t (all trailing): ap_now, ap_max24, kp_max24,
        f107, f107_dev, h_since_storm (hours since Kp >= 5- began, capped at 240)."""
        t = pd.Timestamp(t)
        past = self.three[self.three.index <= t]
        last24 = past[past.index > t - pd.Timedelta("24h")]
        d = self.daily[self.daily.index <= t.floor("D")]
        f107 = float(d.f107.iloc[-1]) if len(d) else np.nan
        f107a = float(d.f107a.iloc[-1]) if len(d) else np.nan
        storm = past[past.kp >= 14 / 3]
        h_since = 240.0 if storm.empty else min(240.0, (t - storm.index[-1]).total_seconds() / 3600)
        return dict(ap_now=float(past.ap.iloc[-1]) if len(past) else np.nan,
                    ap_max24=float(last24.ap.max()) if len(last24) else np.nan,
                    kp_max24=float(last24.kp.max()) if len(last24) else np.nan,
                    f107=f107, f107_dev=f107 - f107a, h_since_storm=h_since)


def extend_with_forecast(three, kp_fc):
    """Append SWPC forecast Kp (converted to ap) after the last observed value."""
    fc = pd.DataFrame({"kp": kp_fc.values, "ap": kp_to_ap(kp_fc.values)}, index=kp_fc.index)
    fc = fc[fc.index > three.index[-1]]
    return pd.concat([three, fc]).sort_index()
