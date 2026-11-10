"""Static website (GitHub Pages, docs/) and machine-readable feed. No external scripts."""
import html
import json

import numpy as np
import pandas as pd

from . import config as C
from .referee import Registry, leaderboard
from .collision import classify

CSS = """
:root{--bg:#fbfaf7;--fg:#1d2430;--muted:#5d6878;--line:#e3e0d8;--card:#ffffff;--accent:#1f5fae;
--hi:#b4321f;--lo:#2361a8;--ok:#2a7a3d;--warn:#a8650a;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#12161c;--fg:#e7eaee;--muted:#9aa6b4;
--line:#2a313b;--card:#181d24;--accent:#6ea8ff;--hi:#ff7a63;--lo:#6aa6ff;--ok:#6cc27f;--warn:#e7a948}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);line-height:1.5}
main{max-width:1080px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.7rem;margin:.2rem 0}h2{font-size:1.15rem;margin:2.2rem 0 .6rem}
.muted{color:var(--muted)}.badge{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.78rem;
font-weight:600;letter-spacing:.04em}.live{background:var(--ok);color:#fff}.backtest{background:var(--warn);color:#fff}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:12px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{padding:6px 8px;border-bottom:1px solid var(--line);
text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600}
.flag-HIGH{color:var(--hi);font-weight:700}.flag-WATCH,.flag-ELEVATED,.flag-MIXED{color:var(--warn);font-weight:600}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
.kpi b{display:block;font-size:1.5rem}svg text{fill:var(--muted);font-size:11px}code{font-size:.85em}
a{color:var(--accent)}
"""


def _esc(x):
    return html.escape(str(x))


def _fmt(x, nd=2, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{100 * x:+.0f}%" if pct else f"{x:.{nd}f}"


def heatmap_svg(bmap, days=None, w=1000, h=300):
    m = bmap.copy()
    m["day"] = pd.to_datetime(m.day)
    if days:
        m = m[m.day >= m.day.max() - pd.Timedelta(days=days)]
    d0, d1 = m.day.min(), m.day.max()
    nd = max((d1 - d0).days + 1, 1)
    bands = sorted(m.band.unique())
    left, top, bh = 70, 10, (h - 40) / max(len(bands), 1)
    cw = (w - left - 10) / nd
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Daily model bias by altitude">']

    def col(r):
        x = float(np.clip(np.log(r) / np.log(1.5), -1, 1))
        if x >= 0:
            return f"rgb({int(255 - 75 * x)},{int(255 - 205 * x)},{int(255 - 215 * x)})"
        x = -x
        return f"rgb({int(255 - 220 * x)},{int(255 - 150 * x)},{int(255 - 70 * x)})"

    for k, b in enumerate(bands[::-1]):
        y = top + k * bh
        out.append(f'<text x="{left - 6}" y="{y + bh * 0.65:.1f}" text-anchor="end">{C.BANDS[b][0]}</text>')
        for _, r in m[m.band == b].iterrows():
            x = left + (r.day - d0).days * cw
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw + 0.4:.2f}" height="{bh - 1:.1f}" fill="{col(r.R)}">'
                       f'<title>{r.day:%Y-%m-%d} {C.band_label(b)}: R={r.R:.2f} (n={r.n})</title></rect>')
    for t in pd.date_range(d0.normalize() + pd.offsets.MonthBegin(0), d1, freq="MS"):
        x = left + (t - d0).days * cw
        out.append(f'<text x="{x:.0f}" y="{h - 12}">{t:%b %y}</text>')
    out.append(f'<text x="8" y="{top + 10}">km</text></svg>')
    return "".join(out)


def line_svg(series, w=1000, h=220, ymin=0.6, ymax=1.0, target=None, labels=None):
    series = {k: s.dropna() for k, s in series.items() if len(s.dropna())}
    if not series:
        return "<p class='muted'>Error bars are still calibrating (they need about 50 verified forecasts per regime).</p>"
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    left, top, bot = 50, 10, 30
    allx = pd.concat([s.index.to_series() for s in series.values()])
    x0, x1 = allx.min(), allx.max()
    span = max((x1 - x0).days, 1)
    X = lambda t: left + (t - x0).days / span * (w - left - 10)
    Y = lambda v: top + (ymax - v) / (ymax - ymin) * (h - top - bot)
    for v in np.linspace(ymin, ymax, 5):
        out.append(f'<line x1="{left}" x2="{w - 10}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--line)"/>'
                   f'<text x="{left - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>')
    if target is not None:
        out.append(f'<line x1="{left}" x2="{w - 10}" y1="{Y(target):.1f}" y2="{Y(target):.1f}" '
                   f'stroke="var(--fg)" stroke-dasharray="4 4"/>')
    palette = ["#1f5fae", "#b4321f", "#2a7a3d", "#a8650a", "#6b4fa3"]
    for k, (name, s) in enumerate(series.items()):
        s = s.dropna()
        if s.empty:
            continue
        pts = " ".join(f"{X(t):.1f},{Y(float(np.clip(v, ymin, ymax))):.1f}" for t, v in s.items())
        out.append(f'<polyline fill="none" stroke="{palette[k % 5]}" stroke-width="2" points="{pts}"/>')
        out.append(f'<text x="{left + 8 + 170 * k}" y="{h - 8}" style="fill:{palette[k % 5]}">{_esc(name)}</text>')
    for t in pd.date_range(x0.normalize() + pd.offsets.MonthBegin(0), x1, freq="2MS"):
        out.append(f'<text x="{X(t):.0f}" y="{h - 18}">{t:%b %y}</text>')
    out.append("</svg>")
    return "".join(out)


def _csv(p, **kw):
    try:
        return pd.read_csv(p, **kw)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _load(source):
    if source == "live":
        reg = Registry(C.REGISTRY, "live")
        return dict(stream="live", scores=reg.scores(), chain=reg.chain(), verify=reg.verify()[1],
                    bmap=_csv(C.STORE / "biasmap.csv"), bul=_csv(C.STORE / "bulletins.csv"),
                    cov=pd.DataFrame(), coll=pd.DataFrame())
    R = C.REPLAY
    reg = Registry(R / "registry", "backtest")
    rd = lambda n: _csv(R / n)
    return dict(stream="backtest", scores=pd.read_csv(R / "scores.csv.gz", parse_dates=["t0", "verified_at"]),
                chain=reg.chain(), verify=reg.verify()[1], bmap=rd("biasmap.csv"), bul=rd("bulletins.csv"),
                cov=rd("coverage.csv"), coll=rd("collision.csv"))


def build(source="auto"):
    if source == "auto":
        # show the live stream once it has verified forecasts; until then the labelled backtest
        has_live = len(Registry(C.REGISTRY, "live").scores()) > 0
        source = "live" if has_live or not (C.REPLAY / "scores.csv.gz").exists() else "backtest"
    d = _load(source)
    S = d["scores"]
    lb_all = leaderboard(S) if len(S) else pd.DataFrame()
    lb_storm = leaderboard(S, regime="storm") if len(S) else pd.DataFrame()
    badge = ('<span class="badge live">LIVE</span>' if d["stream"] == "live" else
             '<span class="badge backtest">BACKTEST</span>')
    note = ("" if d["stream"] == "live" else
            "<p class='muted'><b>Backtest:</b> the system replayed day by day over the ThermoSwarm 2025&ndash;26 "
            "record using only information available at each date. These are not pre-registered live forecasts; "
            "the live registry starts on the first automated run.</p>")

    # bulletin (latest day)
    bul_html = "<p class='muted'>No bulletin yet.</p>"
    if len(d["bul"]):
        b = d["bul"].copy()
        b["day"] = pd.to_datetime(b.day)
        last = b[b.day == b.day.max()].sort_values("band")
        rows = []
        for _, r in last.iterrows():
            flag = str(r.get("flag", "NORMAL")) if "flag" in r else "–"
            cls = flag.split(":")[0]
            rows.append(f"<tr><td>{C.band_label(int(r.band))}</td><td>{_fmt(r.R_pred - 1, pct=True)}</td>"
                        f"<td>{_fmt(r.lo)}&ndash;{_fmt(r.hi)}</td><td>{_fmt(r.bound72_km, 1)} km</td>"
                        f"<td>{_fmt(r.get('multiplier', np.nan), 2)}</td><td class='flag-{_esc(cls)}'>{_esc(flag)}</td></tr>")
        bul_html = (f"<p class='muted'>Issued {b.day.max():%d %b %Y}. Ratio R = true density / NRLMSIS 2.1.</p>"
                    "<table><tr><th>Altitude</th><th>72-h model bias</th><th>90% range of R</th>"
                    "<th>Drag position uncertainty (72 h)</th><th>vs normal</th><th>Flag</th></tr>"
                    + "".join(rows) + "</table>")

    def lb_table(lb):
        if lb.empty:
            return "<p class='muted'>Not enough verified forecasts yet.</p>"
        rows = "".join(f"<tr><td>{_esc(r.model)}</td><td>{r.n:,}</td><td>{r.median_err_km:.2f}</td>"
                       f"<td>{r.rms_err_km:.1f}</td><td>{r.p90_err_km:.1f}</td><td>{_fmt(r.skill_rms, pct=True)}</td>"
                       f"<td>{_fmt(r.coverage, 2)}</td><td>{_fmt(r.interval_score, 3)}</td></tr>"
                       for r in lb.itertuples())
        return ("<table><tr><th>Model</th><th>Forecasts</th><th>Median error (km)</th><th>RMS (km)</th>"
                "<th>90th pct (km)</th><th>RMS skill vs NRLMSIS</th><th>90% interval coverage</th>"
                "<th>Interval score</th></tr>" + rows + "</table>")

    kp = []
    if len(lb_storm):
        best = lb_storm.iloc[0]
        ref = lb_storm[lb_storm.model == "nrlmsis-2.1"]
        kp.append(("Storm-time RMS error, best model", f"{best.rms_err_km:.1f} km",
                    f"{best.model}; NRLMSIS {ref.rms_err_km.iloc[0]:.1f} km" if len(ref) else best.model))
    if len(lb_all):
        g = lb_all[lb_all.model == "thermoswarm-gb"]
        if len(g):
            kp.append(("Interval coverage (target 90%)", f"{100 * g.coverage.iloc[0]:.1f}%", "thermoswarm-gb, all regimes"))
    if len(d["coll"]):
        c = classify(d["coll"])
        kp.append(("Conjunctions where storm drag changes the risk category",
                   f"{100 * (c.escalated | c.diluted).mean():.1f}%",
                   f"{100 * c.escalated.mean():.1f}% could be riskier, {100 * c.diluted.mean():.1f}% diluted (simulated set)"))
    kp.append(("Registry", f"{len(d['chain'])} days", d["verify"]))
    kpis = "".join(f"<div class='kpi'><span class='muted'>{_esc(a)}</span><b>{_esc(b)}</b>"
                   f"<span class='muted'>{_esc(c)}</span></div>" for a, b, c in kp)

    cov_html = ""
    if len(S):
        s = S[np.isfinite(S.interval_score)].copy()
        s["day"] = pd.to_datetime(s.verified_at).dt.floor("D")
        series = {}
        for mname in ["thermoswarm-gb", "persistence", "nrlmsis-2.1"]:
            g = s[s.model == mname].groupby("day").covered.agg(["sum", "count"])
            r = g["sum"].rolling(30, min_periods=10).sum() / g["count"].rolling(30, min_periods=10).sum()
            series[mname] = r
        cov_html = line_svg(series, ymin=0.6, ymax=1.0, target=C.COVERAGE)

    coll_html = "<p class='muted'>No conjunction assessments yet.</p>"
    if len(d["coll"]):
        c = classify(d["coll"])
        sd = c[c.storm_day]
        order = ["high", "medium", "low"]

        def pct(x):
            return f"{100 * x.mean():.1f}%" if len(x) else "&ndash;"
        rows = []
        for cat in order:
            g = c[c.cat_nominal == cat]
            rows.append(f"<tr><td>{cat.capitalize()}</td><td>{len(g):,}</td><td>{pct(g.escalated)}</td>"
                        f"<td>{pct(g.diluted)}</td><td>{pct(~(g.escalated | g.diluted))}</td></tr>")
        rows.append(f"<tr><td><b>All</b></td><td><b>{len(c):,}</b></td><td><b>{pct(c.escalated)}</b></td>"
                    f"<td><b>{pct(c.diluted)}</b></td><td><b>{pct(~(c.escalated | c.diluted))}</b></td></tr>")
        rows.append(f"<tr><td>Storm days only</td><td>{len(sd):,}</td><td>{pct(sd.escalated)}</td>"
                    f"<td>{pct(sd.diluted)}</td><td>{pct(~(sd.escalated | sd.diluted))}</td></tr>")
        coll_html = ("<p>Each conjunction's collision probability (Pc) is recomputed with the drag uncertainty in "
                     "today's bulletin added to the along-track position. Two different things can happen:</p>"
                     "<ul><li><b>Riskier:</b> somewhere inside the drag interval the Pc reaches a higher risk category "
                     "than screened, so the screen understated the risk.</li>"
                     "<li><b>Diluted:</b> the extra uncertainty spreads the Pc so thin that it drops a category. "
                     "This does not mean the encounter is safer; it means the orbit is not known well enough for a "
                     "low Pc to be reassuring.</li></ul>"
                     "<table><tr><th>Category as screened</th><th>Conjunctions</th><th>Riskier</th>"
                     "<th>Diluted</th><th>Unchanged</th></tr>" + "".join(rows) + "</table>"
                     "<p class='muted'>Simulated encounters (realistic LEO geometry and covariances) until real conjunction "
                     "data messages are supplied in <code>conjunctions/</code>. Categories use illustrative thresholds "
                     f"(high &ge; {C.PC_HIGH:g}, medium &ge; {C.PC_MEDIUM:g}), not official ones.</p>")

    heat = heatmap_svg(d["bmap"]) if len(d["bmap"]) else ""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ThermoSwarm Observatory</title>
<style>{CSS}</style></head><body><main>
<p class="muted">Open, self-scoring thermospheric drag service</p>
<h1>ThermoSwarm Observatory {badge}</h1>
<p>About 1,100 tracked objects act as sensors of upper-atmosphere density. Every day the observatory measures how wrong
the NRLMSIS 2.1 model is at each altitude, forecasts drag 72 hours ahead with calibrated uncertainty, shows how that
uncertainty changes collision risk, and scores every forecast &mdash; its own and any submitted model &mdash; once the truth is
known. Forecasts are committed to a hash-chained public registry before they can be checked.</p>{note}
<div class="kpis">{kpis}</div>
<h2>Today's drag bulletin</h2><div class="card">{bul_html}</div>
<h2>Model bias by altitude, day by day</h2><div class="card">{heat}
<p class="muted">Red: more air than NRLMSIS 2.1 predicts (satellites slow down faster than modelled); blue: less.</p></div>
<h2>Leaderboard: 72-hour drag forecasts (all conditions)</h2><div class="card">{lb_table(lb_all)}</div>
<h2>Leaderboard: storm conditions</h2><div class="card">{lb_table(lb_storm)}</div>
<h2>Are the error bars honest?</h2><div class="card">{cov_html}
<p class="muted">Rolling 30-day share of verified outcomes inside the published 90% interval (dashed line = 90%).
Intervals come from adaptive conformal prediction, so coverage is measured, not assumed.</p></div>
<h2>Collision-risk impact</h2><div class="card">{coll_html}</div>
<h2>Verify it yourself</h2><div class="card"><p>Every day's forecasts are in <code>registry/forecasts/</code>; each line of
<code>registry/chain.jsonl</code> holds the file's SHA-256 and a hash chained to the previous day. Run
<code>python -m observatory.pipeline --verify</code> to recompute the chain. Submit your own model's forecasts as a JSON file
in <code>submissions/&lt;team&gt;/YYYY-MM-DD.json</code> before the day's run (see SUBMITTING.md); they are scored with the
same rules. Machine-readable feed: <a href="feed/latest.json">feed/latest.json</a>.</p>
<p class="muted">Built on the ThermoSwarm study (code 10.5281/zenodo.23675174; data 10.5281/zenodo.23674144). Element sets
are not redistributed. Not an official product of any agency.</p></div>
</main></body></html>"""
    (C.DOCS / "index.html").write_text(page)
    feed = dict(stream=d["stream"], leaderboard=lb_all.to_dict("records"), storm=lb_storm.to_dict("records"),
                registry=d["verify"])
    if len(d["bul"]):
        b = d["bul"].copy()
        b["day"] = b.day.astype(str)
        feed["bulletin"] = b[b.day == b.day.max()].to_dict("records")
    (C.DOCS / "feed" / f"{d['stream']}.json").write_text(json.dumps(feed, indent=1, default=float))
    return C.DOCS / "index.html"


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "auto")
