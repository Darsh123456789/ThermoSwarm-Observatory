# ThermoSwarm Observatory

**A public, tamper-evident referee for satellite-drag forecasts, and a daily storm-time
drag-uncertainty bulletin, built from the orbital decay of about a thousand objects already
in orbit.**

Every day the Observatory:

1. **Collects** new element sets for 1,137 low-Earth-orbit objects (200–800 km, near-circular)
   and turns each object's orbital decay into an orbit-averaged measurement of
   thermospheric density, relative to NRLMSIS 2.1 (method of the ThermoSwarm paper,
   [doi:10.5281/zenodo.23675174](https://doi.org/10.5281/zenodo.23675174)).
2. **Maps the bias** of NRLMSIS 2.1 in twelve 50-km altitude bands: how much more or less
   air there is than the model says, with bootstrap confidence intervals.
3. **Forecasts** the density ratio for every object over the next 72 hours with four
   built-in models (NRLMSIS as issued, persistence, a 7-day baseline and the ThermoSwarm
   gradient-boosted model) plus any model submitted by an outside team, and attaches
   **90% error bars** from a model-agnostic adaptive conformal layer, separately for quiet
   and storm conditions.
4. **Translates** the drag uncertainty into **collision-probability uncertainty**: for each
   conjunction it reports the nominal Pc, a storm-aware Pc and the worst case across the
   72-hour error bar, and flags conjunctions whose risk category could change.
5. **Commits** all forecasts to an append-only, SHA-256 hash-chained registry *before* the
   verifying observations exist, then scores them automatically when they arrive, and
   publishes a leaderboard.
6. **Publishes** a web page and a JSON feed through GitHub Actions and GitHub Pages.

> **Status.** Research prototype by a high-school student. It is not an operational
> product, is not affiliated with or endorsed by NOAA, the Office of Space Commerce,
> TraCSS, the U.S. Space Force or any other agency, and its Pc categories are illustrative
> thresholds, not official ones. Until the TraCSS conjunction-assessment verification
> dataset is obtained, collision results use **simulated conjunctions**.

## Why it matters

Drag is the largest source of error in low-Earth-orbit predictions during geomagnetic
storms, and collision-probability (Pc) screening inherits that error. Today a satellite
operator receiving a conjunction warning during a storm has no independent, measured
statement of how wrong the density model is likely to be *at their altitude* over the next
three days. The Observatory provides that statement, and it does so with forecasts that
are publicly committed in advance, so their skill can be checked by anyone rather than
claimed after the fact.

## Full-year replay (backtest)

The pipeline was run causally, day by day, over a full year of archived element sets
(forecasts issued only from information available at the time, models retrained weekly,
error bars updated only from already-verified forecasts). This is a **backtest**: it
was not pre-registered, and it is kept in `replay/` under a separate `backtest` stream so
it can never be confused with live forecasts in `registry/`.

293 daily forecast files, 2025-11-19 to 2026-09-15, 1,129 objects, 98,019 verified
72-hour windows per model (392,076 forecast–outcome pairs). Errors are the 72-hour
along-track position error implied by each density forecast. The registry for this stream
verifies end to end (`293 daily commits verified`).

**All conditions**

| Model | Median error (km) | RMS error (km) | 90th pct (km) | RMS skill vs NRLMSIS | 90% interval coverage | Interval score |
|---|---|---|---|---|---|---|
| persistence | 2.56 | 45.0 | 25.3 | +0.302 | 91.0% | 0.852 |
| thermoswarm-gb | 2.72 | 47.1 | 28.8 | +0.270 | 91.1% | 0.721 |
| baseline-7d | 2.90 | 48.3 | 29.8 | +0.251 | 90.7% | 0.753 |
| nrlmsis-2.1 | 3.51 | 64.5 | 37.6 | +0.000 | 90.5% | 0.883 |

**Storm conditions** (Kp ≥ 5− in the previous 24 h, or within 48 h of storm onset; 24,458 windows)

| Model | Median error (km) | RMS error (km) | 90th pct (km) | RMS skill vs NRLMSIS | 90% interval coverage | Interval score |
|---|---|---|---|---|---|---|
| thermoswarm-gb | 3.12 | 46.6 | 32.4 | +0.171 | 90.8% | 0.748 |
| baseline-7d | 3.27 | 46.6 | 31.1 | +0.171 | 90.5% | 0.744 |
| persistence | 3.11 | 48.8 | 31.9 | +0.133 | 90.4% | 0.987 |
| nrlmsis-2.1 | 3.57 | 56.2 | 35.7 | +0.000 | 91.2% | 0.914 |

What this shows:

- Every model's published **90% intervals cover 90.4–91.2%** of outcomes, in quiet and in
  storm conditions, so the error bars mean what they say.
- All three data-driven models beat NRLMSIS 2.1 as issued: RMS along-track error falls from
  64.5 km to 45.0 km with simple persistence (30% skill).
- **In storms the ThermoSwarm model is the most accurate** (RMS skill 0.171 against 0.133 for
  persistence) and its interval score is 24% better than persistence's (0.748 against
  0.987; lower is better). In quiet conditions persistence has the lowest RMS error.
- **Collision risk (simulated conjunctions).** 20,800 simulated conjunctions over 52
  assessment days. Adding the 72-hour drag uncertainty changed the risk category of
  **20.7%** of them, in two opposite ways:
  - **1.4% became riskier:** somewhere inside the drag interval the Pc reaches a higher
    category than screened (5.1% of medium-risk conjunctions reach high).
  - **19.3% were diluted:** the extra along-track uncertainty spreads the Pc so thin that it
    falls a category (37.7% of high-risk conjunctions). A low Pc in that situation means the
    orbit is not known well enough, not that the encounter is safe, which is exactly what an
    operator needs to be told during a storm.

  These are simulated encounters, so they show the size of the effect, not real-world rates.

## Repository layout

| Path | Contents |
|---|---|
| `observatory/` | the pipeline: `ingest`, `sensors`, `physics`, `biasmap`, `models`, `conformal`, `collision`, `referee`, `pipeline`, `site`, `replay` |
| `store/` | derived, shareable data: object list, densities, drivers, model state |
| `registry/` | **live** hash-chained forecast registry and scores |
| `replay/` | full-year backtest outputs (separate `backtest` stream) |
| `docs/` | GitHub Pages site (`index.html`) and JSON feed (`feed/latest.json`) |
| `submissions/` | forecasts from outside teams (see [SUBMITTING.md](SUBMITTING.md)) |
| `conjunctions/` | optional CCSDS CDM files (`*.cdm`) to assess instead of simulated ones |
| `tests/` | unit tests |

Raw element sets are cached in `.cache/` and are **never committed**: Space-Track's user
agreement does not allow redistribution. Only derived densities and forecasts are
published.

## Running it

```bash
pip install -r requirements.txt
python -m pytest -q tests                      # unit tests
python -m observatory.pipeline                 # one daily run (needs network)
python -m observatory.pipeline --verify        # check the registry's hash chain
python scripts/offline_e2e.py                  # 16-day end-to-end run on archived data, no network
python -m observatory.replay <data_dir>        # full-year backtest
python -m observatory.site                     # rebuild docs/index.html
```

### Setting up the daily run on GitHub

1. **Settings → Pages**: deploy from branch `main`, folder `/docs`.
2. **Settings → Secrets and variables → Actions → Secrets**: add `SPACETRACK_USER` and
   `SPACETRACK_PASS` (a free [Space-Track](https://www.space-track.org) account).
3. Optional **variable** `TRACSS_OMM_URL`: an OMM JSON endpoint from the public TraCSS API.
   If neither source is available the pipeline falls back to CelesTrak.
4. **Actions → daily → Run workflow** once to check it; afterwards it runs daily at 23:30 UTC (05:00 IST).

Each run verifies the existing chain first and stops if anything was altered.

## How the pieces work

- **Density from decay.** For a pair of element sets 1–3 days apart, the change in mean
  motion is compared with the drag integral along the SGP4 orbit through NRLMSIS 2.1.
  Each object's ballistic factor is calibrated causally from its own quiet-time intervals
  over the previous 30 days, so the ratio R = 1 means NRLMSIS is right.
- **Error bars.** Adaptive conformal prediction (Gibbs & Candès, 2021) on the log-ratio
  error, one learner per model and regime. The miscoverage level takes one step per day,
  driven by that day's miss rate, so the published 90% intervals keep their coverage when
  the atmosphere changes. It works for any submitted model, not only the built-in ones.
- **Collision.** Short-encounter 2-D Pc by Gauss–Legendre quadrature (checked against
  Monte Carlo in the tests). The 72-hour along-track error bound from the forecast
  interval is added to the along-track covariance and as a shift, and the Pc is scanned
  across the interval.
- **Referee.** Forecasts for a day are written as a deterministic gzip file; the chain line
  stores its SHA-256 and the hash of the previous line. `--verify` recomputes everything.
  Scores use the paper's along-track error, `e = a · (B·R_pred − f72) · J`, plus interval
  coverage and the Winkler interval score.

## Citation

See [CITATION.cff](CITATION.cff). The underlying method is described in Chaudhary (2026),
*ThermoSwarm: Altitude-Resolved Storm-Time Thermospheric Density From Multi-Object Orbital
Decay, With Physics-Informed Drag Forecasting*.

## Licence

MIT. Derived data in `store/`, `registry/` and `replay/` are released under the same terms.
