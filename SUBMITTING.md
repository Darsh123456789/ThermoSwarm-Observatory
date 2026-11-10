# Submitting a model

Any team can enter the leaderboard. Your forecasts are scored with exactly the same rules,
verification data and conformal error bars as the built-in models.

## What to forecast

For each object in `store/network.csv`, predict **R72**: the mean ratio of true
thermospheric density to NRLMSIS 2.1 along the object's orbit over the 72 hours after its
latest element set. (R = 1 means NRLMSIS is right; R = 1.2 means 20% more air.)

## How

1. Fork the repository and add a file `submissions/<team-name>/<YYYY-MM-DD>.json`:

```json
{
  "team": "your-team",
  "model": "short description",
  "contact": "you@example.org",
  "forecasts": [
    {"norad": 4435, "R_pred": 1.07},
    {"norad": 5599, "R_pred": 0.96}
  ]
}
```

2. Open a pull request before **06:00 UTC** on that date. Forecasts merged after the day's
   run are not used.
3. Objects you leave out are not scored for you. `R_pred` must be positive.

## How it is scored

- The forecast enters the hash-chained registry (`registry/chain.jsonl`) with the day's run,
  before any verifying element set exists.
- When an element set 66–78 h later arrives, the observed ratio over the window is
  computed from the orbit's decay, and your along-track error follows from the paper's
  definition, `e = a · (B·R_pred − f72) · J`.
- Your model gets 90% intervals from the same adaptive conformal layer as every other
  model, learned from your own past errors. A well-calibrated model earns narrower intervals.
- Leaderboard columns: median, RMS and 90th-percentile along-track error (km), RMS skill
  against NRLMSIS 2.1, interval coverage and the interval score (lower is better), for all
  conditions and for storm conditions separately.

## Rules

Use only information available before the start of each object's 72-hour window. Do not
submit forecasts derived from the verifying element sets. Malformed files are skipped.
