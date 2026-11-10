# Warm start (from the backtest)

`thermoswarm-gb.json` (gradient-boosted model) and `conformal_state.json` (adaptive
conformal error-bar state) were learned in the full-year replay, 2025-11-19 to 2026-09-15
(`replay/`). The daily pipeline uses them only until it has saved its own live state in
`store/`; from then on live state replaces them. Live forecasts are scored only against
live observations, so this warm start affects interval widths and the first week's model, not
how forecasts are scored.
