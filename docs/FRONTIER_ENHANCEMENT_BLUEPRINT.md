# Frontier Enhancement Blueprint

> **Implementation review — 2026-08-24.** No end-to-end availability scenario model, unit-continuity feed, play-sequence model, or corresponding product UI was found. The V2 schema is ready to store availability and player facts, and it includes basic travel/context and PBP features; those are prerequisites, not completion of this blueprint. Treat every item below as **planned/research-gated**.

Existing docs already cover advanced/deep models, player props, data pipelines, calibration, weekly performance, and extensive feature roadmaps. The gaps below concern availability scenarios, unit continuity, and play-policy modeling.

## Availability and unit model

Represent practice participation and game status as timestamped evidence feeding player availability and snap-share distributions. Aggregate into unit continuity for offensive line, secondary, pass rush, and receiving groups; propagate scenarios through spread, total, and props.

```python
def scenario_mix(predictions, scenario_probs):
    keys = predictions[0].keys()
    return {k: sum(w * p[k] for w, p in zip(scenario_probs, predictions)) for k in keys}
```

## Play-sequence model

Model drive transitions from down, distance, field position, time, score, timeouts, personnel, motion, weather, and opponent. Separate descriptive fourth-down success from causal policy evaluation; observed choices are selection-biased.

## Data additions

- Participation, inactive, snap, and injury-news observations with `available_at`.
- Player tracking-derived separation, pressure, blocking, and coverage features where licensed.
- OL/skill-unit continuity, coordinator scheme, travel, surface, roof, and wind distributions.
- Referee crew tendencies with partial pooling.
- Timestamped multi-book game and prop markets.

## Product additions

- Injury/participation scenario tree and probability-change timeline.
- Unit matchup cards rather than only team aggregates.
- Drive simulator with policy comparisons and uncertainty.
- Correlated game/prop exposure view.
- Explicit stale-price and unresolved-status no-bet states.

## Gates

Replay every historical week at fixed information horizons. Report log loss, calibration, CRPS/MAE, prop calibration, CLV, and drawdown by weather, quarterback status, injury uncertainty, favorite size, and season week. Use season-forward and team/player cold-start tests; compare to market, Elo, and simple EPA baselines. A feature fails if its original publication time cannot be reconstructed.
