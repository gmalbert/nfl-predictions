# V2 data model and feature catalog

> **Implementation review — 2026-08-24.** Entries marked **Implemented** were confirmed present in the uncommitted `nfl_predictor/` package. The normalized tables and contract/feature/model/backtest primitives are complete as a research foundation. “Implemented when input is present” means the transformation exists, not that a production adapter currently supplies the input. All **Schema-ready** families remain open; no source-run, market-snapshot, availability-snapshot, or player-fact production loader was found. V2 is not yet connected to the legacy UI or its publishing workflow.

Status legend:

- **Implemented:** executable in `nfl_predictor/` and covered by deterministic tests where practical.
- **Schema-ready:** normalized storage and cutoff semantics exist; an external source adapter is still required.
- **Research gate:** do not promote until the listed validation succeeds.

## Point-in-time rule

For a prediction cutoff `T`, every joined observation must satisfy `available_at <= T`. Event time is not enough. An injury report describing Wednesday practice but published Thursday is unavailable to a Wednesday model. Closing odds are unavailable to a model frozen earlier in the day. The `assert_point_in_time` contract enforces this rule.

All rolling results use this sequence:

```python
series.shift(1).rolling(window, min_periods=1).mean()
```

The shift is mandatory. Imputation, encoding, calibration, and feature selection are also fitted inside each training fold.

The legacy wide schedule contains closing-market columns. V2 therefore labels those rows at kickoff (combining `gameday` and nflverse's Eastern `gametime`) and treats them as close-benchmark forecasts. They are not valid early- or midweek snapshots. Earlier decision horizons require timestamped `market_snapshot` rows whose `available_at` precedes the declared cutoff.

## Implemented feature and infrastructure changes

The following list contains more than 30 additional implemented features or changes. Code locations are authoritative.

| ID | Feature or change | Status | Code |
|---:|---|---|---|
| 1 | Canonical team aliases (`LA`/`LAR`, `JAC`/`JAX`, relocations) | Implemented | `features.normalize_team` |
| 2 | Explicit completed-game flag; future scores remain null | Implemented | `features.add_game_targets` |
| 3 | Home margin regression target | Implemented | `features.add_game_targets` |
| 4 | Game total regression target | Implemented | `features.add_game_targets` |
| 5 | Home-win target with ties preserved as pushes | Implemented | `features.add_game_targets` |
| 6 | Home-cover target using NFLverse spread semantics | Implemented | `features.add_game_targets` |
| 7 | Over target with pushes preserved | Implemented | `features.add_game_targets` |
| 8 | Underdog identity, cover, and outright-win targets | Implemented | `features.add_game_targets` |
| 9 | Two-row team perspective for every game | Implemented | `features.to_team_games` |
| 10 | Prior games played without counting scheduled games | Implemented | `features.add_team_form` |
| 11 | Prior points-for mean over 3, 5, 8, and 16 games | Implemented | `features.add_team_form` |
| 12 | Prior points-allowed mean over 3, 5, 8, and 16 games | Implemented | `features.add_team_form` |
| 13 | Prior margin mean over 3, 5, 8, and 16 games | Implemented | `features.add_team_form` |
| 14 | Prior win rate over 3, 5, 8, and 16 games | Implemented | `features.add_team_form` |
| 15 | Prior ATS cover rate over 3, 5, 8, and 16 games | Implemented | `features.add_team_form` |
| 16 | Prior game-total mean over 3, 5, 8, and 16 games | Implemented | `features.add_team_form` |
| 17 | Prior over rate over 3, 5, 8, and 16 games | Implemented | `features.add_team_form` |
| 18 | Prior close-game rate over 3, 5, 8, and 16 games | Implemented | `features.add_team_form` |
| 19 | Prior blowout-win rate over 3, 5, 8, and 16 games | Implemented | `features.add_team_form` |
| 20 | Exponentially weighted version of each form metric | Implemented | `features.add_team_form` |
| 21 | Home/away values plus matchup differences for numeric form | Implemented | `features._join_team_features` |
| 22 | Absolute spread magnitude | Implemented | `features.add_market_features` |
| 23 | Distance to NFL key numbers 3, 6, 7, 10, and 14 | Implemented | `features.add_market_features` |
| 24 | Exact key-number flags for 3, 7, 10, and 14 | Implemented | `features.add_market_features` |
| 25 | Raw home/away moneyline implied probabilities | Implemented | `features.add_market_features` |
| 26 | Multiplicative no-vig home/away moneyline probabilities | Implemented | `features.add_market_features` |
| 27 | Moneyline overround | Implemented | `features.add_market_features` |
| 28 | Spread-side implied probabilities and overround | Implemented | `features.add_market_features` |
| 29 | Total-side raw and no-vig probabilities plus overround | Implemented | `features.add_market_features` |
| 30 | Opening-to-current spread movement | Implemented when input is present | `features.add_market_features` |
| 31 | Opening-to-current total movement | Implemented when input is present | `features.add_market_features` |
| 32 | Opening-to-current moneyline movement by side | Implemented when input is present | `features.add_market_features` |
| 33 | Key-number crossing flag | Implemented when input is present | `features.add_market_features` |
| 34 | Rest differential | Implemented | `features.add_context_features` |
| 35 | Home/away short-rest flags | Implemented | `features.add_context_features` |
| 36 | Home/away extended-rest flags | Implemented | `features.add_context_features` |
| 37 | Material rest-mismatch flag | Implemented | `features.add_context_features` |
| 38 | Dome/closed-roof flag | Implemented | `features.add_context_features` |
| 39 | Weather-observed flag distinct from calm weather | Implemented | `features.add_context_features` |
| 40 | Freezing and hot outdoor flags | Implemented | `features.add_context_features` |
| 41 | Windy and extreme-wind flags using wind, not temperature | Implemented | `features.add_context_features` |
| 42 | Continuous cold-degree and wind-over-10 features | Implemented | `features.add_context_features` |
| 43 | Temperature/wind interaction | Implemented | `features.add_context_features` |
| 44 | Neutral-site and division-game flags | Implemented | `features.add_context_features` |
| 45 | Travel distance, long-travel, time-zone shift, eastbound flags | Implemented when input is present | `features.add_context_features` |
| 46 | Offensive EPA/play | Implemented from PBP | `features.aggregate_pbp_team_games` |
| 47 | Offensive success rate | Implemented from PBP | `features.aggregate_pbp_team_games` |
| 48 | Pass rate and neutral-situation pass rate | Implemented from PBP | `features.aggregate_pbp_team_games` |
| 49 | Pass EPA/dropback and rush EPA/carry | Implemented from PBP | `features.aggregate_pbp_team_games` |
| 50 | Explosive-play rate | Implemented from PBP | `features.aggregate_pbp_team_games` |
| 51 | Sack and quarterback-hit rates allowed | Implemented from PBP | `features.aggregate_pbp_team_games` |
| 52 | Turnover rate | Implemented from PBP | `features.aggregate_pbp_team_games` |
| 53 | Early-down pass rate | Implemented from PBP | `features.aggregate_pbp_team_games` |
| 54 | Third-down and red-zone success proxies | Implemented from PBP | `features.aggregate_pbp_team_games` |
| 55 | Defensive EPA, success, explosive plays, sacks, takeaways | Implemented from PBP | `features.aggregate_pbp_team_games` |
| 56 | Four- and eight-game prior-only PBP windows | Implemented | `features.build_pbp_pregame_features` |
| 57 | Explicit feature denylist for outcomes/returns/predictions | Implemented | `features.feature_columns` |
| 58 | American/decimal/implied odds conversion | Implemented | `markets.py` |
| 59 | Multiplicative and additive no-vig conversion | Implemented | `markets.de_vig` |
| 60 | Unit EV and capped fractional Kelly | Implemented | `markets.expected_profit`, `kelly_fraction` |
| 61 | Moneyline, spread, and total settlement with pushes | Implemented | `markets.py`, `backtest.grade_bets` |
| 62 | Best-price selection | Implemented | `markets.best_price` |
| 63 | Probability-based CLV | Implemented | `markets.probability_clv` |
| 64 | Season-forward train/calibration/test folds | Implemented | `validation.season_walk_forward_splits` |
| 65 | Optional embargo and strict fold assertions | Implemented | `validation.py` |
| 66 | Brier, log loss, AUC, accuracy, ECE, MCE | Implemented | `validation.probability_metrics` |
| 67 | Calibration reliability table | Implemented | `validation.calibration_table` |
| 68 | Held-out isotonic calibration | Implemented | `modeling.IsotonicProbabilityCalibrator` |
| 69 | Learned model/market convex blend | Implemented | `modeling.learn_market_blend_weight` |
| 70 | Coherent correlated margin/total simulation | Implemented | `modeling.ScoreDistributionForecaster` |
| 71 | Moneyline/spread/total probabilities from one simulation | Implemented | `modeling.ScoreSimulation` |
| 72 | Prediction intervals for margin and total | Implemented | `modeling.ScoreSimulation` |
| 73 | Margin-aware pregame Elo baseline | Implemented | `modeling.EloRatings` |
| 74 | Edge/EV/Kelly candidate filter with contradictory-side guard | Implemented | `backtest.recommend_bets` |
| 75 | Bankroll path, profit factor, drawdown, and ROI | Implemented | `backtest.py` |
| 76 | Season-clustered ROI bootstrap | Implemented | `backtest.cluster_bootstrap_roi` |
| 77 | Minimum-bets/ROI/CLV promotion gate | Implemented | `backtest.promotion_gate` |
| 78 | Timestamp, type, key, and point-in-time data contracts | Implemented | `contracts.py` |
| 79 | Append-oriented SQLite warehouse and foreign keys | Implemented | `schema_v2.sql`, `warehouse.py` |
| 80 | CLI for feature build, database initialization, and walk-forward replay | Implemented | `cli.py`, `scripts/nfl_v2.py` |

## Normalized data model

### `source_run`

One immutable record per source fetch. Stores provider, URI, start/end, `available_at`, content hash, row count, status, and error. This separates “the source returned zero rows” from “the fetch failed.”

### `game`

One row per game. Outcomes remain null until final and have a separate `result_available_at`. Kickoff is a timestamp, not only a date. Neutral site, roof, surface, and stadium are game facts.

### `team_game`

Two rows per game, one per team. Postgame PBP summaries live here and are never joined to the same game's pregame feature snapshot.

### `market_snapshot`

One row per game/book/market/participant/side/time. `line` and `price_american` are separate. `observed_at` is what the provider reports; `available_at` is when this system could use it.

### `availability_snapshot`

One row per player/game/evidence update. Supports injury status, practice status, probability active, expected snap share, source, and time.

### `player_game`

Final opportunity facts such as snaps, routes, targets, carries, and dropbacks. These form the basis of subsequent-game prop features.

### `feature_snapshot`

Long-form entity/game/cutoff/version/name/value storage. Missingness is explicit. A materialized wide training frame can be generated for a requested feature-set version and cutoff.

### `model_run`

Stores model and feature versions, target, training/calibration windows, code commit, parameters, and metrics. A prediction cannot exist without a model run.

### `prediction_snapshot`

Immutable probability at a specified cutoff. Stores raw market and calibrated probabilities separately. Unique keys prevent rewriting the same historical prediction.

### `bet_decision` and `bet_settlement`

The decision records the quoted market snapshot, edge, EV, stake rule, time, and policy version. Settlement records result, stake, profit, closing quote, CLV, time, and settlement version. A no-bet remains auditable.

### `data_quality_event`

Persists contract failures, staleness, duplicate keys, missingness spikes, and distribution drift rather than hiding them in console output.

## Schema-ready feature families

These are intentionally not labeled “implemented” until their public, point-in-time source adapters are added.

| Family | Proposed fields | Source and gate |
|---|---|---|
| Quarterback availability | starter probabilities, backup delta, practice trend | NFLverse depth/injury/participation; replay publication times |
| Unit continuity | OL returning snaps, secondary continuity, receiver continuity | weekly rosters + participation; cold-start tests |
| Player opportunity | snap share, route share, target share, carry share, red-zone share | player-game facts; player-forward validation |
| NGS efficiency | time to throw, completion probability, RYOE, separation | nflverse NGS; source/license metadata |
| FTN charting | motion, play action, RPO, personnel, coverage proxies | nflverse FTN; availability delay audit |
| Multi-book market | consensus, dispersion, best price, move velocity, stale quote | timestamped snapshots; bookmaker-specific calibration |
| Injury scenarios | active/snap distributions, replacement allocation | learned availability model; scenario calibration |
| Referee | penalty rate, automatic first downs, home/away differential | partial pooling; crew assignment publication time |
| Travel | distance, time zones, altitude, international sequence | venue registry; era and direction interactions |
| Coaching regime | coordinator IDs, scheme change, tenure, aggressiveness | timestamped staff registry; regime reset tests |
| Strength of schedule | opponent-adjusted EPA and Elo | iterative prior-only calculation; no final standings |
| Portfolio correlation | game factor, team factor, player/opportunity factor | joint simulation; correlation stress tests |

## Target definitions

- `home_margin = home_score - away_score`
- `actual_total = home_score + away_score`
- `target_home_win = I(home_margin > 0)`; ties are pushes/null for binary evaluation
- `home_cover_margin = home_margin - spread_line` because positive NFLverse spread means home favored
- `target_home_cover = I(home_cover_margin > 0)`; equality is a push
- `total_margin = actual_total - total_line`
- `target_over = I(total_margin > 0)`; equality is a push

No target is filled with zero before completion.

## Feature admission checklist

A feature enters a model only if all answers are yes:

1. Is the definition stable and unit-tested?
2. Is the source permitted and reproducible?
3. Are event time and availability time stored?
4. Can the historical value be reconstructed at the prediction cutoff?
5. Are joins one-to-one or explicitly many-to-one with validation?
6. Is missing distinct from measured zero?
7. Is transformation fitted inside the temporal fold?
8. Does ablation improve an untouched period or calibration?
9. Does value repeat across seasons rather than one slice?
10. Is the feature's latency acceptable for the intended decision time?
