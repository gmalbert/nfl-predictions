# Comprehensive repository review (2026)

> **Implementation review — 2026-08-24.** The V2 package, deterministic V2 tests, and a V2 quality workflow now exist in the working tree, but are uncommitted and are not wired into the Streamlit application. The legacy pipeline remains the production path. Completed remediation: V2 prior-only features, outcome/push semantics, market math and settlement, point-in-time contracts, walk-forward splits, calibration primitives, warehouse schema, CLI, and CI definition. Still open: legacy leakage/evaluation fixes, timestamped source ingestion, market/availability adapters, immutable publication, dashboard refactor, operational hardening, and release gates. See `IMPLEMENTATION_PRIORITIES_2026.md` for the authoritative queue.

Reviewed: 2026-08-11  
Scope: application, game and player-prop pipelines, stored model outputs, data model, automation, tests, operational risk, and betting methodology.

## Executive verdict

This is a feature-rich research dashboard, but the legacy model path is not yet a defensible betting backtest. The saved metrics report 52.80% spread accuracy, 66.37% moneyline accuracy, and 52.21% totals accuracy. The saved spread decision-policy audit is 28-26 from 54 bets, or 51.85%, with theoretical ROI of -1.01% at -110. That is below the 52.38% break-even rate. Moneyline accuracy cannot establish profitability without the selected price, no-vig baseline, and settled stake. Totals are also below the usual -110 break-even rate.

The largest problem is evaluation integrity, not model complexity. The legacy code contains full-sample outcome aggregations, shuffled train/test splits, test-set threshold selection, in-sample predictions written beside historical outcomes, and a spread target whose probability is relabeled and inverted after training. Those choices invalidate the README's claims of a leakage-free 60.9% ROI, 91.9% selective win rate, and production-ready calibration.

The new `nfl_predictor/` package added by this review supplies a parallel, testable path. It does not silently replace the dashboard. The package implements prior-only features, explicit schemas, correct market math and pushes, season-forward validation, held-out calibration, coherent margin/total simulation, an Elo baseline, market de-vig utilities, bet grading, CLV, bankroll metrics, and an append-oriented warehouse schema.

## What is already strong

- The application covers game markets, historical play-by-play, player props, parlay construction, exports, email, RSS, and model-performance pages.
- NFLverse supplies reproducible game, schedule, and play-by-play data.
- The player-prop package is separated more cleanly than the main application and already distinguishes passing, rushing, and receiving aggregation.
- Weather, rest, injuries, price context, calibration, and automated refreshes are recognized as important.
- Generated artifacts make the dashboard usable without training during every Streamlit session.
- Large play-by-play data is at least compressed and one artifact is managed through Git LFS.
- Existing roadmap documents show sustained product thought. They should be consolidated around measurable release gates rather than discarded.

## Critical findings

These findings describe the repository as inspected. Items explicitly marked “corrected in this review” have code changes in the current worktree; all others remain remediation work or are addressed only by the parallel v2 path.

### P0: evaluation and target integrity

1. **Full-sample target leakage (corrected in this review).** `nfl-gather-data.py` calculated favored, cover, over, under, and total-hit rates with full-data `groupby(...).mean()` mappings. They now use prior-kickoff observations, including correct postseason ordering.
2. **Shuffled temporal evaluation.** Spread, moneyline, and total models use `train_test_split(..., stratify=...)`. Games from later seasons can train models evaluated on earlier seasons.
3. **Test-set threshold selection.** F1 and spread-EV thresholds are selected on the same test rows later used for reported performance.
4. **In-sample historical probabilities.** After fitting, probabilities are generated over `X_spread`, `X_moneyline`, and `X_totals`, including training observations. These are stored next to outcomes and then used by diagnostic scripts.
5. **Spread target/probability mismatch.** The spread target is `spreadCovered`, the result is stored as `prob_underdogCovered`, and the probability is inverted with `1 - p`. A production model must have one documented orientation from target construction through settlement.
6. **Pushes are not a first-class outcome.** Several labels collapse pushes into one binary side or infer an under as `1 - overHit`, which makes a push a win for the under.
7. **Missing outcomes can become zeros.** Global `fillna(0)` erases the distinction among unknown, structurally absent, not applicable, and measured zero.
8. **Calibration is not temporally isolated.** Internal cross-validation calibration does not reproduce a real historical information boundary, and the calibrated result is then assessed after threshold tuning on the test set.
9. **No market baseline comparison.** A model can be accurate while adding no information beyond the closing line. The current metrics do not consistently compare against de-vigged market probabilities, spread MAE, or total MAE.
10. **No untouched final season.** There is no season reserved from all feature, hyperparameter, calibration, and policy decisions.

### P0: market and settlement correctness

11. **README performance claims are contradicted by saved artifacts.** `data_files/model_metrics.json` contains a negative spread strategy ROI, while the README advertises large positive values.
12. **Accuracy is treated as a betting objective.** Moneyline accuracy without price is economically meaningless; a favorite-only model can be highly accurate and lose money.
13. **Closing-line value is absent from the legacy journal.** Without bet-time and close-time snapshots, it is impossible to distinguish luck from a model that beats the information aggregation of the market.
14. **Odds are single-row game fields.** Opening, current, close, book, timestamp, market, side, and line belong in a time-series snapshot table.
15. **No best-price or book-dispersion logic.** A real decision should choose among books, reject stale quotes, and measure consensus disagreement.
16. **Kelly suggestions are too aggressive for unvalidated probabilities.** The UI advertises bankroll percentages despite weak calibration evidence. Shadow flat stakes are the safe default until forward gates pass.
17. **Correlated bets are summarized as independent.** Same-game markets and multiple props can share most of their risk. Portfolio exposure and cluster-aware uncertainty are missing.
18. **Parlay math needs joint probabilities.** Marginal probabilities cannot simply be multiplied when legs share game script, quarterback, weather, or player usage.

### P0/P1: feature correctness

19. **The windy flag used temperature (corrected in this review).** `isWindy` was calculated from `temp >= 15`; it now uses `wind >= 15` and v2 has a regression test.
20. **Home and away form are location-fragmented.** Several rolling helpers only examine a team's previous rows in the same home/away column, not all prior games from that team's perspective.
21. **Full-sample team encodings leak future target rates.** Favored and cover percentages are particularly dangerous because they are close to target encodings.
22. **Categorical features are listed and then silently discarded.** Team, quarterback, coach, stadium, roof, and surface are named in `features` but `select_dtypes` removes non-numeric values.
23. **Team identifiers are inconsistent.** The app uses `LA` while NFLverse commonly uses `LAR`; historical `OAK`, `SD`, `STL`, and `JAC` aliases require a season-aware registry.
24. **Weather missingness is treated as calm/zero.** Indoor games, unavailable observations, and true zero wind should not share one representation.
25. **Injuries are heuristic probability multipliers.** Reducing an over probability by a fixed percentage and setting the under to its complement is not a learned availability or workload model.
26. **Player matching falls back to last-name substring.** This can select the wrong player. GSIS IDs and an explicit cross-source identity table are required.
27. **Opponent defensive rank is unstable.** Ranks discard magnitude, move sharply in small samples, and can accidentally use future season totals. Use shrunk rate statistics as of the prediction cutoff.
28. **Feature publication time is not stored.** Even a historically dated record can leak if it was published after the modeled decision time.

### P1: code and architecture

29. **The main app is monolithic.** `predictions.py` is roughly 264 KB and mixes UI, networking, PDF generation, persistence, settlement, and business rules.
30. **The training script executes at import time.** `nfl-gather-data.py` loads data, engineers features, trains models, writes outputs, and prints results at module scope.
31. **Exception swallowing is pervasive.** `predictions.py` contains more than 150 broad `except Exception` handlers, many of which hide failed data or settlement logic.
32. **Warnings are globally disabled in player modules.** This can suppress deprecations, invalid joins, chained assignments, and numeric issues.
33. **Duplicate functions exist.** `pages/2_Player_Props.py` defines `load_player_props_predictions` twice.
34. **The play-by-play delimiter is inconsistent.** The primary loader uses tab separation while the chunked loader first tries comma separation and silently falls back.
35. **Side effects occur during page rendering.** Loading the home page can log recommendations and perform ESPN settlement calls.
36. **Generated CSVs are not written atomically.** An interrupted write can leave the dashboard with a partial artifact.
37. **No artifact manifest exists.** Models do not carry feature schema, training cutoff, code commit, source hashes, calibration window, or target definition.
38. **Model fallbacks are not observable.** The UI can mix trained probabilities and historical hit-rate estimates without a complete audit trail.

### P1: testing and quality controls

39. **There is one tracked test module.** It checks RSS link resolution and is network-dependent.
40. **Pytest is not installed by the declared requirements.** The existing test imports it optionally; the repository has no standard local test command.
41. **Core market math had no tests.** Spread orientation, pushes, American odds, de-vigging, EV, and settlement were unprotected.
42. **Temporal invariants had no tests.** There was no assertion that training precedes calibration and calibration precedes test.
43. **Data contracts were absent.** Duplicate game IDs, invalid timestamps, null keys, late observations, and out-of-domain market values could pass silently.
44. **No leakage allowlist/denylist exists.** Numeric outcome and return columns can accidentally enter a model after a wide CSV change.
45. **No reproducibility test exists.** Seeds, source versions, and feature-set versions are not collected in one manifest.

### P1: automation and repository operations

46. **A nightly workflow calls an untracked script.** `.github/workflows/nightly-update.yml` invokes `fetch_espn_weekly_scores.py`, but that file is ignored and not part of the checkout; failure is allowed.
47. **Multiple workflows push generated data directly to `main`.** Concurrent schedule, nightly, performance, and feed jobs can race or fail non-fast-forward.
48. **Broad `continue-on-error` hides stale production data.** A green workflow may still have skipped odds, play-by-play, props, or commits.
49. **Generated data bloats Git history.** The repository tracks dozens of frequently changing CSV/JSON artifacts and a 116 MB LFS play-by-play file.
50. **There is no data-retention or artifact-promotion policy.** Raw sources, derived features, model outputs, UI exports, and logs share `data_files/`.
51. **Dependency bounds are open-ended.** Minimum-only versions allow future major changes to break an automated training run.
52. **A runtime dependency was undeclared (corrected in this review).** The ESPN injury scraper imports Beautiful Soup; `beautifulsoup4` is now declared in `requirements.txt`.
53. **No lint/type/security CI exists.** Syntax, dead code, secrets, vulnerable dependencies, and unsafe workflows are not checked before deployment.

### P2: product and responsible-use controls

54. **Confidence labels imply more certainty than the validation supports.** “Elite” and bankroll ranges should be tied to out-of-sample calibration bands and sample size.
55. **No stale-data state is modeled.** Missing, delayed, cached, provisional, and final should be visible states, not only warnings.
56. **No responsible-gambling notice is prominent.** Research predictions should clearly state uncertainty and provide help resources.
57. **No prediction freeze is recorded.** A historical pick must preserve exactly what the user could see at that time, including price and injuries.
58. **No negative results registry exists.** Discarded features and failed strategies should be retained to reduce repeated data mining.

## The v2 data flow

```text
immutable source runs
        |
        v
normalized game / team-game / player-game facts
        |
        +---- timestamped market snapshots
        +---- timestamped availability snapshots
        |
        v
point-in-time feature snapshots
        |
        v
season-forward train -> held-out calibration -> untouched test
        |
        v
prediction snapshot -> policy decision -> immutable settlement -> CLV/model card
```

The normalized SQL schema is in `nfl_predictor/schema_v2.sql`. The corresponding validation and feature code is in `nfl_predictor/contracts.py` and `nfl_predictor/features.py`.

## Prioritized remediation plan

### Phase 0: stop publishing unsupported performance claims

- Label legacy outputs as research-only.
- Remove “proven,” “leakage-free,” and large ROI claims until reproduced by the v2 walk-forward path.
- Disable Kelly sizing and “elite” language in production views.
- Preserve the current saved spread result as a negative baseline.

### Phase 1: establish the trustworthy baseline

- Build v2 features with `python scripts/nfl_v2.py build-features`.
- Run season-forward model evaluation with `python scripts/nfl_v2.py walk-forward`.
- Compare market-only, Elo-only, form-only, PBP-only, and combined models.
- Publish every target's Brier score, log loss, calibration table, accuracy, MAE where applicable, ROI, CLV, drawdown, and number of bets.

### Phase 2: make data point-in-time

- Populate the v2 warehouse with source runs and availability timestamps.
- Capture multi-book lines at open, scheduled checkpoints, bet time, and close.
- Replace name joins with player/team identity registries.
- Add atomic artifact writes and content hashes.

### Phase 3: refactor the application

- Move loaders, settlement, exports, and model services out of `predictions.py`.
- Make Streamlit read immutable published artifacts only.
- Run refresh, grading, and training outside page-render code.
- Expose freshness, source, cutoff, model version, and calibration range in the UI.

### Phase 4: promote markets selectively

- Require at least two forward seasons.
- Require 500 settled decisions per promoted market or a predeclared smaller-market exception.
- Require positive mean CLV and a confidence interval that does not support a materially negative ROI.
- Re-run results by season, book, price band, favorite size, week, quarterback status, weather, and injury uncertainty.

## Definition of done

A model is not production-ready because it trained successfully or looks accurate. It is ready only when a fresh historical replay can reconstruct the exact information available at each cutoff, reproduce the same artifact from source hashes and code commit, beat declared baselines on an untouched period, settle every market correctly including pushes/voids, and pass the promotion gate without manual cherry-picking.
