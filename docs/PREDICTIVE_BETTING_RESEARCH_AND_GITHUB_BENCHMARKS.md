# Predictive NFL betting research and GitHub benchmarks

> **Implementation review — 2026-08-24.** The research conclusions have been partially operationalized in V2: prior-only PBP/form features, temporal folds, calibration measures, market de-vig/CLV utilities, an outcome denylist, normalized schema, and a coherent game simulation are present. Source manifests/loaders, timestamped multi-book data, availability/player-opportunity adapters, immutable prediction journal, UI decoupling, and all source-gated families remain open. Deferred/rejected ideas remain deferred; none should be promoted from this document alone.

Reviewed: 2026-08-11  
Method: public repositories were inspected for architecture and reproducibility ideas; papers and technical articles were reviewed for data, validation, probability, and market-design implications. Repository performance claims are treated as unverified unless independently reproduced.

## GitHub benchmark review

### nflverse/nflverse-data

[nflverse-data](https://github.com/nflverse/nflverse-data) publishes automated data releases rather than requiring downstream projects to commit every refreshed dataset. Adopt:

- source releases and manifests instead of committing derived artifacts to the application branch;
- scheduled status visibility;
- source licensing metadata;
- one canonical upstream for schedules, play-by-play, rosters, injuries, participation, and stats.

### nflverse/nflreadr and nflverse/nfl_data_py

[nflreadr](https://github.com/nflverse/nflreadr) provides cache modes, data dictionaries, and multiple storage formats. [nfl_data_py](https://github.com/nflverse/nfl_data_py) exposes schedules, play-by-play, player IDs, Next Gen Stats, depth charts, injuries, QBR, snap counts, and FTN charting data. Adopt:

- filesystem caching with explicit refresh rather than implicit page-load downloads;
- Parquet for typed analytical tables;
- data dictionaries and source-version recording;
- stable player IDs, not last-name matching;
- NFLverse injuries, participation, snap counts, and depth charts before brittle page scraping;
- column selection/downcasting at load time.

Important correction: EPA, CPOE, success, and win probability are play-by-play model outputs, not synonymous with Next Gen Stats. NGS contributes tracking-derived passing, rushing, and receiving measures; the documentation should keep those sources distinct.

### nflverse/nflfastR and fastrmodels

[nflfastR](https://github.com/nflverse/nflfastR) ships expected-points, win-probability, completion-probability, xYAC, and expected-pass models. It explicitly offers win probability both with and without the pregame spread. [The model article](https://opensourcefootball.com/posts/2020-09-28-nflfastr-ep-wp-and-cp-models/) reports calibration-oriented evaluation. Adopt:

- EPA and success as opponent-adjustable team-strength inputs;
- market-aware and market-free variants to measure incremental signal;
- model calibration plots, not accuracy alone;
- era-aware features because league scoring and strategy change;
- baseline feature groups for pass, rush, early down, neutral situation, red zone, pressure proxy, and explosive plays.

### nflverse/nfldata

[nfldata's dataset guide](https://github.com/nflverse/nfldata/blob/master/DATASETS.md) documents identifiers, games, standings, teams, and historical aliases. It also confirms the NFLverse schedule convention: a positive `spread_line` means the home team was favored by that many points. Adopt:

- a season-aware team registry;
- join validation and duplicate checks;
- explicit home-margin minus spread target construction;
- stable cross-provider player IDs;
- warnings when a weaker name-based join could create false matches.

### nflverse/ffopportunity

[ffopportunity](https://github.com/ffverse/ffopportunity) models expected fantasy points from opportunity and publishes precomputed versioned outputs. Adopt for props:

- separate opportunity from efficiency;
- targets/routes/carries/dropbacks and expected production before raw results;
- weekly and play-level outputs;
- model/data version fields in every published artifact.

### nflverse/nflseedR and nfl4th

[nflseedR](https://github.com/nflverse/nflseedR) is useful for season simulation and playoff-path logic. [nfl4th](https://github.com/nflverse/nfl4th) demonstrates a focused decision model rather than an all-purpose monolith. Adopt:

- simulate standings and playoff leverage from one coherent game distribution;
- isolate decision engines behind testable APIs;
- keep causal fourth-down policy analysis separate from descriptive team tendencies.

### ShamgarBN/nfl-bet-engine

[nfl-bet-engine](https://github.com/ShamgarBN/nfl-bet-engine) is a recent, small repository whose self-reported results have not been independently verified. Its architecture is nevertheless directly relevant: DuckDB facts, Parquet feature store, walk-forward tests, separate calibration, score-distribution simulation, direct market heads, ablation, tuning, prediction journaling, model cards, CLV, and explicit negative findings. Adopt the design patterns, not the published performance numbers.

Especially useful ideas:

- predict a joint score environment so moneyline, spread, and total probabilities cannot contradict each other;
- report top-edge slices without hiding the all-game result;
- preserve poor tail calibration and failed feature groups in the model card;
- use CLI workflows for backtest, ablation, tuning, training, and prediction;
- keep a prediction journal that cannot be overwritten after kickoff.

### gmalbert/baseball-predictions

[The sibling baseball repository](https://github.com/gmalbert/baseball-predictions) already uses a clearer `src/` split, raw/processed storage, schema validation, Parquet, calibration, pick history, drawdown, Kelly utilities, and page-specific modules. Reuse its proven organizational lessons:

- move shared loading and formatting out of the main UI;
- mirror source modules in tests;
- distinguish raw from processed artifacts;
- expose game-context factors and price movement without hard-coding them into UI functions.

## Research findings and consequences

### The market is the primary baseline

[Szalkowski and Nelson, *The Performance of Betting Lines for Predicting the Outcome of NFL Games*](https://arxiv.org/abs/1211.4000), studied 2,560 NFL games and emphasized opening/closing line information and the 52.38% -110 break-even rate. The result is not a timeless betting rule; it is evidence that line state and era must be preserved.

Implementation consequence:

- store open, bet-time, and close snapshots;
- evaluate point and probability error against the market;
- treat any historical situational edge as a hypothesis requiring a fresh forward test;
- never call 52% ATS accuracy profitable without price and uncertainty.

[Sapra, *Evidence of Betting Market Intraseason Efficiency and Interseason Overreaction*](https://journals.sagepub.com/doi/10.1177/1527002507311726) found the NFL point spread to be a significant predictor and within-season markets broadly efficient in the examined period. A model should therefore demonstrate incremental value over the line, not only outcome prediction.

### Calibration matters more than classification accuracy

[Walsh and Joshi, *Machine learning for sports betting: should model selection be based on accuracy or calibration?*](https://arxiv.org/abs/2303.06021) argues empirically that calibration is more important than accuracy for betting decisions. [Guo et al., *On Calibration of Modern Neural Networks*](https://arxiv.org/abs/1706.04599) shows that strong classifiers can still be miscalibrated and compares post-hoc approaches. [Rossellini et al., *Can a calibration metric be both testable and actionable?*](https://proceedings.mlr.press/v291/rossellini25a.html) explains limitations of common expected calibration error and proposes a cutoff-oriented alternative.

Implementation consequence:

- publish Brier score, log loss, reliability tables, ECE, MCE, and calibration at actual betting cutoffs;
- fit calibration on a held-out chronological block;
- recalibrate only with data that would have been resolved at the time;
- choose a decision policy on calibration data and report it once on untouched test data.

### Random splits and global transforms can create fictional edge

[Hewamalage et al., *Forecast evaluation for data scientists: common pitfalls and best practices*](https://link.springer.com/article/10.1007/s10618-022-00894-5) documents leakage from full-series preprocessing and inappropriate validation. The exact lesson applies to sports features: full-sample scaling, target encoding, rolling means, rankings, and imputation can all leak.

Implementation consequence:

- every feature must have a cutoff and availability time;
- shift before rolling;
- fit imputers/encoders only on a fold's training data;
- use season-forward folds and a final untouched season;
- retain a feature denylist for outcomes, returns, and postgame values.

### EPA and win probability are better foundations than box-score folklore

[Yurko, Ventura, and Horowitz, *nflWAR: A Reproducible Method for Offensive Player Evaluation in Football*](https://arxiv.org/abs/1802.00998) develops public-data expected-points and win-probability models and uses multilevel modeling to isolate player value with uncertainty.

Implementation consequence:

- aggregate EPA/play, success rate, early-down behavior, pressure proxy, explosive rate, and opponent strength;
- build quarterback and availability effects with partial pooling rather than fixed hand-tuned deductions;
- use resampling or hierarchical uncertainty for player contribution;
- benchmark simpler interpretable models before deep sequence models.

### Weather should be continuous and conditional

[Craig, *Quantifying the Impact of Temperature and Wind on NFL Passing and Rushing Performance*](https://scholarship.claremont.edu/cmc_theses/830/) evaluates temperature, wind, home/road response, and weather transitions. The strongest practical implication is not a universal threshold; it is interaction and acclimation.

Implementation consequence:

- preserve continuous temperature and wind;
- distinguish indoor, outdoor, missing, forecast, and observed states;
- add nonlinear cold degree, wind over threshold, and interaction terms;
- model team/player acclimation and uncertainty only after sufficient sample size;
- capture forecast issue time so revised forecasts do not leak.

### Rest effects changed with league rules

[Lopez et al., *Bye-bye, bye advantage: estimating the competitive impact of rest differential in the NFL*](https://www.frontiersin.org/journals/behavioral-economics/articles/10.3389/frbhe.2024.1479832/full) finds that the bye advantage changed after the 2011 collective bargaining agreement and suggests practice time is central.

Implementation consequence:

- use rest differential, not only separate team rest flags;
- include rule/era interactions;
- distinguish bye, mini-bye, Thursday short week, Monday-to-Sunday, and postseason schedules;
- do not assume one historical rest coefficient is stationary.

### Price movement contains information but must not be mined retroactively

The NFL lines paper above and [Gandar, Zuber, and Dare on opening versus closing totals](https://journals.sagepub.com/doi/10.1177/152700250000100205) support treating line movement as information aggregation. The latter studies NBA totals, so it is a methodological analogy, not NFL evidence.

Implementation consequence:

- define the decision horizon before measuring movement;
- never use the closing line as a feature for a pick supposedly placed earlier;
- use close only as a benchmark and CLV outcome;
- measure move velocity, cross-book dispersion, key-number crossing, and stale quotes.

### Longshot bias and market segment behavior require price-band tests

[Borghesi, *The Implications of a Reverse Favourite-Longshot Bias in a Prediction Market*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2151538) examines NFL against-the-spread prediction-market trades. [Daunhawer, Schoch, and Kosub](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2977118) examine favorite-longshot bias in football betting more broadly. These studies do not justify a permanent “bet dogs” rule.

Implementation consequence:

- report calibration and ROI by no-vig probability band;
- model price as a baseline input;
- require segment findings to repeat across seasons and books;
- adjust for multiple hypothesis testing.

### Kelly is only as good as the probability estimate

[Lototsky and Pollok, *Kelly Criterion: From a Simple Random Walk to Lévy Processes*](https://arxiv.org/abs/2002.03448) formalizes long-run growth under an edge. [On Kelly Betting: Some Limitations](https://arxiv.org/abs/1710.01787) highlights practical constraints.

Implementation consequence:

- no positive-EV estimate means no bet;
- use fractional Kelly and a small hard cap only after calibration gates pass;
- reduce correlated exposure across same-game and same-team positions;
- simulate probability error and risk of ruin;
- default to flat shadow stakes during research.

## Research-backed data priority

1. Timestamped multi-book market snapshots and closing consensus.
2. Correct game results and settlement rules.
3. Prior-only EPA, success, pass/rush efficiency, and play volume.
4. Starting-quarterback identity and availability distribution.
5. Snap, route, target, carry, and dropback opportunity.
6. Injury/practice/inactive evidence with publication times.
7. Rest, schedule, travel, surface, roof, and continuous weather.
8. Offensive line, pass-rush, and secondary continuity proxies.
9. Coaching/coordinator and scheme regime changes.
10. Referee and penalty tendencies with partial pooling.

## Ideas intentionally rejected or deferred

- **Scraping proprietary PFF grades:** deferred for licensing, stability, and reproducibility reasons.
- **LLM-generated injury sentiment as a direct model feature:** deferred until sources, timestamps, and a frozen text pipeline exist.
- **Transformer/LSTM as the default:** deferred until simple models beat market/Elo baselines in season-forward tests.
- **Fixed divisional-dog or referee adjustments:** rejected unless learned and repeated out of sample.
- **Using closing lines as early-pick features:** rejected as direct leakage.
- **Promoting small high-confidence slices:** rejected without predeclared slice definitions and adequate sample size.

## Source-use standard

Every future source entry must record provider, URI, license/terms, collection method, observed time, available time, source version, content hash, row count, and failure state. A feature is ineligible if its historical availability cannot be reconstructed.
