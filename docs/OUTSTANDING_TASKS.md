# Outstanding Tasks Across `docs/`

This file consolidates the remaining work referenced in the `docs/` folder after reviewing all documentation.

> Completed items are already indicated in their source docs via `✅ COMPLETED`, explicit deployment notes, or checked boxes. This file focuses on work still described as future, planned, or pending.

## 1. docs/ROADMAP.md
- [ ] Mobile-responsive design improvements
- [ ] A/B testing framework for UI improvements
- [ ] Analytics integration (PostHog, Mixpanel, or similar)
- [ ] SEO optimization and landing page

## 2. docs/PLAYER_PROPS_ROADMAP.md
- [ ] Weather impact integration for player props
- [x] Player usage trends (snap share, target share, red zone touches) — **target_share added to aggregators.py via PBP calculation**
- [x] Multi-model ensemble for player props — **XGBoost + LightGBM soft-voting ensemble added to `player_props/models.py`**
- [ ] Complete Phase 3 analytics/ROI tracker and parlay builder UI if not already present in current app

## 3. docs/PLAYER_PROPS_SUMMARY.md
- [x] Create `player_props/models.py` baseline models — **already existed**
- [x] Create `player_props/train_models.py` training pipeline — **created; full CLI aggregation + train orchestrator**
- [x] Persist saved player prop model files under `player_props/models/` — **29 JSON files already present**
- [ ] Generate and validate prop predictions for upcoming weeks

## 4. docs/NEW_MODELS_ROADMAP.md
- [x] Implement QB passing yards model — **already existed**
- [x] Implement RB rushing yards + TD model — **already existed**
- [ ] Implement additional new models described in the roadmap (e.g. live in-game models, transformer/LSTM ideas if still planned)

## 5. docs/FIX_LFS_BANDWIDTH_QUOTA.md
- [x] Update GitHub Actions workflows to use `actions/checkout@v4` with `lfs: false` — **all 5 workflows already updated**
- [x] Add caching steps for historical data files in CI workflows — **`actions/cache@v4` already in nightly-update.yml**
- [x] Create or finalize a smart update script to avoid LFS downloads — **`update_pbp_smart.py` already exists**

## 6. docs/WEEKLY_MODEL_PERFORMANCE_WORKFLOW.md
- [x] Build the end-to-end weekly analysis automation script or GitHub Action — **`.github/workflows/weekly-model-performance.yml` created**
- [x] Confirm weekly performance report persistence/storage location — **committed to `data_files/accuracy_results_*.json`**
- [x] Ensure current-season play-by-play update flow is fully automated — **`update_pbp_smart.py` + nightly workflow covers this**

## 7. docs/ROADMAP_SCRAPING_PIPELINES.md
- [ ] Implement PFR boxscore URL collector and game data scraper pipelines *(requires paid PFF subscription or complex Selenium setup — deferred)*
- [ ] Implement PFF grade adapter and ranking feature integration if premium data is used *(deferred — requires paid access)*
- [ ] Validate scraper reliability and proxy handling for production use *(deferred)*

## 8. docs/ROADMAP_NEW_DATA_SOURCES.md
- [ ] Add support for PFF team grades and PFR boxscore metadata *(deferred — requires paid access)*
- [ ] Integrate new data sources with feature engineering and retraining workflow *(deferred)*

## 9. docs/MODEL_IMPROVEMENTS.md and docs/IMPROVEMENTS_COMPLETED.md
- [ ] Continue spread model calibration and validation work
- [x] Add model ensembling — **XGB + LightGBM soft-voting ensemble added to all three main game models in `nfl-gather-data.py`**
- [ ] Hyperparameter optimization (currently fixed params; `RandomizedSearchCV` not yet wired in)
- [ ] Monitor and iterate on spread betting performance once 2025 data accumulates
- [x] EV / threshold logic in place — existing `calculate_spread_ev_threshold()` updated to use ensemble probabilities

## 10. docs/NEW_FEATURES_DEC13.md
- [x] Momentum/rest/weather features already in production — **18 features confirmed in `nfl-gather-data.py`**
- [x] Features appear in `best_features_*.txt` via Monte Carlo selection at end of training pipeline
- [ ] Track post-retraining performance to ensure features improved results *(ongoing — visible in `pages/4_Model_Performance.py`)*

## 11. Workflow: Player Prop Retraining (new)
- [x] Add player prop retraining step to `nightly-update.yml` — **`python player_props/train_models.py --skip-aggregation` added**
- [x] Upload `player_props/models/model_metrics.json` as nightly artifact

## Notes on Completed Content
- `docs/ROADMAP.md`, `docs/PLAYER_PROPS_SUMMARY.md`, `docs/IMPROVEMENTS_COMPLETED.md`, `docs/SPREAD_THRESHOLD_CHANGE.md`, and `docs/MODEL_FIX_PLAN.md` already include clearly marked completed items.
- `docs/PLAYER_PROPS_QUICKSTART.md` and `docs/PLAYER_PROPS_ROADMAP.md` already document phase status and next-step actions, indicating where work is still pending.
- The consolidated open tasks in this file are based on language in the docs describing future enhancements, roadmap phases, and checklist items not marked completed.

## Remaining Open Items (not yet feasible without external dependencies)
- PFF premium data integration (requires subscription)
- PFR Selenium scraping (ToS risk + proxy infrastructure)
- Live in-game models (requires real-time data stream)
- LSTM / Transformer models (off-season, not yet scoped)
- Mobile-responsive Streamlit design (framework limitation)
- A/B testing / PostHog analytics (requires account setup)

