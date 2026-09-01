# NFL Predictions — Implementation Priorities

> **Prepared:** 2026-08-24  
> **Scope:** Consolidated implementation queue following a code-level review of the planning, audit, architecture, and research documents.  
> **Status vocabulary:** **Complete (foundation)** = present in the uncommitted V2 package; **Partial** = legacy capability exists but is not trustworthy or production-integrated; **Open** = no end-to-end implementation found; **Blocked** = depends on a source, licensing decision, or verified historical availability.

## Executive decision

Do not build new predictive features, deep-learning models, Kelly sizing, or betting-product enhancements until the V2 foundation is committed, reproducible, supplied with point-in-time data, and used for an immutable shadow-prediction journal. The legacy application still has random train/test splitting, threshold selection on test data, mutable CSV artifacts, and UI-side effects. Those issues invalidate the evidence needed to rank model ideas by expected value.

The immediate goal is therefore **a trustworthy research-to-publication pipeline**, not a higher-confidence pick interface.

## Completion baseline

| Area | Status | Evidence / consequence |
|---|---|---|
| V2 contracts, features, market math, modeling, backtesting, warehouse DDL, CLI | **Complete (foundation)** | `nfl_predictor/` and deterministic V2 test modules are present, but uncommitted. |
| V2 CI workflow | **Complete (foundation)** | `.github/workflows/v2-quality.yml` installs requirements, compiles, and runs V2 tests. |
| V2 test verification in this review environment | **Complete (foundation)** | A pinned V2 environment is declared in `requirements-v2.txt`; all 31 deterministic V2 tests pass in the review environment. |
| Legacy game prediction and player-prop UI | **Partial** | Product pages and models exist, but legacy evaluation/publishing controls do not meet release gates. |
| Point-in-time source ingestion and warehouse loading | **Partial** | Local game and market CSV loaders now create hashed source-run records and quality events. Availability/player sources and scheduled source acquisition remain open. |
| Immutable prediction publishing, decision journal, settlement and CLV record | **Partial** | Atomic manifests and database-backed pass/shadow decision writes exist; a scheduled prediction publisher and final-result/closing-quote settlement job remain open. |
| Streamlit refactor and read-only artifact consumption | **Open** | `predictions.py` remains monolithic and performs operational work. |
| New data families and frontier models | **Blocked/Open** | Require source, licensing, availability-time reconstruction, and baseline ablation. |

## Priority queue

### P0 — Establish a reproducible, safe baseline (next 1–2 weeks)

| Order | Work item | Current status | Deliverable / acceptance criteria |
|---:|---|---|---|
| 0.1 | Commit and protect the V2 foundation | **Complete** | Committed as `c1a34ee`; CI runs compilation and the deterministic V2 suite. |
| 0.2 | Make local verification reproducible | **Complete** | `requirements-v2.txt` pins the V2 runtime and the full V2 suite passes locally. |
| 0.3 | Freeze and label legacy outputs | Open | Preserve current model artifacts with commit/hash metadata; label all legacy picks, ROI, confidence, and bankroll content “research only.” Remove unsupported positive-ROI / leakage-free claims. |
| 0.4 | Disable unvalidated bet sizing and promotion language | **Partial** | The main app now has a research-only warning, research labels, and a disabled bankroll tool. Remaining legacy recommendation language requires the P3 product migration. |
| 0.5 | Fix workflow ownership and failure visibility | Open | One writer promotes generated artifacts; scheduled workflows have concurrency controls, tracked scripts, and fail on missing/stale required inputs. No broad `continue-on-error` for core data. |

### P1 — Build the point-in-time data spine (weeks 2–5)

| Order | Work item | Current status | Deliverable / acceptance criteria |
|---:|---|---|---|
| 1.1 | Source-run manifests | **Partial** | Local game/market CSV ingestion records URI, availability time, content hash, row count, status, and errors. License metadata and remaining sources are open. |
| 1.2 | Warehouse loaders and quality checks | **Partial** | Game and market loaders, FK validation, duplicate handling, and quality events exist. Team/player/availability loaders and scheduled staleness checks are open. |
| 1.3 | Timestamped market capture | Open | At least one reproducible legal provider; snapshots at open, declared checkpoints, decision time, and close with book, side, line, price, observed time, and available time. |
| 1.4 | Immutable artifact publishing | **Partial** | Atomic JSON/table writers and a replay manifest are implemented. The scheduled publisher still needs feature-schema and source-hash enrichment. |
| 1.5 | Prediction journal and settlement | **Partial** | Database-backed shadow/pass decisions require a real prediction and market snapshot. Automated final-result settlement and close capture are open. |

### P2 — Prove the V2 baseline before adding features (weeks 5–8)

| Order | Work item | Current status | Deliverable / acceptance criteria |
|---:|---|---|---|
| 2.1 | Re-run season-forward replay from manifests | **Partial** | A 2023–2025, 855-prediction close-benchmark replay now executes and writes an atomic manifest. Earlier decision horizons await timestamped source adapters. |
| 2.2 | Benchmark report | **Partial** | A predeclared overall/by-season probability report is implemented and generated; market comparisons are marked unavailable until timestamped price inputs exist. Elo/form/PBP ablations remain open. |
| 2.3 | Predeclare policy selection | **Partial** | Flat-stake shadow decisions and a fail-closed evidence gate are implemented. A policy can only be selected after real calibration and price data are ingested. |
| 2.4 | Shadow mode | Open | Publish no-stake, immutable, time-stamped predictions through at least two forward seasons. Use flat tracking stakes only. |
| 2.5 | Promotion gate | Foundation complete; operation open | Promote a market only with 500+ settled decisions (or approved exception), positive mean CLV, reproducible positive post-vig ROI, acceptable drawdown, stable calibration, and no critical data-quality events. |

### P3 — Integrate the safe path into the product (weeks 8–12)

| Order | Work item | Current status | Deliverable / acceptance criteria |
|---:|---|---|---|
| 3.1 | Extract legacy services | Open | Separate data loading, prediction reads, markets, settlement, exports, and scheduled writes from `predictions.py`; page rendering is read-only. |
| 3.2 | V2 prediction reader | Open | Streamlit reads published `prediction_snapshot` artifacts only and displays cutoff, freshness, source, model version, uncertainty, and unresolved-data state. |
| 3.3 | Responsible-use UI | Open | Prominent research/no-bet state and uncertainty notice; only gate-passed markets can display a recommendation. |
| 3.4 | Transparent performance page | Partial | Replace raw accuracy/ROI claims with temporal out-of-sample metrics, reliability plots, CLV, data freshness, and a link to the run manifest. |
| 3.5 | Remove fragile legacy behavior | Open | Eliminate training, ESPN settlement, and recommendation logging from page loads; replace broad exception swallowing in changed paths with explicit state/error reporting. |

### P4 — Add data features in value order (only after P2 is operating)

1. **Quarterback identity, availability, and replacement scenarios** — highest likely value; requires timestamped evidence and a replayable publication history.
2. **Player opportunity facts** — snaps, routes, targets, carries, dropbacks with GSIS identities; prerequisite for trustworthy player props.
3. **Multi-book market quality** — best price, consensus, dispersion, movement, key crossings, stale quote controls; prerequisite for actual EV/CLV decisions.
4. **Opponent-adjusted EPA / strength of schedule** — use prior-only iterative calculations and compare to Elo/market baselines.
5. **Unit continuity** — offensive line, secondary, pass rush, receiving; only from reproducible participation sources.
6. **Travel, surface, roof, coaching regime, referee effects** — add one family at a time with season-forward ablations and partial pooling where sample size is thin.
7. **Licensed tracking / NGS extensions** — CPOE, separation, time to throw, pressure and coverage only after license and historical availability are documented.

### P5 — Defer until the prerequisites win (after P4 gates)

- LSTM/transformer game or prop models.
- Bayesian/multi-library ensembles.
- Same-game parlay optimization, correlated portfolio sizing, and Kelly.
- Drive-policy simulator, causal fourth-down analysis, and scenario tree UI.
- QRC, PFF/DVOA-derived features, and referee adjustments that depend on non-reproducible or licensed sources.
- Cosmetic analytics/UI expansion: radar charts, AI briefing, PDF bet slips, personal tracker, division race, and trend cards.

These are not rejected; they are sequenced behind data provenance, baseline performance, and policy validation.

## Operating rules

1. A feature cannot enter a training frame without a source ID, availability time, version, and point-in-time replay test.
2. Use the market as a benchmark, not as an unexamined label or a future feature.
3. Keep tuning, calibration, and decision-policy selection chronologically isolated from the final test period.
4. Each release must be reproducible from immutable source and model manifests.
5. A failed ablation or policy is recorded in a negative-results registry; it is not silently retried until it looks favorable.

## Definition of implementation readiness

The first production-facing V2 market is ready only when P0–P3 are complete and its P2 promotion gate passes. Until then, roadmap features should be built only as isolated, source-gated research work and surfaced as analysis—not betting recommendations.
