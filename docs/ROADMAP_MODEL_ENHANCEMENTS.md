# Roadmap: Model Enhancements

> Inspired by [thadhutch/sports-quant](https://github.com/thadhutch/sports-quant) NFL ensemble training pipeline.

## Overview

Our current pipeline trains a single XGBoost model per market (spread, moneyline, totals). The sports-quant project uses a significantly more sophisticated approach: **ensemble of 50 models → filter by accuracy → select top 3 by season-weighted score → require consensus**. This document outlines how to adopt these techniques.

---

## 1. Ensemble Training (50 Models Per Game-Day)

### Current Approach
```
1 XGBoost model per market → predict → output probabilities
```

### Sports-Quant Approach
```
50 XGBoost models (different random seeds)
→ Filter: keep only models with >50% validation accuracy
→ Rank: season-weighted accuracy (current season vs last season)
→ Select: top 3 models
→ Consensus: all 3 must agree on prediction
→ Score: weighted algorithm score from confidence bins
```

### Implementation

#### Step 1: Multi-Seed Model Training

Create `scripts/ensemble_trainer.py`:

```python
"""Ensemble XGBoost training with multi-seed diversity and accuracy filtering."""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Confidence bins: 50% to 100% in 5-point increments
CONF_BINS = np.arange(0.5, 1.05, 0.05)
CONF_LABELS = [f"{int(b * 100)}-{int((b + 0.05) * 100)}%" for b in CONF_BINS[:-1]]


@dataclass
class TrainedModel:
    """Container for a single trained XGBoost model and its validation metrics."""
    model: XGBClassifier
    seed: int
    overall_accuracy: float
    current_season_accuracy: float
    last_season_accuracy: float
    confidence_accuracy: pd.DataFrame = field(repr=False)


def build_classifier(seed: int, hyperparameters: dict | None = None) -> XGBClassifier:
    """Create an XGBClassifier with a specific random seed."""
    defaults = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    if hyperparameters:
        defaults.update(hyperparameters)
    return XGBClassifier(random_state=seed, **defaults)


def train_ensemble(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    seasons_full: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    n_models: int = 50,
    test_size: float = 0.2,
    accuracy_threshold: float = 0.50,
    hyperparameters: dict | None = None,
) -> list[TrainedModel]:
    """Train n_models XGBoost classifiers with different seeds.

    Only models exceeding accuracy_threshold on validation data are kept.
    Each model gets a different train/val split via its unique seed.
    """
    models: list[TrainedModel] = []

    for model_idx in range(n_models):
        seed = 42 + model_idx

        try:
            X_train, X_val, y_train, y_val, _, seasons_val = train_test_split(
                X_full, y_full, seasons_full,
                test_size=test_size,
                random_state=seed,
                stratify=y_full if len(np.unique(y_full)) > 1 else None,
            )
        except ValueError as exc:
            logger.debug("train_test_split failed (model %d): %s", model_idx + 1, exc)
            continue

        clf = build_classifier(seed, hyperparameters)
        try:
            clf.fit(X_train, y_train)
        except Exception as exc:
            logger.debug("Training failed (model %d): %s", model_idx + 1, exc)
            continue

        # Validate
        y_val_pred = clf.predict(X_val)
        y_val_proba = clf.predict_proba(X_val)
        val_conf = np.max(y_val_proba, axis=1)

        overall_acc = accuracy_score(y_val, y_val_pred)
        if overall_acc <= accuracy_threshold:
            continue

        # Per-season accuracy
        current_season = seasons_val.max()
        last_season = current_season - 1

        current_mask = seasons_val == current_season
        current_acc = (
            accuracy_score(y_val[current_mask], y_val_pred[current_mask])
            if current_mask.any() else 0.0
        )

        last_mask = seasons_val == last_season
        last_acc = (
            accuracy_score(y_val[last_mask], y_val_pred[last_mask])
            if last_mask.any() else 0.0
        )

        # Confidence-bin accuracy by season
        val_results = pd.DataFrame({
            "Actual": y_val.values,
            "Predicted": y_val_pred,
            "Confidence": val_conf,
            "Season": seasons_val.values,
        })
        val_results["Confidence Bin"] = pd.cut(
            val_results["Confidence"],
            bins=CONF_BINS, labels=CONF_LABELS, include_lowest=True,
        )
        val_results["Correct"] = (val_results["Actual"] == val_results["Predicted"]).astype(int)

        conf_acc = (
            val_results.groupby(["Confidence Bin", "Season"], observed=False)["Correct"]
            .mean()
            .reset_index(name="Accuracy")
        )

        models.append(TrainedModel(
            model=clf,
            seed=seed,
            overall_accuracy=overall_acc,
            current_season_accuracy=current_acc,
            last_season_accuracy=last_acc,
            confidence_accuracy=conf_acc,
        ))

    logger.info("%d / %d models passed threshold (%.0f%%)",
                len(models), n_models, accuracy_threshold * 100)
    return models
```

---

## 2. Season-Weighted Model Scoring

### Concept
Early in the season, there's limited current-season data, so the model should weight last season's accuracy more. As the season progresses, shift weight toward current-season accuracy.

The sports-quant approach: NFL season runs Sep 1 → Jan 15 (~137 days). The fraction of elapsed days determines the weight split.

### Implementation

```python
"""Season-weighted scoring and model selection."""

from datetime import datetime
import numpy as np
import pandas as pd
from ensemble_trainer import TrainedModel, CONF_BINS, CONF_LABELS


def compute_season_progress(current_date: datetime) -> tuple[float, float]:
    """Return (weight_current_season, weight_last_season) based on NFL calendar.

    Sep 1 (start): weight_current=0.0, weight_last=1.0
    Jan 15 (end):  weight_current=1.0, weight_last=0.0
    """
    if isinstance(current_date, (np.datetime64, pd.Timestamp)):
        current_date = pd.Timestamp(current_date).to_pydatetime()

    year = current_date.year
    if current_date.month >= 9:
        season_start = datetime(year, 9, 1)
        season_end = datetime(year + 1, 1, 15)
    else:
        season_start = datetime(year - 1, 9, 1)
        season_end = datetime(year, 1, 15)

    total_days = (season_end - season_start).days
    elapsed = (current_date - season_start).days
    pct = max(0.0, min(1.0, elapsed / total_days))

    return pct, 1.0 - pct


def score_and_select_models(
    models: list[TrainedModel],
    current_date: datetime,
    top_n: int = 3,
) -> list[dict]:
    """Score all models by season-weighted accuracy, return top N.

    Each model's score = weight_current * current_season_acc + weight_last * last_season_acc
    """
    weight_current, weight_last = compute_season_progress(current_date)

    scored = []
    for m in models:
        cur = m.current_season_accuracy if not np.isnan(m.current_season_accuracy) else 0.0
        last = m.last_season_accuracy if not np.isnan(m.last_season_accuracy) else 0.0
        weighted = weight_current * cur + weight_last * last
        scored.append({
            "trained_model": m,
            "weighted_accuracy": weighted,
        })

    # Sort descending by weighted accuracy, take top N
    scored.sort(key=lambda x: x["weighted_accuracy"], reverse=True)
    return scored[:top_n]
```

### Season Progress Example

| Date | Week | weight_current | weight_last |
|------|------|---------------|-------------|
| Sep 7, 2025 | 1 | 0.04 | 0.96 |
| Oct 12, 2025 | 6 | 0.30 | 0.70 |
| Nov 23, 2025 | 12 | 0.61 | 0.39 |
| Dec 28, 2025 | 17 | 0.86 | 0.14 |
| Jan 11, 2026 | WC | 0.96 | 0.04 |

---

## 3. Consensus Prediction

### Concept
Instead of trusting a single model, require **all top models to agree** on the same prediction. This dramatically reduces false positives — if 3 independently-trained models all predict the same class, confidence is much higher.

### Implementation

```python
"""Consensus prediction requiring all top models to agree."""


def predict_with_consensus(
    top_models: list[dict],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_weights: list[float] | None = None,
) -> pd.DataFrame | None:
    """Run top models on test data. Keep only consensus predictions.

    A consensus pick = all top models predict the same class.
    Returns DataFrame of consensus picks or None if no consensus.

    model_weights: how much to weight each model's score in final algorithm
                   score. Default: [0.4, 0.35, 0.25] for 3 models.
    """
    if model_weights is None:
        model_weights = [0.4, 0.35, 0.25]

    n_top = len(top_models)
    predictions = {}

    for idx, entry in enumerate(top_models):
        tm = entry["trained_model"]
        clf = tm.model
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        confidences = np.max(y_proba, axis=1)

        predictions[f"pred_{idx}"] = y_pred
        predictions[f"conf_{idx}"] = confidences

    result_df = pd.DataFrame(predictions, index=X_test.index)
    result_df["actual"] = y_test.values

    # Check consensus: all models agree
    pred_cols = [f"pred_{i}" for i in range(n_top)]
    consensus_mask = pd.Series(True, index=result_df.index)
    for i in range(1, n_top):
        consensus_mask &= result_df[pred_cols[0]] == result_df[pred_cols[i]]

    consensus = result_df[consensus_mask].copy()
    if consensus.empty:
        return None

    consensus["predicted"] = consensus[pred_cols[0]]
    consensus["avg_confidence"] = np.mean(
        [consensus[f"conf_{i}"] for i in range(n_top)], axis=0
    )

    # Weighted algorithm score
    consensus["algorithm_score"] = sum(
        consensus[f"conf_{i}"] * model_weights[i] for i in range(n_top)
    )

    return consensus[["actual", "predicted", "avg_confidence", "algorithm_score"]]
```

### Expected Impact
In the sports-quant pipeline, consensus filtering typically keeps 40–60% of games but with significantly higher accuracy on the remaining picks. This is a **quality over quantity** approach.

---

## 4. Walk-Forward Backtesting

### Concept
Instead of a single train/test split, simulate real-time prediction by training on all data before each game date and predicting that date's games. This is the most realistic evaluation.

### Implementation

```python
"""Walk-forward backtesting for realistic model evaluation."""


def walk_forward_backtest(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    date_col: str = "gameday",
    n_models_per_date: int = 50,
    top_n: int = 3,
    accuracy_threshold: float = 0.50,
    hyperparameters: dict | None = None,
) -> pd.DataFrame:
    """Walk-forward backtest: train on past, predict current date, advance.

    For each unique game date:
      1. Train ensemble on all games before this date
      2. Select top models by season-weighted accuracy
      3. Predict with consensus
      4. Record results

    Returns DataFrame of all consensus predictions across all dates.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    dates = df[date_col].unique()
    all_results = []

    for date_idx, current_date in enumerate(dates):
        train_df = df[df[date_col] < current_date]
        test_df = df[df[date_col] == current_date]

        if test_df.empty or len(train_df) < 100:
            continue

        train_seasons = train_df["season"].unique()
        if len(train_seasons) < 2:
            continue

        X_full = train_df[feature_cols]
        y_full = train_df[target_col]
        seasons_full = train_df["season"]

        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        # 1. Train ensemble
        models = train_ensemble(
            X_full, y_full, seasons_full, X_test, y_test,
            n_models=n_models_per_date,
            accuracy_threshold=accuracy_threshold,
            hyperparameters=hyperparameters,
        )

        if len(models) < top_n:
            continue

        # 2. Select top models
        top_models = score_and_select_models(
            models, current_date, top_n=top_n,
        )

        # 3. Predict with consensus
        consensus = predict_with_consensus(top_models, X_test, y_test)
        if consensus is not None:
            consensus["date"] = current_date
            consensus["season"] = test_df["season"].values[0]
            all_results.append(consensus)

        if (date_idx + 1) % 10 == 0:
            logger.info("Backtest progress: %d / %d dates", date_idx + 1, len(dates))

    if not all_results:
        return pd.DataFrame()

    results = pd.concat(all_results, ignore_index=True)
    results["correct"] = (results["actual"] == results["predicted"]).astype(int)

    accuracy = results["correct"].mean()
    coverage = len(results) / len(df)
    logger.info("Backtest: %.1f%% accuracy on %.1f%% of games (%d picks)",
                accuracy * 100, coverage * 100, len(results))

    return results
```

### Key Difference from Current Pipeline
Our current approach trains once on all historical data. Walk-forward backtesting:
- Trains **separately for each game date**
- Uses **only past data** (no future leakage)
- Provides **realistic accuracy estimates**
- Is computationally expensive (50 models × N dates) but more trustworthy

---

## 5. Confidence-Bin Accuracy Tracking

### Concept
Track how accurate the model is **within each confidence range**. A well-calibrated model should be more accurate when it's more confident.

### Implementation

```python
"""Track and display model accuracy by confidence bin."""


def compute_confidence_bin_accuracy(
    predictions: pd.DataFrame,
    pred_col: str = "predicted",
    actual_col: str = "actual",
    conf_col: str = "avg_confidence",
) -> pd.DataFrame:
    """Compute accuracy within each 5%-wide confidence bin.

    Bins: 50-55%, 55-60%, 60-65%, ..., 95-100%
    """
    bins = np.arange(0.5, 1.05, 0.05)
    labels = [f"{int(b*100)}-{int((b+0.05)*100)}%" for b in bins[:-1]]

    predictions = predictions.copy()
    predictions["conf_bin"] = pd.cut(
        predictions[conf_col], bins=bins, labels=labels, include_lowest=True,
    )
    predictions["correct"] = (predictions[actual_col] == predictions[pred_col]).astype(int)

    summary = (
        predictions.groupby("conf_bin", observed=False)
        .agg(
            total=("correct", "count"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )
    summary["accuracy"] = summary["accuracy"].fillna(0).round(4)

    return summary


def display_confidence_calibration(summary: pd.DataFrame) -> None:
    """Print a confidence calibration table for the Model Performance page."""
    print("\nConfidence Bin  | Games | Correct | Accuracy")
    print("-" * 50)
    for _, row in summary.iterrows():
        print(f"  {row['conf_bin']:>12}  | {row['total']:>5} | {row['correct']:>7} | {row['accuracy']:.1%}")
```

### Streamlit Integration

Add to `pages/4_Model_Performance.py`:

```python
import streamlit as st
import plotly.express as px

# After loading backtest results...
conf_summary = compute_confidence_bin_accuracy(backtest_results)

st.subheader("Confidence Calibration")
fig = px.bar(
    conf_summary,
    x="conf_bin",
    y="accuracy",
    text="total",
    labels={"conf_bin": "Confidence Range", "accuracy": "Accuracy", "total": "Games"},
    title="Model Accuracy by Confidence Level",
)
fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Break-even")
fig.update_layout(yaxis_tickformat=".0%")
st.plotly_chart(fig, use_container_width=True)
```

---

## 6. Algorithm Score (Weighted Confidence-Bin Score)

### Concept
The sports-quant project computes a "Final Algorithm Score" that blends each top model's confidence-bin accuracy, weighted by both:
1. **Model rank** (best model gets weight 0.40, second 0.35, third 0.25)
2. **Season progress** (current season accuracy vs last season accuracy)

This score represents the historical reliability of the model at that specific confidence level.

### Formula

```
For each model i in top 3:
  conf_bin = bin containing this prediction's confidence
  acc_current = model i's accuracy in this conf_bin for current season
  acc_last = model i's accuracy in this conf_bin for last season
  adjusted_score_i = weight_current * acc_current + weight_last * acc_last

Final Algorithm Score = Σ (model_weight_i × adjusted_score_i)
  where model_weights = [0.40, 0.35, 0.25]
```

### Why This Is Better Than Raw Confidence
Raw XGBoost confidence doesn't account for:
- Whether the model is actually accurate at that confidence level
- How much current-season signal is available
- Agreement across multiple models

The algorithm score captures all three dimensions.

---

## 7. Financial Simulation (Kelly Criterion)

### Concept
Instead of just tracking accuracy, simulate actual betting results with realistic vig.

### Implementation

```python
"""1% Kelly criterion betting simulation."""

import numpy as np
import pandas as pd


def simulate_betting(
    picks_df: pd.DataFrame,
    starting_capital: float = 100.0,
) -> pd.DataFrame:
    """Simulate betting with 1% Kelly criterion.

    Bet sizing: 1 unit = 1% of current capital
    Win:  +1 unit  (1:1 payout)
    Loss: -1.1 units (standard -110 vig)
    """
    df = picks_df.sort_values("date").copy()

    # Unit-based profit
    df["profit_units"] = np.where(
        df["correct"] == 1, 1.0, -1.1
    )
    df["cumulative_units"] = df["profit_units"].cumsum()

    # Dollar-based profit (compounding via 1% of rolling capital)
    capital = starting_capital
    cum_dollars = []
    for correct in df["correct"]:
        unit_size = capital * 0.01  # 1% Kelly
        profit = unit_size if correct == 1 else -1.1 * unit_size
        capital += profit
        cum_dollars.append(capital - starting_capital)

    df["cumulative_dollars"] = cum_dollars
    df["current_capital"] = [starting_capital + d for d in cum_dollars]

    return df


def compute_roi_stats(sim_df: pd.DataFrame, starting_capital: float = 100.0) -> dict:
    """Compute summary ROI statistics from simulation results."""
    total_bets = len(sim_df)
    wins = (sim_df["correct"] == 1).sum()
    win_rate = wins / total_bets if total_bets > 0 else 0

    final_capital = sim_df["current_capital"].iloc[-1] if len(sim_df) > 0 else starting_capital
    total_roi = (final_capital - starting_capital) / starting_capital
    total_units = sim_df["cumulative_units"].iloc[-1] if len(sim_df) > 0 else 0

    return {
        "total_bets": total_bets,
        "wins": wins,
        "losses": total_bets - wins,
        "win_rate": win_rate,
        "total_units": total_units,
        "total_roi_pct": total_roi * 100,
        "final_capital": final_capital,
        "break_even_rate": 1.1 / 2.1,  # ~52.38% needed to break even at -110
    }
```

### Break-Even Math
At standard -110 vig:
- Win: +$100, Loss: -$110
- Break-even win rate: 110 / (100 + 110) = **52.38%**
- Our current spread model hits ~46% at 50%+ confidence
- Consensus ensemble should push this above break-even

---

## 8. Integration Plan

### Phase 1: Ensemble Training (Immediate)
1. Add `train_ensemble()` to `nfl-gather-data.py` alongside existing single-model training
2. Keep existing single-model as fallback
3. Compare accuracy: ensemble consensus vs single model

### Phase 2: Season-Weighted Scoring (Week 2)
1. Implement `compute_season_progress()` and `score_and_select_models()`
2. Add season progress display to predictions dashboard

### Phase 3: Walk-Forward Backtest (Week 3)
1. Implement `walk_forward_backtest()` as a standalone script
2. Run on 2020–2024 data to establish baseline metrics
3. Compare with current test-split evaluation

### Phase 4: Financial Simulation (Week 4)
1. Add `simulate_betting()` to Model Performance page
2. Display cumulative P/L charts, ROI, win rate
3. Show break-even line for reference

### Config-Driven Setup

Create `model_config.yaml` (inspired by sports-quant):

```yaml
spread:
  models_to_train: 50
  top_models: 3
  accuracy_threshold: 0.50
  model_weights: [0.40, 0.35, 0.25]
  starting_capital: 100.0
  hyperparameters:
    objective: binary:logistic
    eval_metric: logloss
  backtest:
    min_training_seasons: 2
    test_size: 0.2

moneyline:
  models_to_train: 50
  top_models: 3
  accuracy_threshold: 0.28
  model_weights: [0.40, 0.35, 0.25]
  starting_capital: 100.0

totals:
  models_to_train: 50
  top_models: 3
  accuracy_threshold: 0.50
  model_weights: [0.40, 0.35, 0.25]
  starting_capital: 100.0
```

---

## 9. Multi-Class Totals Model

### Current vs Sports-Quant
- **Our model**: Binary classification (over/under)
- **Sports-quant**: 3-class classification (over/under/push) with `multi:softprob`

### Why 3-Class Matters
Pushes (~3-5% of games) are currently misclassified as either over or under. A 3-class model can:
- Correctly identify push-likely games
- Avoid betting on games near the line
- Improve calibration for over/under predictions

### Implementation Change

```python
# In nfl-gather-data.py, for totals model:

# Current: binary target
# df['total_result'] = (total_score > ou_line).astype(int)

# New: 3-class target
def classify_total(row):
    total = row['home_score'] + row['away_score']
    if total > row['ou_line']:
        return 1  # over
    elif total < row['ou_line']:
        return 0  # under
    else:
        return 2  # push

df['total_result'] = df.apply(classify_total, axis=1)

# Update XGBoost params
xgb_params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
}
```

---

## Summary

| Enhancement | Expected Impact | Complexity |
|------------|----------------|------------|
| Ensemble (50 models) | +5-10% accuracy on consensus picks | Medium |
| Season-weighted scoring | Better early-season predictions | Low |
| Consensus prediction | Higher precision, fewer false positives | Low |
| Walk-forward backtest | Realistic accuracy estimates | Medium |
| Confidence-bin tracking | Better calibration visibility | Low |
| Financial simulation | Concrete ROI metrics | Low |
| 3-class totals | Better push handling | Low |
| Config-driven training | Easier experimentation | Low |
