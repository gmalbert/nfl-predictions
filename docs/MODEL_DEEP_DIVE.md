# NFL Predictions — Model Deep Dive & Enhancement Recommendations

> **Implementation review — 2026-08-24.** V2 completes the prior-only rolling pattern, PBP EPA/success/pass-rush feature primitives, weather corrections, chronological validation, held-out isotonic calibration, and a coherent score-distribution model. It does **not** implement CPOE ingestion, pressure/blitz source adapters, the proposed XGBoost/LightGBM/CatBoost ensemble, or a V2 production training/publishing path. The legacy pipeline still uses random splits, so its leakage-prevention claim is not complete.

> Generated: 2026-07-31

---

## Current Architecture

XGBoost models for spread, moneyline, and totals trained on `nfl_data_py`
historical data. Recent fix (Dec 2025) corrected spread model inversion.

---

## Critical Improvements

### 1. Temporal Leakage Prevention

**Current state**: Verify all rolling features use `shift(1)` before rolling.

```python
# CORRECT pattern — prevents data leakage
df["off_epa_r4"] = (
    df.groupby("team")["epa_per_play"]
    .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
)

# WRONG — uses current game data
df["off_epa_r4"] = (
    df.groupby("team")["epa_per_play"]
    .transform(lambda x: x.rolling(4, min_periods=1).mean())  # BUG: includes current game
)
```

### 2. NGS Feature Integration

Next-Gen Stats (EPA, CPOE, success rate) are the strongest predictive
features. Priority implementation:

```python
PRIORITY_NGS_FEATURES = [
    "off_epa_r4",           # Offensive EPA per play (last 4 games)
    "def_epa_r4",           # Defensive EPA allowed per play
    "pass_epa_r4",          # Passing EPA per dropback
    "rush_epa_r4",          # Rushing EPA per carry
    "cpoe_r4",              # QB CPOE (completion % over expected)
    "success_rate_r4",      # Offensive success rate
    "def_success_rate_r4",  # Defensive success rate allowed
    "pressure_rate_r4",     # % of dropbacks with pressure
    "blitz_rate_r4",        # Opposing team's blitz rate
]
```

### 3. Bayesian Ensemble (Recommended Architecture)

```python
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def build_nfl_ensemble(feature_weights: dict | None = None) -> VotingClassifier:
    """
    Soft-voting ensemble. Weights based on validation performance.
    Expected: +2-3% AUC vs single XGBoost.
    """
    estimators = [
        ("xgb", XGBClassifier(n_estimators=800, max_depth=5, learning_rate=0.02,
                               subsample=0.8, colsample_bytree=0.7,
                               scale_pos_weight=1.0)),
        ("lgb", LGBMClassifier(n_estimators=800, learning_rate=0.02,
                                num_leaves=31, min_data_in_leaf=20, verbose=-1)),
        ("cat", CatBoostClassifier(iterations=600, depth=5, learning_rate=0.03,
                                    verbose=False)),
    ]
    weights = feature_weights or [0.45, 0.35, 0.20]
    return VotingClassifier(estimators=estimators, voting="soft", weights=weights)
```

### 4. Spread Model Calibration Fix

The spread model's 50% threshold should be validated properly:

```python
from sklearn.calibration import calibration_curve
import numpy as np

def validate_spread_model_calibration(
    y_true: np.ndarray, y_prob: np.ndarray
) -> dict:
    """Validate spread model calibration."""
    # ECE calculation
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.mean() * abs(acc - conf)

    # Check threshold
    at_50 = y_true[y_prob >= 0.5].mean()
    return {
        "ece": round(ece, 4),
        "accuracy_at_50pct": round(at_50, 3),
        "recommended_threshold": 0.50 if at_50 > 0.52 else 0.53,
    }
```

### 5. Weather Feature Engineering

```python
def build_weather_features(game: dict) -> dict:
    """Quantify weather impact on NFL totals."""
    temp = game.get("temperature", 70)
    wind = game.get("wind_speed", 5)
    precip = game.get("precipitation_prob", 0)
    is_dome = game.get("is_dome", False)

    if is_dome:
        return {"weather_impact": 0.0, "cold_flag": 0, "wind_flag": 0}

    cold_adj = max(0, (45 - temp) / 45 * 1.5) if temp < 45 else 0
    wind_adj = max(0, (wind - 15) / 15 * 1.0) if wind > 15 else 0
    precip_adj = precip / 100 * 0.5

    return {
        "weather_total_reduction": cold_adj + wind_adj + precip_adj,
        "cold_flag": int(temp < 35),
        "wind_flag": int(wind > 20),
        "extreme_flag": int(cold_adj + wind_adj > 1.5),
    }
```

### 6. Target Variable Refinement for Spread Model

```python
def create_spread_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create spread betting target with proper bookmaker-line alignment.
    
    NOTE: Always use the closing spread line, not opening line.
    The model predicts vs CLOSING line for proper CLV measurement.
    """
    df["home_margin"] = df["home_score"] - df["away_score"]
    # Positive spread = home team favored
    # Negative spread = away team favored
    df["home_covered"] = df["home_margin"] + df["spread_line"] > 0
    # Remove pushes for cleaner binary target
    df = df[df["home_margin"] + df["spread_line"] != 0]
    return df
```

---

## Model Performance Targets

| Metric | Current | Target (6 months) | Target (12 months) |
|--------|---------|-------------------|-------------------|
| Spread AUC | 0.54 | 0.57 | 0.59 |
| Moneyline AUC | 0.58 | 0.61 | 0.63 |
| Total AUC | 0.56 | 0.59 | 0.61 |
| Spread Win Rate | 53% | 55% | 56% |
| Spread ROI | +1.4% | +3% | +5% |

---

## Feature Importance Rankings (Current)

Based on typical NFL model feature importance, recommend this priority:

1. **Offensive EPA per play** (most important)
2. **Quarterback quality (QRC)**
3. **Defensive EPA allowed**
4. **Home/away/neutral**
5. **Rest advantage**
6. **Weather adjustments**
7. **Injury-adjusted power ratings**
8. **Divisional game flag**
9. **ATS trend (team covers rate)**
10. **Referee tendencies**
