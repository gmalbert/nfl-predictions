# Roadmap: New Features & Processing Pipeline

> Inspired by [thadhutch/sports-quant](https://github.com/thadhutch/sports-quant) NFL feature engineering pipeline.

## Overview

The sports-quant project demonstrates a disciplined feature engineering approach: raw PFF grade data → rolling season averages → per-date rankings → model features. This document covers new features we can adopt, the processing pipeline, and additional suggestions.

---

## 1. PFF Rank Features (22 Features)

### Concept
Instead of using raw PFF grades (which vary in scale), convert season-to-date rolling averages into **per-date rankings** (1–32). Rankings are ordinal and robust to scale differences.

### Feature List

The sports-quant model uses exactly **23 features**: 22 PFF rank features + 1 O/U line.

| Feature | Description |
|---------|-------------|
| `home-off-avg-rank` | Home team offense ranking |
| `away-off-avg-rank` | Away team offense ranking |
| `home-pass-avg-rank` | Home team passing ranking |
| `away-pass-avg-rank` | Away team passing ranking |
| `home-pblk-avg-rank` | Home team pass blocking ranking |
| `away-pblk-avg-rank` | Away team pass blocking ranking |
| `home-recv-avg-rank` | Home team receiving ranking |
| `away-recv-avg-rank` | Away team receiving ranking |
| `home-run-avg-rank` | Home team rushing ranking |
| `away-run-avg-rank` | Away team rushing ranking |
| `home-rblk-avg-rank` | Home team run blocking ranking |
| `away-rblk-avg-rank` | Away team run blocking ranking |
| `home-def-avg-rank` | Home team defense ranking |
| `away-def-avg-rank` | Away team defense ranking |
| `home-rdef-avg-rank` | Home team run defense ranking |
| `away-rdef-avg-rank` | Away team run defense ranking |
| `home-tack-avg-rank` | Home team tackling ranking |
| `away-tack-avg-rank` | Away team tackling ranking |
| `home-prsh-avg-rank` | Home team pass rush ranking |
| `away-prsh-avg-rank` | Away team pass rush ranking |
| `home-cov-avg-rank` | Home team coverage ranking |
| `away-cov-avg-rank` | Away team coverage ranking |
| `ou_line` | Vegas Over/Under line |

### Why Ranks Beat Raw Values
- **Scale invariant**: Rank 1 always means best, regardless of absolute grade
- **Robust to outliers**: One extreme grade doesn't skew the feature
- **Interpretable**: Model learns "top-5 defense vs bottom-5 offense" patterns
- **Stable across seasons**: Rankings are comparable year-to-year

---

## 2. Rolling Average Processing Pipeline

### Current vs Proposed

**Current** (our pipeline): Rolling stats computed in `nfl-gather-data.py` with complex pandas operations.

**Proposed** (sports-quant style): Simple iterative approach that's easier to verify for leakage.

### Implementation

Create `scripts/processing/compute_rolling_stats.py`:

```python
"""Compute rolling season-average statistics for team features.

The key invariant: for each game row, averages use ONLY prior games
in the current season. Averages reset at the start of each new season.
No data leakage is possible because we compute before updating.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def initialize_team_stats(stat_columns: list[str]) -> dict:
    """Initialize zeroed stat counters for a team."""
    return {stat: 0.0 for stat in stat_columns}


def calculate_avg(cumulative: dict, game_count: int) -> dict:
    """Calculate average stats from cumulative totals."""
    if game_count > 0:
        return {stat: cumulative[stat] / game_count for stat in cumulative}
    return {stat: 0.0 for stat in cumulative}


def compute_rolling_averages(
    df: pd.DataFrame,
    stat_columns: list[str],
    date_col: str = "gameday",
    season_col: str = "season",
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
) -> pd.DataFrame:
    """Compute rolling season-to-date averages for specified stat columns.

    For each game:
    1. Look up the team's cumulative stats from prior games this season
    2. Compute average = cumulative / games_played
    3. Store the average as pre-game features (no leakage)
    4. Update cumulative stats with current game values

    Args:
        df: Game data sorted by date. Must have columns:
            - date_col, season_col, home_team_col, away_team_col
            - For each stat in stat_columns: home-{stat} and away-{stat}
        stat_columns: List of stat category names (e.g., ["off", "pass", "def"])

    Returns:
        DataFrame with new columns: home-{stat}-avg, away-{stat}-avg for each stat
    """
    df = df.sort_values(date_col).reset_index(drop=True)

    # Track cumulative stats per team
    team_stats: dict[str, dict] = {}

    for idx, row in df.iterrows():
        home = row[home_team_col]
        away = row[away_team_col]
        season = row[season_col]

        # Initialize or reset on new season
        for team in (home, away):
            if team not in team_stats:
                team_stats[team] = {
                    "stats": initialize_team_stats(stat_columns),
                    "count": 0,
                    "season": season,
                }
            elif team_stats[team]["season"] != season:
                # New season: reset accumulators
                team_stats[team] = {
                    "stats": initialize_team_stats(stat_columns),
                    "count": 0,
                    "season": season,
                }

        # Step 1: STORE averages BEFORE current game (no leakage)
        avg_home = calculate_avg(team_stats[home]["stats"], team_stats[home]["count"])
        avg_away = calculate_avg(team_stats[away]["stats"], team_stats[away]["count"])

        for stat in stat_columns:
            df.at[idx, f"home-{stat}-avg"] = avg_home[stat]
            df.at[idx, f"away-{stat}-avg"] = avg_away[stat]

        # Step 2: UPDATE cumulative stats WITH current game
        for stat in stat_columns:
            team_stats[home]["stats"][stat] += row[f"home-{stat}"]
            team_stats[away]["stats"][stat] += row[f"away-{stat}"]
        team_stats[home]["count"] += 1
        team_stats[away]["count"] += 1

    # Optimize dtypes
    for stat in stat_columns:
        for prefix in ("home", "away"):
            col = f"{prefix}-{stat}-avg"
            df[col] = df[col].astype("float32")

    logger.info("Computed rolling averages for %d stats", len(stat_columns))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example: compute rolling averages for existing play-by-play derived stats
    input_path = Path("data_files/nfl_games_historical.csv")
    df = pd.read_csv(input_path, sep="\t")

    # Define stat columns that exist as home-{stat} and away-{stat} in the data
    # Adapt this to your actual column naming
    example_stats = ["points_scored", "yards_gained", "turnovers"]

    df = compute_rolling_averages(df, example_stats)
    print(df[[f"home-{s}-avg" for s in example_stats]].head(10))
```

---

## 3. Per-Date Team Rankings

### Concept
After computing rolling averages, rank all teams for each game date. This converts continuous averages into ordinal features that XGBoost can split efficiently.

### Implementation

Create `scripts/processing/compute_rankings.py`:

```python
"""Convert rolling averages into per-date team rankings.

For each date, rank all 32 teams based on their rolling average for each stat.
Rank 1 = best (highest value for offensive stats, depends on context for defensive).
Only uses data available up to that date (no lookahead).
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def compute_per_date_rankings(
    df: pd.DataFrame,
    stat_columns: list[str],
    date_col: str = "gameday",
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Rank teams per game date based on rolling averages.

    For each unique date in the dataset:
    1. Gather the latest rolling average for each team (from all prior rows)
    2. Rank teams 1-32 for each stat category
    3. Assign ranks to home/away columns for games on that date

    Args:
        df: DataFrame with rolling average columns (home-{stat}-avg, away-{stat}-avg)
        stat_columns: List of stat categories
        higher_is_better: If True, rank 1 = highest average value
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    dates = sorted(df[date_col].unique())

    for date in dates:
        # Collect latest average for each team from ALL rows up to this date
        prior_mask = df[date_col] <= date
        prior_df = df[prior_mask]

        team_avgs: dict[str, dict[str, float]] = {}

        for _, row in prior_df.iterrows():
            # Home team averages
            home = row[home_team_col]
            team_avgs[home] = {
                stat: row[f"home-{stat}-avg"] for stat in stat_columns
            }
            # Away team averages
            away = row[away_team_col]
            team_avgs[away] = {
                stat: row[f"away-{stat}-avg"] for stat in stat_columns
            }

        # Rank for each stat
        for stat in stat_columns:
            values = {t: avgs[stat] for t, avgs in team_avgs.items()}
            sorted_teams = sorted(
                values, key=values.get, reverse=higher_is_better
            )
            rankings = {team: rank + 1 for rank, team in enumerate(sorted_teams)}

            # Apply to games happening on this date
            game_mask = df[date_col] == date
            for idx in df[game_mask].index:
                home = df.at[idx, home_team_col]
                away = df.at[idx, away_team_col]
                df.at[idx, f"home-{stat}-avg-rank"] = float(rankings.get(home, 16))
                df.at[idx, f"away-{stat}-avg-rank"] = float(rankings.get(away, 16))

    # Optimize dtypes
    for stat in stat_columns:
        for prefix in ("home", "away"):
            col = f"{prefix}-{stat}-avg-rank"
            df[col] = df[col].astype("float32")

    logger.info("Computed rankings for %d stats", len(stat_columns))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Run after compute_rolling_stats.py to add rank features")
```

---

## 4. Games Played Tracking

### Concept
Track how many games each team has played in the current season. This is useful as a feature and also for filtering (don't trust averages from < 3 games).

### Implementation

```python
"""Track games played per team per season."""


def add_games_played(
    df: pd.DataFrame,
    date_col: str = "gameday",
    season_col: str = "season",
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
) -> pd.DataFrame:
    """Add home_gp and away_gp columns tracking games played before current game."""
    df = df.sort_values(date_col).reset_index(drop=True)

    team_games: dict[str, dict] = {}  # team -> {"count": int, "season": int}

    for idx, row in df.iterrows():
        home = row[home_team_col]
        away = row[away_team_col]
        season = row[season_col]

        for team in (home, away):
            if team not in team_games or team_games[team]["season"] != season:
                team_games[team] = {"count": 0, "season": season}

        # Store count BEFORE this game
        df.at[idx, "home_gp"] = team_games[home]["count"]
        df.at[idx, "away_gp"] = team_games[away]["count"]

        # Increment after
        team_games[home]["count"] += 1
        team_games[away]["count"] += 1

    df["home_gp"] = df["home_gp"].astype("Int8")
    df["away_gp"] = df["away_gp"].astype("Int8")

    return df
```

### Use in Model Training

```python
# Filter out early-season games with unreliable averages
min_games = 3
reliable_mask = (df["home_gp"] >= min_games) & (df["away_gp"] >= min_games)
train_df = df[reliable_mask]
```

---

## 5. Differential Features

### Concept
Instead of just raw home/away features, compute the **difference** between home and away. This captures relative matchup quality.

### Implementation

```python
"""Create differential features from home/away pairs."""


def create_differential_features(
    df: pd.DataFrame,
    stat_columns: list[str],
    suffix: str = "-avg-rank",
) -> pd.DataFrame:
    """Create home-minus-away differential features.

    For each stat, creates: {stat}-diff = home-{stat}{suffix} - away-{stat}{suffix}
    Positive diff = home team is better ranked (lower rank number).
    """
    for stat in stat_columns:
        home_col = f"home-{stat}{suffix}"
        away_col = f"away-{stat}{suffix}"
        if home_col in df.columns and away_col in df.columns:
            # Note: for ranks, LOWER is better, so negative diff = home advantage
            df[f"{stat}-rank-diff"] = (df[home_col] - df[away_col]).astype("float32")

    return df


# Example usage with PFF categories
PFF_CATEGORIES = ["off", "pass", "pblk", "recv", "run", "rblk",
                   "def", "rdef", "tack", "prsh", "cov"]

# Creates 11 differential features:
# off-rank-diff, pass-rank-diff, pblk-rank-diff, ...
df = create_differential_features(df, PFF_CATEGORIES)
```

### Why Differentials Help
- Reduces feature count (11 diff features vs 22 home/away features)
- Captures the matchup directly
- XGBoost can learn "big offense-defense gap → high-scoring game" patterns

---

## 6. Vegas Line Features

### Concept
Vegas lines are the market's best estimate of game outcomes. They incorporate information that our model may not capture (injuries, matchup history, coaching tendencies, etc.).

### From PFR or nfl_data_py

```python
"""Vegas line feature extraction and processing."""


def extract_vegas_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and compute Vegas-derived features.

    Assumes df has: spread_line, total_line (from nfl_data_py schedules)
    """
    if "spread_line" in df.columns:
        # Absolute spread = how lopsided the game is expected to be
        df["vegas_spread_abs"] = df["spread_line"].abs().astype("float32")

        # Spread category: close game vs blowout
        df["vegas_close_game"] = (df["vegas_spread_abs"] <= 3.0).astype("Int8")
        df["vegas_big_favorite"] = (df["vegas_spread_abs"] >= 7.0).astype("Int8")

    if "total_line" in df.columns:
        df["vegas_total"] = df["total_line"].astype("float32")

        # Total category: low-scoring vs high-scoring expected
        df["vegas_high_total"] = (df["total_line"] >= 48.0).astype("Int8")
        df["vegas_low_total"] = (df["total_line"] <= 40.0).astype("Int8")

    return df
```

### Checking nfl_data_py for Existing Vegas Data

```python
import nfl_data_py as nfl

# Check what schedule columns exist
sched = nfl.import_schedules(years=[2024])
vegas_cols = [c for c in sched.columns if any(
    kw in c.lower() for kw in ["spread", "over", "under", "line", "odds", "vegas"]
)]
print("Available Vegas columns:", vegas_cols)

# Expected output may include:
# spread_line, total_line, moneyline_home, moneyline_away
# If these exist, we don't need PFR scraping for lines!
```

---

## 7. O/U Target Variable (3-Class)

### Current Approach
Our totals model uses binary classification (over/under).

### Sports-Quant Approach
The sports-quant model uses **3-class classification** to handle pushes explicitly: 0=under, 1=over, 2=push.

### Implementation

```python
"""Three-class Over/Under target variable creation."""

import numpy as np


def create_ou_target(
    df: pd.DataFrame,
    total_score_col: str = "total_score",
    ou_line_col: str = "ou_line",
) -> pd.DataFrame:
    """Create 3-class O/U target: 0=under, 1=over, 2=push.

    total_score = home_score + away_score
    """
    if total_score_col not in df.columns:
        df[total_score_col] = df["home_score"] + df["away_score"]

    df["ou_target"] = np.where(
        df[total_score_col] > df[ou_line_col], 1,  # Over
        np.where(
            df[total_score_col] < df[ou_line_col], 0,  # Under
            2  # Push
        )
    ).astype("Int8")

    # Distribution check
    counts = df["ou_target"].value_counts()
    total = len(df)
    print(f"O/U Distribution: Over={counts.get(1,0)} ({counts.get(1,0)/total:.1%}), "
          f"Under={counts.get(0,0)} ({counts.get(0,0)/total:.1%}), "
          f"Push={counts.get(2,0)} ({counts.get(2,0)/total:.1%})")

    return df
```

### XGBoost Config for 3-Class

```python
xgb_params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 200,
}
```

---

## 8. Feature Interaction: Offense vs Defense Matchup

### Concept
Create features that capture specific matchup dynamics, e.g., "strong passing offense vs weak coverage defense."

```python
"""Offense-vs-defense matchup interaction features."""


def create_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create offense-vs-defense matchup interaction features.

    Uses PFF rank features to identify favorable/unfavorable matchups.
    Lower rank = better, so home_rank < away_rank = home advantage.
    """
    matchups = [
        # (home offense, away defense) → home offensive advantage
        ("home-pass-avg-rank", "away-cov-avg-rank", "pass_vs_cov_matchup"),
        ("home-run-avg-rank", "away-rdef-avg-rank", "run_vs_rdef_matchup"),
        ("home-pblk-avg-rank", "away-prsh-avg-rank", "pblk_vs_prsh_matchup"),
        # (away offense, home defense) → away offensive advantage
        ("away-pass-avg-rank", "home-cov-avg-rank", "away_pass_vs_cov_matchup"),
        ("away-run-avg-rank", "home-rdef-avg-rank", "away_run_vs_rdef_matchup"),
        ("away-pblk-avg-rank", "home-prsh-avg-rank", "away_pblk_vs_prsh_matchup"),
    ]

    for off_col, def_col, feature_name in matchups:
        if off_col in df.columns and def_col in df.columns:
            # Negative = offense ranked better than opposing defense
            df[feature_name] = (df[off_col] - df[def_col]).astype("float32")

    return df
```

---

## 9. Complete Processing Pipeline

### End-to-End Pipeline

Create `scripts/processing/run_processing.py`:

```python
"""Run the complete feature processing pipeline."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

root = Path(__file__).parent.parent.parent


def run_pipeline():
    """Full processing pipeline: load → rolling averages → rankings → features."""

    # Step 1: Load base data
    logger.info("Step 1: Loading base data...")
    games_path = root / "data_files" / "nfl_games_historical.csv"
    df = pd.read_csv(games_path, sep="\t")
    logger.info("  Loaded %d games", len(df))

    # Step 2: Load and merge PFF data (if available)
    pff_path = root / "data_files" / "pff_team_grades.csv"
    if pff_path.exists():
        logger.info("Step 2: Processing PFF data...")
        from scripts.scrapers.pff_adapter import (
            load_pff_grades,
            compute_rolling_pff_averages,
            compute_pff_rankings,
        )
        pff_df = load_pff_grades(pff_path)
        pff_df = compute_rolling_pff_averages(pff_df)
        pff_df = compute_pff_rankings(pff_df)

        # Merge PFF rank features
        pff_rank_cols = [c for c in pff_df.columns if c.endswith("-avg-rank")]
        merge_cols = ["Formatted Date", "home_team", "away_team"] + pff_rank_cols
        df = df.merge(pff_df[merge_cols], on=["Formatted Date", "home_team", "away_team"], how="left")
        logger.info("  Merged %d PFF rank features", len(pff_rank_cols))
    else:
        logger.info("Step 2: No PFF data found, skipping")

    # Step 3: Load and merge PFR data (if available)
    pfr_path = root / "data_files" / "pfr_game_data.csv"
    if pfr_path.exists():
        logger.info("Step 3: Processing PFR data...")
        from scripts.scrapers.pff_adapter import process_pfr_data  # reuse adapter
        pfr_df = process_pfr_data(pfr_path)
        # Merge PFR features
        pfr_cols = ["vegas_spread", "ou_line", "roof_encoded", "surface_encoded"]
        df = df.merge(pfr_df, on=["Formatted Date", "home_team", "away_team"], how="left")
        logger.info("  Merged PFR features")
    else:
        logger.info("Step 3: No PFR data found, skipping")

    # Step 4: Add derived features
    logger.info("Step 4: Computing derived features...")

    # Games played tracking
    from scripts.processing.compute_rolling_stats import add_games_played
    df = add_games_played(df)

    # Differential features (if PFF ranks available)
    pff_cats = ["off", "pass", "pblk", "recv", "run", "rblk",
                "def", "rdef", "tack", "prsh", "cov"]
    if f"home-{pff_cats[0]}-avg-rank" in df.columns:
        from scripts.processing.compute_rankings import create_differential_features
        df = create_differential_features(df, pff_cats)
        logger.info("  Added %d differential features", len(pff_cats))

    # Vegas features
    from scripts.processing.compute_rankings import extract_vegas_features
    df = extract_vegas_features(df)

    # Step 5: Save
    output_path = root / "data_files" / "nfl_games_enriched.csv"
    df.to_csv(output_path, sep="\t", index=False)
    logger.info("Saved enriched data to %s (%d rows, %d cols)", output_path, len(df), len(df.columns))

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_pipeline()
```

---

## 10. Feature Importance Analysis

### Compare Old vs New Features

After training with new features, compare feature importances:

```python
"""Analyze feature importances across model versions."""

import json
from pathlib import Path

import pandas as pd


def compare_feature_importance(
    old_importances_path: str = "data_files/model_feature_importances.csv",
    new_importances_path: str = "data_files/model_feature_importances_v2.csv",
) -> pd.DataFrame:
    """Compare feature importances between old and new model versions."""
    old = pd.read_csv(old_importances_path)
    new = pd.read_csv(new_importances_path)

    comparison = old.merge(new, on="feature", suffixes=("_old", "_new"), how="outer")
    comparison = comparison.fillna(0)
    comparison["change"] = comparison["importance_new"] - comparison["importance_old"]
    comparison = comparison.sort_values("importance_new", ascending=False)

    print("\nTop 20 Features (New Model):")
    print(comparison.head(20).to_string(index=False))

    # New features added
    new_only = comparison[comparison["importance_old"] == 0]
    if not new_only.empty:
        print(f"\nNew features ({len(new_only)}):")
        print(new_only[["feature", "importance_new"]].to_string(index=False))

    return comparison
```

---

## 11. Summary: All New Features

| Category | Features | Count | Source |
|----------|----------|-------|--------|
| PFF Rank (home/away) | off, pass, pblk, recv, run, rblk, def, rdef, tack, prsh, cov | 22 | PFF scraper |
| PFF Rank Differentials | off-diff, pass-diff, pblk-diff, recv-diff, run-diff, rblk-diff, def-diff, rdef-diff, tack-diff, prsh-diff, cov-diff | 11 | Computed |
| Matchup Interactions | pass_vs_cov, run_vs_rdef, pblk_vs_prsh (× home/away) | 6 | Computed |
| Vegas | vegas_spread, vegas_spread_abs, vegas_close_game, vegas_big_favorite | 4 | PFR or nfl_data_py |
| Totals | ou_line, vegas_total, vegas_high_total, vegas_low_total | 4 | PFR or nfl_data_py |
| Venue | roof_encoded, surface_encoded | 2 | PFR |
| Metadata | home_gp, away_gp | 2 | Computed |
| **Total New** | | **51** | |

### Recommended Priority

1. **Immediate** (no new data needed):
   - Vegas spread/total from `nfl_data_py` (check availability first)
   - Games played tracking
   - Differential features from existing rolling stats

2. **Short-term** (PFR scraping):
   - O/U line, Vegas spread (if not in nfl_data_py)
   - Roof/Surface encoding

3. **Medium-term** (PFF subscription):
   - 22 PFF rank features
   - 11 PFF differential features
   - 6 matchup interaction features

---

## 12. Directory Structure

```
scripts/
  processing/
    __init__.py
    compute_rolling_stats.py    # Rolling season averages
    compute_rankings.py         # Per-date team rankings
    run_processing.py           # End-to-end pipeline orchestrator
  scrapers/
    pff_adapter.py              # PFF data loading and processing
    ...                         # (see ROADMAP_SCRAPING_PIPELINES.md)
```
