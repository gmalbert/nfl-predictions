# Roadmap: New Data Sources

> Inspired by [thadhutch/sports-quant](https://github.com/thadhutch/sports-quant) NFL pipeline.

## Overview

Our current pipeline relies on `nfl_data_py` for schedule/play-by-play data and ESPN for live scores. The sports-quant project demonstrates two powerful additional data sources — **PFF team grades** and **Pro Football Reference (PFR) boxscore metadata** — that can significantly improve model accuracy, especially for totals (O/U) and spread predictions.

---

## 1. PFF Team Grades (Premium)

### What It Provides
PFF grades every player on every play and aggregates into **11 team-level categories** per game:

| Category | Column | Description |
|----------|--------|-------------|
| Offense Overall | `off` | Composite offensive grade |
| Passing | `pass` | QB + passing game grade |
| Pass Blocking | `pblk` | Pass protection grade |
| Receiving | `recv` | Receiver performance grade |
| Rushing | `run` | Run game grade |
| Run Blocking | `rblk` | Run blocking grade |
| Defense Overall | `def` | Composite defensive grade |
| Run Defense | `rdef` | Run stopping grade |
| Tackling | `tack` | Tackling efficiency grade |
| Pass Rush | `prsh` | Pass rush grade |
| Coverage | `cov` | Coverage grade |

### Why It Matters
- PFF grades are **scheme-adjusted** — they measure player/unit quality independent of opponent
- Rolling PFF averages capture team quality trends that box-score stats miss
- When converted to **per-date rankings** (1–32), they provide powerful ordinal features
- The sports-quant model uses **only 23 features** (22 PFF rank features + O/U line) and achieves strong O/U prediction accuracy

### Integration Path

#### Step 1: Create PFF Data Adapter

Create `scripts/scrapers/pff_adapter.py` to store and process PFF data:

```python
"""Adapter to load and process PFF team grade data for model features."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# PFF grade categories (11 total)
PFF_CATEGORIES = [
    "off", "pass", "pblk", "recv", "run", "rblk",
    "def", "rdef", "tack", "prsh", "cov",
]

# Expected columns in the raw PFF CSV
PFF_HOME_COLS = [f"home-{cat}" for cat in PFF_CATEGORIES]
PFF_AWAY_COLS = [f"away-{cat}" for cat in PFF_CATEGORIES]
PFF_REQUIRED_COLS = (
    ["Formatted Date", "season", "home_team", "away_team"]
    + PFF_HOME_COLS
    + PFF_AWAY_COLS
)


def load_pff_grades(pff_csv: str | Path) -> pd.DataFrame:
    """Load raw PFF grades CSV and validate columns.

    Expected format: one row per game with home/away grades for each category.
    """
    df = pd.read_csv(pff_csv)

    missing = set(PFF_REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"PFF CSV missing columns: {missing}")

    df["Formatted Date"] = pd.to_datetime(df["Formatted Date"])
    df = df.sort_values("Formatted Date").reset_index(drop=True)

    # Optimize dtypes
    for col in PFF_HOME_COLS + PFF_AWAY_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    df["season"] = df["season"].astype("Int16")

    logger.info("Loaded %d PFF game records", len(df))
    return df


def compute_rolling_pff_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling season-average PFF grades for each team.

    Uses only prior games in the current season (no data leakage).
    Resets averages at the start of each new season.
    """
    team_stats: dict[str, dict] = {}

    for idx, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        season = row["season"]

        # Initialize or reset on new season
        for team in (home, away):
            if team not in team_stats:
                team_stats[team] = {
                    "stats": {cat: 0.0 for cat in PFF_CATEGORIES},
                    "count": 0,
                    "season": season,
                }
            elif team_stats[team]["season"] != season:
                team_stats[team] = {
                    "stats": {cat: 0.0 for cat in PFF_CATEGORIES},
                    "count": 0,
                    "season": season,
                }

        # Store PRIOR averages (before this game) — no leakage
        for cat in PFF_CATEGORIES:
            home_count = team_stats[home]["count"]
            away_count = team_stats[away]["count"]
            df.at[idx, f"home-{cat}-avg"] = (
                team_stats[home]["stats"][cat] / home_count
                if home_count > 0
                else 0.0
            )
            df.at[idx, f"away-{cat}-avg"] = (
                team_stats[away]["stats"][cat] / away_count
                if away_count > 0
                else 0.0
            )

        # Update cumulative stats WITH current game
        for cat in PFF_CATEGORIES:
            team_stats[home]["stats"][cat] += row[f"home-{cat}"]
            team_stats[away]["stats"][cat] += row[f"away-{cat}"]
        team_stats[home]["count"] += 1
        team_stats[away]["count"] += 1

    # Optimize new columns
    avg_cols = [f"{side}-{cat}-avg" for side in ("home", "away") for cat in PFF_CATEGORIES]
    for col in avg_cols:
        df[col] = df[col].astype("float32")

    logger.info("Computed rolling PFF averages (%d columns)", len(avg_cols))
    return df


def compute_pff_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert rolling PFF averages into per-date team rankings (1=best).

    For each game date, rank all teams based on their rolling average
    using only data available up to that date (no lookahead).
    """
    avg_cols = [f"{side}-{cat}-avg" for side in ("home", "away") for cat in PFF_CATEGORIES]

    dates = df["Formatted Date"].unique()
    dates.sort()

    for date in dates:
        date_mask = df["Formatted Date"] <= date
        prior_df = df[date_mask]

        # Collect latest average for each team
        team_avgs: dict[str, dict[str, float]] = {}
        for _, row in prior_df.iterrows():
            for side, team_col in [("home", "home_team"), ("away", "away_team")]:
                team = row[team_col]
                team_avgs[team] = {
                    cat: row[f"{side}-{cat}-avg"] for cat in PFF_CATEGORIES
                }

        # Rank teams for each category
        for cat in PFF_CATEGORIES:
            cat_values = {t: avgs[cat] for t, avgs in team_avgs.items()}
            # Higher PFF grade = better = rank 1
            sorted_teams = sorted(cat_values, key=cat_values.get, reverse=True)
            rankings = {team: rank + 1 for rank, team in enumerate(sorted_teams)}

            # Apply rankings to games on this date
            game_mask = df["Formatted Date"] == date
            for idx in df[game_mask].index:
                home = df.at[idx, "home_team"]
                away = df.at[idx, "away_team"]
                df.at[idx, f"home-{cat}-avg-rank"] = rankings.get(home, 16)
                df.at[idx, f"away-{cat}-avg-rank"] = rankings.get(away, 16)

    rank_cols = [f"{side}-{cat}-avg-rank" for side in ("home", "away") for cat in PFF_CATEGORIES]
    for col in rank_cols:
        df[col] = df[col].astype("float32")

    logger.info("Computed PFF rankings (%d columns)", len(rank_cols))
    return df


if __name__ == "__main__":
    """Example usage: process raw PFF data end-to-end."""
    import sys

    logging.basicConfig(level=logging.INFO)

    pff_path = Path("data_files/pff_team_grades.csv")
    if not pff_path.exists():
        print(f"PFF data file not found: {pff_path}")
        sys.exit(1)

    df = load_pff_grades(pff_path)
    df = compute_rolling_pff_averages(df)
    df = compute_pff_rankings(df)

    output_path = Path("data_files/pff_ranked.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved ranked PFF data to {output_path}")
```

#### Step 2: Integrate PFF Features into Training

In `nfl-gather-data.py`, merge PFF features with existing game data:

```python
# After loading nfl_games_historical.csv
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

pff_ranked_path = Path("data_files/pff_ranked.csv")
if pff_ranked_path.exists():
    pff_df = pd.read_csv(pff_ranked_path)
    pff_df["Formatted Date"] = pd.to_datetime(pff_df["Formatted Date"])

    # Merge on date + teams (adjust column names to match your schema)
    pff_rank_cols = [
        f"{side}-{cat}-avg-rank"
        for side in ("home", "away")
        for cat in [
            "off", "pass", "pblk", "recv", "run", "rblk",
            "def", "rdef", "tack", "prsh", "cov",
        ]
    ]
    merge_cols = ["Formatted Date", "home_team", "away_team"] + pff_rank_cols

    df = df.merge(
        pff_df[merge_cols],
        on=["Formatted Date", "home_team", "away_team"],
        how="left",
    )
    print(f"Merged {len(pff_rank_cols)} PFF rank features")
else:
    print("No PFF data found — skipping PFF features")
```

---

## 2. Pro Football Reference (PFR) Boxscore Metadata

### What It Provides
PFR boxscore pages contain game-context metadata not available in play-by-play:

| Field | Values | Use Case |
|-------|--------|----------|
| **Roof** | `outdoors`, `dome`, `retractable` | Weather exposure indicator |
| **Surface** | `grass`, `fieldturf`, `a_turf`, etc. | Surface type affects game pace |
| **Vegas Line** | e.g., `Kansas City Chiefs -3.0` | Market-implied team quality |
| **Over/Under** | e.g., `47.5 (over)` | Market-implied total + actual result |

### Why It Matters
- **Vegas lines** are the market's best estimate of team strength — powerful baseline features
- **Roof/Surface** interact with weather to affect scoring (domes = higher totals)
- **O/U line** is the single most predictive feature in the sports-quant totals model
- Actual O/U result (over/under/push) provides the target variable for totals models

### Integration Path

#### Step 1: Create PFR Data Adapter

```python
"""Adapter to load and process PFR boxscore metadata."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_pfr_game_data(pfr_csv: str | Path) -> pd.DataFrame:
    """Load PFR boxscore game data.

    Expected columns: Title, Roof, Surface, Vegas Line, Over/Under
    """
    df = pd.read_csv(pfr_csv)
    logger.info("Loaded %d PFR game records", len(df))
    return df


def extract_vegas_line(vegas_str: str) -> tuple[str, float] | tuple[None, None]:
    """Parse 'Kansas City Chiefs -3.0' into (team, spread).

    Returns (favorite_team, spread_value) or (None, None) on failure.
    """
    if not isinstance(vegas_str, str) or vegas_str == "N/A":
        return None, None
    try:
        parts = vegas_str.rsplit(" ", 1)
        team = parts[0].strip()
        spread = float(parts[1])
        return team, spread
    except (ValueError, IndexError):
        return None, None


def extract_ou_line(ou_str: str) -> tuple[float, int] | tuple[None, None]:
    """Parse '47.5 (over)' into (line, result).

    result: 1=over, 0=under, 2=push
    """
    if not isinstance(ou_str, str) or ou_str == "N/A":
        return None, None
    try:
        line = float(ou_str.split()[0])
        if "(under)" in ou_str:
            result = 0
        elif "(over)" in ou_str:
            result = 1
        else:
            result = 2  # push
        return line, result
    except (ValueError, IndexError):
        return None, None


def encode_roof(roof: str) -> int:
    """Encode roof type as numeric. 0=outdoors, 1=dome, 2=retractable."""
    mapping = {"outdoors": 0, "dome": 1, "retractable roof": 2}
    return mapping.get(str(roof).lower().strip(), 0)


def encode_surface(surface: str) -> int:
    """Encode surface type. 0=grass, 1=artificial."""
    if not isinstance(surface, str):
        return 0
    s = surface.lower().strip()
    return 0 if s in ("grass",) else 1


def process_pfr_data(pfr_csv: str | Path) -> pd.DataFrame:
    """Load and process PFR data into model-ready features."""
    df = load_pfr_game_data(pfr_csv)

    # Extract Vegas line
    parsed = df["Vegas Line"].apply(extract_vegas_line)
    df["vegas_favorite"] = [p[0] for p in parsed]
    df["vegas_spread"] = [p[1] for p in parsed]
    df["vegas_spread"] = df["vegas_spread"].astype("float32")

    # Extract O/U line and result
    parsed_ou = df["Over/Under"].apply(extract_ou_line)
    df["ou_line"] = [p[0] for p in parsed_ou]
    df["ou_result"] = [p[1] for p in parsed_ou]
    df["ou_line"] = df["ou_line"].astype("float32")
    df["ou_result"] = df["ou_result"].astype("Int8")

    # Encode categorical
    df["roof_encoded"] = df["Roof"].apply(encode_roof).astype("Int8")
    df["surface_encoded"] = df["Surface"].apply(encode_surface).astype("Int8")

    logger.info("Processed PFR data: %d games", len(df))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pfr_path = Path("data_files/pfr_game_data.csv")
    if pfr_path.exists():
        df = process_pfr_data(pfr_path)
        print(df[["vegas_spread", "ou_line", "ou_result", "roof_encoded", "surface_encoded"]].head(10))
```

---

## 3. Combining PFF + PFR with Existing Data

### Merge Strategy

The sports-quant pipeline merges PFF and PFR data on (date, home_team, away_team). We should follow the same pattern:

```python
"""Merge PFF ranks + PFR metadata with our existing game data."""

def merge_all_sources(
    games_df: pd.DataFrame,
    pff_df: pd.DataFrame | None = None,
    pfr_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge all data sources into a single training DataFrame.

    games_df: Our existing nfl_games_historical.csv with features
    pff_df: PFF ranked data (optional)
    pfr_df: PFR game metadata (optional)
    """
    result = games_df.copy()

    if pff_df is not None:
        pff_rank_cols = [
            col for col in pff_df.columns if col.endswith("-avg-rank")
        ]
        result = result.merge(
            pff_df[["Formatted Date", "home_team", "away_team"] + pff_rank_cols],
            on=["Formatted Date", "home_team", "away_team"],
            how="left",
        )
        print(f"  Merged {len(pff_rank_cols)} PFF rank features")

    if pfr_df is not None:
        pfr_features = ["vegas_spread", "ou_line", "roof_encoded", "surface_encoded"]
        result = result.merge(
            pfr_df[["Formatted Date", "home_team", "away_team"] + pfr_features],
            on=["Formatted Date", "home_team", "away_team"],
            how="left",
        )
        print(f"  Merged {len(pfr_features)} PFR features")

    return result
```

### Feature List Addition

Update `best_features_spread.txt` and similar files:

```
# PFF Rank Features (add to best_features_*.txt)
home-off-avg-rank
away-off-avg-rank
home-pass-avg-rank
away-pass-avg-rank
home-pblk-avg-rank
away-pblk-avg-rank
home-recv-avg-rank
away-recv-avg-rank
home-run-avg-rank
away-run-avg-rank
home-rblk-avg-rank
away-rblk-avg-rank
home-def-avg-rank
away-def-avg-rank
home-rdef-avg-rank
away-rdef-avg-rank
home-tack-avg-rank
away-tack-avg-rank
home-prsh-avg-rank
away-prsh-avg-rank
home-cov-avg-rank
away-cov-avg-rank

# PFR Features
ou_line
vegas_spread
roof_encoded
surface_encoded
```

---

## 4. Data Acquisition Requirements

### PFF (Paid)
- **Cost**: PFF Premium subscription (~$35/month during season, ~$10/month offseason)
- **Access**: Requires Selenium scraper with manual login (see [ROADMAP_SCRAPING_PIPELINES.md](ROADMAP_SCRAPING_PIPELINES.md))
- **Volume**: ~288 games/season × 22 grade columns = manageable
- **Alternative**: Some PFF data appears in aggregated form on free sites; accuracy varies

### PFR (Free, rate-limited)
- **Cost**: Free, but heavily rate-limited (requires proxy rotation)
- **Access**: BeautifulSoup for URL collection, Selenium for boxscores (see [ROADMAP_SCRAPING_PIPELINES.md](ROADMAP_SCRAPING_PIPELINES.md))
- **Volume**: ~288 boxscore URLs/season
- **Alternative**: Some Vegas lines available via `nfl_data_py` — check `nfl.import_sc_lines()` before building a scraper

### Quick Win: Check Existing Data First

Before building scrapers, check what `nfl_data_py` already provides:

```python
import nfl_data_py as nfl

# Check available betting data
lines = nfl.import_sc_lines(years=[2024])
print(lines.columns.tolist())
# May include: spread_line, total_line (O/U), moneyline odds

# Check available team stats
schedules = nfl.import_schedules(years=[2024])
print(schedules.columns.tolist())
# May include: roof, surface, spread_line, over_under_line
```

If `nfl_data_py` already provides Vegas lines and O/U lines, we may only need PFF for the grade features.

---

## 5. Priority & Effort Estimates

| Data Source | Priority | Effort | Impact |
|------------|----------|--------|--------|
| PFR O/U Line | **High** | Low (may be in nfl_data_py) | Direct totals feature |
| PFR Vegas Spread | **High** | Low (may be in nfl_data_py) | Direct spread feature |
| PFF Team Grades | **Medium** | High (requires subscription + scraper) | 22 new rank features |
| PFR Roof/Surface | **Low** | Low | Marginal improvement |

---

## Next Steps

1. **Audit existing data**: Check if `nfl_data_py` provides spread/total lines
2. **If not**: Build PFR scraper (see [ROADMAP_SCRAPING_PIPELINES.md](ROADMAP_SCRAPING_PIPELINES.md))
3. **Evaluate PFF ROI**: Consider whether PFF Premium is worth the cost for 22 extra features
4. **Integrate features**: Follow the merge pattern above, add to feature lists, retrain models
5. **Validate**: Ensure no data leakage — all features must be pre-game only
