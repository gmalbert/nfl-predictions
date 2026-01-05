# Player Props Prediction System - Roadmap

## Overview
Build a prediction system for individual player performance props similar to DraftKings Pick 6, predicting:
- Passing Yards
- Rushing Yards
- Receiving Yards
- Rush + Rec TDs
- Receptions (5+ threshold)
- And other player-specific markets

## Data Foundation
**Existing Asset**: `data_files/nfl_play_by_play_historical.csv.gz` (~282k plays)
- Contains play-by-play data from 2020-2025 seasons (auto-updates with current year)
- Includes player-level stats: yards_gained, touchdowns, pass attempts, rush attempts, receptions
- Team context: score differential, game situation, weather

---

## SHORT TERM (Weeks 1-2): Data Exploration & Baseline Models

### Phase 1A: Player Stat Exploration (Days 1-2)
**Goal**: Understand available player data and build aggregation functions

**Tasks**:
1. Explore play-by-play columns for player stats
2. Map player IDs to names
3. Aggregate season/game-level stats
4. Identify data quality issues

**Deliverable**: `scripts/explore_player_stats.py`

```python
"""
Explore play-by-play data for player-level statistics.
Run: python scripts/explore_player_stats.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load play-by-play data
DATA_DIR = Path('data_files')
pbp_file = DATA_DIR / 'nfl_play_by_play_historical.csv.gz'

print("Loading play-by-play data...")
pbp = pd.read_csv(pbp_file, compression='gzip', low_memory=False)

print(f"\n📊 Dataset: {len(pbp):,} plays")
print(f"Seasons: {pbp['season'].min()}-{pbp['season'].max()}")
print(f"Columns: {pbp.shape[1]}")

# Key player stat columns
player_cols = [
    # Passers
    'passer_player_name', 'passer_player_id', 'passing_yards', 'pass_touchdown',
    # Rushers
    'rusher_player_name', 'rusher_player_id', 'rushing_yards', 'rush_touchdown',
    # Receivers
    'receiver_player_name', 'receiver_player_id', 'receiving_yards', 'reception',
    # Game context
    'game_id', 'season', 'week', 'game_date', 'posteam', 'defteam'
]

available_cols = [c for c in player_cols if c in pbp.columns]
print(f"\n✅ Available player columns: {len(available_cols)}")
for col in available_cols:
    print(f"  - {col}")

# Sample: Top passers by yards
print("\n🏈 Top 10 Passers (2024 season):")
if 'passer_player_name' in pbp.columns and 'passing_yards' in pbp.columns:
    passers_2024 = pbp[
        (pbp['season'] == 2024) & 
        (pbp['passer_player_name'].notna())
    ].groupby('passer_player_name').agg({
        'passing_yards': 'sum',
        'pass_touchdown': 'sum',
        'game_id': 'nunique'
    }).sort_values('passing_yards', ascending=False).head(10)
    passers_2024.columns = ['Pass Yards', 'Pass TDs', 'Games']
    print(passers_2024)

# Sample: Top rushers
print("\n🏃 Top 10 Rushers (2024 season):")
if 'rusher_player_name' in pbp.columns and 'rushing_yards' in pbp.columns:
    rushers_2024 = pbp[
        (pbp['season'] == 2024) & 
        (pbp['rusher_player_name'].notna())
    ].groupby('rusher_player_name').agg({
        'rushing_yards': 'sum',
        'rush_touchdown': 'sum',
        'game_id': 'nunique'
    }).sort_values('rushing_yards', ascending=False).head(10)
    rushers_2024.columns = ['Rush Yards', 'Rush TDs', 'Games']
    print(rushers_2024)

# Sample: Top receivers
print("\n🎯 Top 10 Receivers (2024 season):")
if 'receiver_player_name' in pbp.columns and 'receiving_yards' in pbp.columns:
    receivers_2024 = pbp[
        (pbp['season'] == 2024) & 
        (pbp['receiver_player_name'].notna())
    ].groupby('receiver_player_name').agg({
        'receiving_yards': 'sum',
        'reception': 'sum',
        'game_id': 'nunique'
    }).sort_values('receiving_yards', ascending=False).head(10)
    receivers_2024.columns = ['Rec Yards', 'Receptions', 'Games']
    print(receivers_2024)

# Data quality check
print("\n🔍 Data Quality:")
print(f"Missing passer names: {pbp['passer_player_name'].isna().sum():,}")
print(f"Missing rusher names: {pbp['rusher_player_name'].isna().sum():,}")
print(f"Missing receiver names: {pbp['receiver_player_name'].isna().sum():,}")

# Save sample to CSV for manual inspection
sample_out = DATA_DIR / 'player_stats_sample.csv'
pbp[available_cols].head(1000).to_csv(sample_out, index=False)
print(f"\n💾 Saved sample to {sample_out}")
```

---

### Phase 1B: Player Game-Level Aggregation (Days 3-4)
**Goal**: Build functions to aggregate player stats by game

**Deliverable**: `player_props/aggregators.py`

```python
"""
Player stat aggregation functions.
Converts play-by-play to game-level player stats.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

def aggregate_passing_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate passing stats by player and game.
    
    Returns:
        DataFrame with columns: [player_id, player_name, game_id, season, week,
                                  team, opponent, passing_yards, pass_tds, 
                                  completions, attempts, interceptions]
    """
    passing_plays = pbp[pbp['passer_player_id'].notna()].copy()
    
    agg_stats = passing_plays.groupby([
        'passer_player_id', 'passer_player_name', 'game_id', 
        'season', 'week', 'posteam', 'defteam'
    ]).agg({
        'passing_yards': 'sum',
        'pass_touchdown': 'sum',
        'complete_pass': 'sum',
        'pass_attempt': 'sum',
        'interception': 'sum'
    }).reset_index()
    
    agg_stats.columns = [
        'player_id', 'player_name', 'game_id', 'season', 'week',
        'team', 'opponent', 'passing_yards', 'pass_tds', 
        'completions', 'attempts', 'interceptions'
    ]
    
    return agg_stats


def aggregate_rushing_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rushing stats by player and game."""
    rushing_plays = pbp[pbp['rusher_player_id'].notna()].copy()
    
    agg_stats = rushing_plays.groupby([
        'rusher_player_id', 'rusher_player_name', 'game_id',
        'season', 'week', 'posteam', 'defteam'
    ]).agg({
        'rushing_yards': 'sum',
        'rush_touchdown': 'sum',
        'rush_attempt': 'sum'
    }).reset_index()
    
    agg_stats.columns = [
        'player_id', 'player_name', 'game_id', 'season', 'week',
        'team', 'opponent', 'rushing_yards', 'rush_tds', 'rush_attempts'
    ]
    
    return agg_stats


def aggregate_receiving_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate receiving stats by player and game."""
    receiving_plays = pbp[pbp['receiver_player_id'].notna()].copy()
    
    agg_stats = receiving_plays.groupby([
        'receiver_player_id', 'receiver_player_name', 'game_id',
        'season', 'week', 'posteam', 'defteam'
    ]).agg({
        'receiving_yards': 'sum',
        'reception': 'sum',
        'pass_touchdown': 'sum'  # Receiving TDs
    }).reset_index()
    
    agg_stats.columns = [
        'player_id', 'player_name', 'game_id', 'season', 'week',
        'team', 'opponent', 'receiving_yards', 'receptions', 'rec_tds'
    ]
    
    return agg_stats


def calculate_rolling_averages(
    player_stats: pd.DataFrame,
    stat_col: str,
    windows: List[int] = [3, 5, 10]
) -> pd.DataFrame:
    """
    Calculate rolling averages for a stat column.
    
    Args:
        player_stats: Game-level player stats (must have player_id, season, week)
        stat_col: Column to calculate rolling average for
        windows: List of window sizes (e.g., [3, 5, 10] for L3, L5, L10)
    
    Returns:
        DataFrame with additional columns for rolling averages
    """
    player_stats = player_stats.sort_values(['player_id', 'season', 'week'])
    
    for window in windows:
        col_name = f'{stat_col}_L{window}'
        player_stats[col_name] = player_stats.groupby('player_id')[stat_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
    
    return player_stats


# Example usage:
if __name__ == '__main__':
    import gzip
    from pathlib import Path
    
    DATA_DIR = Path('data_files')
    pbp = pd.read_csv(
        DATA_DIR / 'nfl_play_by_play_historical.csv.gz',
        compression='gzip',
        low_memory=False
    )
    
    # Aggregate stats
    print("Aggregating passing stats...")
    passing = aggregate_passing_stats(pbp)
    print(f"✅ {len(passing):,} player-game records")
    
    # Add rolling averages
    passing = calculate_rolling_averages(passing, 'passing_yards', windows=[3, 5, 10])
    
    # Save
    passing.to_csv(DATA_DIR / 'player_passing_game_stats.csv', index=False)
    print(f"💾 Saved to player_passing_game_stats.csv")
```

---

### Phase 1C: Baseline Prop Models (Days 5-7)
**Goal**: Build simple XGBoost models for 2-3 prop types

**Deliverable**: `player_props/models.py`

```python
"""
Baseline player prop prediction models.
Start with Passing Yards, Rushing Yards, Receiving Yards.
"""
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path

class PlayerPropModel:
    """Base class for player prop prediction models."""
    
    def __init__(self, prop_type: str):
        """
        Args:
            prop_type: 'passing_yards', 'rushing_yards', 'receiving_yards', etc.
        """
        self.prop_type = prop_type
        self.model = None
        self.features = []
        self.target = prop_type
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for modeling.
        Override in subclasses for prop-specific features.
        """
        raise NotImplementedError
        
    def train(self, df: pd.DataFrame, test_size: float = 0.2):
        """Train the model."""
        # Prepare features
        df_features = self.prepare_features(df)
        
        # Split
        X = df_features[self.features]
        y = df_features[self.target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train XGBoost
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"\n✅ {self.prop_type} Model Trained")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        
        return {'mae': mae, 'rmse': rmse}
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        df_features = self.prepare_features(df)
        return self.model.predict(df_features[self.features])
    
    def save(self, filepath: str):
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'features': self.features,
            'prop_type': self.prop_type
        }, filepath)
        
    @classmethod
    def load(cls, filepath: str):
        """Load model from disk."""
        data = joblib.load(filepath)
        instance = cls(data['prop_type'])
        instance.model = data['model']
        instance.features = data['features']
        return instance


class PassingYardsModel(PlayerPropModel):
    """Model for predicting passing yards."""
    
    def __init__(self):
        super().__init__('passing_yards')
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare passing-specific features."""
        df = df.copy()
        
        # Rolling averages (already calculated)
        self.features = [
            'passing_yards_L3',
            'passing_yards_L5',
            'passing_yards_L10',
            'completions_L3',
            'attempts_L3',
            'pass_tds_L3',
            # Add opponent defense stats when available
        ]
        
        # Only keep rows with all features
        df = df.dropna(subset=self.features + [self.target])
        
        return df


class RushingYardsModel(PlayerPropModel):
    """Model for predicting rushing yards."""
    
    def __init__(self):
        super().__init__('rushing_yards')
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        self.features = [
            'rushing_yards_L3',
            'rushing_yards_L5',
            'rushing_yards_L10',
            'rush_attempts_L3',
            'rush_tds_L3',
        ]
        
        df = df.dropna(subset=self.features + [self.target])
        return df


class ReceivingYardsModel(PlayerPropModel):
    """Model for predicting receiving yards."""
    
    def __init__(self):
        super().__init__('receiving_yards')
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        self.features = [
            'receiving_yards_L3',
            'receiving_yards_L5',
            'receiving_yards_L10',
            'receptions_L3',
            'rec_tds_L3',
        ]
        
        df = df.dropna(subset=self.features + [self.target])
        return df


# Training script
if __name__ == '__main__':
    from pathlib import Path
    
    DATA_DIR = Path('data_files')
    MODEL_DIR = Path('player_props/models')
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    
    # Load aggregated stats
    passing_stats = pd.read_csv(DATA_DIR / 'player_passing_game_stats.csv')
    
    # Train passing model
    print("Training Passing Yards Model...")
    model = PassingYardsModel()
    metrics = model.train(passing_stats)
    model.save(MODEL_DIR / 'passing_yards_model.pkl')
    
    print("\n✅ Models saved to player_props/models/")
```

---

## MEDIUM TERM (Weeks 3-8): Feature Engineering & UI Integration

### Phase 2A: Advanced Feature Engineering (Weeks 3-4)
**Goal**: Add opponent adjustments, game context, weather

**New Features**:
```python
# Opponent defensive stats
- 'opp_pass_yards_allowed_L3'  # Opponent's average pass yards allowed (last 3 games)
- 'opp_rush_yards_allowed_L3'
- 'opp_sacks_L3'

# Game situation
- 'home_game'  # 1 if home, 0 if away
- 'division_game'  # 1 if division rival
- 'days_rest'  # Days since last game

# Weather (from existing weather data)
- 'temperature'
- 'wind_speed'
- 'is_dome'

# Player usage trends
- 'target_share_L3'  # % of team's targets
- 'snap_share_L3'  # % of offensive snaps (if available)
```

**Deliverable**: `player_props/feature_engineering.py`

---

### Phase 2B: Prop Type Expansion (Weeks 4-5)
**Goal**: Build models for all major prop types

**Prop Types to Add**:
1. **Anytime TD** (classification model)
2. **Rush + Rec TDs** (combo model)
3. **5+ Receptions** (classification model)
4. **Longest Reception** (quantile regression)
5. **Completions** (regression)

**Deliverable**: Update `player_props/models.py` with new model classes

---

### Phase 2C: UI Integration (Weeks 6-7)
**Goal**: Add player props tab to Streamlit app

**UI Features**:
- **Player Props Dashboard** (new tab in `predictions.py`)
  - Filter by position, team, prop type
  - Display predictions with confidence intervals
  - Show recent performance trends
  - Export to CSV/PDF

**Deliverable**: Update `predictions.py` with new "Player Props" tab

```python
# In predictions.py, add new tab:
with st.tabs(["...", "🎯 Player Props"]):
    st.write("### Player Prop Predictions")
    
    # Load player prop predictions
    props_df = load_player_props()
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        position_filter = st.multiselect("Position", ["QB", "RB", "WR", "TE"])
    with col2:
        prop_type_filter = st.selectbox("Prop Type", [
            "Passing Yards", "Rushing Yards", "Receiving Yards", 
            "Anytime TD", "5+ Receptions"
        ])
    with col3:
        team_filter = st.multiselect("Team", sorted(props_df['team'].unique()))
    
    # Display predictions table
    st.dataframe(filtered_props, use_container_width=True)
```

---

### Phase 2D: Backtesting & Validation (Week 8)
**Goal**: Validate model performance on historical props

**Tasks**:
1. Collect historical prop lines (if available)
2. Calculate historical hit rates by confidence tier
3. Track ROI by prop type
4. Build calibration plots

**Deliverable**: `scripts/backtest_player_props.py`

---

## LONG TERM (Months 3-6): Production & Advanced Features

### Phase 3A: Daily Prediction Pipeline (Month 3)
**Goal**: Automate daily prop predictions

**Components**:
1. **Data refresh script**: Update play-by-play weekly
2. **Model retrain schedule**: Monthly or weekly retraining
3. **Prediction generation**: Daily prop predictions for upcoming games
4. **Email notifications**: Send prop alerts for high-confidence plays

**Deliverable**: `scripts/daily_player_props_pipeline.py`

---

### Phase 3B: Lineup Optimization (Month 4)
**Goal**: Build DraftKings Pick 6 lineup optimizer

**Features**:
- Correlated player stacking (QB + WR from same team)
- Optimal lineup selection given prop lines
- Variance optimization (balanced vs. boom-or-bust lineups)

**Deliverable**: `player_props/lineup_optimizer.py`

```python
"""
DraftKings Pick 6 Lineup Optimizer.
Given prop predictions, build optimal 6-player lineups.
"""
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
import pandas as pd

def optimize_lineup(
    player_props: pd.DataFrame,
    n_players: int = 6,
    max_same_team: int = 3
) -> pd.DataFrame:
    """
    Optimize lineup selection for DraftKings Pick 6.
    
    Args:
        player_props: DataFrame with columns [player_name, team, position, 
                                              prop_type, prediction, confidence]
        n_players: Number of players to select (default 6 for Pick 6)
        max_same_team: Max players from same team
    
    Returns:
        Optimal lineup DataFrame
    """
    # Linear programming optimization
    prob = LpProblem("DK_Pick6_Lineup", LpMaximize)
    
    # Decision variables
    player_vars = LpVariable.dicts(
        "player",
        player_props.index,
        cat='Binary'
    )
    
    # Objective: Maximize sum of confidence scores
    prob += lpSum([
        player_vars[i] * player_props.loc[i, 'confidence']
        for i in player_props.index
    ])
    
    # Constraint: Select exactly n_players
    prob += lpSum([player_vars[i] for i in player_props.index]) == n_players
    
    # Constraint: Max players from same team
    for team in player_props['team'].unique():
        team_players = player_props[player_props['team'] == team].index
        prob += lpSum([player_vars[i] for i in team_players]) <= max_same_team
    
    # Solve
    prob.solve()
    
    # Extract selected players
    selected = [i for i in player_props.index if player_vars[i].varValue == 1]
    return player_props.loc[selected]
```

---

### Phase 3C: Advanced Models (Month 5)
**Goal**: Implement ensemble and neural network models

**Techniques**:
1. **Ensemble Models**: Combine XGBoost, LightGBM, CatBoost
2. **Neural Networks**: LSTM for time-series player performance
3. **Bayesian Models**: Uncertainty quantification for prop predictions

**Deliverable**: `player_props/advanced_models.py`

---

### Phase 3D: Injury & Lineup Integration (Month 6)
**Goal**: Incorporate real-time injury/lineup data

**Data Sources**:
- ESPN injury reports API
- Depth chart scraping
- Vegas line movements (injury indicators)

**Impact**:
- Adjust predictions when key players are out
- Boost backup RB/WR projections
- Detect volume opportunity shifts

**Deliverable**: `player_props/injury_adjustments.py`

---

## Summary Timeline

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **1A: Data Exploration** | Days 1-2 | `explore_player_stats.py` |
| **1B: Aggregation** | Days 3-4 | `aggregators.py` |
| **1C: Baseline Models** | Days 5-7 | `models.py` (3 prop types) |
| **2A: Feature Engineering** | Weeks 3-4 | `feature_engineering.py` |
| **2B: Prop Expansion** | Weeks 4-5 | 5 new prop models |
| **2C: UI Integration** | Weeks 6-7 | Player Props tab in UI |
| **2D: Backtesting** | Week 8 | `backtest_player_props.py` |
| **3A: Daily Pipeline** | Month 3 | Automated prop generation |
| **3B: Lineup Optimizer** | Month 4 | `lineup_optimizer.py` |
| **3C: Advanced Models** | Month 5 | Ensemble/neural models |
| **3D: Injury Integration** | Month 6 | Real-time adjustments |

---

## Installation & Setup

### 1. Create Player Props Module
```bash
# From project root
mkdir -p player_props/models
touch player_props/__init__.py
```

### 2. Install Additional Dependencies
```bash
# Activate venv
.\venv\Scripts\Activate.ps1

# Install optimization library for lineup optimizer
pip install pulp

# Install neural network libraries (for Phase 3C)
pip install tensorflow keras

# Update requirements.txt
pip freeze > requirements.txt
```

### 3. Run Initial Exploration
```bash
# Create scripts directory if it doesn't exist
mkdir -p scripts

# Run data exploration
python scripts/explore_player_stats.py
```

---

## Next Steps (Week 1)
1. ✅ Create `scripts/explore_player_stats.py` and run it
2. Build `player_props/aggregators.py` 
3. Generate game-level player stats CSVs
4. Start baseline model development

**Decision Point**: After Phase 1C, evaluate model performance. If MAE is acceptable (e.g., <20 yards for passing), proceed to Phase 2. Otherwise, iterate on features.
