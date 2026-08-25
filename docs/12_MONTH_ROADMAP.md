# NFL Predictions — 12-Month Feature Roadmap

> **Implementation review — 2026-08-24.** Q1 is only partially complete. V2 has prior-only team EPA, success rate, pass/rush EPA, explosive-play, sack/hit, rest, weather, and travel-ready primitives; it lacks CPOE and source adapters. Legacy player-prop code has rolling target-share-related aggregation, but no V2 player opportunity feed. QRC, referee profiles, LSTM/transformer models, the Q2–Q4 product features, and all roadmap UI modules are not present. Reprioritization is required before continuing with this feature sequence; see `IMPLEMENTATION_PRIORITIES_2026.md`.

> Generated: 2026-07-31 | Horizon: August 2026 – July 2027

---

## Executive Summary

This roadmap expands the NFL platform from spread/moneyline/total core models into a
full-season analytics suite with live-game intelligence, advanced player props, deep
team tendencies analysis, and an AI-powered pre-game briefing system.

---

## Q1 (Aug–Oct 2026) — Season Prep & Data Hardening

### Feature 1 — Next-Gen Stats Integration (EPA, CPOE, Success Rate)

Pull NGS data via `nfl_data_py` to add EPA per play, CPOE (completion % over
expected), and Success Rate as first-class model features.

```python
# src/features/ngs_features.py
import nfl_data_py as nfl
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data_files")

def build_ngs_team_features(season: int) -> pd.DataFrame:
    """Build weekly team-level NGS aggregates."""
    pbp = nfl.import_pbp_data([season])
    pbp = pbp[pbp["play_type"].isin(["pass", "run"])]

    # EPA aggregates
    team_epa = (
        pbp.groupby(["season", "week", "posteam"])
        .agg(
            off_epa=("epa", "mean"),
            pass_epa=("pass_epa", "mean"),
            rush_epa=("rushing_epa", "mean"),
            success_rate=("success", "mean"),
            cpoe=("cpoe", "mean"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    # Defensive EPA (allowed)
    def_epa = (
        pbp.groupby(["season", "week", "defteam"])
        .agg(def_epa=("epa", "mean"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    combined = team_epa.merge(def_epa, on=["season", "week", "team"], how="left")
    out = DATA_DIR / f"ngs_features_{season}.parquet"
    combined.to_parquet(out, index=False)
    return combined

def rolling_ngs(df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """Add rolling-window NGS features without data leakage."""
    df = df.sort_values(["team", "season", "week"])
    for col in ["off_epa", "pass_epa", "rush_epa", "success_rate", "cpoe", "def_epa"]:
        df[f"{col}_r{window}"] = (
            df.groupby("team")[col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
    return df
```

### Feature 2 — Quarterback Rating Composite (QRC)

Build a single composite QB rating from PFF grade, EPA, CPOE, and air yards
to replace simplistic passer-rating features.

```python
# src/features/qb_composite.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def compute_qrc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Quarterback Rating Composite (QRC).
    df must contain: pff_grade, epa_per_dropback, cpoe, air_yards_per_att,
                     td_pct, int_pct, pressure_rate_allowed
    """
    features = ["pff_grade", "epa_per_dropback", "cpoe",
                 "air_yards_per_att", "td_pct", "int_pct_neg"]
    df["int_pct_neg"] = -df["int_pct"]  # invert so higher = better

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features].fillna(0))

    # Weights reflect predictive importance for game outcomes
    weights = np.array([0.25, 0.30, 0.20, 0.10, 0.10, 0.05])
    df["qrc"] = scaled @ weights
    df["qrc"] = (df["qrc"] - df["qrc"].min()) / (df["qrc"].max() - df["qrc"].min()) * 100
    return df
```

### Feature 3 — Offensive Line Grade Tracker

Ingest PFF OL grades weekly. Compute pass-block / run-block efficiency.
Feed into spread and total models as team-level features.

```python
# src/features/oline_grades.py
import pandas as pd
import requests
from pathlib import Path

DATA_DIR = Path("data_files")

# PFF grades require a subscription — alternatively scrape ProFootballReference
PFR_OL_BASE = "https://www.pro-football-reference.com/years/{year}/blocking.htm"

def load_oline_grades(season: int) -> pd.DataFrame:
    """Load or fetch OL efficiency metrics for given season."""
    cached = DATA_DIR / f"oline_grades_{season}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)

    import nfl_data_py as nfl
    pbp = nfl.import_pbp_data([season])
    pbp = pbp[pbp["play_type"].isin(["pass", "run"])]

    oline = (
        pbp.groupby(["season", "week", "posteam"])
        .agg(
            sacks_allowed=("sack", "sum"),
            pressures_approx=("qb_hit", "sum"),
            pass_att=("pass_attempt", "sum"),
            rush_yds=("rushing_yards", "mean"),
        )
        .reset_index()
    )
    oline["pressure_rate"] = (
        (oline["sacks_allowed"] + oline["pressures_approx"]) / oline["pass_att"].clip(1)
    )
    oline.to_parquet(cached, index=False)
    return oline
```

### Feature 4 — Target Share & Air Yards Distribution

For player props, compute weekly target share and air yards per receiver.
Use as primary feature for receiving yards / TD prop models.

```python
# src/features/receiver_targets.py
import nfl_data_py as nfl
import pandas as pd

def build_receiver_target_profiles(season: int, week: int) -> pd.DataFrame:
    pbp = nfl.import_pbp_data([season])
    passes = pbp[
        (pbp["play_type"] == "pass")
        & (pbp["week"] <= week)
        & (pbp["receiver_player_id"].notna())
    ]

    team_totals = (
        passes.groupby(["posteam", "week"])["pass_attempt"].sum()
        .reset_index().rename(columns={"pass_attempt": "team_att"})
    )

    receiver = (
        passes.groupby(["receiver_player_name", "posteam", "week"])
        .agg(
            targets=("pass_attempt", "sum"),
            air_yards=("air_yards", "mean"),
            yards=("yards_gained", "sum"),
            tds=("touchdown", "sum"),
            wopr=("wopr", "mean"),
        )
        .reset_index()
    )
    receiver = receiver.merge(team_totals, on=["posteam", "week"])
    receiver["target_share"] = receiver["targets"] / receiver["team_att"].clip(1)
    return receiver
```

### Feature 5 — Referee Tendency Analysis

Compute penalty rates, scoring, and game pace per referee crew.
Integrate penalty-adjusted totals and pace features into model training.

```python
# src/features/referee_tendencies.py
import pandas as pd
import nfl_data_py as nfl

def build_ref_profiles(seasons: list[int]) -> pd.DataFrame:
    games = nfl.import_schedules(seasons)
    games = games[games["game_type"] == "REG"]

    ref_stats = (
        games.groupby("referee")
        .agg(
            games=("game_id", "count"),
            avg_total=("total", "mean"),
            avg_home_score=("home_score", "mean"),
            avg_away_score=("away_score", "mean"),
            avg_penalties=("home_rush_atts", "mean"),  # proxy if penalty data unavailable
        )
        .reset_index()
    )
    league_avg = games["total"].mean()
    ref_stats["pace_factor"] = ref_stats["avg_total"] / league_avg
    return ref_stats
```

---

## Q2 (Nov 2026 – Jan 2027) — Player Props Deep Expansion

### Feature 6 — LSTM Sequence Model for Player Props

Replace XGBoost rolling-window props with an LSTM that natures sequential
game-by-game player performance patterns.

```python
# src/models/lstm_props.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

SEQUENCE_LEN = 6  # last 6 games

def build_lstm_prop_model(n_features: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(SEQUENCE_LEN, n_features)),
        layers.LSTM(64, return_sequences=True, dropout=0.2),
        layers.LSTM(32, dropout=0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),  # regression output
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def prepare_sequences(df: pd.DataFrame, features: list[str], target: str):
    """Build (X, y) sequences for LSTM training."""
    df = df.sort_values(["player_id", "season", "week"])
    Xs, ys = [], []
    for _, player_df in df.groupby("player_id"):
        vals = player_df[features].values
        targets = player_df[target].values
        for i in range(SEQUENCE_LEN, len(vals)):
            Xs.append(vals[i - SEQUENCE_LEN:i])
            ys.append(targets[i])
    return np.array(Xs), np.array(ys)

def train_lstm_props(df: pd.DataFrame, features: list[str], target: str,
                     epochs: int = 50) -> tf.keras.Model:
    X, y = prepare_sequences(df, features, target)
    split = int(len(X) * 0.8)
    model = build_lstm_prop_model(len(features))
    model.fit(X[:split], y[:split], validation_data=(X[split:], y[split:]),
              epochs=epochs, batch_size=32, verbose=0)
    mae = model.evaluate(X[split:], y[split:], verbose=0)[1]
    print(f"LSTM props {target} MAE: {mae:.2f}")
    return model
```

### Feature 7 — Defensive Position Matchup Matrix

For each skill-position player, compute historical stats vs opposing
defensive schemes (Cover-2, Cover-3, man, zone). Feed into prop models.

```python
# src/features/defensive_matchup.py
import pandas as pd
import nfl_data_py as nfl

COVERAGE_COLS = ["cover_1", "cover_2", "cover_3", "quarters", "prevent", "man_coverage"]

def build_defensive_matchup_table(season: int) -> pd.DataFrame:
    """Compute yards allowed by defense vs position per coverage shell."""
    pbp = nfl.import_pbp_data([season])
    pbp = pbp[pbp["play_type"] == "pass"]
    pbp = pbp[pbp["receiver_player_name"].notna()]

    # Coverage type available in NGS data (season 2016+)
    if "coverage_type" not in pbp.columns:
        pbp["coverage_type"] = "unknown"

    matchup = (
        pbp.groupby(["defteam", "coverage_type", "receiver_position"])
        .agg(
            targets=("pass_attempt", "sum"),
            yards=("yards_gained", "mean"),
            tds=("touchdown", "mean"),
            comp_pct=("complete_pass", "mean"),
        )
        .reset_index()
    )
    return matchup
```

### Feature 8 — Snap Count Efficiency Tracker

Track snap count by player. Flag newly elevated snap-count players as
injury-driven value in player prop markets.

```python
# src/features/snap_counts.py
import pandas as pd
import nfl_data_py as nfl

def get_snap_count_trends(season: int, team: str) -> pd.DataFrame:
    snaps = nfl.import_snap_counts([season])
    team_snaps = snaps[snaps["team"] == team].copy()
    team_snaps = team_snaps.sort_values(["player", "week"])

    team_snaps["snap_pct_delta"] = (
        team_snaps.groupby("player")["offense_pct"]
        .transform(lambda x: x.diff())
    )
    # Flag players with snap share jump > 15 ppts
    team_snaps["snap_elevation_flag"] = team_snaps["snap_pct_delta"] > 0.15
    return team_snaps[
        ["player", "week", "offense_pct", "snap_pct_delta", "snap_elevation_flag"]
    ]
```

### Feature 9 — Anytime TD Scorer Probability Model

Train a binary classifier per game predicting which players score a TD.
Use target share, red zone opportunities, snap count, and defensive TD-allowed.

```python
# src/models/td_scorer.py
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib
from pathlib import Path

FEATURES = [
    "target_share_l4", "red_zone_targets_l4", "snap_pct_l4",
    "def_td_allowed_per_game", "def_red_zone_td_pct",
    "home_flag", "game_total", "team_implied_score",
    "player_td_rate_l8",
]

def train_td_scorer(df: pd.DataFrame) -> XGBClassifier:
    X = df[FEATURES].fillna(0)
    y = df["scored_td"].astype(int)
    model = XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.7,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        eval_metric="auc",
    )
    model.fit(X, y, verbose=False)
    joblib.dump(model, Path("models/td_scorer.joblib"))
    return model
```

### Feature 10 — Two-Minute Drill & Clutch Performance Index

Track team performance specifically in fourth-quarter, within-7 game situations.
Quantify clutch vs choke tendencies to adjust spread model for close-game scenarios.

```python
# src/features/clutch_index.py
import nfl_data_py as nfl
import pandas as pd

def compute_clutch_index(seasons: list[int]) -> pd.DataFrame:
    """Win rate and scoring efficiency in Q4 close-game situations."""
    pbp = nfl.import_pbp_data(seasons)
    clutch = pbp[
        (pbp["qtr"] == 4) &
        (pbp["score_differential"].abs() <= 8) &
        (pbp["play_type"].isin(["pass", "run"]))
    ]

    off = (
        clutch.groupby(["season", "posteam"])
        .agg(
            clutch_epa=("epa", "mean"),
            clutch_success=("success", "mean"),
            clutch_tds=("touchdown", "sum"),
        )
        .reset_index().rename(columns={"posteam": "team"})
    )
    def_ = (
        clutch.groupby(["season", "defteam"])
        .agg(def_clutch_epa=("epa", "mean"))
        .reset_index().rename(columns={"defteam": "team"})
    )
    df = off.merge(def_, on=["season", "team"])
    df["clutch_index"] = df["clutch_epa"] - df["def_clutch_epa"]
    return df
```

---

## Q3 (Feb–Apr 2027) — Visual Analytics & UI Overhaul

### Feature 11 — Matchup Radar Chart per Game

For each upcoming game, render a radar chart comparing 8 key team stats.
Team A vs Team B on offense EPA, defense EPA, pass rush, run blocking, etc.

```python
# pages/4_matchup_analysis.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np

def render_matchup_radar(home_stats: dict, away_stats: dict,
                          home_team: str, away_team: str) -> None:
    categories = [
        "Off EPA/Play", "Def EPA/Play", "Pass Rush",
        "Run Block", "Clutch Index", "Red Zone Eff",
        "Third Down %", "Turnover Margin",
    ]
    home_vals = [home_stats.get(k, 0) for k in categories]
    away_vals = [away_stats.get(k, 0) for k in categories]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=home_vals, theta=categories, fill="toself",
                                   name=home_team, line=dict(color="#2196F3")))
    fig.add_trace(go.Scatterpolar(r=away_vals, theta=categories, fill="toself",
                                   name=away_team, line=dict(color="#FF5722")))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
        showlegend=True, template="plotly_dark",
        title=f"{home_team} vs {away_team} — Team Radar",
    )
    st.plotly_chart(fig, width="stretch")
```

### Feature 12 — Model Transparency Explainer (SHAP)

Generate SHAP waterfall plots for each game prediction so users can see
which features drove the model toward a specific pick.

```python
# pages/explainer.py
import streamlit as st
import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def render_shap_explainer(game_features: pd.DataFrame, model_name: str = "spread") -> None:
    model = joblib.load(Path("models") / f"{model_name}_model.joblib")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(game_features)

    st.subheader(f"Model Explanation — {model_name.title()} Prediction")
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=game_features.iloc[0],
            feature_names=game_features.columns.tolist(),
        ),
        show=False,
    )
    st.pyplot(fig, use_container_width=False)
    plt.close()
```

### Feature 13 — Drive Chart Visualizer

After game completion, render an SVG-like drive chart showing each possession:
direction, yards gained, outcome (punt/TD/FG/turnover).

```python
# pages/components/drive_chart.py
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def render_drive_chart(drives: pd.DataFrame) -> None:
    """drives columns: drive_num, team, start_yardline, yards_gained, result, color."""
    fig = go.Figure()
    for _, d in drives.iterrows():
        fig.add_shape(
            type="rect",
            x0=d["start_yardline"], x1=d["start_yardline"] + d["yards_gained"],
            y0=d["drive_num"] - 0.4, y1=d["drive_num"] + 0.4,
            fillcolor=d.get("color", "#2196F3"), opacity=0.7,
            line=dict(width=0),
        )
        fig.add_annotation(
            x=d["start_yardline"] + d["yards_gained"] / 2,
            y=d["drive_num"], text=d["result"], showarrow=False,
            font=dict(size=9, color="white"),
        )
    fig.update_layout(
        xaxis=dict(range=[0, 100], title="Field Position"),
        yaxis=dict(title="Drive #"),
        title="Drive Chart",
        template="plotly_dark",
        height=max(300, len(drives) * 30),
    )
    st.plotly_chart(fig, width="stretch")
```

### Feature 14 — Division Race Tracker

Live division standings, playoff seed probabilities (from season simulator),
and magic-number calculator. Updated daily via GitHub Actions.

```python
# pages/division_race.py
import streamlit as st
import pandas as pd
import nfl_data_py as nfl

def render_division_race(sim_df: pd.DataFrame) -> None:
    st.title("Division Race Tracker")
    games = nfl.import_schedules([2026])
    standings = _compute_standings(games)
    divisions = standings["division"].unique()

    for div in sorted(divisions):
        st.subheader(div)
        div_df = standings[standings["division"] == div].merge(
            sim_df[["team", "playoff_pct", "div_winner_pct"]],
            on="team", how="left",
        )
        st.dataframe(
            div_df[["team", "wins", "losses", "ties", "pct",
                    "div_winner_pct", "playoff_pct"]]
            .sort_values("pct", ascending=False)
            .style.format({"div_winner_pct": "{:.1%}", "playoff_pct": "{:.1%}",
                           "pct": "{:.3f}"}),
            width="stretch",
        )

def _compute_standings(games: pd.DataFrame) -> pd.DataFrame:
    finished = games[games["home_score"].notna()]
    # … win/loss/tie aggregation by team …
    return pd.DataFrame()  # placeholder
```

### Feature 15 — Line Movement Tracker

Capture opening and current spread/total every 30 minutes. Display line
movement charts and flag sharp-money moves (line moves opposite public %).

```python
# src/ingestion/line_movement.py
import pandas as pd, requests, json
from datetime import datetime
from pathlib import Path

SNAPSHOTS_DIR = Path("data_files/line_snapshots")
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

def capture_line_snapshot(sport: str = "americanfootball_nfl") -> None:
    resp = requests.get(
        f"https://api.the-odds-api.com/v4/sports/{sport}/odds/",
        params={"apiKey": os.environ["ODDS_API_KEY"], "regions": "us",
                "markets": "spreads,totals", "oddsFormat": "american"},
        timeout=10,
    )
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    snap_file = SNAPSHOTS_DIR / f"snapshot_{ts}.json"
    snap_file.write_text(json.dumps(resp.json()))

def load_line_history(game_id: str) -> pd.DataFrame:
    rows = []
    for snap in sorted(SNAPSHOTS_DIR.glob("snapshot_*.json")):
        data = json.loads(snap.read_text())
        ts = snap.stem.replace("snapshot_", "")
        for game in data:
            if game["id"] != game_id:
                continue
            for book in game.get("bookmakers", []):
                for mkt in book.get("markets", []):
                    rows.append({
                        "timestamp": ts, "book": book["key"],
                        "market": mkt["key"],
                        "outcome": mkt["outcomes"][0]["name"],
                        "point": mkt["outcomes"][0].get("point"),
                        "price": mkt["outcomes"][0]["price"],
                    })
    return pd.DataFrame(rows)
```

---

## Q4 (May–Jul 2027) — Advanced ML & Automation

### Feature 16 — Transformer-Based Game Outcome Predictor

Fine-tune a small transformer encoder on game-level tabular data as a
supplement to gradient boosting. Use cross-attention between home/away features.

```python
# src/models/transformer_predictor.py
import torch, torch.nn as nn
import pandas as pd, numpy as np

class GameTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2):
        super().__init__()
        self.home_embed = nn.Linear(n_features, d_model)
        self.away_embed = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=256, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, 64), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, home: torch.Tensor, away: torch.Tensor) -> torch.Tensor:
        h = self.home_embed(home).unsqueeze(1)
        a = self.away_embed(away).unsqueeze(1)
        combined = torch.cat([h, a], dim=1)
        enc = self.encoder(combined)
        flat = enc.flatten(1)
        return self.head(flat).squeeze(-1)

def train_transformer(X_home, X_away, y, epochs: int = 100) -> GameTransformer:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GameTransformer(n_features=X_home.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCELoss()

    Xh = torch.FloatTensor(X_home).to(device)
    Xa = torch.FloatTensor(X_away).to(device)
    Yt = torch.FloatTensor(y).to(device)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(Xh, Xa)
        loss = loss_fn(pred, Yt)
        loss.backward()
        opt.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: loss={loss.item():.4f}")
    return model
```

### Feature 17 — Injury-Adjusted Power Rankings

Weekly team power rankings incorporating current injury report (IR, Questionable,
Doubtful). Adjust EPA-based ranking by estimated WAR lost.

```python
# src/models/power_rankings.py
import pandas as pd, numpy as np
from src.features.ngs_features import build_ngs_team_features

INJURY_MULTIPLIERS = {"Out": -0.8, "Doubtful": -0.4, "Questionable": -0.15}

def compute_power_rankings(season: int, week: int, injuries: pd.DataFrame) -> pd.DataFrame:
    ngs = build_ngs_team_features(season)
    recent = ngs[ngs["week"] <= week].groupby("team").tail(4)
    base = recent.groupby("team")[["off_epa", "def_epa"]].mean().reset_index()
    base["raw_power"] = base["off_epa"] - base["def_epa"]

    # Apply injury penalties
    for _, inj in injuries.iterrows():
        mult = INJURY_MULTIPLIERS.get(inj["status"], 0)
        base.loc[base["team"] == inj["team"], "raw_power"] += inj["war_per_game"] * mult

    base["power_rank"] = base["raw_power"].rank(ascending=False).astype(int)
    return base.sort_values("power_rank")
```

### Feature 18 — Weather Impact on Totals Model

Train a dedicated weather → scoring adjustment model. Features: temperature,
wind speed, precipitation probability, dome flag.

```python
# src/models/weather_totals.py
import pandas as pd
from xgboost import XGBRegressor
import joblib
from pathlib import Path

WEATHER_FEATURES = [
    "temperature_f", "wind_speed_mph", "wind_direction_cross",
    "precip_probability", "is_dome", "is_cold_weather_team",
]

def train_weather_model(df: pd.DataFrame) -> None:
    """Predict total run/point deviation from model baseline due to weather."""
    X = df[WEATHER_FEATURES].fillna(0)
    y = df["total_deviation"]  # actual total - pre-weather model total
    model = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05)
    model.fit(X, y)
    joblib.dump(model, Path("models/weather_totals.joblib"))

def predict_weather_adjustment(park: str, date: str, hour: int) -> float:
    from src.ingestion.weather_engine import fetch_game_weather
    w = fetch_game_weather(park, date, hour)
    model = joblib.load(Path("models/weather_totals.joblib"))
    X = pd.DataFrame([{
        "temperature_f": w["temp_f"],
        "wind_speed_mph": w["wind_mph"],
        "wind_direction_cross": abs(np.sin(np.radians(w["wind_dir"]))),
        "precip_probability": w["precip_pct"] / 100,
        "is_dome": 0, "is_cold_weather_team": 0,
    }])
    return float(model.predict(X)[0])
```

### Feature 19 — AI Pre-Game Briefing (LLM Summary)

Use a local LLM (Ollama) or OpenAI GPT-4o-mini to generate a 3-paragraph
pre-game briefing covering model edge, key matchup factors, and injury news.

```python
# src/picks/ai_briefing.py
import os, json
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

BRIEFING_TEMPLATE = """
You are an expert NFL analyst and sharp bettor. Given the following game data,
write a concise 3-paragraph pre-game briefing (max 200 words total):
1. Model recommendation and edge explanation
2. Key matchup factors (top 3)
3. Injury news and impact

Game Data:
{game_json}

Style: direct, data-driven, no fluff.
"""

def generate_briefing(game: dict) -> str:
    prompt = BRIEFING_TEMPLATE.format(game_json=json.dumps(game, indent=2))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.4,
    )
    return resp.choices[0].message.content

def render_briefing_in_ui(game: dict) -> None:
    import streamlit as st
    with st.expander("🤖 AI Pre-Game Briefing", expanded=True):
        with st.spinner("Generating analysis..."):
            briefing = generate_briefing(game)
        st.markdown(briefing)
```

### Feature 20 — Historical ATS / O-U Trend Cards

For each team, calculate against-the-spread (ATS) and over/under records
over the last 1/2/3 seasons. Surface home/away/divisional splits.

```python
# src/analytics/ats_trends.py
import pandas as pd
import nfl_data_py as nfl

def compute_ats_record(games: pd.DataFrame) -> pd.DataFrame:
    """Compute ATS and O/U record for each team."""
    g = games.copy()
    g = g[g["spread_line"].notna() & g["total_line"].notna()]

    # Home ATS
    g["home_ats_cover"] = (
        g["home_score"] - g["away_score"]
    ) > -g["spread_line"]  # negative spread = home favored

    # Total
    g["went_over"] = g["home_score"] + g["away_score"] > g["total_line"]

    home_ats = (
        g.groupby("home_team")
        .agg(home_ats_covers=("home_ats_cover", "sum"),
             home_games=("game_id", "count"),
             home_overs=("went_over", "sum"))
        .reset_index().rename(columns={"home_team": "team"})
    )
    away_ats = (
        g.groupby("away_team")
        .agg(away_ats_covers=("home_ats_cover", lambda x: (~x).sum()),
             away_games=("game_id", "count"),
             away_overs=("went_over", "sum"))
        .reset_index().rename(columns={"away_team": "team"})
    )
    combined = home_ats.merge(away_ats, on="team")
    combined["total_ats_covers"] = combined["home_ats_covers"] + combined["away_ats_covers"]
    combined["total_games"] = combined["home_games"] + combined["away_games"]
    combined["ats_pct"] = combined["total_ats_covers"] / combined["total_games"]
    combined["total_overs"] = combined["home_overs"] + combined["away_overs"]
    combined["over_pct"] = combined["total_overs"] / combined["total_games"]
    return combined.sort_values("ats_pct", ascending=False)
```

### Feature 21 — Punt Return / Special Teams Value Model

Quantify hidden value in special teams play (field position, return TDs,
kick coverage). Feed into game-total and moneyline models.

```python
# src/features/special_teams.py
import nfl_data_py as nfl
import pandas as pd

def compute_st_value(seasons: list[int]) -> pd.DataFrame:
    pbp = nfl.import_pbp_data(seasons)
    punts = pbp[pbp["play_type"] == "punt"]
    kicks = pbp[pbp["play_type"] == "kickoff"]

    punt_val = (
        punts.groupby(["season", "posteam"])
        .agg(
            avg_net_punt=("kick_distance", "mean"),
            punt_tds=("touchdown", "sum"),
        )
        .reset_index().rename(columns={"posteam": "team"})
    )
    kick_val = (
        kicks.groupby(["season", "return_team"])
        .agg(
            avg_kr_yds=("return_yards", "mean"),
            kr_tds=("touchdown", "sum"),
        )
        .reset_index().rename(columns={"return_team": "team"})
    )
    st = punt_val.merge(kick_val, on=["season", "team"], how="outer")
    lg_avg_net = punts["kick_distance"].mean()
    st["punt_value_over_avg"] = st["avg_net_punt"] - lg_avg_net
    return st
```

### Feature 22 — Divisional Dog Bias Adjustment

NFL division games historically produce smaller spreads and more upsets.
Detect and adjust model probability when teams meet in division.

```python
# src/models/divisional_adjustment.py
import pandas as pd
import nfl_data_py as nfl

DIVISIONAL_ATS_ADJUSTMENT = 0.03  # ~3% win-prob shift toward dog in div games

def apply_divisional_adjustment(
    home_team: str, away_team: str, home_win_prob: float,
    team_division_map: dict[str, str],
) -> float:
    same_div = team_division_map.get(home_team) == team_division_map.get(away_team)
    if not same_div:
        return home_win_prob
    # Regress toward 50% slightly in divisional games
    return home_win_prob * (1 - DIVISIONAL_ATS_ADJUSTMENT) + 0.5 * DIVISIONAL_ATS_ADJUSTMENT
```

### Feature 23 — Bet Slip PDF Export

Allow users to export the day's picks as a formatted PDF pick slip with
game time, pick, odds, model prob, and edge for each bet.

```python
# scripts/export_pick_slip.py
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from io import BytesIO
import pandas as pd

def generate_pick_slip_pdf(picks: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                             rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = [Paragraph("🏈 NFL Daily Pick Slip", styles["Title"]), Spacer(1, 12)]

    headers = ["Game", "Pick", "Bet Type", "Odds", "Edge", "Model Prob", "Tier"]
    data = [headers]
    for _, row in picks.iterrows():
        data.append([
            row["game"], row["pick"], row["bet_type"],
            f"{row['odds']:+d}", f"{row['edge']:+.1%}",
            f"{row['model_prob']:.1%}", row["tier"],
        ])

    table = Table(data, colWidths=[140, 80, 70, 50, 55, 70, 55])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565C0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#E3F2FD")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(table)
    doc.build(story)
    return buffer.getvalue()
```

### Feature 24 — Feedback Loop: Track Your Own Bets

Let users log their own bets (with notes). Compare their personal ROI against
the model's recommended bets. Stored in browser localStorage via Streamlit.

```python
# streamlit_app/pages/my_bets.py
import streamlit as st
import pandas as pd
import json
from datetime import date

def render_my_bets() -> None:
    st.title("My Bet Tracker")

    if "my_bets" not in st.session_state:
        st.session_state["my_bets"] = []

    with st.form("add_bet"):
        col1, col2, col3 = st.columns(3)
        game = col1.text_input("Game")
        pick = col2.text_input("Pick")
        odds = col3.number_input("Odds", value=-110)
        stake = st.number_input("Stake ($)", value=10.0, step=5.0)
        outcome = st.selectbox("Outcome", ["Pending", "Win", "Loss", "Push"])
        submitted = st.form_submit_button("Add Bet")
        if submitted and game:
            pnl = 0.0
            if outcome == "Win":
                pnl = stake * ((100 / abs(odds) + 1) if odds < 0 else (odds / 100 + 1)) - stake
            elif outcome == "Loss":
                pnl = -stake
            st.session_state["my_bets"].append({
                "date": str(date.today()), "game": game, "pick": pick,
                "odds": odds, "stake": stake, "outcome": outcome, "pnl": pnl,
            })

    if st.session_state["my_bets"]:
        df = pd.DataFrame(st.session_state["my_bets"])
        total_pnl = df["pnl"].sum()
        st.metric("Total P&L", f"${total_pnl:+.2f}")
        st.metric("Win Rate", f"{(df['outcome']=='Win').mean():.1%}")
        st.dataframe(df, width="stretch")
```

---

## Timeline Summary

| Quarter | Focus | Key Deliverables |
|---------|-------|-----------------|
| Q1 Aug–Oct 2026 | Data hardening | NGS features, QRC, OLine grades, target share, referee tendencies |
| Q2 Nov 2026–Jan 2027 | Player props deep | LSTM props, defensive matchup, snap counts, TD scorer, clutch index |
| Q3 Feb–Apr 2027 | Visual analytics | Radar charts, SHAP explainer, drive chart, division race, line movement |
| Q4 May–Jul 2027 | Advanced ML | Transformer model, injury power rankings, weather model, AI briefing, ATS trends, ST value, divisional adj, PDF export, personal tracker |
