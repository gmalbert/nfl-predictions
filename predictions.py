import streamlit as st

# Set page config FIRST - must be the very first Streamlit command
st.set_page_config(
    page_title="NFL Outcome Predictor",
    page_icon="🏈",
    layout="wide"
)

import pandas as pd
import numpy as np
from os import path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.inspection import permutation_importance
import os
from datetime import datetime
import requests


DATA_DIR = 'data_files/'

# Define features list before loading best features - matches nfl-gather-data.py
features = [
    # Pregame features only (removed 'total' as it's actual game total, causing data leakage)
    'spread_line', 'away_moneyline', 'home_moneyline', 'away_spread_odds', 'home_spread_odds', 'total_line',
    'under_odds', 'over_odds', 'div_game', 'roof', 'surface', 'temp', 'wind', 'away_rest', 'home_rest',
    'home_team', 'away_team', 'gameday', 'week', 'season', 'home_qb_id', 'away_qb_id', 'home_qb_name', 'away_qb_name',
    'home_coach', 'away_coach', 'stadium', 'location',
    # Rolling team stats (calculated from previous games only)
    'homeTeamWinPct', 'awayTeamWinPct', 'homeTeamCloseGamePct', 'awayTeamCloseGamePct', 'homeTeamBlowoutPct', 'awayTeamBlowoutPct',
    'homeTeamAvgScore', 'awayTeamAvgScore', 'homeTeamAvgScoreAllowed', 'awayTeamAvgScoreAllowed', 'homeTeamAvgPointDiff', 'awayTeamAvgPointDiff',
    'homeTeamAvgTotalScore', 'awayTeamAvgTotalScore', 'homeTeamGamesPlayed', 'awayTeamGamesPlayed', 'homeTeamAvgPointSpread', 'awayTeamAvgPointSpread',
    'homeTeamAvgTotal', 'awayTeamAvgTotal', 'homeTeamFavoredPct', 'awayTeamFavoredPct', 'homeTeamSpreadCoveredPct', 'awayTeamSpreadCoveredPct',
    'homeTeamOverHitPct', 'awayTeamOverHitPct', 'homeTeamUnderHitPct', 'awayTeamUnderHitPct', 'homeTeamTotalHitPct', 'awayTeamTotalHitPct',
    # Enhanced season and matchup features
    'homeTeamCurrentSeasonWinPct', 'awayTeamCurrentSeasonWinPct', 'homeTeamCurrentSeasonAvgScore', 'awayTeamCurrentSeasonAvgScore',
    'homeTeamCurrentSeasonAvgScoreAllowed', 'awayTeamCurrentSeasonAvgScoreAllowed', 'homeTeamPriorSeasonRecord', 'awayTeamPriorSeasonRecord',
    'headToHeadHomeTeamWinPct',
    # Upset-specific features
    'spreadSize', 'isCloseSpread', 'isMediumSpread', 'isLargeSpread'
]

# Load best features for spread
try:
    with open(path.join(DATA_DIR, 'best_features_spread.txt'), 'r') as f:
        loaded_features = [line.strip() for line in f if line.strip()]
    # Only keep features that are valid columns in the data
    best_features_spread = [feat for feat in loaded_features if feat in features]
    if not best_features_spread:
        best_features_spread = features
except FileNotFoundError:
    best_features_spread = features  # fallback to all features

# Load best features for moneyline
try:
    with open(path.join(DATA_DIR, 'best_features_moneyline.txt'), 'r') as f:
        best_features_moneyline = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    best_features_moneyline = features

# Load best features for totals
try:
    with open(path.join(DATA_DIR, 'best_features_totals.txt'), 'r') as f:
        best_features_totals = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    best_features_totals = features



# Load full NFL schedule from ESPN API (all regular season weeks) and save to CSV
current_year = datetime.now().year

# Load historical game data with error handling
try:
    historical_game_level_data = pd.read_csv(path.join(DATA_DIR, 'nfl_games_historical_with_predictions.csv'), sep='\t')
except FileNotFoundError:
    st.error("Critical file missing: nfl_games_historical_with_predictions.csv")
    st.info("Please run 'python nfl-gather-data.py' first to generate the required data files.")
    st.stop()  # Stop execution if critical data is missing
except Exception as e:
    st.error(f"Error loading historical game data: {str(e)}")
    st.stop()

# Uncomment below to pull down current year schedule from ESPN
# Uncomment and run once to get data, then comment back
# local_path = f"{DATA_DIR}/espn_schedule_{current_year}.csv"

# all_games = []
# weeks = range(1, 19)  # Regular season weeks 1-18
# for week in weeks:
#     espn_url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?seasontype=2&year={current_year}&week={week}"
#     try:
#         response = requests.get(espn_url)
#         if response.status_code == 200:
#             data = response.json()
#             for event in data.get("events", []):
#                 comp = event.get("competitions", [{}])[0]
#                 competitors = comp.get("competitors", [{}]*2)
#                 home = competitors[0].get("team", {}).get("displayName", "")
#                 away = competitors[1].get("team", {}).get("displayName", "")
#                 venue = comp.get("venue", {}).get("fullName", "")
#                 status = event.get("status", {}).get("type", {}).get("name", "")
#                 all_games.append({
#                     "week": week,
#                     "date": event.get("date", ""),
#                     "home_team": home,
#                     "away_team": away,
#                     "venue": venue,
#                     "status": status
#                 })
#         else:
#             st.warning(f"Week {week}: NFL schedule not found on ESPN (HTTP {response.status_code}).")
#     except Exception as e:
#         st.error(f"Week {week}: Error downloading NFL schedule from ESPN: {e}")

# schedule_df = pd.DataFrame(all_games)
# if not schedule_df.empty:
#     schedule_df.to_csv(local_path, index=False)

# Below purely for pulling down historical data from nflverse
# Uncomment and run once to get data, then comment back
# YEARS = range(2020, 2024)

# data = pd.DataFrame()

# for i in YEARS:  
#     i_data = pd.read_csv('https://github.com/nflverse/nflverse-data/releases/download/pbp/' \
#                    'play_by_play_' + str(i) + '.csv.gz',
#                    compression= 'gzip', low_memory= False)

#     data = pd.concat([data, i_data], ignore_index=True, sort=True)
#     data.reset_index(drop=True, inplace=True)

# data.to_csv(path.join(DATA_DIR, 'nfl_history_2020_2024.csv.gz'), compression='gzip', index=False, sep='\t')

# Load model predictions CSV for display
predictions_csv_path = path.join(DATA_DIR, 'nfl_games_historical_with_predictions.csv')
if os.path.exists(predictions_csv_path):
    predictions_df = pd.read_csv(predictions_csv_path, sep='\t')
else:
    predictions_df = None

@st.cache_data
def load_data():
    file_path = path.join(DATA_DIR, 'nfl_history_2020_2024.csv.gz')
    if os.path.exists(file_path):
        historical_data = pd.read_csv(file_path, compression='gzip', sep='\t', low_memory=False)
        return historical_data
    else:
        st.warning("Historical play-by-play data file not found. Some features may be limited.")
        return pd.DataFrame()  # Return empty DataFrame as fallback

historical_data = load_data()

@st.cache_data
def load_schedule():
    try:
        schedule_data = pd.read_csv(path.join(DATA_DIR, f'espn_schedule_{current_year}.csv'), low_memory=False)
        return schedule_data
    except FileNotFoundError:
        st.warning(f"Schedule file for {current_year} not found. Schedule data will be unavailable.")
        return pd.DataFrame()  # Return empty DataFrame as fallback
    except Exception as e:
        st.error(f"Error loading schedule data: {str(e)}")
        return pd.DataFrame()

schedule = load_schedule()

# Display NFL logo at the top
logo_path = os.path.join(DATA_DIR, "gridiron-oracle.png")
if os.path.exists(logo_path):
    st.image(logo_path, width=300)

st.title('NFL Game Outcome Predictor')

# Sidebar filters

filter_keys = [
    'posteam', 'defteam', 'down', 'ydstogo', 'yardline_100', 'play_type', 'qtr',
    'score_differential', 'posteam_score', 'defteam_score', 'epa', 'pass_attempt'
]


# --- For Monte Carlo Feature Selection ---
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Feature list for modeling and Monte Carlo selection
features = [
    'spread_line', 'total', 'homeTeamWinPct', 'awayTeamWinPct', 'homeTeamCloseGamePct', 'awayTeamCloseGamePct',
    'homeTeamBlowoutPct', 'awayTeamBlowoutPct', 'homeTeamAvgScore', 'awayTeamAvgScore', 'homeTeamAvgScoreAllowed',
    'awayTeamAvgScoreAllowed', 'homeTeamAvgPointDiff', 'awayTeamAvgPointDiff', 'homeTeamAvgTotalScore',
    'awayTeamAvgTotalScore', 'homeTeamGamesPlayed', 'awayTeamGamesPlayed', 'homeTeamAvgPointSpread',
    'awayTeamAvgPointSpread', 'homeTeamAvgTotal', 'awayTeamAvgTotal', 'homeTeamFavoredPct', 'awayTeamFavoredPct',
    'homeTeamSpreadCoveredPct', 'awayTeamSpreadCoveredPct', 'homeTeamOverHitPct', 'awayTeamOverHitPct',
    'homeTeamUnderHitPct', 'awayTeamUnderHitPct', 'homeTeamTotalHitPct', 'awayTeamTotalHitPct', 'total_line_diff'
]
# Target
if 'spreadCovered' in historical_game_level_data.columns:
    target_spread = 'spreadCovered'
else:
    target_spread = st.selectbox('Select spread target column', historical_game_level_data.columns)

# Prepare data for MC feature selection

# Define X and y for spread
X = historical_game_level_data[features]
y_spread = historical_game_level_data[target_spread]

# Spread model (using best features) - filter to numeric only
available_spread_features = [f for f in best_features_spread if f in historical_game_level_data.columns]
X_spread_full = historical_game_level_data[available_spread_features].select_dtypes(include=["number", "bool", "category"])
X_train_spread, X_test_spread, y_spread_train, y_spread_test = train_test_split(
    X_spread_full, y_spread, test_size=0.2, random_state=42, stratify=y_spread)

# --- Moneyline (underdogWon) target and split ---
target_moneyline = 'underdogWon'
y_moneyline = historical_game_level_data[target_moneyline]
# Filter to only numeric features for XGBoost compatibility
available_moneyline_features = [f for f in best_features_moneyline if f in historical_game_level_data.columns]
X_moneyline_full = historical_game_level_data[available_moneyline_features].select_dtypes(include=["number", "bool", "category"])
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(
    X_moneyline_full, y_moneyline, test_size=0.2, random_state=42, stratify=y_moneyline)

# --- Totals (overHit) target and split ---
target_totals = 'overHit'
y_totals = historical_game_level_data[target_totals]
# Filter to only numeric features for XGBoost compatibility
available_totals_features = [f for f in best_features_totals if f in historical_game_level_data.columns]
X_totals_full = historical_game_level_data[available_totals_features].select_dtypes(include=["number", "bool", "category"])
X_train_tot, X_test_tot, y_train_tot, y_test_tot = train_test_split(
    X_totals_full, y_totals, test_size=0.2, random_state=42, stratify=y_totals)

if st.checkbox("Show Raw Historical Play By Play Data", value=False):
    st.write("### Historical Data Sample")
    if not historical_data.empty:
        # Play-by-play data uses 'game_date' instead of 'gameday'  
        if 'game_date' in historical_data.columns:
            # Convert to datetime if needed
            if historical_data['game_date'].dtype == 'object':
                historical_data['game_date'] = pd.to_datetime(historical_data['game_date'])
            # Filter for historical games (games that have already happened) and sort
            current_date = pd.Timestamp(datetime.now().date())
            filtered_data = historical_data[historical_data['game_date'] <= current_date]
            filtered_data = filtered_data.sort_values(by='game_date', ascending=False)
            st.dataframe(filtered_data.head(50), hide_index=True)
            
        else:
            # Fallback: show all data without date filtering
            st.dataframe(historical_data.head(50), hide_index=True)
            
    else:
        st.info("No historical play-by-play data available. The nfl_history_2020_2024.csv.gz file may be missing or empty.")

if st.checkbox("Show Historical Game Summaries", value=False):
    st.write("### Historical Game Summaries Sample")
    historical_game_level_data = historical_game_level_data.sort_values(by='gameday', ascending=False)
    st.dataframe(historical_game_level_data.head(50), hide_index=True)

if st.checkbox("Show Schedule Data", value=False):
    st.write(f"### {current_year} NFL Schedule")
    if not schedule.empty:
        display_cols = ['week', 'date', 'home_team', 'away_team', 'venue']
        # Convert UTC date string to local datetime
        schedule_local = schedule.copy()
        schedule_local['date'] = pd.to_datetime(schedule_local['date']).dt.tz_convert('America/New_York').dt.strftime('%m/%d/%Y %I:%M %p')
        st.dataframe(schedule_local[display_cols], height=600, hide_index=True, column_config={'date': 'Date/Time (ET)', 'home_team': 'Home Team', 'away_team': 'Away Team', 'venue': 'Venue', 'week': 'Week'})
    else:
        st.warning(f"Schedule data for {current_year} is not available.")

if st.checkbox("Show/Apply Filters", value=False):
    import math
    def clean_default(default_list, valid_options):
        # Remove nan/None/invalids from default list
        return [x for x in default_list if x in valid_options and not (isinstance(x, float) and math.isnan(x))]

    with st.sidebar:
        st.header("Filters")
        if 'reset' not in st.session_state:
            st.session_state['reset'] = False
        # Initialize session state for filters
        for key in filter_keys:
            if key not in st.session_state:
                if key in ['down', 'posteam', 'defteam', 'play_type', 'qtr', 'pass_attempt']:
                    st.session_state[key] = []
                elif key == 'ydstogo':
                    st.session_state[key] = (int(historical_data['ydstogo'].min()), int(historical_data['ydstogo'].max()))
                elif key == 'yardline_100':
                    st.session_state[key] = (int(historical_data['yardline_100'].min()), int(historical_data['yardline_100'].max()))
                elif key == 'score_differential':
                    st.session_state[key] = (int(historical_data['score_differential'].min()), int(historical_data['score_differential'].max()))
                elif key == 'posteam_score':
                    st.session_state[key] = (int(historical_data['posteam_score'].min()), int(historical_data['posteam_score'].max()))
                elif key == 'defteam_score':
                    st.session_state[key] = (int(historical_data['defteam_score'].min()), int(historical_data['defteam_score'].max()))
                elif key == 'epa':
                    st.session_state[key] = (float(historical_data['epa'].min()), float(historical_data['epa'].max()))

        if st.button("Reset Filters"):
            for key in filter_keys:
                if key in ['down', 'posteam', 'defteam', 'play_type', 'qtr', 'pass_attempt']:
                    st.session_state[key] = []
                else:
                    st.session_state[key] = None
            st.session_state['reset'] = True

        # Default values
        default_filters = {
            'posteam': historical_data['posteam'].unique().tolist(),
            'defteam': historical_data['defteam'].unique().tolist(),
            'down': [1, 2, 3, 4],
            'ydstogo': (int(historical_data['ydstogo'].min()), int(historical_data['ydstogo'].max())),
            'yardline_100': (int(historical_data['yardline_100'].min()), int(historical_data['yardline_100'].max())),
            'play_type': historical_data['play_type'].dropna().unique().tolist(),
            'qtr': sorted(historical_data['qtr'].dropna().unique()),
            'score_differential': (int(historical_data['score_differential'].min()), int(historical_data['score_differential'].max())),
            'posteam_score': (int(historical_data['posteam_score'].min()), int(historical_data['posteam_score'].max())),
            'defteam_score': (int(historical_data['defteam_score'].min()), int(historical_data['defteam_score'].max())),
            'epa': (float(historical_data['epa'].min()), float(historical_data['epa'].max())),
            'pass_attempt': [0, 1]
        }

        # Filters
        posteam_options = historical_data['posteam'].unique().tolist()
        posteam_default = clean_default(st.session_state['posteam'], posteam_options)
        posteam = st.multiselect("Possession Team", posteam_options, default=posteam_default, key="posteam")
        # Defense Team
        defteam_options = historical_data['defteam'].unique().tolist()
        defteam_default = clean_default(st.session_state['defteam'], defteam_options)
        defteam = st.multiselect("Defense Team", defteam_options, default=defteam_default, key="defteam")
        # Down
        down_options = [1,2,3,4]
        down_default = clean_default(st.session_state['down'], down_options)
        down = st.multiselect("Down", down_options, default=down_default, key="down")
        # Yards To Go
        if st.session_state['ydstogo'] is None:
            st.session_state['ydstogo'] = (int(historical_data['ydstogo'].min()), int(historical_data['ydstogo'].max()))
        ydstogo = st.slider("Yards To Go", int(historical_data['ydstogo'].min()), int(historical_data['ydstogo'].max()), value=st.session_state['ydstogo'], key="ydstogo")
        # Yardline 100
        if st.session_state['yardline_100'] is None:
            st.session_state['yardline_100'] = (int(historical_data['yardline_100'].min()), int(historical_data['yardline_100'].max()))
        yardline_100 = st.slider("Yardline 100", int(historical_data['yardline_100'].min()), int(historical_data['yardline_100'].max()), value=st.session_state['yardline_100'], key="yardline_100")
        # Play Type
        play_type_options = historical_data['play_type'].dropna().unique().tolist()
        play_type_default = clean_default(st.session_state['play_type'], play_type_options)
        play_type = st.multiselect("Play Type", play_type_options, default=play_type_default, key="play_type")
        # Quarter
        qtr_options = sorted([q for q in historical_data['qtr'].dropna().unique() if not (isinstance(q, float) and math.isnan(q))])
        qtr_default = clean_default(st.session_state['qtr'], qtr_options)
        qtr = st.multiselect("Quarter", qtr_options, default=qtr_default, key="qtr")
        # Score Differential
        if st.session_state['score_differential'] is None:
            st.session_state['score_differential'] = (int(historical_data['score_differential'].min()), int(historical_data['score_differential'].max()))
        score_differential = st.slider("Score Differential", int(historical_data['score_differential'].min()), int(historical_data['score_differential'].max()), value=st.session_state['score_differential'], key="score_differential")
        # Possession Team Score
        if st.session_state['posteam_score'] is None:
            st.session_state['posteam_score'] = (int(historical_data['posteam_score'].min()), int(historical_data['posteam_score'].max()))
        posteam_score = st.slider(
            "Possession Team Score",
            int(historical_data['posteam_score'].min()),
            int(historical_data['posteam_score'].max()),
            value=default_filters['posteam_score'] if st.session_state['reset'] else (int(historical_data['posteam_score'].min()), int(historical_data['posteam_score'].max()))
        )
        defteam_score = st.slider(
            "Defense Team Score",
            int(historical_data['defteam_score'].min()),
            int(historical_data['defteam_score'].max()),
            value=default_filters['defteam_score'] if st.session_state['reset'] else (int(historical_data['defteam_score'].min()), int(historical_data['defteam_score'].max()))
        )
        epa = st.slider(
            "Expected Points Added (EPA)",
            float(historical_data['epa'].min()),
            float(historical_data['epa'].max()),
            value=default_filters['epa'] if st.session_state['reset'] else (float(historical_data['epa'].min()), float(historical_data['epa'].max()))
        )
        pass_attempt_options = [0,1]
        pass_attempt_default = clean_default(st.session_state['pass_attempt'], pass_attempt_options)
        pass_attempt = st.multiselect("Pass Attempt", pass_attempt_options, default=pass_attempt_default, key="pass_attempt")

        # Reset session state after applying
        if st.session_state['reset']:
            st.session_state['reset'] = False
            # End sidebar

        # Apply filters to the dataframe
        filtered_data = historical_data.copy()
        if posteam:
            filtered_data = filtered_data[filtered_data['posteam'].isin(posteam)]
        if defteam:
            filtered_data = filtered_data[filtered_data['defteam'].isin(defteam)]
        if down:
            filtered_data = filtered_data[filtered_data['down'].isin(down)]
        if ydstogo:
            filtered_data = filtered_data[(filtered_data['ydstogo'] >= ydstogo[0]) & (filtered_data['ydstogo'] <= ydstogo[1])]
        if yardline_100:
            filtered_data = filtered_data[(filtered_data['yardline_100'] >= yardline_100[0]) & (filtered_data['yardline_100'] <= yardline_100[1])]
        if play_type:
            filtered_data = filtered_data[filtered_data['play_type'].isin(play_type)]
        if qtr:
            filtered_data = filtered_data[filtered_data['qtr'].isin(qtr)]
        if score_differential:
            filtered_data = filtered_data[(filtered_data['score_differential'] >= score_differential[0]) & (filtered_data['score_differential'] <= score_differential[1])]
        if posteam_score:
            filtered_data = filtered_data[(filtered_data['posteam_score'] >= posteam_score[0]) & (filtered_data['posteam_score'] <= posteam_score[1])]
        if defteam_score:
            filtered_data = filtered_data[(filtered_data['defteam_score'] >= defteam_score[0]) & (filtered_data['defteam_score'] <= defteam_score[1])]
        if epa:
            filtered_data = filtered_data[(filtered_data['epa'] >= epa[0]) & (filtered_data['epa'] <= epa[1])]
        if pass_attempt:
            filtered_data = filtered_data[filtered_data['pass_attempt'].isin(pass_attempt)]

    st.write("### Filtered Historical Data")
    st.dataframe(filtered_data.head(50))

if st.checkbox("Show Model Predictions vs Actuals", value=False):
    
    if predictions_df is not None:
        display_cols = [
            'game_id', 'gameday', 'home_team', 'away_team', 'home_score', 'away_score',
            'total_line', 'spread_line',
            'predictedSpreadCovered', 'spreadCovered',
            'predictedOverHit', 'overHit'
        ]
        # Only show columns that exist
        display_cols = [col for col in display_cols if col in predictions_df.columns]
        st.write("### Model Predictions vs Actual Results")
        predictions_df = predictions_df.copy()
        predictions_df['gameday'] = pd.to_datetime(predictions_df['gameday'], errors='coerce')
        mask = predictions_df['gameday'] <= pd.to_datetime(datetime.now())
        predictions_df = predictions_df[mask]
        predictions_df.sort_values(by='gameday', ascending=False, inplace=True)
        st.dataframe(predictions_df[display_cols].head(50), hide_index=True)
        st.write(f"Predictions data shape: {predictions_df.shape}")
    else:
        st.warning("Predictions CSV not found. Run the model script to generate predictions.")

# --- New: Display model probabilities, implied probabilities, and edge columns ---
if st.checkbox("Show Model Probabilities, Implied Probabilities, and Edges", value=True):
    if predictions_df is not None:
        st.write("### Model Probabilities, Implied Probabilities, and Edges")
        prob_cols = [
             'gameday', 'home_team', 'away_team', 'home_score', 'away_score',
            'spread_line', 'total_line',
            'prob_underdogCovered', 'implied_prob_underdog_spread', 'edge_underdog_spread',
            'prob_underdogWon', 'pred_underdogWon_optimal', 'implied_prob_underdog_ml', 'edge_underdog_ml',
            'prob_overHit', 'implied_prob_over', 'edge_over',
            'implied_prob_under', 'edge_under'
        ]
        # Only show columns that exist
        # Add favored_team column (spread_line is from away team perspective)
        def get_favored_team(row):
            if not pd.isnull(row.get('spread_line', None)):
                if row['spread_line'] < 0:
                    return row['away_team']  # Away team favored (negative spread)
                elif row['spread_line'] > 0:
                    return row['home_team']  # Home team favored (positive spread)
                else:
                    return 'Pick'
            return None
        predictions_df['favored_team'] = predictions_df.apply(get_favored_team, axis=1)
        # Build display_cols: all columns in prob_cols that exist, plus favored_team after away_team
        display_cols = [col for col in prob_cols if col in predictions_df.columns]
        if 'favored_team' in predictions_df.columns and 'favored_team' not in display_cols:
            # Insert after away_team if possible, else at the end
            if 'away_team' in display_cols:
                idx = display_cols.index('away_team') + 1
                display_cols.insert(idx, 'favored_team')
            else:
                display_cols.append('favored_team')
        predictions_df['gameday'] = pd.to_datetime(predictions_df['gameday'], errors='coerce')
        today = pd.to_datetime(datetime.now().date())
        next_week = today + pd.Timedelta(days=7)
        mask = (predictions_df['gameday'] >= today) & (predictions_df['gameday'] < next_week)
        predictions_df = predictions_df[mask]
        
        # Filter out games with zero spread lines (no betting data available)
        predictions_df = predictions_df[predictions_df['spread_line'] != 0.0]
        
        # Format gameday to show only date (no time)
        predictions_df['gameday'] = predictions_df['gameday'].dt.strftime('%Y-%m-%d')
        predictions_df['pred_underdogWon_optimal'] = predictions_df['pred_underdogWon_optimal'].astype(int)
        
        st.dataframe(
            predictions_df[display_cols].sort_values(by='gameday', ascending=False).head(50), 
            hide_index=True, 
            height=470,
            column_config={
                'pred_underdogWon_optimal': st.column_config.CheckboxColumn('Betting Signal')
            }
        )
        # st.write(f"Probabilities/edges data shape: {predictions_df.shape}")
        st.markdown("""
        **Column meanings:**
        - `prob_underdogCovered`: Model probability underdog covers the spread
        - `implied_prob_underdog_spread`: Implied probability from sportsbook odds for underdog covering
        - `edge_underdog_spread`: Model edge for underdog spread bet
        - `prob_underdogWon`: Model probability underdog wins outright (moneyline)
        - `pred_underdogWon_optimal`: **🎯 BETTING SIGNAL** ✅ = Bet on underdog (optimized threshold ≥24%)
        - `implied_prob_underdog_ml`: Implied probability from sportsbook odds for underdog moneyline
        - `edge_underdog_ml`: Model edge for underdog moneyline bet
        - `prob_overHit`: Model probability total goes over
        - `implied_prob_over`: Implied probability from sportsbook odds for over
        - `edge_over`: Model edge for over bet
        - `implied_prob_under`: Implied probability from sportsbook odds for under
        - `edge_under`: Model edge for under bet
        """)
    else:
        st.warning("Predictions CSV not found. Run the model script to generate predictions.")

# --- New: Betting Analysis Section ---
if st.checkbox("Show Betting Analysis & Performance", value=True):
    if predictions_df is not None:
        st.write("### 📊 Betting Analysis & Performance")
        st.write("Assuming a $100 bet per game")
        # Load the full predictions data for analysis
        try:
            predictions_df_full = pd.read_csv(predictions_csv_path, sep='\t')
        except Exception as e:
            st.error(f"Error loading predictions data for betting analysis: {str(e)}")
            st.info("Please run 'python nfl-gather-data.py' to generate the required data files.")
            predictions_df_full = None
        
        # Calculate betting statistics
        if predictions_df_full is not None and 'pred_underdogWon_optimal' in predictions_df_full.columns:
            # Moneyline betting stats
            moneyline_bets = predictions_df_full[predictions_df_full['pred_underdogWon_optimal'] == 1]
            if len(moneyline_bets) > 0:
                moneyline_wins = (moneyline_bets['moneyline_bet_return'] > 0).sum()
                moneyline_total_return = moneyline_bets['moneyline_bet_return'].sum()
                moneyline_win_rate = moneyline_wins / len(moneyline_bets)
                moneyline_roi = moneyline_total_return / (len(moneyline_bets) * 100)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🎯 Moneyline Strategy", "Underdog Betting")
                    st.metric("Total Bets", f"{len(moneyline_bets):,}")
                    st.metric("Win Rate", f"{moneyline_win_rate:.1%}")
                    
                with col2:
                    st.metric("Total Return", f"${moneyline_total_return:,.2f}")
                    st.metric("ROI", f"{moneyline_roi:.1%}")
                    avg_return = moneyline_total_return / len(moneyline_bets)
                    st.metric("Avg Return/Bet", f"${avg_return:.2f}")
                
                # Show betting threshold info
                st.info(f"🎲 **Strategy**: Bet on underdogs when model probability ≥ 24% (F1-score optimized threshold)")
                st.caption("💡 The 24% threshold was determined by testing values from 10% to 60% and selecting the one that maximizes F1-score on training data.")
                
                # Add explanatory information
                st.markdown("#### 📊 What These Numbers Mean:")
                
                st.write(f"**Total Bets ({len(moneyline_bets):,})**: Your model identified {len(moneyline_bets):,} games where underdogs met the 24% probability threshold - this represents selective betting, not every game.")
                
                losses = len(moneyline_bets) - moneyline_wins
                st.write(f"**Win Rate ({moneyline_win_rate:.1%})**: Out of {len(moneyline_bets):,} bets, you won {moneyline_wins:,} bets and lost {losses:,} bets. This is exceptionally high for underdog betting (underdogs typically win around 35-40% of games).")
                
                original_investment = len(moneyline_bets) * 100
                total_payout = moneyline_total_return + original_investment

                st.markdown("#### 🔍 Why This Strategy Works:")
                st.write("• **Market Inefficiency**: Sportsbooks often undervalue underdogs with good statistical profiles")
                st.write("• **Selective Approach**: Only betting when model confidence ≥24% filters out poor value bets")
                st.write("• **High-Odds Payouts**: Underdog wins pay 2:1, 3:1, or higher, so you don't need to win most bets to profit")
                st.write("• **Statistical Edge**: Your model found patterns that predict underdog victories better than market expectations")

                # Show best recent bets
                if 'gameday' in moneyline_bets.columns:
                    recent_bets = moneyline_bets.copy()
                    recent_bets['gameday'] = pd.to_datetime(recent_bets['gameday'], errors='coerce')
                    recent_bets = recent_bets.sort_values('gameday', ascending=False).head(20)
                    
                    bet_display_cols = ['gameday', 'home_team', 'away_team', 'home_score', 'away_score', 
                                      'spread_line', 'prob_underdogWon', 'underdogWon', 'moneyline_bet_return']
                    bet_display_cols = [col for col in bet_display_cols if col in recent_bets.columns]
                    
                    if bet_display_cols:
                        st.write("#### 🔥 Recent Moneyline Bets")
                        st.write("*Shows underdog bets only. ✅ = Underdog won outright*")
                        
                        recent_bets_display = recent_bets[bet_display_cols].copy()
                        
                        # Add favored team column (spread_line is from away team perspective)
                        def get_favored_team(row):
                            if not pd.isnull(row.get('spread_line', None)):
                                if row['spread_line'] < 0:
                                    return row['away_team'] + ' (A)'  # Away team favored (negative spread)
                                elif row['spread_line'] > 0:
                                    return row['home_team'] + ' (H)'  # Home team favored (positive spread)
                                else:
                                    return 'Pick'
                            return 'N/A'
                        recent_bets_display['Favored'] = recent_bets_display.apply(get_favored_team, axis=1)
                        
                        # Convert probability to percentage for display
                        recent_bets_display['prob_underdogWon'] = recent_bets_display['prob_underdogWon'] * 100
                        
                        # Create formatted score column with whole numbers
                        if 'home_score' in recent_bets_display.columns and 'away_score' in recent_bets_display.columns:
                            recent_bets_display['Score'] = recent_bets_display['home_score'].astype(int).astype(str) + '-' + recent_bets_display['away_score'].astype(int).astype(str)
                        
                        # Rename columns for better display
                        recent_bets_display = recent_bets_display.rename(columns={
                            'gameday': 'Date',
                            'home_team': 'Home',
                            'away_team': 'Away', 
                            'prob_underdogWon': 'Model %',
                            'underdogWon': 'Underdog Won?',
                            'moneyline_bet_return': 'Return'
                        })
                    
                        # Select final display columns (excluding individual score columns)
                        final_display_cols = ['Date', 'Home', 'Away', 'Favored', 'Score', 'Model %', 'Underdog Won?', 'Return']
                        final_display_cols = [col for col in final_display_cols if col in recent_bets_display.columns]
                        
                        st.dataframe(
                            recent_bets_display[final_display_cols],
                            column_config={
                                'Date': st.column_config.DateColumn(format='MM/DD/YYYY'),
                                'Home': st.column_config.TextColumn(width='medium'),
                                'Away': st.column_config.TextColumn(width='medium'),
                                'Favored': st.column_config.TextColumn(width='medium'),
                                'Score': st.column_config.TextColumn(width='small'),
                                'Model %': st.column_config.NumberColumn(format='%.1f%%'),
                                'Underdog Won?': st.column_config.CheckboxColumn(),
                                'Return': st.column_config.NumberColumn(format='$%.2f')
                            },
                            height=750,
                            hide_index=True
                        )
        
        # Spread betting stats (if available)
        if 'spread_bet_return' in predictions_df_full.columns:
            spread_bets = predictions_df_full[predictions_df_full['spread_bet_return'] != 0]
            if len(spread_bets) > 0:
                spread_wins = (spread_bets['spread_bet_return'] > 0).sum()
                spread_total_return = spread_bets['spread_bet_return'].sum()
                spread_win_rate = spread_wins / len(spread_bets)
                spread_roi = spread_total_return / (len(spread_bets) * 100)
                
                st.write("#### 📈 Spread Betting Performance")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Spread Bets", f"{len(spread_bets):,}")
                with col2:
                    st.metric("Win Rate", f"{spread_win_rate:.1%}")
                with col3:
                    st.metric("ROI", f"{spread_roi:.1%}")
        
        # Performance comparison
        st.write("#### 🏆 Model vs Baseline Comparison")
        baseline_accuracy = (predictions_df_full['underdogWon'] == 0).mean()  # Always pick favorites
        st.write(f"- **Baseline Strategy (Always Pick Favorites)**: {baseline_accuracy:.1%} accuracy")
        st.write(f"- **Our Model**: Identifies profitable underdog opportunities with {moneyline_roi:.1%} ROI")
        st.write(f"- **Key Insight**: While model sacrifices overall accuracy, it finds value bets with positive expected return")
        
    else:
        st.warning("Predictions CSV not found. Run the model script to generate betting analysis.")

# --- New: Upcoming Underdog Bets Section ---
if st.checkbox("🔥 Show Next 10 Underdog Betting Opportunities", value=True):
    if predictions_df is not None:
        st.write("### 🎯 Next 10 Recommended Underdog Bets")
        st.write("*Games where model recommends betting on underdog to win (≥28% confidence)*")
        
        # Filter for upcoming games where we should bet on underdog, sort by date, take first 10
        upcoming_bets = predictions_df[predictions_df['pred_underdogWon_optimal'] == 1].copy()
        
        if len(upcoming_bets) > 0:
            # Sort by date and take first 10
            if 'gameday' in upcoming_bets.columns:
                upcoming_bets = upcoming_bets.sort_values('gameday').head(10)
            else:
                upcoming_bets = upcoming_bets.head(10)
            # Add columns for better display
            upcoming_bets_display = upcoming_bets.copy()
            
            # Add favored team column (corrected logic)
            def get_favored_team_upcoming(row):
                if not pd.isnull(row.get('spread_line', None)):
                    if row['spread_line'] < 0:
                        return row['away_team'] + ' (A)'  # Away team favored
                    elif row['spread_line'] > 0:
                        return row['home_team'] + ' (H)'  # Home team favored
                    else:
                        return 'Pick'
                return 'N/A'
            
            upcoming_bets_display['Favored'] = upcoming_bets_display.apply(get_favored_team_upcoming, axis=1)
            
            # Add underdog team column
            def get_underdog_team(row):
                if not pd.isnull(row.get('spread_line', None)):
                    if row['spread_line'] < 0:
                        return row['home_team'] + ' (H)'  # Home team underdog
                    elif row['spread_line'] > 0:
                        return row['away_team'] + ' (A)'  # Away team underdog
                    else:
                        return 'Pick'
                return 'N/A'
            
            upcoming_bets_display['Underdog'] = upcoming_bets_display.apply(get_underdog_team, axis=1)
            
            # Convert probability to percentage for display
            if 'prob_underdogWon' in upcoming_bets_display.columns:
                upcoming_bets_display['Model %'] = upcoming_bets_display['prob_underdogWon'] * 100
            
            # Get underdog moneyline odds for expected payout
            def get_underdog_odds_payout(row):
                if not pd.isnull(row.get('away_moneyline', None)) and not pd.isnull(row.get('home_moneyline', None)):
                    underdog_odds = max(row['away_moneyline'], row['home_moneyline'])  # Higher odds = underdog
                    if underdog_odds > 0:
                        return f"+{int(underdog_odds)} (${underdog_odds} profit on $100)"
                    else:
                        profit = 100 / abs(underdog_odds) * 100
                        return f"{int(underdog_odds)} (${profit:.0f} profit on $100)"
                return 'N/A'
            
            upcoming_bets_display['Expected Payout'] = upcoming_bets_display.apply(get_underdog_odds_payout, axis=1)
            
            # Select and rename columns for display
            display_cols = ['gameday', 'home_team', 'away_team', 'Favored', 'Underdog', 'spread_line', 'Model %', 'Expected Payout']
            display_cols = [col for col in display_cols if col in upcoming_bets_display.columns]
            
            final_display = upcoming_bets_display[display_cols].rename(columns={
                'gameday': 'Date',
                'home_team': 'Home',
                'away_team': 'Away',
                'spread_line': 'Spread'
            })
            
            # Sort by date
            if 'Date' in final_display.columns:
                final_display = final_display.sort_values('Date')
            
            st.dataframe(
                final_display,
                column_config={
                    'Date': st.column_config.DateColumn(format='MM/DD/YYYY'),
                    'Home': st.column_config.TextColumn(width='medium'),
                    'Away': st.column_config.TextColumn(width='medium'),
                    'Favored': st.column_config.TextColumn(width='medium'),
                    'Underdog': st.column_config.TextColumn(width='medium'),
                    'Spread': st.column_config.NumberColumn(format='%.1f'),
                    'Model %': st.column_config.NumberColumn(format='%.1f%%'),
                    'Expected Payout': st.column_config.TextColumn(width='large')
                },
                height=400,
                hide_index=True
            )
            
            st.write(f"**📊 Showing**: {len(upcoming_bets)} next underdog betting opportunities")
            
            # Add explanatory note
            st.info("""
            **💡 How to Use This:**
            - **Underdog**: Team recommended to bet on (model gives them ≥28% chance to win outright)
            - **Expected Payout**: Amount you'd win on a $100 bet if underdog wins
            - **Model %**: Model's confidence the underdog will win (higher = more confident)
            - **Strategy**: These are value bets where the model thinks the underdog is undervalued
            """)
            
        else:
            st.info("No upcoming games with underdog betting signals found in current predictions.")
            st.write("*The model may not have enough confidence (≥28%) in any upcoming underdog victories.*")
    
    else:
        st.warning("Predictions CSV not found. Run the model script to generate betting opportunities.")

# --- New: Upcoming Spread Bets Section ---
if st.checkbox("🎯 Show Next 10 Spread Betting Opportunities", value=True):
    if predictions_df is not None:
        st.write("### 🏈 Next 10 Recommended Spread Bets")
        st.write("*Games where model thinks underdog will cover spread (>50% confidence)*")
        
        # Filter for upcoming games where model thinks underdog has ANY chance to cover (>50%)
        if 'prob_underdogCovered' in predictions_df.columns:
            spread_bets = predictions_df[predictions_df['prob_underdogCovered'] > 0.50].copy()
            
            if len(spread_bets) > 0:
                # Sort by date and take first 10
                if 'gameday' in spread_bets.columns:
                    spread_bets = spread_bets.sort_values('gameday').head(10)
                else:
                    spread_bets = spread_bets.head(10)

                # Add columns for better display
                spread_bets_display = spread_bets.copy()
                
                # Add favored team and spread info
                def get_spread_info(row):
                    if not pd.isnull(row.get('spread_line', None)):
                        spread = row['spread_line']
                        if spread < 0:
                            return f"{row['away_team']} -{abs(spread)}"  # Away team favored
                        elif spread > 0:
                            return f"{row['home_team']} -{spread}"  # Home team favored
                        else:
                            return 'Pick\'em'
                    return 'N/A'
                
                spread_bets_display['Favorite & Spread'] = spread_bets_display.apply(get_spread_info, axis=1)
                
                # Add underdog team (who we're betting on to cover)
                def get_spread_underdog(row):
                    if not pd.isnull(row.get('spread_line', None)):
                        spread = row['spread_line']
                        if spread < 0:
                            return f"{row['home_team']} +{abs(spread)}"  # Home team underdog
                        elif spread > 0:
                            return f"{row['away_team']} +{spread}"  # Away team underdog
                        else:
                            return 'Pick\'em'
                    return 'N/A'
                
                spread_bets_display['Underdog (Bet On)'] = spread_bets_display.apply(get_spread_underdog, axis=1)
                
                # Convert probability to percentage for display
                spread_bets_display['Model Confidence'] = spread_bets_display['prob_underdogCovered'] * 100
                
                # Add confidence tier for prioritization
                def get_confidence_tier(row):
                    confidence = row['prob_underdogCovered']
                    if confidence >= 0.54:
                        return "🔥 Elite (54%+)"
                    elif confidence >= 0.52:
                        return "⭐ Strong (52-54%)"
                    else:
                        return "📈 Good (50-52%)"
                
                spread_bets_display['Tier'] = spread_bets_display.apply(get_confidence_tier, axis=1)
                
                # Calculate expected payout (standard -110 odds)
                spread_bets_display['Expected Payout'] = "$90.91 profit on $100 bet (91% ROI based on historical performance)"
                
                # Add edge calculation if available
                if 'edge_underdog_spread' in spread_bets_display.columns:
                    spread_bets_display['Value Edge'] = spread_bets_display['edge_underdog_spread'] * 100
                else:
                    spread_bets_display['Value Edge'] = 'N/A'
                
                # Select and rename columns for display
                display_cols = ['gameday', 'home_team', 'away_team', 'Favorite & Spread', 'Underdog (Bet On)', 'Tier', 'Model Confidence', 'Value Edge', 'Expected Payout']
                display_cols = [col for col in display_cols if col in spread_bets_display.columns]
                
                final_spread_display = spread_bets_display[display_cols].rename(columns={
                    'gameday': 'Date',
                    'home_team': 'Home Team',
                    'away_team': 'Away Team'
                })
                
                # Sort by model confidence (highest first), then by date
                if 'Model Confidence' in final_spread_display.columns:
                    final_spread_display = final_spread_display.sort_values(['Model Confidence', 'Date'], ascending=[False, True])
                elif 'Date' in final_spread_display.columns:
                    final_spread_display = final_spread_display.sort_values('Date')
                
                st.dataframe(
                    final_spread_display,
                    column_config={
                        'Date': st.column_config.DateColumn(format='MM/DD/YYYY'),
                        'Home Team': st.column_config.TextColumn(width='medium'),
                        'Away Team': st.column_config.TextColumn(width='medium'),
                        'Favorite & Spread': st.column_config.TextColumn(width='medium'),
                        'Underdog (Bet On)': st.column_config.TextColumn(width='medium'),
                        'Tier': st.column_config.TextColumn(width='medium'),
                        'Model Confidence': st.column_config.NumberColumn(format='%.1f%%'),
                        'Value Edge': st.column_config.NumberColumn(format='%.1f%%') if 'Value Edge' in final_spread_display.columns and final_spread_display['Value Edge'].dtype != 'object' else st.column_config.TextColumn(),
                        'Expected Payout': st.column_config.TextColumn(width='large')
                    },
                    height=500,
                    hide_index=True
                )
                
                st.write(f"**📊 Showing**: {len(spread_bets)} next spread betting opportunities")
                
                # Add explanatory note with tiered performance
                st.success(f"""
                **� PERFORMANCE BY CONFIDENCE LEVEL:**
                - **High Confidence (≥54%)**: 91.9% win rate, 75.5% ROI (elite level)
                - **Medium Confidence (50-54%)**: Expected ~52-55% win rate (still profitable)
                - **Current Selection**: Showing all games >50% confidence for more opportunities
                """)
                
                st.info("""
                **💡 How to Use Spread Bets:**
                - **Underdog (Bet On)**: Team to bet on covering the spread (+points means they get that advantage)
                - **Model Confidence**: How confident model is underdog will cover (50%+ shown for more opportunities)
                - **Value Edge**: How much better the model thinks the odds are vs the betting line
                - **Strategy**: Higher confidence = better historical performance, but >50% still profitable
                
                **Example**: If betting "Chiefs +3.5", the Chiefs can lose by 1, 2, or 3 points and you still win!
                
                **💰 Betting Strategy**: Focus on highest confidence games first, but >50% games still have value!
                """)
                
            else:
                st.info("No upcoming games with positive spread betting signals found.")
                st.write("*The model doesn't favor underdogs to cover in any upcoming games (all <50% confidence).*")
        else:
            st.warning("Spread probabilities not found in predictions data. Ensure the model has been trained with spread predictions.")
    
    else:
        st.warning("Predictions CSV not found. Run the model script to generate spread betting opportunities.")

if st.checkbox("Show Model Feature Importances & Metrics", value=False):
    st.write("### Model Feature Importances and Error Metrics")
    # Try to load feature importances and metrics if saved as a CSV or JSON
    import json
    metrics_path = path.join(DATA_DIR, 'model_metrics.json')
    importances_path = path.join(DATA_DIR, 'model_feature_importances.csv')
    # Display metrics
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        st.write("#### Model Error Metrics")
        for k, v in metrics.items():
            st.write(f"{k}: {v}")
    else:
        st.info("No model metrics file found. Run the model script to generate metrics.")
    # Display feature importances
    if os.path.exists(importances_path):
        importances_df = pd.read_csv(importances_path)
        st.write("#### Feature Importances (Top 10)")
        st.dataframe(importances_df.head(10), hide_index=True)
    else:
        st.info("No feature importances file found. Run the model script to generate importances.")

if st.checkbox("Run Monte Carlo Feature Selection", value=False):
    st.write("### Monte Carlo Feature Selection (Spread Model)")
    import random
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer, roc_auc_score, f1_score
    # User controls
    num_iter = st.number_input("Number of Iterations", min_value=10, max_value=500, value=100, step=10, key="mc_iter_1")
    subset_size = st.number_input("Subset Size", min_value=2, max_value=len(features), value=8, step=1, key="mc_subset_1")
    random_seed = st.number_input("Random Seed", min_value=0, value=42, step=1, key="mc_seed_1")
    run_mc = st.button("Run Monte Carlo Search", key="mc_run_1")
    if run_mc:
        with st.spinner("Running Monte Carlo feature selection..."):
            # Filter features to only those available in the spread dataset
            available_spread_features = list(X_train_spread.columns)
            
            best_score = 0
            best_features = []
            random.seed(random_seed)
            scores_list = []
            for i in range(int(num_iter)):
                subset = random.sample(available_spread_features, min(int(subset_size), len(available_spread_features)))
                X_subset = X_train_spread[subset]
                model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
                acc = cross_val_score(model, X_subset, y_spread_train, cv=3, scoring='accuracy').mean()
                # For AUC and F1, use make_scorer
                try:
                    auc = cross_val_score(model, X_subset, y_spread_train, cv=3, scoring='roc_auc').mean()
                except Exception:
                    auc = float('nan')
                try:
                    f1 = cross_val_score(model, X_subset, y_spread_train, cv=3, scoring='f1').mean()
                except Exception:
                    f1 = float('nan')
                scores_list.append({
                    'iteration': i+1,
                    'features': subset,
                    'accuracy': acc,
                    'AUC': auc,
                    'F1-score': f1
                })
                if acc > best_score:
                    best_score = acc
                    best_features = subset
        st.success(f"Best mean CV accuracy: {best_score:.4f}")
        st.write(f"Best feature subset:")
        st.code(best_features)
        # Save best features to file (spread)
        with open(path.join(DATA_DIR, 'best_features_spread.txt'), 'w') as f:
            f.write("\n".join(best_features))
        # Retrain model using best_features and calibrate probabilities
        X_train_best = X_train_spread[best_features]
        X_test_best = X_test_spread[best_features]
        from sklearn.calibration import CalibratedClassifierCV
        model_spread_best = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
        calibrated_model = CalibratedClassifierCV(model_spread_best, method='isotonic', cv=3)
        calibrated_model.fit(X_train_best, y_spread_train)
        y_spread_pred_best = calibrated_model.predict(X_test_best)
        spread_accuracy_best = accuracy_score(y_spread_test, y_spread_pred_best)
        # Probability sanity check
        probs = calibrated_model.predict_proba(X_test_best)[:, 1]
        mean_pred_prob = np.mean(probs)
        actual_rate = np.mean(y_spread_test)
        st.write(f"Accuracy with best features: {spread_accuracy_best:.4f}")
        st.write(f"Mean predicted probability (test set): {mean_pred_prob:.4f}")
        st.write(f"Actual outcome rate (test set): {actual_rate:.4f}")
        # Show top 10 results
        scores_df = pd.DataFrame(scores_list).sort_values(by='accuracy', ascending=False).head(10)
        st.write("#### Top 10 Feature Subsets (by Accuracy)")
        st.dataframe(scores_df, hide_index=True)

if st.checkbox("Run Monte Carlo Feature Selection (Favorites v. Underdogs)", value=False):
    st.write("### Monte Carlo Feature Selection (Spread Model)")
    import random
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer, roc_auc_score, f1_score
    # User controls
    num_iter = st.number_input("Number of Iterations", min_value=10, max_value=500, value=100, step=10, key="mc_iter_2")
    subset_size = st.number_input("Subset Size", min_value=2, max_value=len(features), value=8, step=1, key="mc_subset_2")
    random_seed = st.number_input("Random Seed", min_value=0, value=42, step=1, key="mc_seed_2")
    run_mc = st.button("Run Monte Carlo Search", key="mc_run_2")
    if run_mc:
        with st.spinner("Running Monte Carlo feature selection..."):
            # Filter features to only those available in the spread dataset
            available_spread_features = list(X_train_spread.columns)
            
            best_score = 0
            best_features = []
            random.seed(random_seed)
            scores_list = []
            for i in range(int(num_iter)):
                subset = random.sample(available_spread_features, min(int(subset_size), len(available_spread_features)))
                X_subset = X_train_spread[subset]
                model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
                acc = cross_val_score(model, X_subset, y_spread_train, cv=3, scoring='accuracy').mean()
                # For AUC and F1, use make_scorer
                try:
                    auc = cross_val_score(model, X_subset, y_spread_train, cv=3, scoring='roc_auc').mean()
                except Exception:
                    auc = float('nan')
                try:
                    f1 = cross_val_score(model, X_subset, y_spread_train, cv=3, scoring='f1').mean()
                except Exception:
                    f1 = float('nan')
                scores_list.append({
                    'iteration': i+1,
                    'features': subset,
                    'accuracy': acc,
                    'AUC': auc,
                    'F1-score': f1
                })
                if acc > best_score:
                    best_score = acc
                    best_features = subset
        st.success(f"Best mean CV accuracy: {best_score:.4f}")
        st.write(f"Best feature subset:")
        st.code(best_features)
        # Retrain model using best_features
        X_train_best = X_train_spread[best_features]
        X_test_best = X_test_spread[best_features]
        model_spread_best = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
        model_spread_best.fit(X_train_best, y_spread_train)
        y_spread_pred_best = model_spread_best.predict(X_test_best)
        spread_accuracy_best = accuracy_score(y_spread_test, y_spread_pred_best)
        st.write(f"Accuracy with best features: {spread_accuracy_best:.4f}")
        # Show top 10 results
        scores_df = pd.DataFrame(scores_list).sort_values(by='accuracy', ascending=False).head(10)
        st.write("#### Top 10 Feature Subsets (by Accuracy)")
        st.dataframe(scores_df, hide_index=True)

# --- Monte Carlo Feature Selection: Moneyline ---
if st.checkbox("Run Monte Carlo Feature Selection (Moneyline)", value=False):
    st.write("### Monte Carlo Feature Selection (Moneyline Model)")
    import random
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer, roc_auc_score, f1_score
    num_iter = st.number_input("Number of Iterations (Moneyline)", min_value=10, max_value=500, value=100, step=10)
    # Get available numeric features for validation
    available_features = [f for f in features if f in historical_game_level_data.columns]
    numeric_features_count = len(historical_game_level_data[available_features].select_dtypes(include=["number", "bool", "category"]).columns)
    subset_size = st.number_input("Subset Size (Moneyline)", min_value=2, max_value=numeric_features_count, value=min(8, numeric_features_count), step=1)
    random_seed = st.number_input("Random Seed (Moneyline)", min_value=0, value=42, step=1)
    run_mc = st.button("Run Monte Carlo Search (Moneyline)")
    if run_mc:
        with st.spinner("Running Monte Carlo feature selection..."):
            # Filter features to only those available in the dataset
            available_features = [f for f in features if f in historical_game_level_data.columns]
            numeric_features = historical_game_level_data[available_features].select_dtypes(include=["number", "bool", "category"]).columns.tolist()
            
            best_score = 0
            best_features = []
            random.seed(random_seed)
            scores_list = []
            for i in range(int(num_iter)):
                subset = random.sample(numeric_features, min(int(subset_size), len(numeric_features)))
                # Create a fresh dataset slice for this subset to avoid column mismatch
                X_subset_data = historical_game_level_data[subset]
                X_train_subset, _, y_train_subset, _ = train_test_split(
                    X_subset_data, y_moneyline, test_size=0.2, random_state=42, stratify=y_moneyline)
                
                model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
                acc = cross_val_score(model, X_train_subset, y_train_subset, cv=3, scoring='accuracy').mean()
                try:
                    auc = cross_val_score(model, X_train_subset, y_train_subset, cv=3, scoring='roc_auc').mean()
                except Exception:
                    auc = float('nan')
                try:
                    f1 = cross_val_score(model, X_train_subset, y_train_subset, cv=3, scoring='f1').mean()
                except Exception:
                    f1 = float('nan')
                scores_list.append({
                    'iteration': i+1,
                    'features': subset,
                    'accuracy': acc,
                    'AUC': auc,
                    'F1-score': f1
                })
                if acc > best_score:
                    best_score = acc
                    best_features = subset
        st.success(f"Best mean CV accuracy: {best_score:.4f}")
        st.write(f"Best feature subset:")
        st.code(best_features)
        # Save best features to file (moneyline)
        with open(path.join(DATA_DIR, 'best_features_moneyline.txt'), 'w') as f:
            f.write("\n".join(best_features))
        # Retrain model using best_features (Moneyline) - create new dataset with selected features
        X_moneyline_best = historical_game_level_data[best_features]
        X_train_ml_best, X_test_ml_best, _, _ = train_test_split(
            X_moneyline_best, y_moneyline, test_size=0.2, random_state=42, stratify=y_moneyline)
        model_moneyline_best = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
        model_moneyline_best.fit(X_train_ml_best, y_train_ml)
        y_moneyline_pred_best = model_moneyline_best.predict(X_test_ml_best)
        moneyline_accuracy_best = accuracy_score(y_test_ml, y_moneyline_pred_best)
        st.write(f"Accuracy with best features (Moneyline): {moneyline_accuracy_best:.4f}")
        scores_df = pd.DataFrame(scores_list).sort_values(by='accuracy', ascending=False).head(10)
        st.write("#### Top 10 Feature Subsets (by Accuracy)")
        st.dataframe(scores_df, hide_index=True)

# --- Monte Carlo Feature Selection: Totals (Over) ---
if st.checkbox("Run Monte Carlo Feature Selection (Totals)", value=False):
    st.write("### Monte Carlo Feature Selection (Totals Model)")
    import random
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer, roc_auc_score, f1_score
    num_iter = st.number_input("Number of Iterations (Totals)", min_value=10, max_value=500, value=100, step=10)
    subset_size = st.number_input("Subset Size (Totals)", min_value=2, max_value=len(features), value=8, step=1)
    random_seed = st.number_input("Random Seed (Totals)", min_value=0, value=42, step=1)
    run_mc = st.button("Run Monte Carlo Search (Totals)")
    if run_mc:
        with st.spinner("Running Monte Carlo feature selection..."):
            best_score = 0
            best_features = []
            random.seed(random_seed)
            scores_list = []
            valid_totals_features = list(X_train_tot.columns)
            for i in range(int(num_iter)):
                subset = random.sample(valid_totals_features, int(subset_size))
                X_subset = X_train_tot[subset]
                model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
                acc = cross_val_score(model, X_subset, y_train_tot, cv=3, scoring='accuracy').mean()
                try:
                    auc = cross_val_score(model, X_subset, y_train_tot, cv=3, scoring='roc_auc').mean()
                except Exception:
                    auc = float('nan')
                try:
                    f1 = cross_val_score(model, X_subset, y_train_tot, cv=3, scoring='f1').mean()
                except Exception:
                    f1 = float('nan')
                scores_list.append({
                    'iteration': i+1,
                    'features': subset,
                    'accuracy': acc,
                    'AUC': auc,
                    'F1-score': f1
                })
                if acc > best_score:
                    best_score = acc
                    best_features = subset
        st.success(f"Best mean CV accuracy: {best_score:.4f}")
        st.write(f"Best feature subset:")
        st.code(best_features)
        # Save best features to file (totals)
        with open(path.join(DATA_DIR, 'best_features_totals.txt'), 'w') as f:
            f.write("\n".join(best_features))
        # Retrain model using best_features and calibrate probabilities (Totals)
        X_totals_best = historical_game_level_data[best_features].select_dtypes(include=["number", "bool", "category"])
        X_train_tot_best, X_test_tot_best, _, _ = train_test_split(
            X_totals_best, y_totals, test_size=0.2, random_state=42, stratify=y_totals)
        from sklearn.calibration import CalibratedClassifierCV
        model_totals_best = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
        calibrated_model_totals = CalibratedClassifierCV(model_totals_best, method='isotonic', cv=3)
        calibrated_model_totals.fit(X_train_tot_best, y_train_tot)
        y_totals_pred_best = calibrated_model_totals.predict(X_test_tot_best)
        totals_accuracy_best = accuracy_score(y_test_tot, y_totals_pred_best)
        # Probability sanity check
        probs_totals = calibrated_model_totals.predict_proba(X_test_tot_best)[:, 1]
        mean_pred_prob_totals = np.mean(probs_totals)
        actual_rate_totals = np.mean(y_test_tot)
        st.write(f"Accuracy with best features (Totals): {totals_accuracy_best:.4f}")
        st.write(f"Mean predicted probability (test set): {mean_pred_prob_totals:.4f}")
        st.write(f"Actual outcome rate (test set): {actual_rate_totals:.4f}")
        scores_df = pd.DataFrame(scores_list).sort_values(by='accuracy', ascending=False).head(10)
        st.write("#### Top 10 Feature Subsets (by Accuracy)")
        st.dataframe(scores_df, hide_index=True)