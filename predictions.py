import streamlit as st
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

# Set page config at the top
st.set_page_config(

    page_title="NFL Play Outcome Predictor",
    page_icon="data_files/favicon.ico"  # or use an emoji, e.g. "🏈"
)


# Load full NFL schedule from ESPN API (all regular season weeks) and save to CSV
current_year = datetime.now().year
historical_game_level_data = pd.read_csv(path.join(DATA_DIR, 'nfl_games_historical.csv'), sep='\t')

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
    historical_data = pd.read_csv(path.join(DATA_DIR, 'nfl_history_2020_2024.csv.gz'), compression='gzip', sep='\t', low_memory=False)
    return historical_data

historical_data = load_data()

@st.cache_data
def load_schedule():
    schedule_data = pd.read_csv(path.join(DATA_DIR, f'espn_schedule_{current_year}.csv'), low_memory=False)
    return schedule_data

schedule = load_schedule()

# Display NFL logo at the top
logo_path = os.path.join(DATA_DIR, "gridiron-oracle.png")
if os.path.exists(logo_path):
    st.image(logo_path, width=300)

st.title('NFL Play Outcome Predictor')

# Sidebar filters

filter_keys = [
    'posteam', 'defteam', 'down', 'ydstogo', 'yardline_100', 'play_type', 'qtr',
    'score_differential', 'posteam_score', 'defteam_score', 'epa', 'pass_attempt'
]

if st.checkbox("Show Raw Historical Play By Play Data", value=False):
    st.write("### Historical Data Sample")
    st.dataframe(historical_data.head(50), width=800, hide_index=True)
    st.write(f"Data shape: {historical_data.shape}")

if st.checkbox("Show Historical Game Summaries", value=False):
    st.write("### Historical Game Summaries Sample")
    st.dataframe(historical_game_level_data.head(50), width=800, hide_index=True)
    st.write(f"Game summaries data shape: {historical_game_level_data.shape}")

if st.checkbox("Show Schedule Data", value=False):
    st.write(f"### {current_year} NFL Schedule")
    if not schedule.empty:
        display_cols = ['week', 'date', 'home_team', 'away_team', 'venue']
        # Convert UTC date string to local datetime
        schedule_local = schedule.copy()
        schedule_local['date'] = pd.to_datetime(schedule_local['date']).dt.tz_convert('America/New_York').dt.strftime('%m/%d/%Y %I:%M %p')
        st.dataframe(schedule_local[display_cols], width=800, height=600, hide_index=True, column_config={'date': 'Date/Time (ET)', 'home_team': 'Home Team', 'away_team': 'Away Team', 'venue': 'Venue', 'week': 'Week'})
        st.write(f"Schedule data shape: {schedule.shape}")
    else:
        st.warning(f"Schedule data for {current_year} is not available.")



if st.checkbox("Show Model Predictions vs Actuals", value=False):
    if predictions_df is not None:
        display_cols = [
            'game_id', 'date', 'home_team', 'away_team', 'home_score', 'away_score',
            'total_line', 'spread_line',
            'predictedSpreadCovered', 'spreadCovered',
            'predictedOverHit', 'overHit'
        ]
        # Only show columns that exist
        display_cols = [col for col in display_cols if col in predictions_df.columns]
        st.write("### Model Predictions vs Actual Results")
        st.dataframe(predictions_df[display_cols].head(50), width=1000, hide_index=True)
        st.write(f"Predictions data shape: {predictions_df.shape}")
    else:
        st.warning("Predictions CSV not found. Run the model script to generate predictions.")

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
    st.dataframe(filtered_data.head(50), width=800)
    st.write(f"Filtered data shape: {filtered_data.shape}")