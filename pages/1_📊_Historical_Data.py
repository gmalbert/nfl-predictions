import streamlit as st
import pandas as pd
import os
from os import path
from datetime import datetime

st.set_page_config(
    page_title="Historical Data - NFL Predictor",
    page_icon="📊",
    layout="wide"
)

DATA_DIR = 'data_files/'

# Load historical play-by-play data with optimizations
@st.cache_data
def load_historical_data():
    file_path = path.join(DATA_DIR, 'nfl_history_2020_2024.csv.gz')
    if os.path.exists(file_path):
        # Use chunksize and only load what we need to reduce memory usage
        data = pd.read_csv(
            file_path, 
            compression='gzip', 
            sep='\t', 
            low_memory=False,
            dtype={
                'down': 'float32',
                'qtr': 'float32', 
                'ydstogo': 'float32',
                'yardline_100': 'float32',
                'score_differential': 'float32',
                'posteam_score': 'float32',
                'defteam_score': 'float32',
                'epa': 'float32',
                'pass_attempt': 'Int8',
                'rush_attempt': 'Int8'
            }
        )
        return data
    else:
        return pd.DataFrame()

st.title("📊 Historical Data & Filters")
st.write("Historical play-by-play data and game summaries")

# Link back to predictions page
if st.button("🏈 Back to Predictions", type="secondary"):
    st.switch_page("predictions.py")

# Load data
with st.spinner("Loading historical data..."):
    historical_data = load_historical_data()

if historical_data.empty:
    st.error("Historical play-by-play data file not found. The nfl_history_2020_2024.csv.gz file may be missing or empty.")
    st.stop()

current_year = datetime.now().year

st.write("### Historical Play-by-Play Data Sample")
st.write(f"*Showing data from {current_year-4} to {current_year-1} seasons*")

if 'game_date' in historical_data.columns:
    # Don't copy - work with original dataframe to save memory
    # Only convert game_date once if needed
    if historical_data['game_date'].dtype == 'object':
        historical_data['game_date'] = pd.to_datetime(historical_data['game_date'], errors='coerce')
    
    current_date = pd.Timestamp(datetime.now().date())
    
    # Create a view instead of copy for initial filtering
    filtered_data = historical_data[historical_data['game_date'] <= current_date]
    
    # Sort only when needed (not on the full dataset)
    filtered_data = filtered_data.sort_values(by='game_date', ascending=False)

    # Select key play-by-play columns for display
    display_cols = [
        'game_date', 'week', 'season', 'home_team', 'away_team', 'posteam', 'defteam',
        'game_seconds_remaining', 'qtr', 'down', 'ydstogo', 'yardline_100',
        'play_type', 'yards_gained', 'desc', 'epa', 'wp',
        'posteam_score', 'defteam_score', 'score_differential',
        'pass_attempt', 'rush_attempt', 'complete_pass', 'interception', 'fumble_lost',
        'td_prob', 'touchdown', 'field_goal_result'
    ]

    # Only use columns that exist
    display_cols = [col for col in display_cols if col in filtered_data.columns]
   
    # Initialize reset flag
    if 'reset' not in st.session_state:
        st.session_state['reset'] = False
    
    with st.sidebar:
        st.header("🔍 Filters")
        
        # Reset button - sets reset flag
        if st.button("🔄 Reset All Filters"):
            st.session_state['reset'] = True
            st.rerun()
        
        st.divider()
        
        # Default filter values
        default_filters = {
            'posteam': [],
            'defteam': [],
            'down': [],
            'qtr': [],
            'play_type': [],
            'pass_only': False
        }
        
        # Team Filters
        st.subheader("Teams")
        posteam_options = sorted(filtered_data['posteam'].dropna().unique().tolist())
        selected_offense = st.multiselect(
            "Offense Team", 
            posteam_options,
            default=default_filters['posteam'] if st.session_state['reset'] else st.session_state.get('posteam', []),
            key='posteam'
        )
        
        defteam_options = sorted(filtered_data['defteam'].dropna().unique().tolist())
        selected_defense = st.multiselect(
            "Defense Team", 
            defteam_options,
            default=default_filters['defteam'] if st.session_state['reset'] else st.session_state.get('defteam', []),
            key='defteam'
        )
        
        st.divider()
        
        # Game Context Filters
        st.subheader("Game Context")
        down_options = sorted([int(d) for d in filtered_data['down'].dropna().unique() if d > 0])
        selected_downs = st.multiselect(
            "Down", 
            down_options,
            default=default_filters['down'] if st.session_state['reset'] else st.session_state.get('down', []),
            key='down'
        )
        
        qtr_options = sorted([q for q in filtered_data['qtr'].dropna().unique() if q > 0])
        selected_qtrs = st.multiselect(
            "Quarter", 
            qtr_options,
            default=default_filters['qtr'] if st.session_state['reset'] else st.session_state.get('qtr', []),
            key='qtr'
        )
        
        st.divider()
        
        # Play Type Filter
        st.subheader("Play Type")
        play_type_options = sorted(filtered_data['play_type'].dropna().unique().tolist())
        selected_play_types = st.multiselect(
            "Play Type", 
            play_type_options,
            default=default_filters['play_type'] if st.session_state['reset'] else st.session_state.get('play_type', []),
            key='play_type'
        )
        
        pass_only = st.checkbox(
            "Pass Attempts Only",
            value=default_filters['pass_only'] if st.session_state['reset'] else st.session_state.get('pass_only', False),
            key='pass_only'
        )
        
        # Clear reset flag after widgets are defined
        if st.session_state['reset']:
            st.session_state['reset'] = False
        
        st.divider()
        
        # Numeric Filters with Sliders
        st.subheader("Field Position & Yards")
        
        if 'ydstogo' in filtered_data.columns:
            ydstogo_min = int(filtered_data['ydstogo'].min())
            ydstogo_max = int(filtered_data['ydstogo'].max())
            ydstogo_range = st.slider("Yards To Go", ydstogo_min, ydstogo_max, (ydstogo_min, ydstogo_max))
        else:
            ydstogo_range = None
        
        if 'yardline_100' in filtered_data.columns:
            yardline_min = int(filtered_data['yardline_100'].min())
            yardline_max = int(filtered_data['yardline_100'].max())
            yardline_range = st.slider("Yardline (distance from opponent endzone)", yardline_min, yardline_max, (yardline_min, yardline_max))
        else:
            yardline_range = None
        
        st.divider()
        
        # Score Filters
        st.subheader("Score")
        
        if 'score_differential' in filtered_data.columns:
            score_diff_min = int(filtered_data['score_differential'].min())
            score_diff_max = int(filtered_data['score_differential'].max())
            score_diff_range = st.slider("Score Differential", score_diff_min, score_diff_max, (score_diff_min, score_diff_max))
        else:
            score_diff_range = None
        
        if 'posteam_score' in filtered_data.columns:
            posteam_score_min = int(filtered_data['posteam_score'].min())
            posteam_score_max = int(filtered_data['posteam_score'].max())
            posteam_score_range = st.slider("Offense Team Score", posteam_score_min, posteam_score_max, (posteam_score_min, posteam_score_max))
        else:
            posteam_score_range = None
        
        if 'defteam_score' in filtered_data.columns:
            defteam_score_min = int(filtered_data['defteam_score'].min())
            defteam_score_max = int(filtered_data['defteam_score'].max())
            defteam_score_range = st.slider("Defense Team Score", defteam_score_min, defteam_score_max, (defteam_score_min, defteam_score_max))
        else:
            defteam_score_range = None
        
        st.divider()
        
        # Advanced Metrics
        st.subheader("Advanced Metrics")
        
        if 'epa' in filtered_data.columns:
            epa_min = float(filtered_data['epa'].min())
            epa_max = float(filtered_data['epa'].max())
            epa_range = st.slider("Expected Points Added (EPA)", epa_min, epa_max, (epa_min, epa_max))
        else:
            epa_range = None
    
    # Apply filters to the data
    if selected_offense:
        filtered_data = filtered_data[filtered_data['posteam'].isin(selected_offense)]
    
    if selected_defense:
        filtered_data = filtered_data[filtered_data['defteam'].isin(selected_defense)]
    
    if selected_downs:
        filtered_data = filtered_data[filtered_data['down'].isin(selected_downs)]
    
    if selected_qtrs:
        filtered_data = filtered_data[filtered_data['qtr'].isin(selected_qtrs)]
    
    if selected_play_types:
        filtered_data = filtered_data[filtered_data['play_type'].isin(selected_play_types)]
    
    if pass_only:
        filtered_data = filtered_data[filtered_data['pass_attempt'] == 1]
    
    if ydstogo_range:
        filtered_data = filtered_data[(filtered_data['ydstogo'] >= ydstogo_range[0]) & (filtered_data['ydstogo'] <= ydstogo_range[1])]
    
    if yardline_range:
        filtered_data = filtered_data[(filtered_data['yardline_100'] >= yardline_range[0]) & (filtered_data['yardline_100'] <= yardline_range[1])]
    
    if score_diff_range:
        filtered_data = filtered_data[(filtered_data['score_differential'] >= score_diff_range[0]) & (filtered_data['score_differential'] <= score_diff_range[1])]
    
    if posteam_score_range:
        filtered_data = filtered_data[(filtered_data['posteam_score'] >= posteam_score_range[0]) & (filtered_data['posteam_score'] <= posteam_score_range[1])]
    
    if defteam_score_range:
        filtered_data = filtered_data[(filtered_data['defteam_score'] >= defteam_score_range[0]) & (filtered_data['defteam_score'] <= defteam_score_range[1])]
    
    if epa_range:
        filtered_data = filtered_data[(filtered_data['epa'] >= epa_range[0]) & (filtered_data['epa'] <= epa_range[1])]

    # Add pagination
    rows_per_page = st.selectbox("Rows per page", [50, 100, 250, 500], index=0)
    total_rows = len(filtered_data)
    total_pages = (total_rows - 1) // rows_per_page + 1
    
    # Warn if showing too many rows
    if rows_per_page > 100 and total_rows > 10000:
        st.warning(f"⚠️ Displaying {rows_per_page} rows from a large dataset ({total_rows:,} total). Consider using filters to narrow down results for better performance.")
    
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    start_idx = (page - 1) * rows_per_page
    end_idx = min(start_idx + rows_per_page, total_rows)
    
    st.write(f"Showing rows {start_idx + 1:,} to {end_idx:,} of {total_rows:,}")
    
    st.dataframe(
        filtered_data[display_cols].iloc[start_idx:end_idx],
        hide_index=True,
        height=600,
        column_config={
            'game_date': st.column_config.DateColumn('Game Date', format='MM/DD/YYYY'),
            'week': st.column_config.NumberColumn('Week', format='%d'),
            'season': st.column_config.NumberColumn('Season', format='%d'),
            'home_team': st.column_config.TextColumn('Home Team', width='small'),
            'away_team': st.column_config.TextColumn('Away Team', width='small'),
            'posteam': st.column_config.TextColumn('Offense', width='small'),
            'defteam': st.column_config.TextColumn('Defense', width='small'),
            'game_seconds_remaining': st.column_config.NumberColumn('Time Left (s)', format='%d'),
            'qtr': st.column_config.NumberColumn('Qtr', format='%d'),
            'down': st.column_config.NumberColumn('Down', format='%d'),
            'ydstogo': st.column_config.NumberColumn('To Go', format='%d'),
            'yardline_100': st.column_config.NumberColumn('Yardline', format='%d', help='Distance from opponent endzone'),
            'play_type': st.column_config.TextColumn('Play Type', width='small'),
            'yards_gained': st.column_config.NumberColumn('Yards', format='%d'),
            'desc': st.column_config.TextColumn('Play Description', width='large'),
            'epa': st.column_config.NumberColumn('EPA', format='%.2f', help='Expected Points Added'),
            'wp': st.column_config.NumberColumn('Win Prob', format='%.1f%%', help='Win probability after play'),
            'posteam_score': st.column_config.NumberColumn('Off Score', format='%d'),
            'defteam_score': st.column_config.NumberColumn('Def Score', format='%d'),
            'score_differential': st.column_config.NumberColumn('Score Diff', format='%d'),
            'pass_attempt': st.column_config.CheckboxColumn('Pass?'),
            'rush_attempt': st.column_config.CheckboxColumn('Rush?'),
            'complete_pass': st.column_config.CheckboxColumn('Complete?'),
            'interception': st.column_config.CheckboxColumn('INT?'),
            'fumble_lost': st.column_config.CheckboxColumn('Fumble?'),
            'td_prob': st.column_config.NumberColumn('TD Prob', format='%.1f%%'),
            'touchdown': st.column_config.CheckboxColumn('TD?'),
        'field_goal_result': st.column_config.TextColumn('FG Result', width='small')
        }
    )

else:
    # Fallback: show all data without date filtering
    st.dataframe(historical_data.head(50), hide_index=True)