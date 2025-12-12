import streamlit as st
import pandas as pd
import os
import re
from os import path
from datetime import datetime

st.set_page_config(
    page_title="Historical Data - NFL Predictor",
    page_icon="📊",
    layout="wide"
)

DATA_DIR = 'data_files/'

def get_dataframe_height(df, row_height=35, header_height=38, padding=2, max_height=600):
    """
    Calculate the optimal height for a Streamlit dataframe based on number of rows.
    
    Args:
        df (pd.DataFrame): The dataframe to display
        row_height (int): Height per row in pixels. Default: 35
        header_height (int): Height of header row in pixels. Default: 38
        padding (int): Extra padding in pixels. Default: 2
        max_height (int): Maximum height cap in pixels. Default: 600 (None for no limit)
    
    Returns:
        int: Calculated height in pixels
    
    Example:
        height = get_dataframe_height(my_df)
        st.dataframe(my_df, height=height)
    """
    num_rows = len(df)
    calculated_height = (num_rows * row_height) + header_height + padding
    
    if max_height is not None:
        return min(calculated_height, max_height)
    return calculated_height

def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV bytes for Streamlit download_button."""
    try:
        return df.to_csv(index=False).encode('utf-8')
    except Exception:
        # Fallback: coerce to strings then export
        df2 = df.copy()
        for c in df2.columns:
            try:
                df2[c] = df2[c].astype(str)
            except Exception:
                pass
        return df2.to_csv(index=False).encode('utf-8')

# Load historical play-by-play data with optimizations
@st.cache_data(show_spinner=False)
def load_historical_data():
    file_path = path.join(DATA_DIR, 'nfl_play_by_play_historical.csv.gz')
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
                'rush_attempt': 'Int8',
                'complete_pass': 'Int8',
                'interception': 'Int8',
                'fumble_lost': 'Int8',
                'touchdown': 'Int8'
            }
        )
        return data
    else:
        return pd.DataFrame()


def _find_most_recent_pbp_info(candidates=None, chunksize=100000):
    """Scan candidate PBP files in DATA_DIR to detect the most recent season and game_date.
    Returns a dict: {'file': filename, 'season': int or None, 'game_date': Timestamp or None}
    """
    if candidates is None:
        candidates = ['nfl_play_by_play_historical.csv.gz']

    best = {'file': None, 'season': None, 'game_date': None}

    for fname in candidates:
        p = path.join(DATA_DIR, fname)
        if not os.path.exists(p):
            continue
        compression = 'gzip' if fname.endswith('.gz') else None
        try:
            for chunk in pd.read_csv(p, compression=compression, sep='\t', usecols=['season', 'game_date'], chunksize=chunksize, low_memory=True):
                # season
                try:
                    if 'season' in chunk.columns:
                        s = pd.to_numeric(chunk['season'], errors='coerce').dropna()
                        if not s.empty:
                            maxs = int(s.max())
                            if best['season'] is None or maxs > best['season']:
                                best['season'] = maxs
                                best['file'] = fname
                except Exception:
                    pass

                # game_date
                try:
                    if 'game_date' in chunk.columns:
                        g = pd.to_datetime(chunk['game_date'], errors='coerce')
                        # Only consider game dates that are today or earlier (compare by calendar date)
                        try:
                            today_date = datetime.now().date()
                            g = g[g.dt.date <= today_date]
                        except Exception:
                            # Fallback: compare timestamps if .dt is unavailable
                            today = pd.Timestamp(datetime.now().date())
                            g = g[g <= today]

                        if not g.empty and not g.isna().all():
                            maxd = g.max()
                            if best['game_date'] is None or maxd > best['game_date']:
                                best['game_date'] = maxd
                                best['file'] = fname
                except Exception:
                    pass

                # If we have both season and game_date and they look recent, stop early
                if best['season'] is not None and best['game_date'] is not None:
                    # small heuristic: if game_date year matches season, likely recent enough
                    try:
                        if best['game_date'].year == best['season']:
                            return best
                    except Exception:
                        return best
        except Exception:
            continue

    return best


def _collect_recent_pbp_rows(filename, season=None, max_rows=450, chunksize=5000):
    """Collect up to max_rows recent rows from filename for the given season (if provided).
    Returns a DataFrame (possibly empty).
    """
    p = path.join(DATA_DIR, filename)
    if not os.path.exists(p):
        return pd.DataFrame()
    compression = 'gzip' if filename.endswith('.gz') else None
    collected = []
    try:
        for chunk in pd.read_csv(p, compression=compression, sep='\t', chunksize=chunksize, low_memory=True):
            # normalize season if present
            if season is not None and 'season' in chunk.columns:
                try:
                    chunk['season'] = pd.to_numeric(chunk['season'], errors='coerce')
                except Exception:
                    pass
                subset = chunk[chunk['season'] == season]
            else:
                subset = chunk

                if not subset.empty:
                    # Filter out any future-dated games: compare by calendar date to avoid timezone issues
                    if 'game_date' in subset.columns:
                        try:
                            subset['game_date'] = pd.to_datetime(subset['game_date'], errors='coerce')
                            today_date = datetime.now().date()
                            subset = subset[subset['game_date'].dt.date <= today_date]
                        except Exception:
                            try:
                                today = pd.Timestamp(datetime.now().date())
                                subset['game_date'] = pd.to_datetime(subset['game_date'], errors='coerce')
                                subset = subset[subset['game_date'] <= today]
                            except Exception:
                                pass

                    if not subset.empty:
                        collected.append(subset)
            if sum(len(c) for c in collected) >= max_rows:
                break
    except Exception:
        return pd.DataFrame()

    if not collected:
        return pd.DataFrame()

    df = pd.concat(collected, ignore_index=True)
    # Enforce final filtering and sorting by calendar date to ensure results are <= today and descending
    if 'game_date' in df.columns:
        try:
            df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
            today_date = datetime.now().date()
            df = df[df['game_date'].dt.date <= today_date]
            df = df.sort_values(by='game_date', ascending=False)
        except Exception:
            try:
                df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
                today = pd.Timestamp(datetime.now().date())
                df = df[df['game_date'] <= today]
                df = df.sort_values(by='game_date', ascending=False)
            except Exception:
                pass

    return df.head(max_rows)


def _scan_all_pbp_files_for_latest_season(sample_rows=5000):
    """Quick-scan all CSV/CSV.GZ files in DATA_DIR by reading up to sample_rows
    and returning the file with the maximum season observed.
    This is bounded and safe for a quick check.
    Returns (best_file, best_season) or (None, None).
    """
    best_file = None
    best_season = None
    for fname in os.listdir(DATA_DIR):
        if not (fname.endswith('.csv') or fname.endswith('.csv.gz')):
            continue
        p = path.join(DATA_DIR, fname)
        compression = 'gzip' if fname.endswith('.gz') else None
        try:
            df = pd.read_csv(p, compression=compression, sep='\t', usecols=['season'], nrows=sample_rows, low_memory=True)
            if 'season' in df.columns:
                svals = pd.to_numeric(df['season'], errors='coerce').dropna().astype(int)
                if not svals.empty:
                    maxs = int(svals.max())
                    if best_season is None or maxs > best_season:
                        best_season = maxs
                        best_file = fname
        except Exception:
            continue
    return best_file, best_season

st.title("📊 Historical Data & Filters")
st.write("Historical play-by-play data and game summaries")

# Link back to predictions page
if st.button("🏈 Back to Predictions", type="secondary"):
    st.switch_page("predictions.py")

# Do not load the large play-by-play data automatically. Load on demand from the UI.
if 'historical_data' not in st.session_state or st.session_state.get('historical_data') is None:
    # Show a small sample immediately so users see content quickly.
    file_path = path.join(DATA_DIR, 'nfl_play_by_play_historical.csv.gz')
    sample_df = pd.DataFrame()
    if os.path.exists(file_path):
        try:
            # Read a small sample of rows and a minimal set of columns for quick display
            sample_usecols = [
                'game_date', 'week', 'season', 'home_team', 'away_team', 'posteam', 'defteam',
                'qtr', 'down', 'ydstogo', 'yardline_100', 'play_type', 'yards_gained', 'desc',
                # Advanced / numeric columns often used by filters
                'epa', 'td_prob', 'pass_attempt', 'rush_attempt', 'posteam_score', 'defteam_score', 'score_differential'
            ]

            # Try to bias the sample to the most recent season.
            # Prefer the predictions CSV (fast). If that doesn't contain recent seasons,
            # probe any up-to-date play-by-play file (e.g. `nfl_play_by_play_thru_2025.csv.gz`).
            most_recent_season = None
            try:
                preds_path = path.join(DATA_DIR, 'nfl_games_historical_with_predictions.csv')
                if os.path.exists(preds_path):
                    seasons_df = pd.read_csv(preds_path, sep='\t', usecols=['season'], low_memory=True)
                    seasons = pd.to_numeric(seasons_df['season'], errors='coerce').dropna().astype(int).unique().tolist()
                    if seasons:
                        most_recent_season = int(max(seasons))
                        # st.write(f"*Detected most recent season from predictions data: {most_recent_season}*")
            except Exception:
                most_recent_season = None

            # If predictions CSV didn't provide a recent season, check common pbp files
            if most_recent_season is None:
                try:
                    pbp_candidates = [
                        # 'nfl_play_by_play_thru_2025.csv.gz',
                        'nfl_play_by_play_historical.csv.gz'
                    ]
                    for fname in pbp_candidates:
                        p = path.join(DATA_DIR, fname)
                        if os.path.exists(p):
                            try:
                                sdf = pd.read_csv(p, compression='gzip', sep='\t', usecols=['season'], nrows=2000)
                                svals = pd.to_numeric(sdf['season'], errors='coerce').dropna().astype(int).unique().tolist()
                                if svals:
                                    most_recent_season = int(max(svals))
                                    break
                            except Exception:
                                continue
                except Exception:
                    most_recent_season = None

            # Determine available columns first to avoid errors
            header_cols = pd.read_csv(file_path, compression='gzip', sep='\t', nrows=0).columns.tolist()
            cols = [c for c in sample_usecols if c in header_cols]

            # If we have a recent season, try to read matching rows in chunks until we have 200 rows
            if most_recent_season is not None and 'season' in cols:
                collected = []
                for chunk in pd.read_csv(file_path, compression='gzip', sep='\t', usecols=cols, chunksize=5000):
                    try:
                        # Normalize season column to numeric for robust matching
                        if chunk['season'].dtype == object:
                            chunk['season'] = pd.to_numeric(chunk['season'], errors='coerce')
                        match = chunk[chunk['season'] == most_recent_season]
                        if not match.empty:
                            collected.append(match)
                        if sum(len(c) for c in collected) >= 200:
                            break
                    except Exception:
                        # If chunk processing fails, ignore and continue
                        continue

                if collected:
                    sample_df = pd.concat(collected, ignore_index=True).head(200)

            # Fallback: quick nrows read if we didn't collect recent-season rows
            if sample_df.empty:
                sample_df = pd.read_csv(file_path, compression='gzip', sep='\t', usecols=cols, nrows=200)

            # Lightweight dtype casts
            if 'yards_gained' in sample_df.columns:
                sample_df['yards_gained'] = pd.to_numeric(sample_df['yards_gained'], errors='coerce').astype('float32')
            if 'game_date' in sample_df.columns:
                sample_df['game_date'] = pd.to_datetime(sample_df['game_date'], errors='coerce')
            # If we detected a most recent season, prefer rows from that season
            try:
                if most_recent_season is not None and 'season' in sample_df.columns:
                    # normalize season and filter
                    sample_df['season'] = pd.to_numeric(sample_df['season'], errors='coerce')
                    if 'game_date' in sample_df.columns:
                        sample_df = sample_df[sample_df['season'] == most_recent_season]
                        sample_df = sample_df.sort_values(by='game_date', ascending=False).head(200)
                    else:
                        sample_df = sample_df[sample_df['season'] == most_recent_season].head(200)
            except Exception:
                # If filtering fails, keep whatever sample we have
                pass
        except Exception:
            sample_df = pd.DataFrame()
    else:
        st.error("Historical play-by-play data file not found. The nfl_play_by_play_historical.csv.gz file may be missing or empty.")
        st.stop()

    if not sample_df.empty:
        # Provide the sample to downstream code as `historical_data` so
        # the rest of the page (tabs) can operate on a DataFrame view
        # without requiring the full dataset to be loaded. The sample
        # will be displayed in the "Historical Play-by-Play Data Sample"
        # tab below to avoid duplicate tables on the page.
        # Prefer recent-season rows first so the sample looks current.
        try:
            if 'season' in sample_df.columns:
                sample_df['season'] = pd.to_numeric(sample_df['season'], errors='coerce')
                if 'game_date' in sample_df.columns:
                    sample_df = sample_df.sort_values(by=['season', 'game_date'], ascending=[False, False])
                else:
                    sample_df = sample_df.sort_values(by='season', ascending=False)
            elif 'game_date' in sample_df.columns:
                sample_df = sample_df.sort_values(by='game_date', ascending=False)
        except Exception:
            # If sorting fails for any reason, keep the original sample
            pass

        historical_data = sample_df
    else:
        st.info("No sample available. You can load the full historical dataset below.")
        # Ensure downstream code has a defined DataFrame to work with
        historical_data = pd.DataFrame()

    if st.button("Load full historical play-by-play data"):
        try:
            with st.spinner("Loading historical play-by-play (this may take several minutes)..."):
                df = load_historical_data()
                st.session_state['historical_data'] = df
                try:
                    st.success(f"Loaded {len(df):,} play-by-play rows")
                except Exception:
                    pass
                try:
                    st.experimental_rerun()
                except Exception:
                    pass
        except Exception as e:
            st.error(f"Failed to load historical data: {e}")
            st.stop()
else:
    # Get historical_data from session if present, otherwise use empty DataFrame
    historical_data = st.session_state.get('historical_data', pd.DataFrame())

current_year = datetime.now().year

# Top sample / snapshot removed per user request.
# The bottom table uses the same `historical_data` DataFrame and the
# filtering logic below (date filter, sorting, and sidebar filters).

# Render Quick Presets in the sidebar unconditionally so they never
# disappear when the page reruns or when data is loading.
with st.sidebar:
    st.subheader("Quick Presets")
    try:
        # Small targeted reset button that only unchecks the three preset boxes
        if st.button("Reset Presets"):
            # Uncheck the preset checkboxes by setting their keys to False,
            # then immediately rerun so the UI reflects the change.
            st.session_state['qp_redzone'] = False
            st.session_state['qp_3rd_short'] = False
            st.session_state['qp_pass_only'] = False
            st.rerun()

        # Track previous preset states to detect changes
        prev_redzone = st.session_state.get('_prev_qp_redzone', False)
        prev_3rd_short = st.session_state.get('_prev_qp_3rd_short', False)
        prev_pass_only = st.session_state.get('_prev_qp_pass_only', False)

        # Persistent checkboxes used as quick-presets so they remain visible
        qp_redzone = st.checkbox("Red Zone", value=st.session_state.get('qp_redzone', False), key='qp_redzone')
        qp_3rd_short = st.checkbox("3rd & Short", value=st.session_state.get('qp_3rd_short', False), key='qp_3rd_short')
        qp_pass_only = st.checkbox("Pass Attempts Only", value=st.session_state.get('qp_pass_only', False), key='qp_pass_only')
        
        # Detect if any preset was just checked (changed from False to True)
        preset_changed = False
        
        # Red Zone preset: set yardline_range to 0-20
        if qp_redzone and not prev_redzone and 'yardline_100' in historical_data.columns:
            try:
                ymin = int(historical_data['yardline_100'].min())
                ymax = int(historical_data['yardline_100'].max())
                st.session_state['yardline_range'] = (max(ymin, 0), min(20, ymax))
                preset_changed = True
            except Exception:
                pass
        
        # 3rd & Short preset: set down to [3] and ydstogo to 0-3
        if qp_3rd_short and not prev_3rd_short:
            st.session_state['down'] = [3]
            if 'ydstogo' in historical_data.columns:
                try:
                    ymin = int(historical_data['ydstogo'].min())
                    ymax = int(historical_data['ydstogo'].max())
                    st.session_state['ydstogo_range'] = (max(ymin, 0), min(3, ymax))
                except Exception:
                    pass
            preset_changed = True
        
        # Pass Only preset: set pass_only checkbox to True
        if qp_pass_only and not prev_pass_only:
            st.session_state['pass_only'] = True
            preset_changed = True
        
        # Update previous states for next run
        st.session_state['_prev_qp_redzone'] = qp_redzone
        st.session_state['_prev_qp_3rd_short'] = qp_3rd_short
        st.session_state['_prev_qp_pass_only'] = qp_pass_only
        
        # Rerun if any preset was just activated to update filter widgets
        if preset_changed:
            st.rerun()
            
    except Exception:
        pass

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
    
    # Initialize session state for preset-affected filters if they don't exist
    # This prevents widget conflicts when presets modify these values
    if 'down' not in st.session_state:
        st.session_state['down'] = []
    if 'pass_only' not in st.session_state:
        st.session_state['pass_only'] = False
    if 'ydstogo_range' not in st.session_state and 'ydstogo' in filtered_data.columns:
        ydstogo_min = int(filtered_data['ydstogo'].min())
        ydstogo_max = int(filtered_data['ydstogo'].max())
        st.session_state['ydstogo_range'] = (ydstogo_min, ydstogo_max)
    if 'yardline_range' not in st.session_state and 'yardline_100' in filtered_data.columns:
        yardline_min = int(filtered_data['yardline_100'].min())
        yardline_max = int(filtered_data['yardline_100'].max())
        st.session_state['yardline_range'] = (yardline_min, yardline_max)
    
    # Filters live in the left sidebar (restore original behavior per user request)
    with st.sidebar:
        st.header("🔍 Filters")
        
        # Reset button - sets reset flag
        if st.button("🔄 Reset All Filters"):
            st.session_state['reset'] = True
            st.rerun()
        
        st.divider()

        # (Quick Presets were moved to the top of the sidebar)
        
        # Default filter values
        default_filters = {
            'season': [],
            'posteam': [],
            'defteam': [],
            'down': [],
            'qtr': [],
            'play_type': [],
            'pass_only': False
        }
        
        # Team Filters
        # Season / Year filter
        st.subheader("Season / Year")
        season_options = []
        # Prefer to read available seasons from the predictions CSV (small, fast)
        try:
            preds_path = path.join(DATA_DIR, 'nfl_games_historical_with_predictions.csv')
            if os.path.exists(preds_path):
                seasons_df = pd.read_csv(preds_path, sep='\t', usecols=['season'], low_memory=True)
                season_options = sorted(pd.to_numeric(seasons_df['season'], errors='coerce').dropna().astype(int).unique().tolist())
        except Exception:
            season_options = []

        # Fallback to filtered_data if predictions CSV not available or empty
        if not season_options and 'season' in filtered_data.columns:
            try:
                season_options = sorted(filtered_data['season'].dropna().unique().tolist())
            except Exception:
                season_options = []

        # Do not preselect any seasons by default; allow users to choose explicitly
        selected_seasons = st.multiselect(
            "Season (Year)",
            season_options,
            default=default_filters['season'] if st.session_state['reset'] else st.session_state.get('season', []),
            key='season'
        )

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
        # Reset down filter if reset flag is set
        if st.session_state['reset']:
            st.session_state['down'] = default_filters['down']
        selected_downs = st.multiselect(
            "Down", 
            down_options,
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
        
        # Reset pass_only filter if reset flag is set
        if st.session_state['reset']:
            st.session_state['pass_only'] = default_filters['pass_only']
        pass_only = st.checkbox(
            "Pass Attempts Only",
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
            # Reset ydstogo_range if reset flag is set
            if st.session_state['reset']:
                st.session_state['ydstogo_range'] = (ydstogo_min, ydstogo_max)
            ydstogo_range = st.slider("Yards To Go", ydstogo_min, ydstogo_max, key='ydstogo_range')
        else:
            ydstogo_range = None
        
        if 'yardline_100' in filtered_data.columns:
            yardline_min = int(filtered_data['yardline_100'].min())
            yardline_max = int(filtered_data['yardline_100'].max())
            # Reset yardline_range if reset flag is set
            if st.session_state['reset']:
                st.session_state['yardline_range'] = (yardline_min, yardline_max)
            yardline_range = st.slider("Yardline (distance from opponent endzone)", yardline_min, yardline_max, key='yardline_range')
        else:
            yardline_range = None
        
        st.divider()
        
        # Score Filters
        st.subheader("Score")
        
        if 'score_differential' in filtered_data.columns:
            score_diff_min = int(filtered_data['score_differential'].min())
            score_diff_max = int(filtered_data['score_differential'].max())
            score_diff_range = st.slider("Score Differential", score_diff_min, score_diff_max, (score_diff_min, score_diff_max), key='score_diff_range')
        else:
            score_diff_range = None
        
        if 'posteam_score' in filtered_data.columns:
            posteam_score_min = int(filtered_data['posteam_score'].min())
            posteam_score_max = int(filtered_data['posteam_score'].max())
            posteam_score_range = st.slider("Offense Team Score", posteam_score_min, posteam_score_max, (posteam_score_min, posteam_score_max), key='posteam_score_range')
        else:
            posteam_score_range = None
        
        if 'defteam_score' in filtered_data.columns:
            defteam_score_min = int(filtered_data['defteam_score'].min())
            defteam_score_max = int(filtered_data['defteam_score'].max())
            defteam_score_range = st.slider("Defense Team Score", defteam_score_min, defteam_score_max, (defteam_score_min, defteam_score_max), key='defteam_score_range')
        else:
            defteam_score_range = None
        
        st.divider()
        
        # Advanced Metrics
        st.subheader("Advanced Metrics")
        
        if 'epa' in filtered_data.columns:
            epa_min = float(filtered_data['epa'].min())
            epa_max = float(filtered_data['epa'].max())
            epa_range = st.slider("Expected Points Added (EPA)", epa_min, epa_max, (epa_min, epa_max), key='epa_range')
        else:
            epa_range = None

        # (Quick Presets were moved to the top of the sidebar)
    
    # Apply filters to the data
    # Compute effective filter values including quick-preset overrides (persistent checkboxes)
    effective_selected_downs = selected_downs
    effective_ydstogo_range = ydstogo_range
    effective_yardline_range = yardline_range
    effective_pass_only = pass_only

    # Quick Preset overrides (checkbox-driven)
    if st.session_state.get('qp_redzone'):
        if 'yardline_100' in filtered_data.columns and not filtered_data.empty:
            try:
                ymin = int(filtered_data['yardline_100'].min())
                ymax = int(filtered_data['yardline_100'].max())
                low = max(ymin, 0)
                high = min(20, ymax)
                if low > high:
                    low, high = ymin, ymax
                effective_yardline_range = (low, high)
            except Exception:
                effective_yardline_range = None
        else:
            effective_yardline_range = None

    if st.session_state.get('qp_3rd_short'):
        effective_selected_downs = [3]
        if 'ydstogo' in filtered_data.columns and not filtered_data.empty:
            try:
                ymin = int(filtered_data['ydstogo'].min())
                ymax = int(filtered_data['ydstogo'].max())
                low = max(ymin, 0)
                high = min(3, ymax)
                if low > high:
                    low, high = ymin, ymax
                effective_ydstogo_range = (low, high)
            except Exception:
                effective_ydstogo_range = None
        else:
            effective_ydstogo_range = None

    # Pass-only presets
    if st.session_state.get('qp_pass_only'):
        effective_pass_only = True

    if selected_offense:
        filtered_data = filtered_data[filtered_data['posteam'].isin(selected_offense)]

    if selected_seasons:
        filtered_data = filtered_data[filtered_data['season'].isin(selected_seasons)]
    
    if selected_defense:
        filtered_data = filtered_data[filtered_data['defteam'].isin(selected_defense)]
    
    if effective_selected_downs:
        filtered_data = filtered_data[filtered_data['down'].isin(effective_selected_downs)]
    
    if selected_qtrs:
        filtered_data = filtered_data[filtered_data['qtr'].isin(selected_qtrs)]
    
    if selected_play_types:
        filtered_data = filtered_data[filtered_data['play_type'].isin(selected_play_types)]
    
    if effective_pass_only:
        filtered_data = filtered_data[filtered_data['pass_attempt'] == 1]
    
    if effective_ydstogo_range:
        filtered_data = filtered_data[(filtered_data['ydstogo'] >= effective_ydstogo_range[0]) & (filtered_data['ydstogo'] <= effective_ydstogo_range[1])]
    
    if effective_yardline_range:
        filtered_data = filtered_data[(filtered_data['yardline_100'] >= effective_yardline_range[0]) & (filtered_data['yardline_100'] <= effective_yardline_range[1])]
    
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

    if total_rows == 0:
        st.info("No rows match the current filters. Adjust filters or load the full dataset.")
        # Show empty dataframe with display columns if available
        empty_df = filtered_data[display_cols].head(0) if len(display_cols) > 0 else pd.DataFrame()
        st.dataframe(empty_df, hide_index=True, height=200)
    else:
        total_pages = (total_rows - 1) // rows_per_page + 1

        # Warn if showing too many rows
        if rows_per_page > 100 and total_rows > 10000:
            st.warning(f"⚠️ Displaying {rows_per_page} rows from a large dataset ({total_rows:,} total). Consider using filters to narrow down results for better performance.")

        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        start_idx = (page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, total_rows)

        st.write(f"Showing rows {start_idx + 1:,} to {end_idx:,} of {total_rows:,}")

        height = get_dataframe_height(filtered_data[display_cols].iloc[start_idx:end_idx])
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

    # Download button for current filtered view (only when rows exist)
    try:
        if total_rows > 0:
            to_download = filtered_data[display_cols].iloc[start_idx:end_idx]
            if not to_download.empty:
                csv_bytes = convert_df_to_csv(to_download)
                filename = f'nfl_historical_data_{datetime.now().strftime("%Y%m%d")}.csv'

                # CSV download button (show small icon from `data_files/` if available)
                icons_dir = path.join(DATA_DIR)
                csv_icon = path.join(icons_dir, 'csv_icon.png')
                fallback_icon = path.join(icons_dir, 'favicon.ico')

                # Render HTML-based download button with embedded icon (base64 data-URI).
                # Fallback to the existing Streamlit download button if something goes wrong.
                import base64
                try:
                    # Prepare data URI for download link
                    b64_file = base64.b64encode(csv_bytes).decode('ascii')
                    file_data_uri = f"data:text/csv;base64,{b64_file}"

                    # Choose icon (csv_icon preferred, fallback to favicon)
                    chosen_icon_path = None
                    if os.path.exists(csv_icon):
                        chosen_icon_path = csv_icon
                    elif os.path.exists(fallback_icon):
                        chosen_icon_path = fallback_icon

                    img_tag = ''
                    if chosen_icon_path is not None:
                        try:
                            with open(chosen_icon_path, 'rb') as ifh:
                                img_b64 = base64.b64encode(ifh.read()).decode('ascii')
                            # Larger icon and rounded corners for nicer appearance
                            img_tag = f'<img src="data:image/png;base64,{img_b64}" style="width:36px;height:36px;margin-right:10px;vertical-align:middle;border-radius:6px;">'
                        except Exception:
                            img_tag = ''

                    # HTML for a compact icon + button-like link; wrap the image inside the anchor so it's clickable
                    html = (
                        f'<div style="display:flex;align-items:center;margin:6px 0;">'
                        f'<a download="{filename}" href="{file_data_uri}" '
                        f'style="display:flex;align-items:center;padding:6px 12px;background:#1976d2;color:#fff;border-radius:6px;text-decoration:none;font-weight:600;">'
                        f'{img_tag}'
                        f'<span style="color:#fff;">Export Filtered Data</span>'
                        f'</a></div>'
                    )

                    st.markdown(html, unsafe_allow_html=True)
                except Exception:
                    # fallback to Streamlit native button
                    st.download_button(
                        label="📥 Export Filtered Data",
                        data=csv_bytes,
                        file_name=filename,
                        mime='text/csv'
                    )
    except Exception:
        # If export fails, don't break the page
        pass

else:
    # Fallback: show all data without date filtering
    st.dataframe(historical_data.head(50), hide_index=True)