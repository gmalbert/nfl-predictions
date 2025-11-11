import streamlit as st

# Set page config FIRST - must be the very first Streamlit command
st.set_page_config(
    page_title="NFL Outcome Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Import required libraries
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
import sys

# Debug: Print to logs to verify app is running
print("🚀 Starting NFL Predictor app...", file=sys.stderr, flush=True)

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

# Load historical game data with caching and error handling
@st.cache_data
def load_historical_data():
    """Load historical game data with caching"""
    try:
        data = pd.read_csv(path.join(DATA_DIR, 'nfl_games_historical_with_predictions.csv'), sep='\t')
        return data
    except FileNotFoundError:
        st.error("Critical file missing: nfl_games_historical_with_predictions.csv")
        st.info("Please run 'python nfl-gather-data.py' first to generate the required data files.")
        st.stop()  # Stop execution if critical data is missing
    except Exception as e:
        st.error(f"Error loading historical game data: {str(e)}")
        st.stop()

# DON'T load data at module level - will be loaded lazily when needed
historical_game_level_data = None

# Load model predictions CSV for display (cached)
@st.cache_data
def load_predictions_csv():
    """Load predictions CSV with caching"""
    predictions_csv_path = path.join(DATA_DIR, 'nfl_games_historical_with_predictions.csv')
    if os.path.exists(predictions_csv_path):
        return pd.read_csv(predictions_csv_path, sep='\t')
    return None

# DON'T load predictions at module level - will be loaded lazily when needed
predictions_csv_path = path.join(DATA_DIR, 'nfl_games_historical_with_predictions.csv')
predictions_df = None

# Function to automatically log betting recommendations
def log_betting_recommendations(predictions_df):
    """Automatically log betting recommendations to CSV for tracking"""
    if predictions_df is None:
        return
    
    log_path = path.join(DATA_DIR, 'betting_recommendations_log.csv')
    
    # Filter for upcoming games only (future games)
    upcoming_df = predictions_df.copy()
    if 'gameday' in upcoming_df.columns:
        upcoming_df['gameday'] = pd.to_datetime(upcoming_df['gameday'], errors='coerce')
        upcoming_df = upcoming_df[upcoming_df['gameday'] > pd.to_datetime(datetime.now())]
    
    if len(upcoming_df) == 0:
        return  # No upcoming games to log
    
    # Prepare records for both moneyline and spread bets
    records = []
    
    # Moneyline bets (underdog)
    if 'pred_underdogWon_optimal' in upcoming_df.columns:
        moneyline_bets = upcoming_df[upcoming_df['pred_underdogWon_optimal'] == 1].copy()
        
        for _, row in moneyline_bets.iterrows():
            # Determine underdog team
            if pd.notna(row.get('spread_line')):
                if row['spread_line'] < 0:
                    recommended_team = row['home_team']  # Home is underdog
                elif row['spread_line'] > 0:
                    recommended_team = row['away_team']  # Away is underdog
                else:
                    recommended_team = 'Pick'
            else:
                recommended_team = 'Unknown'
            
            # Determine confidence tier
            prob = row.get('prob_underdogWon', 0)
            if prob >= 0.75:
                confidence = 'Elite'
            elif prob >= 0.65:
                confidence = 'Strong'
            elif prob >= 0.54:
                confidence = 'Good'
            else:
                confidence = 'Standard'
            
            records.append({
                'log_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'season': row.get('season', current_year),
                'week': row.get('week', ''),
                'game_id': row.get('game_id', ''),
                'gameday': row.get('gameday', ''),
                'home_team': row.get('home_team', ''),
                'away_team': row.get('away_team', ''),
                'bet_type': 'moneyline_underdog',
                'recommended_team': recommended_team,
                'spread_line': row.get('spread_line', ''),
                'total_line': row.get('total_line', ''),
                'moneyline_odds': row.get('away_moneyline' if recommended_team == row.get('away_team') else 'home_moneyline', ''),
                'model_probability': row.get('prob_underdogWon', ''),
                'edge': row.get('edge_underdog_ml', ''),
                'confidence_tier': confidence,
                'actual_home_score': '',
                'actual_away_score': '',
                'bet_result': 'pending',
                'bet_profit': ''
            })
    
    # Spread bets
    if 'pred_spreadCovered_optimal' in upcoming_df.columns:
        spread_bets = upcoming_df[upcoming_df['pred_spreadCovered_optimal'] == 1].copy()
        
        for _, row in spread_bets.iterrows():
            # Determine underdog team for spread
            if pd.notna(row.get('spread_line')):
                if row['spread_line'] < 0:
                    recommended_team = row['home_team']  # Home is underdog
                elif row['spread_line'] > 0:
                    recommended_team = row['away_team']  # Away is underdog
                else:
                    recommended_team = 'Pick'
            else:
                recommended_team = 'Unknown'
            
            # Determine confidence tier
            prob = row.get('prob_underdogCovered', 0)
            if prob >= 0.75:
                confidence = 'Elite'
            elif prob >= 0.65:
                confidence = 'Strong'
            elif prob >= 0.54:
                confidence = 'Good'
            else:
                confidence = 'Standard'
            
            records.append({
                'log_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'season': row.get('season', current_year),
                'week': row.get('week', ''),
                'game_id': row.get('game_id', ''),
                'gameday': row.get('gameday', ''),
                'home_team': row.get('home_team', ''),
                'away_team': row.get('away_team', ''),
                'bet_type': 'spread',
                'recommended_team': recommended_team,
                'spread_line': row.get('spread_line', ''),
                'total_line': row.get('total_line', ''),
                'moneyline_odds': '',
                'model_probability': row.get('prob_underdogCovered', ''),
                'edge': row.get('edge_underdog_spread', ''),
                'confidence_tier': confidence,
                'actual_home_score': '',
                'actual_away_score': '',
                'bet_result': 'pending',
                'bet_profit': ''
            })
    
    if len(records) == 0:
        return  # No bets to log
    
    # Convert to DataFrame
    new_records_df = pd.DataFrame(records)
    
    # Load existing log or create new one
    if os.path.exists(log_path):
        existing_log = pd.read_csv(log_path)
        # Avoid duplicate entries: check if game_id + bet_type already exists with pending status
        existing_pending = existing_log[existing_log['bet_result'] == 'pending']
        
        # Filter out records that already exist as pending
        new_records_df = new_records_df[
            ~new_records_df.apply(
                lambda x: ((existing_pending['game_id'] == x['game_id']) & 
                          (existing_pending['bet_type'] == x['bet_type'])).any(),
                axis=1
            )
        ]
        
        if len(new_records_df) > 0:
            combined_log = pd.concat([existing_log, new_records_df], ignore_index=True)
            combined_log.to_csv(log_path, index=False)
    else:
        new_records_df.to_csv(log_path, index=False)

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

# Function to automatically update completed game results
def update_completed_games():
    """Fetch scores from ESPN API and update betting log for completed games"""
    log_path = path.join(DATA_DIR, 'betting_recommendations_log.csv')
    
    if not os.path.exists(log_path):
        return  # No log to update
    
    log_df = pd.read_csv(log_path)
    
    # Filter for pending bets only
    pending_bets = log_df[log_df['bet_result'] == 'pending'].copy()
    
    if len(pending_bets) == 0:
        return  # No pending bets to update
    
    # Convert gameday to datetime
    pending_bets['gameday'] = pd.to_datetime(pending_bets['gameday'], errors='coerce')
    
    # Only check games that should be completed (game day has passed)
    today = pd.to_datetime(datetime.now().date())
    completed_games = pending_bets[pending_bets['gameday'] < today].copy()
    
    if len(completed_games) == 0:
        return  # No games to check
    
    # Fetch scores from ESPN API for each completed game
    updates_made = False
    
    for idx, bet in completed_games.iterrows():
        season = int(bet['season']) if pd.notna(bet['season']) else current_year
        week = int(bet['week']) if pd.notna(bet['week']) else 1
        
        try:
            # Fetch ESPN data for that week
            espn_url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?seasontype=2&year={season}&week={week}"
            response = requests.get(espn_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Find matching game by team names
                for event in data.get("events", []):
                    comp = event.get("competitions", [{}])[0]
                    competitors = comp.get("competitors", [])
                    
                    if len(competitors) >= 2:
                        # ESPN format: competitors[0] is home, competitors[1] is away
                        home_team = competitors[0].get("team", {}).get("displayName", "")
                        away_team = competitors[1].get("team", {}).get("displayName", "")
                        
                        # Match by team names (case-insensitive)
                        if (home_team.lower() == str(bet['home_team']).lower() and 
                            away_team.lower() == str(bet['away_team']).lower()):
                            
                            # Check if game is completed
                            status = event.get("status", {}).get("type", {}).get("completed", False)
                            
                            if status:
                                # Get scores
                                home_score = int(competitors[0].get("score", 0))
                                away_score = int(competitors[1].get("score", 0))
                                
                                # Update the log dataframe
                                log_df.at[idx, 'actual_home_score'] = home_score
                                log_df.at[idx, 'actual_away_score'] = away_score
                                
                                # Determine bet result based on bet type
                                bet_type = bet['bet_type']
                                recommended_team = bet['recommended_team']
                                
                                if bet_type == 'moneyline_underdog':
                                    # Moneyline: did underdog win?
                                    if recommended_team == bet['home_team']:
                                        bet_won = home_score > away_score
                                    else:
                                        bet_won = away_score > home_score
                                    
                                    # Calculate profit
                                    if bet_won:
                                        # Get underdog odds to calculate payout
                                        odds = float(bet['moneyline_odds']) if pd.notna(bet['moneyline_odds']) else 0
                                        if odds > 0:
                                            profit = odds  # Win $odds on $100 bet
                                        else:
                                            profit = 100 / abs(odds) * 100 if odds != 0 else 0
                                    else:
                                        profit = -100  # Lost the $100 bet
                                    
                                elif bet_type == 'spread':
                                    # Spread: did underdog cover?
                                    spread_line = float(bet['spread_line']) if pd.notna(bet['spread_line']) else 0
                                    
                                    if spread_line < 0:
                                        # Home is underdog, gets + points
                                        adjusted_score = home_score - spread_line  # Subtracting negative = adding
                                        bet_won = adjusted_score > away_score
                                    else:
                                        # Away is underdog, gets + points
                                        adjusted_score = away_score + spread_line
                                        bet_won = adjusted_score > home_score
                                    
                                    # Standard spread profit (assuming -110 odds)
                                    profit = 90.91 if bet_won else -100
                                
                                else:
                                    bet_won = False
                                    profit = 0
                                
                                # Update result and profit
                                log_df.at[idx, 'bet_result'] = 'win' if bet_won else 'loss'
                                log_df.at[idx, 'bet_profit'] = profit
                                
                                updates_made = True
                                break  # Found the game, move to next bet
        
        except Exception as e:
            # Silently continue if ESPN API fails for a particular week
            continue
    
    # Save updated log if any changes were made
    if updates_made:
        log_df.to_csv(log_path, index=False)

# Automatically log recommendations when predictions are loaded
if predictions_df is not None:
    log_betting_recommendations(predictions_df)
    # Update completed games with scores
    update_completed_games()

@st.cache_data
def load_data():
    file_path = path.join(DATA_DIR, 'nfl_history_2020_2024.csv.gz')
    if os.path.exists(file_path):
        historical_data = pd.read_csv(file_path, compression='gzip', sep='\t', low_memory=False)
        return historical_data
    else:
        st.warning("Historical play-by-play data file not found. Some features may be limited.")
        return pd.DataFrame()  # Return empty DataFrame as fallback

# DON'T load at module level - will be loaded lazily when needed
historical_data = None

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

# Load data NOW (lazily, only when user accesses the app)
print("📊 About to load historical data...", file=sys.stderr, flush=True)
with st.spinner("🏈 Loading NFL data and predictions..."):
    if historical_game_level_data is None:
        print("📂 Loading historical game data from CSV...", file=sys.stderr, flush=True)
        historical_game_level_data = load_historical_data()
        print(f"✅ Loaded {len(historical_game_level_data)} rows", file=sys.stderr, flush=True)
    
    if predictions_df is None:
        print("📂 Loading predictions CSV...", file=sys.stderr, flush=True)
        predictions_df = load_predictions_csv()
        print(f"✅ Loaded predictions: {len(predictions_df) if predictions_df is not None else 0} rows", file=sys.stderr, flush=True)

print("🎉 Data loading complete, proceeding with app...", file=sys.stderr, flush=True)

# Feature list for modeling and Monte Carlo selection
print("📋 Setting up features...", file=sys.stderr, flush=True)
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
print("🎯 Setting up target variables...", file=sys.stderr, flush=True)
if 'spreadCovered' in historical_game_level_data.columns:
    target_spread = 'spreadCovered'
else:
    target_spread = st.selectbox('Select spread target column', historical_game_level_data.columns)

# Prepare data for MC feature selection

# Define X and y for spread
print("📊 Preparing spread model data...", file=sys.stderr, flush=True)
X = historical_game_level_data[features]
y_spread = historical_game_level_data[target_spread]

# Spread model (using best features) - filter to numeric only
available_spread_features = [f for f in best_features_spread if f in historical_game_level_data.columns]
X_spread_full = historical_game_level_data[available_spread_features].select_dtypes(include=["number", "bool", "category"])
print(f"✅ Spread features: {len(available_spread_features)} columns", file=sys.stderr, flush=True)
X_train_spread, X_test_spread, y_spread_train, y_spread_test = train_test_split(
    X_spread_full, y_spread, test_size=0.2, random_state=42, stratify=y_spread)
print("✅ Spread train/test split complete", file=sys.stderr, flush=True)

# --- Moneyline (underdogWon) target and split ---
print("💰 Preparing moneyline model data...", file=sys.stderr, flush=True)
target_moneyline = 'underdogWon'
y_moneyline = historical_game_level_data[target_moneyline]
# Filter to only numeric features for XGBoost compatibility
available_moneyline_features = [f for f in best_features_moneyline if f in historical_game_level_data.columns]
X_moneyline_full = historical_game_level_data[available_moneyline_features].select_dtypes(include=["number", "bool", "category"])
print(f"✅ Moneyline features: {len(available_moneyline_features)} columns", file=sys.stderr, flush=True)
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(
    X_moneyline_full, y_moneyline, test_size=0.2, random_state=42, stratify=y_moneyline)
print("✅ Moneyline train/test split complete", file=sys.stderr, flush=True)

# --- Totals (overHit) target and split ---
print("🎯 Preparing totals/over-under model data...", file=sys.stderr, flush=True)
try:
    target_totals = 'overHit'
    y_totals = historical_game_level_data[target_totals]
    print(f"✅ Target 'overHit' loaded: {len(y_totals)} rows", file=sys.stderr, flush=True)
    
    # Filter to only numeric features for XGBoost compatibility
    available_totals_features = [f for f in best_features_totals if f in historical_game_level_data.columns]
    print(f"✅ Totals features: {len(available_totals_features)} columns", file=sys.stderr, flush=True)
    
    X_totals_full = historical_game_level_data[available_totals_features].select_dtypes(include=["number", "bool", "category"])
    print(f"✅ X_totals_full shape: {X_totals_full.shape}", file=sys.stderr, flush=True)
    
    X_train_tot, X_test_tot, y_train_tot, y_test_tot = train_test_split(
        X_totals_full, y_totals, test_size=0.2, random_state=42, stratify=y_totals)
    print("✅ Totals train/test split complete", file=sys.stderr, flush=True)
except Exception as e:
    print(f"❌ ERROR in totals model setup: {type(e).__name__}: {str(e)}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
    raise

# TEMPORARILY SKIP: Create expander for data views (collapsed by default)
# This section causes timeout issues with 196k rows - will fix later
if False:  # Disable this entire expander section for now
    with st.expander("📊 Historical Data & Filters", expanded=False):
        # Lazy load historical data only when expander is accessed
        if historical_data is None:
            print("📂 Loading historical play-by-play data...", file=sys.stderr, flush=True)
            historical_data = load_data()
            print(f"✅ Loaded historical data: {len(historical_data)} rows", file=sys.stderr, flush=True)
    
    # Create tabs for different data views
    tab1, tab2, tab3, tab4 = st.tabs(["🏈 Play-by-Play Data", "📊 Game Summaries", "📅 Schedule", "🔍 Filters"])

    with tab1:
        st.write("### Historical Play-by-Play Data Sample for " + f"{current_year-4} to {current_year-1} Seasons")
        if not historical_data.empty:
            # Play-by-play data uses 'game_date' instead of 'gameday'  
            if 'game_date' in historical_data.columns:
                # Convert to datetime and filter for completed games
                filtered_data = historical_data.copy()
                if filtered_data['game_date'].dtype == 'object':
                    filtered_data['game_date'] = pd.to_datetime(filtered_data['game_date'], errors='coerce')
                current_date = pd.Timestamp(datetime.now().date())
                filtered_data = filtered_data[filtered_data['game_date'] <= current_date]
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
            
                st.dataframe(
                    filtered_data[display_cols].head(50),
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
            
        else:
            st.info("No historical play-by-play data available. The nfl_history_2020_2024.csv.gz file may be missing or empty.")

    with tab2:
        st.write("### Historical Game Summaries")
        historical_game_level_data_display = historical_game_level_data.copy()
        
        # Convert gameday to datetime and filter for completed games only (≤ today)
        historical_game_level_data_display['gameday'] = pd.to_datetime(historical_game_level_data_display['gameday'], errors='coerce')
        today = pd.to_datetime(datetime.now().date())
        historical_game_level_data_display = historical_game_level_data_display[historical_game_level_data_display['gameday'] <= today]
        
        # Sort by most recent games first
        historical_game_level_data_display = historical_game_level_data_display.sort_values(by='gameday', ascending=False)
        
        # Select key columns for display
        display_cols = [
            'gameday', 'week', 'season', 'home_team', 'away_team', 'home_qb_name', 'away_qb_name',
            'home_score', 'away_score', 'spread_line', 'home_spread_odds', 'away_spread_odds',
            'total_line', 'spreadCovered', 'overHit', 'underdogWon',
            'homeTeamWinPct', 'awayTeamWinPct', 'home_moneyline', 'away_moneyline'
        ]
        # Only use columns that exist
        display_cols = [col for col in display_cols if col in historical_game_level_data_display.columns]
        
        st.dataframe(
            historical_game_level_data_display[display_cols].head(50),
            hide_index=True,
            height=600,
            column_config={
                'gameday': st.column_config.DateColumn('Game Date', format='MM/DD/YYYY'),
                'week': st.column_config.NumberColumn('Week', format='%d'),
                'season': st.column_config.NumberColumn('Season', format='%d'),
                'home_team': st.column_config.TextColumn('Home Team', width='medium'),
                'away_team': st.column_config.TextColumn('Away Team', width='medium'),
                'home_qb_name': st.column_config.TextColumn('Home QB', width='medium', help='Shows 0 for upcoming games'),
                'away_qb_name': st.column_config.TextColumn('Away QB', width='medium', help='Shows 0 for upcoming games'),
                'home_score': st.column_config.NumberColumn('Home Score', format='%d'),
                'away_score': st.column_config.NumberColumn('Away Score', format='%d'),
                'spread_line': st.column_config.NumberColumn('Spread', format='%.1f', help='Negative = away favored'),
                'home_spread_odds': st.column_config.NumberColumn('Home Spread Odds', format='%d'),
                'away_spread_odds': st.column_config.NumberColumn('Away Spread Odds', format='%d'),
                'total_line': st.column_config.NumberColumn('O/U Line', format='%.1f'),
                'spreadCovered': st.column_config.CheckboxColumn('Spread Covered?'),
                'overHit': st.column_config.CheckboxColumn('Over Hit?'),
                'underdogWon': st.column_config.CheckboxColumn('Underdog Won?'),
                'homeTeamWinPct': st.column_config.NumberColumn('Home Win %', format='%.1f%%', help='Historical win percentage'),
                'awayTeamWinPct': st.column_config.NumberColumn('Away Win %', format='%.1f%%', help='Historical win percentage'),
                'home_moneyline': st.column_config.NumberColumn('Home ML', format='%d'),
                'away_moneyline': st.column_config.NumberColumn('Away ML', format='%d')
            }
        )

    with tab3:
        st.write(f"### {current_year} NFL Schedule")
        if not schedule.empty:
            display_cols = ['week', 'date', 'home_team', 'away_team', 'venue']
            # Convert UTC date string to local datetime
            schedule_local = schedule.copy()
            schedule_local['date'] = pd.to_datetime(schedule_local['date']).dt.tz_convert('America/New_York').dt.strftime('%m/%d/%Y %I:%M %p')
            st.dataframe(schedule_local[display_cols], height=600, hide_index=True, column_config={'date': 'Date/Time (ET)', 'home_team': 'Home Team', 'away_team': 'Away Team', 'venue': 'Venue', 'week': 'Week'})
        else:
            st.warning(f"Schedule data for {current_year} is not available.")

    with tab4:
            import math
        
            # Helpful message to open sidebar
            st.info("👈 Open the sidebar (click arrow in top-left) to see filter controls")
        
            with st.sidebar:
                st.header("Filters")
                if 'reset' not in st.session_state:
                    st.session_state['reset'] = False
                # Initialize session state for filters
                for key in filter_keys:
                    if key not in st.session_state:
                        if key in ['down', 'posteam', 'defteam', 'play_type', 'qtr']:
                            st.session_state[key] = []
                        elif key == 'pass_attempt':
                            st.session_state[key] = False  # Checkbox default
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
                        if key in ['down', 'posteam', 'defteam', 'play_type', 'qtr']:
                            st.session_state[key] = []
                        elif key == 'pass_attempt':
                            st.session_state[key] = False  # Reset checkbox
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
                    'pass_attempt': False
                }

                # Filters
                posteam_options = historical_data['posteam'].unique().tolist()
                posteam = st.multiselect("Possession Team", posteam_options, key="posteam")
                # Defense Team
                defteam_options = historical_data['defteam'].unique().tolist()
                defteam = st.multiselect("Defense Team", defteam_options, key="defteam")
                # Down
                down_options = [1,2,3,4]
                down = st.multiselect("Down", down_options, key="down")
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
                play_type = st.multiselect("Play Type", play_type_options, key="play_type")
                # Quarter
                qtr_options = sorted([q for q in historical_data['qtr'].dropna().unique() if not (isinstance(q, float) and math.isnan(q))])
                qtr = st.multiselect("Quarter", qtr_options, key="qtr")
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
                pass_attempt = st.checkbox("Pass Attempts Only", key="pass_attempt")

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
                filtered_data = filtered_data[filtered_data['pass_attempt'] == 1]

            st.write("### Filtered Historical Data")

            # Create display copy and convert probability columns to percentages
            display_data = filtered_data.head(50).copy()

            # Identify probability columns (typically range 0-1) and convert to percentages
            prob_columns = ['wp', 'def_wp', 'home_wp', 'away_wp', 'vegas_wp', 'vegas_home_wp', 
                            'cp', 'cpoe', 'success', 'pass_oe', 'qb_epa', 'xyac_epa']

            for col in prob_columns:
                if col in display_data.columns:
                    # Check if values are in 0-1 range (probabilities)
                    if display_data[col].notna().any() and display_data[col].between(0, 1).all():
                        display_data[col] = display_data[col] * 100

            # Configure columns with appropriate formatting
            column_config = {}
            for col in display_data.columns:
                if col in prob_columns and col in display_data.columns:
                    column_config[col] = st.column_config.NumberColumn(col, format='%.1f%%')
                elif col in ['epa', 'wpa', 'air_epa', 'yac_epa', 'comp_air_epa', 'comp_yac_epa',
                             'air_wpa', 'yac_wpa', 'comp_air_wpa', 'comp_yac_wpa', 'ep', 'vegas_wpa']:
                    column_config[col] = st.column_config.NumberColumn(col, format='%.3f')
                elif col in ['yards_gained', 'air_yards', 'yards_after_catch', 'ydstogo', 
                             'yardline_100', 'score_differential', 'posteam_score', 'defteam_score']:
                    column_config[col] = st.column_config.NumberColumn(col, format='%d')

            st.dataframe(display_data, hide_index=True, column_config=column_config if column_config else None)

print("🎨 Starting main UI rendering...", file=sys.stderr, flush=True)

# Create tabs for prediction and betting sections
st.write("---")
st.write("## 📈 Model Performance & Betting Analysis")

# Upcoming Games Schedule with Predictions
if not schedule.empty and predictions_df is not None:
    with st.expander("📅 Upcoming Games Schedule (Click to expand)", expanded=False):
        st.write("### This Week's Games with Model Predictions")
        
        # Team name mapping from full names to abbreviations
        team_abbrev_map = {
            'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
            'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
            'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
            'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
            'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
            'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
            'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
            'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
            'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
            'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
            'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
        }
        
        # Filter for upcoming games (not STATUS_FINAL and future dates)
        current_time = pd.Timestamp.now(tz='UTC')
        upcoming_schedule = schedule.copy()
        upcoming_schedule['date'] = pd.to_datetime(upcoming_schedule['date'], utc=True)
        upcoming_mask = (upcoming_schedule['status'] != 'STATUS_FINAL') & (upcoming_schedule['date'] > current_time)
        upcoming_games = upcoming_schedule[upcoming_mask].copy()
        
        if not upcoming_games.empty:
            # Convert to local time for display
            upcoming_games['date'] = upcoming_games['date'].dt.tz_convert('America/New_York')
            upcoming_games['date_display'] = upcoming_games['date'].dt.strftime('%m/%d/%Y %I:%M %p ET')
            
            # Sort by date
            upcoming_games = upcoming_games.sort_values('date').head(15)  # Show next 15 games
            
            # Create display dataframe
            schedule_display = []
            
            for _, game in upcoming_games.iterrows():
                # Convert team names to abbreviations for matching
                home_abbrev = team_abbrev_map.get(game['home_team'], game['home_team'])
                away_abbrev = team_abbrev_map.get(game['away_team'], game['away_team'])
                
                # Find matching prediction (if available)
                pred_match = predictions_df[
                    ((predictions_df['home_team'] == home_abbrev) & (predictions_df['away_team'] == away_abbrev)) |
                    ((predictions_df['home_team'] == away_abbrev) & (predictions_df['away_team'] == home_abbrev))
                ]
                
                if not pred_match.empty:
                    pred = pred_match.iloc[0]
                    
                    # Determine underdog and favorite
                    if pred.get('spread_line', 0) < 0:
                        favorite = pred['away_team']
                        underdog = pred['home_team']
                        spread = abs(pred['spread_line'])
                    else:
                        favorite = pred['home_team']
                        underdog = pred['away_team']
                        spread = pred['spread_line']
                    
                    schedule_display.append({
                        'Date': game['date_display'],
                        'Matchup': f"{game['away_team']} @ {game['home_team']}",
                        'Spread': f"{favorite} -{spread}" if spread > 0 else "Pick'em",
                        'Total': f"{pred.get('total_line', 'N/A')}",
                        'Underdog Win %': f"{pred.get('prob_underdogWon', 0):.1%}",
                        'Spread Cover %': f"{pred.get('prob_underdogCovered', 0):.1%}",
                        'Over Hit %': f"{pred.get('prob_overHit', 0):.1%}",
                        'ML Edge': f"{pred.get('edge_underdog_ml', 0):.1f}",
                        'Spread Edge': f"{pred.get('edge_underdog_spread', 0):.1f}",
                        'Total Edge': f"{pred.get('edge_over', 0):.1f}"
                    })
                else:
                    # No prediction available
                    schedule_display.append({
                        'Date': game['date_display'],
                        'Matchup': f"{game['away_team']} @ {game['home_team']}",
                        'Spread': "TBD",
                        'Total': "TBD",
                        'Underdog Win %': "N/A",
                        'Spread Cover %': "N/A",
                        'Over Hit %': "N/A",
                        'ML Edge': "N/A",
                        'Spread Edge': "N/A",
                        'Total Edge': "N/A"
                    })
            
            if schedule_display:
                schedule_df = pd.DataFrame(schedule_display)
                height = get_dataframe_height(schedule_df)
                st.dataframe(
                    schedule_df,
                    hide_index=True,
                    height=height,
                    column_config={
                        'Date': st.column_config.TextColumn('Date/Time', width='medium'),
                        'Matchup': st.column_config.TextColumn('Matchup', width='large'),
                        'Spread': st.column_config.TextColumn('Spread', width='medium'),
                        'Total': st.column_config.TextColumn('O/U', width='small'),
                        'Underdog Win %': st.column_config.TextColumn('Underdog Win %', width='small'),
                        'Spread Cover %': st.column_config.TextColumn('Cover %', width='small'),
                        'Over Hit %': st.column_config.TextColumn('Over %', width='small'),
                        'ML Edge': st.column_config.NumberColumn('ML Edge', format='%.1f', width='small'),
                        'Spread Edge': st.column_config.NumberColumn('Spread Edge', format='%.1f', width='small'),
                        'Total Edge': st.column_config.NumberColumn('Total Edge', format='%.1f', width='small')
                    }
                )
                st.caption(f"Showing next {len(schedule_display)} upcoming games • Green edges indicate positive expected value bets")
            else:
                st.info("No upcoming games found in schedule data.")
        else:
            st.info("No upcoming games scheduled.")
else:
    with st.expander("📅 Upcoming Games Schedule", expanded=False):
        st.info("Schedule or prediction data not available.")

print("✅ Main tabs section loaded", file=sys.stderr, flush=True)
pred_tab1, pred_tab2, pred_tab3, pred_tab4, pred_tab5, pred_tab6, pred_tab7 = st.tabs([
    "📊 Model Predictions", 
    "🎯 Probabilities & Edges",
    "💰 Betting Performance",
    "🔥 Underdog Bets",
    "🏈 Spread Bets",
    "🎯 Over/Under Bets",
    "📋 Betting Log"
])

with pred_tab1:
    
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
        
        st.dataframe(
            predictions_df[display_cols].head(50), 
            hide_index=True,
            height=600,
            column_config={
                'game_id': None,
                'gameday': st.column_config.DateColumn('Game Date', format='MM/DD/YYYY'),
                'home_team': st.column_config.TextColumn('Home Team', width='medium'),
                'away_team': st.column_config.TextColumn('Away Team', width='medium'),
                'home_score': st.column_config.NumberColumn('Home Score', format='%d'),
                'away_score': st.column_config.NumberColumn('Away Score', format='%d'),
                'total_line': st.column_config.NumberColumn('O/U Line', format='%.1f', help='Over/Under betting line'),
                'spread_line': st.column_config.NumberColumn('Spread', format='%.1f', help='Point spread (negative = away favored)'),
                'predictedSpreadCovered': st.column_config.CheckboxColumn('Predicted Spread', help='Model prediction: underdog covers spread'),
                'spreadCovered': st.column_config.CheckboxColumn('Actual Spread', help='Actual result: underdog covered spread'),
                'predictedOverHit': st.column_config.CheckboxColumn('Predicted Over', help='Model prediction: total goes over'),
                'overHit': st.column_config.CheckboxColumn('Actual Over', help='Actual result: total went over')
            }
        )
        st.caption(f"Showing 50 most recent completed games • Total predictions: {predictions_df.shape[0]:,} games")
    else:
        st.warning("Predictions CSV not found. Run the model script to generate predictions.")

with pred_tab2:
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
        
        # Don't convert gameday to string yet - keep as datetime for sorting
        predictions_df['pred_underdogWon_optimal'] = predictions_df['pred_underdogWon_optimal'].astype(int)
        
        # Create a display copy and convert probabilities/edges to percentages
        display_df = predictions_df[display_cols].sort_values(by='gameday', ascending=False).head(50).copy()
        
        # Convert probability and edge columns from decimal to percentage
        prob_cols = ['prob_underdogCovered', 'implied_prob_underdog_spread', 'edge_underdog_spread',
                     'prob_underdogWon', 'implied_prob_underdog_ml', 'edge_underdog_ml',
                     'prob_overHit', 'implied_prob_over', 'edge_over', 'implied_prob_under', 'edge_under']
        for col in prob_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col] * 100
        
        st.dataframe(
            display_df, 
            hide_index=True, 
            height=600,
            column_config={
                'gameday': st.column_config.DateColumn('Date', format='MM/DD/YYYY', width='small'),
                'home_team': st.column_config.TextColumn('Home', width='small'),
                'away_team': st.column_config.TextColumn('Away', width='small'),
                'favored_team': st.column_config.TextColumn('Favored', width='small', help='Team favored to win'),
                'home_score': st.column_config.NumberColumn('Home Pts', format='%d', width='small'),
                'away_score': st.column_config.NumberColumn('Away Pts', format='%d', width='small'),
                'spread_line': st.column_config.NumberColumn('Spread', format='%.1f', width='small', help='Point spread (negative = away favored)'),
                'total_line': st.column_config.NumberColumn('O/U', format='%.1f', width='small', help='Over/Under line'),
                'prob_underdogCovered': st.column_config.NumberColumn('Spread Prob', format='%.1f%%', width='small', help='Model probability underdog covers spread'),
                'implied_prob_underdog_spread': st.column_config.NumberColumn('Implied Spread', format='%.1f%%', width='small', help='Sportsbook implied probability for underdog covering'),
                'edge_underdog_spread': st.column_config.NumberColumn('Spread Edge', format='%.1f%%', width='small', help='Model edge for underdog spread bet'),
                'prob_underdogWon': st.column_config.NumberColumn('ML Prob', format='%.1f%%', width='small', help='Model probability underdog wins outright'),
                'pred_underdogWon_optimal': st.column_config.CheckboxColumn('ML Signal', help='🎯 Betting signal: Bet on underdog (≥28% threshold)'),
                'implied_prob_underdog_ml': st.column_config.NumberColumn('Implied ML', format='%.1f%%', width='small', help='Sportsbook implied probability for underdog moneyline'),
                'edge_underdog_ml': st.column_config.NumberColumn('ML Edge', format='%.1f%%', width='small', help='Model edge for underdog moneyline bet'),
                'prob_overHit': st.column_config.NumberColumn('Over Prob', format='%.1f%%', width='small', help='Model probability total goes over'),
                'implied_prob_over': st.column_config.NumberColumn('Implied Over', format='%.1f%%', width='small', help='Sportsbook implied probability for over'),
                'edge_over': st.column_config.NumberColumn('Over Edge', format='%.1f%%', width='small', help='Model edge for over bet'),
                'implied_prob_under': st.column_config.NumberColumn('Implied Under', format='%.1f%%', width='small', help='Sportsbook implied probability for under'),
                'edge_under': st.column_config.NumberColumn('Under Edge', format='%.1f%%', width='small', help='Model edge for under bet')
            }
        )
        st.caption(f"Showing next 50 upcoming games with betting data • {len(predictions_df):,} games in next week")
        
        st.info("""
        **📌 Quick Reference:**
        - **Prob Columns**: Model's predicted probability (higher = more confident)
        - **Implied Columns**: Probability implied by sportsbook odds
        - **Edge Columns**: Model advantage vs sportsbook (positive = value bet)
        - **ML Signal (✅)**: Automated betting signal when model probability ≥ 28% (F1-optimized threshold)
        
        💡 **Positive edge = value bet opportunity** (model thinks probability is higher than sportsbook odds suggest)
        """)
    else:
        st.warning("Predictions CSV not found. Run the model script to generate predictions.")

with pred_tab3:
    if predictions_df is not None:
        st.write("### 📊 Betting Analysis & Performance")
        st.write("Assuming a $100 bet per game")
        # Load the full predictions data for analysis
        try:
            predictions_df_full = pd.read_csv(predictions_csv_path, sep='\t')
            # Calculate moneyline_bet_return column
            def calc_moneyline_return(row):
                if row['pred_underdogWon_optimal'] == 1:
                    if row['underdogWon'] == 1:
                        underdog_odds = max(row['away_moneyline'], row['home_moneyline'])
                        if underdog_odds > 0:
                            return underdog_odds
                        else:
                            return 100 / abs(underdog_odds) * 100 if underdog_odds != 0 else 0
                    else:
                        return -100
                else:
                    return 0
            predictions_df_full['moneyline_bet_return'] = predictions_df_full.apply(calc_moneyline_return, axis=1)
        except Exception as e:
            st.error(f"Error loading predictions data for betting analysis: {str(e)}")
            st.info("Please run 'python nfl-gather-data.py' to generate the required data files.")
            predictions_df_full = None
        
        # Calculate betting statistics
        if predictions_df_full is not None and 'pred_underdogWon_optimal' in predictions_df_full.columns and 'moneyline_bet_return' in predictions_df_full.columns:
            # Moneyline betting stats
            moneyline_bets = predictions_df_full[predictions_df_full['pred_underdogWon_optimal'] == 1].copy()
            if len(moneyline_bets) > 0:
                bet_returns = moneyline_bets['moneyline_bet_return']
                moneyline_wins = (bet_returns > 0).sum()
                moneyline_total_return = bet_returns.sum()
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
            
            # Over/Under betting performance
            if 'totals_bet_return' in predictions_df_full.columns:
                totals_bets = predictions_df_full[predictions_df_full['totals_bet_return'].notna()]
                totals_wins = (totals_bets['totals_bet_return'] > 0).sum()
                totals_total_return = totals_bets['totals_bet_return'].sum()
                totals_win_rate = totals_wins / len(totals_bets)
                totals_roi = totals_total_return / (len(totals_bets) * 100)
                
                st.write("#### 🎯 Over/Under Betting Performance")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Over/Under Bets", f"{len(totals_bets):,}")
                with col2:
                    st.metric("Win Rate", f"{totals_win_rate:.1%}")
                with col3:
                    st.metric("ROI", f"{totals_roi:.1%}")
        
        # Performance comparison
        st.write("#### 🏆 Model vs Baseline Comparison")
        baseline_accuracy = (predictions_df_full['underdogWon'] == 0).mean()  # Always pick favorites
        st.write(f"- **Baseline Strategy (Always Pick Favorites)**: {baseline_accuracy:.1%} accuracy")
        st.write(f"- **Our Model**: Identifies profitable underdog opportunities with {moneyline_roi:.1%} ROI")
        st.write(f"- **Key Insight**: While model sacrifices overall accuracy, it finds value bets with positive expected return")
        
    else:
        st.warning("Predictions CSV not found. Run the model script to generate betting analysis.")

with pred_tab4:
    if predictions_df is not None:
        st.write("### 🎯 Next 10 Recommended Underdog Bets")
        st.write("*Games where model recommends betting on underdog to win (≥28% confidence)*")
        
        # Reload predictions_df fresh and filter for upcoming games only
        predictions_df_upcoming = pd.read_csv(predictions_csv_path, sep='\t')
        predictions_df_upcoming['gameday'] = pd.to_datetime(predictions_df_upcoming['gameday'], errors='coerce')
        
        # Filter for future games only
        today = pd.to_datetime(datetime.now().date())
        predictions_df_upcoming = predictions_df_upcoming[predictions_df_upcoming['gameday'] > today]
        
        # Filter for upcoming games where we should bet on underdog, sort by date, take first 10
        upcoming_bets = predictions_df_upcoming[predictions_df_upcoming['pred_underdogWon_optimal'] == 1].copy()
        
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
                    if underdog_odds == 0:
                        return 'N/A (even odds)'
                    elif underdog_odds > 0:
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

with pred_tab5:
    if predictions_df is not None:
        st.write("### 🏈 Next 10 Recommended Spread Bets")
        st.write("*Games where model thinks underdog will cover spread (>50% confidence)*")
        
        # Reload predictions_df fresh and filter for upcoming games only
        predictions_df_spread = pd.read_csv(predictions_csv_path, sep='\t')
        predictions_df_spread['gameday'] = pd.to_datetime(predictions_df_spread['gameday'], errors='coerce')
        
        # Filter for future games only
        today = pd.to_datetime(datetime.now().date())
        predictions_df_spread = predictions_df_spread[predictions_df_spread['gameday'] > today]
        
        # Filter for upcoming games where model thinks underdog has ANY chance to cover (>50%)
        if 'prob_underdogCovered' in predictions_df_spread.columns:
            spread_bets = predictions_df_spread[predictions_df_spread['prob_underdogCovered'] > 0.50].copy()
            
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
                
                height = get_dataframe_height(final_spread_display)
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
                    height=height,
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

with pred_tab6:
    st.write("### 🎯 Over/Under Betting Opportunities")
    st.write("*Top games where the model predicts profitable over/under bets based on optimal threshold*")
    
    if os.path.exists(predictions_csv_path):
        predictions_df_full = pd.read_csv(predictions_csv_path, sep='\t')
        
        # Check for the required column
        if 'pred_overHit_optimal' not in predictions_df_full.columns:
            st.error("Over/under predictions not found. Ensure pred_overHit_optimal column exists in the predictions CSV.")
        else:
            # Filter for games with over/under betting signals AND that haven't been played yet
            totals_bets = predictions_df_full[
                (predictions_df_full['pred_overHit_optimal'] == 1) & 
                (pd.to_datetime(predictions_df_full['gameday']) > pd.Timestamp.now().normalize())
            ].copy()
            
            if len(totals_bets) > 0:
                # Add confidence tiers based on probability
                def get_totals_confidence_tier(prob):
                    if prob >= 0.65:
                        return "🔥 Elite"
                    elif prob >= 0.60:
                        return "💪 Strong"
                    elif prob >= 0.55:
                        return "✓ Good"
                    else:
                        return "→ Standard"
                
                totals_bets['confidence_tier'] = totals_bets['prob_overHit'].apply(get_totals_confidence_tier)
                
                # Calculate expected payout on $100 bet
                def calculate_over_payout(row):
                    # If predicting over, use over odds, else under odds
                    if row['pred_over'] == 1:
                        odds = row['over_odds']
                        bet_on = 'Over'
                    else:
                        odds = row['under_odds']
                        bet_on = 'Under'
                    
                    if odds > 0:
                        payout = 100 + (odds)
                    else:
                        payout = 100 + (100 * 100 / abs(odds))
                    
                    return payout, bet_on
                
                totals_bets[['expected_payout', 'bet_on']] = totals_bets.apply(
                    lambda row: pd.Series(calculate_over_payout(row)), axis=1
                )
                
                # Calculate value edge
                totals_bets['value_edge'] = (
                    totals_bets['prob_overHit'] * totals_bets['expected_payout'] - 100
                ) / 100
                
                # Sort by value edge
                totals_bets = totals_bets.sort_values('value_edge', ascending=False)
                
                # Summary metrics
                st.write("#### 📊 Over/Under Betting Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Opportunities", len(totals_bets))
                with col2:
                    avg_prob = totals_bets['prob_overHit'].mean()
                    st.metric("Avg Probability", f"{avg_prob:.1%}")
                with col3:
                    avg_edge = totals_bets['value_edge'].mean()
                    st.metric("Avg Value Edge", f"{avg_edge:.1%}")
                
                # Display top opportunities
                st.write("#### 🎯 Top 15 Over/Under Opportunities")
                
                display_totals = totals_bets.head(15).copy()
                
                # Format for display
                display_totals['matchup'] = display_totals['away_team'] + ' @ ' + display_totals['home_team']
                display_totals['game_date'] = pd.to_datetime(display_totals['gameday']).dt.strftime('%m/%d/%Y')
                display_totals['total_line'] = display_totals['total_line'].round(1)
                display_totals['prob_pct'] = (display_totals['prob_overHit'] * 100).round(1)
                display_totals['value_pct'] = (display_totals['value_edge'] * 100).round(1)
                display_totals['payout_fmt'] = '$' + display_totals['expected_payout'].round(0).astype(int).astype(str)
                
                # Create display dataframe
                display_cols = {
                    'game_date': 'Date',
                    'matchup': 'Matchup',
                    'bet_on': 'Bet',
                    'total_line': 'Line',
                    'prob_pct': 'Model Prob %',
                    'payout_fmt': 'Expected Payout',
                    'value_pct': 'Value Edge %',
                    'confidence_tier': 'Confidence'
                }
                
                height = get_dataframe_height(display_totals)
                st.dataframe(
                    display_totals[list(display_cols.keys())].rename(columns=display_cols),
                    column_config={
                        'Date': st.column_config.TextColumn('Date', width='small'),
                        'Matchup': st.column_config.TextColumn('Matchup', width='large'),
                        'Bet': st.column_config.TextColumn('Bet', width='small'),
                        'Line': st.column_config.NumberColumn('Line', format='%.1f'),
                        'Model Prob %': st.column_config.NumberColumn('Model Prob %', format='%.1f'),
                        'Expected Payout': st.column_config.TextColumn('Expected Payout', width='medium'),
                        'Value Edge %': st.column_config.NumberColumn('Value Edge %', format='%.1f'),
                        'Confidence': st.column_config.TextColumn('Confidence', width='medium')
                    },
                    hide_index=True,
                    height=height,
                    use_container_width=True
                )
                
                # Confidence tier breakdown
                st.write("#### 📈 Opportunities by Confidence Tier")
                tier_counts = display_totals['confidence_tier'].value_counts().sort_index(ascending=False)
                
                cols = st.columns(len(tier_counts))
                for i, (tier, count) in enumerate(tier_counts.items()):
                    with cols[i]:
                        st.metric(tier, count)
                
                # Betting strategy guide
                st.info("""
                **💡 Over/Under Betting Strategy:**
                - **Elite (≥65%)**: Highest confidence bets with strong value edge
                - **Strong (60-65%)**: Very good betting opportunities
                - **Good (55-60%)**: Solid value bets worth considering
                - **Standard (<55%)**: Meets threshold but requires careful consideration
                
                **Value Edge** represents the expected profit percentage on a $100 bet based on model probability.
                """)
                
            else:
                st.info("No over/under betting opportunities found for current games. Check back when new predictions are available.")
    
    else:
        st.warning("Predictions CSV not found. Run the model script to generate over/under betting opportunities.")

with pred_tab7:
    st.write("### 📋 Betting Recommendations Tracking Log")
    st.write("*All logged betting recommendations with performance tracking*")
    
    log_path = path.join(DATA_DIR, 'betting_recommendations_log.csv')
    
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        
        if len(log_df) > 0:
            # Convert dates for filtering
            log_df['gameday'] = pd.to_datetime(log_df['gameday'], errors='coerce')
            log_df['log_date'] = pd.to_datetime(log_df['log_date'], errors='coerce')
            
            # Sidebar filters for log
            st.write("#### Filter Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Filter by bet type
                bet_types = ['All'] + list(log_df['bet_type'].unique())
                selected_bet_type = st.selectbox("Bet Type", bet_types, key="log_bet_type")
            
            with col2:
                # Filter by status
                statuses = ['All'] + list(log_df['bet_result'].unique())
                selected_status = st.selectbox("Bet Status", statuses, key="log_status")
            
            with col3:
                # Filter by confidence tier
                tiers = ['All'] + list(log_df['confidence_tier'].unique())
                selected_tier = st.selectbox("Confidence Tier", tiers, key="log_tier")
            
            # Apply filters
            filtered_log = log_df.copy()
            if selected_bet_type != 'All':
                filtered_log = filtered_log[filtered_log['bet_type'] == selected_bet_type]
            if selected_status != 'All':
                filtered_log = filtered_log[filtered_log['bet_result'] == selected_status]
            if selected_tier != 'All':
                filtered_log = filtered_log[filtered_log['confidence_tier'] == selected_tier]
            
            # Summary statistics
            st.write("#### 📊 Summary Statistics")
            
            total_bets = len(filtered_log)
            pending_bets = len(filtered_log[filtered_log['bet_result'] == 'pending'])
            completed_bets = len(filtered_log[filtered_log['bet_result'] != 'pending'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Bets", total_bets)
            with col2:
                st.metric("Pending", pending_bets)
            with col3:
                st.metric("Completed", completed_bets)
            with col4:
                if completed_bets > 0:
                    wins = len(filtered_log[filtered_log['bet_result'] == 'win'])
                    win_rate = wins / completed_bets
                    st.metric("Win Rate", f"{win_rate:.1%}")
                else:
                    st.metric("Win Rate", "N/A")
            
            # Performance by confidence tier (if any completed bets)
            if completed_bets > 0:
                st.write("#### 🎯 Performance by Confidence Tier")
                
                tier_stats = []
                for tier in ['Elite', 'Strong', 'Good', 'Standard']:
                    tier_bets = filtered_log[
                        (filtered_log['confidence_tier'] == tier) & 
                        (filtered_log['bet_result'] != 'pending')
                    ]
                    if len(tier_bets) > 0:
                        wins = len(tier_bets[tier_bets['bet_result'] == 'win'])
                        win_rate = wins / len(tier_bets)
                        
                        # Calculate profit if bet_profit column exists and has values
                        if 'bet_profit' in tier_bets.columns:
                            tier_bets['bet_profit'] = pd.to_numeric(tier_bets['bet_profit'], errors='coerce')
                            total_profit = tier_bets['bet_profit'].sum()
                            roi = total_profit / (len(tier_bets) * 100) if len(tier_bets) > 0 else 0
                        else:
                            total_profit = 0
                            roi = 0
                        
                        tier_stats.append({
                            'Tier': tier,
                            'Bets': len(tier_bets),
                            'Wins': wins,
                            'Win Rate': f"{win_rate:.1%}",
                            'Total Profit': f"${total_profit:.2f}",
                            'ROI': f"{roi:.1%}"
                        })
                
                if tier_stats:
                    tier_df = pd.DataFrame(tier_stats)
                    st.dataframe(tier_df, hide_index=True, use_container_width=True)
            
            # Display the log
            st.write("#### 📋 Detailed Betting Log")
            st.write(f"*Showing {len(filtered_log)} bets*")
            
            # Format for display
            display_log = filtered_log.copy()
            display_log['gameday'] = display_log['gameday'].dt.strftime('%Y-%m-%d')
            display_log['log_date'] = display_log['log_date'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Select columns to display
            display_cols = [
                'log_date', 'gameday', 'week', 'home_team', 'away_team', 
                'bet_type', 'recommended_team', 'spread_line', 'model_probability',
                'edge', 'confidence_tier', 'bet_result', 'bet_profit'
            ]
            display_cols = [col for col in display_cols if col in display_log.columns]
            
            # Sort by game date (most recent first)
            display_log = display_log.sort_values('gameday', ascending=False)
            
            st.dataframe(
                display_log[display_cols],
                column_config={
                    'log_date': st.column_config.TextColumn('Logged At', width='medium'),
                    'gameday': st.column_config.TextColumn('Game Date', width='medium'),
                    'week': st.column_config.NumberColumn('Week', format='%d'),
                    'home_team': st.column_config.TextColumn('Home', width='medium'),
                    'away_team': st.column_config.TextColumn('Away', width='medium'),
                    'bet_type': st.column_config.TextColumn('Bet Type', width='medium'),
                    'recommended_team': st.column_config.TextColumn('Bet On', width='medium'),
                    'spread_line': st.column_config.NumberColumn('Spread', format='%.1f'),
                    'model_probability': st.column_config.NumberColumn('Model Prob', format='%.1%'),
                    'edge': st.column_config.NumberColumn('Edge', format='%.2%'),
                    'confidence_tier': st.column_config.TextColumn('Tier', width='small'),
                    'bet_result': st.column_config.TextColumn('Result', width='small'),
                    'bet_profit': st.column_config.NumberColumn('Profit', format='$%.2f')
                },
                height=600,
                hide_index=True
            )
            
            # Instructions for automatic updates
            st.success("""
            **✨ Automatic Updates Enabled:**
            - Scores are automatically fetched from ESPN API for completed games
            - Win/loss results and profit calculations are done automatically
            - Just refresh the app after game day to see updated statistics
            - No manual data entry required!
            
            **Note**: The system checks for completed games each time you load the app.
            """)
            
        else:
            st.info("No betting recommendations have been logged yet. Run the app when predictions are available.")
    
    else:
        st.warning("No betting log file found. The log will be created automatically when predictions with betting signals are available.")

# Create tabs for advanced model features
st.write("---")
st.write("## 🔬 Advanced Model Features")
adv_tab1, adv_tab2 = st.tabs([
    "📊 Feature Importances",
    "🎲 Monte Carlo Selection"
])

with adv_tab1:
    st.write("### Model Feature Importances and Error Metrics")
    # Try to load feature importances and metrics if saved as a CSV or JSON
    import json
    metrics_path = path.join(DATA_DIR, 'model_metrics.json')
    importances_path = path.join(DATA_DIR, 'model_feature_importances.csv')
    # Display metrics
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        st.write("#### 📊 Model Performance Metrics")
        
        # Organize metrics into a clean table format
        metrics_data = []
        
        # Extract model names and their metrics
        models = set()
        metric_model_names = {}  # Map from metrics key to display name
        for key in metrics.keys():
            if 'Spread' in key:
                models.add('Spread')
                metric_model_names['Spread'] = 'Spread'
            elif 'Moneyline' in key:
                models.add('Moneyline')
                metric_model_names['Moneyline'] = 'Moneyline'
            elif 'Totals' in key:
                models.add('Totals')
                metric_model_names['Totals'] = 'Over/Under'
        
        # Build table rows
        for model in sorted(models):
            display_name = metric_model_names.get(model, model)
            row = {'Model': display_name}
            
            # Get accuracy
            acc_key = f"{model} Accuracy"
            if acc_key in metrics:
                row['Accuracy'] = f"{metrics[acc_key]:.1%}"
            
            # Get MAE
            mae_key = f"{model} MAE"
            if mae_key in metrics:
                row['MAE'] = f"{metrics[mae_key]:.3f}"
            
            # Get threshold
            threshold_key = f"Optimal {model} Threshold"
            if threshold_key in metrics:
                row['Optimal Threshold'] = f"{metrics[threshold_key]:.1%}"
            
            metrics_data.append(row)
        
        # Display as a clean dataframe
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(
                metrics_df,
                hide_index=True,
                width=600,
                column_config={
                    'Model': st.column_config.TextColumn('Model', width='medium', help='Betting market type'),
                    'Accuracy': st.column_config.TextColumn('Accuracy', width='medium', help='Out-of-sample prediction accuracy'),
                    'MAE': st.column_config.TextColumn('MAE', width='medium', help='Mean Absolute Error (lower is better)'),
                    'Optimal Threshold': st.column_config.TextColumn('Betting Threshold', width='medium', help='Probability threshold for placing bets (F1-score optimized)')
                }
            )
            
            # Add helpful explanation
            st.info("""
            **📌 Quick Guide:**
            - **Accuracy**: How often the model correctly predicts outcomes on unseen data
            - **MAE**: Average prediction error (lower = better calibration)
            - **Betting Threshold**: Minimum probability to trigger a bet (optimized for F1-score, NOT 50%)
            
            💡 **Why thresholds aren't 50%**: These are optimized to maximize the F1-score (balance of precision and recall), 
            which produces better long-term betting results than simple 50% cutoffs.
            """)
        
    else:
        st.info("No model metrics file found. Run the model script to generate metrics.")
    
    # Display feature importances with separate tabs
    if os.path.exists(importances_path):
        importances_df = pd.read_csv(importances_path)
        
        # Create tabs for each model
        st.write("#### 🔍 Feature Importances by Model")
        feat_tab1, feat_tab2, feat_tab3 = st.tabs([
            "📈 Spread Model",
            "💰 Moneyline Model", 
            "🎯 Over/Under Model"
        ])
        
        with feat_tab1:
            st.write("### Spread Model Feature Importances (Top 25)")
            spread_features = importances_df[importances_df['model'] == 'spread'].head(25)
            if len(spread_features) > 0:
                height = get_dataframe_height(spread_features)
                st.dataframe(
                    spread_features[['feature', 'importance_mean']],
                    hide_index=True,
                    height=height,
                    width=400,
                    column_config={
                        'feature': st.column_config.TextColumn('Feature Name'),
                        'importance_mean': st.column_config.NumberColumn('Importance', format='%.4f', help='XGBoost feature importance (gain-based)')
                    }
                )
            else:
                st.warning("No spread feature importances found.")
        
        with feat_tab2:
            st.write("### Moneyline Model Feature Importances (Top 25)")
            moneyline_features = importances_df[importances_df['model'] == 'moneyline'].head(25)
            if len(moneyline_features) > 0:
                height = get_dataframe_height(moneyline_features)
                st.dataframe(
                    moneyline_features[['feature', 'importance_mean']],
                    hide_index=True,
                    height=height,
                    width=400,
                    column_config={
                        'feature': st.column_config.TextColumn('Feature Name'),
                        'importance_mean': st.column_config.NumberColumn('Importance', format='%.4f', help='XGBoost feature importance (gain-based)')
                    }
                )
            else:
                st.warning("No moneyline feature importances found.")
        
        with feat_tab3:
            st.write("### Over/Under Model Feature Importances (Top 25)")
            totals_features = importances_df[importances_df['model'] == 'totals'].head(25)
            if len(totals_features) > 0:
                height = get_dataframe_height(totals_features)
                st.dataframe(
                    totals_features[['feature', 'importance_mean']],
                    hide_index=True,
                    height=height,
                    width=400,
                    column_config={
                        'feature': st.column_config.TextColumn('Feature Name'),
                        'importance_mean': st.column_config.NumberColumn('Importance', format='%.4f', help='XGBoost feature importance (gain-based)')
                    }
                )
            else:
                st.warning("No over/under feature importances found.")
    else:
        st.info("No feature importances file found. Run the model script to generate importances.")

with adv_tab2:
    st.write("### 🎲 Monte Carlo Feature Selection")
    st.write("*Advanced feature selection using Monte Carlo simulation*")
    
    mc_sub_tab1, mc_sub_tab2, mc_sub_tab3 = st.tabs([
        "📈 Spread Model",
        "💰 Moneyline Model",
        "🎯 Totals Model"
    ])
    
    with mc_sub_tab1:
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
            st.success(f"Best mean CV accuracy: {best_score:.3f}")
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
            st.write(f"Accuracy with best features: {spread_accuracy_best:.3f}")
            st.write(f"Mean predicted probability (test set): {mean_pred_prob:.3f}")
            st.write(f"Actual outcome rate (test set): {actual_rate:.3f}")
            # Show top 10 results
            scores_df = pd.DataFrame(scores_list).sort_values(by='accuracy', ascending=False).head(10)
            st.write("#### Top 10 Feature Subsets (by Accuracy)")
            
            st.dataframe(
                scores_df,
                hide_index=True,
                column_config={
                    'iteration': st.column_config.NumberColumn('Iteration', format='%d'),
                    'accuracy': st.column_config.NumberColumn('Accuracy', format='%.3f'),
                    'AUC': st.column_config.NumberColumn('AUC', format='%.3f'),
                    'F1-score': st.column_config.NumberColumn('F1-Score', format='%.3f'),
                    'features': st.column_config.TextColumn('Features', width='large')
                }
            )

    with mc_sub_tab2:
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
            st.success(f"Best mean CV accuracy: {best_score:.3f}")
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
            st.write(f"Accuracy with best features (Moneyline): {moneyline_accuracy_best:.3f}")
            scores_df = pd.DataFrame(scores_list).sort_values(by='accuracy', ascending=False).head(10)
            st.write("#### Top 10 Feature Subsets (by Accuracy)")
            
            st.dataframe(
                scores_df,
                hide_index=True,
                column_config={
                    'iteration': st.column_config.NumberColumn('Iteration', format='%d'),
                    'accuracy': st.column_config.NumberColumn('Accuracy', format='%.3f'),
                    'AUC': st.column_config.NumberColumn('AUC', format='%.3f'),
                    'F1-score': st.column_config.NumberColumn('F1-Score', format='%.3f'),
                    'features': st.column_config.TextColumn('Features', width='large')
                }
            )

    with mc_sub_tab3:
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
            st.success(f"Best mean CV accuracy: {best_score:.3f}")
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
            st.write(f"Accuracy with best features (Totals): {totals_accuracy_best:.3f}")
            st.write(f"Mean predicted probability (test set): {mean_pred_prob_totals:.3f}")
            st.write(f"Actual outcome rate (test set): {actual_rate_totals:.3f}")
            scores_df = pd.DataFrame(scores_list).sort_values(by='accuracy', ascending=False).head(10)
            st.write("#### Top 10 Feature Subsets (by Accuracy)")
            
            st.dataframe(
                scores_df,
                hide_index=True,
                column_config={
                    'iteration': st.column_config.NumberColumn('Iteration', format='%d'),
                    'accuracy': st.column_config.NumberColumn('Accuracy', format='%.3f'),
                    'AUC': st.column_config.NumberColumn('AUC', format='%.3f'),
                    'F1-score': st.column_config.NumberColumn('F1-Score', format='%.3f'),
                    'features': st.column_config.TextColumn('Features', width='large')
                }
            )