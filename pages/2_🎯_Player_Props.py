"""
🎯 Player Props - Individual Player Predictions
Streamlit page for NFL player prop predictions (passing yards, rushing yards, TDs, receptions, etc.)
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import xgboost as xgb

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Page config
st.set_page_config(
    page_title="Player Props | NFL Predictions",
    page_icon="🎯",
    layout="wide"
)

# ============================================================================
# DATA LOADING (Lazy with Caching)
# ============================================================================

@st.cache_data(ttl=3600)
def load_xgb_models():
    """Load trained XGBoost models for player props."""
    models = {}
    models_dir = Path('player_props/models')
    
    if not models_dir.exists():
        return None
    
    # Define which models we need for DK calculator
    model_configs = {
        'passing_yards': ['elite_qb', 'star_qb', 'good_qb', 'starter'],
        'passing_tds': ['elite_qb', 'star_qb', 'good_qb', 'starter'],
        'rushing_yards': ['elite_rb', 'star_rb', 'good_rb', 'starter'],
        'rushing_tds': ['elite_rb', 'star_rb', 'good_rb', 'anytime'],
        'receiving_yards': ['star_wr', 'good_wr', 'starter'],
        'receiving_tds': ['star_wr', 'good_wr', 'anytime'],
        'receptions': ['star_wr', 'good_wr', 'starter']
    }
    
    for prop_type, tiers in model_configs.items():
        models[prop_type] = {}
        for tier in tiers:
            model_name = f"{prop_type}_{tier}"
            model_path = models_dir / f"{model_name}.json"
            
            if model_path.exists():
                try:
                    model = xgb.XGBClassifier()
                    model.load_model(str(model_path))
                    models[prop_type][tier] = model
                except Exception:
                    pass  # Silently skip failed models
    
    return models if models else None


@st.cache_data(ttl=3600)
def load_player_passing_stats():
    """Load aggregated passing stats with memory optimization."""
    file_path = Path('data_files/player_passing_stats.csv')
    if not file_path.exists():
        return None
    
    df = pd.read_csv(file_path)
    # Memory optimization
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].max() < 2147483647:
            df[col] = df[col].astype('int32')
    
    return df


@st.cache_data(ttl=3600)
def load_player_rushing_stats():
    """Load aggregated rushing stats with memory optimization."""
    file_path = Path('data_files/player_rushing_stats.csv')
    if not file_path.exists():
        return None
    
    df = pd.read_csv(file_path)
    # Memory optimization
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].max() < 2147483647:
            df[col] = df[col].astype('int32')
    
    return df


@st.cache_data(ttl=3600)
def load_player_receiving_stats():
    """Load aggregated receiving stats with memory optimization."""
    file_path = Path('data_files/player_receiving_stats.csv')
    if not file_path.exists():
        return None
    
    df = pd.read_csv(file_path)
    # Memory optimization
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].max() < 2147483647:
            df[col] = df[col].astype('int32')
    
    return df


@st.cache_data(ttl=3600)
def load_nfl_schedule():
    """Load NFL schedule for upcoming games."""
    file_path = Path('data_files/nfl_schedule_2025.csv')
    if not file_path.exists():
        return None
    return pd.read_csv(file_path)


@st.cache_data(ttl=3600)
def load_players():
    """Load player metadata with full names."""
    file_path = Path('data_files/players.csv')
    if not file_path.exists():
        return None
    return pd.read_csv(file_path)[['gsis_id', 'short_name', 'display_name', 'last_name', 'position']]


@st.cache_data(ttl=3600)
def load_player_props_predictions():
    """Load player prop predictions with memory optimization."""
    file_path = Path('data_files/player_props_predictions.csv')
    if not file_path.exists():
        return None
    
    df = pd.read_csv(file_path)
    
    # Memory optimization
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    # Parse game_date
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Add position and display_name information by joining with players data (only if not already present)
    if 'display_name' not in df.columns or 'position' not in df.columns:
        players_df = load_players()
        if players_df is not None:
            df = df.merge(
                players_df[['short_name', 'display_name', 'position']], 
                left_on='player_name', 
                right_on='short_name', 
                how='left'
            )
            # Use display_name for display, keep short_name for data operations
            if 'display_name' in df.columns:
                df['display_name'] = df['display_name'].fillna(df['player_name'])  # Fallback to short_name if no display_name
            df = df.drop('short_name', axis=1, errors='ignore')
    
@st.cache_data(ttl=3600)
def load_player_props_predictions():
    """Load player prop predictions with memory optimization."""
    file_path = Path('data_files/player_props_predictions.csv')
    if not file_path.exists():
        return None
    
    df = pd.read_csv(file_path)
    
    # Memory optimization
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    # Parse game_date
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Add position and display_name information by joining with players data (only if not already present)
    if 'display_name' not in df.columns or 'position' not in df.columns:
        players_df = load_players()
        if players_df is not None:
            df = df.merge(
                players_df[['short_name', 'display_name', 'position']], 
                left_on='player_name', 
                right_on='short_name', 
                how='left'
            )
            # Use display_name for display, keep short_name for data operations
            if 'display_name' in df.columns:
                df['display_name'] = df['display_name'].fillna(df['player_name'])  # Fallback to short_name if no display_name
            df = df.drop('short_name', axis=1, errors='ignore')
    
    # Ensure opponent_def_rank column exists (for backward compatibility)
    if 'opponent_def_rank' not in df.columns:
        st.warning("⚠️ opponent_def_rank column missing from predictions. Please regenerate predictions.")
        df['opponent_def_rank'] = 16.0  # Default to league average
    
    # Ensure injury_note column exists (for backward compatibility)
    if 'injury_note' not in df.columns:
        df['injury_note'] = None
    
    return df

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

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_upcoming_games(schedule_df):
    """Filter schedule to upcoming games only."""
    if schedule_df is None:
        return None
    
    today = pd.Timestamp.now(tz='UTC')
    
    # Handle different schedule formats (date vs commence_time)
    if 'date' in schedule_df.columns:
        date_col = pd.to_datetime(schedule_df['date'])
        # Check if already timezone-aware
        if date_col.dt.tz is None:
            schedule_df['game_date_utc'] = date_col.dt.tz_localize('UTC')
        else:
            schedule_df['game_date_utc'] = date_col.dt.tz_convert('UTC')
    elif 'commence_time' in schedule_df.columns:
        date_col = pd.to_datetime(schedule_df['commence_time'])
        if date_col.dt.tz is None:
            schedule_df['game_date_utc'] = date_col.dt.tz_localize('UTC')
        else:
            schedule_df['game_date_utc'] = date_col.dt.tz_convert('UTC')
    else:
        return None
    
    upcoming = schedule_df[schedule_df['game_date_utc'] >= today].copy()
    return upcoming.sort_values('game_date_utc')


def filter_stats_by_season(stats_df, season=2025):
    """Filter stats to specific season."""
    if stats_df is None or 'season' not in stats_df.columns:
        return stats_df
    return stats_df[stats_df['season'] == season]


def get_recent_form(stats_df, player_name, n_games=5):
    """Get player's recent game performance."""
    if stats_df is None:
        return None
    
    player_stats = stats_df[stats_df['player_name'] == player_name].sort_values('game_date', ascending=False)
    return player_stats.head(n_games)


# ============================================================================
# MAIN PAGE
# ============================================================================

def main():
    st.title("🎯 Player Props Predictions")
    st.markdown("---")
    
    # Load data
    predictions = load_player_props_predictions()
    passing_stats = load_player_passing_stats()
    rushing_stats = load_player_rushing_stats()
    receiving_stats = load_player_receiving_stats()
    schedule = load_nfl_schedule()
    players = load_players()
    
    # Check if stats are available
    stats_available = all([
        passing_stats is not None,
        rushing_stats is not None,
        receiving_stats is not None
    ])
    
    if not stats_available:
        st.warning("⚠️ Player stats not yet generated. Run the aggregation pipeline first.")
        
        with st.expander("📋 How to Generate Player Stats", expanded=True):
            st.markdown("""
            ### Generate player stats from play-by-play data:
            
            **Option 1: Run aggregation script**
            ```bash
            python player_props/aggregators.py
            ```
            
            **Option 2: Explore data first**
            ```bash
            python scripts/explore_player_stats.py
            ```
            
            This will create three CSV files in `data_files/`:
            - `player_passing_stats.csv` (~5,000 QB games)
            - `player_rushing_stats.csv` (~8,000 RB games)
            - `player_receiving_stats.csv` (~12,000 WR/TE games)
            
            Each file includes rolling averages (Last 3, Last 5, L10 games) for trend analysis.
            """)
        
        return
    
    # Upcoming games section
    upcoming = get_upcoming_games(schedule)
    
    if upcoming is not None and len(upcoming) > 0:
        st.info(f"📅 **{len(upcoming)} upcoming games** this week")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "⭐ Top Picks",
        "📊 DK Pick 6 Calculator",
        "🎯 Top QBs",
        "🏃 Top RBs", 
        "🙌 Top WRs/TEs",
        "🔍 Player Search"
    ])
    
    # ========================================================================
    # TAB 1: Top Picks (Model Predictions)
    # ========================================================================
    with tab1:
        st.subheader("Top Player Prop Recommendations")
        
        if predictions is not None and not predictions.empty:
            # Confidence tiers
            col1, col2, col3 = st.columns(3)
            
            with col1:
                elite_count = len(predictions[predictions['confidence'] >= 0.65])
                st.metric("🔥 Elite (≥65%)", elite_count)
            
            with col2:
                strong_count = len(predictions[(predictions['confidence'] >= 0.60) & (predictions['confidence'] < 0.65)])
                st.metric("💪 Strong (60-65%)", strong_count)
            
            with col3:
                good_count = len(predictions[(predictions['confidence'] >= 0.55) & (predictions['confidence'] < 0.60)])
                st.metric("✅ Good (55-60%)", good_count)
            
            st.markdown("---")
            
            # Filters
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                min_conf = st.slider("Minimum Confidence", 0.50, 0.90, 0.60, 0.05, key="min_conf_top")
            
            with col2:
                prop_types = st.multiselect(
                    "Prop Types",
                    options=predictions['prop_type'].unique(),
                    default=predictions['prop_type'].unique(),
                    key="prop_filter_top"
                )
            
            with col3:
                injury_filter = st.selectbox(
                    "Injury Status",
                    ["All Players", "Healthy Only", "Injured Only"],
                    key="injury_filter_top"
                )
            
            with col4:
                sort_option = st.selectbox(
                    "Sort By",
                    ["Confidence", "Player Name", "Avg L3"],
                    key="sort_top"
                )
            
            # Filter predictions
            filtered = predictions[
                (predictions['confidence'] >= min_conf) &
                (predictions['prop_type'].isin(prop_types))
            ].copy()
            
            # Apply injury filter
            if injury_filter == "Healthy Only":
                filtered = filtered[filtered['injury_note'].isna() | (filtered['injury_note'] == '')]
            elif injury_filter == "Injured Only":
                filtered = filtered[filtered['injury_note'].notna() & (filtered['injury_note'] != '')]
            
            # Sort
            if sort_option == "Confidence":
                filtered = filtered.sort_values('confidence', ascending=False)
            elif sort_option == "Player Name":
                filtered = filtered.sort_values('display_name')
            else:
                filtered = filtered.sort_values('avg_L3', ascending=False)
            
            # Display
            if not filtered.empty:
                # Format display columns
                display_df = filtered.copy()
                display_df['Confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
                display_df['Recommendation'] = display_df.apply(
                    lambda row: f"{row['recommendation']} {row['line_value']:.1f}", axis=1
                )
                display_df['Tier'] = display_df['confidence'].apply(
                    lambda x: '🔥 Elite' if x >= 0.65 else ('💪 Strong' if x >= 0.60 else '✅ Good')
                )
                display_df['Last 3 Avg'] = display_df['avg_L3'].apply(lambda x: f"{x:.1f}")
                display_df['Last 5 Avg'] = display_df['avg_L5'].apply(lambda x: f"{x:.1f}")
                display_df['Last 10 Avg'] = display_df['avg_L10'].apply(lambda x: f"{x:.1f}")
                display_df['Defense Rank'] = display_df['opponent_def_rank'].apply(
                    lambda x: f"#{int(x)}/32" + (" 🛡️" if x <= 8 else (" ⚠️" if x >= 24 else ""))
                )
                
                # Add injury information if available
                if 'injury_note' in display_df.columns:
                    display_df['Injury Status'] = display_df['injury_note'].fillna('')
                else:
                    display_df['Injury Status'] = ''
                
                # Select columns to show
                show_cols = [
                    'display_name', 'position', 'team', 'opponent', 'Defense Rank', 'trend', 'prop_type', 
                    'Recommendation', 'Confidence', 'Tier', 'Last 3 Avg', 'Last 5 Avg', 'Last 10 Avg', 'weather_conditions', 'Injury Status'
                ]
                
                height = get_dataframe_height(display_df[show_cols])

                st.dataframe(
                    display_df[show_cols].rename(columns={
                        'display_name': 'Player',
                        'position': 'Pos',
                        'team': 'Team',
                        'opponent': 'Opp',
                        'trend': 'Trend',
                        'prop_type': 'Prop Type',
                        'weather_conditions': 'Weather'
                    }),
                    width='stretch',
                    height=height,
                    hide_index=True
                )
                
                injured_count = len(filtered[filtered['injury_note'].notna() & (filtered['injury_note'] != '')]) if 'injury_note' in filtered.columns else 0
                filter_desc = f" ({injury_filter})" if injury_filter != "All Players" else ""
                st.caption(f"Showing {len(filtered):,} predictions{filter_desc} • {injured_count} with injury adjustments")
                
                # Explanation
                with st.expander("ℹ️ How to Read These Predictions"):
                    st.markdown("""
                    **Confidence Tiers:**
                    - 🔥 **Elite (≥65%)**: Highest confidence predictions
                    - 💪 **Strong (60-65%)**: Very good predictions  
                    - ✅ **Good (55-60%)**: Solid predictions above breakeven
                    
                    **Recommendation Format:**
                    - `OVER 225.5` means the model predicts the player will go OVER this yardage line
                    - `UNDER 65.5` means the model predicts the player will go UNDER this yardage line
                    
                    **Recent Averages:**
                    - **Last 3/Last 5 Avg**: Player's average performance over Last 3/5 games
                    - Compare to the line value to gauge difficulty
                    
                    **Weather Impact:**
                    - **Weather**: Shows weather conditions for outdoor games (temperature, wind, precipitation)
                    - Weather adjustments are automatically applied to predictions for outdoor stadiums
                    - Dome games show "Dome" with no weather impact
                    
                    **Model Info:**
                    - Trained on 2020-2025 historical data
                    - Uses rolling averages, TDs, completions, attempts as features
                    - ROC-AUC scores: Passing 0.72, Rushing 0.84, Receiving 0.81
                    """)
            else:
                st.info("No predictions match the selected filters")
        
        else:
            st.warning("⚠️ No predictions available. Generate predictions first.")
            
            with st.expander("📋 How to Generate Predictions", expanded=True):
                st.markdown("""
                Run the prediction pipeline to generate prop picks for upcoming games:
                
                ```bash
                python player_props/predict.py
                ```
                
                This will:
                1. Load upcoming games from the schedule
                2. Get each player's recent performance (Last 3, Last 5, L10 games)
                3. Run XGBoost models to predict Over/Under probabilities
                4. Save predictions to `player_props_predictions.csv`
                
                Then refresh this page to see the picks!
                """)
    
    # ========================================================================
    # TAB 2: DraftKings Pick 6 Line Calculator
    # ========================================================================
    with tab2:
        st.subheader("📊 DraftKings Pick 6 - Line Comparison")
        st.markdown("Enter a player and their DraftKings Pick 6 line to get the model's prediction")
        
        # Load models
        xgb_models = load_xgb_models()
        
        # Check if we have the necessary data
        if passing_stats is None or rushing_stats is None or receiving_stats is None:
            st.warning("⚠️ Player stats not available. Run aggregation pipeline first.")
            return
        
        # Create two columns for inputs
        input_col1, input_col2 = st.columns([2, 1])
        
        with input_col1:
            # Player selection with full name search
            if players is not None:
                player_search = st.text_input(
                    "🔍 Search Player", 
                    placeholder="Type player name (e.g., Josh Allen, Patrick Mahomes)",
                    key="dk_player_search"
                )
                
                if player_search:
                    # Search in display_name and last_name
                    search_lower = player_search.lower()
                    matches = players[
                        players['display_name'].str.lower().str.contains(search_lower, na=False) |
                        players['last_name'].str.lower().str.contains(search_lower, na=False)
                    ].copy()
                    
                    if not matches.empty:
                        # Sort by relevance (exact matches first)
                        matches['exact_match'] = matches['display_name'].str.lower() == search_lower
                        matches = matches.sort_values(['exact_match', 'display_name'], ascending=[False, True])
                        
                        # Create display options with team info
                        player_options = {}
                        for _, row in matches.head(20).iterrows():
                            display = f"{row['display_name']} - {row['position']}"
                            player_options[display] = row['gsis_id']
                        
                        selected_display = st.selectbox(
                            "Select Player",
                            options=list(player_options.keys()),
                            key="dk_player_select"
                        )
                        selected_player_id = player_options[selected_display]
                        selected_player_name = matches[matches['gsis_id'] == selected_player_id].iloc[0]['short_name']
                        selected_player_full = matches[matches['gsis_id'] == selected_player_id].iloc[0]['display_name']
                        selected_position = matches[matches['gsis_id'] == selected_player_id].iloc[0]['position']
                    else:
                        st.info("No players found. Try a different search.")
                        selected_player_id = None
                        selected_player_name = None
                        selected_player_full = None
                        selected_position = None
                else:
                    selected_player_id = None
                    selected_player_name = None
                    selected_player_full = None
                    selected_position = None
            else:
                st.error("Player data not loaded")
                selected_player_id = None
                selected_player_name = None
                selected_player_full = None
                selected_position = None
        
        with input_col2:
            # Stat category selection based on position
            if selected_position:
                if selected_position == 'QB':
                    stat_options = ['Passing Yards', 'Passing TDs']
                elif selected_position in ['RB', 'FB']:
                    stat_options = ['Rushing Yards', 'Rushing TDs', 'Receiving Yards', 'Receiving TDs']
                elif selected_position in ['WR', 'TE']:
                    stat_options = ['Receiving Yards', 'Receiving TDs', 'Receptions']
                else:
                    stat_options = ['Passing Yards', 'Rushing Yards', 'Receiving Yards']
                
                stat_category = st.selectbox(
                    "Stat Category",
                    options=stat_options,
                    key="dk_stat_category"
                )
            else:
                stat_category = None
        
        # Line input
        if selected_player_id and stat_category:
            dk_line = st.number_input(
                f"DraftKings Pick 6 Line for {stat_category}",
                min_value=0.5,
                max_value=500.0,
                value=100.5,
                step=0.5,
                key="dk_line_value",
                help="Enter the exact line from DraftKings Pick 6 (e.g., 100.5, 225.5)"
            )
            
            # Get player stats based on category
            if stat_category in ['Passing Yards', 'Passing TDs']:
                player_stats_df = passing_stats[passing_stats['player_id'] == selected_player_id].copy()
                if stat_category == 'Passing Yards':
                    stat_col = 'passing_yards'
                    td_col = 'pass_tds'
                else:
                    stat_col = 'pass_tds'
                    td_col = None
            elif stat_category in ['Rushing Yards', 'Rushing TDs']:
                player_stats_df = rushing_stats[rushing_stats['player_id'] == selected_player_id].copy()
                if stat_category == 'Rushing Yards':
                    stat_col = 'rushing_yards'
                    td_col = 'rush_tds'
                else:
                    stat_col = 'rush_tds'
                    td_col = None
            else:  # Receiving stats
                player_stats_df = receiving_stats[receiving_stats['player_id'] == selected_player_id].copy()
                if stat_category == 'Receiving Yards':
                    stat_col = 'receiving_yards'
                    td_col = 'rec_tds'
                elif stat_category == 'Receptions':
                    stat_col = 'receptions'
                    td_col = 'rec_tds'
                else:
                    stat_col = 'rec_tds'
                    td_col = None
            
            # Filter to 2025 season
            if not player_stats_df.empty and 'season' in player_stats_df.columns:
                player_stats_df = player_stats_df[player_stats_df['season'] == 2025].sort_values('week', ascending=False)
            
            if not player_stats_df.empty:
                st.markdown("---")
                
                # Calculate statistics
                recent_games = player_stats_df.head(10)
                last_3_avg = recent_games.head(3)[stat_col].mean()
                last_5_avg = recent_games.head(5)[stat_col].mean()
                last_10_avg = recent_games.head(10)[stat_col].mean()
                season_avg = player_stats_df[stat_col].mean()
                
                # === MODEL-BASED PREDICTION ===
                model_prob_over = None
                model_confidence = None
                
                if xgb_models:
                    prop_type = stat_category.replace(' ', '_').lower()
                    
                    # Select appropriate model tier based on season average
                    tier = None
                    if prop_type in ['passing_yards', 'passing_tds']:
                        if season_avg >= 280:
                            tier = 'elite_qb'
                        elif season_avg >= 250:
                            tier = 'star_qb'
                        elif season_avg >= 220:
                            tier = 'good_qb'
                        else:
                            tier = 'starter'
                    elif prop_type in ['rushing_yards', 'rushing_tds']:
                        if season_avg >= 80:
                            tier = 'elite_rb'
                        elif season_avg >= 60:
                            tier = 'star_rb'
                        elif season_avg >= 45:
                            tier = 'good_rb'
                        else:
                            tier = 'starter' if prop_type == 'rushing_yards' else 'anytime'
                    else:  # receiving
                        if season_avg >= 75:
                            tier = 'star_wr'
                        elif season_avg >= 55:
                            tier = 'star_wr'
                        elif season_avg >= 40:
                            tier = 'good_wr'
                        else:
                            tier = 'starter' if prop_type != 'receiving_tds' else 'anytime'
                    
                    # Get the model
                    if tier and prop_type in xgb_models and tier in xgb_models[prop_type]:
                        model = xgb_models[prop_type][tier]
                        
                        # Build features
                        features = {}
                        
                        # Core rolling features
                        features[f'{stat_col}_L3'] = last_3_avg
                        features[f'{stat_col}_L5'] = last_5_avg
                        features[f'{stat_col}_L10'] = last_10_avg
                        
                        # Add auxiliary features based on stat type
                        if prop_type in ['passing_yards', 'passing_tds']:
                            features['pass_tds_L3'] = recent_games.head(3)['pass_tds'].mean() if 'pass_tds' in recent_games.columns else 0
                            features['pass_tds_L5'] = recent_games.head(5)['pass_tds'].mean() if 'pass_tds' in recent_games.columns else 0
                            features['completions_L3'] = recent_games.head(3)['completions'].mean() if 'completions' in recent_games.columns else 0
                            features['completions_L5'] = recent_games.head(5)['completions'].mean() if 'completions' in recent_games.columns else 0
                            features['attempts_L3'] = recent_games.head(3)['attempts'].mean() if 'attempts' in recent_games.columns else 0
                            features['attempts_L5'] = recent_games.head(5)['attempts'].mean() if 'attempts' in recent_games.columns else 0
                            if prop_type == 'passing_tds':
                                features['pass_tds_L10'] = last_10_avg
                                features['completions_L10'] = recent_games.head(10)['completions'].mean() if 'completions' in recent_games.columns else 0
                                features['attempts_L10'] = recent_games.head(10)['attempts'].mean() if 'attempts' in recent_games.columns else 0
                        
                        elif prop_type in ['rushing_yards', 'rushing_tds']:
                            features['rush_tds_L3'] = recent_games.head(3)['rush_tds'].mean() if 'rush_tds' in recent_games.columns else 0
                            features['rush_tds_L5'] = recent_games.head(5)['rush_tds'].mean() if 'rush_tds' in recent_games.columns else 0
                            features['rush_attempts_L3'] = recent_games.head(3)['rush_attempts'].mean() if 'rush_attempts' in recent_games.columns else 0
                            features['rush_attempts_L5'] = recent_games.head(5)['rush_attempts'].mean() if 'rush_attempts' in recent_games.columns else 0
                            if prop_type == 'rushing_tds':
                                features['rush_tds_L10'] = last_10_avg
                                features['rush_attempts_L10'] = recent_games.head(10)['rush_attempts'].mean() if 'rush_attempts' in recent_games.columns else 0
                        
                        else:  # receiving
                            features['rec_tds_L3'] = recent_games.head(3)['rec_tds'].mean() if 'rec_tds' in recent_games.columns else 0
                            features['rec_tds_L5'] = recent_games.head(5)['rec_tds'].mean() if 'rec_tds' in recent_games.columns else 0
                            features['receptions_L3'] = recent_games.head(3)['receptions'].mean() if 'receptions' in recent_games.columns else 0
                            features['receptions_L5'] = recent_games.head(5)['receptions'].mean() if 'receptions' in recent_games.columns else 0
                            features['targets_L3'] = recent_games.head(3)['targets'].mean() if 'targets' in recent_games.columns else 0
                            features['targets_L5'] = recent_games.head(5)['targets'].mean() if 'targets' in recent_games.columns else 0
                            if prop_type == 'receiving_tds':
                                features['rec_tds_L10'] = last_10_avg
                                features['receptions_L10'] = recent_games.head(10)['receptions'].mean() if 'receptions' in recent_games.columns else 0
                                features['targets_L10'] = recent_games.head(10)['targets'].mean() if 'targets' in recent_games.columns else 0
                        
                        # Add matchup features (defaults)
                        features['opponent_def_rank'] = 16.0
                        features['is_home'] = 1
                        features['days_rest'] = 7
                        
                        # Create feature DataFrame and predict
                        try:
                            feature_df = pd.DataFrame([features])
                            model_prob_over = model.predict_proba(feature_df)[0][1]
                        except Exception:
                            model_prob_over = None
                
                # === HISTORICAL HIT RATE (Fallback) ===
                total_games = len(recent_games)
                games_over = len(recent_games[recent_games[stat_col] > dk_line])
                games_under = total_games - games_over
                hist_prob_over = (games_over + 1) / (total_games + 2)
                
                # Use model prediction if available
                if model_prob_over is not None:
                    prob_over = model_prob_over
                    prob_under = 1 - prob_over
                    prediction_source = "🤖 Machine Learning Model"
                else:
                    prob_over = hist_prob_over
                    prob_under = 1 - prob_over
                    prediction_source = "📊 Historical"
                
                # Determine recommendation
                if prob_over >= 0.60:
                    recommendation = "OVER"
                    confidence = prob_over
                    tier = "🔥 ELITE" if prob_over >= 0.65 else "💪 STRONG"
                elif prob_under >= 0.60:
                    recommendation = "UNDER"
                    confidence = prob_under
                    tier = "🔥 ELITE" if prob_under >= 0.65 else "💪 STRONG"
                else:
                    recommendation = "OVER" if prob_over > prob_under else "UNDER"
                    confidence = max(prob_over, prob_under)
                    tier = "✅ GOOD" if confidence >= 0.55 else "⚠️ LEAN"
                
                # Display results in a prominent card
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 2rem; border-radius: 15px; color: white; margin: 1rem 0;">
                    <h2 style="margin: 0; font-size: 1.8rem;">{selected_player_full}</h2>
                    <p style="margin: 0.5rem 0; font-size: 1.1rem; opacity: 0.9;">{stat_category}</p>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1.5rem;">
                        <div>
                            <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">DraftKings Line</p>
                            <p style="margin: 0; font-size: 2.5rem; font-weight: bold;">{dk_line}</p>
                        </div>
                        <div style="text-align: right;">
                            <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">Recommendation</p>
                            <p style="margin: 0; font-size: 2.5rem; font-weight: bold;">{recommendation}</p>
                            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">{tier}</p>
                        </div>
                    </div>
                    <div style="margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.2);">
                        <p style="margin: 0; font-size: 1rem;">Confidence: <strong>{confidence:.1%}</strong> ({prediction_source})</p>
                        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.8;">
                            Historical: {total_games} games ({games_over} over, {games_under} under)
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed statistics
                st.markdown("### 📈 Performance Analysis")
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    delta_3 = last_3_avg - dk_line
                    st.metric(
                        "Last 3 Games Avg",
                        f"{last_3_avg:.1f}",
                        f"{delta_3:+.1f} vs line",
                        delta_color="normal"
                    )
                
                with metric_col2:
                    delta_5 = last_5_avg - dk_line
                    st.metric(
                        "Last 5 Games Avg",
                        f"{last_5_avg:.1f}",
                        f"{delta_5:+.1f} vs line",
                        delta_color="normal"
                    )
                
                with metric_col3:
                    delta_10 = last_10_avg - dk_line
                    st.metric(
                        "Last 10 Games Avg",
                        f"{last_10_avg:.1f}",
                        f"{delta_10:+.1f} vs line",
                        delta_color="normal"
                    )
                
                with metric_col4:
                    delta_season = season_avg - dk_line
                    st.metric(
                        "Season Average",
                        f"{season_avg:.1f}",
                        f"{delta_season:+.1f} vs line",
                        delta_color="normal"
                    )
                
                # Game log
                st.markdown("### 📊 Recent Game Log")
                
                display_cols = ['week', 'opponent', stat_col]
                if td_col and td_col in recent_games.columns:
                    display_cols.append(td_col)
                
                game_log = recent_games[display_cols].copy()
                game_log['Result'] = game_log[stat_col].apply(
                    lambda x: f"{'✅ OVER' if x > dk_line else '❌ UNDER'} ({x:.1f})"
                )
                
                rename_map = {
                    'week': 'Week',
                    'opponent': 'Opponent',
                    stat_col: stat_category
                }
                if td_col:
                    rename_map[td_col] = 'TDs'
                
                st.dataframe(
                    game_log.rename(columns=rename_map),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Hit rate by game count
                st.markdown("### 🎯 Hit Rate Analysis")
                
                hit_col1, hit_col2, hit_col3 = st.columns(3)
                
                with hit_col1:
                    last_3 = recent_games.head(3)
                    over_3 = len(last_3[last_3[stat_col] > dk_line])
                    rate_3 = over_3 / 3 * 100 if len(last_3) >= 3 else 0
                    st.metric("Last 3 Games", f"{over_3}/3 Over", f"{rate_3:.0f}% hit rate")
                
                with hit_col2:
                    last_5 = recent_games.head(5)
                    over_5 = len(last_5[last_5[stat_col] > dk_line])
                    rate_5 = over_5 / 5 * 100 if len(last_5) >= 5 else 0
                    st.metric("Last 5 Games", f"{over_5}/5 Over", f"{rate_5:.0f}% hit rate")
                
                with hit_col3:
                    over_10 = len(recent_games[recent_games[stat_col] > dk_line])
                    rate_10 = over_10 / len(recent_games) * 100 if len(recent_games) > 0 else 0
                    st.metric(f"Last {len(recent_games)} Games", f"{over_10}/{len(recent_games)} Over", f"{rate_10:.0f}% hit rate")
                
                # Explanation
                with st.expander("ℹ️ How This Works"):
                    if model_prob_over is not None:
                        st.markdown(f"""
                        **🤖 Machine Learning Model Prediction Method:**
                        - Uses trained XGBoost model (tier: {tier})
                        - Season average: {season_avg:.1f} → model tier selection
                        - Analyzes rolling stats (L3, L5, L10 games)
                        - Considers TDs, attempts/targets, matchup factors
                        - Trained on 2020-2024 data (ROC-AUC: 0.72-0.84)
                        
                        **Prediction Comparison:**
                        - **Machine Learning Model**: {model_prob_over:.1%} probability OVER
                        - **Historical**: {hist_prob_over:.1%} hit rate ({games_over}/{total_games} games)
                        - **Recommendation**: {recommendation} with {confidence:.1%} confidence
                        """)
                    else:
                        st.markdown(f"""
                        **📊 Historical Analysis Method:**
                        - Analyzes {selected_player_full}'s {total_games} games in 2025
                        - Compares actual performance to line ({dk_line})
                        - Uses Laplace smoothing for reliability
                        - Machine Learning model not available for this stat/tier
                        """)
                    
                    st.markdown("""
                    **Confidence Tiers:**
                    - 🔥 **ELITE (≥65%)**: Highest confidence picks
                    - 💪 **STRONG (60-65%)**: Very confident picks  
                    - ✅ **GOOD (55-60%)**: Solid picks above breakeven
                    - ⚠️ **LEAN (<55%)**: Lower confidence
                    
                    **Pro Tips:**
                    - Model auto-selects tier based on season average
                    - Recent form (L3/L5) weighted heavily
                    - Check opponent defense rank for full analysis
                    - Weather/injuries not included in quick calculator
                    """)
            else:
                st.info(f"No 2025 season data available for {selected_player_full}")
        
        else:
            st.info("👆 Search for a player above to get started")
            
            # Show example
            with st.expander("📖 How to Use This Tool"):
                st.markdown("""
                ### Step-by-Step Guide:
                
                1. **Search for a Player**: Type the player's name in the search box
                2. **Select from Results**: Choose the correct player from the dropdown
                3. **Pick a Stat Category**: The tool auto-suggests categories based on position
                4. **Enter DraftKings Line**: Input the exact over/under line from DK Pick 6
                5. **Get Recommendation**: See the model's OVER/UNDER prediction with confidence
                
                ### Example:
                - Player: **Josh Allen**
                - Category: **Passing Yards**
                - DK Line: **242.5**
                - Model says: **OVER 242.5** (💪 STRONG - 63% confidence)
                
                The model analyzes recent performance, season averages, and hit rates to give you data-driven recommendations!
                """)
    
    # ========================================================================
    # TAB 3: Top QBs (Season Leaders)
    # ========================================================================
    with tab3:
        st.subheader("🏆 2025 Season Leaders - Quarterbacks")
        
        if passing_stats is not None:
            # Filter to 2025 season
            season_stats = passing_stats[passing_stats['season'] == 2025].copy()
            
            if not season_stats.empty:
                # Aggregate to season totals per player
                season_totals = season_stats.groupby(['player_name', 'team'], observed=True).agg({
                    'passing_yards': 'sum',
                    'pass_tds': 'sum',
                    'completions': 'sum',
                    'attempts': 'sum',
                    'interceptions': 'sum',
                    'game_id': 'count'  # Games played
                }).reset_index()
                
                season_totals.rename(columns={'game_id': 'games_played'}, inplace=True)
                
                # Calculate per-game averages
                season_totals['yards_per_game'] = season_totals['passing_yards'] / season_totals['games_played']
                season_totals['completion_pct'] = (season_totals['completions'] / season_totals['attempts'] * 100).round(1)
                
                # Add position information
                players_df = load_players()
                if players_df is not None:
                    season_totals = season_totals.merge(
                        players_df[['short_name', 'position']], 
                        left_on='player_name', 
                        right_on='short_name', 
                        how='left'
                    )
                    season_totals = season_totals.drop('short_name', axis=1)
                
                # Sort by total yards
                season_totals = season_totals.sort_values('passing_yards', ascending=False).head(30)
                
                # Create display dataframe
                display_df = season_totals[[
                    'player_name', 'position', 'team', 'games_played', 'passing_yards', 'yards_per_game',
                    'pass_tds', 'interceptions', 'completions', 'attempts', 'completion_pct'
                ]].copy()
                
                # Rename columns
                display_df.columns = [
                    'Player', 'Pos', 'Team', 'GP', 'Total Yds', 'Yds/G', 
                    'TDs', 'INT', 'Comp', 'Att', 'Comp%'
                ]
                
                # Format numbers
                display_df['Total Yds'] = display_df['Total Yds'].astype(int)
                display_df['Yds/G'] = display_df['Yds/G'].round(1)
                
                st.dataframe(
                    display_df,
                    width='stretch',
                    height=600,
                    hide_index=True
                )
                
                st.caption(f"Top 30 QBs by total passing yards (2025 season)")
            else:
                st.info("No 2025 season data available yet")
        else:
            st.error("Passing stats not loaded")
    
    # ========================================================================
    # TAB 4: Top RBs (Season Leaders)
    # ========================================================================
    with tab4:
        st.subheader("🏆 2025 Season Leaders - Running Backs")
        
        if rushing_stats is not None:
            # Filter to 2025 season
            season_stats = rushing_stats[rushing_stats['season'] == 2025].copy()
            
            if not season_stats.empty:
                # Aggregate to season totals per player
                season_totals = season_stats.groupby(['player_name', 'team'], observed=True).agg({
                    'rushing_yards': 'sum',
                    'rush_tds': 'sum',
                    'rush_attempts': 'sum',
                    'game_id': 'count'
                }).reset_index()
                
                season_totals.rename(columns={'game_id': 'games_played'}, inplace=True)
                
                # Calculate per-game and efficiency stats
                season_totals['yards_per_game'] = season_totals['rushing_yards'] / season_totals['games_played']
                season_totals['yards_per_carry'] = (season_totals['rushing_yards'] / season_totals['rush_attempts']).round(1)
                season_totals['att_per_game'] = (season_totals['rush_attempts'] / season_totals['games_played']).round(1)
                
                # Add position information
                players_df = load_players()
                if players_df is not None:
                    season_totals = season_totals.merge(
                        players_df[['short_name', 'position']], 
                        left_on='player_name', 
                        right_on='short_name', 
                        how='left'
                    )
                    season_totals = season_totals.drop('short_name', axis=1)
                
                # Sort by total yards
                season_totals = season_totals.sort_values('rushing_yards', ascending=False).head(30)
                
                # Create display dataframe
                display_df = season_totals[[
                    'player_name', 'position', 'team', 'games_played', 'rushing_yards', 'yards_per_game',
                    'rush_tds', 'rush_attempts', 'yards_per_carry', 'att_per_game'
                ]].copy()
                
                # Rename columns
                display_df.columns = [
                    'Player', 'Pos', 'Team', 'GP', 'Total Yds', 'Yds/G',
                    'TDs', 'Attempts', 'YPC', 'Att/G'
                ]
                
                # Format numbers
                display_df['Total Yds'] = display_df['Total Yds'].astype(int)
                display_df['Yds/G'] = display_df['Yds/G'].round(1)
                
                st.dataframe(
                    display_df,
                    width='stretch',
                    height=600,
                    hide_index=True
                )
                
                st.caption(f"Top 30 RBs by total rushing yards (2025 season)")
            else:
                st.info("No 2025 season data available yet")
        else:
            st.error("Rushing stats not loaded")
    
    # ========================================================================
    # TAB 5: Top WRs/TEs (Season Leaders)
    # ========================================================================
    with tab5:
        st.subheader("🏆 2025 Season Leaders - Wide Receivers & Tight Ends")
        
        if receiving_stats is not None:
            # Filter to 2025 season
            season_stats = receiving_stats[receiving_stats['season'] == 2025].copy()
            
            if not season_stats.empty:
                # Aggregate to season totals per player
                season_totals = season_stats.groupby(['player_name', 'team'], observed=True).agg({
                    'receiving_yards': 'sum',
                    'receptions': 'sum',
                    'rec_tds': 'sum',
                    'targets': 'sum',
                    'game_id': 'count'
                }).reset_index()
                
                season_totals.rename(columns={'game_id': 'games_played'}, inplace=True)
                
                # Calculate per-game and efficiency stats
                season_totals['yards_per_game'] = season_totals['receiving_yards'] / season_totals['games_played']
                season_totals['rec_per_game'] = (season_totals['receptions'] / season_totals['games_played']).round(1)
                season_totals['yards_per_rec'] = (season_totals['receiving_yards'] / season_totals['receptions']).round(1)
                season_totals['catch_rate'] = (season_totals['receptions'] / season_totals['targets'] * 100).round(1)
                
                # Add position information
                players_df = load_players()
                if players_df is not None:
                    season_totals = season_totals.merge(
                        players_df[['short_name', 'position']], 
                        left_on='player_name', 
                        right_on='short_name', 
                        how='left'
                    )
                    season_totals = season_totals.drop('short_name', axis=1)
                
                # Sort by total yards
                season_totals = season_totals.sort_values('receiving_yards', ascending=False).head(40)
                
                # Create display dataframe
                display_df = season_totals[[
                    'player_name', 'position', 'team', 'games_played', 'receiving_yards', 'yards_per_game',
                    'receptions', 'rec_tds', 'targets', 'yards_per_rec', 'catch_rate'
                ]].copy()
                
                # Rename columns
                display_df.columns = [
                    'Player', 'Pos', 'Team', 'GP', 'Total Yds', 'Yds/G',
                    'Rec', 'TDs', 'Targets', 'YPR', 'Catch%'
                ]
                
                # Format numbers
                display_df['Total Yds'] = display_df['Total Yds'].astype(int)
                display_df['Yds/G'] = display_df['Yds/G'].round(1)
                
                st.dataframe(
                    display_df,
                    width='stretch',
                    height=600,
                    hide_index=True
                )
                
                st.caption(f"Top 40 WRs/TEs by total receiving yards (2025 season)")
            else:
                st.info("No 2025 season data available yet")
        else:
            st.error("Receiving stats not loaded")
    
    # ========================================================================
    # TAB 6: Player Search
    # ========================================================================
    with tab6:
        st.subheader("Search Individual Player Stats")
        
        # Get unique player IDs from stats
        all_player_ids = []
        if passing_stats is not None:
            all_player_ids.extend(passing_stats['player_id'].unique())
        if rushing_stats is not None:
            all_player_ids.extend(rushing_stats['player_id'].unique())
        if receiving_stats is not None:
            all_player_ids.extend(receiving_stats['player_id'].unique())
        
        all_player_ids = list(set(all_player_ids))
        
        # Build player options with full names
        if players is not None and len(all_player_ids) > 0:
            player_options = []
            for player_id in all_player_ids:
                match = players[players['gsis_id'] == player_id]
                if not match.empty:
                    display_name = match.iloc[0]['display_name']
                    last_name = match.iloc[0]['last_name']
                    # Only add if we found a valid display name
                    if pd.notna(display_name) and display_name:
                        player_options.append((display_name, player_id, last_name))
            
            # Sort by last name
            player_options = sorted(player_options, key=lambda x: x[2] if pd.notna(x[2]) else '')
            display_names = [opt[0] for opt in player_options]
            player_ids = [opt[1] for opt in player_options]
        else:
            display_names = []
            player_ids = []
        
        if display_names:
            # Add search filter
            search_term = st.text_input("Search for a player (type to filter)", "", key="player_search_filter")
            
            # Filter players based on search term
            if search_term:
                filtered_indices = [i for i, name in enumerate(display_names) 
                                   if search_term.lower() in name.lower()]
                filtered_display_names = [display_names[i] for i in filtered_indices]
                filtered_player_ids = [player_ids[i] for i in filtered_indices]
            else:
                filtered_display_names = display_names
                filtered_player_ids = player_ids
            
            if filtered_display_names:
                selected_display = st.selectbox("Select Player", filtered_display_names, key="player_search")
                
                # Get the corresponding player_id
                selected_index = filtered_display_names.index(selected_display)
                selected_player_id = filtered_player_ids[selected_index]
            else:
                st.info("No players match your search")
                selected_player_id = None
            
            if selected_player_id:
                # Get player position
                player_info = players[players['gsis_id'] == selected_player_id]
                if not player_info.empty:
                    player_position = player_info.iloc[0]['position']
                    st.markdown(f"**Position:** {player_position}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Passing Stats**")
                    if passing_stats is not None:
                        player_pass = passing_stats[passing_stats['player_id'] == selected_player_id]
                        if not player_pass.empty:
                            # Filter to 2025 season only
                            if 'season' in player_pass.columns:
                                player_pass = player_pass[player_pass['season'] == 2025]
                            display_cols = ['week', 'opponent', 'passing_yards', 'pass_tds', 'completions', 'attempts', 'interceptions']
                            available = [c for c in display_cols if c in player_pass.columns]
                            st.dataframe(
                                player_pass[available].rename(columns={
                                    'week': 'Wk',
                                    'opponent': 'Opp',
                                    'passing_yards': 'Yds',
                                    'pass_tds': 'TDs',
                                    'completions': 'Comp',
                                    'attempts': 'Att',
                                    'interceptions': 'INT'
                                }),
                                width='stretch',
                                hide_index=True
                            )
                        else:
                            st.info("No passing stats")
                    else:
                        st.info("Data not loaded")
                
                with col2:
                    st.markdown("**Rushing Stats**")
                    if rushing_stats is not None:
                        player_rush = rushing_stats[rushing_stats['player_id'] == selected_player_id]
                        if not player_rush.empty:
                            # Filter to 2025 season only
                            if 'season' in player_rush.columns:
                                player_rush = player_rush[player_rush['season'] == 2025]
                            display_cols = ['week', 'opponent', 'rushing_yards', 'rush_tds', 'rush_attempts']
                            available = [c for c in display_cols if c in player_rush.columns]
                            st.dataframe(
                                player_rush[available].rename(columns={
                                    'week': 'Wk',
                                    'opponent': 'Opp',
                                    'rushing_yards': 'Yds',
                                    'rush_tds': 'TDs',
                                    'rush_attempts': 'Att'
                                }),
                                width='stretch',
                                hide_index=True
                            )
                        else:
                            st.info("No rushing stats")
                    else:
                        st.info("Data not loaded")
                
                with col3:
                    st.markdown("**Receiving Stats**")
                    if receiving_stats is not None:
                        player_rec = receiving_stats[receiving_stats['player_id'] == selected_player_id]
                        if not player_rec.empty:
                            # Filter to 2025 season only
                            if 'season' in player_rec.columns:
                                player_rec = player_rec[player_rec['season'] == 2025]
                            display_cols = ['week', 'opponent', 'receiving_yards', 'receptions', 'rec_tds', 'targets']
                            available = [c for c in display_cols if c in player_rec.columns]
                            st.dataframe(
                                player_rec[available].rename(columns={
                                    'week': 'Wk',
                                    'opponent': 'Opp',
                                    'receiving_yards': 'Yds',
                                    'receptions': 'Rec',
                                    'rec_tds': 'TDs',
                                    'targets': 'Targets'
                                }),
                                width='stretch',
                                hide_index=True
                            )
                        else:
                            st.info("No receiving stats")
                    else:
                        st.info("Data not loaded")
                
                # Season Averages Section
                st.markdown("---")
                st.markdown("### 📊 Season Averages (2025)")
                
                avg_col1, avg_col2, avg_col3 = st.columns(3)
                
                with avg_col1:
                    st.markdown("**Passing Averages**")
                    if passing_stats is not None:
                        player_pass = passing_stats[passing_stats['player_id'] == selected_player_id]
                        if not player_pass.empty and 'season' in player_pass.columns:
                            player_pass_2025 = player_pass[player_pass['season'] == 2025]
                            if not player_pass_2025.empty:
                                games = len(player_pass_2025)
                                avg_yards = player_pass_2025['passing_yards'].mean()
                                avg_tds = player_pass_2025['pass_tds'].mean()
                                avg_comp = player_pass_2025['completions'].mean()
                                avg_att = player_pass_2025['attempts'].mean()
                                avg_int = player_pass_2025['interceptions'].mean()
                                comp_pct = (player_pass_2025['completions'].sum() / player_pass_2025['attempts'].sum() * 100) if player_pass_2025['attempts'].sum() > 0 else 0
                                
                                st.metric("Games Played", f"{games}")
                                st.metric("Yards/Game", f"{avg_yards:.1f}")
                                st.metric("TDs/Game", f"{avg_tds:.2f}")
                                st.metric("Comp/Att", f"{avg_comp:.1f}/{avg_att:.1f}")
                                st.metric("Comp %", f"{comp_pct:.1f}%")
                                st.metric("INT/Game", f"{avg_int:.2f}")
                            else:
                                st.info("No 2025 passing stats")
                        else:
                            st.info("No passing stats")
                    else:
                        st.info("Data not loaded")
                
                with avg_col2:
                    st.markdown("**Rushing Averages**")
                    if rushing_stats is not None:
                        player_rush = rushing_stats[rushing_stats['player_id'] == selected_player_id]
                        if not player_rush.empty and 'season' in player_rush.columns:
                            player_rush_2025 = player_rush[player_rush['season'] == 2025]
                            if not player_rush_2025.empty:
                                games = len(player_rush_2025)
                                avg_yards = player_rush_2025['rushing_yards'].mean()
                                avg_tds = player_rush_2025['rush_tds'].mean()
                                avg_att = player_rush_2025['rush_attempts'].mean()
                                ypc = (player_rush_2025['rushing_yards'].sum() / player_rush_2025['rush_attempts'].sum()) if player_rush_2025['rush_attempts'].sum() > 0 else 0
                                
                                st.metric("Games Played", f"{games}")
                                st.metric("Yards/Game", f"{avg_yards:.1f}")
                                st.metric("TDs/Game", f"{avg_tds:.2f}")
                                st.metric("Att/Game", f"{avg_att:.1f}")
                                st.metric("YPC", f"{ypc:.2f}")
                            else:
                                st.info("No 2025 rushing stats")
                        else:
                            st.info("No rushing stats")
                    else:
                        st.info("Data not loaded")
                
                with avg_col3:
                    st.markdown("**Receiving Averages**")
                    if receiving_stats is not None:
                        player_rec = receiving_stats[receiving_stats['player_id'] == selected_player_id]
                        if not player_rec.empty and 'season' in player_rec.columns:
                            player_rec_2025 = player_rec[player_rec['season'] == 2025]
                            if not player_rec_2025.empty:
                                games = len(player_rec_2025)
                                avg_yards = player_rec_2025['receiving_yards'].mean()
                                avg_rec = player_rec_2025['receptions'].mean()
                                avg_tds = player_rec_2025['rec_tds'].mean()
                                avg_tgts = player_rec_2025['targets'].mean()
                                ypr = (player_rec_2025['receiving_yards'].sum() / player_rec_2025['receptions'].sum()) if player_rec_2025['receptions'].sum() > 0 else 0
                                catch_rate = (player_rec_2025['receptions'].sum() / player_rec_2025['targets'].sum() * 100) if player_rec_2025['targets'].sum() > 0 else 0
                                
                                st.metric("Games Played", f"{games}")
                                st.metric("Yards/Game", f"{avg_yards:.1f}")
                                st.metric("Rec/Game", f"{avg_rec:.1f}")
                                st.metric("TDs/Game", f"{avg_tds:.2f}")
                                st.metric("Targets/Game", f"{avg_tgts:.1f}")
                                st.metric("YPR", f"{ypr:.2f}")
                                st.metric("Catch Rate", f"{catch_rate:.1f}%")
                            else:
                                st.info("No 2025 receiving stats")
                        else:
                            st.info("No receiving stats")
                    else:
                        st.info("Data not loaded")
    
    # Footer
   

if __name__ == "__main__":
    main()
