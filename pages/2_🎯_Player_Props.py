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
    
    # Ensure opponent_def_rank column exists (for backward compatibility)
    if 'opponent_def_rank' not in df.columns:
        st.warning("⚠️ opponent_def_rank column missing from predictions. Please regenerate predictions.")
        df['opponent_def_rank'] = 16.0  # Default to league average
    
    return df


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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⭐ Top Picks",
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
            col1, col2, col3 = st.columns(3)
            
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
                
                # Select columns to show
                show_cols = [
                    'display_name', 'position', 'team', 'opponent', 'Defense Rank', 'prop_type', 
                    'Recommendation', 'Confidence', 'Tier', 'Last 3 Avg', 'Last 5 Avg', 'Last 10 Avg'
                ]
                
                st.dataframe(
                    display_df[show_cols].rename(columns={
                        'display_name': 'Player',
                        'position': 'Pos',
                        'team': 'Team',
                        'opponent': 'Opp',
                        'prop_type': 'Prop Type'
                    }),
                    width='stretch',
                    height=600,
                    hide_index=True
                )
                
                st.caption(f"Showing {len(filtered):,} predictions")
                
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
    # TAB 2: Top QBs (Season Leaders)
    # ========================================================================
    with tab2:
        st.subheader("🏆 2025 Season Leaders - Quarterbacks")
        
        if passing_stats is not None:
            # Filter to 2025 season
            season_stats = passing_stats[passing_stats['season'] == 2025].copy()
            
            if not season_stats.empty:
                # Aggregate to season totals per player
                season_totals = season_stats.groupby(['player_name', 'team']).agg({
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
    # TAB 3: Top RBs (Season Leaders)
    # ========================================================================
    with tab3:
        st.subheader("🏆 2025 Season Leaders - Running Backs")
        
        if rushing_stats is not None:
            # Filter to 2025 season
            season_stats = rushing_stats[rushing_stats['season'] == 2025].copy()
            
            if not season_stats.empty:
                # Aggregate to season totals per player
                season_totals = season_stats.groupby(['player_name', 'team']).agg({
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
    # TAB 4: Top WRs/TEs (Season Leaders)
    # ========================================================================
    with tab4:
        st.subheader("🏆 2025 Season Leaders - Wide Receivers & Tight Ends")
        
        if receiving_stats is not None:
            # Filter to 2025 season
            season_stats = receiving_stats[receiving_stats['season'] == 2025].copy()
            
            if not season_stats.empty:
                # Aggregate to season totals per player
                season_totals = season_stats.groupby(['player_name', 'team']).agg({
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
    # TAB 5: Player Search
    # ========================================================================
    with tab5:
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
    st.markdown("---")
    st.caption("💡 **Coming Soon**: ML-powered prop predictions, lineup optimizer, parlay builder")


if __name__ == "__main__":
    main()
