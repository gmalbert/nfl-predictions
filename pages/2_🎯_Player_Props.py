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
    passing_stats = load_player_passing_stats()
    rushing_stats = load_player_rushing_stats()
    receiving_stats = load_player_receiving_stats()
    schedule = load_nfl_schedule()
    
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
            
            Each file includes rolling averages (L3, L5, L10 games) for trend analysis.
            """)
        
        return
    
    # Upcoming games section
    upcoming = get_upcoming_games(schedule)
    
    if upcoming is not None and len(upcoming) > 0:
        st.info(f"📅 **{len(upcoming)} upcoming games** this week")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Passing Props",
        "🏃 Rushing Props", 
        "🤲 Receiving Props",
        "🔍 Player Search"
    ])
    
    # ========================================================================
    # TAB 1: Passing Props
    # ========================================================================
    with tab1:
        st.subheader("Quarterback Passing Props")
        
        if passing_stats is not None:
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_games = st.slider("Minimum Games", 1, 10, 5, key="pass_min_games")
            
            with col2:
                season_filter = st.selectbox("Season", [2025, 2024, 2023, 2022, 2021, 2020], key="pass_season")
            
            with col3:
                sort_by = st.selectbox(
                    "Sort By",
                    ["passing_yards", "passing_tds", "completions", "attempts", "avg_yards_L5"],
                    key="pass_sort"
                )
            
            # Filter data
            filtered = filter_stats_by_season(passing_stats, season_filter)
            if 'games_played' in filtered.columns:
                filtered = filtered[filtered['games_played'] >= min_games]
            
            # Display stats
            if not filtered.empty:
                # Select key columns
                display_cols = ['player_name', 'team', 'week', 'passing_yards', 'passing_tds', 
                               'completions', 'attempts', 'interceptions']
                
                # Add rolling averages if available
                rolling_cols = [c for c in filtered.columns if c.startswith('avg_') or c.startswith('std_')]
                display_cols.extend(rolling_cols[:6])  # Limit to first 6 rolling cols
                
                available_cols = [c for c in display_cols if c in filtered.columns]
                
                st.dataframe(
                    filtered[available_cols].sort_values(sort_by, ascending=False).head(50),
                    use_container_width=True,
                    height=500
                )
                
                st.caption(f"Showing top 50 of {len(filtered):,} QB performances")
            else:
                st.info("No data matches the selected filters")
        else:
            st.error("Passing stats not loaded")
    
    # ========================================================================
    # TAB 2: Rushing Props
    # ========================================================================
    with tab2:
        st.subheader("Running Back Rushing Props")
        
        if rushing_stats is not None:
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_games = st.slider("Minimum Games", 1, 10, 5, key="rush_min_games")
            
            with col2:
                season_filter = st.selectbox("Season", [2025, 2024, 2023, 2022, 2021, 2020], key="rush_season")
            
            with col3:
                sort_by = st.selectbox(
                    "Sort By",
                    ["rushing_yards", "rushing_tds", "rush_attempts", "avg_yards_L5"],
                    key="rush_sort"
                )
            
            # Filter data
            filtered = filter_stats_by_season(rushing_stats, season_filter)
            if 'games_played' in filtered.columns:
                filtered = filtered[filtered['games_played'] >= min_games]
            
            # Display stats
            if not filtered.empty:
                display_cols = ['player_name', 'team', 'week', 'rushing_yards', 'rushing_tds', 
                               'rush_attempts']
                
                rolling_cols = [c for c in filtered.columns if c.startswith('avg_') or c.startswith('std_')]
                display_cols.extend(rolling_cols[:6])
                
                available_cols = [c for c in display_cols if c in filtered.columns]
                
                st.dataframe(
                    filtered[available_cols].sort_values(sort_by, ascending=False).head(50),
                    use_container_width=True,
                    height=500
                )
                
                st.caption(f"Showing top 50 of {len(filtered):,} RB performances")
            else:
                st.info("No data matches the selected filters")
        else:
            st.error("Rushing stats not loaded")
    
    # ========================================================================
    # TAB 3: Receiving Props
    # ========================================================================
    with tab3:
        st.subheader("Wide Receiver / Tight End Receiving Props")
        
        if receiving_stats is not None:
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_games = st.slider("Minimum Games", 1, 10, 5, key="rec_min_games")
            
            with col2:
                season_filter = st.selectbox("Season", [2025, 2024, 2023, 2022, 2021, 2020], key="rec_season")
            
            with col3:
                sort_by = st.selectbox(
                    "Sort By",
                    ["receiving_yards", "receptions", "receiving_tds", "avg_yards_L5"],
                    key="rec_sort"
                )
            
            # Filter data
            filtered = filter_stats_by_season(receiving_stats, season_filter)
            if 'games_played' in filtered.columns:
                filtered = filtered[filtered['games_played'] >= min_games]
            
            # Display stats
            if not filtered.empty:
                display_cols = ['player_name', 'team', 'week', 'receiving_yards', 'receptions', 
                               'receiving_tds', 'targets']
                
                rolling_cols = [c for c in filtered.columns if c.startswith('avg_') or c.startswith('std_')]
                display_cols.extend(rolling_cols[:6])
                
                available_cols = [c for c in display_cols if c in filtered.columns]
                
                st.dataframe(
                    filtered[available_cols].sort_values(sort_by, ascending=False).head(50),
                    use_container_width=True,
                    height=500
                )
                
                st.caption(f"Showing top 50 of {len(filtered):,} WR/TE performances")
            else:
                st.info("No data matches the selected filters")
        else:
            st.error("Receiving stats not loaded")
    
    # ========================================================================
    # TAB 4: Player Search
    # ========================================================================
    with tab4:
        st.subheader("Search Individual Player Stats")
        
        # Combine all players
        all_players = []
        if passing_stats is not None:
            all_players.extend(passing_stats['player_name'].unique())
        if rushing_stats is not None:
            all_players.extend(rushing_stats['player_name'].unique())
        if receiving_stats is not None:
            all_players.extend(receiving_stats['player_name'].unique())
        
        all_players = sorted(set(all_players))
        
        if all_players:
            selected_player = st.selectbox("Select Player", all_players, key="player_search")
            
            if selected_player:
                st.markdown(f"### {selected_player}")
                
                # Show stats from all categories
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Passing Stats**")
                    if passing_stats is not None:
                        player_pass = passing_stats[passing_stats['player_name'] == selected_player]
                        if not player_pass.empty:
                            st.dataframe(player_pass.tail(10), use_container_width=True)
                        else:
                            st.info("No passing stats")
                    else:
                        st.info("Data not loaded")
                
                with col2:
                    st.markdown("**Rushing Stats**")
                    if rushing_stats is not None:
                        player_rush = rushing_stats[rushing_stats['player_name'] == selected_player]
                        if not player_rush.empty:
                            st.dataframe(player_rush.tail(10), use_container_width=True)
                        else:
                            st.info("No rushing stats")
                    else:
                        st.info("Data not loaded")
                
                with col3:
                    st.markdown("**Receiving Stats**")
                    if receiving_stats is not None:
                        player_rec = receiving_stats[receiving_stats['player_name'] == selected_player]
                        if not player_rec.empty:
                            st.dataframe(player_rec.tail(10), use_container_width=True)
                        else:
                            st.info("No receiving stats")
                    else:
                        st.info("Data not loaded")
        else:
            st.info("No player data available")
    
    # Footer
    st.markdown("---")
    st.caption("💡 **Coming Soon**: ML-powered prop predictions, lineup optimizer, parlay builder")


if __name__ == "__main__":
    main()
