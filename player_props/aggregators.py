"""
Player stat aggregation functions.
Converts play-by-play data to game-level player statistics.

Usage:
    from player_props.aggregators import aggregate_passing_stats
    
    pbp = pd.read_csv('data_files/nfl_play_by_play_historical.csv.gz', compression='gzip')
    passing_stats = aggregate_passing_stats(pbp)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def aggregate_passing_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate passing stats by player and game.
    
    Args:
        pbp: Play-by-play DataFrame
    
    Returns:
        DataFrame with columns: 
            [player_id, player_name, game_id, season, week, game_date,
             team, opponent, passing_yards, pass_tds, completions, 
             attempts, interceptions, completion_pct]
    """
    # Filter to passing plays only
    passing_plays = pbp[pbp['passer_player_id'].notna()].copy()
    
    if passing_plays.empty:
        print("⚠️  No passing plays found in dataset")
        return pd.DataFrame()
    
    # Aggregate by player and game
    agg_stats = passing_plays.groupby([
        'passer_player_id', 
        'passer_player_name', 
        'game_id', 
        'season', 
        'week',
        'posteam',  # Team the player is on
        'defteam'   # Opponent
    ]).agg({
        'passing_yards': 'sum',
        'pass_touchdown': 'sum',
        'complete_pass': 'sum',
        'pass_attempt': 'sum',
        'interception': 'sum',
        'game_date': 'first'  # Get game date
    }).reset_index()
    
    # Rename columns
    agg_stats.columns = [
        'player_id', 'player_name', 'game_id', 'season', 'week',
        'team', 'opponent', 'passing_yards', 'pass_tds', 
        'completions', 'attempts', 'interceptions', 'game_date'
    ]
    
    # Calculate completion percentage
    agg_stats['completion_pct'] = (
        agg_stats['completions'] / agg_stats['attempts'] * 100
    ).round(1)
    
    # Sort by date
    agg_stats = agg_stats.sort_values(['player_id', 'season', 'week'])
    
    print(f"✅ Aggregated {len(agg_stats):,} player-game passing records")
    print(f"   Players: {agg_stats['player_id'].nunique():,}")
    print(f"   Games: {agg_stats['game_id'].nunique():,}")
    
    return agg_stats


def aggregate_rushing_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate rushing stats by player and game.
    
    Returns:
        DataFrame with columns:
            [player_id, player_name, game_id, season, week, game_date,
             team, opponent, rushing_yards, rush_tds, rush_attempts, ypc]
    """
    rushing_plays = pbp[pbp['rusher_player_id'].notna()].copy()
    
    if rushing_plays.empty:
        print("⚠️  No rushing plays found in dataset")
        return pd.DataFrame()
    
    agg_stats = rushing_plays.groupby([
        'rusher_player_id',
        'rusher_player_name',
        'game_id',
        'season',
        'week',
        'posteam',
        'defteam'
    ]).agg({
        'rushing_yards': 'sum',
        'rush_touchdown': 'sum',
        'rush_attempt': 'sum',
        'game_date': 'first'
    }).reset_index()
    
    agg_stats.columns = [
        'player_id', 'player_name', 'game_id', 'season', 'week',
        'team', 'opponent', 'rushing_yards', 'rush_tds', 
        'rush_attempts', 'game_date'
    ]
    
    # Calculate yards per carry
    agg_stats['ypc'] = (
        agg_stats['rushing_yards'] / agg_stats['rush_attempts']
    ).round(1)
    
    agg_stats = agg_stats.sort_values(['player_id', 'season', 'week'])
    
    print(f"✅ Aggregated {len(agg_stats):,} player-game rushing records")
    print(f"   Players: {agg_stats['player_id'].nunique():,}")
    
    return agg_stats


def aggregate_receiving_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate receiving stats by player and game.
    
    Returns:
        DataFrame with columns:
            [player_id, player_name, game_id, season, week, game_date,
             team, opponent, receiving_yards, receptions, rec_tds, 
             targets, ypr, catch_rate]
    """
    receiving_plays = pbp[pbp['receiver_player_id'].notna()].copy()
    
    if receiving_plays.empty:
        print("⚠️  No receiving plays found in dataset")
        return pd.DataFrame()
    
    # Count targets (all passes to this receiver, caught or not)
    receiving_plays['target'] = 1
    
    agg_stats = receiving_plays.groupby([
        'receiver_player_id',
        'receiver_player_name',
        'game_id',
        'season',
        'week',
        'posteam',
        'defteam'
    ]).agg({
        'receiving_yards': 'sum',
        'complete_pass': 'sum',  # Receptions = completed passes
        'pass_touchdown': 'sum',  # Receiving TDs
        'pass_attempt': 'sum',    # Targets
        'game_date': 'first'
    }).reset_index()
    
    agg_stats.columns = [
        'player_id', 'player_name', 'game_id', 'season', 'week',
        'team', 'opponent', 'receiving_yards', 'receptions', 
        'rec_tds', 'targets', 'game_date'
    ]
    
    # Calculate yards per reception
    agg_stats['ypr'] = (
        agg_stats['receiving_yards'] / agg_stats['receptions'].replace(0, np.nan)
    ).round(1)
    
    # Calculate catch rate
    agg_stats['catch_rate'] = (
        agg_stats['receptions'] / agg_stats['targets'] * 100
    ).round(1)
    
    agg_stats = agg_stats.sort_values(['player_id', 'season', 'week'])
    
    print(f"✅ Aggregated {len(agg_stats):,} player-game receiving records")
    print(f"   Players: {agg_stats['player_id'].nunique():,}")
    
    return agg_stats


def calculate_rolling_averages(
    player_stats: pd.DataFrame,
    stat_cols: List[str],
    windows: List[int] = [3, 5, 10],
    player_id_col: str = 'player_id'
) -> pd.DataFrame:
    """
    Calculate rolling averages for stat columns.
    
    Args:
        player_stats: Game-level player stats (must have player_id, season, week)
        stat_cols: List of stat columns to calculate rolling averages for
        windows: List of window sizes (e.g., [3, 5, 10] for L3, L5, L10)
        player_id_col: Column name for player ID
    
    Returns:
        DataFrame with additional columns for rolling averages
        (e.g., passing_yards_L3, passing_yards_L5)
    """
    # Ensure sorted by player and time
    player_stats = player_stats.sort_values([player_id_col, 'season', 'week']).copy()
    
    for stat_col in stat_cols:
        if stat_col not in player_stats.columns:
            print(f"⚠️  Column '{stat_col}' not found, skipping")
            continue
            
        for window in windows:
            col_name = f'{stat_col}_L{window}'
            
            # Calculate rolling average, excluding current game (shift by 1)
            player_stats[col_name] = player_stats.groupby(player_id_col)[stat_col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
    
    print(f"✅ Calculated rolling averages: windows={windows}")
    
    return player_stats


def aggregate_all_stats(
    pbp: pd.DataFrame,
    rolling_windows: List[int] = [3, 5, 10],
    save_to_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aggregate all player stats and calculate rolling averages.
    
    Args:
        pbp: Play-by-play DataFrame
        rolling_windows: Window sizes for rolling averages
        save_to_dir: If provided, save aggregated stats to this directory
    
    Returns:
        Tuple of (passing_stats, rushing_stats, receiving_stats)
    """
    print("\n🏈 Aggregating Player Stats...")
    print("=" * 60)
    
    # Aggregate each stat type
    print("\n1. Passing Stats:")
    passing = aggregate_passing_stats(pbp)
    
    print("\n2. Rushing Stats:")
    rushing = aggregate_rushing_stats(pbp)
    
    print("\n3. Receiving Stats:")
    receiving = aggregate_receiving_stats(pbp)
    
    # Calculate rolling averages
    print("\n📊 Calculating Rolling Averages...")
    
    if not passing.empty:
        passing = calculate_rolling_averages(
            passing,
            stat_cols=['passing_yards', 'pass_tds', 'completions', 'attempts', 'interceptions'],
            windows=rolling_windows
        )
    
    if not rushing.empty:
        rushing = calculate_rolling_averages(
            rushing,
            stat_cols=['rushing_yards', 'rush_tds', 'rush_attempts', 'ypc'],
            windows=rolling_windows
        )
    
    if not receiving.empty:
        receiving = calculate_rolling_averages(
            receiving,
            stat_cols=['receiving_yards', 'receptions', 'rec_tds', 'targets'],
            windows=rolling_windows
        )
    
    # Save if directory provided
    if save_to_dir:
        save_to_dir = Path(save_to_dir)
        save_to_dir.mkdir(exist_ok=True, parents=True)
        
        if not passing.empty:
            passing.to_csv(save_to_dir / 'player_passing_stats.csv', index=False)
            print(f"\n💾 Saved player_passing_stats.csv ({len(passing):,} rows)")
        
        if not rushing.empty:
            rushing.to_csv(save_to_dir / 'player_rushing_stats.csv', index=False)
            print(f"💾 Saved player_rushing_stats.csv ({len(rushing):,} rows)")
        
        if not receiving.empty:
            receiving.to_csv(save_to_dir / 'player_receiving_stats.csv', index=False)
            print(f"💾 Saved player_receiving_stats.csv ({len(receiving):,} rows)")
    
    print("\n✅ All stats aggregated successfully!")
    
    return passing, rushing, receiving


# CLI script for running aggregation
if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    print("🏈 NFL Player Stats Aggregator")
    print("=" * 60)
    
    # Load play-by-play data
    DATA_DIR = Path('data_files')
    pbp_file = DATA_DIR / 'nfl_play_by_play_historical.csv.gz'
    
    if not pbp_file.exists():
        print(f"\n❌ File not found: {pbp_file}")
        print("Please ensure nfl_play_by_play_historical.csv.gz exists")
        sys.exit(1)
    
    print(f"\n📂 Loading {pbp_file.name}...")
    pbp = pd.read_csv(pbp_file, compression='gzip', sep='\t', low_memory=False)
    print(f"✅ Loaded {len(pbp):,} plays from {pbp['season'].min()}-{pbp['season'].max()}")
    
    # Aggregate all stats
    passing, rushing, receiving = aggregate_all_stats(
        pbp,
        rolling_windows=[3, 5, 10],
        save_to_dir=DATA_DIR
    )
    
    print("\n" + "=" * 60)
    print("✅ Aggregation complete! Ready for model training.")
    print("\nNext steps:")
    print("  1. Review generated CSV files in data_files/")
    print("  2. Run player_props/models.py to train baseline models")
