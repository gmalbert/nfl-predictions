"""
Explore play-by-play data for player-level statistics.
Run: python scripts/explore_player_stats.py

This script analyzes the nfl_play_by_play_historical.csv.gz file to understand
available player data for building prop prediction models.

Data source: NFLverse play-by-play (2020-2025 seasons, auto-updates yearly)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Set UTF-8 encoding for Windows console output
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    # Load play-by-play data
    DATA_DIR = Path('data_files')
    pbp_file = DATA_DIR / 'nfl_play_by_play_historical.csv.gz'
    
    if not pbp_file.exists():
        print(f"❌ File not found: {pbp_file}")
        print("Please ensure nfl_play_by_play_historical.csv.gz exists in data_files/")
        return
    
    print("Loading play-by-play data...")
    pbp = pd.read_csv(pbp_file, compression='gzip', sep='\t', low_memory=False)
    
    print(f"\n📊 Dataset Overview")
    print(f"  Total plays: {len(pbp):,}")
    print(f"  Seasons: {pbp['season'].min()}-{pbp['season'].max()}")
    print(f"  Columns: {pbp.shape[1]}")
    print(f"  Memory: {pbp.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Key player stat columns
    player_cols = {
        'Passing': ['passer_player_name', 'passer_player_id', 'passing_yards', 
                    'pass_touchdown', 'complete_pass', 'pass_attempt', 'interception'],
        'Rushing': ['rusher_player_name', 'rusher_player_id', 'rushing_yards', 
                    'rush_touchdown', 'rush_attempt'],
        'Receiving': ['receiver_player_name', 'receiver_player_id', 'receiving_yards', 
                      'reception', 'pass_touchdown'],
    }
    
    print(f"\n✅ Available Player Columns:")
    for category, cols in player_cols.items():
        available = [c for c in cols if c in pbp.columns]
        print(f"\n  {category}: {len(available)}/{len(cols)} columns")
        for col in available:
            non_null = pbp[col].notna().sum()
            print(f"    - {col:30s} ({non_null:,} non-null)")
    
    # Analyze 2024 season (most recent)
    pbp_2024 = pbp[pbp['season'] == 2024].copy()
    print(f"\n\n🏈 2024 Season Analysis ({len(pbp_2024):,} plays)")
    
    # Top passers
    print("\n📊 Top 10 Passers by Total Yards:")
    if 'passer_player_name' in pbp.columns and 'passing_yards' in pbp.columns:
        passers = pbp_2024[pbp_2024['passer_player_name'].notna()].groupby('passer_player_name').agg({
            'passing_yards': 'sum',
            'pass_touchdown': 'sum',
            'complete_pass': 'sum',
            'pass_attempt': 'sum',
            'interception': 'sum',
            'game_id': 'nunique'
        }).sort_values('passing_yards', ascending=False).head(10)
        passers.columns = ['Pass Yds', 'TDs', 'Comp', 'Att', 'INT', 'Games']
        passers['Avg/Game'] = (passers['Pass Yds'] / passers['Games']).round(1)
        print(passers[['Pass Yds', 'TDs', 'Comp', 'Att', 'INT', 'Games', 'Avg/Game']].to_string())
    
    # Top rushers
    print("\n\n🏃 Top 10 Rushers by Total Yards:")
    if 'rusher_player_name' in pbp.columns and 'rushing_yards' in pbp.columns:
        rushers = pbp_2024[pbp_2024['rusher_player_name'].notna()].groupby('rusher_player_name').agg({
            'rushing_yards': 'sum',
            'rush_touchdown': 'sum',
            'rush_attempt': 'sum',
            'game_id': 'nunique'
        }).sort_values('rushing_yards', ascending=False).head(10)
        rushers.columns = ['Rush Yds', 'TDs', 'Attempts', 'Games']
        rushers['Avg/Game'] = (rushers['Rush Yds'] / rushers['Games']).round(1)
        rushers['YPC'] = (rushers['Rush Yds'] / rushers['Attempts']).round(1)
        print(rushers[['Rush Yds', 'TDs', 'Attempts', 'Games', 'Avg/Game', 'YPC']].to_string())
    
    # Top receivers
    print("\n\n🎯 Top 10 Receivers by Total Yards:")
    if 'receiver_player_name' in pbp.columns and 'receiving_yards' in pbp.columns:
        receivers = pbp_2024[pbp_2024['receiver_player_name'].notna()].groupby('receiver_player_name').agg({
            'receiving_yards': 'sum',
            'complete_pass': 'sum',  # Receptions = completed passes
            'pass_touchdown': 'sum',
            'game_id': 'nunique'
        }).sort_values('receiving_yards', ascending=False).head(10)
        receivers.columns = ['Rec Yds', 'Rec', 'TDs', 'Games']
        receivers['Avg/Game'] = (receivers['Rec Yds'] / receivers['Games']).round(1)
        receivers['YPR'] = (receivers['Rec Yds'] / receivers['Rec']).round(1)
        print(receivers[['Rec Yds', 'Rec', 'TDs', 'Games', 'Avg/Game', 'YPR']].to_string())
    
    # Data quality metrics
    print("\n\n🔍 Data Quality Assessment:")
    total_plays = len(pbp)
    print(f"  Total plays: {total_plays:,}")
    print(f"  Plays with passer: {pbp['passer_player_name'].notna().sum():,} ({pbp['passer_player_name'].notna().sum()/total_plays*100:.1f}%)")
    print(f"  Plays with rusher: {pbp['rusher_player_name'].notna().sum():,} ({pbp['rusher_player_name'].notna().sum()/total_plays*100:.1f}%)")
    print(f"  Plays with receiver: {pbp['receiver_player_name'].notna().sum():,} ({pbp['receiver_player_name'].notna().sum()/total_plays*100:.1f}%)")
    
    # Check for additional useful columns
    print("\n\n📋 Additional Context Columns:")
    context_cols = ['game_date', 'week', 'posteam', 'defteam', 'home_team', 'away_team',
                    'score_differential', 'game_seconds_remaining', 'qtr', 
                    'down', 'ydstogo', 'yardline_100']
    available_context = [c for c in context_cols if c in pbp.columns]
    print(f"  Available: {len(available_context)}/{len(context_cols)}")
    for col in available_context:
        print(f"    ✓ {col}")
    
    # Save sample for manual inspection
    print("\n\n💾 Saving Samples...")
    sample_dir = DATA_DIR / 'samples'
    sample_dir.mkdir(exist_ok=True)
    
    # Sample of passing plays
    if 'passer_player_name' in pbp.columns:
        passing_sample = pbp[pbp['passer_player_name'].notna()][
            ['game_date', 'week', 'passer_player_name', 'posteam', 'defteam',
             'passing_yards', 'pass_touchdown', 'complete_pass', 'interception']
        ].head(500)
        passing_sample.to_csv(sample_dir / 'passing_plays_sample.csv', index=False)
        print(f"  ✓ Saved passing_plays_sample.csv (500 plays)")
    
    # Sample of rushing plays
    if 'rusher_player_name' in pbp.columns:
        rushing_sample = pbp[pbp['rusher_player_name'].notna()][
            ['game_date', 'week', 'rusher_player_name', 'posteam', 'defteam',
             'rushing_yards', 'rush_touchdown']
        ].head(500)
        rushing_sample.to_csv(sample_dir / 'rushing_plays_sample.csv', index=False)
        print(f"  ✓ Saved rushing_plays_sample.csv (500 plays)")
    
    # Sample of receiving plays
    if 'receiver_player_name' in pbp.columns:
        receiving_sample = pbp[pbp['receiver_player_name'].notna()][
            ['game_date', 'week', 'receiver_player_name', 'posteam', 'defteam',
             'receiving_yards', 'complete_pass', 'pass_touchdown']
        ].head(500)
        receiving_sample.to_csv(sample_dir / 'receiving_plays_sample.csv', index=False)
        print(f"  ✓ Saved receiving_plays_sample.csv (500 plays)")
    
    print(f"\n✅ Exploration complete! Sample files saved to {sample_dir}")
    print("\nNext steps:")
    print("  1. Review sample CSVs to understand data structure")
    print("  2. Run player_props/aggregators.py to create game-level stats")
    print("  3. Start building baseline models")

if __name__ == '__main__':
    main()
