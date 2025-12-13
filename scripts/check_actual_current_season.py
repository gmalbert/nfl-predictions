import pandas as pd
from datetime import datetime

# Check what actual game data we have
df = pd.read_csv('data_files/nfl_games_historical_with_predictions.csv', sep='\t')
df['gameday'] = pd.to_datetime(df['gameday'])

today = datetime(2025, 12, 13)

print("CURRENT DATE: December 13, 2025")
print("We should be in Week 15/16 of 2025-2026 NFL season")
print("=" * 70)

# Check games around today's date
recent = df[(df['gameday'] >= '2025-09-01') & (df['gameday'] <= '2025-12-31')].sort_values('gameday')
print(f"\nGames Sept-Dec 2025 (current season):")
print(f"Total: {len(recent)}")

if len(recent) > 0:
    print(f"\nDate range: {recent['gameday'].min()} to {recent['gameday'].max()}")
    print(f"\nSample of games from this period:")
    for _, game in recent.head(10).iterrows():
        date_str = game['gameday'].strftime('%Y-%m-%d')
        has_score = 'PLAYED' if pd.notna(game['home_score']) else 'FUTURE'
        print(f"  {date_str} Week {game['week']:2.0f}: {game['away_team']} @ {game['home_team']} - {has_score}")
    
    # Check games happening THIS WEEK (around Dec 13-15, 2025)
    this_week = recent[(recent['gameday'] >= '2025-12-13') & (recent['gameday'] <= '2025-12-16')]
    print(f"\nGames THIS WEEKEND (Dec 13-16, 2025): {len(this_week)}")
    for _, game in this_week.iterrows():
        date_str = game['gameday'].strftime('%Y-%m-%d %a')
        has_score = pd.notna(game['home_score'])
        print(f"  {date_str}: {game['away_team']} @ {game['home_team']} - {'PLAYED' if has_score else 'UPCOMING'}")
    
    # Check if scores exist
    played = recent[recent['home_score'].notna()]
    upcoming = recent[recent['home_score'].isna()]
    print(f"\nStatus of 2025 season games:")
    print(f"  Already played (have scores): {len(played)}")
    print(f"  Upcoming (no scores yet): {len(upcoming)}")
    
    if len(upcoming) > 0:
        print(f"\n  Upcoming games confidence:")
        print(f"    Max probability: {upcoming['prob_underdogCovered'].max():.1%}")
        print(f"    Games >=52.4%: {len(upcoming[upcoming['prob_underdogCovered'] >= 0.524])}")
        print(f"    Games >=45%: {len(upcoming[upcoming['prob_underdogCovered'] >= 0.45])}")

print("\n" + "=" * 70)
print("CHECKING WHAT 'season' COLUMN SAYS")
print("=" * 70)
print(f"Games with season=2025: {len(df[df['season'] == 2025])}")
print(f"Games with season=2026: {len(df[df['season'] == 2026])}")

# The issue might be season labeling
if len(recent) > 0:
    print(f"\nWhat 'season' value do Sept-Dec 2025 games have?")
    print(recent[['gameday', 'season', 'week', 'away_team', 'home_team']].head(5).to_string())
