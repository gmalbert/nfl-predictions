import pandas as pd
from datetime import datetime

# Load the predictions file
df = pd.read_csv('data_files/nfl_games_historical_with_predictions.csv', sep='\t')
df['gameday'] = pd.to_datetime(df['gameday'])

print("=" * 60)
print("DATA INVENTORY")
print("=" * 60)

# Check what seasons we have
print(f"\nSeasons in dataset: {sorted(df['season'].unique())}")
print(f"Total games: {len(df)}")

# Check 2024 vs 2025
df_2024 = df[df['season'] == 2024]
df_2025 = df[df['season'] == 2025]

print(f"\n2024 Season:")
print(f"  Total games: {len(df_2024)}")
print(f"  With scores: {len(df_2024[df_2024['home_score'].notna()])}")
print(f"  Without scores (future): {len(df_2024[df_2024['home_score'].isna()])}")
print(f"  Date range: {df_2024['gameday'].min()} to {df_2024['gameday'].max()}")

print(f"\n2025 Season:")
print(f"  Total games: {len(df_2025)}")
print(f"  With scores: {len(df_2025[df_2025['home_score'].notna()])}")
print(f"  Without scores (future): {len(df_2025[df_2025['home_score'].isna()])}")
if len(df_2025) > 0:
    print(f"  Date range: {df_2025['gameday'].min()} to {df_2025['gameday'].max()}")

# Check what the model considers "future"
today = datetime.now()
future_games = df[df['gameday'] > today]
print(f"\nFuture games (after {today.date()}):")
print(f"  Total: {len(future_games)}")
print(f"  By season: {dict(future_games['season'].value_counts().sort_index())}")

# Check remaining 2024 games
remaining_2024 = df[(df['season'] == 2024) & (df['gameday'] > today)]
print(f"\nRemaining 2024 games: {len(remaining_2024)}")
if len(remaining_2024) > 0:
    print(f"  Date range: {remaining_2024['gameday'].min()} to {remaining_2024['gameday'].max()}")
    print(f"\n  Sample games:")
    for _, game in remaining_2024.head(5).iterrows():
        print(f"    {game['gameday'].date()}: {game['away_team']} @ {game['home_team']}")
