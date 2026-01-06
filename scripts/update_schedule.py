import pandas as pd

df = pd.read_csv('data_files/nfl_games_historical.csv', sep='\t')
df_2025 = df[df['season']==2025]

schedule = df_2025[['week', 'gameday', 'home_team', 'away_team', 'stadium', 'game_type']].copy()
schedule.columns = ['week', 'date', 'home_team', 'away_team', 'venue', 'status']

schedule.to_csv('data_files/nfl_schedule_2025.csv', index=False)
print(f'✅ Saved {len(schedule)} games to nfl_schedule_2025.csv')
print(f'   Including playoff games: {len(schedule[schedule["status"] != "REG"])}')
print(f'   Weeks: {sorted(schedule["week"].unique())}')
