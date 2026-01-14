import nfl_data_py as nfl
import pandas as pd
from os import path
from datetime import datetime

DATA_DIR = 'data_files/'
# Use 2025 for the current NFL season (starts in 2025, playoffs in 2026)
season_year = 2025

# Fetch schedule for current season
print(f"Fetching NFL schedule for {season_year}...")
schedule = nfl.import_schedules([season_year])
print("Columns in schedule:", schedule.columns.tolist())

# Select and format columns for the schedule file
# Assuming columns: adjust based on actual
stadium_col = 'stadium' if 'stadium' in schedule.columns else 'venue'

schedule_df = schedule[['week', 'gameday', 'home_team', 'away_team', stadium_col]].copy()
schedule_df['date'] = pd.to_datetime(schedule_df['gameday']).dt.strftime('%Y-%m-%d')
schedule_df['venue'] = schedule_df[stadium_col]
schedule_df['status'] = 'REG'
schedule_df = schedule_df[['week', 'date', 'home_team', 'away_team', 'venue', 'status']]

# Save to CSV
out_path = f"nfl_schedule_{season_year}.csv"
full_path = path.join(DATA_DIR, out_path)
schedule_df.to_csv(full_path, index=False)
print(f"Saved updated schedule to {full_path}")
print(f"Total games: {len(schedule_df)}")