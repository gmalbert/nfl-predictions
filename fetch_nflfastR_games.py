import nfl_data_py as nfl
import pandas as pd
from os import path

import os
from datetime import datetime
import requests

DATA_DIR = 'data_files/'

# Get game summaries for 2020-2025 (includes current season)
seasons = list(range(2020, 2026))  # Updated to include 2025 season
games = nfl.import_schedules(seasons)

# Show a sample
print(games.head())
print(games.columns.to_list())
# Save to CSV
out_path = "nfl_games_historical.csv"
games.to_csv(path.join(DATA_DIR, out_path), index=False, sep='\t')
print(f"Saved game summaries to {out_path}")

# Example: filter for 2024 regular season weeks only
games_2024 = games[(games['season'] == 2024) & (games['game_type'] == 'REG')]
print(games_2024[['game_id', 'week', 'home_team', 'away_team', 'home_score', 'away_score']].head())
