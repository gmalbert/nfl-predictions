import nfl_data_py as nfl
import pandas as pd
from os import path

import os
from datetime import datetime
import requests

DATA_DIR = 'data_files/'
current_year = datetime.now().year
YEARS = range(2020, current_year + 1)

# Get game summaries for 2020-2025
seasons = list(range(2020, current_year + 1))
games = nfl.import_schedules(seasons)

# Show a sample
print(games.tail(15))

# Save to CSV
out_path = "nfl_games_historical.csv"
games.to_csv(path.join(DATA_DIR, out_path), index=False, sep='\t')
print(f"Saved game summaries to {out_path}")
