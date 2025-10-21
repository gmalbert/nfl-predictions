import nfl_data_py as nfl
import pandas as pd
from os import path

import os
from datetime import datetime
import requests

DATA_DIR = 'data_files/'
current_year = datetime.now().year
YEARS = range(2020, current_year + 1)

data = pd.DataFrame()

for i in YEARS:  
    i_data = pd.read_csv('https://github.com/nflverse/nflverse-data/releases/download/pbp/' \
                   'play_by_play_' + str(i) + '.csv.gz',
                   compression= 'gzip', low_memory= False)

    data = pd.concat([data, i_data], ignore_index=True, sort=True)
    data.reset_index(drop=True, inplace=True)

data.to_csv(path.join(DATA_DIR, 'nfl_play_by_play_thru_' + str(current_year) + '.csv.gz'), compression='gzip', index=False, sep='\t')
print(f"Saved play-by-play data through {current_year} to nfl_play_by_play_thru_{current_year}.csv.gz")
