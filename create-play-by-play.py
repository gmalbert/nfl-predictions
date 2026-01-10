import nfl_data_py as nfl
import pandas as pd
from os import path

import os
from datetime import datetime
import requests
import argparse

DATA_DIR = 'data_files/'
current_year = datetime.now().year
# Only fetch through current year if season has started (September+), otherwise use last year
# NFL regular season starts in September, so if we're before September, use previous year
if datetime.now().month < 9:
    end_year = current_year - 1
else:
    end_year = current_year

def main():
    parser = argparse.ArgumentParser(description='Download NFL play-by-play data')
    parser.add_argument('--current-season-only', action='store_true', 
                       help='Only download current season data (much faster for nightly updates)')
    args = parser.parse_args()
    
    if args.current_season_only:
        # Only download current season for nightly updates
        YEARS = [end_year]
        print(f"📅 Downloading only current season ({end_year}) data...")
    else:
        # Download all historical seasons
        YEARS = range(2020, end_year + 1)
        print(f"📚 Downloading all seasons ({min(YEARS)}-{end_year}) data...")

    data = pd.DataFrame()

    for i in YEARS:
        try:
            print(f"Downloading {i} season data...")
            i_data = pd.read_csv('https://github.com/nflverse/nflverse-data/releases/download/pbp/' \
                           'play_by_play_' + str(i) + '.csv.gz',
                           compression= 'gzip', low_memory= False)
            
            data = pd.concat([data, i_data], ignore_index=True, sort=True)
            data.reset_index(drop=True, inplace=True)
            print(f"✓ {i} season: {len(i_data):,} plays")
        except Exception as e:
            print(f"⚠️  Warning: Could not download {i} season data - {e}")
            print(f"   Continuing with available data...")

    data.to_csv(path.join(DATA_DIR, 'nfl_play_by_play_historical.csv.gz'), compression='gzip', index=False, sep='\t')
    print(f"\n✅ Saved {len(data):,} plays ({min(YEARS)}-{end_year}) to nfl_play_by_play_historical.csv.gz")

if __name__ == '__main__':
    main()
