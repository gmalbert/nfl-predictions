#!/usr/bin/env python3
"""
Smart Play-by-Play Data Updater

Only downloads play-by-play data when there are new games to update.
This prevents unnecessary downloads on non-game days.

Usage:
    python update_pbp_smart.py
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import requests
import sys

DATA_DIR = Path('data_files')
PBP_FILE = DATA_DIR / 'nfl_play_by_play_historical.csv.gz'

def get_current_season():
    """Get current NFL season year"""
    now = datetime.now()
    if now.month <= 2:  # Jan-Feb
        return now.year - 1
    else:
        return now.year

def get_last_game_date(pbp_df):
    """Get the date of the most recent game in the dataset"""
    if pbp_df is None or pbp_df.empty:
        return None

    # Convert game_date to datetime if it's not already
    if 'game_date' in pbp_df.columns:
        pbp_df['game_date'] = pd.to_datetime(pbp_df['game_date'], errors='coerce')
        return pbp_df['game_date'].max()

    return None

def check_for_new_games(last_update_date):
    """Check if there should be new games since last update"""
    if last_update_date is None:
        return True  # No data yet, need to download

    now = datetime.now()
    days_since_update = (now - last_update_date).days

    # If it's been more than 3 days since last game, likely no new games
    # (Games are played ~4-5 days/week, so 3+ days means we should check)
    return days_since_update >= 3

def download_current_season_pbp(season):
    """Download only current season PBP data"""
    try:
        print(f"📥 Downloading {season} season play-by-play data...")
        url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"

        # Download current season data
        current_data = pd.read_csv(url, compression='gzip', low_memory=False)
        print(f"✅ Downloaded {len(current_data):,} plays for {season}")

        return current_data

    except Exception as e:
        print(f"❌ Failed to download {season} data: {e}")
        return None

def merge_with_existing_data(new_data):
    """Merge new season data with existing historical data"""
    if not PBP_FILE.exists():
        print("📝 No existing data file, creating new one")
        return new_data

    try:
        print("📚 Loading existing data...")
        existing_data = pd.read_csv(PBP_FILE, compression='gzip', sep='\t', low_memory=False)
        print(f"📊 Existing data: {len(existing_data):,} plays")

        # Remove old data for the current season if it exists
        current_season = get_current_season()
        existing_data = existing_data[existing_data['season'] != current_season]
        print(f"🗑️  Removed old {current_season} data from existing file")

        # Merge with new data
        combined_data = pd.concat([existing_data, new_data], ignore_index=True, sort=True)
        combined_data.reset_index(drop=True, inplace=True)

        print(f"🔄 Combined data: {len(combined_data):,} total plays")
        return combined_data

    except Exception as e:
        print(f"⚠️  Error merging data: {e}")
        return new_data

def main():
    print("🧠 Smart PBP Updater")
    print("=" * 50)

    current_season = get_current_season()
    print(f"🏈 Current season: {current_season}")

    # Check if we need to update
    needs_update = True
    last_game_date = None

    if PBP_FILE.exists():
        try:
            # Read the entire file to get accurate max date (data is sorted by season)
            existing_data = pd.read_csv(PBP_FILE, compression='gzip', sep='\t', 
                                      usecols=['game_date'], low_memory=False)
            last_game_date = get_last_game_date(existing_data)
            print(f"📅 Last game in existing data: {last_game_date}")

            # Check if update is needed
            needs_update = check_for_new_games(last_game_date)
            print(f"🔍 Update needed: {needs_update}")

        except Exception as e:
            print(f"⚠️  Error checking existing data: {e}")
            needs_update = True

    if not needs_update:
        print("✅ Data is up to date, skipping download")
        return 0

    # Download current season data
    new_data = download_current_season_pbp(current_season)
    if new_data is None:
        print("❌ Download failed")
        return 1

    # Merge with existing data
    final_data = merge_with_existing_data(new_data)

    # Save updated data
    DATA_DIR.mkdir(exist_ok=True)
    final_data.to_csv(PBP_FILE, compression='gzip', index=False, sep='\t')
    print(f"💾 Saved {len(final_data):,} plays to {PBP_FILE}")

    print("✅ PBP data update complete!")
    return 0

if __name__ == '__main__':
    sys.exit(main())