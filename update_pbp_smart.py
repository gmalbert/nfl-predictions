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
import json

DATA_DIR = Path('data_files')
PBP_FILE = DATA_DIR / 'nfl_play_by_play_historical.csv.gz'
METADATA_FILE = DATA_DIR / 'pbp_metadata.json'
HISTORICAL_CACHE = DATA_DIR / 'nfl_play_by_play_historical_2020_2024.csv.gz'

def get_current_season():
    """Get current NFL season year"""
    now = datetime.now()
    if now.month <= 2:  # Jan-Feb
        return now.year - 1
    else:
        return now.year

def read_metadata():
    """Read metadata file to get last update info"""
    if not METADATA_FILE.exists():
        return None
    
    try:
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Error reading metadata: {e}")
        return None

def write_metadata(total_plays):
    """Write metadata file with current update info"""
    metadata = {
        'last_update': datetime.now().isoformat(),
        'total_plays': total_plays,
        'seasons': f"{min([2020, get_current_season()])}-{get_current_season()}"
    }
    
    try:
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📝 Updated metadata: {metadata}")
    except Exception as e:
        print(f"⚠️  Error writing metadata: {e}")

def should_update():
    """Check if we need to update based on metadata"""
    metadata = read_metadata()
    
    if metadata is None:
        print("📋 No metadata file found, update needed")
        return True
    
    try:
        last_update = datetime.fromisoformat(metadata['last_update'])
        hours_since_update = (datetime.now() - last_update).total_seconds() / 3600
        
        print(f"📅 Last update: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏰ Hours since update: {hours_since_update:.1f}")
        
        # Update if it's been more than 12 hours
        # (Games typically finish by midnight ET, cron runs at 3 AM UTC)
        if hours_since_update >= 12:
            print("✅ Update needed (>12 hours since last update)")
            return True
        else:
            print("⏭️  Skipping update (recent update found)")
            return False
            
    except Exception as e:
        print(f"⚠️  Error checking metadata: {e}, will update")
        return True

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
    """Deprecated - now using metadata file instead"""
    pass

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

def download_all_seasons_pbp(start_year=2020, end_year=None):
    """Download seasons from start_year to end_year (or current season if not specified)"""
    if end_year is None:
        end_year = get_current_season()
    
    all_data = []
    
    for year in range(start_year, end_year + 1):
        try:
            print(f"📥 Downloading {year} season play-by-play data...")
            url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.csv.gz"
            
            season_data = pd.read_csv(url, compression='gzip', low_memory=False)
            all_data.append(season_data)
            print(f"✅ Downloaded {len(season_data):,} plays for {year}")
            
        except Exception as e:
            print(f"❌ Failed to download {year} data: {e}")
            # Continue with other years
            continue
    
    if not all_data:
        return None
    
    # Combine all seasons
    combined_data = pd.concat(all_data, ignore_index=True, sort=True)
    print(f"🔄 Combined {len(combined_data):,} total plays ({start_year}-{end_year})")
    return combined_data

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

def is_lfs_pointer(filepath):
    """Check if file is an LFS pointer instead of actual data"""
    if not filepath.exists():
        return False
    
    try:
        # LFS pointers are always < 200 bytes, real files are 60-75MB
        if filepath.stat().st_size < 1000:
            return True
        return False
    except:
        return False

def main():
    print("🧠 Smart PBP Updater")
    print("=" * 50)

    current_season = get_current_season()
    print(f"🏈 Current season: {current_season}")
    
    DATA_DIR.mkdir(exist_ok=True)

    # Check if we have real data (not LFS pointer)
    has_real_pbp_file = PBP_FILE.exists() and not is_lfs_pointer(PBP_FILE)
    
    # Check metadata to see if update is needed (local optimization)
    if has_real_pbp_file and not should_update():
        print("✅ Data is up to date, skipping download")
        return 0

    # Strategy 1: Local development with real file - incremental update
    if has_real_pbp_file:
        print("📦 Local file found - downloading current season only")
        new_data = download_current_season_pbp(current_season)
        if new_data is None:
            print("❌ Download failed")
            return 1
        
        final_data = merge_with_existing_data(new_data)
    
    # Strategy 2: CI/CD with cached historical data - merge with current season
    elif HISTORICAL_CACHE.exists() and not is_lfs_pointer(HISTORICAL_CACHE):
        print("✓ Using cached historical data (2020-2024)")
        historical_data = pd.read_csv(HISTORICAL_CACHE, compression='gzip', sep='\t', low_memory=False)
        print(f"📚 Loaded {len(historical_data):,} historical plays")
        
        print(f"📥 Downloading current season ({current_season}) only...")
        current_data = download_current_season_pbp(current_season)
        if current_data is None:
            print("❌ Download failed")
            return 1
        
        # Merge historical cache with current season
        final_data = pd.concat([historical_data, current_data], ignore_index=True, sort=True)
        print(f"🔄 Combined {len(final_data):,} total plays")
    
    # Strategy 3: Fresh install or cache miss - download everything
    else:
        print("🌐 No cache found - downloading all seasons (2020-present)")
        
        # Build historical cache (2020-2024)
        print("📦 Building historical cache (2020-2024)...")
        historical_data = download_all_seasons_pbp(start_year=2020, end_year=2024)
        if historical_data is None:
            print("❌ Failed to download historical data")
            return 1
        
        # Save to cache for future runs
        historical_data.to_csv(HISTORICAL_CACHE, compression='gzip', index=False, sep='\t')
        print(f"💾 Cached {len(historical_data):,} historical plays")
        
        # Download current season
        current_data = download_current_season_pbp(current_season)
        if current_data is None:
            print("❌ Failed to download current season")
            return 1
        
        # Merge
        final_data = pd.concat([historical_data, current_data], ignore_index=True, sort=True)
        print(f"🔄 Combined {len(final_data):,} total plays")

    # Save updated data
    final_data.to_csv(PBP_FILE, compression='gzip', index=False, sep='\t')
    print(f"💾 Saved {len(final_data):,} plays to {PBP_FILE}")

    # Update metadata
    write_metadata(len(final_data))

    print("✅ PBP data update complete!")
    return 0

if __name__ == '__main__':
    sys.exit(main())