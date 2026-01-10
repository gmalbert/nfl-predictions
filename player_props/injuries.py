"""
NFL Injury Data Scraper
Scrapes injury reports from ESPN.com for player prop adjustments.
"""
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(__file__).parent.parent / 'data_files'
INJURIES_FILE = DATA_DIR / 'espn_injuries.csv'

# ESPN injury report URL
ESPN_INJURIES_URL = "https://www.espn.com/nfl/injuries"

# Request headers to avoid blocking
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

# ============================================================================
# SCRAPING FUNCTIONS
# ============================================================================

def scrape_espn_injuries():
    """
    Scrape NFL injury reports from ESPN.com

    Returns:
        pd.DataFrame: DataFrame with injury data
    """
    print("🔍 Scraping ESPN injury reports...")

    try:
        # Make request with headers
        response = requests.get(ESPN_INJURIES_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        injuries = []

        # Find all team injury tables
        # ESPN structures injuries by team in separate sections
        team_sections = soup.find_all('div', class_='ResponsiveTable')

        for section in team_sections:
            # Get team name
            team_header = section.find_previous('h2')
            if team_header:
                team_name = team_header.text.strip()
            else:
                # Try alternative team identification
                team_name = "Unknown"

            # Find injury table
            table = section.find('table')
            if not table:
                continue

            # Parse table rows
            rows = table.find_all('tr')[1:]  # Skip header row

            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 4:
                    try:
                        player_name = cols[0].text.strip()
                        position = cols[1].text.strip() if len(cols) > 1 else ""
                        injury_type = cols[2].text.strip() if len(cols) > 2 else ""
                        status = cols[3].text.strip() if len(cols) > 3 else ""

                        # Extract practice participation if available
                        practice_participation = ""
                        if len(cols) > 4:
                            practice_participation = cols[4].text.strip()

                        # Skip empty rows
                        if not player_name:
                            continue

                        injuries.append({
                            'player_name': player_name,
                            'position': position,
                            'injury_type': injury_type,
                            'status': status,
                            'practice_participation': practice_participation,
                            'team': team_name,
                            'source': 'ESPN'
                        })

                    except Exception as e:
                        print(f"⚠️  Error parsing row: {e}")
                        continue

        if injuries:
            df = pd.DataFrame(injuries)
            print(f"✅ Scraped {len(df)} injuries from {df['team'].nunique()} teams")
            return df
        else:
            print("⚠️  No injuries found - ESPN may have changed their page structure")
            return pd.DataFrame()

    except requests.RequestException as e:
        print(f"❌ Request error: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Scraping error: {e}")
        return pd.DataFrame()


def clean_injury_data(df):
    """
    Clean and standardize injury data

    Args:
        df (pd.DataFrame): Raw injury data

    Returns:
        pd.DataFrame: Cleaned injury data
    """
    if df.empty:
        return df

    # Standardize status values
    status_mapping = {
        'questionable': 'Questionable',
        'probable': 'Probable',
        'doubtful': 'Doubtful',
        'out': 'Out',
        'injured reserve': 'IR',
        'pup': 'PUP',
        'nf-inj': 'NFI',
        'suspended': 'Suspended'
    }

    df['status'] = df['status'].str.lower().map(status_mapping).fillna(df['status'])

    # Standardize practice participation
    practice_mapping = {
        'full': 'Full',
        'limited': 'Limited',
        'dnp': 'DNP',
        'did not participate': 'DNP',
        'out': 'Out'
    }

    df['practice_participation'] = df['practice_participation'].str.lower().map(practice_mapping).fillna(df['practice_participation'])

    # Clean player names (remove extra spaces, standardize format)
    df['player_name'] = df['player_name'].str.strip()

    # Add timestamp
    df['scraped_at'] = pd.Timestamp.now()

    return df


def save_injuries_to_csv(df, filename=None):
    """
    Save injury data to CSV

    Args:
        df (pd.DataFrame): Injury data
        filename (str): Output filename (optional)
    """
    if filename is None:
        filename = INJURIES_FILE

    try:
        df.to_csv(filename, index=False)
        print(f"💾 Saved {len(df)} injuries to {filename}")
    except Exception as e:
        print(f"❌ Error saving to CSV: {e}")


def load_cached_injuries(max_age_hours=6):
    """
    Load cached injury data if it's recent enough

    Args:
        max_age_hours (int): Maximum age of cached data in hours

    Returns:
        pd.DataFrame or None: Cached injury data, or None if too old/no cache
    """
    if not INJURIES_FILE.exists():
        return None

    try:
        df = pd.read_csv(INJURIES_FILE)
        df['scraped_at'] = pd.to_datetime(df['scraped_at'])

        # Check if data is recent enough
        age_hours = (pd.Timestamp.now() - df['scraped_at'].max()).total_seconds() / 3600

        if age_hours <= max_age_hours:
            print(f"📋 Using cached injuries ({age_hours:.1f} hours old)")
            return df
        else:
            print(f"📋 Cached injuries too old ({age_hours:.1f} hours) - refreshing")
            return None

    except Exception as e:
        print(f"⚠️  Error loading cached injuries: {e}")
        return None


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def get_injury_report(use_cache=True, max_cache_age_hours=6):
    """
    Get current NFL injury report

    Args:
        use_cache (bool): Whether to use cached data if available
        max_cache_age_hours (int): Maximum age of cached data

    Returns:
        pd.DataFrame: Current injury data
    """
    # Try to load cached data first
    if use_cache:
        cached_data = load_cached_injuries(max_cache_age_hours)
        if cached_data is not None:
            return cached_data

    # Scrape fresh data
    raw_data = scrape_espn_injuries()

    if not raw_data.empty:
        # Clean the data
        clean_data = clean_injury_data(raw_data)

        # Save to cache
        save_injuries_to_csv(clean_data)

        return clean_data
    else:
        # Return cached data as fallback if scraping fails
        print("⚠️  Scraping failed, trying cached data as fallback...")
        cached_fallback = load_cached_injuries(max_age_hours=24)  # Allow older data as fallback
        if cached_fallback is not None:
            return cached_fallback

        return pd.DataFrame()


def find_player_injury(player_name, injuries_df):
    """
    Find injury information for a specific player

    Args:
        player_name (str): Player name to search for
        injuries_df (pd.DataFrame): Injury data

    Returns:
        dict or None: Injury info for the player
    """
    if injuries_df.empty or not isinstance(player_name, str) or not player_name:
        return None

    # Try exact match first
    exact_match = injuries_df[injuries_df['player_name'] == player_name]
    if not exact_match.empty:
        return exact_match.iloc[0].to_dict()

    # Try partial match (last name)
    last_name = player_name.split()[-1] if player_name else ""
    if last_name:
        partial_matches = injuries_df[injuries_df['player_name'].str.contains(last_name, case=False, na=False)]
        if not partial_matches.empty:
            return partial_matches.iloc[0].to_dict()

    return None


# ============================================================================
# PREDICTION ADJUSTMENT FUNCTIONS
# ============================================================================

def adjust_prediction_for_injury(prediction, injury_info):
    """
    Adjust a player prop prediction based on injury status

    Args:
        prediction (dict): Prediction dictionary
        injury_info (dict): Injury information

    Returns:
        dict or None: Adjusted prediction, or None if player is out
    """
    if not injury_info:
        return prediction

    # Ensure status and practice are strings
    status = str(injury_info.get('status', '')).lower() if injury_info.get('status') is not None else ''
    practice = str(injury_info.get('practice_participation', '')).lower() if injury_info.get('practice_participation') is not None else ''

    # Player is out - remove prediction entirely
    if status in ['out', 'ir', 'pup', 'nfi', 'suspended'] or practice == 'out':
        return None

    # Questionable players - reduce confidence
    if status == 'questionable':
        reduction_factor = 0.80  # Reduce confidence by 20%
        prediction['confidence'] *= reduction_factor
        prediction['prob_over'] *= reduction_factor
        prediction['prob_under'] = 1 - prediction['prob_over']

        # Add injury note
        injury_type = injury_info.get('injury_type', 'Unknown')
        prediction['injury_note'] = f"⚠️ {injury_type} injury (Questionable)"

    # Doubtful players - reduce confidence more
    elif status == 'doubtful':
        reduction_factor = 0.70  # Reduce confidence by 30%
        prediction['confidence'] *= reduction_factor
        prediction['prob_over'] *= reduction_factor
        prediction['prob_under'] = 1 - prediction['prob_over']

        injury_type = injury_info.get('injury_type', 'Unknown')
        prediction['injury_note'] = f"❌ {injury_type} injury (Doubtful)"

    # Limited practice - slight reduction
    elif practice == 'limited':
        reduction_factor = 0.90  # Reduce confidence by 10%
        prediction['confidence'] *= reduction_factor
        prediction['prob_over'] *= reduction_factor
        prediction['prob_under'] = 1 - prediction['prob_over']

        prediction['injury_note'] = "🟡 Limited practice participation"

    return prediction


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_injury_summary(injuries_df):
    """
    Print summary of injury data

    Args:
        injuries_df (pd.DataFrame): Injury data
    """
    if injuries_df.empty:
        print("📋 No injury data available")
        return

    print("📋 Injury Report Summary:")
    print(f"   Total injuries: {len(injuries_df)}")
    print(f"   Teams affected: {injuries_df['team'].nunique()}")
    print(f"   Status breakdown: {injuries_df['status'].value_counts().to_dict()}")
    print(f"   Practice participation: {injuries_df['practice_participation'].value_counts().to_dict()}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🏥 NFL Injury Report Scraper")
    print("=" * 60)

    # Get injury data
    injuries = get_injury_report()

    if not injuries.empty:
        print_injury_summary(injuries)

        # Show sample injuries
        print("\n📋 Sample Injuries:")
        sample = injuries.head(10)[['player_name', 'team', 'status', 'injury_type', 'practice_participation']]
        print(sample.to_string(index=False))

    else:
        print("❌ No injury data retrieved")

    print("=" * 60)