"""
Weather Impact Integration for NFL Player Props

Fetches weather data for outdoor NFL games and adjusts predictions accordingly.
Uses Open-Meteo API for historical weather data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import openmeteo_requests
from typing import Dict, Optional, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(__file__).parent.parent / 'data_files'

# NFL Stadium coordinates (latitude, longitude, is_dome)
# True = dome/retractable roof (no weather impact)
STADIUM_COORDINATES = {
    # AFC East
    'Gillette Stadium': (42.0911, -71.2642, False),  # New England Patriots
    'MetLife Stadium': (40.8135, -74.0745, False),   # NY Jets/Giants
    'Highmark Stadium': (40.4468, -80.0158, False),  # Buffalo Bills
    'Northwest Stadium': (38.9076, -76.8645, False), # Washington Commanders

    # AFC North
    'FirstEnergy Stadium': (41.5061, -81.6995, False),  # Cleveland Browns
    'Paycor Stadium': (39.0954, -84.5161, False),       # Cincinnati Bengals
    'M&T Bank Stadium': (39.2780, -76.6227, False),     # Baltimore Ravens
    'Acrisure Stadium': (40.4468, -80.0158, False),     # Pittsburgh Steelers

    # AFC South
    'Lucas Oil Stadium': (39.7601, -86.1639, False),    # Indianapolis Colts
    'TIAA Bank Stadium': (30.3239, -81.6373, False),    # Jacksonville Jaguars
    'Nissan Stadium': (36.1664, -86.7713, False),       # Tennessee Titans
    'Hard Rock Stadium': (25.9580, -80.2389, False),    # Miami Dolphins

    # AFC West
    'GEHA Field at Arrowhead Stadium': (39.0489, -94.4840, False),  # Kansas City Chiefs
    'Empower Field at Mile High': (39.7439, -105.0201, False),      # Denver Broncos
    'Allegiant Stadium': (36.0907, -115.1831, False),               # Las Vegas Raiders
    'SoFi Stadium': (33.9535, -118.3391, False),                    # LA Chargers/Rams

    # NFC East
    'Lincoln Financial Field': (39.9008, -75.1675, False),  # Philadelphia Eagles
    'AT&T Stadium': (32.7473, -97.0945, False),             # Dallas Cowboys
    'Northwest Stadium': (38.9076, -76.8645, False),        # Washington Commanders

    # NFC North
    'Soldier Field': (41.8623, -87.6167, False),           # Chicago Bears
    'Ford Field': (42.3400, -83.0456, True),               # Detroit Lions (dome)
    'Lambeau Field': (44.5013, -88.0622, False),           # Green Bay Packers
    'U.S. Bank Stadium': (44.9737, -93.2576, False),       # Minnesota Vikings

    # NFC South
    'Mercedes-Benz Stadium': (33.7554, -84.4009, False),   # Atlanta Falcons
    'TIAA Bank Stadium': (30.3239, -81.6373, False),       # Jacksonville Jaguars
    'Mercedes-Benz Superdome': (29.9508, -90.0812, True),  # New Orleans Saints (dome)
    'Raymond James Stadium': (27.9759, -82.5033, False),   # Tampa Bay Buccaneers

    # NFC West
    'State Farm Stadium': (33.5276, -112.2626, False),     # Arizona Cardinals
    'Levi\'s Stadium': (37.4030, -121.9695, False),        # San Francisco 49ers
    'Lumen Field': (47.5952, -122.3316, False),            # Seattle Seahawks
    'SoFi Stadium': (33.9535, -118.3391, False),           # LA Chargers/Rams

    # Alternative names
    'NRG Stadium': (29.6847, -95.4107, False),             # Houston Texans
    'Bank of America Stadium': (35.2258, -80.8528, False), # Carolina Panthers
}

# ============================================================================
# WEATHER FETCHING
# ============================================================================

def setup_openmeteo_client():
    """Setup Open-Meteo API client."""
    return openmeteo_requests.Client()

def get_game_weather(stadium_name: str, game_date) -> Dict:
    """
    Fetch weather forecast for game location and date.

    Args:
        stadium_name: Name of the stadium/venue
        game_date: Date string in YYYY-MM-DD format or pandas Timestamp

    Returns:
        Dict with weather data or None if dome/not found
    """
    # Convert game_date to string if it's a Timestamp
    if hasattr(game_date, 'strftime'):
        game_date = game_date.strftime('%Y-%m-%d')
    elif not isinstance(game_date, str):
        game_date = str(game_date)
    
    # Extract date part only if it has time component
    if len(game_date) > 10:
        game_date = game_date[:10]
    if stadium_name not in STADIUM_COORDINATES:
        # Try to find partial matches
        for stadium, coords in STADIUM_COORDINATES.items():
            if stadium_name.lower() in stadium.lower() or stadium.lower() in stadium_name.lower():
                latitude, longitude, is_dome = coords
                break
        else:
            # Default to outdoor if not found (most NFL games are outdoor)
            print(f"Stadium '{stadium_name}' not found in coordinates, assuming outdoor")
            latitude, longitude, is_dome = 39.8283, -98.5795, False  # Geographic center of US
    else:
        latitude, longitude, is_dome = STADIUM_COORDINATES[stadium_name]

    # If it's a dome, no weather impact
    if is_dome:
        return {
            'is_dome': True,
            'wind_mph': 0.0,
            'temp_f': 72.0,  # Standard indoor temp
            'precipitation_mm': 0.0,
            'humidity_pct': 50.0
        }

    try:
        client = setup_openmeteo_client()

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": game_date,
            "end_date": game_date,
            "hourly": ["temperature_2m", "precipitation", "relative_humidity_2m", "wind_speed_10m"],
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph"
        }

        # Use different URLs depending on whether we are seeking current or past weather
        if datetime.strptime(game_date, '%Y-%m-%d') < datetime.now():
            url = "https://archive-api.open-meteo.com/v1/archive"
        elif datetime.strptime(game_date, '%Y-%m-%d') >= datetime.now() and datetime.strptime(game_date, '%Y-%m-%d') <= (datetime.now() + timedelta(days=16)):
            url = "https://api.open-meteo.com/v1/forecast"
        else:
            print("Date too far in future for weather API")
            # Return neutral weather conditions
            return {
                'is_dome': False,
                'wind_mph': 5.0,  # Light wind
                'temp_f': 65.0,   # Mild temperature
                'precipitation_mm': 0.0,
                'humidity_pct': 60.0
            }

        responses = client.weather_api(url, params=params)
        response = responses[0]

        # Get hourly data
        hourly = response.Hourly()
        temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        precipitation = hourly.Variables(1).ValuesAsNumpy()
        relative_humidity_2m = hourly.Variables(2).ValuesAsNumpy()
        wind_speed_10m = hourly.Variables(3).ValuesAsNumpy()

        # Calculate daily aggregates (game time is typically afternoon/evening)
        # Use hours 12-23 (noon to 11pm) to cover typical game times
        game_hours_mask = slice(12, 24)  # Hours 12-23

        return {
            'is_dome': False,
            'latitude': latitude,
            'longitude': longitude,
            'wind_mph': float(np.mean(wind_speed_10m[game_hours_mask])),
            'temp_f': float(np.mean(temperature_2m[game_hours_mask])),
            'precipitation_mm': float(np.sum(precipitation[game_hours_mask])),
            'humidity_pct': float(np.mean(relative_humidity_2m[game_hours_mask]))
        }

    except Exception as e:
        print(f"Failed to fetch weather for {stadium_name} on {game_date}: {e}")
        # Return neutral weather conditions
        return {
            'is_dome': False,
            'wind_mph': 5.0,  # Light wind
            'temp_f': 65.0,   # Mild temperature
            'precipitation_mm': 0.0,
            'humidity_pct': 60.0
        }

# ============================================================================
# WEATHER ADJUSTMENTS
# ============================================================================

def adjust_for_weather(prob_over: float, prop_type: str, weather: Dict) -> float:
    """
    Adjust prediction probability based on weather conditions.

    Args:
        prob_over: Original probability of going OVER the line
        prop_type: Type of prop ('passing_yards', 'rushing_yards', etc.)
        weather: Weather data dict from get_game_weather()

    Returns:
        Adjusted probability (clamped to [0, 1])
    """
    if weather.get('is_dome', False):
        return prob_over  # No weather impact for dome games

    adjusted_prob = prob_over

    # High wind (>15 mph) reduces passing accuracy
    if prop_type in ['passing_yards', 'passing_tds'] and weather['wind_mph'] > 15:
        wind_penalty = min(0.15, (weather['wind_mph'] - 15) * 0.02)  # 2% penalty per mph over 15
        adjusted_prob *= (1.0 - wind_penalty)
        print(f"Wind penalty: -{wind_penalty:.1%} for {prop_type} (wind: {weather['wind_mph']:.1f} mph)")

    # Cold weather (<32F) reduces passing efficiency
    if prop_type in ['passing_yards', 'passing_tds'] and weather['temp_f'] < 32:
        cold_penalty = min(0.10, (32 - weather['temp_f']) * 0.02)  # 2% penalty per degree below 32
        adjusted_prob *= (1.0 - cold_penalty)
        print(f"Cold penalty: -{cold_penalty:.1%} for {prop_type} (temp: {weather['temp_f']:.1f}F)")

    # Heavy rain (>5mm) affects ball handling and field conditions
    if weather['precipitation_mm'] > 5.0:
        rain_penalty = min(0.08, weather['precipitation_mm'] * 0.015)  # 1.5% penalty per mm
        if prop_type in ['passing_yards', 'passing_tds']:
            adjusted_prob *= (1.0 - rain_penalty)
            print(f"Rain penalty: -{rain_penalty:.1%} for {prop_type} (precip: {weather['precipitation_mm']:.1f}mm)")
        elif prop_type in ['rushing_yards', 'receiving_yards']:
            # Rain can help rushing by making passes less effective
            adjusted_prob *= (1.0 + rain_penalty * 0.5)
            print(f"Rain bonus: +{rain_penalty * 0.5:.1%} for {prop_type} (precip: {weather['precipitation_mm']:.1f}mm)")

    # High humidity (>80%) can affect ball grip
    if weather['humidity_pct'] > 80 and prop_type in ['passing_tds', 'rushing_tds', 'receiving_yards']:
        humidity_penalty = min(0.05, (weather['humidity_pct'] - 80) * 0.001)  # 0.1% penalty per % over 80
        adjusted_prob *= (1.0 - humidity_penalty)
        print(f"Humidity penalty: -{humidity_penalty:.1%} for {prop_type} (humidity: {weather['humidity_pct']:.1f}%)")

    # Clamp to reasonable bounds
    return max(0.05, min(0.95, adjusted_prob))  # Never go below 5% or above 95%

# ============================================================================
# INTEGRATION FUNCTIONS
# ============================================================================

def get_weather_for_game(home_team: str, venue: str, game_date: str) -> Dict:
    """
    Get weather data for a specific NFL game.

    Args:
        home_team: Home team name
        venue: Stadium name
        game_date: Game date in YYYY-MM-DD format

    Returns:
        Weather data dict
    """
    return get_game_weather(venue, game_date)

def apply_weather_adjustments(predictions_df: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply weather adjustments to all predictions in a dataframe.

    Args:
        predictions_df: DataFrame with player prop predictions
        schedule_df: DataFrame with game schedule including venue info

    Returns:
        DataFrame with weather-adjusted predictions
    """
    adjusted_predictions = []

    for _, pred in predictions_df.iterrows():
        # Find the game in schedule
        game = schedule_df[
            (schedule_df['home_team'] == pred['team']) |
            (schedule_df['away_team'] == pred['team'])
        ]

        if game.empty:
            # No schedule match, use prediction as-is
            adjusted_predictions.append(pred)
            continue

        game_row = game.iloc[0]
        weather = get_weather_for_game(
            game_row['home_team'],
            game_row['venue'],
            game_row['date']
        )

        # Apply weather adjustment
        original_prob = pred['confidence']
        adjusted_prob = adjust_for_weather(original_prob, pred['prop_type'], weather)

        # Update prediction
        adjusted_pred = pred.copy()
        adjusted_pred['confidence'] = adjusted_prob
        adjusted_pred['weather_adjusted'] = True
        adjusted_pred['weather_conditions'] = f"{weather['temp_f']:.0f}F, {weather['wind_mph']:.1f} mph wind"

        adjusted_predictions.append(adjusted_pred)

    return pd.DataFrame(adjusted_predictions)

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test weather fetching
    test_weather = get_game_weather("SoFi Stadium", "2025-01-10")
    print("Test weather for SoFi Stadium:", test_weather)

    # Test weather adjustment
    test_prob = 0.65
    adjusted = adjust_for_weather(test_prob, 'passing_yards', test_weather)
    print("Original prob: {:.3f}, Adjusted: {:.3f}".format(test_prob, adjusted))