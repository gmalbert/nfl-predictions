"""
Player Props Prediction Pipeline
Generate prop predictions for upcoming NFL games using trained XGBoost models.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path('data_files')
MODELS_DIR = Path('player_props/models')

# Prop lines matching training configuration
PROP_LINES = {
    'passing_yards': {
        'elite_qb': 275.5,
        'starter': 225.5
    },
    'rushing_yards': {
        'starter': 65.5
    },
    'receiving_yards': {
        'starter': 55.5
    }
}

# Minimum confidence threshold for recommendations
MIN_CONFIDENCE = 0.55  # 55% probability for a recommendation

# Position mapping for prop types
POSITION_MAP = {
    'QB': 'passing_yards',
    'RB': 'rushing_yards',
    'WR': 'receiving_yards',
    'TE': 'receiving_yards'
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_schedule():
    """Load upcoming games schedule."""
    schedule_path = DATA_DIR / 'nfl_schedule_2025.csv'
    if not schedule_path.exists():
        print(f"❌ Schedule not found: {schedule_path}")
        return None
    
    df = pd.read_csv(schedule_path)
    df['game_date'] = pd.to_datetime(df['date'])
    
    # Handle timezone
    if df['game_date'].dt.tz is None:
        df['game_date'] = df['game_date'].dt.tz_localize('UTC')
    else:
        df['game_date'] = df['game_date'].dt.tz_convert('UTC')
    
    # Try to get upcoming games
    now_utc = pd.Timestamp.now(tz='UTC')
    cutoff = now_utc - pd.Timedelta(hours=12)
    upcoming = df[df['game_date'] >= cutoff].copy()
    
    # If no upcoming games, use most recent week for demonstration
    if upcoming.empty:
        print("⚠️  No upcoming games. Using most recent week for demonstration...")
        max_week = df['week'].max()
        upcoming = df[df['week'] == max_week].copy()
    
    print(f"✅ Found {len(upcoming)} games (Week {upcoming['week'].iloc[0] if len(upcoming) > 0 else 'N/A'})")
    return upcoming


def load_player_stats():
    """Load all player stats files."""
    stats = {}
    
    for stat_type in ['passing', 'rushing', 'receiving']:
        file_map = {
            'passing': 'player_passing_stats.csv',
            'rushing': 'player_rushing_stats.csv',
            'receiving': 'player_receiving_stats.csv'
        }
        
        file_path = DATA_DIR / file_map[stat_type]
        if file_path.exists():
            stats[stat_type] = pd.read_csv(file_path)
            print(f"✅ Loaded {len(stats[stat_type]):,} {stat_type} records")
        else:
            print(f"⚠️  {stat_type} stats not found")
            stats[stat_type] = None
    
    return stats


def load_models():
    """Load trained XGBoost models."""
    models = {}
    
    # Map model names to their configurations
    model_configs = [
        ('passing_yards_elite_qb', 'passing_yards', 'elite_qb'),
        ('passing_yards_starter', 'passing_yards', 'starter'),
        ('rushing_yards_starter', 'rushing_yards', 'starter'),
        ('receiving_yards_starter', 'receiving_yards', 'starter')
    ]
    
    for model_name, prop_type, line_type in model_configs:
        model_path = MODELS_DIR / f'{model_name}.json'
        
        if model_path.exists():
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            
            # Derive stat type from prop_type (passing_yards -> passing)
            stat_type = prop_type.replace('_yards', '') if '_yards' in prop_type else prop_type
            
            models[model_name] = {
                'model': model,
                'prop_type': prop_type,
                'stat_type': stat_type,  # For matching with stats files
                'line_type': line_type,
                'line_value': PROP_LINES[prop_type][line_type]
            }
            print(f"✅ Loaded {model_name}")
        else:
            print(f"⚠️  Model not found: {model_name}")
    
    return models


# ============================================================================
# PLAYER IDENTIFICATION
# ============================================================================

def get_recent_starters(stats_df, team, n_games=3):
    """
    Get players who have recently played for a team.
    Uses last N games to identify likely starters.
    """
    if stats_df is None or stats_df.empty:
        return []
    
    # Filter to this team's recent games
    team_stats = stats_df[stats_df['team'] == team].copy()
    
    if team_stats.empty:
        return []
    
    # Get most recent games
    team_stats = team_stats.sort_values('game_date', ascending=False)
    recent_games = team_stats['game_id'].unique()[:n_games]
    recent_stats = team_stats[team_stats['game_id'].isin(recent_games)]
    
    # Find players with most activity
    player_activity = recent_stats.groupby('player_name').agg({
        'game_id': 'nunique',  # Games played
        'game_date': 'max'      # Last appearance
    }).reset_index()
    
    # Filter to players who played in at least 1 of last 3 games
    active_players = player_activity[player_activity['game_id'] >= 1]['player_name'].tolist()
    
    return active_players


def get_player_features(stats_df, player_name, team, stat_type):
    """
    Get latest feature values for a player.
    """
    if stats_df is None:
        return None
    
    # Get player's most recent game
    player_stats = stats_df[
        (stats_df['player_name'] == player_name) & 
        (stats_df['team'] == team)
    ].sort_values('game_date', ascending=False)
    
    if player_stats.empty:
        return None
    
    latest = player_stats.iloc[0]
    
    # Build feature dict based on stat type
    stat_col_map = {
        'passing': 'passing_yards',
        'rushing': 'rushing_yards',
        'receiving': 'receiving_yards'
    }
    
    stat_col = stat_col_map[stat_type]
    
    features = {
        f'{stat_col}_L3': latest.get(f'{stat_col}_L3'),
        f'{stat_col}_L5': latest.get(f'{stat_col}_L5'),
        f'{stat_col}_L10': latest.get(f'{stat_col}_L10')
    }
    
    # Add position-specific features
    if stat_type == 'passing':
        features.update({
            'pass_tds_L3': latest.get('pass_tds_L3'),
            'pass_tds_L5': latest.get('pass_tds_L5'),
            'completions_L3': latest.get('completions_L3'),
            'completions_L5': latest.get('completions_L5'),
            'attempts_L3': latest.get('attempts_L3'),
            'attempts_L5': latest.get('attempts_L5')
        })
    elif stat_type == 'rushing':
        features.update({
            'rush_tds_L3': latest.get('rush_tds_L3'),
            'rush_tds_L5': latest.get('rush_tds_L5'),
            'rush_attempts_L3': latest.get('rush_attempts_L3'),
            'rush_attempts_L5': latest.get('rush_attempts_L5')
        })
    elif stat_type == 'receiving':
        features.update({
            'rec_tds_L3': latest.get('rec_tds_L3'),
            'rec_tds_L5': latest.get('rec_tds_L5'),
            'receptions_L3': latest.get('receptions_L3'),
            'receptions_L5': latest.get('receptions_L5'),
            'targets_L3': latest.get('targets_L3'),
            'targets_L5': latest.get('targets_L5')
        })
    
    # Check if we have required features (L3 stats at minimum)
    if pd.isna(features[f'{stat_col}_L3']):
        return None
    
    return features, latest


# ============================================================================
# PREDICTION GENERATION
# ============================================================================

def predict_props_for_game(game_row, all_stats, models):
    """
    Generate prop predictions for all players in a game.
    """
    predictions = []
    
    teams = [game_row['home_team'], game_row['away_team']]
    game_date = game_row['game_date']
    week = game_row['week']
    
    for team in teams:
        opponent = game_row['away_team'] if team == game_row['home_team'] else game_row['home_team']
        is_home = team == game_row['home_team']
        
        # Process each stat type
        for stat_type, stats_df in all_stats.items():
            if stats_df is None:
                continue
            
            # Get recent starters for this team
            starters = get_recent_starters(stats_df, team)
            
            for player_name in starters:
                # Get player's latest features
                result = get_player_features(stats_df, player_name, team, stat_type)
                if result is None:
                    continue
                
                features, player_latest = result
                
                # Try each relevant model
                for model_name, model_info in models.items():
                    # Match model's stat_type with current stat_type
                    if model_info['stat_type'] != stat_type:
                        continue
                    
                    try:
                        # Prepare features as DataFrame
                        feature_df = pd.DataFrame([features])
                        
                        # Make prediction
                        prob_over = model_info['model'].predict_proba(feature_df)[0, 1]
                        
                        # Only include if meets minimum confidence
                        if prob_over >= MIN_CONFIDENCE or prob_over <= (1 - MIN_CONFIDENCE):
                            prediction = {
                                'week': week,
                                'game_date': game_date,
                                'player_name': player_name,
                                'team': team,
                                'opponent': opponent,
                                'is_home': is_home,
                                'prop_type': stat_type,
                                'line_type': model_info['line_type'],
                                'line_value': model_info['line_value'],
                                'prob_over': prob_over,
                                'prob_under': 1 - prob_over,
                                'recommendation': 'OVER' if prob_over >= MIN_CONFIDENCE else 'UNDER',
                                'confidence': max(prob_over, 1 - prob_over),
                                'avg_L3': features.get(f"{model_info['prop_type']}_L3"),
                                'avg_L5': features.get(f"{model_info['prop_type']}_L5"),
                                'avg_L10': features.get(f"{model_info['prop_type']}_L10")
                            }
                            predictions.append(prediction)
                    
                    except Exception as e:
                        # Skip if prediction fails (missing features, etc.)
                        continue
    
    return predictions


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def generate_predictions():
    """Main pipeline to generate all prop predictions."""
    print("=" * 70)
    print("🎯 NFL Player Props Prediction Pipeline")
    print("=" * 70)
    print()
    
    # Load data
    print("📂 Loading data...")
    schedule = load_schedule()
    if schedule is None or schedule.empty:
        print("❌ No upcoming games found")
        return
    
    all_stats = load_player_stats()
    models = load_models()
    
    if not models:
        print("❌ No models found. Run 'python player_props/models.py' first")
        return
    
    print()
    print("🔮 Generating predictions...")
    print("-" * 70)
    
    # Generate predictions for each game
    all_predictions = []
    
    for idx, game_row in schedule.iterrows():
        print(f"\n📅 {game_row['away_team']} @ {game_row['home_team']} (Week {game_row['week']})")
        
        game_predictions = predict_props_for_game(game_row, all_stats, models)
        all_predictions.extend(game_predictions)
        
        print(f"   Generated {len(game_predictions)} prop predictions")
    
    # Save predictions
    if all_predictions:
        pred_df = pd.DataFrame(all_predictions)
        
        # Sort by confidence descending
        pred_df = pred_df.sort_values('confidence', ascending=False)
        
        # Save to CSV
        output_path = DATA_DIR / 'player_props_predictions.csv'
        pred_df.to_csv(output_path, index=False)
        
        print()
        print("=" * 70)
        print(f"✅ Generated {len(pred_df)} total prop predictions")
        print(f"💾 Saved to: {output_path}")
        print()
        
        # Summary statistics
        print("📊 Prediction Summary:")
        print(f"   Total props: {len(pred_df)}")
        print(f"   High confidence (≥65%): {len(pred_df[pred_df['confidence'] >= 0.65])}")
        print(f"   Medium confidence (60-65%): {len(pred_df[(pred_df['confidence'] >= 0.60) & (pred_df['confidence'] < 0.65)])}")
        print(f"   Standard confidence (55-60%): {len(pred_df[(pred_df['confidence'] >= 0.55) & (pred_df['confidence'] < 0.60)])}")
        print()
        
        # Top recommendations
        print("🔥 Top 10 Recommendations:")
        print("-" * 70)
        top_props = pred_df.head(10)
        for idx, row in top_props.iterrows():
            print(f"{row['player_name']:20s} {row['team']:3s} | "
                  f"{row['prop_type']:15s} {row['recommendation']:5s} {row['line_value']:6.1f} | "
                  f"Conf: {row['confidence']:.1%} | L3: {row['avg_L3']:.1f}")
        
        print()
        print("=" * 70)
        print("✅ Prediction pipeline complete!")
        print("\nNext step: View predictions in Player Props UI page")
        print("=" * 70)
    else:
        print("\n⚠️  No predictions generated. Check if player stats are available.")


if __name__ == '__main__':
    generate_predictions()
