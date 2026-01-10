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

# Import injury functions
try:
    from .injuries import get_injury_report, find_player_injury, adjust_prediction_for_injury
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from injuries import get_injury_report, find_player_injury, adjust_prediction_for_injury

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(__file__).parent.parent / 'data_files'
MODELS_DIR = Path(__file__).parent / 'models'

# Prop lines matching training configuration - UPDATED TO REALISTIC VALUES
PROP_LINES = {
    'passing_yards': {
        'elite_qb': 275.0,   # For QBs averaging 280+ ypg (Stafford, Jones, Goff)
        'star_qb': 250.0,    # For QBs averaging 250-280 ypg (Prescott, Mahomes)
        'good_qb': 225.0,    # For QBs averaging 220-250 ypg (Lawrence, Herbert)
        'starter': 200.0     # For QBs averaging <220 ypg
    },
    'rushing_yards': {
        'elite_rb': 75.0,    # For RBs averaging 80+ ypg
        'star_rb': 55.0,     # For RBs averaging 60-80 ypg
        'good_rb': 40.0,     # For RBs averaging 45-60 ypg
        'starter': 24.0      # For RBs averaging <45 ypg
    },
    'receiving_yards': {
        'elite_wr': 70.0,    # For WRs averaging 75+ ypg
        'star_wr': 50.0,     # For WRs averaging 55-75 ypg
        'good_wr': 35.0,     # For WRs averaging 40-55 ypg
        'starter': 27.0      # For WRs averaging <40 ypg
    },
    'passing_tds': {
        'elite_qb': 2.5,     # Elite QBs: Over 2.5 TDs (will they throw 3+?)
        'star_qb': 1.5,      # Star QBs: Over 1.5 TDs (will they throw 2+?)
        'good_qb': 1.5,      # Good QBs: Over 1.5 TDs (will they throw 2+?)
        'starter': 0.5       # Starter QBs: Over 0.5 TDs (will they throw at least 1?)
    },
    'rushing_tds': {
        'elite_rb': 1.5,     # Elite RBs: Over 1.5 TDs (will they score 2+?)
        'star_rb': 0.5,      # Star RBs: Over 0.5 TDs (will they score at least 1?)
        'good_rb': 0.5,      # Good RBs: Over 0.5 TDs (will they score at least 1?)
        'anytime': 0.5       # Backup RBs: Over 0.5 TDs (will they score at least 1?)
    },
    'receiving_tds': {
        'elite_wr': 1.5,     # Elite WRs: Over 1.5 TDs (will they score 2+?)
        'star_wr': 0.5,      # Star WRs: Over 0.5 TDs (will they score at least 1?)
        'good_wr': 0.5,      # Good WRs: Over 0.5 TDs (will they score at least 1?)
        'anytime': 0.5       # Role WRs: Over 0.5 TDs (will they score at least 1?)
    },
    'receptions': {
        'elite_wr': 7.5,     # Elite WRs: Over 7.5 receptions
        'star_wr': 5.5,      # Star WRs: Over 5.5 receptions
        'good_wr': 4.5,      # Good WRs: Over 4.5 receptions
        'starter': 3.5       # Role players: Over 3.5 receptions
    }
}

# Minimum confidence threshold for recommendations
MIN_CONFIDENCE = 0.55  # 55% probability for a recommendation

# Position mapping for prop types
POSITION_MAP = {
    'QB': ['passing_yards', 'passing_tds'],
    'RB': ['rushing_yards', 'rushing_tds', 'receptions'],
    'WR': ['receiving_yards', 'receiving_tds', 'receptions'],
    'TE': ['receiving_yards', 'receiving_tds', 'receptions']
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_players():
    """Load player metadata with full names."""
    players_path = DATA_DIR / 'players.csv'
    if not players_path.exists():
        print(f"❌ Players data not found: {players_path}")
        return None
    return pd.read_csv(players_path)[['gsis_id', 'short_name', 'display_name', 'last_name', 'position']]


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
        # Passing Yards Models
        ('passing_yards_elite_qb', 'passing_yards', 'elite_qb'),
        ('passing_yards_star_qb', 'passing_yards', 'star_qb'),
        ('passing_yards_good_qb', 'passing_yards', 'good_qb'),
        ('passing_yards_starter', 'passing_yards', 'starter'),
        
        # Passing TDs Models
        ('passing_tds_elite_qb', 'passing_tds', 'elite_qb'),
        ('passing_tds_star_qb', 'passing_tds', 'star_qb'),
        ('passing_tds_good_qb', 'passing_tds', 'good_qb'),
        ('passing_tds_starter', 'passing_tds', 'starter'),
        
        # Rushing Yards Models
        ('rushing_yards_elite_rb', 'rushing_yards', 'elite_rb'),
        ('rushing_yards_star_rb', 'rushing_yards', 'star_rb'),
        ('rushing_yards_good_rb', 'rushing_yards', 'good_rb'),
        ('rushing_yards_starter', 'rushing_yards', 'starter'),
        
        # Rushing TDs Models
        ('rushing_tds_elite_rb', 'rushing_tds', 'elite_rb'),
        ('rushing_tds_star_rb', 'rushing_tds', 'star_rb'),
        ('rushing_tds_good_rb', 'rushing_tds', 'good_rb'),
        ('rushing_tds_anytime', 'rushing_tds', 'anytime'),
        
        # Receiving Yards Models
        ('receiving_yards_elite_wr', 'receiving_yards', 'elite_wr'),
        ('receiving_yards_star_wr', 'receiving_yards', 'star_wr'),
        ('receiving_yards_good_wr', 'receiving_yards', 'good_wr'),
        ('receiving_yards_starter', 'receiving_yards', 'starter'),
        
        # Receiving TDs Models
        ('receiving_tds_elite_wr', 'receiving_tds', 'elite_wr'),
        ('receiving_tds_star_wr', 'receiving_tds', 'star_wr'),
        ('receiving_tds_good_wr', 'receiving_tds', 'good_wr'),
        ('receiving_tds_anytime', 'receiving_tds', 'anytime'),
        
        # Receptions Models
        ('receptions_elite_wr', 'receptions', 'elite_wr'),
        ('receptions_star_wr', 'receptions', 'star_wr'),
        ('receptions_good_wr', 'receptions', 'good_wr'),
        ('receptions_starter', 'receptions', 'starter')
    ]
    
    for model_name, prop_type, line_type in model_configs:
        model_path = MODELS_DIR / f'{model_name}.json'
        
        if model_path.exists():
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            
            # Derive stat type and stat column from prop_type
            if prop_type in ['passing_yards', 'passing_tds']:
                stat_type = 'passing'
                stat_col = 'passing_yards' if prop_type == 'passing_yards' else 'pass_tds'
            elif prop_type in ['rushing_yards', 'rushing_tds']:
                stat_type = 'rushing'
                stat_col = 'rushing_yards' if prop_type == 'rushing_yards' else 'rush_tds'
            elif prop_type in ['receiving_yards', 'receiving_tds', 'receptions']:
                stat_type = 'receiving'
                if prop_type == 'receiving_yards':
                    stat_col = 'receiving_yards'
                elif prop_type == 'receiving_tds':
                    stat_col = 'rec_tds'
                elif prop_type == 'receptions':
                    stat_col = 'receptions'
                else:
                    continue
            else:
                continue
            
            models[model_name] = {
                'model': model,
                'prop_type': prop_type,
                'stat_type': stat_type,
                'stat_col': stat_col,
                'line_type': line_type,
                'line_value': PROP_LINES[prop_type][line_type]
            }
        else:
            print(f"⚠️  Model {model_name} not found, skipping")
    
    # Handle missing models with fallbacks
    if 'receiving_yards_elite_wr' not in models and 'receiving_yards_star_wr' in models:
        print("🔄 Using star_wr model as fallback for elite_wr")
        star_model = models['receiving_yards_star_wr']
        models['receiving_yards_elite_wr'] = {
            'model': star_model['model'],
            'prop_type': star_model['prop_type'],
            'stat_type': star_model['stat_type'],
            'stat_col': star_model['stat_col'],
            'line_type': 'elite_wr',
            'line_value': PROP_LINES['receiving_yards']['elite_wr']
        }
    
    # Add fallback for receptions elite_wr (use star_wr model)
    if 'receptions_elite_wr' not in models and 'receptions_star_wr' in models:
        print("🔄 Using star_wr model as fallback for receptions elite_wr")
        star_model = models['receptions_star_wr']
        models['receptions_elite_wr'] = {
            'model': star_model['model'],
            'prop_type': star_model['prop_type'],
            'stat_type': star_model['stat_type'],
            'stat_col': star_model['stat_col'],
            'line_type': 'elite_wr',
            'line_value': PROP_LINES['receptions']['elite_wr']
        }
    
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


def get_player_features(stats_df, player_name, team, prop_type, opponent=None, is_home=None, all_stats=None):
    """
    Get latest feature values for a player.
    
    Args:
        stats_df: Stats DataFrame for the stat type
        player_name: Player name
        team: Team name
        prop_type: Prop type (passing_yards, passing_tds, etc.)
        opponent: Opponent team name
        is_home: Whether the game is home for the player's team
        all_stats: All stats DataFrames for defense ranking calculation
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
    
    # Determine stat type and column
    if prop_type in ['passing_yards', 'passing_tds']:
        stat_type = 'passing'
        stat_col = 'passing_yards' if prop_type == 'passing_yards' else 'pass_tds'
    elif prop_type in ['rushing_yards', 'rushing_tds']:
        stat_type = 'rushing'
        stat_col = 'rushing_yards' if prop_type == 'rushing_yards' else 'rush_tds'
    elif prop_type in ['receiving_yards', 'receiving_tds', 'receptions']:
        stat_type = 'receiving'
        if prop_type == 'receiving_yards':
            stat_col = 'receiving_yards'
        elif prop_type == 'receiving_tds':
            stat_col = 'rec_tds'
        elif prop_type == 'receptions':
            stat_col = 'receptions'
        else:
            return None
    else:
        return None
    
    # Build base features
    features = {
        f'{stat_col}_L3': latest.get(f'{stat_col}_L3'),
        f'{stat_col}_L5': latest.get(f'{stat_col}_L5'),
        f'{stat_col}_L10': latest.get(f'{stat_col}_L10')
    }
    
    # Add position-specific features
    if stat_type == 'passing':
        if prop_type == 'passing_tds':
            # TD models get L10 for all features
            features.update({
                'pass_tds_L3': latest.get('pass_tds_L3'),
                'pass_tds_L5': latest.get('pass_tds_L5'),
                'pass_tds_L10': latest.get('pass_tds_L10'),
                'completions_L3': latest.get('completions_L3'),
                'completions_L5': latest.get('completions_L5'),
                'completions_L10': latest.get('completions_L10'),
                'attempts_L3': latest.get('attempts_L3'),
                'attempts_L5': latest.get('attempts_L5'),
                'attempts_L10': latest.get('attempts_L10')
            })
        else:
            # Yards models only get L3/L5 for auxiliary features
            features.update({
                'pass_tds_L3': latest.get('pass_tds_L3'),
                'pass_tds_L5': latest.get('pass_tds_L5'),
                'completions_L3': latest.get('completions_L3'),
                'completions_L5': latest.get('completions_L5'),
                'attempts_L3': latest.get('attempts_L3'),
                'attempts_L5': latest.get('attempts_L5')
            })
    elif stat_type == 'rushing':
        if prop_type == 'rushing_tds':
            # TD models get L10 for all features
            features.update({
                'rush_tds_L3': latest.get('rush_tds_L3'),
                'rush_tds_L5': latest.get('rush_tds_L5'),
                'rush_tds_L10': latest.get('rush_tds_L10'),
                'rush_attempts_L3': latest.get('rush_attempts_L3'),
                'rush_attempts_L5': latest.get('rush_attempts_L5'),
                'rush_attempts_L10': latest.get('rush_attempts_L10')
            })
        else:
            # Yards models only get L3/L5 for auxiliary features
            features.update({
                'rush_tds_L3': latest.get('rush_tds_L3'),
                'rush_tds_L5': latest.get('rush_tds_L5'),
                'rush_attempts_L3': latest.get('rush_attempts_L3'),
                'rush_attempts_L5': latest.get('rush_attempts_L5')
            })
    elif stat_type == 'receiving':
        if prop_type == 'receiving_tds':
            # TD models get L10 for all features
            features.update({
                'rec_tds_L3': latest.get('rec_tds_L3'),
                'rec_tds_L5': latest.get('rec_tds_L5'),
                'rec_tds_L10': latest.get('rec_tds_L10'),
                'receptions_L3': latest.get('receptions_L3'),
                'receptions_L5': latest.get('receptions_L5'),
                'receptions_L10': latest.get('receptions_L10'),
                'targets_L3': latest.get('targets_L3'),
                'targets_L5': latest.get('targets_L5'),
                'targets_L10': latest.get('targets_L10')
            })
        else:
            # Yards models only get L3/L5 for auxiliary features
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
    
    # Add matchup and situational features
    if opponent and all_stats:
        features['opponent_def_rank'] = get_opponent_defense_rank(opponent, stat_type, all_stats)
    else:
        features['opponent_def_rank'] = 16  # Default league average
    
    features['is_home'] = 1 if is_home else 0
    
    # Calculate days rest (simplified - would need player's last game date for accuracy)
    if hasattr(latest, 'game_date'):
        features['days_rest'] = calculate_days_rest(latest['game_date'])
    else:
        features['days_rest'] = 7  # Default
    
    return features, latest


def get_opponent_defense_rank(opponent, stat_type, all_stats):
    """
    Calculate opponent's defensive ranking for a stat type.
    Lower rank = better defense (harder matchup).
    """
    if stat_type == 'passing':
        # Get all QBs' passing yards against this defense
        stats_df = all_stats.get('passing')
        if stats_df is None or stats_df.empty:
            return 16  # Default to league average
        
        # Calculate average passing yards allowed by each defense
        avg_allowed = stats_df.groupby('opponent')['passing_yards'].mean()
        # Rank defenses (1 = stingiest, 32 = most generous)
        defense_ranks = avg_allowed.rank(method='min')
        return defense_ranks.get(opponent, 16)  # Default to league average
    
    elif stat_type == 'rushing':
        # Get all RBs' rushing yards against this defense
        stats_df = all_stats.get('rushing')
        if stats_df is None or stats_df.empty:
            return 16
        
        # Calculate average rushing yards allowed by each defense
        avg_allowed = stats_df.groupby('opponent')['rushing_yards'].mean()
        defense_ranks = avg_allowed.rank(method='min')
        return defense_ranks.get(opponent, 16)
    
    elif stat_type == 'receiving':
        # Get all WRs/TEs' receiving yards against this defense
        stats_df = all_stats.get('receiving')
        if stats_df is None or stats_df.empty:
            return 16
        
        # Calculate average receiving yards allowed by each defense
        avg_allowed = stats_df.groupby('opponent')['receiving_yards'].mean()
        defense_ranks = avg_allowed.rank(method='min')
        return defense_ranks.get(opponent, 16)
    
    return 16  # Default middle rank


def calculate_days_rest(game_date):
    """
    Calculate days of rest before a game.
    """
    from datetime import datetime, timezone
    
    if pd.isna(game_date):
        return 7  # Default to 1 week rest
    
    # Convert to datetime if needed
    if isinstance(game_date, str):
        game_date = pd.to_datetime(game_date)
    
    # Ensure game_date is timezone-aware
    if game_date.tz is None:
        game_date = game_date.tz_localize('UTC')
    
    # Get current time in UTC
    now = pd.Timestamp.now(tz='UTC')
    
    # Calculate days since last game (assuming game_date is next game)
    # This is a simplification - in reality we'd need the player's last game date
    days_rest = (game_date - now).days
    
    # Clamp to reasonable range
    return max(1, min(days_rest, 14))


# ============================================================================
# PREDICTION GENERATION
# ============================================================================

def get_player_position(player_name):
    """Get player position from players database."""
    players_df = load_players()
    if players_df is not None:
        player_record = players_df[players_df['short_name'] == player_name]
        if not player_record.empty:
            # If multiple players with same short_name, prefer QB for passing props, etc.
            # For now, just return the first match
            record = player_record.iloc[0]
            position = record['position']
            display_name = record['display_name']
            # Ensure we return strings or None, not NaN
            position = position if isinstance(position, str) else None
            display_name = display_name if isinstance(display_name, str) else None
            return position, display_name
    return None, None


def get_player_performance_tier(player_name, prop_type, all_stats):
    """
    Determine player's performance tier based on recent stats.
    Returns appropriate line_type for prop betting.
    """
    if prop_type == 'passing_yards':
        # Get relevant stats DataFrame
        stats_df = all_stats.get('passing')
        if stats_df is None or stats_df.empty:
            return 'starter'
        
        # Get player's recent performance (last 8 games)
        player_stats = stats_df[stats_df['player_name'] == player_name].copy()
        if player_stats.empty:
            return 'starter'
        
        player_stats = player_stats.sort_values('game_date', ascending=False)
        recent_stats = player_stats.head(8)  # Last 8 games
        
        if len(recent_stats) < 3:  # Need at least 3 games
            return 'starter'
        
        # Calculate average performance
        avg_performance = recent_stats['passing_yards'].mean()
        
        if avg_performance >= 280:
            return 'elite_qb'
        elif avg_performance >= 250:
            return 'star_qb'
        elif avg_performance >= 220:
            return 'good_qb'
        else:
            return 'starter'
    
    elif prop_type == 'passing_tds':
        # Get relevant stats DataFrame
        stats_df = all_stats.get('passing')
        if stats_df is None or stats_df.empty:
            return 'starter'
        
        # Get player's recent performance (last 8 games)
        player_stats = stats_df[stats_df['player_name'] == player_name].copy()
        if player_stats.empty:
            return 'starter'
        
        player_stats = player_stats.sort_values('game_date', ascending=False)
        recent_stats = player_stats.head(8)  # Last 8 games
        
        if len(recent_stats) < 3:  # Need at least 3 games
            return 'starter'
        
        # Calculate average performance
        avg_performance = recent_stats['pass_tds'].mean()
        
        if avg_performance >= 2.0:
            return 'elite_qb'
        elif avg_performance >= 1.5:
            return 'star_qb'
        elif avg_performance >= 1.0:
            return 'good_qb'
        else:
            return 'starter'
    
    elif prop_type == 'rushing_yards':
        # Get relevant stats DataFrame
        stats_df = all_stats.get('rushing')
        if stats_df is None or stats_df.empty:
            return 'starter'
        
        # Get player's recent performance (last 8 games)
        player_stats = stats_df[stats_df['player_name'] == player_name].copy()
        if player_stats.empty:
            return 'starter'
        
        player_stats = player_stats.sort_values('game_date', ascending=False)
        recent_stats = player_stats.head(8)  # Last 8 games
        
        if len(recent_stats) < 3:  # Need at least 3 games
            return 'starter'
        
        # Calculate average performance
        avg_performance = recent_stats['rushing_yards'].mean()
        
        if avg_performance >= 80:
            return 'elite_rb'
        elif avg_performance >= 60:
            return 'star_rb'
        elif avg_performance >= 45:
            return 'good_rb'
        else:
            return 'starter'
    
    elif prop_type == 'rushing_tds':
        # Get relevant stats DataFrame
        stats_df = all_stats.get('rushing')
        if stats_df is None or stats_df.empty:
            return 'anytime'
        
        # Get player's recent performance (last 8 games)
        player_stats = stats_df[stats_df['player_name'] == player_name].copy()
        if player_stats.empty:
            return 'anytime'
        
        player_stats = player_stats.sort_values('game_date', ascending=False)
        recent_stats = player_stats.head(8)  # Last 8 games
        
        if len(recent_stats) < 3:  # Need at least 3 games
            return 'anytime'
        
        # Calculate average performance
        avg_performance = recent_stats['rush_tds'].mean()
        
        if avg_performance >= 1.0:
            return 'elite_rb'
        elif avg_performance >= 0.7:
            return 'star_rb'
        elif avg_performance >= 0.4:
            return 'good_rb'
        else:
            return 'anytime'
    
    elif prop_type == 'receiving_yards':
        # Get relevant stats DataFrame
        stats_df = all_stats.get('receiving')
        if stats_df is None or stats_df.empty:
            return 'starter'
        
        # Get player's recent performance (last 8 games)
        player_stats = stats_df[stats_df['player_name'] == player_name].copy()
        if player_stats.empty:
            return 'starter'
        
        player_stats = player_stats.sort_values('game_date', ascending=False)
        recent_stats = player_stats.head(8)  # Last 8 games
        
        if len(recent_stats) < 3:  # Need at least 3 games
            return 'starter'
        
        # Calculate average performance
        avg_performance = recent_stats['receiving_yards'].mean()
        
        if avg_performance >= 75:
            return 'elite_wr'
        elif avg_performance >= 55:
            return 'star_wr'
        elif avg_performance >= 40:
            return 'good_wr'
        else:
            return 'starter'
    
    elif prop_type == 'receiving_tds':
        # Get relevant stats DataFrame
        stats_df = all_stats.get('receiving')
        if stats_df is None or stats_df.empty:
            return 'anytime'
        
        # Get player's recent performance (last 8 games)
        player_stats = stats_df[stats_df['player_name'] == player_name].copy()
        if player_stats.empty:
            return 'anytime'
        
        player_stats = player_stats.sort_values('game_date', ascending=False)
        recent_stats = player_stats.head(8)  # Last 8 games
        
        if len(recent_stats) < 3:  # Need at least 3 games
            return 'anytime'
        
        # Calculate average performance
        avg_performance = recent_stats['rec_tds'].mean()
        
        if avg_performance >= 1.0:
            return 'elite_wr'
        elif avg_performance >= 0.7:
            return 'star_wr'
        elif avg_performance >= 0.4:
            return 'good_wr'
        else:
            return 'anytime'
    
    elif prop_type == 'receptions':
        # Get relevant stats DataFrame
        stats_df = all_stats.get('receiving')
        if stats_df is None or stats_df.empty:
            return 'starter'
        
        # Get player's recent performance (last 8 games)
        player_stats = stats_df[stats_df['player_name'] == player_name].copy()
        if player_stats.empty:
            return 'starter'
        
        player_stats = player_stats.sort_values('game_date', ascending=False)
        recent_stats = player_stats.head(8)  # Last 8 games
        
        if len(recent_stats) < 3:  # Need at least 3 games
            return 'starter'
        
        # Calculate average performance
        avg_performance = recent_stats['receptions'].mean()
        
        if avg_performance >= 7.5:
            return 'elite_wr'
        elif avg_performance >= 5.5:
            return 'star_wr'
        elif avg_performance >= 4.5:
            return 'good_wr'
        else:
            return 'starter'
    
    # Default fallback
    return 'starter'


def predict_props_for_game(game_row, all_stats, models):
    """
    Generate prop predictions for all players in a game.
    """
    predictions = []
    
    teams = [game_row['home_team'], game_row['away_team']]
    game_date = game_row['game_date']
    week = game_row['week']
    
    # Load injury data for adjustment
    injuries_df = get_injury_report()
    
    for team in teams:
        opponent = game_row['away_team'] if team == game_row['home_team'] else game_row['home_team']
        is_home = team == game_row['home_team']
        
        # Get recent starters for this team (from any stat type)
        all_starters = set()
        for stat_type, stats_df in all_stats.items():
            if stats_df is not None:
                team_starters = get_recent_starters(stats_df, team)
                all_starters.update(team_starters)
        
        for player_name in all_starters:
            # Skip if player_name is not a valid string
            if not isinstance(player_name, str) or not player_name:
                continue
            
            # Get player position and display name
            player_position, player_display_name = get_player_position(player_name)
            
            # Get all prop types this player is eligible for based on position
            eligible_props = []
            if player_position and player_position in POSITION_MAP:
                eligible_props = POSITION_MAP[player_position]
            
            # For each eligible prop type, determine tier and use appropriate model
            for prop_type in eligible_props:
                # Get player's performance tier for this prop type
                player_tier = get_player_performance_tier(player_name, prop_type, all_stats)
                
                # Find the model that matches this tier
                matching_model = None
                for model_name, model_info in models.items():
                    if (model_info['prop_type'] == prop_type and 
                        model_info['line_type'] == player_tier):
                        matching_model = (model_name, model_info)
                        break
                
                # Skip if no matching model found
                if matching_model is None:
                    continue
                    
                model_name, model_info = matching_model
                stat_type = model_info['stat_type']
                
                # Get stats DataFrame for this stat type
                stats_df = all_stats.get(stat_type)
                if stats_df is None:
                    continue
                
                # Get player's latest features for this prop type
                result = get_player_features(stats_df, player_name, team, prop_type, opponent, is_home, all_stats)
                if result is None:
                    continue
                
                features, player_latest = result
                
                # Get tier-appropriate line value
                line_value = PROP_LINES[prop_type].get(player_tier, 0.5)  # Default to 0.5 if tier not found
                
                # Use the model's trained line_type for prediction, but display the tier-appropriate line
                model_line_value = model_info['line_value']
                
                try:
                    # Prepare features as DataFrame
                    feature_df = pd.DataFrame([features])
                    
                    # Make prediction using model's trained threshold
                    prob_over = model_info['model'].predict_proba(feature_df)[0, 1]
                    
                    # For star players with tier-appropriate lines, adjust recommendations
                    stat_col = model_info['stat_col']
                    recent_avg = features.get(f"{stat_col}_L5", 0)  # Last 5 games average
                    
                    # If player is a star (averages much more than the line), force OVER
                    if player_tier in ['elite_rb', 'star_rb', 'elite_wr', 'star_wr'] and recent_avg > line_value * 1.3:
                        prob_over = max(prob_over, 0.75)  # Ensure high confidence OVER for stars
                    # If player averages much less than the line, force UNDER
                    elif recent_avg < line_value * 0.6:
                        prob_over = min(prob_over, 0.25)  # Ensure high confidence UNDER
                    
                    # Only include if meets minimum confidence
                    if prob_over >= MIN_CONFIDENCE or prob_over <= (1 - MIN_CONFIDENCE):
                        prediction = {
                            'week': week,
                            'game_date': game_date,
                            'player_name': player_name,
                            'display_name': player_display_name or player_name,  # Use display_name if available, fallback to short_name
                            'position': player_position or 'Unknown',
                            'team': team,
                            'opponent': opponent,
                            'is_home': is_home,
                            'prop_type': prop_type,
                            'line_type': player_tier,  # Use performance-based tier
                            'line_value': line_value,  # Use tier-appropriate line
                            'prob_over': prob_over,
                            'prob_under': 1 - prob_over,
                            'recommendation': 'OVER' if prob_over >= MIN_CONFIDENCE else 'UNDER',
                            'confidence': max(prob_over, 1 - prob_over),
                            'opponent_def_rank': features.get('opponent_def_rank', 16),  # Add defensive ranking
                            'avg_L3': features.get(f"{model_info['stat_col']}_L3"),
                            'avg_L5': features.get(f"{model_info['stat_col']}_L5"),
                            'avg_L10': features.get(f"{model_info['stat_col']}_L10")
                        }
                        
                        # Check for injuries and adjust prediction
                        # Try matching with display_name first, then player_name
                        display_name = player_display_name if player_display_name and isinstance(player_display_name, str) else player_name
                        if isinstance(display_name, str):
                            injury_info = find_player_injury(display_name, injuries_df)
                            if not injury_info and display_name != player_name:
                                injury_info = find_player_injury(player_name, injuries_df)
                        else:
                            injury_info = None
                        
                        if injury_info:
                            adjusted_prediction = adjust_prediction_for_injury(prediction, injury_info)
                            if adjusted_prediction is None:
                                # Player is out - skip this prediction
                                continue
                            prediction = adjusted_prediction
                        
                        predictions.append(prediction)
                    
                except Exception as e:
                    # Skip if prediction fails (missing features, etc.)
                    print(f"⚠️  Prediction failed for {player_name} {prop_type}: {e}")
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
