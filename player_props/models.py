"""
Player Props XGBoost Models
Train baseline models for predicting player prop over/under outcomes.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path('data_files')
MODELS_DIR = Path('player_props/models')
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Common prop betting lines - UPDATED TO REALISTIC VALUES BASED ON HISTORICAL PERFORMANCE
PROP_LINES = {
    'passing_yards': {
        'elite_qb': 255.0,      # 70th percentile - elite QBs
        'starter': 231.0,        # 60th percentile - average starter
        'backup': 175.5          # Backup/struggling QB (unchanged)
    },
    'rushing_yards': {
        'elite_rb': 85.5,        # Henry, Barkley (unchanged)
        'starter': 24.0,         # 60th percentile - much more realistic!
        'committee': 45.5        # RB in committee (unchanged)
    },
    'receiving_yards': {
        'elite_wr': 75.5,        # Top WRs (unchanged)
        'starter': 27.0,         # 60th percentile - much more realistic!
        'role_player': 35.5      # WR4+, TEs (unchanged)
    },
    'passing_tds': {
        'over': 1.0,             # 60th percentile - will QB throw 2+ TDs?
        'high': 2.0              # 70th percentile - will QB throw 3+ TDs?
    },
    'rush_rec_tds': {
        'anytime': 0.0           # 60th percentile - will player score a TD?
    }
}

# Training configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_GAMES_PLAYED = 5  # Minimum games to have reliable rolling stats

# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_player_stats(stat_type='passing'):
    """Load aggregated player stats."""
    file_map = {
        'passing': 'player_passing_stats.csv',
        'rushing': 'player_rushing_stats.csv',
        'receiving': 'player_receiving_stats.csv'
    }
    
    file_path = DATA_DIR / file_map[stat_type]
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        print(f"   Run 'python player_props/aggregators.py' first")
        return None
    
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {len(df):,} {stat_type} records")
    return df


def create_prop_targets(df, stat_col, line_values):
    """
    Create binary targets for different prop lines.
    
    Args:
        df: DataFrame with player stats
        stat_col: Column name for the stat (e.g., 'passing_yards')
        line_values: Dict of prop lines to create targets for
        
    Returns:
        DataFrame with added target columns
    """
    for line_name, line_value in line_values.items():
        target_col = f'over_{line_name}'
        df[target_col] = (df[stat_col] > line_value).astype(int)
    
    return df


def prepare_training_features(df, stat_type='passing'):
    """
    Prepare features for model training.
    
    Features include:
    - Rolling averages (L3, L5, L10)
    - Rolling std dev for consistency
    - Team strength indicators
    - Home/away
    - Recent trends
    """
    # Core stat column mapping
    stat_col_map = {
        'passing': 'passing_yards',
        'rushing': 'rushing_yards',
        'receiving': 'receiving_yards'
    }
    
    stat_col = stat_col_map[stat_type]
    
    # Filter to only games with valid rolling averages
    # (need at least 3 games for L3 average)
    rolling_col = f'{stat_col}_L3'
    df = df.dropna(subset=[rolling_col]).copy()
    
    # Base features (rolling averages already calculated by aggregator)
    feature_cols = [
        f'{stat_col}_L3',
        f'{stat_col}_L5', 
        f'{stat_col}_L10'
    ]
    
    # Add position-specific features
    if stat_type == 'passing':
        extra_cols = [
            'pass_tds_L3', 'pass_tds_L5',
            'completions_L3', 'completions_L5',
            'attempts_L3', 'attempts_L5'
        ]
    elif stat_type == 'rushing':
        extra_cols = [
            'rush_tds_L3', 'rush_tds_L5',
            'rush_attempts_L3', 'rush_attempts_L5'
        ]
    elif stat_type == 'receiving':
        extra_cols = [
            'rec_tds_L3', 'rec_tds_L5',
            'receptions_L3', 'receptions_L5',
            'targets_L3', 'targets_L5'
        ]
    
    # Only add features that exist in the dataframe
    extra_cols = [c for c in extra_cols if c in df.columns]
    feature_cols.extend(extra_cols)
    
    # Filter to only existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    print(f"📊 Using {len(feature_cols)} features: {feature_cols[:5]}...")
    
    return df, feature_cols


def prepare_td_training_features(df, stat_type='passing'):
    """
    Prepare features for TD model training.
    
    TD models use different features than yards models:
    - Focus on TD-related stats and volume metrics
    - Exclude yards (to avoid data leakage for TD predictions)
    """
    # Filter to only games with valid rolling TD averages
    td_col = 'pass_tds' if stat_type == 'passing' else ('rush_tds' if stat_type == 'rushing' else 'rec_tds')
    rolling_td_col = f'{td_col}_L3'
    df = df.dropna(subset=[rolling_td_col]).copy()
    
    # TD-specific features (no yards to avoid leakage)
    if stat_type == 'passing':
        feature_cols = [
            'pass_tds_L3', 'pass_tds_L5', 'pass_tds_L10',
            'completions_L3', 'completions_L5', 'completions_L10',
            'attempts_L3', 'attempts_L5', 'attempts_L10'
        ]
    elif stat_type == 'rushing':
        feature_cols = [
            'rush_tds_L3', 'rush_tds_L5', 'rush_tds_L10',
            'rush_attempts_L3', 'rush_attempts_L5', 'rush_attempts_L10'
        ]
    elif stat_type == 'receiving':
        feature_cols = [
            'rec_tds_L3', 'rec_tds_L5', 'rec_tds_L10',
            'receptions_L3', 'receptions_L5', 'receptions_L10',
            'targets_L3', 'targets_L5', 'targets_L10'
        ]
    
    # Filter to only existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    print(f"🏈 Using {len(feature_cols)} TD features: {feature_cols[:5]}...")
    
    return df, feature_cols


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_prop_model(df, features, target_col, model_name):
    """
    Train XGBoost model for a specific prop line.
    
    Args:
        df: DataFrame with features and target
        features: List of feature column names
        target_col: Target column name
        model_name: Name for saving model
        
    Returns:
        Trained model, metrics dict
    """
    # Remove rows with missing features or target
    df_clean = df[features + [target_col]].dropna()
    
    if len(df_clean) < 100:
        print(f"⚠️  Not enough data for {target_col}: {len(df_clean)} rows")
        return None, None
    
    X = df_clean[features]
    y = df_clean[target_col]
    
    # Check class balance
    class_dist = y.value_counts()
    print(f"\n📊 Target distribution for {target_col}:")
    print(f"   Over (1): {class_dist.get(1, 0):,} ({class_dist.get(1, 0)/len(y)*100:.1f}%)")
    print(f"   Under (0): {class_dist.get(0, 0):,} ({class_dist.get(0, 0)/len(y)*100:.1f}%)")
    
    # Skip if too imbalanced (less than 10% of either class)
    if min(class_dist.get(0, 0), class_dist.get(1, 0)) / len(y) < 0.1:
        print(f"⚠️  Skipping {target_col}: too imbalanced")
        return None, None
    
    # Train/test split (chronological would be better but requires date sorting)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Calculate scale_pos_weight for imbalanced classes
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Train XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'model_name': model_name,
        'target': target_col,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\n✅ {model_name} Results:")
    print(f"   Accuracy: {metrics['accuracy']:.3f}")
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall: {metrics['recall']:.3f}")
    print(f"   F1: {metrics['f1']:.3f}")
    print(f"   ROC-AUC: {metrics['roc_auc']:.3f}")
    
    # Save model
    model_path = MODELS_DIR / f'{model_name}.json'
    model.save_model(model_path)
    print(f"💾 Saved model to {model_path}")
    
    return model, metrics


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_all_models():
    """Train all player prop models."""
    print("=" * 70)
    print("🏈 NFL Player Props Model Training")
    print("=" * 70)
    
    all_metrics = []
    
    # ========================================================================
    # 1. PASSING YARDS MODELS
    # ========================================================================
    print("\n\n📊 PASSING YARDS MODELS")
    print("-" * 70)
    
    passing_df = load_player_stats('passing')
    if passing_df is not None:
        # Create targets for different QB tiers
        passing_df = create_prop_targets(
            passing_df, 
            'passing_yards',
            {
                'elite_qb': PROP_LINES['passing_yards']['elite_qb'],
                'starter': PROP_LINES['passing_yards']['starter']
            }
        )
        
        passing_df, pass_features = prepare_training_features(passing_df, 'passing')
        
        # Train models for each line
        for line_type in ['elite_qb', 'starter']:
            target_col = f'over_{line_type}'
            model_name = f'passing_yards_{line_type}'
            
            model, metrics = train_prop_model(
                passing_df, pass_features, target_col, model_name
            )
            if metrics:
                all_metrics.append(metrics)
    
    # ========================================================================
    # 1.5. PASSING TDS MODELS
    # ========================================================================
    print("\n\n🏈 PASSING TDS MODELS")
    print("-" * 70)
    
    if passing_df is not None:
        # Create targets for TD props
        passing_df = create_prop_targets(
            passing_df, 
            'pass_tds',
            PROP_LINES['passing_tds']
        )
        
        # Prepare TD-specific features (no yards leakage)
        passing_df, pass_td_features = prepare_td_training_features(passing_df, 'passing')
        
        # Train models for each TD line
        for line_type in PROP_LINES['passing_tds'].keys():
            target_col = f'over_{line_type}'
            model_name = f'passing_tds_{line_type}'
            
            model, metrics = train_prop_model(
                passing_df, pass_td_features, target_col, model_name
            )
            if metrics:
                all_metrics.append(metrics)
    
    # ========================================================================
    # 2. RUSHING YARDS MODELS
    # ========================================================================
    print("\n\n🏃 RUSHING YARDS MODELS")
    print("-" * 70)
    
    rushing_df = load_player_stats('rushing')
    if rushing_df is not None:
        rushing_df = create_prop_targets(
            rushing_df,
            'rushing_yards',
            {
                'elite_rb': PROP_LINES['rushing_yards']['elite_rb'],
                'starter': PROP_LINES['rushing_yards']['starter']
            }
        )
        
        rushing_df, rush_features = prepare_training_features(rushing_df, 'rushing')
        
        for line_type in ['elite_rb', 'starter']:
            target_col = f'over_{line_type}'
            model_name = f'rushing_yards_{line_type}'
            
            model, metrics = train_prop_model(
                rushing_df, rush_features, target_col, model_name
            )
            if metrics:
                all_metrics.append(metrics)
    
    # ========================================================================
    # 2.5. RUSHING TDS MODELS
    # ========================================================================
    print("\n\n🏃 RUSHING TDS MODELS")
    print("-" * 70)
    
    if rushing_df is not None:
        # Create targets for TD props
        rushing_df = create_prop_targets(
            rushing_df,
            'rush_tds',
            PROP_LINES['rush_rec_tds']
        )
        
        # Prepare TD-specific features (no yards leakage)
        rushing_df, rush_td_features = prepare_td_training_features(rushing_df, 'rushing')
        
        # Train TD model
        target_col = 'over_anytime'
        model_name = 'rushing_tds_anytime'
        
        model, metrics = train_prop_model(
            rushing_df, rush_td_features, target_col, model_name
        )
        if metrics:
            all_metrics.append(metrics)
    
    # ========================================================================
    # 3. RECEIVING YARDS MODELS
    # ========================================================================
    print("\n\n🤲 RECEIVING YARDS MODELS")
    print("-" * 70)
    
    receiving_df = load_player_stats('receiving')
    if receiving_df is not None:
        receiving_df = create_prop_targets(
            receiving_df,
            'receiving_yards',
            {
                'elite_wr': PROP_LINES['receiving_yards']['elite_wr'],
                'starter': PROP_LINES['receiving_yards']['starter']
            }
        )
        
        receiving_df, rec_features = prepare_training_features(receiving_df, 'receiving')
        
        for line_type in ['elite_wr', 'starter']:
            target_col = f'over_{line_type}'
            model_name = f'receiving_yards_{line_type}'
            
            model, metrics = train_prop_model(
                receiving_df, rec_features, target_col, model_name
            )
            if metrics:
                all_metrics.append(metrics)
    
    # ========================================================================
    # 3.5. RECEIVING TDS MODELS
    # ========================================================================
    print("\n\n🤲 RECEIVING TDS MODELS")
    print("-" * 70)
    
    if receiving_df is not None:
        # Create targets for TD props
        receiving_df = create_prop_targets(
            receiving_df,
            'rec_tds',
            PROP_LINES['rush_rec_tds']
        )
        
        # Prepare TD-specific features (no yards leakage)
        receiving_df, rec_td_features = prepare_td_training_features(receiving_df, 'receiving')
        
        # Train TD model
        target_col = 'over_anytime'
        model_name = 'receiving_tds_anytime'
        
        model, metrics = train_prop_model(
            receiving_df, rec_td_features, target_col, model_name
        )
        if metrics:
            all_metrics.append(metrics)
    
    # ========================================================================
    # SAVE SUMMARY METRICS
    # ========================================================================
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = MODELS_DIR / 'model_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\n\n💾 Saved metrics summary to {metrics_path}")
        
        # Save JSON version too
        metrics_json_path = MODELS_DIR / 'model_metrics.json'
        with open(metrics_json_path, 'w') as f:
            json.dump({
                'trained_at': datetime.now().isoformat(),
                'models': all_metrics
            }, f, indent=2)
        
        # Print summary table
        print("\n" + "=" * 70)
        print("📊 MODEL PERFORMANCE SUMMARY")
        print("=" * 70)
        print(metrics_df[['model_name', 'accuracy', 'f1', 'roc_auc']].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print(f"📁 Models saved to: {MODELS_DIR}")
    print("\nNext steps:")
    print("  1. Review metrics in player_props/models/model_metrics.csv")
    print("  2. Create prediction pipeline for upcoming games")
    print("  3. Integrate with Player Props UI page")
    print("=" * 70)


if __name__ == '__main__':
    train_all_models()
