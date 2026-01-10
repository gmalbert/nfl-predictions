"""
Prediction Accuracy Tracking and Backtesting Module

This module provides tools to track prediction accuracy by comparing model predictions
against actual game results. It calculates hit rates, ROI analysis, and performance metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import nfl_data_py as nfl
from typing import Dict, List, Tuple, Optional


def collect_actual_results(week: int, season: int = 2025) -> tuple[pd.DataFrame, str]:
    """
    Fetch actual player stats for completed games in a given week.
    
    Tries two methods in order:
    1. Pre-aggregated weekly stats (fast, preferred)
    2. PBP data aggregation (slower, fallback)

    Args:
        week: NFL week number
        season: NFL season year

    Returns:
        Tuple of (DataFrame with actual player statistics, error message if any)
    """
    # Method 1: Try pre-aggregated weekly stats first (much faster)
    try:
        print(f"📊 Attempting to load pre-aggregated stats for {season} Season, Week {week}...")
        actual_stats = nfl.import_weekly_data([season], columns=[
            'player_name', 'week', 'season', 'passing_yards', 'passing_tds',
            'rushing_yards', 'rushing_tds', 'receiving_yards', 'receiving_tds',
            'receptions', 'completions', 'attempts'
        ])
        
        # Filter to specific week
        week_stats = actual_stats[actual_stats['week'] == week].copy()
        
        if week_stats.empty:
            print(f"⚠️  Pre-aggregated stats found but empty for Week {week}, trying PBP aggregation...")
            raise ValueError("Empty weekly stats")
        
        # Clean up player names
        week_stats['player_name'] = week_stats['player_name'].str.strip()
        
        print(f"✅ Collected pre-aggregated stats for {len(week_stats)} players in Week {week}")
        return week_stats, ""
        
    except Exception as e:
        # Method 2: Fallback to PBP aggregation
        print(f"   Pre-aggregated stats unavailable ({str(e)[:50]})")
        print(f"   Falling back to play-by-play data aggregation...")
        
        try:
            # Load play-by-play data directly from nflverse parquet files
            pbp_url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
            
            pbp_data = pd.read_parquet(pbp_url)
            
            # Filter to specific week
            week_pbp = pbp_data[pbp_data['week'] == week].copy()
            
            if week_pbp.empty:
                return pd.DataFrame(), f"No play-by-play data found for Week {week}, Season {season}"
            
            print(f"   Loaded {len(week_pbp):,} plays for Week {week}")
            
            # Aggregate passing stats
            passing_plays = week_pbp[week_pbp['pass'] == 1].copy()
            passing_stats = passing_plays.groupby('passer_player_name', observed=True).agg({
                'passing_yards': 'sum',
                'pass_touchdown': 'sum'
            }).reset_index()
            passing_stats.columns = ['player_name', 'passing_yards', 'passing_tds']
            
            # Aggregate rushing stats
            rushing_plays = week_pbp[week_pbp['rush'] == 1].copy()
            rushing_stats = rushing_plays.groupby('rusher_player_name', observed=True).agg({
                'rushing_yards': 'sum',
                'rush_touchdown': 'sum'
            }).reset_index()
            rushing_stats.columns = ['player_name', 'rushing_yards', 'rushing_tds']
            
            # Aggregate receiving stats
            receiving_plays = week_pbp[(week_pbp['pass'] == 1) & 
                                        (week_pbp['receiver_player_name'].notna())].copy()
            receiving_stats = receiving_plays.groupby('receiver_player_name', observed=True).agg({
                'receiving_yards': 'sum',
                'complete_pass': 'sum',  # Receptions
                'pass_touchdown': 'sum'   # Receiving TDs
            }).reset_index()
            receiving_stats.columns = ['player_name', 'receiving_yards', 'receptions', 'receiving_tds']
            
            # Merge all stats together
            combined_stats = pd.DataFrame()
            
            if not passing_stats.empty:
                combined_stats = passing_stats
            
            if not rushing_stats.empty:
                if combined_stats.empty:
                    combined_stats = rushing_stats
                else:
                    combined_stats = combined_stats.merge(
                        rushing_stats, on='player_name', how='outer'
                    )
            
            if not receiving_stats.empty:
                if combined_stats.empty:
                    combined_stats = receiving_stats
                else:
                    combined_stats = combined_stats.merge(
                        receiving_stats, on='player_name', how='outer'
                    )
            
            # Fill NaN with 0 and add metadata
            combined_stats = combined_stats.fillna(0)
            combined_stats['week'] = week
            combined_stats['season'] = season
            
            # Clean up player names
            combined_stats['player_name'] = combined_stats['player_name'].str.strip()
            
            print(f"✅ Collected PBP-aggregated stats for {len(combined_stats)} players in Week {week}")
            
            return combined_stats, ""
            
        except Exception as pbp_error:
            error_msg = f"Both methods failed - Pre-aggregated: {str(e)[:50]}, PBP: {str(pbp_error)[:50]}"
            print(f"❌ Error collecting actual results: {error_msg}")
            return pd.DataFrame(), error_msg


def calculate_hit_rate(predictions_df: pd.DataFrame, actuals_df: pd.DataFrame) -> Dict:
    """
    Calculate prediction accuracy (hit rate) by comparing predictions vs actual results.

    Args:
        predictions_df: DataFrame with model predictions
        actuals_df: DataFrame with actual game results

    Returns:
        Dictionary with accuracy metrics
    """
    if predictions_df.empty or actuals_df.empty:
        return {
            'overall_accuracy': 0.0,
            'by_confidence_tier': pd.Series(),
            'by_prop_type': pd.Series(),
            'total_predictions': 0,
            'detailed_results': pd.DataFrame()
        }

    # Merge predictions with actual results
    merged = predictions_df.merge(
        actuals_df,
        on=['player_name'],
        suffixes=('_pred', '_actual'),
        how='left'
    )

    # Filter out players with no actual stats (DNP, etc.)
    # Check if any of the key stat columns have data
    stat_columns = ['passing_yards', 'rushing_yards', 'receiving_yards']
    merged = merged.dropna(subset=stat_columns, how='all')

    if merged.empty:
        print("⚠️  No matching predictions found with actual results")
        return {
            'overall_accuracy': 0.0,
            'by_confidence_tier': pd.Series(),
            'by_prop_type': pd.Series(),
            'total_predictions': 0,
            'detailed_results': pd.DataFrame()
        }

    results = []

    for _, row in merged.iterrows():
        prop_type = row['prop_type']
        line_value = row['line_value']
        predicted_rec = row['recommendation']
        confidence = row['confidence']

        # Get actual stat value based on prop type
        stat_mapping = {
            'passing_yards': 'passing_yards',
            'passing_tds': 'passing_tds',
            'rushing_yards': 'rushing_yards',
            'rushing_tds': 'rushing_tds',
            'receiving_yards': 'receiving_yards',
            'receiving_tds': 'receiving_tds',
            'receptions': 'receptions'
        }

        stat_col = stat_mapping.get(prop_type)
        if not stat_col or pd.isna(row.get(stat_col, np.nan)):
            continue

        actual_value = row[stat_col]

        # Determine actual outcome
        actual_outcome = 'OVER' if actual_value > line_value else 'UNDER'

        # Check if prediction was correct
        hit = (predicted_rec == actual_outcome)

        results.append({
            'player_name': row['player_name'],
            'display_name': row.get('display_name', row['player_name']),
            'prop_type': prop_type,
            'line_value': line_value,
            'predicted': predicted_rec,
            'actual': actual_outcome,
            'actual_value': actual_value,
            'hit': hit,
            'confidence': confidence,
            'team': row.get('team', ''),
            'week': row.get('week_pred', 0)
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return {
            'overall_accuracy': 0.0,
            'by_confidence_tier': pd.Series(),
            'by_prop_type': pd.Series(),
            'total_predictions': 0,
            'detailed_results': pd.DataFrame()
        }

    # Calculate overall accuracy
    overall_hit_rate = results_df['hit'].mean()

    # Calculate accuracy by confidence tier
    confidence_bins = [0.50, 0.60, 0.65, 0.70, 0.75, 1.0]
    confidence_labels = ['50-60%', '60-65%', '65-70%', '70-75%', '75%+']

    results_df['confidence_tier'] = pd.cut(
        results_df['confidence'],
        bins=confidence_bins,
        labels=confidence_labels,
        include_lowest=True
    )

    by_confidence = results_df.groupby('confidence_tier', observed=True)['hit'].mean()

    # Calculate accuracy by prop type
    by_prop_type = results_df.groupby('prop_type', observed=True)['hit'].mean()

    print(f"📊 Accuracy Analysis Complete:")
    print(f"   Total predictions evaluated: {len(results_df)}")
    print(f"   Overall hit rate: {overall_hit_rate:.1%}")

    return {
        'overall_accuracy': overall_hit_rate,
        'by_confidence_tier': by_confidence,
        'by_prop_type': by_prop_type,
        'total_predictions': len(results_df),
        'detailed_results': results_df
    }


def calculate_roi(results_df: pd.DataFrame, odds: float = -110) -> Dict:
    """
    Calculate hypothetical ROI assuming standard betting odds.

    Args:
        results_df: DataFrame with prediction results (must have 'hit' column)
        odds: American odds format (default -110)

    Returns:
        Dictionary with ROI metrics
    """
    if results_df.empty or 'hit' not in results_df.columns:
        return {
            'total_bets': 0,
            'wins': 0,
            'losses': 0,
            'hit_rate': 0.0,
            'total_wagered': 0,
            'net_profit': 0,
            'roi': 0.0,
            'breakeven_rate': 52.4
        }

    total_bets = len(results_df)
    total_hits = results_df['hit'].sum()

    # Convert American odds to decimal for calculations
    if odds < 0:  # Negative odds (e.g., -110)
        decimal_odds = (100 / abs(odds)) + 1
    else:  # Positive odds (e.g., +150)
        decimal_odds = (odds / 100) + 1

    # Calculate P&L
    # Each bet: risk $110 to win $100 (or lose $110)
    risk_amount = abs(odds) if odds < 0 else 100
    win_amount = abs(odds) if odds > 0 else 100

    total_wagered = total_bets * risk_amount
    total_won = total_hits * win_amount
    total_lost = (total_bets - total_hits) * risk_amount

    net_profit = total_won - total_lost
    roi = (net_profit / total_wagered) * 100 if total_wagered > 0 else 0

    # Calculate breakeven rate
    breakeven_rate = (risk_amount / (risk_amount + win_amount)) * 100

    return {
        'total_bets': total_bets,
        'wins': total_hits,
        'losses': total_bets - total_hits,
        'hit_rate': total_hits / total_bets if total_bets > 0 else 0,
        'total_wagered': total_wagered,
        'net_profit': net_profit,
        'roi': roi,
        'breakeven_rate': breakeven_rate
    }


def profitable_subset(results_df: pd.DataFrame, min_confidence: float = 0.65) -> pd.DataFrame:
    """
    Calculate ROI for different confidence thresholds.

    Args:
        results_df: DataFrame with prediction results
        min_confidence: Minimum confidence threshold to test

    Returns:
        DataFrame with ROI analysis by confidence threshold
    """
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75]
    roi_by_threshold = {}

    for threshold in thresholds:
        subset = results_df[results_df['confidence'] >= threshold].copy()
        if len(subset) > 0:
            roi_metrics = calculate_roi(subset)
            roi_by_threshold[threshold] = roi_metrics

    return pd.DataFrame(roi_by_threshold).T


def save_accuracy_results(accuracy_metrics: Dict, week: int, filepath: Optional[str] = None) -> None:
    """
    Save accuracy results to a timestamped file for historical tracking.

    Args:
        accuracy_metrics: Dictionary returned by calculate_hit_rate
        week: NFL week number
        filepath: Optional custom filepath
    """
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"data_files/accuracy_results_week{week}_{timestamp}.json"

    # Convert Series to dict for JSON serialization
    results_to_save = accuracy_metrics.copy()
    results_to_save['by_confidence_tier'] = accuracy_metrics['by_confidence_tier'].to_dict()
    results_to_save['by_prop_type'] = accuracy_metrics['by_prop_type'].to_dict()

    # Convert detailed results to dict (without the DataFrame)
    results_to_save['detailed_results'] = accuracy_metrics['detailed_results'].to_dict('records')

    import json
    with open(filepath, 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)

    print(f"💾 Accuracy results saved to: {filepath}")


def load_accuracy_history() -> pd.DataFrame:
    """
    Load historical accuracy results from saved files.

    Returns:
        DataFrame with historical accuracy data
    """
    import glob
    import json

    accuracy_files = glob.glob("data_files/accuracy_results_week*.json")

    if not accuracy_files:
        return pd.DataFrame()

    history_data = []

    for filepath in accuracy_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Extract week and timestamp from filename
            # Filename format: accuracy_results_week{week}_{date}_{time}.json
            # Example: accuracy_results_week18_20260110_141431.json
            import os
            filename = os.path.basename(filepath)  # Remove directory path
            parts = filename.split('_')
            
            # Extract week number from 'week18' -> 18
            week_str = parts[2]  # 'week18'
            week = int(week_str.replace('week', ''))
            
            # Parse timestamp from date and time parts
            if len(parts) >= 5:
                date_str = parts[3]  # e.g., '20260110'
                time_str = parts[4].replace('.json', '')  # e.g., '141431'
                
                # Format as readable datetime: 20260110 -> 2026-01-10, 141431 -> 14:14:31
                try:
                    formatted_timestamp = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
                except:
                    formatted_timestamp = f"{date_str} {time_str}"
            else:
                formatted_timestamp = data.get('timestamp', 'Unknown')

            history_data.append({
                'week': week,
                'overall_accuracy': data.get('overall_accuracy', 0),
                'total_predictions': data.get('total_predictions', 0),
                'timestamp': formatted_timestamp
            })

        except Exception as e:
            print(f"⚠️  Error loading {filepath}: {e}")
            continue

    if not history_data:
        return pd.DataFrame()

    # Create DataFrame and deduplicate by week (keep most recent timestamp)
    df = pd.DataFrame(history_data)
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Group by week and keep the row with the most recent timestamp
    df_deduped = df.loc[df.groupby('week')['timestamp_dt'].idxmax()].copy()
    df_deduped = df_deduped.drop('timestamp_dt', axis=1)
    
    return df_deduped.sort_values('week')


def load_accuracy_results_for_week(week: int, season: int = 2025) -> Optional[Dict]:
    """
    Load the most recent accuracy results for a specific week.

    Args:
        week: NFL week number
        season: NFL season year

    Returns:
        Dictionary with accuracy results, or None if not found
    """
    import glob
    import json

    # Find all accuracy result files for this week
    pattern = f"data_files/accuracy_results_week{week}_*.json"
    accuracy_files = glob.glob(pattern)

    if not accuracy_files:
        return None

    # Find the most recent file (by timestamp in filename)
    most_recent_file = max(accuracy_files, key=lambda f: f.split('_')[-1].replace('.json', ''))

    try:
        with open(most_recent_file, 'r') as f:
            data = json.load(f)

        # Convert back to proper format
        data['by_confidence_tier'] = pd.Series(data['by_confidence_tier'])
        data['by_prop_type'] = pd.Series(data['by_prop_type'])
        data['detailed_results'] = pd.DataFrame(data['detailed_results'])

        return data

    except Exception as e:
        print(f"⚠️  Error loading {most_recent_file}: {e}")
        return None


# Example usage and testing functions
def run_weekly_accuracy_check(week: int, season: int = 2025) -> Dict:
    """
    Complete workflow to check accuracy for a given week.

    Args:
        week: NFL week number
        season: NFL season year

    Returns:
        Dictionary with complete accuracy analysis
    """
    print(f"🔍 Running accuracy check for Week {week}, Season {season}")
    print("=" * 60)

    # Load predictions for the week
    predictions_file = f"data_files/player_props_predictions_week{week}.csv"
    if not Path(predictions_file).exists():
        # Try the general predictions file
        predictions_file = "data_files/player_props_predictions.csv"

    if not Path(predictions_file).exists():
        print(f"❌ No predictions file found for week {week}")
        return {}

    predictions_df = pd.read_csv(predictions_file)
    print(f"📂 Loaded {len(predictions_df)} predictions")

    # Collect actual results
    actuals_df, error_msg = collect_actual_results(week, season)

    if actuals_df.empty:
        print(f"❌ No actual results available for week {week}: {error_msg}")
        return {}

    # Calculate accuracy
    accuracy_metrics = calculate_hit_rate(predictions_df, actuals_df)

    # Calculate ROI analysis
    if accuracy_metrics['total_predictions'] > 0:
        roi_metrics = calculate_roi(accuracy_metrics['detailed_results'])
        accuracy_metrics['roi_analysis'] = roi_metrics

        print(f"💰 ROI Analysis (at -110 odds):")
        print(f"   Hit Rate: {roi_metrics['hit_rate']:.1%}")
        print(f"   ROI: {roi_metrics['roi']:.1f}%")
        print(f"   Breakeven Rate: {roi_metrics['breakeven_rate']:.1f}%")

    # Save results
    save_accuracy_results(accuracy_metrics, week)

    print("=" * 60)
    print("✅ Weekly accuracy check complete!")

    return accuracy_metrics


if __name__ == '__main__':
    # Example: Check accuracy for last week
    current_week = 19  # Adjust based on current NFL week
    results = run_weekly_accuracy_check(current_week)

    if results:
        print("\n📈 Key Metrics:")
        print(f"Overall Accuracy: {results['overall_accuracy']:.1%}")
        print(f"Total Predictions: {results['total_predictions']}")

        if 'by_confidence_tier' in results and not results['by_confidence_tier'].empty:
            print("\n🎯 By Confidence Tier:")
            for tier, accuracy in results['by_confidence_tier'].items():
                print(f"  {tier}: {accuracy:.1%}")
