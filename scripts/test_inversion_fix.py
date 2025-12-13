"""
Test if inverting predictions fixes the calibration
"""
import pandas as pd

df = pd.read_csv('data_files/nfl_games_historical_with_predictions.csv', sep='\t')

# Get training data
train_data = df[(df['season'] < 2025) & (df['underdogCovered'].notna())].copy()

print("TESTING PREDICTION INVERSION FIX")
print("=" * 80)

# Test current predictions
print("\n1. CURRENT PREDICTIONS (prob_underdogCovered as-is)")
print("-" * 80)

bins = [0, 0.4, 0.45, 0.5, 0.55, 0.6, 1.0]
labels = ['<40%', '40-45%', '45-50%', '50-55%', '55-60%', '60%+']
train_data['confidence_bin'] = pd.cut(train_data['prob_underdogCovered'], bins=bins, labels=labels)

current_calib = train_data.groupby('confidence_bin').agg({
    'prob_underdogCovered': 'mean',
    'underdogCovered': 'mean',
    'game_id': 'count'
}).round(3)
current_calib.columns = ['Predicted', 'Actual', 'Count']
current_calib['Error'] = (current_calib['Predicted'] - current_calib['Actual']).abs()

print(current_calib.to_string())
print(f"\nAverage Error: {current_calib['Error'].mean():.1%}")

# Test inverted predictions
print("\n\n2. INVERTED PREDICTIONS (1 - prob_underdogCovered)")
print("-" * 80)

train_data['prob_inverted'] = 1 - train_data['prob_underdogCovered']
train_data['confidence_bin_inv'] = pd.cut(train_data['prob_inverted'], bins=bins, labels=labels)

inverted_calib = train_data.groupby('confidence_bin_inv').agg({
    'prob_inverted': 'mean',
    'underdogCovered': 'mean',
    'game_id': 'count'
}).round(3)
inverted_calib.columns = ['Predicted', 'Actual', 'Count']
inverted_calib['Error'] = (inverted_calib['Predicted'] - inverted_calib['Actual']).abs()

print(inverted_calib.to_string())
print(f"\nAverage Error: {inverted_calib['Error'].mean():.1%}")

# Calculate ROI for inverted predictions
print("\n\n3. BETTING PERFORMANCE WITH INVERSION")
print("-" * 80)

for threshold in [0.50, 0.524, 0.55, 0.60]:
    bets = train_data[train_data['prob_inverted'] >= threshold]
    if len(bets) > 0:
        wins = bets['underdogCovered'].sum()
        win_rate = wins / len(bets)
        roi = (wins * 90.91 - (len(bets) - wins) * 100) / (len(bets) * 100) * 100
        
        print(f"\nBetting on games ≥{threshold:.1%} confidence (inverted):")
        print(f"  Games: {len(bets)}")
        print(f"  Win rate: {win_rate:.1%}")
        print(f"  ROI: {roi:+.1f}%")
        print(f"  Status: {'✅ PROFITABLE' if roi > 0 else '❌ LOSING'}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)

if inverted_calib['Error'].mean() < current_calib['Error'].mean():
    improvement = (current_calib['Error'].mean() - inverted_calib['Error'].mean()) * 100
    print(f"\n✅ INVERSION WORKS!")
    print(f"   Calibration error improves by {improvement:.1f} percentage points")
    print(f"\n   RECOMMENDATION: Invert all spread predictions immediately")
else:
    print(f"\n❌ INVERSION DOESN'T HELP")
    print(f"   Problem is deeper - need to fix target variable logic")
