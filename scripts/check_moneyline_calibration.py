import pandas as pd
import numpy as np

# Load predictions
df = pd.read_csv('data_files/nfl_games_historical_with_predictions.csv', sep='\t')

# Filter for completed games with actual results
completed = df[df['underdogWon'].notna()].copy()

print("=== MONEYLINE MODEL CALIBRATION CHECK ===\n")
print(f"Total completed games with moneyline results: {len(completed)}")

# Bin by predicted probability
bins = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
labels = ['0-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-100%']

completed['prob_bin'] = pd.cut(completed['prob_underdogWon'], bins=bins, labels=labels)

# Calculate actual win rate in each bin
calibration = completed.groupby('prob_bin').agg({
    'underdogWon': ['count', 'mean']
}).round(3)

calibration.columns = ['Count', 'Actual Win Rate']
calibration['Actual Win Rate'] = calibration['Actual Win Rate'] * 100

print("\nCALIBRATION TABLE:")
print(calibration)

# Check if model is inverted
print("\n=== INVERSION TEST ===")
print("\nIf model is CORRECT:")
print("  - Low predicted probability (0-30%) should have LOW actual win rate")
print("  - High predicted probability (60%+) should have HIGH actual win rate")

print("\nIf model is INVERTED:")
print("  - Low predicted probability (0-30%) should have HIGH actual win rate")
print("  - High predicted probability (60%+) should have LOW actual win rate")

# Calculate correlation
correlation = completed['prob_underdogWon'].corr(completed['underdogWon'])
print(f"\nCorrelation between predicted prob and actual outcome: {correlation:.3f}")
print(f"  - Positive correlation = Model is CORRECT")
print(f"  - Negative correlation = Model is INVERTED")

# Look at extremes
low_prob = completed[completed['prob_underdogWon'] < 0.3]
high_prob = completed[completed['prob_underdogWon'] >= 0.6]

print(f"\nLow probability (<30%): {len(low_prob)} games")
print(f"  Actual win rate: {low_prob['underdogWon'].mean():.1%}")

print(f"\nHigh probability (≥60%): {len(high_prob)} games")
print(f"  Actual win rate: {high_prob['underdogWon'].mean():.1%}")
