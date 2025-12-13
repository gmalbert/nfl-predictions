import pandas as pd
from datetime import datetime

df = pd.read_csv('data_files/nfl_games_historical_with_predictions.csv', sep='\t')
df['gameday'] = pd.to_datetime(df['gameday'], errors='coerce')

# Filter for future 2025 games
today = pd.to_datetime(datetime.now().date())
future = df[(df['season'] == 2025) & (df['gameday'] > today)].copy()

print("="*70)
print("FEATURE ANALYSIS FOR FUTURE 2025 GAMES")
print("="*70)

print(f"\n📊 Key Features Statistics:")
print(f"\nhomeTeamWinPct:")
print(f"   Min: {future['homeTeamWinPct'].min():.3f}")
print(f"   Max: {future['homeTeamWinPct'].max():.3f}")
print(f"   Mean: {future['homeTeamWinPct'].mean():.3f}")

print(f"\nawayTeamWinPct:")
print(f"   Min: {future['awayTeamWinPct'].min():.3f}")
print(f"   Max: {future['awayTeamWinPct'].max():.3f}")
print(f"   Mean: {future['awayTeamWinPct'].mean():.3f}")

print(f"\nhomeTeamGamesPlayed:")
print(f"   Min: {future['homeTeamGamesPlayed'].min():.0f}")
print(f"   Max: {future['homeTeamGamesPlayed'].max():.0f}")
print(f"   Mean: {future['homeTeamGamesPlayed'].mean():.1f}")

print(f"\nhomeTeamSpreadCoveredPct:")
print(f"   Min: {future['homeTeamSpreadCoveredPct'].min():.3f}")
print(f"   Max: {future['homeTeamSpreadCoveredPct'].max():.3f}")
print(f"   Mean: {future['homeTeamSpreadCoveredPct'].mean():.3f}")

print(f"\n🎯 Prediction Probabilities:")
print(f"\nprob_underdogCovered:")
print(f"   Min: {future['prob_underdogCovered'].min():.3f}")
print(f"   Max: {future['prob_underdogCovered'].max():.3f}")
print(f"   Mean: {future['prob_underdogCovered'].mean():.3f}")

# Show distribution
print(f"\nDistribution of prob_underdogCovered:")
print(f"   < 40%: {(future['prob_underdogCovered'] < 0.40).sum()} games")
print(f"   40-45%: {((future['prob_underdogCovered'] >= 0.40) & (future['prob_underdogCovered'] < 0.45)).sum()} games")
print(f"   45-50%: {((future['prob_underdogCovered'] >= 0.45) & (future['prob_underdogCovered'] < 0.50)).sum()} games")
print(f"   ≥ 50%: {(future['prob_underdogCovered'] >= 0.50).sum()} games ⚠️")

print(f"\n💡 DIAGNOSIS:")
print(f"   All future 2025 games have win % and stats from the 2025 season so far")
print(f"   Games played ranges from {future['homeTeamGamesPlayed'].min():.0f} to {future['homeTeamGamesPlayed'].max():.0f}")
print(f"   But predictions are very conservative (max {future['prob_underdogCovered'].max():.1%})")
print(f"   The 50% threshold is too high for upcoming 2025 games!")

print("="*70)
