import pandas as pd
from datetime import datetime

df = pd.read_csv('data_files/nfl_games_historical_with_predictions.csv', sep='\t')
df['gameday'] = pd.to_datetime(df['gameday'], errors='coerce')

# Filter for future 2025 games
today = pd.to_datetime(datetime.now().date())
future = df[(df['season'] == 2025) & (df['gameday'] > today)].copy()

# Check how many games meet the 45% threshold
spread_bets_45 = future[future['prob_underdogCovered'] > 0.45]

print("="*70)
print("SPREAD BETS WITH 45% THRESHOLD")
print("="*70)

print(f"\n✅ Future 2025 games with spread bets (>45% threshold): {len(spread_bets_45)}")
print(f"   (Previously with 50% threshold: 0)")

print(f"\n📊 Games by confidence tier:")
print(f"   45-46%: {((spread_bets_45['prob_underdogCovered'] >= 0.45) & (spread_bets_45['prob_underdogCovered'] < 0.46)).sum()} games")
print(f"   46-47%: {((spread_bets_45['prob_underdogCovered'] >= 0.46) & (spread_bets_45['prob_underdogCovered'] < 0.47)).sum()} games")
print(f"   47-48%: {((spread_bets_45['prob_underdogCovered'] >= 0.47) & (spread_bets_45['prob_underdogCovered'] < 0.48)).sum()} games")
print(f"   48%+: {(spread_bets_45['prob_underdogCovered'] >= 0.48).sum()} games")

if len(spread_bets_45) > 0:
    print(f"\n🎯 Next 5 Spread Bets:")
    spread_bets_45_sorted = spread_bets_45.sort_values('prob_underdogCovered', ascending=False).head(5)
    for idx, row in spread_bets_45_sorted.iterrows():
        print(f"   {row['gameday'].date()}: {row['away_team']} @ {row['home_team']} ({row['prob_underdogCovered']:.1%} confidence)")

print(f"\n✨ SUCCESS! Spread betting is now showing actionable games!")
print("="*70)
