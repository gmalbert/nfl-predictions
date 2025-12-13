import pandas as pd
from datetime import datetime

df = pd.read_csv('data_files/nfl_games_historical_with_predictions.csv', sep='\t')
df['gameday'] = pd.to_datetime(df['gameday'], errors='coerce')

today = pd.to_datetime(datetime.now().date())
future = df[(df['season'] == 2025) & (df['gameday'] > today)].copy()

print("="*70)
print("SPREAD BETTING THRESHOLD ANALYSIS")
print("="*70)

print(f"\n📅 Future games (after {today.date()}): {len(future)}")

print(f"\n📊 Threshold Comparison:")
print(f"   Yesterday (45%+ confidence): {(future['prob_underdogCovered'] > 0.45).sum()} games")
print(f"   Old approach (50%+): {(future['prob_underdogCovered'] > 0.50).sum()} games")
print(f"   Current (52.4%+ EV-based): {(future['prob_underdogCovered'] >= 0.524).sum()} games")

print(f"\n🎯 Probability Distribution:")
print(f"   Max: {future['prob_underdogCovered'].max():.1%}")
print(f"   75th percentile: {future['prob_underdogCovered'].quantile(0.75):.1%}")
print(f"   Median: {future['prob_underdogCovered'].median():.1%}")
print(f"   Mean: {future['prob_underdogCovered'].mean():.1%}")

if 'ev_spread' in future.columns:
    print(f"\n💰 Expected Value (EV) Analysis:")
    print(f"   Games with EV > 0: {(future['ev_spread'] > 0).sum()}")
    print(f"   Games with 52.4%+ AND EV > 0: {((future['prob_underdogCovered'] >= 0.524) & (future['ev_spread'] > 0)).sum()}")
    print(f"   Max EV: ${future['ev_spread'].max():.2f}")
    
    # Show games closest to threshold
    print(f"\n🔍 Top 5 Games (by probability):")
    top5 = future.nlargest(5, 'prob_underdogCovered')
    for idx, row in top5.iterrows():
        print(f"   {row['gameday'].date()}: {row['away_team']} @ {row['home_team']}")
        print(f"      Prob: {row['prob_underdogCovered']:.1%}, EV: ${row['ev_spread']:.2f}")

print(f"\n⚠️ WHY NO BETS TODAY:")
print(f"   The EV-based system requires ≥52.4% confidence (breakeven for -110 odds)")
print(f"   All upcoming games are below this threshold (max: {future['prob_underdogCovered'].max():.1%})")
print(f"   Yesterday you may have seen bets with the 45% threshold")

print(f"\n💡 OPTIONS:")
print(f"   1. Keep 52.4% threshold (mathematically sound, no bad bets)")
print(f"   2. Lower to 50% (show more bets, but some may be -EV)")
print(f"   3. Lower to 45% (original threshold, show 4 games)")

print("="*70)
