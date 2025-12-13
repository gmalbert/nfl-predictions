import pandas as pd
from datetime import datetime

df = pd.read_csv('data_files/nfl_games_historical_with_predictions.csv', sep='\t')
df['gameday'] = pd.to_datetime(df['gameday'])

today = datetime(2025, 12, 13)
future = df[(df['gameday'] > today) & (df['season'] == 2025)]

print(f'Games after Dec 13, 2025 (2025 season): {len(future)}')
print(f'Max confidence: {future["prob_underdogCovered"].max():.1%}')
print(f'Games >=52.4%: {len(future[future["prob_underdogCovered"] >= 0.524])}')
print(f'Games >=45%: {len(future[future["prob_underdogCovered"] >= 0.45])}')

print('\nTop 10 Upcoming Games:')
top = future.nlargest(10, 'prob_underdogCovered')[['gameday', 'week', 'away_team', 'home_team', 'prob_underdogCovered', 'ev_spread', 'spread_line']]
for _, g in top.iterrows():
    date = g['gameday'].strftime('%m/%d')
    matchup = f"{g['away_team']} @ {g['home_team']}"
    prob = g['prob_underdogCovered'] * 100
    ev = g['ev_spread']
    spread = g['spread_line']
    status = '✅ +EV' if ev > 0 else '⚠️ -EV'
    print(f"  {date} Wk{g['week']:2.0f}: {matchup:25s} | {prob:4.1f}% | ${ev:6.2f} {status} | Spread: {spread:+.1f}")

# Check if momentum data exists for these games
print(f"\nMomentum Data Check:")
sample = future.iloc[0]
if 'homeTeamLast3WinPct' in future.columns:
    print(f"  Home team last 3 win%: {sample['homeTeamLast3WinPct']:.1%}")
    print(f"  Away team last 3 win%: {sample['awayTeamLast3WinPct']:.1%}")
    print(f"  ✅ Momentum features are populated")
else:
    print(f"  ❌ Momentum features missing")
