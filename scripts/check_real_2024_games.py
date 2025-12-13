from datetime import datetime
import pandas as pd

print(f'System thinks today is: {datetime.now()}')
print(f'ACTUAL today: December 13, 2024')

df = pd.read_csv('data_files/nfl_games_historical_with_predictions.csv', sep='\t')
df['gameday'] = pd.to_datetime(df['gameday'])

# Use ACTUAL current date (Dec 13, 2024)
real_today = datetime(2024, 12, 13)

print("\n" + "=" * 60)
print("REMAINING 2024 GAMES (Dec 13, 2024 onwards)")
print("=" * 60)

df_2024 = df[df['season'] == 2024]
remaining = df_2024[df_2024['gameday'] > real_today].copy()

print(f"\nTotal remaining 2024 games: {len(remaining)}")
print(f"Date range: {remaining['gameday'].min()} to {remaining['gameday'].max()}")

print(f"\nSpread Betting Analysis:")
print(f"  Games >=52.4% confidence: {len(remaining[remaining['prob_underdogCovered'] >= 0.524])}")
print(f"  Games >=45% confidence: {len(remaining[remaining['prob_underdogCovered'] >= 0.45])}")
print(f"  Max probability: {remaining['prob_underdogCovered'].max():.1%}")
print(f"  Games with +EV: {len(remaining[remaining['ev_spread'] > 0])}")

# Show top games
print(f"\nTop 10 Confidence Games:")
top = remaining.nlargest(10, 'prob_underdogCovered')[['gameday', 'away_team', 'home_team', 'prob_underdogCovered', 'ev_spread', 'spread_line']]
for idx, row in top.iterrows():
    date = row['gameday'].strftime('%m/%d')
    matchup = f"{row['away_team']} @ {row['home_team']}"
    prob = row['prob_underdogCovered'] * 100
    ev = row['ev_spread']
    spread = row['spread_line']
    ev_flag = "✅" if ev > 0 else "⚠️"
    print(f"  {date}: {matchup:25s} | {prob:4.1f}% | EV: ${ev:6.2f} {ev_flag} | Spread: {spread:+.1f}")

# Check if model sees these as having momentum data
print(f"\nMomentum Feature Check (sample game):")
sample = remaining.iloc[0]
momentum_cols = [col for col in df.columns if 'Last3' in col or 'Trend' in col]
if momentum_cols:
    print(f"  Momentum columns exist: {len(momentum_cols)}")
    for col in momentum_cols[:5]:
        print(f"    {col}: {sample[col]}")
else:
    print(f"  No momentum columns found!")
