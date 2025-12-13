import pandas as pd
from datetime import datetime

df = pd.read_csv('data_files/nfl_games_historical_with_predictions.csv', sep='\t')
df['gameday'] = pd.to_datetime(df['gameday'])

today = datetime(2025, 12, 13)

print("=" * 70)
print("NFL SEASON vs CALENDAR YEAR")
print("=" * 70)

# 2024 NFL Season games (Sept 2024 - Feb 2025)
season_2024 = df[df['season'] == 2024]
print(f"\n2024 NFL Season (labeled season=2024):")
print(f"  Total games: {len(season_2024)}")
print(f"  Date range: {season_2024['gameday'].min()} to {season_2024['gameday'].max()}")
print(f"  Games in calendar 2024: {len(season_2024[season_2024['gameday'].dt.year == 2024])}")
print(f"  Games in calendar 2025: {len(season_2024[season_2024['gameday'].dt.year == 2025])}")

# Current 2024 season - games still to be played
remaining_2024_season = season_2024[season_2024['gameday'] > today]
print(f"\n  Remaining 2024 season games (after Dec 13, 2025): {len(remaining_2024_season)}")

# 2025 NFL Season games (Sept 2025 - Feb 2026)  
season_2025 = df[df['season'] == 2025]
print(f"\n2025 NFL Season (labeled season=2025):")
print(f"  Total games: {len(season_2025)}")
if len(season_2025) > 0:
    print(f"  Date range: {season_2025['gameday'].min()} to {season_2025['gameday'].max()}")
    print(f"  These are FUTURE scheduled games for next season")
    
# What the app considers "future"
future_all = df[df['gameday'] > today]
print(f"\n{'='*70}")
print(f"GAMES AFTER TODAY (Dec 13, 2025):")
print(f"{'='*70}")
print(f"Total: {len(future_all)}")
print(f"  From 2024 season: {len(future_all[future_all['season'] == 2024])}")
print(f"  From 2025 season: {len(future_all[future_all['season'] == 2025])}")

if len(future_all[future_all['season'] == 2024]) > 0:
    print(f"\nREMAINING 2024 SEASON GAMES:")
    remaining = future_all[future_all['season'] == 2024]
    print(f"  Count: {len(remaining)}")
    print(f"  Confidence range: {remaining['prob_underdogCovered'].min():.1%} to {remaining['prob_underdogCovered'].max():.1%}")
    print(f"  Games >=52.4%: {len(remaining[remaining['prob_underdogCovered'] >= 0.524])}")
    print(f"  Games >=45%: {len(remaining[remaining['prob_underdogCovered'] >= 0.45])}")
    
    print(f"\n  Top 5 games:")
    top5 = remaining.nlargest(5, 'prob_underdogCovered')
    for _, g in top5.iterrows():
        print(f"    {g['gameday'].strftime('%m/%d/%y')}: {g['away_team']} @ {g['home_team']} - {g['prob_underdogCovered']:.1%} (EV: ${g['ev_spread']:.2f})")

print(f"\n2025 SEASON GAMES (next season - haven't been played):")
if len(future_all[future_all['season'] == 2025]) > 0:
    future_2025 = future_all[future_all['season'] == 2025]
    print(f"  Count: {len(future_2025)}")
    print(f"  These start in September 2025")
