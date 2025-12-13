import pandas as pd
from datetime import datetime

predictions_csv_path = 'data_files/nfl_games_historical_with_predictions.csv'
predictions_df_spread = pd.read_csv(predictions_csv_path, sep='\t')
predictions_df_spread['gameday'] = pd.to_datetime(predictions_df_spread['gameday'], errors='coerce')

# Filter for future games only
today = pd.to_datetime(datetime.now().date())
print(f"Today's date: {today}")
print(f"Total games in file: {len(predictions_df_spread)}")

future_games = predictions_df_spread[predictions_df_spread['gameday'] > today]
print(f"Future games (gameday > today): {len(future_games)}")

if 'prob_underdogCovered' in future_games.columns:
    print(f"prob_underdogCovered column exists: Yes")
    print(f"Games with prob >= 0.45: {(future_games['prob_underdogCovered'] >= 0.45).sum()}")
    
    spread_bets = future_games[future_games['prob_underdogCovered'] >= 0.45].copy()
    print(f"\nFiltered spread_bets: {len(spread_bets)}")
    
    if len(spread_bets) > 0:
        print(f"\nTop 5 games:")
        for idx, row in spread_bets.head(5).iterrows():
            print(f"  {row['gameday'].date()}: {row['away_team']} @ {row['home_team']} - {row['prob_underdogCovered']:.1%}")
    else:
        print("\n⚠️ No games found after filtering!")
        print(f"Max probability in future games: {future_games['prob_underdogCovered'].max():.1%}")
else:
    print("prob_underdogCovered column: MISSING")

if 'ev_spread' in future_games.columns:
    print(f"\nev_spread column exists: Yes")
else:
    print(f"\nev_spread column: MISSING")
