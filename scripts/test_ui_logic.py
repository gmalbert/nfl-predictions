"""
Test the exact logic used in the UI to see if it produces the expected results
"""
import pandas as pd
from datetime import datetime

predictions_csv_path = 'data_files/nfl_games_historical_with_predictions.csv'

print("Loading predictions CSV...")
predictions_df = pd.read_csv(predictions_csv_path, sep='\t', low_memory=False)

print(f"Total rows loaded: {len(predictions_df)}")
print(f"Columns: {list(predictions_df.columns)}")

# Convert gameday to datetime
predictions_df['gameday'] = pd.to_datetime(predictions_df['gameday'])

# Filter for future games
today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
predictions_df_spread = predictions_df[predictions_df['gameday'] > today].copy()

print(f"\nFuture games: {len(predictions_df_spread)}")

# Check if required column exists
if 'prob_underdogCovered' in predictions_df_spread.columns:
    print("+ prob_underdogCovered column exists")
    
    # Apply the exact filter from UI
    spread_bets = predictions_df_spread[
        predictions_df_spread['prob_underdogCovered'] >= 0.45
    ].copy()
    
    print(f"\nGames with prob >= 0.45: {len(spread_bets)}")
    
    # Add EV status flag if column exists
    if 'ev_spread' in spread_bets.columns:
        print("+ ev_spread column exists")
        spread_bets['ev_status'] = spread_bets['ev_spread'].apply(
            lambda x: '+EV' if x > 0 else '-EV'
        )
        print(f"+ ev_status column created")
    
    if len(spread_bets) > 0:
        print(f"\n+ SHOULD DISPLAY {len(spread_bets)} GAMES")
        print("\nGames to display:")
        for _, row in spread_bets.head(10).iterrows():
            prob = row['prob_underdogCovered'] * 100
            ev = row.get('ev_spread', 'N/A')
            ev_status = row.get('ev_status', 'N/A')
            matchup = f"{row['away_team']} @ {row['home_team']}"
            print(f"  {matchup}: {prob:.1f}% prob, EV: ${ev:.2f}, Status: {ev_status}")
    else:
        print("\n- WOULD SHOW: No games message")
else:
    print("- prob_underdogCovered column NOT FOUND")
