import pandas as pd

# Load predictions and players
predictions = pd.read_csv('data_files/player_props_predictions.csv')
players = pd.read_csv('data_files/players.csv')

print('=== INVESTIGATING POSITION MISMATCH ===')
print()

# Check the specific players mentioned
problem_players = ['C.McCaffrey', 'T.Atwell', 'D.Moore', 'K.Allen']

for player_name in problem_players:
    print(f'\n=== {player_name} ===')

    # Find predictions for this player
    player_preds = predictions[predictions['player_name'] == player_name]
    print(f'Predictions: {len(player_preds)}')

    # Find player records
    player_records = players[players['short_name'] == player_name]
    print(f'Player records in CSV: {len(player_records)}')

    if len(player_records) > 0:
        print('Positions found:')
        for _, row in player_records.iterrows():
            print(f'  - {row["position"]} ({row["display_name"]})')

    # Show passing yards predictions
    passing_preds = player_preds[player_preds['prop_type'] == 'passing_yards']
    if len(passing_preds) > 0:
        print('Passing yards predictions:')
        for _, pred in passing_preds.iterrows():
            print(f'  - {pred["recommendation"]} {pred["line_value"]:.1f} (conf: {pred["confidence"]:.1%})')