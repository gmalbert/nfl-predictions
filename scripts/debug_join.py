import pandas as pd

# Load predictions directly
df = pd.read_csv('data_files/player_props_predictions.csv')
print('Columns in predictions file:')
print(df.columns.tolist())

# Load players
players = pd.read_csv('data_files/players.csv')
print(f'\nPlayers loaded: {len(players)}')

# Check if short_name matching works
sample_players = ['C.McCaffrey', 'T.Atwell', 'D.Moore', 'K.Allen']
for player in sample_players:
    matches = players[players['short_name'] == player]
    print(f'{player}: {len(matches)} matches')
    if len(matches) > 0:
        print(f'  Positions: {matches["position"].tolist()}')

# Try the join manually
print('\nTrying manual join...')
predictions_with_pos = df.merge(
    players[['short_name', 'position']],
    left_on='player_name',
    right_on='short_name',
    how='left'
)
predictions_with_pos = predictions_with_pos.drop('short_name', axis=1)

print(f'After join: {len(predictions_with_pos)} rows')
print(f'Position column exists: {"position" in predictions_with_pos.columns}')

if 'position' in predictions_with_pos.columns:
    passing_yards = predictions_with_pos[predictions_with_pos['prop_type'] == 'passing_yards']
    non_qb = passing_yards[passing_yards['position'] != 'QB']
    print(f'Non-QB passing predictions: {len(non_qb)}')
    if len(non_qb) > 0:
        print('Non-QB players:')
        for player in non_qb['player_name'].unique():
            pos = non_qb[non_qb['player_name'] == player]['position'].iloc[0]
            print(f'  {player} ({pos})')