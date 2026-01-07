import pandas as pd

df = pd.read_csv('data_files/player_props_predictions.csv')
print(f'Loaded {len(df)} predictions')

if 'position' in df.columns:
    print('Position column found')

    passing_yards = df[df['prop_type'] == 'passing_yards']
    print(f'Passing yards predictions: {len(passing_yards)}')

    positions = passing_yards['position'].unique()
    print(f'Positions with passing yards: {list(positions)}')

    non_qb = passing_yards[passing_yards['position'] != 'QB']
    if len(non_qb) > 0:
        print(f'FOUND {len(non_qb)} NON-QB PASSING PREDICTIONS')
        unique_non_qb = non_qb[['player_name', 'position']].drop_duplicates()
        for _, row in unique_non_qb.iterrows():
            print(f'  {row["player_name"]} ({row["position"]})')
    else:
        print('SUCCESS: All passing yards predictions are for QBs!')

else:
    print('Position column NOT found')