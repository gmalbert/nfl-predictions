import pandas as pd
df = pd.read_csv('data_files/player_props_predictions.csv')

print('=== CURRENT BETTING LINES ANALYSIS ===')
print('Yards Props:')
yards_props = df[df['prop_type'].str.contains('yards')]
for prop in yards_props['prop_type'].unique():
    prop_data = yards_props[yards_props['prop_type'] == prop]
    print(f'{prop}: avg line = {prop_data["line_value"].mean():.1f}, range = {prop_data["line_value"].min():.1f}-{prop_data["line_value"].max():.1f}')

print('\nTD Props:')
td_props = df[df['prop_type'].str.contains('tds')]
for prop in td_props['prop_type'].unique():
    prop_data = td_props[td_props['prop_type'] == prop]
    print(f'{prop}: avg line = {prop_data["line_value"].mean():.1f}, range = {prop_data["line_value"].min():.1f}-{prop_data["line_value"].max():.1f}')

print('\n=== RECOMMENDATION DISTRIBUTION ===')
print('Overall:')
print(f'OVER: {len(df[df["recommendation"] == "OVER"]):>3} ({len(df[df["recommendation"] == "OVER"])/len(df)*100:>5.1f}%)')
print(f'UNDER: {len(df[df["recommendation"] == "UNDER"]):>3} ({len(df[df["recommendation"] == "UNDER"])/len(df)*100:>5.1f}%)')

print('\nBy Confidence Tier:')
tiers = [('Elite', 0.85, 1.0), ('Strong', 0.75, 0.85), ('Good', 0.65, 0.75), ('Standard', 0.55, 0.65)]
for tier_name, min_conf, max_conf in tiers:
    tier_data = df[(df['confidence'] >= min_conf) & (df['confidence'] < max_conf)]
    if len(tier_data) > 0:
        over_pct = (tier_data['recommendation'] == 'OVER').mean() * 100
        print(f'{tier_name:<8}: {len(tier_data):>3} predictions ({over_pct:>5.1f}% OVER)')