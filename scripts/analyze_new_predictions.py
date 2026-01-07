import pandas as pd
df = pd.read_csv('data_files/player_props_predictions.csv')

print('=== NEW PREDICTIONS WITH REALISTIC LINES ===')
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

print('\n=== ACTIONABLE RECOMMENDATIONS (60-85% confidence) ===')
actionable = df[(df['confidence'] >= 0.60) & (df['confidence'] <= 0.85)]
actionable = actionable.sort_values('confidence', ascending=False).head(15)
for _, row in actionable.iterrows():
    player = row['player_name'][:15]
    prop = row['prop_type'][:15]
    rec = row['recommendation']
    conf = row['confidence']
    l3 = row['avg_L3']
    line = row['line_value']
    print(f'{player:<15} {prop:<15} {rec:<5} {conf:.1%} | L3: {l3:.1f} vs {line}')