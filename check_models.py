from player_props.predict import load_models
models = load_models()
print('Loaded models with receptions:')
for name, info in models.items():
    if 'receptions' in name:
        print(f'  {name}: {info["prop_type"]} {info["line_type"]}')