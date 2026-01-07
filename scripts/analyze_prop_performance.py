import pandas as pd

files = ['data_files/player_passing_stats.csv', 'data_files/player_rushing_stats.csv', 'data_files/player_receiving_stats.csv']

for file in files:
    try:
        df = pd.read_csv(file)
        print(f'\n=== {file.split("/")[-1].replace("_stats.csv", "").upper()} PERFORMANCE ANALYSIS ===')

        if 'passing_yards' in df.columns:
            prop_col = 'passing_yards'
            line = 225.5
            over_rate = (df[prop_col] > line).mean() * 100
            avg_perf = df[prop_col].mean()
            print(f'Passing Yards > {line}: {over_rate:.1f}% (avg: {avg_perf:.1f})')

        if 'rushing_yards' in df.columns:
            prop_col = 'rushing_yards'
            line = 65.5
            over_rate = (df[prop_col] > line).mean() * 100
            avg_perf = df[prop_col].mean()
            print(f'Rushing Yards > {line}: {over_rate:.1f}% (avg: {avg_perf:.1f})')

        if 'receiving_yards' in df.columns:
            prop_col = 'receiving_yards'
            line = 55.5
            over_rate = (df[prop_col] > line).mean() * 100
            avg_perf = df[prop_col].mean()
            print(f'Receiving Yards > {line}: {over_rate:.1f}% (avg: {avg_perf:.1f})')

        if 'pass_tds' in df.columns:
            over_rate_1_5 = (df['pass_tds'] > 1.5).mean() * 100
            over_rate_2_5 = (df['pass_tds'] > 2.5).mean() * 100
            avg_tds = df['pass_tds'].mean()
            print(f'Passing TDs > 1.5: {over_rate_1_5:.1f}%, > 2.5: {over_rate_2_5:.1f}% (avg: {avg_tds:.2f})')

        if 'rush_tds' in df.columns:
            over_rate = (df['rush_tds'] > 0.5).mean() * 100
            avg_tds = df['rush_tds'].mean()
            print(f'Rush TDs > 0.5: {over_rate:.1f}% (avg: {avg_tds:.2f})')

        if 'rec_tds' in df.columns:
            over_rate = (df['rec_tds'] > 0.5).mean() * 100
            avg_tds = df['rec_tds'].mean()
            print(f'Rec TDs > 0.5: {over_rate:.1f}% (avg: {avg_tds:.2f})')

    except Exception as e:
        print(f'Error loading {file}: {e}')