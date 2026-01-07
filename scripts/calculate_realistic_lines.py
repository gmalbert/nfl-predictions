import pandas as pd
import numpy as np

def calculate_realistic_lines():
    """Calculate realistic betting lines based on historical performance percentiles."""

    files = {
        'passing': 'data_files/player_passing_stats.csv',
        'rushing': 'data_files/player_rushing_stats.csv',
        'receiving': 'data_files/player_receiving_stats.csv'
    }

    realistic_lines = {}

    for stat_type, file_path in files.items():
        try:
            df = pd.read_csv(file_path)
            print(f"\n=== {stat_type.upper()} STATS ANALYSIS ===")

            if stat_type == 'passing':
                # Passing yards - use 60th percentile for realistic line
                yards_data = df['passing_yards'].dropna()
                realistic_lines['passing_yards_starter'] = np.percentile(yards_data, 60)
                realistic_lines['passing_yards_elite_qb'] = np.percentile(yards_data, 70)

                # Passing TDs - use 60th percentile
                td_data = df['pass_tds'].dropna()
                realistic_lines['passing_tds_over'] = np.percentile(td_data, 60)
                realistic_lines['passing_tds_high'] = np.percentile(td_data, 70)

                print(f"Passing Yards: 60th={realistic_lines['passing_yards_starter']:.1f}, 70th={realistic_lines['passing_yards_elite_qb']:.1f}")
                print(f"Passing TDs: 60th={realistic_lines['passing_tds_over']:.1f}, 70th={realistic_lines['passing_tds_high']:.1f}")

            elif stat_type == 'rushing':
                # Rushing yards - use 60th percentile
                yards_data = df['rushing_yards'].dropna()
                realistic_lines['rushing_yards_starter'] = np.percentile(yards_data, 60)

                # Rushing TDs - use 60th percentile
                td_data = df['rush_tds'].dropna()
                realistic_lines['rushing_tds_anytime'] = np.percentile(td_data, 60)

                print(f"Rushing Yards: 60th={realistic_lines['rushing_yards_starter']:.1f}")
                print(f"Rushing TDs: 60th={realistic_lines['rushing_tds_anytime']:.1f}")

            elif stat_type == 'receiving':
                # Receiving yards - use 60th percentile
                yards_data = df['receiving_yards'].dropna()
                realistic_lines['receiving_yards_starter'] = np.percentile(yards_data, 60)

                # Receiving TDs - use 60th percentile
                td_data = df['rec_tds'].dropna()
                realistic_lines['receiving_tds_anytime'] = np.percentile(td_data, 60)

                print(f"Receiving Yards: 60th={realistic_lines['receiving_yards_starter']:.1f}")
                print(f"Receiving TDs: 60th={realistic_lines['receiving_tds_anytime']:.1f}")

        except Exception as e:
            print(f"Error processing {stat_type}: {e}")

    return realistic_lines

if __name__ == '__main__':
    realistic_lines = calculate_realistic_lines()

    print("\n=== REALISTIC PROP LINES ===")
    print("# Copy these into player_props/predict.py PROP_LINES")
    print("PROP_LINES = {")
    print("    'passing_yards': {")
    print(f"        'elite_qb': {realistic_lines.get('passing_yards_elite_qb', 275.5):.1f},")
    print(f"        'starter': {realistic_lines.get('passing_yards_starter', 225.5):.1f}")
    print("    },")
    print("    'rushing_yards': {")
    print(f"        'starter': {realistic_lines.get('rushing_yards_starter', 65.5):.1f}")
    print("    },")
    print("    'receiving_yards': {")
    print(f"        'starter': {realistic_lines.get('receiving_yards_starter', 55.5):.1f}")
    print("    },")
    print("    'passing_tds': {")
    print(f"        'over': {realistic_lines.get('passing_tds_over', 1.5):.1f},")
    print(f"        'high': {realistic_lines.get('passing_tds_high', 2.5):.1f}")
    print("    },")
    print("    'rushing_tds': {")
    print(f"        'anytime': {realistic_lines.get('rushing_tds_anytime', 0.5):.1f}")
    print("    },")
    print("    'receiving_tds': {")
    print(f"        'anytime': {realistic_lines.get('receiving_tds_anytime', 0.5):.1f}")
    print("    }")
    print("}")