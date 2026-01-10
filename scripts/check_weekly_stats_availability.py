"""
Test if pre-aggregated weekly player stats are available for 2025.
This is much faster than aggregating PBP data.
"""
import nfl_data_py as nfl
import pandas as pd

print("=" * 80)
print("TESTING PRE-AGGREGATED WEEKLY STATS AVAILABILITY")
print("=" * 80)

# Test 2025 weekly stats (the fast way)
print("\n1. Testing nfl_data_py.import_weekly_data() for 2025...")
try:
    weekly_stats = nfl.import_weekly_data([2025], columns=[
        'player_name', 'week', 'season', 'passing_yards', 'passing_tds',
        'rushing_yards', 'rushing_tds', 'receiving_yards', 'receiving_tds', 'receptions'
    ])
    
    print(f"✅ SUCCESS! Pre-aggregated weekly stats ARE available!")
    print(f"   Total records: {len(weekly_stats):,}")
    print(f"   Weeks available: {sorted(weekly_stats['week'].unique())}")
    print(f"   Columns: {weekly_stats.columns.tolist()}")
    
    # Test a specific week
    week_18 = weekly_stats[weekly_stats['week'] == 18]
    print(f"\n   Week 18 sample:")
    print(f"   Players: {len(week_18)}")
    print(weekly_stats[weekly_stats['week'] == 18].head(10)[['player_name', 'passing_yards', 'rushing_yards', 'receiving_yards']])
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("✅ Pre-aggregated stats ARE available - USE THIS METHOD!")
    print("   Much faster than PBP aggregation (instant vs ~5 seconds)")
    print("   Should use try/except: try weekly_data first, fallback to PBP if fails")
    
except Exception as e:
    print(f"❌ FAILED: {str(e)}")
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("❌ Pre-aggregated stats NOT available - must use PBP aggregation")
    print("   Will need to aggregate play-by-play data (slower but works)")
