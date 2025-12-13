import pandas as pd
from datetime import datetime

# Load data files
schedule = pd.read_csv('data_files/nfl_schedule_2025.csv', sep='\t')
predictions = pd.read_csv('data_files/nfl_games_historical_with_predictions.csv', sep='\t')

print("="*60)
print("CHECKING UI DATA FOR SPREAD BETS")
print("="*60)

print(f"\n📁 Data Files:")
print(f"   Schedule (2025): {len(schedule)} games")
print(f"   Historical predictions: {len(predictions)} games")

print(f"\n📊 Spread Bets in Historical Data:")
spread_bets_hist = predictions[predictions['pred_spreadCovered_optimal'] == 1]
print(f"   Games with spread bets: {len(spread_bets_hist)} ({len(spread_bets_hist)/len(predictions)*100:.1f}%)")

print(f"\n🔍 Schedule File Structure:")
print(f"   Columns in schedule: {schedule.columns.tolist()[:15]}")
print(f"\n   Sample schedule row:")
print(schedule.head(1).T)

# Check if schedule has prediction columns
has_pred_cols = 'pred_spreadCovered_optimal' in schedule.columns
print(f"\n❓ Schedule has prediction columns: {has_pred_cols}")

if not has_pred_cols:
    print(f"\n⚠️  PROBLEM FOUND:")
    print(f"   The schedule file doesn't have prediction columns!")
    print(f"   The UI loads predictions from nfl_schedule_2025.csv")
    print(f"   But predictions are only in nfl_games_historical_with_predictions.csv")
    print(f"\n💡 SOLUTION:")
    print(f"   Need to run: python nfl-gather-data.py")
    print(f"   This should add predictions to the schedule file")
else:
    spread_bets_schedule = schedule[schedule['pred_spreadCovered_optimal'] == 1]
    print(f"\n✅ Schedule has {len(spread_bets_schedule)} games with spread bets")
    
    # Check for upcoming games
    schedule['gameday'] = pd.to_datetime(schedule['gameday'])
    today = datetime.now()
    upcoming = schedule[schedule['gameday'] >= today]
    upcoming_spread = upcoming[upcoming['pred_spreadCovered_optimal'] == 1]
    
    print(f"\n📅 Upcoming Games:")
    print(f"   Total upcoming: {len(upcoming)}")
    print(f"   With spread bets: {len(upcoming_spread)}")

print("="*60)
