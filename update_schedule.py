from pathlib import Path

import pandas as pd

from nfl_schedule_data import get_current_season, load_schedules

DATA_DIR = Path("data_files")


def main():
    season = get_current_season()
    print(f"Fetching NFL schedule for {season}...")
    games = load_schedules([season])
    schedule = games[["week", "gameday", "home_team", "away_team", "stadium"]].rename(
        columns={"gameday": "date", "stadium": "venue"}
    )
    schedule["date"] = pd.to_datetime(schedule["date"]).dt.strftime("%Y-%m-%d")
    schedule["status"] = "REG"

    out_path = DATA_DIR / f"nfl_schedule_{season}.csv"
    schedule.to_csv(out_path, index=False)
    print(f"Saved {len(schedule)} games to {out_path}")


if __name__ == "__main__":
    main()
