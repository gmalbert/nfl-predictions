from pathlib import Path

from nfl_schedule_data import get_current_season, load_schedules

DATA_DIR = Path("data_files")


def main():
    seasons = range(2020, get_current_season() + 1)
    games = load_schedules(seasons)
    out_path = DATA_DIR / "nfl_games_historical.csv"
    games.to_csv(out_path, index=False, sep="\t")
    print(f"Saved {len(games)} game summaries to {out_path}")


if __name__ == "__main__":
    main()
