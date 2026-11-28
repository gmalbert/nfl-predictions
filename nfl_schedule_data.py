"""Shared source for NFL game summaries and schedules."""

from datetime import datetime
from io import BytesIO

import pandas as pd
import requests


SCHEDULE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def get_current_season():
    """Return the season year, including the preceding season's playoffs."""
    now = datetime.now()
    return now.year - 1 if now.month <= 2 else now.year


def load_schedules(seasons):
    response = requests.get(SCHEDULE_URL, timeout=30)
    response.raise_for_status()
    games = pd.read_csv(BytesIO(response.content))
    games = games.loc[games["season"].isin(seasons)]
    if games.empty:
        raise ValueError(f"nflverse returned no games for {list(seasons)}")
    return games
