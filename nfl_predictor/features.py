"""Point-in-time game and play-by-play feature engineering.

Every historical outcome is shifted before it is rolled.  Upcoming rows may be
present in the same dataframe: because their outcome columns are null, they do
not silently become losses or contaminate later windows.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from .contracts import GAME_FACT_CONTRACT, SchemaError, require_columns
from .markets import american_to_implied


TEAM_ALIASES = {
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "JAC": "JAX",
    "LA": "LAR",
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LAR",
}

FORM_METRICS = (
    "points_for",
    "points_against",
    "margin",
    "win",
    "cover",
    "game_total",
    "over",
    "close_game",
    "blowout_win",
)


def normalize_team(value: object) -> object:
    if pd.isna(value):
        return value
    code = str(value).strip().upper()
    return TEAM_ALIASES.get(code, code)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _binary_with_push(margin: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(margin > 0, 1.0, np.where(margin < 0, 0.0, np.nan)),
        index=margin.index,
        dtype=float,
    )


def add_game_targets(games: pd.DataFrame) -> pd.DataFrame:
    """Create unambiguous home-oriented targets while preserving pushes."""

    result = GAME_FACT_CONTRACT.validate(games)
    result["home_score"] = _numeric(result, "home_score")
    result["away_score"] = _numeric(result, "away_score")
    result["spread_line"] = _numeric(result, "spread_line")
    result["total_line"] = _numeric(result, "total_line")
    completed = result["home_score"].notna() & result["away_score"].notna()
    result["is_completed"] = completed
    result["home_margin"] = (result["home_score"] - result["away_score"]).where(completed)
    result["actual_total"] = (result["home_score"] + result["away_score"]).where(completed)
    result["target_home_win"] = _binary_with_push(result["home_margin"])

    # nflverse spread_line is positive when the home team is favored.  A home
    # cover therefore requires home_margin > spread_line.
    result["home_cover_margin"] = (result["home_margin"] - result["spread_line"]).where(
        completed & result["spread_line"].notna()
    )
    result["target_home_cover"] = _binary_with_push(result["home_cover_margin"])
    result["total_margin"] = (result["actual_total"] - result["total_line"]).where(
        completed & result["total_line"].notna()
    )
    result["target_over"] = _binary_with_push(result["total_margin"])

    home_is_underdog = result["spread_line"] < 0
    away_is_underdog = result["spread_line"] > 0
    result["underdog_team"] = np.where(
        home_is_underdog,
        result["home_team"],
        np.where(away_is_underdog, result["away_team"], pd.NA),
    )
    result["target_underdog_cover"] = np.where(
        home_is_underdog,
        result["target_home_cover"],
        np.where(away_is_underdog, 1.0 - result["target_home_cover"], np.nan),
    )
    result["target_underdog_win"] = np.where(
        home_is_underdog,
        result["target_home_win"],
        np.where(away_is_underdog, 1.0 - result["target_home_win"], np.nan),
    )
    return result


def to_team_games(games: pd.DataFrame) -> pd.DataFrame:
    """Convert one game row into two team-perspective rows."""

    games = add_game_targets(games).sort_values(["gameday", "game_id"], kind="stable")
    spread = games["spread_line"]

    common = ["game_id", "season", "week", "gameday", "is_completed"]
    home = games[common].copy()
    home["team"] = games["home_team"].map(normalize_team)
    home["opponent"] = games["away_team"].map(normalize_team)
    home["is_home"] = 1
    home["points_for"] = games["home_score"]
    home["points_against"] = games["away_score"]
    home["team_line"] = -spread

    away = games[common].copy()
    away["team"] = games["away_team"].map(normalize_team)
    away["opponent"] = games["home_team"].map(normalize_team)
    away["is_home"] = 0
    away["points_for"] = games["away_score"]
    away["points_against"] = games["home_score"]
    away["team_line"] = spread

    team_games = pd.concat([home, away], ignore_index=True)
    team_games = team_games.sort_values(["team", "gameday", "game_id"], kind="stable")
    completed = team_games["is_completed"]
    team_games["margin"] = (team_games["points_for"] - team_games["points_against"]).where(completed)
    team_games["win"] = _binary_with_push(team_games["margin"])
    team_games["cover_margin"] = (team_games["margin"] + team_games["team_line"]).where(
        completed & team_games["team_line"].notna()
    )
    team_games["cover"] = _binary_with_push(team_games["cover_margin"])
    team_games["game_total"] = (team_games["points_for"] + team_games["points_against"]).where(completed)

    totals = games.set_index("game_id")["total_line"]
    team_games["total_line"] = team_games["game_id"].map(totals)
    team_games["over"] = _binary_with_push(team_games["game_total"] - team_games["total_line"])
    team_games["close_game"] = (team_games["margin"].abs() <= 3).astype(float).where(completed)
    team_games["blowout_win"] = (team_games["margin"] >= 20).astype(float).where(completed)
    return team_games.reset_index(drop=True)


def _shifted_rolling(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=1).mean()


def add_team_form(
    team_games: pd.DataFrame,
    *,
    windows: Sequence[int] = (3, 5, 8, 16),
    ewm_halflife: float = 4.0,
) -> pd.DataFrame:
    """Add prior-only rolling and exponentially weighted form features."""

    require_columns(
        team_games,
        ("team", "gameday", "game_id", "is_completed", *FORM_METRICS),
        context="add_team_form",
    )
    if not windows or any(int(window) < 1 for window in windows):
        raise ValueError("windows must contain positive integers")

    result = team_games.sort_values(["team", "gameday", "game_id"], kind="stable").copy()
    grouped = result.groupby("team", sort=False, group_keys=False)
    result["prior_games_played"] = grouped["is_completed"].transform(
        lambda values: values.astype(int).shift(1, fill_value=0).cumsum()
    )
    for metric in FORM_METRICS:
        for window in windows:
            result[f"{metric}_mean_w{int(window)}"] = grouped[metric].transform(
                lambda values, size=int(window): _shifted_rolling(values, size)
            )
        result[f"{metric}_ewm"] = grouped[metric].transform(
            lambda values: values.shift(1).ewm(halflife=ewm_halflife, adjust=False).mean()
        )
    return result


def _implied_series(odds: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(odds, errors="coerce")
    valid = numeric.notna() & numeric.ne(0)
    result = pd.Series(np.nan, index=odds.index, dtype=float)
    result.loc[valid] = numeric.loc[valid].map(american_to_implied)
    return result


def add_market_features(games: pd.DataFrame) -> pd.DataFrame:
    result = games.copy()
    result["spread_line"] = _numeric(result, "spread_line")
    result["total_line"] = _numeric(result, "total_line")
    result["spread_abs"] = result["spread_line"].abs()
    key_numbers = np.array([3.0, 6.0, 7.0, 10.0, 14.0])
    result["spread_key_distance"] = result["spread_abs"].map(
        lambda value: np.nan if pd.isna(value) else float(np.min(np.abs(key_numbers - value)))
    )
    for key in (3, 7, 10, 14):
        result[f"spread_on_key_{key}"] = pd.Series(
            np.isclose(result["spread_abs"], key), index=result.index, dtype=float
        ).where(result["spread_abs"].notna())

    for side in ("home", "away"):
        column = f"{side}_moneyline"
        result[f"{side}_ml_implied_raw"] = _implied_series(
            result[column] if column in result else pd.Series(np.nan, index=result.index)
        )
    ml_sum = result["home_ml_implied_raw"] + result["away_ml_implied_raw"]
    result["moneyline_overround"] = ml_sum - 1.0
    result["home_ml_implied_fair"] = result["home_ml_implied_raw"] / ml_sum
    result["away_ml_implied_fair"] = result["away_ml_implied_raw"] / ml_sum

    for side in ("home", "away"):
        column = f"{side}_spread_odds"
        result[f"{side}_spread_implied_raw"] = _implied_series(
            result[column] if column in result else pd.Series(np.nan, index=result.index)
        )
    spread_sum = result["home_spread_implied_raw"] + result["away_spread_implied_raw"]
    result["spread_overround"] = spread_sum - 1.0

    for side in ("over", "under"):
        column = f"{side}_odds"
        result[f"{side}_implied_raw"] = _implied_series(
            result[column] if column in result else pd.Series(np.nan, index=result.index)
        )
    total_sum = result["over_implied_raw"] + result["under_implied_raw"]
    result["total_overround"] = total_sum - 1.0
    result["over_implied_fair"] = result["over_implied_raw"] / total_sum
    result["under_implied_fair"] = result["under_implied_raw"] / total_sum

    optional_movements = {
        "spread_move": ("spread_line", "open_spread_line"),
        "total_move": ("total_line", "open_total_line"),
        "home_ml_move": ("home_moneyline", "open_home_moneyline"),
        "away_ml_move": ("away_moneyline", "open_away_moneyline"),
    }
    for output, (current, opening) in optional_movements.items():
        result[output] = (
            _numeric(result, current) - _numeric(result, opening)
            if opening in result
            else np.nan
        )
    result["spread_crossed_key"] = np.nan
    if "open_spread_line" in result:
        opening = _numeric(result, "open_spread_line").abs()
        current = result["spread_abs"]
        crossed = pd.Series(False, index=result.index)
        for key in key_numbers:
            crossed |= (opening - key) * (current - key) < 0
        result["spread_crossed_key"] = crossed.astype(float).where(
            opening.notna() & current.notna()
        )
    return result


def add_context_features(games: pd.DataFrame) -> pd.DataFrame:
    result = games.copy()
    home_rest = _numeric(result, "home_rest")
    away_rest = _numeric(result, "away_rest")
    result["rest_diff"] = home_rest - away_rest
    result["home_short_rest"] = (home_rest <= 6).astype(int).where(home_rest.notna())
    result["away_short_rest"] = (away_rest <= 6).astype(int).where(away_rest.notna())
    result["home_extended_rest"] = (home_rest >= 10).astype(int).where(home_rest.notna())
    result["away_extended_rest"] = (away_rest >= 10).astype(int).where(away_rest.notna())
    result["rest_mismatch"] = (result["rest_diff"].abs() >= 3).astype(int).where(
        result["rest_diff"].notna()
    )

    roof = result.get("roof", pd.Series("", index=result.index)).astype(str).str.lower()
    outdoor = roof.isin({"outdoors", "open"})
    temperature = _numeric(result, "temp")
    wind = _numeric(result, "wind")
    result["is_dome"] = roof.isin({"dome", "closed"}).astype(int)
    result["weather_observed"] = (outdoor & (temperature.notna() | wind.notna())).astype(int)
    result["is_freezing"] = ((temperature <= 32) & outdoor).astype(int)
    result["is_hot"] = ((temperature >= 85) & outdoor).astype(int)
    result["is_windy"] = ((wind >= 15) & outdoor).astype(int)
    result["is_extreme_wind"] = ((wind >= 20) & outdoor).astype(int)
    result["cold_degree"] = (45 - temperature).clip(lower=0).where(outdoor)
    result["wind_over_10"] = (wind - 10).clip(lower=0).where(outdoor)
    result["temperature_wind_interaction"] = result["cold_degree"] * result["wind_over_10"]
    location = result.get("location", pd.Series(pd.NA, index=result.index)).astype("string").str.lower()
    result["neutral_site"] = pd.Series(
        np.where(location.eq("neutral"), 1.0, np.where(location.eq("home"), 0.0, np.nan)),
        index=result.index,
        dtype=float,
    )
    result["division_game"] = _numeric(result, "div_game").fillna(0).astype(int)

    if "away_travel_miles" in result:
        travel = _numeric(result, "away_travel_miles")
        result["away_travel_1000mi"] = travel / 1000.0
        result["away_long_travel"] = (travel >= 1500).astype(int).where(travel.notna())
    if "away_timezone_shift" in result:
        shift = _numeric(result, "away_timezone_shift")
        result["away_timezone_shift_abs"] = shift.abs()
        result["away_eastbound"] = (shift > 0).astype(int).where(shift.notna())
    return result


def _join_team_features(games: pd.DataFrame, team_form: pd.DataFrame) -> pd.DataFrame:
    identity = {
        "game_id",
        "season",
        "week",
        "gameday",
        "team",
        "opponent",
        "is_home",
        "is_completed",
        *FORM_METRICS,
        "team_line",
        "cover_margin",
        "total_line",
    }
    feature_columns = [column for column in team_form.columns if column not in identity]
    home = team_form.loc[team_form["is_home"].eq(1), ["game_id", *feature_columns]].copy()
    away = team_form.loc[team_form["is_home"].eq(0), ["game_id", *feature_columns]].copy()
    home = home.rename(columns={column: f"home_{column}" for column in feature_columns})
    away = away.rename(columns={column: f"away_{column}" for column in feature_columns})
    result = games.merge(home, on="game_id", how="left", validate="one_to_one")
    result = result.merge(away, on="game_id", how="left", validate="one_to_one")
    for column in feature_columns:
        home_column = f"home_{column}"
        away_column = f"away_{column}"
        if pd.api.types.is_numeric_dtype(result[home_column]) and pd.api.types.is_numeric_dtype(
            result[away_column]
        ):
            result[f"diff_{column}"] = result[home_column] - result[away_column]
    return result


def aggregate_pbp_team_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate public nflverse play-by-play into stable team-game signals."""

    require_columns(pbp, ("game_id", "posteam", "defteam"), context="aggregate_pbp_team_games")
    plays = pbp.copy()
    if "play_type" in plays:
        plays = plays[plays["play_type"].isin(["pass", "run"])]
    if plays.empty:
        return pd.DataFrame(columns=["game_id", "team"])

    for column in (
        "epa",
        "success",
        "pass_attempt",
        "rush_attempt",
        "sack",
        "qb_hit",
        "interception",
        "fumble_lost",
        "yards_gained",
        "down",
        "yardline_100",
        "wp",
        "third_down_converted",
    ):
        plays[column] = _numeric(plays, column)
    plays["dropback"] = plays["pass_attempt"].fillna(0) + plays["sack"].fillna(0)
    plays["turnover"] = (
        plays["interception"].fillna(0) + plays["fumble_lost"].fillna(0)
    ).clip(upper=1)
    plays["explosive"] = (plays["yards_gained"] >= 20).astype(float)
    plays["early_down_pass"] = (
        (plays["down"].isin([1, 2])) & plays["dropback"].gt(0)
    ).astype(float)
    plays["early_down"] = plays["down"].isin([1, 2]).astype(float)
    plays["neutral"] = plays["wp"].between(0.2, 0.8, inclusive="both").astype(float)
    plays["neutral_pass"] = (plays["neutral"].eq(1) & plays["dropback"].gt(0)).astype(float)
    plays["red_zone"] = (plays["yardline_100"] <= 20).astype(float)

    def _aggregate(group: pd.DataFrame) -> pd.Series:
        play_count = max(len(group), 1)
        dropbacks = max(group["dropback"].sum(), 1.0)
        rushes = max(group["rush_attempt"].sum(), 1.0)
        early_downs = max(group["early_down"].sum(), 1.0)
        neutral_plays = max(group["neutral"].sum(), 1.0)
        third_downs = max(group["down"].eq(3).sum(), 1)
        red_zone_plays = max(group["red_zone"].sum(), 1.0)
        return pd.Series(
            {
                "plays": len(group),
                "off_epa_per_play": group["epa"].sum() / play_count,
                "off_success_rate": group["success"].mean(),
                "pass_rate": group["dropback"].sum() / play_count,
                "pass_epa_per_dropback": group.loc[group["dropback"].gt(0), "epa"].sum()
                / dropbacks,
                "rush_epa_per_carry": group.loc[group["rush_attempt"].gt(0), "epa"].sum()
                / rushes,
                "explosive_rate": group["explosive"].sum() / play_count,
                "sack_rate_allowed": group["sack"].sum() / dropbacks,
                "qb_hit_rate_allowed": group["qb_hit"].sum() / dropbacks,
                "turnover_rate": group["turnover"].sum() / play_count,
                "early_down_pass_rate": group["early_down_pass"].sum() / early_downs,
                "neutral_pass_rate": group["neutral_pass"].sum() / neutral_plays,
                "third_down_success_rate": group.loc[
                    group["down"].eq(3), "third_down_converted"
                ].sum()
                / third_downs,
                "red_zone_success_rate": group.loc[group["red_zone"].eq(1), "success"].sum()
                / red_zone_plays,
            }
        )

    offense = plays.groupby(["game_id", "posteam"], observed=True).apply(
        _aggregate, include_groups=False
    )
    offense = offense.reset_index().rename(columns={"posteam": "team"})
    offense["team"] = offense["team"].map(normalize_team)
    defense = plays.groupby(["game_id", "defteam"], observed=True).agg(
        def_epa_allowed=("epa", "mean"),
        def_success_allowed=("success", "mean"),
        def_explosive_allowed=("explosive", "mean"),
        def_sacks=("sack", "sum"),
        def_takeaways=("turnover", "sum"),
    )
    defense = defense.reset_index().rename(columns={"defteam": "team"})
    defense["team"] = defense["team"].map(normalize_team)
    return offense.merge(defense, on=["game_id", "team"], how="outer", validate="one_to_one")


def feature_cutoff_at(games: pd.DataFrame) -> pd.Series:
    """Return the latest safe cutoff represented by the legacy game row.

    The wide nflverse schedule artifact contains closing markets, so it is a
    kickoff benchmark rather than an early-week decision snapshot.  nflverse
    ``gametime`` values are Eastern time; date-only inputs fall back to midnight
    UTC and must not be presented as intraday forecasts.
    """

    base = pd.to_datetime(games["gameday"], utc=True, errors="coerce")
    if "gametime" not in games:
        return base
    date = base.dt.strftime("%Y-%m-%d")
    clock = games["gametime"].astype("string").str.strip()
    combined = pd.to_datetime(date + " " + clock, format="%Y-%m-%d %H:%M", errors="coerce")
    kickoff = combined.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="shift_forward"
    ).dt.tz_convert("UTC")
    return kickoff.fillna(base)


def build_pbp_pregame_features(
    games: pd.DataFrame,
    pbp: pd.DataFrame,
    *,
    windows: Sequence[int] = (4, 8),
) -> pd.DataFrame:
    """Build prior-game PBP features and return one row per game."""

    team_games = to_team_games(games)[
        ["game_id", "gameday", "team", "is_home", "is_completed"]
    ]
    aggregates = aggregate_pbp_team_games(pbp)
    history = team_games.merge(aggregates, on=["game_id", "team"], how="left", validate="one_to_one")
    metric_columns = [
        column
        for column in aggregates.columns
        if column not in {"game_id", "team"}
    ]
    history = history.sort_values(["team", "gameday", "game_id"], kind="stable")
    grouped = history.groupby("team", sort=False, group_keys=False)
    for metric in metric_columns:
        for window in windows:
            history[f"{metric}_mean_w{int(window)}"] = grouped[metric].transform(
                lambda values, size=int(window): _shifted_rolling(values, size)
            )
    keep = [
        "game_id",
        "is_home",
        *[column for column in history if "_mean_w" in column],
    ]
    identity_games = add_game_targets(games)
    return _join_team_features(identity_games, history[keep])


def build_pregame_features(
    games: pd.DataFrame,
    *,
    pbp: pd.DataFrame | None = None,
    form_windows: Sequence[int] = (3, 5, 8, 16),
    pbp_windows: Sequence[int] = (4, 8),
) -> pd.DataFrame:
    """Build the full point-in-time feature frame used by v2 models."""

    clean = GAME_FACT_CONTRACT.validate(games)
    clean["home_team"] = clean["home_team"].map(normalize_team)
    clean["away_team"] = clean["away_team"].map(normalize_team)
    clean = add_game_targets(clean)
    form = add_team_form(to_team_games(clean), windows=form_windows)
    result = _join_team_features(clean, form)
    result = add_market_features(result)
    result = add_context_features(result)

    if pbp is not None and not pbp.empty:
        pbp_features = build_pbp_pregame_features(clean, pbp, windows=pbp_windows)
        pbp_columns = [
            column
            for column in pbp_features.columns
            if "_mean_w" in column and column not in result
        ]
        result = result.merge(
            pbp_features[["game_id", *pbp_columns]],
            on="game_id",
            how="left",
            validate="one_to_one",
        )

    result["feature_cutoff_at"] = feature_cutoff_at(result)
    result = result.sort_values(["gameday", "game_id"], kind="stable").reset_index(drop=True)
    if result["game_id"].duplicated().any():
        raise SchemaError("build_pregame_features produced duplicate game rows")
    return result


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric, pregame-safe columns while excluding identifiers/targets."""

    excluded_prefixes = (
        "target_",
        "actual_",
        "pred_",
        "predicted",
        "prob_",
        "ev_",
    )
    excluded = {
        # Provider identifiers are join keys, not football signal.
        "old_game_id",
        "gsis",
        "pff",
        "espn",
        "ftn",
        "home_score",
        "away_score",
        "home_margin",
        "home_cover_margin",
        "total_margin",
        "is_completed",
        # Legacy outcome fields that can be present on the input wide CSV.
        "result",
        "total",
        "overtime",
        "gameLineAccuracy",
        "overUnderAccuracy",
        "homeWin",
        "awayWin",
        "isCloseGame",
        "isBlowout",
        "totalScore",
        "pointDiff",
        "spreadCovered",
        "favoriteCovered",
        "underdogCovered",
        "underdogWon",
        "overHit",
        "underHit",
        "totalHit",
        "total_line_diff",
        "moneyline_bet_return",
        "spread_bet_return",
        "totals_bet_return",
    }
    return [
        column
        for column in frame.select_dtypes(include=["number", "bool"]).columns
        if column not in excluded and not column.startswith(excluded_prefixes)
    ]
