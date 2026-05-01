"""
scripts/export_best_bets.py — NFL
Reads data_files/betting_recommendations_log.csv and writes
data_files/best_bets_today.json in the unified Sports Picks Grid schema.
"""
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

SPORT = "NFL"
MODEL_VERSION = "1.0.0"
SEASON = str(date.today().year)
OUT_PATH = Path("data_files/best_bets_today.json")
LOG_PATH = Path("data_files/betting_recommendations_log.csv")


def _write(bets: list, notes: str = "") -> None:
    payload: dict = {
        "meta": {
            "sport": SPORT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_version": MODEL_VERSION,
            "season": SEASON,
        },
        "bets": bets,
    }
    if notes:
        payload["meta"]["notes"] = notes
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[{SPORT}] Wrote {len(bets)} bets -> {OUT_PATH}")


def main() -> None:
    today = date.today()

    # Off-season: still write empty bets so the aggregator doesn't error
    month = today.month
    if not (month >= 9 or month <= 2):
        _write([], "NFL off-season")
        return

    if not LOG_PATH.exists():
        _write([], f"Source file not found: {LOG_PATH}")
        return

    try:
        df = pd.read_csv(LOG_PATH)
    except Exception as e:
        _write([], f"Failed to read source: {e}")
        return

    # Normalise date column
    date_col = next((c for c in df.columns if "game" in c.lower() and "day" in c.lower()), None)
    if date_col is None and "gameday" in df.columns:
        date_col = "gameday"
    if date_col is None:
        _write([], "No date column found in recommendations log")
        return

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    today_df = df[df[date_col] == today].copy()

    TIER_KEEP = {"Elite", "Strong", "Good"}
    tier_col = next((c for c in df.columns if "tier" in c.lower() or "confidence_tier" in c.lower()), None)
    if tier_col and tier_col in today_df.columns:
        today_df = today_df[today_df[tier_col].isin(TIER_KEEP)]

    if today_df.empty:
        _write([], f"No picks for {today}")
        return

    bets = []
    for _, row in today_df.iterrows():
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        game = f"{away} @ {home}" if home and away else str(row.get("game", ""))
        bet_type_raw = str(row.get("bet_type", "")).strip()
        bt_map = {"Spread": "Spread", "Moneyline": "Moneyline", "Over/Under": "Over/Under",
                  "spread": "Spread", "moneyline": "Moneyline", "total": "Over/Under"}
        bet_type = bt_map.get(bet_type_raw, bet_type_raw)

        pick = str(row.get("recommended_team", row.get("pick", "")))
        confidence = _safe_float(row.get("model_probability", row.get("confidence")))
        edge = _safe_float(row.get("edge"))
        tier = str(row.get(tier_col, row.get("tier", "Good"))) if tier_col else "Good"
        odds = _safe_int(row.get("moneyline_odds", row.get("odds")))
        line_val = _safe_float(row.get("spread_line", row.get("total_line", row.get("line"))))

        bet: dict = {
            "game_date": str(today),
            "game_time": str(row.get("gametime", "")) or None,
            "game": game,
            "home_team": home,
            "away_team": away,
            "bet_type": bet_type,
            "pick": pick,
            "confidence": confidence,
            "edge": edge,
            "tier": tier,
            "odds": odds,
            "line": line_val,
            "notes": str(row.get("notes", "")) or None,
        }
        bets.append(bet)

    _write(bets)


def _safe_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
