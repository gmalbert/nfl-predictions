# NFL Predictions — Next 5 Features to Implement

> **Based on:** Codebase gap analysis as of July 2025

---

## Feature 1: Coaching Matchup Feature

**Why:** Head coach win rate against specific coordinators/coaches is a publicly available signal that is almost never captured in public models. Teams with winning head coaches have a measurable edge in close games (4th-down decisions, 2-point conversions, clock management).

**How:**
1. Add `data_files/nfl_coaches.csv` mapping team → head coach + offensive/defensive coordinator (manually maintained per season)
2. Compute per-coach features: career ATS win%, career over/under record, ATS % as home/away favorite
3. Add `home_hc_ats_pct` and `away_hc_ats_pct` to the feature set in `nfl-gather-data.py`
4. Use rolling 3-season window to reduce sample size issues with new coaches

**Complexity:** Medium

---

## Feature 2: Quarterback Change Detection

**Why:** Backup QBs are one of the biggest model blind spots. A starter injury that requires a backup elevates uncertainty dramatically and is often a value bet opportunity when the market hasn't fully adjusted. Starting QB is confirmed ~1 hour before kickoff.

**How:**
1. Add `scripts/fetch_qb_status.py` that polls the ESPN NFL injury/roster API for starting QB confirmation per game
2. If the announced starter differs from the `nfl_data_py` season-long starter, set `home_qb_backup = 1` flag
3. Apply a statistical adjustment: backup QBs average ~6% lower completion rate and ~1.5 fewer points per game
4. Display "⚠ Backup QB" alert prominently on the spread and moneyline tabs

**Complexity:** Medium

---

## Feature 3: Weather Impact Feature (Full Integration)

**Why:** `NEW_FEATURES_DEC13.md` documents weather features (cold ≤32°F, windy ≥15mph, extreme) were added in December 2025 — but these use historical weather, not actual game-day forecasts. Switching to real forecast data for upcoming games would materially improve totals predictions.

**How:**
1. Add `scripts/fetch_game_weather.py` that calls Open-Meteo forecast API for each game's stadium coordinates on game day
2. Fetch: `temperature_2m`, `wind_speed_10m`, `precipitation_probability` per hour at kickoff time
3. Compute: `is_cold`, `is_windy`, `is_dome_game` (binary: is the stadium a closed/dome venue)
4. Update the totals model to use forecast weather rather than historical averages for upcoming games

**Complexity:** Low

---

## Feature 4: Division Rivalry Boost Feature

**Why:** Divisional games in the NFL have significantly higher variance than non-divisional games. Divisional opponents know each other's tendencies, game plans are more aggressive, and teams often perform differently vs divisional rivals regardless of current form.

**How:**
1. Add a lookup in `nfl-gather-data.py` that determines if the two teams are in the same division (AFC East, NFC North, etc.)
2. Set `is_divisional_game` binary feature
3. Compute per-team `divisional_ats_pct` (ATS record in divisional games, last 3 seasons)
4. Feature is especially relevant for the spread model — division games have higher upset rates

**Complexity:** Low

---

## Feature 5: DK Pick 6 Model Improvement (Target Share for Receivers)

**Why:** The player prop models in `player_props/` currently use L3/L5/L10 rolling stats but `target_share` has already been identified as a key usage feature. Ensuring `target_share` is consistently included for WR/TE predictions would improve Pick 6 recommendation accuracy.

**How:**
1. Verify `target_share` is in the feature vector for WR/TE models in `player_props/predict.py`
2. If missing: compute `target_share = targets / team_total_targets` from `nfl_data_py` play-by-play data
3. Add `air_yards_share` as a companion feature (correlated with big-play receiving targets)
4. Retrain WR/TE models in `player_props/train_models.py` with these additions
5. Validate improvement using the `pages/2_Player_Props.py` probability vs hit-rate comparison

**Complexity:** Medium
