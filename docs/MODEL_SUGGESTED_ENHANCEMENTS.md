# NFL Predictions — Model Suggested Enhancements

## Priority 1: Spread Model (Current Threshold: 50%)

### DVOA Integration
- Football Outsiders DVOA (Defense-adjusted Value Over Average) is among the most predictive NFL metrics.
- Add `home_dvoa`, `away_dvoa`, and `dvoa_differential` from the free weekly DVOA summary.

### Quarterback Rating Features
- Add `qbr_l4` (ESPN QBR rolling 4-week) for starting QBs. QB changes mid-season should reset this feature to league average.

### Weather Impact Refinement
- Current weather features: `is_cold`, `is_windy`, `is_extreme`. Add `wind_chill_feels_like` as a continuous feature instead of binary thresholds.

### Offensive Line Quality
- ProFootballFocus O-line grades are free in aggregate form. Add `home_oline_rank` and `away_oline_rank`.

## Priority 2: Moneyline Model (Current Threshold: 28%)

### Underdog Value
- The 28% threshold captures underdogs systematically. Add `implied_market_prob_from_dk` as a direct feature so the model learns where the market over/undervalues dogs.

### Home-Field Advantage Decay
- Analyst research shows HFA has declined in the post-COVID era. Add `season_year` as a feature to allow the model to learn HFA trends over time.

## Priority 3: Player Props

### Target Share Stability
- `target_share` is already integrated as a feature. Add `target_share_vs_cb_rank` to capture matchup-specific value.

### Red Zone Receiving Rate
- Players who catch a high % of targets inside the 20 have elevated TD probability. Add `rz_target_rate`.

## Priority 4: Calibration

- Continue tracking CLV (closing line value) for all bets placed.
- Publish weekly calibration curve on the Model Performance tab.
- Maintain the spread model inversion fix: `prob_underdogCovered = 1 - prob_underdogCovered` in training.
