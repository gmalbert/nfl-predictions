# Spread Betting Threshold Change - December 13, 2025

## What Changed

**Yesterday (Dec 12)**: Implemented strict EV-based threshold (52.4%) for spread betting  
**Today (Dec 13)**: Reverted to hybrid approach (45% threshold) with EV status indicators

## Update — Dec 13, 2025 (Model Fix)

After investigation we discovered the spread model predictions were inverted due to a mislabeled target in the training pipeline. The pipeline was updated to flip the spread probability immediately after prediction (`prob_underdogCovered = 1 - prob_underdogCovered`). After applying the fix and retraining:

- The model now identifies a large set of positive-EV opportunities (62 of 63 remaining games flagged as profitable in the immediate post-fix run).
- Betting ROI on historical/backtest runs improved from approximately -90% to roughly **+60%**.
- Calibration error reduced (from ~45% to ~28%) and maximum model confidence rose to **89.5%**.

See `MODEL_FIX_PLAN.md` for a full technical write-up and `NEW_FEATURES_DEC13.md` for details on the 18 new features added alongside this fix.

## Why The Change?

### Problem Discovered
After implementing the EV-based system yesterday:
- **0 upcoming games** met the 52.4% threshold
- **All 63 future games have negative EV** (max EV: -$4.69)
- **Max probability: 49.9%** (below breakeven)
- Users couldn't see any spread betting opportunities

### Root Cause
The model is appropriately conservative on future 2025 games:
- Limited historical data for 2025 season
- Probabilities cluster around 38-50% (below breakeven)
- Model correctly identifies these as unprofitable bets

### Solution: Hybrid Approach
Show games ≥45% confidence BUT clearly label their EV status:
- **✅ +EV**: Positive expected value = mathematically profitable
- **⚠️ -EV**: Negative expected value = losing bet (entertainment only)

## What Users Will See Now

### Current Situation (Dec 13)
- **6 games displayed** with ≥45% confidence
- **0 games with +EV** (all are -EV)
- **Clear warning** that all bets are expected to lose money
- **EV Status column** shows ⚠️ -EV for all games

### Example Display
| Date | Teams | Confidence | EV Status | Expected Value |
|------|-------|-----------|-----------|----------------|
| 12/21 | CIN @ MIA | 49.9% | ⚠️ -EV | -$4.69 |
| 12/14 | NYJ @ JAX | 47.4% | ⚠️ -EV | -$9.52 |
| 12/20 | PHI @ WAS | 46.6% | ⚠️ -EV | -$10.94 |

## User Messaging

### Warning Displayed
```
⚠️ NO POSITIVE EV BETS AVAILABLE:
- All 6 games shown have negative expected value (-EV)
- These are for entertainment/research only - expect to lose money over time
- Max probability: 49.9% (need 52.4%+ for breakeven)
```

### Info Panel
```
💡 How to Use This Table:
- EV Status: ✅ +EV = Bet this! | ⚠️ -EV = Avoid (will lose money long-term)
- Model Confidence: Probability that underdog covers the spread
- Expected Value: Predicted profit/loss per $100 bet
- Strategy: Only bet on +EV games with highest confidence

⚠️ Important: -EV bets are shown for reference but are NOT recommended.
```

## Technical Details

### Code Changes (predictions.py)

**Line 3054-3068**: Filter changed from 52.4% to 45%
```python
# OLD (strict EV-based):
spread_bets = predictions_df_spread[
    (predictions_df_spread['prob_underdogCovered'] >= 0.524) & 
    (predictions_df_spread['ev_spread'] > 0)
].copy()

# NEW (hybrid with EV indicators):
spread_bets = predictions_df_spread[
    predictions_df_spread['prob_underdogCovered'] >= 0.45
].copy()

# Add EV status flag
spread_bets['ev_status'] = spread_bets['ev_spread'].apply(
    lambda x: '✅ +EV' if x > 0 else '⚠️ -EV'
)
```

**Line 3045**: Header updated
```python
# OLD:
"*Games with positive expected value (EV > 0) and ≥52.4% win probability*"

# NEW:
"*Games where model has ≥45% confidence. ✅ +EV = mathematically profitable, ⚠️ -EV = for entertainment only*"
```

**Display Changes**:
- Added "EV Status" column with ✅/⚠️ icons
- Show count of +EV vs total games
- Display warning when no +EV bets available
- Sort by confidence instead of EV (since all EV is negative)

## Philosophy

### Transparency Over Restriction
- **Show the data** even when it's not favorable
- **Clearly label** which bets are expected to lose money
- **Educate users** on Expected Value concept
- **Let users decide** (with full information)

### Alternative: Strict EV-Only
Could have kept 52.4% threshold showing 0 bets, but:
- Users wanted to see SOMETHING
- Educational value in showing WHY bets are bad
- Research/entertainment use cases exist
- Clear warnings prevent uninformed betting

## Next Steps

### Short-term
1. **Monitor user feedback** on hybrid approach
2. **Track if +EV bets appear** as more 2025 games are played
3. **Update warnings** if situation changes

### Medium-term
1. **Add momentum features** to improve 2025 predictions
2. **Implement separate model** for in-season vs pre-season
3. **Add weather/rest features** to boost probabilities

### Long-term
1. **Build confidence** as 2025 season progresses
2. **Retrain models** with 2025 data monthly
3. **Consider ensemble methods** for better calibration

## Summary

**Trade-off Made**: Transparency + Education > Strict Mathematical Purity

Users now see:
- ✅ All available games (6 vs 0)
- ✅ Clear EV status for each
- ✅ Strong warnings about -EV bets
- ✅ Educational content on Expected Value
- ⚠️ But risk of users ignoring warnings and betting anyway

**Recommendation**: Only bet when ✅ +EV appears. Currently 0 spread bets meet this criteria.

---

**Date**: December 13, 2025  
**Trigger**: User feedback ("why aren't any spread bets shown?")  
**Status**: ✅ Implemented  
**User Impact**: Will now see 6 games with clear -EV warnings
