# Current-model backtest audit (2026)

> **Implementation review — 2026-08-24.** The audit verdict still applies to the legacy production path. V2 now provides corrected settlement helpers, walk-forward evaluation, calibration metrics, CLV math, and promotion-gate code, but it has not produced a promoted, price-aware forward record. Accordingly, the “no-bet / flat shadow stakes” decision remains open and in force for production.

## Verdict

Saved overall metrics are 52.80% spread accuracy, 66.37% moneyline accuracy, and 52.21% totals accuracy. The only explicit wager-policy audit is the spread EV strategy: 54 bets from 339 games, 28-26 (51.85%), with theoretical ROI -1.01%. At standard -110, break-even is 52.38%, so the selected spread policy did not clear vig. The historical prediction CSV contains return fields whose loss/zero encoding can generate impossible positive ROIs if naively averaged; those fields should not be used until settlement accounting is repaired.

## Changes justified by the result

1. Suspend the spread EV policy; optimize probability calibration and closing-line value rather than thresholding the same test set.
2. Use rolling-season/week validation with tuning confined to prior seasons. Store fold-specific thresholds and an untouched final season.
3. Model margin and total distributions, not only binary labels; compare against closing spread/total and de-vigged moneyline.
4. Add quarterback/lineup uncertainty, EPA/play success, trenches, rest/travel/weather, coaching, and market movement with strict as-of cutoffs.

## Betting strategy decision

- **Spread:** current policy -1.01%; no-bet until forward improvement.
- **Moneyline:** 66.4% accuracy lacks price context; favorites can be accurate and unprofitable.
- **Totals/team totals:** 52.2% is below typical -110 break-even.
- **Player props:** separate opportunity/efficiency models and book-specific settlement.
- **Teasers/parlays/survivor:** require correlated simulations and rule/price comparison; teaser key-number logic must be explicit.
- **Staking:** no Kelly; flat shadow stakes only.

## Release gate

Two forward seasons, corrected settlement tests, positive CLV, 500+ bets per promoted market, and profitability after vig with season-clustered uncertainty.
