"""
Test the updated backtest module with 2025 season data.
"""
import sys
sys.path.append('.')
from player_props.backtest import run_weekly_accuracy_check

print('Testing updated backtest with 2025 Season, Week 18...')
print('=' * 80)

result = run_weekly_accuracy_check(18, 2025)

if result:
    print('\n✅ SUCCESS! Backtest completed')
    print(f'   Total predictions: {result.get("total_predictions", 0)}')
    print(f'   Overall accuracy: {result.get("overall_accuracy", 0):.1%}')
    print(f'   ROI: {result.get("roi", 0):.1f}%')
    
    # Show by prop type
    if 'by_prop_type' in result and not result['by_prop_type'].empty:
        print('\n   Accuracy by prop type:')
        for prop_type, acc in result['by_prop_type'].items():
            print(f'      {prop_type}: {acc:.1%}')
    
    # Show by confidence tier
    if 'by_confidence_tier' in result and not result['by_confidence_tier'].empty:
        print('\n   Accuracy by confidence tier:')
        for tier, acc in result['by_confidence_tier'].items():
            print(f'      {tier}: {acc:.1%}')
else:
    print('\n❌ Backtest returned no results')
