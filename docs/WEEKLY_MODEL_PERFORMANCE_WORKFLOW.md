# Weekly Model Performance Analysis Workflow

This document outlines the steps to run model performance analysis on a weekly basis to track prediction accuracy and ROI.

## Overview

The model performance analysis compares your predictions against actual game results to calculate:
- Overall accuracy (hit rate)
- Accuracy by confidence tier
- Accuracy by prop type
- Hypothetical ROI analysis
- Historical trends

## Prerequisites

- Python environment with all dependencies installed
- Access to NFL data sources
- Predictions generated for the target week

## Weekly Workflow

### Step 1: Update Play-by-Play Data (Required for Current Season)

For the current season (2025), actual results are not available through the standard NFL API, so you need to download play-by-play data first:

```bash
# Activate your virtual environment
& venv\Scripts\Activate.ps1

# Download/update PBP data (includes all seasons 2020-2025)
python create-play-by-play.py
```

**What this does:**
- Downloads ~333MB of compressed play-by-play data from nflverse
- Extracts player stats (passing, rushing, receiving) from game data
- Saves to `data_files/nfl_play_by_play_historical.csv.gz`
- Takes ~2-3 minutes to complete

### Step 2: Run Model Performance Analysis

You have two options to run the analysis:

#### Option A: Through Streamlit UI (Recommended)

1. Start the Streamlit app:
```bash
streamlit run predictions.py
```

2. Navigate to the **📈 Model Performance** page in the sidebar

3. Select your analysis parameters:
   - **Season**: Choose the season you want to analyze
   - **Week**: Select the specific week
   - **Analysis Type**: Choose "Current Week" for individual week analysis

4. Click **"🔄 Run Fresh Analysis"** button

5. Wait for the analysis to complete (shows progress spinner)

#### Option B: Through Python Script (Advanced)

If you prefer to run analysis programmatically:

```python
from player_props.backtest import run_weekly_accuracy_check

# Run analysis for a specific week and season
results = run_weekly_accuracy_check(week=18, season=2025)

if results:
    print(f"Accuracy: {results['overall_accuracy']:.1%}")
    print(f"Total predictions: {results['total_predictions']}")
    print(f"ROI: {results.get('roi', 'N/A')}")
```

## Understanding the Results

### Key Metrics Displayed

1. **Overall Accuracy**: Percentage of predictions that were correct
2. **Total Predictions**: Number of predictions evaluated
3. **Average Confidence**: Mean confidence score of evaluated predictions
4. **Hypothetical ROI**: Expected return at -110 odds

### Analysis Breakdowns

- **By Confidence Tier**: Shows accuracy for different confidence levels (Elite ≥65%, Strong 60-65%, etc.)
- **By Prop Type**: Performance comparison across different prop types (passing yards, rushing yards, receptions, etc.)
- **Detailed Results Table**: Individual prediction outcomes with actual vs predicted values

## Automation Options

### Option 1: Scheduled Task (Windows)

Create a batch file (`weekly_model_analysis.bat`):

```batch
@echo off
cd C:\path\to\nfl-predictions
call venv\Scripts\activate.bat
python create-play-by-play.py
echo PBP data updated. Run analysis through Streamlit UI.
pause
```

Then use Windows Task Scheduler to run weekly.

### Option 2: GitHub Actions (Recommended for Production)

Set up a weekly GitHub Action that:
1. Downloads latest PBP data
2. Runs model performance analysis
3. Saves results to database/files
4. Generates weekly performance reports

## Troubleshooting

### Common Issues

1. **"Data Collection Failed"**
   - Check internet connection
   - Verify PBP data was downloaded successfully
   - Try running `create-play-by-play.py` again

2. **"No predictions found"**
   - Ensure predictions were generated for the target week
   - Check `data_files/player_props_predictions.csv` exists
   - Verify the week number matches

3. **"No matching predictions with actual results"**
   - Player names might not match between predictions and actuals
   - Check for name formatting differences
   - Verify the season/week combination has completed games

### Data Sources

- **Historical seasons (2023-2024)**: Uses nfl_data_py API (fast, reliable)
- **Current season (2025)**: Uses play-by-play data aggregation (slower but comprehensive)
- **Future seasons**: Will automatically switch to API when available

## Best Practices

1. **Run analysis immediately after games complete** (typically Monday morning)
2. **Compare performance across weeks** to identify trends
3. **Focus on high-confidence predictions** (≥65%) for best ROI
4. **Track prop type performance** to optimize model focus areas
5. **Save historical results** for long-term trend analysis

## File Locations

- **Predictions**: `data_files/player_props_predictions.csv`
- **PBP Data**: `data_files/nfl_play_by_play_historical.csv.gz`
- **Analysis Results**: Stored in memory (consider saving to database for persistence)
- **Scripts**: `scripts/` folder contains testing utilities

## Contact/Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed (`pip install -r requirements.txt`)
3. Ensure you're using Python 3.12 (not 3.13)
4. Check the scripts folder for diagnostic tools