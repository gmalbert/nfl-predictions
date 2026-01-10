"""
Model Performance Dashboard

Displays prediction accuracy tracking, ROI analysis, and model calibration metrics.
Shows historical performance trends and helps validate model improvements.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from player_props.backtest import (
        run_weekly_accuracy_check,
        load_accuracy_history,
        calculate_hit_rate,
        calculate_roi,
        profitable_subset,
        collect_actual_results
    )
except ImportError:
    st.error("❌ Could not import backtest module. Please ensure player_props/backtest.py exists.")
    st.stop()

# Set page to wide layout for better chart display
st.set_page_config(layout="wide")

st.title("📈 Model Performance Dashboard")

st.markdown("""
Track prediction accuracy, ROI analysis, and model calibration over time.
Validate that our model improvements are actually working!
""")


def get_current_nfl_week() -> int:
    """
    Calculate the current NFL week based on the actual date.

    NFL season structure:
    - Regular season: 18 weeks (Weeks 1-18)
    - Playoffs: Weeks 19-22 (Wild Card, Divisional, Conference Championships, Super Bowl)

    Returns:
        Current NFL week number (1-22)
    """
    today = datetime.now()

    # Determine NFL season year
    # NFL season runs from September to February, so if we're in Jan-Feb, it's the previous year
    if today.month <= 2:  # January-February
        season_year = today.year - 1
    else:
        season_year = today.year

    # NFL season typically starts on the first Thursday in September
    # Find the first Thursday in September of the season year
    from datetime import date
    import calendar

    # Start from September 1st of the season year
    sept_1 = date(season_year, 9, 1)

    # Find the first Thursday in September
    # weekday() returns 0=Monday, 3=Thursday
    days_to_first_thursday = (3 - sept_1.weekday()) % 7
    season_start = sept_1 + timedelta(days=days_to_first_thursday)

    # If we're before the season start, return week 1
    if today.date() < season_start:
        return 1

    # Calculate weeks since season start
    days_since_start = (today.date() - season_start).days
    weeks_since_start = days_since_start // 7 + 1  # +1 because week 1 starts immediately

    # NFL regular season is 18 weeks, then playoffs
    if weeks_since_start <= 18:
        return min(weeks_since_start, 18)  # Cap at 18 for regular season
    else:
        # Playoffs: Week 19 = Wild Card, 20 = Divisional, 21 = Conference Championships, 22 = Super Bowl
        playoff_week = 18 + ((weeks_since_start - 18) // 7) + 1
        return min(playoff_week, 22)  # Cap at 22 for Super Bowl week


def get_season_for_week(week: int) -> int:
    """
    Determine the NFL season year for a given week number.

    This accounts for the fact that playoff games (weeks 19-22) from season N
    are played in January/February of year N+1.

    Args:
        week: NFL week number (1-22)

    Returns:
        Season year for the given week
    """
    current_week = get_current_nfl_week()
    today = datetime.now()

    # Determine current season year
    # NFL season runs from September to February
    if today.month <= 2:  # January-February (playoffs)
        current_season = today.year - 1  # 2025 season plays in Jan/Feb 2026
    else:
        current_season = today.year

    # If the requested week is <= current week, it's from the current season
    if week <= current_week:
        return current_season

    # If the requested week > current week, we need to look backwards
    # This handles historical analysis (e.g., analyzing week 18 when we're in week 19)
    else:
        # For now, assume we're only looking at recent weeks within the same season
        # If we need to analyze older seasons, this logic would need enhancement
        return current_season

def get_dataframe_height(df, row_height=35, header_height=38, padding=2, max_height=600):
    """
    Calculate the optimal height for a Streamlit dataframe based on number of rows.
    
    Args:
        df (pd.DataFrame): The dataframe to display
        row_height (int): Height per row in pixels. Default: 35
        header_height (int): Height of header row in pixels. Default: 38
        padding (int): Extra padding in pixels. Default: 2
        max_height (int): Maximum height cap in pixels. Default: 600 (None for no limit)
    
    Returns:
        int: Calculated height in pixels
    
    Example:
        height = get_dataframe_height(my_df)
        st.dataframe(my_df, height=height)
    """
    num_rows = len(df)
    calculated_height = (num_rows * row_height) + header_height + padding
    
    if max_height is not None:
        return min(calculated_height, max_height)
    return calculated_height

# Sidebar controls
st.sidebar.header("📊 Analysis Controls")

# Season selection - current season and two prior seasons
today = datetime.now()
if today.month <= 2:  # January-February (playoffs)
    current_season = today.year - 1
else:
    current_season = today.year

available_seasons = [current_season - 2, current_season - 1, current_season]
season_labels = {
    current_season - 2: f"{current_season - 2} Season",
    current_season - 1: f"{current_season - 1} Season",
    current_season: f"{current_season} Season"
}

selected_season_idx = st.sidebar.selectbox(
    "Select Season to Analyze",
    options=range(len(available_seasons)),
    format_func=lambda i: season_labels[available_seasons[i]],
    index=len(available_seasons)-1,  # Default to current season
    help="Choose which NFL season to analyze. Note: Current season data may not be available yet from the API."
)
selected_season = available_seasons[selected_season_idx]

# Week selection - show appropriate weeks for selected season
if selected_season == 2025:
    # Current season - we're in week 19 (playoffs)
    current_week = get_current_nfl_week()
    max_week = min(current_week, 18)  # Regular season only for now
    st.sidebar.info(f"ℹ️ **2025 Season Note**: We're currently in Week {current_week}. Data for 2025 may not be available yet. If analysis fails, try 2024 Season.")
elif selected_season == 2024:
    max_week = 18  # Full regular season available
elif selected_season == 2023:
    max_week = 18  # Full regular season available
else:
    max_week = 18

available_weeks = list(range(1, max_week + 1))
selected_week = st.sidebar.selectbox(
    "Select Week to Analyze",
    options=available_weeks,
    index=len(available_weeks)-1,  # Default to most recent
    help=f"Choose which week from {selected_season} season to analyze against actual results"
)

# Analysis type
analysis_type = st.sidebar.radio(
    "Analysis Type",
    ["Current Week", "Historical Trends", "ROI Analysis"],
    help="Choose what type of analysis to display"
)

# Auto-run analysis button
if st.sidebar.button("🔄 Run Fresh Analysis", help="Re-run accuracy analysis for selected week"):
    with st.spinner("Running accuracy analysis..."):
        # First check if we can collect actual results
        test_df, error_msg = collect_actual_results(selected_week, selected_season)

        if test_df.empty and error_msg:
            st.sidebar.error(f"❌ **Data Collection Failed**: {error_msg}")
            st.sidebar.info("💡 **Troubleshooting Tips:**\n"
                           "- Check your internet connection\n"
                           "- The NFL data API may be temporarily unavailable\n"
                           "- Historical data may not be available for recent seasons\n"
                           "- Try selecting an earlier week with completed games")
        else:
            accuracy_results = run_weekly_accuracy_check(selected_week, selected_season)
            if accuracy_results:
                st.sidebar.success(f"✅ Analysis complete for Week {selected_week}")
                st.rerun()
            else:
                st.sidebar.error("❌ Analysis failed - check data availability")


def display_current_week_analysis(week: int, season: int):
    """Display accuracy analysis for a specific week."""
    st.header(f"🎯 Week {week} Accuracy Analysis")

    # Load predictions and actual results
    predictions_file = f"data_files/player_props_predictions_week{week}.csv"
    if not Path(predictions_file).exists():
        predictions_file = "data_files/player_props_predictions.csv"

    if not Path(predictions_file).exists():
        st.error(f"❌ No predictions file found for Week {week}")
        return

    predictions_df = pd.read_csv(predictions_file)

    # Filter to high-confidence predictions
    high_conf_predictions = predictions_df[predictions_df['confidence'] >= 0.55]

    if high_conf_predictions.empty:
        st.warning(f"⚠️ No high-confidence predictions (≥55%) found for Week {week}")
        return

    # Collect actual results
    actuals_df, error_msg = collect_actual_results(week, season)

    if actuals_df.empty:
        if error_msg:
            st.error(f"❌ **Data Collection Failed**: {error_msg}")
            st.info("💡 **Troubleshooting Tips:**\n"
                   "- Check your internet connection\n"
                   "- The NFL data API may be temporarily unavailable\n"
                   "- Historical data may not be available for recent seasons\n"
                   "- Try selecting an earlier week with completed games")
        else:
            st.info(f"ℹ️ Actual results for Week {week} are not yet available. Games may still be in progress.")
        st.markdown("**Preview Analysis** (based on available data)")

        # Show prediction distribution
        fig = px.histogram(
            high_conf_predictions,
            x='confidence',
            nbins=20,
            title=f"Week {week} Prediction Confidence Distribution",
            labels={'confidence': 'Model Confidence', 'count': 'Number of Predictions'}
        )
        st.plotly_chart(fig, width='stretch')

        return

    # Calculate accuracy metrics
    accuracy_metrics = calculate_hit_rate(high_conf_predictions, actuals_df)

    if accuracy_metrics['total_predictions'] == 0:
        st.warning("⚠️ No matching predictions found with actual results")
        return

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Overall Accuracy",
            f"{accuracy_metrics['overall_accuracy']:.1%}",
            help="Percentage of predictions that were correct"
        )

    with col2:
        st.metric(
            "Total Predictions",
            accuracy_metrics['total_predictions'],
            help="Number of predictions evaluated"
        )

    with col3:
        # Calculate average confidence
        avg_conf = accuracy_metrics['detailed_results']['confidence'].mean()
        st.metric(
            "Avg Confidence",
            f"{avg_conf:.1%}",
            help="Average model confidence for evaluated predictions"
        )

    with col4:
        # Calculate ROI
        roi_metrics = calculate_roi(accuracy_metrics['detailed_results'])
        st.metric(
            "Hypothetical ROI",
            f"{roi_metrics['roi']:.1f}%",
            delta=f"{roi_metrics['roi']:.1f}%" if roi_metrics['roi'] != 0 else None,
            help=f"ROI at -110 odds. Breakeven: {roi_metrics['breakeven_rate']:.1f}%"
        )

    # Accuracy by confidence tier
    st.subheader("🎯 Accuracy by Confidence Tier")

    if not accuracy_metrics['by_confidence_tier'].empty:
        # Create bar chart
        conf_df = accuracy_metrics['by_confidence_tier'].reset_index()
        conf_df.columns = ['Confidence Tier', 'Accuracy']

        fig = px.bar(
            conf_df,
            x='Confidence Tier',
            y='Accuracy',
            title="Prediction Accuracy by Confidence Level",
            labels={'Accuracy': 'Hit Rate'},
            color='Accuracy',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig, width='stretch')

        # Show as table too
        st.dataframe(
            conf_df.style.format({'Accuracy': '{:.1%}'}),
            width=600,
            height=get_dataframe_height(conf_df),
            hide_index=True
        )
    else:
        st.info("Not enough data for confidence tier analysis")

    # Accuracy by prop type
    st.subheader("🏈 Accuracy by Prop Type")

    if not accuracy_metrics['by_prop_type'].empty:
        prop_df = accuracy_metrics['by_prop_type'].reset_index()
        prop_df.columns = ['Prop Type', 'Accuracy']

        # Sort by accuracy
        prop_df = prop_df.sort_values('Accuracy', ascending=False)

        fig = px.bar(
            prop_df,
            x='Prop Type',
            y='Accuracy',
            title="Prediction Accuracy by Prop Type",
            labels={'Accuracy': 'Hit Rate'},
            color='Accuracy',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Not enough data for prop type analysis")

    # Detailed results table
    st.subheader("📋 Detailed Results")

    detailed_df = accuracy_metrics['detailed_results'][[
        'display_name', 'prop_type', 'line_value', 'predicted', 'actual', 'actual_value', 'hit', 'confidence'
    ]].copy()

    # Format columns
    detailed_df['prop_type'] = detailed_df['prop_type'].str.replace('_', ' ').str.title()
    detailed_df['confidence'] = detailed_df['confidence'].map('{:.1%}'.format)
    detailed_df['hit'] = detailed_df['hit'].map({True: '✅', False: '❌'})

    height = get_dataframe_height(detailed_df)

    st.dataframe(
        detailed_df,
        column_config={
            'display_name': st.column_config.TextColumn('Player', width='medium'),
            'prop_type': st.column_config.TextColumn('Prop Type', width='medium'),
            'line_value': st.column_config.NumberColumn('Line', width='small'),
            'predicted': st.column_config.TextColumn('Predicted', width='small'),
            'actual': st.column_config.TextColumn('Actual', width='small'),
            'actual_value': st.column_config.NumberColumn('Actual Value', width='small'),
            'hit': st.column_config.TextColumn('Hit', width='small'),
            'confidence': st.column_config.TextColumn('Confidence', width='small'),
        },
        width=1000,
        height=height,
        hide_index=True
    )


def display_historical_trends():
    """Display historical accuracy trends across multiple weeks."""
    st.header("📈 Historical Performance Trends")

    # Load historical accuracy data
    history_df = load_accuracy_history()

    if history_df.empty:
        st.info("ℹ️ No historical accuracy data found. Run some weekly analyses first!")
        st.markdown("""
        **To build historical data:**
        1. Select "Current Week" analysis
        2. Choose different weeks
        3. Click "🔄 Run Fresh Analysis" for each week
        4. Return here to see trends
        """)
        return

    # Display summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        avg_accuracy = history_df['overall_accuracy'].mean()
        st.metric("Average Accuracy", f"{avg_accuracy:.1%}")

    with col2:
        total_predictions = history_df['total_predictions'].sum()
        st.metric("Total Predictions", f"{total_predictions:,}")

    with col3:
        weeks_analyzed = len(history_df)
        st.metric("Weeks Analyzed", weeks_analyzed)

    # Accuracy trend over time
    st.subheader("📊 Accuracy Trend Over Time")

    fig = px.line(
        history_df,
        x='week',
        y='overall_accuracy',
        title="Weekly Prediction Accuracy Trend",
        labels={'week': 'NFL Week', 'overall_accuracy': 'Accuracy'},
        markers=True
    )
    fig.update_layout(yaxis_tickformat='.1%')
    fig.update_xaxes(tickmode='linear')
    st.plotly_chart(fig, width='stretch')

    # Volume trend
    st.subheader("📈 Prediction Volume Trend")

    fig = px.bar(
        history_df,
        x='week',
        y='total_predictions',
        title="Weekly Prediction Volume",
        labels={'week': 'NFL Week', 'total_predictions': 'Predictions'}
    )
    fig.update_xaxes(tickmode='linear')
    st.plotly_chart(fig, width='stretch')

    # Historical data table
    st.subheader("📋 Historical Results")

    display_df = history_df.copy()
    display_df['overall_accuracy'] = display_df['overall_accuracy'].map('{:.1%}'.format)
    display_df = display_df.sort_values('week', ascending=False)

    st.dataframe(
        display_df,
        column_config={
            'week': st.column_config.NumberColumn('Week', width='small'),
            'overall_accuracy': st.column_config.TextColumn('Accuracy', width='small'),
            'total_predictions': st.column_config.NumberColumn('Predictions', width='small'),
            'timestamp': st.column_config.TextColumn('Analysis Date', width='medium'),
        },
        width='stretch',
        hide_index=True
    )


def display_roi_analysis(week: int, season: int):
    """Display ROI analysis for different confidence thresholds."""
    st.header("💰 ROI Analysis")

    # Load predictions and actual results
    predictions_file = f"data_files/player_props_predictions_week{week}.csv"
    if not Path(predictions_file).exists():
        predictions_file = "data_files/player_props_predictions.csv"

    if not Path(predictions_file).exists():
        st.error(f"❌ No predictions file found for Week {week}")
        return

    predictions_df = pd.read_csv(predictions_file)
    actuals_df, error_msg = collect_actual_results(week, season)

    if actuals_df.empty:
        if error_msg:
            st.error(f"❌ **Data Collection Failed**: {error_msg}")
            st.info("💡 **Troubleshooting Tips:**\n"
                   "- Check your internet connection\n"
                   "- The NFL data API may be temporarily unavailable\n"
                   "- Historical data may not be available for recent seasons\n"
                   "- Try selecting an earlier week with completed games")
        else:
            st.info(f"ℹ️ Actual results for Week {week} are not yet available for ROI analysis.")
        return

    # Calculate accuracy metrics
    accuracy_metrics = calculate_hit_rate(predictions_df, actuals_df)

    if accuracy_metrics['total_predictions'] == 0:
        st.warning("⚠️ No matching predictions found with actual results")
        return

    # ROI analysis for different confidence thresholds
    roi_table = profitable_subset(accuracy_metrics['detailed_results'])

    if roi_table.empty:
        st.warning("⚠️ Not enough data for ROI analysis")
        return

    st.subheader("💵 ROI by Confidence Threshold")

    # Format the ROI table for display
    display_roi = roi_table.copy()
    display_roi['hit_rate'] = display_roi['hit_rate'].map('{:.1%}'.format)
    display_roi['roi'] = display_roi['roi'].map('{:.1f}%'.format)
    display_roi['breakeven_rate'] = display_roi['breakeven_rate'].map('{:.1f}%'.format)

    st.dataframe(
        display_roi[['total_bets', 'hit_rate', 'roi', 'breakeven_rate']],
        column_config={
            'total_bets': st.column_config.NumberColumn('Bets', width='small'),
            'hit_rate': st.column_config.TextColumn('Hit Rate', width='small'),
            'roi': st.column_config.TextColumn('ROI', width='small'),
            'breakeven_rate': st.column_config.TextColumn('Breakeven', width='small'),
        },
        width='stretch',
    )

    # Find best performing threshold
    if not roi_table.empty:
        best_threshold = roi_table['roi'].idxmax()
        best_roi = roi_table.loc[best_threshold, 'roi']

        st.success(f"🎯 **Best Strategy**: Bet on {best_threshold:.0%}+ confidence props (ROI: {best_roi:.1f}%)")

    # ROI vs Confidence Threshold Chart
    st.subheader("📊 ROI vs Confidence Threshold")

    fig = px.line(
        roi_table.reset_index(),
        x='index',
        y='roi',
        title="ROI by Minimum Confidence Threshold",
        labels={'index': 'Minimum Confidence', 'roi': 'ROI (%)'},
        markers=True
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Breakeven")
    fig.update_layout(yaxis_tickformat='.1f')
    st.plotly_chart(fig, width='stretch')

    # Hit Rate vs Confidence Threshold
    st.subheader("🎯 Hit Rate vs Confidence Threshold")

    fig = px.line(
        roi_table.reset_index(),
        x='index',
        y='hit_rate',
        title="Hit Rate by Minimum Confidence Threshold",
        labels={'index': 'Minimum Confidence', 'hit_rate': 'Hit Rate'},
        markers=True
    )
    fig.update_layout(yaxis_tickformat='.1%')
    st.plotly_chart(fig, width='stretch')

    # Betting strategy recommendations
    st.subheader("🎲 Betting Strategy Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**✅ Profitable Thresholds:**")
        profitable = roi_table[roi_table['roi'] > 0]
        if not profitable.empty:
            for threshold, row in profitable.iterrows():
                st.write(f"• {threshold:.0%}+ confidence: {row['roi']:.1f}% ROI")
        else:
            st.write("• None found - model needs improvement")

    with col2:
        st.markdown("**📈 Improvement Opportunities:**")
        st.write("• Focus on prop types with low accuracy")
        st.write("• Investigate high-confidence misses")
        st.write("• Consider adjusting confidence thresholds")


# Main content based on analysis type
if analysis_type == "Current Week":
    display_current_week_analysis(selected_week, selected_season)

elif analysis_type == "Historical Trends":
    display_historical_trends()

elif analysis_type == "ROI Analysis":
    display_roi_analysis(selected_week, selected_season)


# Footer
st.markdown("---")
st.markdown("*Dashboard automatically updates when new accuracy analyses are run.*")
st.markdown("*ROI calculations assume standard -110 American odds.*")