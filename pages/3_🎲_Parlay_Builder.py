import streamlit as st
import pandas as pd
from pathlib import Path

from footer import add_betting_oracle_footer

st.title("🎲 Parlay Builder")

# Load predictions
@st.cache_data(ttl=3600)
def load_player_props_predictions():
    """Load player prop predictions with memory optimization."""
    file_path = Path('data_files/player_props_predictions.csv')
    if not file_path.exists():
        return None

    df = pd.read_csv(file_path)

    # Memory optimization
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    # Parse game_date
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])

    # Add position and display_name information by joining with players data (only if not already present)
    if 'display_name' not in df.columns or 'position' not in df.columns:
        players_df = load_players()
        if players_df is not None:
            df = df.merge(
                players_df[['short_name', 'display_name', 'position']],
                left_on='player_name',
                right_on='short_name',
                how='left'
            )
            # Use display_name for display, keep short_name for data operations
            if 'display_name' in df.columns:
                df['display_name'] = df['display_name'].fillna(df['player_name'])  # Fallback to short_name if no display_name
            df = df.drop('short_name', axis=1, errors='ignore')

    # Ensure opponent_def_rank column exists (for backward compatibility)
    if 'opponent_def_rank' not in df.columns:
        df['opponent_def_rank'] = 16.0  # Default to league average

    # Ensure injury_note column exists (for backward compatibility)
    if 'injury_note' not in df.columns:
        df['injury_note'] = None

    return df

@st.cache_data
def load_players():
    """Load players data for display names and positions."""
    try:
        players_df = pd.read_csv('data_files/players.csv')
        return players_df
    except FileNotFoundError:
        return None

def detect_correlations(props_df):
    """
    Detect correlated props that may not be independent.
    """
    warnings = []

    # Check for same-game correlations
    for (team, opponent), game_props in props_df.groupby(['team', 'opponent']):
        if len(game_props) > 1:
            # QB passing yards + team total OVERs = correlated
            qb_pass = game_props[game_props['prop_type'] == 'passing_yards']
            if not qb_pass.empty and qb_pass.iloc[0]['recommendation'] == 'OVER':
                warnings.append({
                    'warning': f"QB {qb_pass.iloc[0]['display_name']} passing OVER may correlate with other OVERs in this game ({team} vs {opponent})"
                })

    # Check for opposing players (zero-sum correlations)
    teams = props_df['team'].unique()
    opponents = props_df['opponent'].unique()
    if len(teams) == 2 and len(opponents) == 2 and set(teams) == set(opponents):
        # Opposing QBs, RBs, etc. in same game
        warnings.append({
            'warning': "Legs contain opposing teams - outcomes may be negatively correlated"
        })

    # Check for same player multiple props
    player_counts = props_df['player_name'].value_counts()
    for player, count in player_counts.items():
        if count > 1:
            player_props = props_df[props_df['player_name'] == player]
            warnings.append({
                'warning': f"Multiple props for {player_props.iloc[0]['display_name']} - outcomes are highly correlated"
            })

    return warnings

def calculate_parlay_metrics(selected_props):
    """Calculate parlay probability and odds."""
    if len(selected_props) == 0:
        return None

    # Calculate combined probability (assuming independence)
    combined_prob = selected_props['confidence'].prod()

    # Calculate decimal odds
    decimal_odds = 1 / combined_prob

    # Convert to American odds
    if decimal_odds >= 2:
        american_odds = (decimal_odds - 1) * 100
        odds_str = f"+{american_odds:.0f}"
    else:
        american_odds = -100 / (decimal_odds - 1)
        odds_str = f"{american_odds:.0f}"

    # Calculate implied win probability
    implied_prob = 1 / decimal_odds

    return {
        'combined_prob': combined_prob,
        'decimal_odds': decimal_odds,
        'american_odds': american_odds,
        'odds_str': odds_str,
        'implied_prob': implied_prob,
        'num_legs': len(selected_props)
    }

# Load predictions
predictions_df = load_player_props_predictions()

if predictions_df is None or predictions_df.empty:
    st.error("❌ No player prop predictions found. Please generate predictions first.")
    st.stop()

# Filter to high-confidence predictions only
high_conf_predictions = predictions_df[predictions_df['confidence'] >= 0.55].copy()

if high_conf_predictions.empty:
    st.warning("⚠️ No high-confidence predictions available (≥55%).")
    st.stop()

# Create options for multiselect
def create_prop_option(row):
    """Create a readable option string for each prop."""
    trend_emoji = row.get('trend', '➡️')
    return f"{row['display_name']} ({row['team']}) - {row['prop_type'].replace('_', ' ').title()} {row['recommendation']} {row['line_value']} {trend_emoji} ({row['confidence']:.1%})"

prop_options = []
prop_indices = []

for idx, row in high_conf_predictions.iterrows():
    option = create_prop_option(row)
    prop_options.append(option)
    prop_indices.append(idx)

# Multiselect for parlay legs
st.subheader("Build Your Parlay")
st.markdown("Select 2-6 high-confidence props to combine into a parlay. Higher confidence = better value!")

selected_options = st.multiselect(
    "Select props to combine (2-6 legs recommended)",
    options=prop_options,
    max_selections=8,  # Allow up to 8 for flexibility
    help="Choose props with confidence ≥55%. Correlation warnings will appear for risky combinations."
)

if len(selected_options) >= 2:
    # Get selected rows
    selected_indices = [prop_indices[prop_options.index(option)] for option in selected_options]
    selected_props = high_conf_predictions.loc[selected_indices].copy()

    # Calculate parlay metrics
    metrics = calculate_parlay_metrics(selected_props)

    # Detect correlations
    correlations = detect_correlations(selected_props)

    # Display parlay summary
    st.subheader("📊 Parlay Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Combined Probability", f"{metrics['combined_prob']:.1%}")
    with col2:
        st.metric("Fair Odds", metrics['odds_str'])
    with col3:
        st.metric("Legs", metrics['num_legs'])
    with col4:
        # Show edge (difference between our prob and implied prob)
        edge = metrics['combined_prob'] - metrics['implied_prob']
        st.metric("Edge", f"{edge:.1%}", delta=f"{edge:.1%}" if edge > 0 else None)

    # Show correlation warnings
    if correlations:
        st.warning("⚠️ Correlation Warnings Detected")
        st.markdown("**These combinations may not be truly independent:**")
        for corr in correlations:
            st.write(f"• {corr['warning']}")
        st.info("💡 Consider removing correlated legs or reducing parlay size for better value.")

    # Parlay legs table
    st.subheader("🎯 Selected Legs")

    # Prepare display dataframe
    display_df = selected_props[[
        'display_name', 'position', 'team', 'opponent', 'prop_type',
        'recommendation', 'line_value', 'confidence', 'trend',
        'avg_L3', 'avg_L5', 'avg_L10'
    ]].copy()

    # Format columns
    display_df['prop_type'] = display_df['prop_type'].str.replace('_', ' ').str.title()
    display_df['confidence'] = display_df['confidence'].map('{:.1%}'.format)
    display_df['avg_L3'] = display_df['avg_L3'].round(1)
    display_df['avg_L5'] = display_df['avg_L5'].round(1)
    display_df['avg_L10'] = display_df['avg_L10'].round(1)

    st.dataframe(
        display_df,
        column_config={
            'display_name': st.column_config.TextColumn('Player', width='medium'),
            'position': st.column_config.TextColumn('Pos', width='small'),
            'team': st.column_config.TextColumn('Team', width='small'),
            'opponent': st.column_config.TextColumn('Opp', width='small'),
            'prop_type': st.column_config.TextColumn('Prop Type', width='medium'),
            'recommendation': st.column_config.TextColumn('Bet', width='small'),
            'line_value': st.column_config.NumberColumn('Line', width='small'),
            'confidence': st.column_config.TextColumn('Conf', width='small'),
            'trend': st.column_config.TextColumn('Trend', width='small'),
            'avg_L3': st.column_config.NumberColumn('L3 Avg', width='small'),
            'avg_L5': st.column_config.NumberColumn('L5 Avg', width='small'),
            'avg_L10': st.column_config.NumberColumn('L10 Avg', width='small'),
        },
        width='stretch',
        hide_index=True
    )

    # Betting strategy tips
    st.subheader("💡 Betting Strategy Tips")

    tips_col1, tips_col2 = st.columns(2)

    with tips_col1:
        st.markdown("**🎲 Parlay Best Practices:**")
        st.markdown("- Aim for 3-5 legs maximum")
        st.markdown("- Mix different prop types")
        st.markdown("- Avoid same-game correlations")
        st.markdown("- Higher confidence = better value")

    with tips_col2:
        st.markdown("**📈 Odds Explanation:**")
        st.markdown(f"- **Fair Odds**: {metrics['odds_str']} (what you'd get if betting our probability)")
        st.markdown(f"- **Implied Win Rate**: {metrics['implied_prob']:.1%} (what sportsbook expects)")
        st.markdown(f"- **Our Edge**: {edge:.1%} (positive = good value)")

    # Export option
    if st.button("📄 Export Parlay Summary"):
        # Create summary text
        summary = f"""
NFL Player Props Parlay - {len(selected_props)} Legs

Combined Probability: {metrics['combined_prob']:.1%}
Fair Odds: {metrics['odds_str']}
Edge: {edge:.1%}

Selected Props:
"""
        for _, row in selected_props.iterrows():
            summary += f"- {row['display_name']} ({row['team']}): {row['prop_type']} {row['recommendation']} {row['line_value']} ({row['confidence']:.1%})\n"

        if correlations:
            summary += "\nCorrelation Warnings:\n"
            for corr in correlations:
                summary += f"- {corr['warning']}\n"

        st.download_button(
            label="📥 Download Summary",
            data=summary,
            file_name="nfl_parlay_summary.txt",
            mime="text/plain"
        )

elif len(selected_options) == 1:
    st.info("💡 Add at least one more prop to create a parlay!")

else:
    st.info("🎯 **How to Build a Winning Parlay:**")
    st.markdown("""
    1. **Start with High Confidence**: Look for props with ≥65% confidence (Elite tier)
    2. **Mix Prop Types**: Combine passing, rushing, and receiving props
    3. **Avoid Correlations**: Don't bet multiple props from the same game or player
    4. **Balance Risk**: 3-5 legs typically offer the best risk/reward ratio
    5. **Check Trends**: 🔥 hot players and ➡️ stable performers are good picks
    """)

    # Show top recommendations
    st.subheader("🔥 Top Recommendations")
    top_picks = high_conf_predictions.nlargest(10, 'confidence')

    for _, row in top_picks.iterrows():
        trend_emoji = row.get('trend', '➡️')
        st.write(f"**{row['display_name']}** ({row['team']}) - {row['prop_type'].replace('_', ' ').title()} {row['recommendation']} {row['line_value']} {trend_emoji} ({row['confidence']:.1%})")

# Add footer to the page
add_betting_oracle_footer()