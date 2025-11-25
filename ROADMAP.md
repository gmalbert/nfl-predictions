# 🗺️ NFL Predictions - Product Roadmap

[⬅️ Back to README](./README.md)

## 📋 Current Status: Beta 0.1 (November 2025)

### ✅ Shipped Features
- Three specialized XGBoost models (spread, moneyline, totals)
- Interactive predictions dashboard with confidence tiers
- Historical data page with advanced filtering (196k+ records)
- Multi-page Streamlit app with optimized performance
- Data leakage-free models with proven ROI
- Git LFS integration for large datasets
- Lazy loading architecture for fast startup

---

## 🚀 Release Schedule & Enhancement Plan

### **Beta 0.1** (Ship: **November 5, 2025** ✅)
**Focus**: Core betting predictions and historical analysis

**Included**:
- ✅ Spread, moneyline, and totals predictions
- ✅ Top 10 betting recommendations per strategy
- ✅ Historical play-by-play data filtering
- ✅ Confidence tiers with visual indicators
- ✅ Performance-optimized for Streamlit Cloud
- ✅ Proper error handling and graceful fallbacks

**Success Metrics**:
- 10-20 beta users onboarded
- <10 second initial load time
- No timeout errors on Historical Data page
- User feedback collected

---

### **Beta 0.2** (Target: **Late November 2025** - 2-3 weeks)
**Focus**: User experience enhancements based on beta 0.1 feedback

#### 🎯 Planned Enhancements

#### ~~1. **Loading Progress Indicators** (Priority: HIGH)~~ ✅ **COMPLETED**
~~**Problem**: Users don't know what's happening during 5-10 second initial load~~

~~**Implementation**:
```python
# In predictions.py
with st.spinner("🏈 Loading NFL data and predictions..."):
    progress_bar = st.progress(0)
    
    # Load historical data
    progress_bar.progress(25, text="Loading historical games...")
    historical_game_level_data = load_historical_data()
    
    # Load predictions
    progress_bar.progress(50, text="Loading model predictions...")
    predictions_df = load_predictions()
    
    # Load play-by-play data
    progress_bar.progress(75, text="Loading play-by-play data...")
    historical_data = load_data()
    
    progress_bar.progress(100, text="Ready!")
    time.sleep(0.5)  # Brief pause to show completion
    progress_bar.empty()
```

**Effort**: 2-3 hours  
**Impact**: Significantly improves perceived performance~~

---

#### 2. **Data Export Functionality** (Priority: MEDIUM)
**Problem**: Users want to analyze filtered data in Excel or save betting recommendations

**Implementation**:
```python
# Add to each tab with dataframes
import io

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# In betting recommendations tabs
st.download_button(
    label="📥 Download Recommendations as CSV",
    data=convert_df_to_csv(filtered_df),
    file_name=f'nfl_spread_bets_{datetime.now().strftime("%Y%m%d")}.csv',
    mime='text/csv',
)

# In Historical Data page
st.download_button(
    label="📥 Export Filtered Data",
    data=convert_df_to_csv(filtered_data),
    file_name=f'nfl_historical_data_{datetime.now().strftime("%Y%m%d")}.csv',
    mime='text/csv',
)
```

**Effort**: 3-4 hours  
**Impact**: Enables power users to do custom analysis

---

#### ~~3. **Cache Management UI** (Priority: MEDIUM)~~ ✅ **COMPLETED**
~~**Problem**: Users can't refresh predictions or clear cached data manually~~

~~**Implementation**:
```python
# Add to sidebar
with st.sidebar:
    st.write("### ⚙️ Settings")
    
    if st.button("🔄 Refresh Data", help="Clear cache and reload all data"):
        st.cache_data.clear()
        st.rerun()
```

**Effort**: 2 hours  
**Impact**: Gives users control over data freshness~~

---

#### 3. **Enhanced Filtering UI** (Priority: LOW)
**Problem**: Historical Data page filters could be more intuitive

**Implementation**:
```python
# Group filters into expandable sections
with st.sidebar:
    with st.expander("⚙️ Team Filters", expanded=True):
        selected_offense = st.multiselect(...)
        selected_defense = st.multiselect(...)
    
    with st.expander("📊 Game Situation Filters"):
        selected_downs = st.multiselect(...)
        selected_qtrs = st.multiselect(...)
        
    with st.expander("📈 Advanced Metrics"):
        epa_range = st.slider(...)
        wp_range = st.slider(...)
    
    # Add preset filters
    st.write("**Quick Filters**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Red Zone"):
            # Auto-set yardline_100 to 0-20
            pass
    with col2:
        if st.button("2-Minute Drill"):
            # Auto-set game_seconds_remaining < 120
            pass
```

**Effort**: 4-5 hours  
**Impact**: Makes complex filtering more accessible

---

#### **Beta 0.2 Priorities** (Choose 2-3 based on user feedback)

**If users say**:
- *"I don't know if it's working"* → **Loading Progress** (MUST HAVE)
- *"I want to track my bets in Excel"* → **Data Export** (HIGH VALUE)
- *"Data seems stale"* → **Cache Management** (QUICK WIN)
- *"Filters are overwhelming"* → **Enhanced Filtering** (NICE TO HAVE)

**Estimated Timeline**: 10-15 hours total development

---

### **Beta 0.3** (Target: **Mid-December 2025** - 4-6 weeks)
**Focus**: Analytics and engagement features

#### 🎯 Planned Enhancements

#### ~~1. **Model Performance Dashboard** (Priority: HIGH)~~ ✅ **COMPLETED**
~~**Problem**: Users don't know if predictions are actually accurate~~

~~**Implementation**:
```python
# New tab in main app
with tab_performance:
    st.write("### 📊 Model Performance Tracking")
    
    # Load betting log with results
    betting_log = pd.read_csv('data_files/betting_recommendations_log.csv')
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Bets", len(betting_log))
    with col2:
        win_rate = (betting_log['result'] == 'WIN').mean() * 100
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col3:
        roi = calculate_roi(betting_log)
        st.metric("ROI", f"{roi:.1f}%")
    with col4:
        st.metric("Units Won", f"+{betting_log['profit'].sum():.1f}")
    
    # Performance by confidence tier
    st.write("#### Performance by Confidence Level")
    confidence_performance = betting_log.groupby('confidence_tier').agg({
        'result': lambda x: (x == 'WIN').mean() * 100,
        'profit': 'sum'
    })
    st.dataframe(confidence_performance)
    
    # Week-over-week tracking
    st.write("#### Weekly Performance")
    weekly_data = betting_log.groupby('week').agg({
        'result': lambda x: (x == 'WIN').mean() * 100,
        'profit': 'sum'
    })
    st.line_chart(weekly_data)
```

**Effort**: 6-8 hours  
**Impact**: Builds trust and transparency~~

---

#### ~~1. **Notification System** (Priority: MEDIUM)~~ ✅ **COMPLETED**
**Problem**: Users miss high-value betting opportunities

**Implementation (deployed)**:
- Automatic in-app toasts for Elite (≥65%) and Strong (60–65%) opportunities. Toasts are deduplicated via `st.session_state` so the same game doesn't re-notify within a session.
- Toasts are actionable and link to per-alert pages, reachable with the query parameter `?alert=<guid>`; per-alert pages render friendly bet labels, team logos, gameday/time, and a small recommendation table.
- The app attempts to detect its public base URL from the browser and persists it to `data_files/app_config.json`; `scripts/generate_rss.py` uses that value (if present) to build `data_files/alerts_feed.xml` with working per-alert links.
- The running app exposes a sidebar `"🔁 Rebuild RSS"` button to re-generate the feed in-place and report success/errors.

**Files & locations**:
- Per-alert rendering: `predictions.py` (reads `data_files/betting_recommendations_log.csv` and responds to `?alert=` query param)
- Persisted config: `data_files/app_config.json` (contains `{"app_base_url": "https://..."}` when detected)
- RSS generator: `scripts/generate_rss.py` → output `data_files/alerts_feed.xml`

**Effort**: ~3-6 hours (implemented, testing + UI polish)  
**Impact**: Improves engagement and enables external alerting via RSS

---

#### ~~2. **Bankroll Management Tool** (Priority: MEDIUM)~~ ✅ **COMPLETED**
~~**Problem**: Users don't know how much to bet on each game~~

~~**Implementation**:
```python
# New tab for bankroll management
with tab_bankroll:
    st.write("### 💰 Bankroll Management")
    
    # User inputs
    col1, col2 = st.columns(2)
    with col1:
        bankroll = st.number_input("Current Bankroll ($)", value=1000, step=100)
    with col2:
        risk_tolerance = st.select_slider(
            "Risk Tolerance",
            options=["Conservative", "Moderate", "Aggressive"],
            value="Moderate"
        )
    
    # Kelly Criterion calculator
    risk_multipliers = {
        "Conservative": 0.25,  # 1/4 Kelly
        "Moderate": 0.5,       # 1/2 Kelly
        "Aggressive": 1.0      # Full Kelly
    }
    
    # Calculate bet sizes for each recommendation
    st.write("#### Recommended Bet Sizes")
    for _, bet in elite_bets.iterrows():
        # Kelly formula: f = (bp - q) / b
        # f = fraction of bankroll to bet
        # b = odds received (decimal - 1)
        # p = probability of winning
        # q = probability of losing (1-p)
        
        decimal_odds = american_to_decimal(bet['odds'])
        b = decimal_odds - 1
        p = bet['win_probability']
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        kelly_fraction = max(0, kelly_fraction)  # Never negative
        
        recommended_bet = bankroll * kelly_fraction * risk_multipliers[risk_tolerance]
        
        st.write(f"**{bet['matchup']}**: Bet ${recommended_bet:.2f} ({recommended_bet/bankroll*100:.1f}% of bankroll)")
```

**Effort**: 5-6 hours  
**Impact**: Helps users bet responsibly and optimally~~

---

#### 3. **Comparison Tool** (Priority: LOW)
**Problem**: Users want to compare multiple games side-by-side

**Implementation**:
```python
# Add comparison mode
with tab_compare:
    st.write("### 🔍 Game Comparison Tool")
    
    # Let users select 2-4 games to compare
    available_games = predictions_df['matchup'].unique()
    selected_games = st.multiselect(
        "Select games to compare (max 4)",
        options=available_games,
        max_selections=4
    )
    
    if len(selected_games) >= 2:
        comparison_df = predictions_df[predictions_df['matchup'].isin(selected_games)]
        
        # Show side-by-side metrics
        st.write("#### Head-to-Head Comparison")
        
        metrics_to_compare = [
            'spread_prediction', 'spread_confidence',
            'moneyline_prediction', 'moneyline_confidence',
            'total_prediction', 'over_under_confidence',
            'expected_roi'
        ]
        
        comparison_table = comparison_df.pivot_table(
            index='matchup',
            values=metrics_to_compare,
            aggfunc='first'
        )
        
        st.dataframe(comparison_table.style.highlight_max(axis=0, color='lightgreen'))
        
        # Recommendation summary
        st.write("#### Best Betting Opportunities")
        best_spread = comparison_df.loc[comparison_df['spread_confidence'].idxmax()]
        best_moneyline = comparison_df.loc[comparison_df['moneyline_confidence'].idxmax()]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Best Spread Bet", best_spread['matchup'], 
                     f"{best_spread['spread_confidence']:.1f}% confidence")
        with col2:
            st.metric("Best Moneyline Bet", best_moneyline['matchup'],
                     f"{best_moneyline['moneyline_confidence']:.1f}% confidence")
```

**Effort**: 4-5 hours  
**Impact**: Helps users make informed decisions when choosing between games

---

#### **Per-Game Detail Page** (Priority: MEDIUM)
**Problem**: Users want to inspect a single game's full context (predictions, play-by-play, team stats, and alert history) in one place.

**Goal**: Add a dedicated per-game page modeled on the alert/per-alert pages already implemented. The page should be reachable via a query parameter (e.g. `?game=<game_id>`) or a friendly route, and provide a complete, shareable view of an individual matchup.

**Key Features**:
- **Summary Header**: Team names, logos, gameday/time, venue, spread, moneyline, total line, and quick confidence badges (Elite/Strong/Good).
- **Model Predictions**: Spread, Moneyline, and Totals probabilities with calibrated probabilities and the model's recommended action(s).
- **Recent Betting History**: Any historical alerts or betting-log entries for the game (reads from `data_files/betting_recommendations_log.csv`).
- **Play-by-Play Viewer**: Paginated play-by-play table for the game with filters (quarter, down, play type) and expandable details per play.
- **Team Season Context**: Side-by-side team stats (season averages, recent 3-game form, head-to-head) using the same memory-efficient dtype patterns.
- **Edge & Expected Value**: Calculation panel showing implied odds vs model probability, expected value, and suggested bet sizing (Kelly-inspired) when applicable.
- **Shareable Alert Link**: One-click copy link to the game's per-game page; compatible with the RSS/per-alert generator so external feeds can link directly.
- **Export & Persist**: Download game-specific CSV (predictions + pbp) and optionally persist a server-side snapshot under `data_files/exports/` for auditability.

**Implementation Notes**:
- Use the same lazy-loading pattern (`@st.cache_data`) and progress spinner as other pages.
- Render the page as an independent Streamlit view that reads `st.experimental_get_query_params()` for `game` or `alert` params.
- Reuse the alert-rendering components and copy the toast/deduplication session-state behavior for in-app notifications.
- Keep the play-by-play viewer paginated and memory-efficient (views, `float32`, `Int8`), and show a warning if the dataset for a single game is unexpectedly large.

**Effort**: 6-10 hours (including UI polish and testing)

**Impact**: Improves transparency and allows analysts to deep-dive on individual matchups, increases shareability of alerts, and provides a single-source-of-truth page for support and audit.


#### **Beta 0.3 Priorities** (Choose 2-3 features)

**Recommended Priority**:
1. **Model Performance Dashboard** (MUST HAVE) - Builds trust
2. **In-App Notifications** (HIGH VALUE) - Increases engagement
3. **Bankroll Management** (UNIQUE FEATURE) - Differentiates from competitors

**Estimated Timeline**: 15-20 hours total development

---

### **Version 1.0** (Target: **January 2026** - 8-10 weeks)
**Focus**: Full production release with polish

#### Final Features Before Launch
- [ ] Complete documentation and help system
- [ ] User authentication (if needed for notifications)

## ✅ Recent Progress (Nov 25, 2025)

- **Per-Game Detail Page (Deployed)**: Implemented a `?game=<game_id>` per-game view with matchup summary, model recommendations, shareable links, and lightweight pagination for play-by-play when explicitly requested. The page avoids loading the entire play-by-play dataset by default to preserve Streamlit Cloud memory.
- **UX & Navigation Improvements**: Schedule rows now link to per-game pages using path-relative `?game=` and open in the same tab (`target="_self"`). Return navigation replaced markdown links with explicit same-tab anchors.
- **Underdog Render Logic**: Per-game header marks the underdog in bold using spread-based logic with a moneyline fallback.
- **Season-aware Schedule Matching**: Schedule→prediction matching updated to prefer candidate prediction rows with matching `season` to avoid historical collisions.
- **Download Placeholder Pattern**: Sidebar reserves lightweight placeholders for download controls and populates them after data load to avoid premature widget creation and UI flicker.

These updates complete several Beta 0.2/0.3 items earlier than planned and reduce Streamlit Cloud memory pressure while improving shareability and per-game analysis.
- [ ] Mobile-responsive design improvements
- [ ] A/B testing framework for UI improvements
- [ ] Analytics integration (PostHog, Mixpanel, or similar)
- [ ] Terms of service and responsible gambling disclaimers
- [ ] SEO optimization and landing page
- [ ] Social sharing features

---

## 📊 Decision Framework

### When to Prioritize Each Enhancement

| Enhancement | Build If... | Skip If... |
|------------|-------------|------------|
| **Loading Progress** | Users complain about "app hanging" | Load times consistently <5s |
| **Data Export** | Users ask "can I download this?" | No one mentions it after 2 weeks |
| **Cache Management** | Data freshness is critical | Data updated infrequently |
| **Performance Dashboard** | Users question accuracy | Models already trusted |
| **Notifications** | Engagement drops week-over-week | Users check app daily |
| **Bankroll Management** | Users ask about bet sizing | Users are experienced bettors |

---

## 🎯 Success Metrics by Release

### Beta 0.1
- [ ] 10-20 active users
- [ ] <5 critical bugs reported
- [ ] Average session length >5 minutes
- [ ] At least 3 pieces of actionable feedback

### Beta 0.2
- [ ] 50+ active users
- [ ] 20%+ increase in user retention
- [ ] <10 second average load time
- [ ] 80%+ of users engage with new features

### Beta 0.3
- [ ] 100+ active users
- [ ] 50%+ weekly active users
- [ ] Positive feedback on prediction accuracy
- [ ] Users placing actual bets based on recommendations

### Version 1.0
- [ ] 500+ active users
- [ ] <1% error rate
- [ ] 4+ star average rating
- [ ] Documented success stories

---

## 💡 Implementation Philosophy

### Core Principles
1. **Ship fast, iterate faster** - Weekly releases during beta
2. **Data-driven decisions** - Build what users actually want
3. **Quality over features** - Better to have 5 excellent features than 20 mediocre ones
4. **Performance first** - Every feature must load <2 seconds
5. **User trust** - Transparency about model performance and limitations

### Technical Debt Management
- Refactor after every 3 features
- Maintain <10% test coverage regression
- Document all new features in copilot-instructions.md
- Keep dependencies up-to-date monthly

---

## 📝 Notes & Lessons Learned

### From Beta 0.1 Development
- ✅ Lazy loading pattern crucial for Streamlit Cloud performance
- ✅ Git LFS essential for large datasets
- ✅ Module-level imports cause silent failures
- ✅ User-facing error messages need to be specific
- ⚠️ Watch out for data leakage in feature engineering

### Feedback Template for Beta Users
```
Thanks for testing NFL Predictions Beta 0.1! Quick questions:

1. What's the ONE feature you use most?
2. What's the ONE thing that frustrates you?
3. What feature would make you check this daily?
4. Would you recommend this to a friend? Why/why not?
5. How much would you pay for this? (Be honest!)
```

---

## 🤝 Contributing

See individual feature implementation notes above. All enhancements should:
- Follow lazy loading pattern
- Include proper error handling
- Be documented in copilot-instructions.md
- Include user-facing help text
- Load in <2 seconds

---

**Last Updated**: November 5, 2025  
**Next Review**: After Beta 0.1 user feedback (approx. November 20, 2025)

---

## Progress Update (Nov 23, 2025)

- **Data Export (COMPLETED)**: Download helpers and CSV export buttons for predictions and the betting log were implemented. The UI uses lightweight sidebar placeholders that are populated with `st.download_button` controls after data loading finishes, ensuring the sidebar renders immediately even when large datasets are being read.
- **Sidebar Placement**: Export buttons are located in the sidebar to reduce main-page clutter. Note that the sidebar may be collapsed by default — expanding it reveals the download controls.
- **Move Export Buttons**: The export buttons were moved to the sidebar and the placeholder pattern was implemented to avoid render-order issues (previously downloads sometimes didn't appear because the sidebar rendered before data loaded).
- **Smoke Test**: Added `smoke_test.py` to validate imports and lazy-loading behavior in CI without launching the full Streamlit server.
- **Memory & Stability Work**: `float32` and `Int8` dtype changes, DataFrame view usage, `@st.cache_data` lazy loaders, and pagination are in place to keep memory usage compatible with Streamlit Cloud limits (~1.5GB observed during peak loads).

### Immediate Next Priorities

1. **Automated CI** — Add a GitHub Action running `python smoke_test.py` for PRs to catch regressions early.
2. **Streamlit Staging Deploy** — Deploy to Streamlit Cloud staging and monitor memory and session behavior with real users.
3. **Server-side Export Persistence (optional)** — Save copies of user exports under `data_files/exports/` with timestamps for auditing and support.
4. **Sidebar UX** — Consider defaulting to `initial_sidebar_state='expanded'` in `predictions.py` to make exports discoverable for new users.

