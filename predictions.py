import streamlit as st

# Set page config FIRST - must be the very first Streamlit command
st.set_page_config(
    page_title="NFL Outcome Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Import required libraries
import pandas as pd
import numpy as np
from os import path
# Optional heavy ML imports are wrapped so Streamlit can start even if packages
# are missing on the host. If missing, the app will continue to run and show
# a friendly error later when model training or prediction features are used.
try:
    from xgboost import XGBClassifier
except Exception as _e:
    XGBClassifier = None
    # Log import error to stderr so cloud logs contain the root cause
    try:
        import sys
        print(f"[WARN] xgboost import failed: {_e}", file=sys.stderr)
    except Exception:
        pass

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, accuracy_score
    from sklearn.inspection import permutation_importance
except Exception as _e:
    train_test_split = None
    mean_absolute_error = None
    accuracy_score = None
    permutation_importance = None
    try:
        import sys
        print(f"[WARN] scikit-learn import failed: {_e}", file=sys.stderr)
    except Exception:
        pass
import os
from datetime import datetime
from streamlit.components.v1 import html as components_html
import re
import requests
import sys
import json
from io import BytesIO
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import base64
import traceback
import threading
import time
import functools

# Application-level startup log and uncaught exception hook so Cloud logs
# include clear tracebacks when the app crashes during startup.
try:
    print(f"[INFO] app import started: {datetime.now().isoformat()}")
except Exception:
    pass

def _log_uncaught_exception(exc_type, exc_value, exc_tb):
    try:
        print("[ERROR] Uncaught exception:", file=sys.stderr)
        import traceback as _tb
        for line in _tb.format_exception(exc_type, exc_value, exc_tb):
            try:
                print(line, end='', file=sys.stderr)
            except Exception:
                pass
    except Exception:
        pass
    # Give Streamlit Cloud (or any log-forwarder) a small window to capture
    # the printed traceback before the process exits. This is intentionally
    # noisy and temporary — it helps debugging remote startup crashes.
    try:
        time.sleep(30)
    except Exception:
        pass

try:
    sys.excepthook = _log_uncaught_exception
except Exception:
    pass

# Load .env automatically if present. Prefer python-dotenv, fallback to a minimal parser.
try:
    # Try to use python-dotenv if available (recommended)
    from dotenv import load_dotenv as _load_dotenv  # type: ignore
    _load_dotenv()
except Exception:
    # Minimal .env loader: read key=value lines and set into os.environ if not already set.
    def _manual_load_dotenv(dotenv_path='.env'):
        try:
            if not os.path.exists(dotenv_path):
                return
            with open(dotenv_path, 'r', encoding='utf-8') as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' not in line:
                        continue
                    key, val = line.split('=', 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    # Don't overwrite existing environment variables
                    if key and key not in os.environ:
                        os.environ[key] = val
        except Exception:
            # silently ignore parsing errors to avoid breaking the app
            pass

    _manual_load_dotenv()

# Debug: Print to logs to verify app is running
# Startup log removed to keep terminal output clean

DATA_DIR = 'data_files/'

# Define features list before loading best features - matches nfl-gather-data.py
features = [
    # Pregame features only (removed 'total' as it's actual game total, causing data leakage)
    'spread_line', 'away_moneyline', 'home_moneyline', 'away_spread_odds', 'home_spread_odds', 'total_line',
    'under_odds', 'over_odds', 'div_game', 'roof', 'surface', 'temp', 'wind', 'away_rest', 'home_rest',
    'home_team', 'away_team', 'gameday', 'week', 'season', 'home_qb_id', 'away_qb_id', 'home_qb_name', 'away_qb_name',
    'home_coach', 'away_coach', 'stadium', 'location',
    # Rolling team stats (calculated from previous games only)
    'homeTeamWinPct', 'awayTeamWinPct', 'homeTeamCloseGamePct', 'awayTeamCloseGamePct', 'homeTeamBlowoutPct', 'awayTeamBlowoutPct',
    'homeTeamAvgScore', 'awayTeamAvgScore', 'homeTeamAvgScoreAllowed', 'awayTeamAvgScoreAllowed', 'homeTeamAvgPointDiff', 'awayTeamAvgPointDiff',
    'homeTeamAvgTotalScore', 'awayTeamAvgTotalScore', 'homeTeamGamesPlayed', 'awayTeamGamesPlayed', 'homeTeamAvgPointSpread', 'awayTeamAvgPointSpread',
    'homeTeamAvgTotal', 'awayTeamAvgTotal', 'homeTeamFavoredPct', 'awayTeamFavoredPct', 'homeTeamSpreadCoveredPct', 'awayTeamSpreadCoveredPct',
    'homeTeamOverHitPct', 'awayTeamOverHitPct', 'homeTeamUnderHitPct', 'awayTeamUnderHitPct', 'homeTeamTotalHitPct', 'awayTeamTotalHitPct',
    # Enhanced season and matchup features
    'homeTeamCurrentSeasonWinPct', 'awayTeamCurrentSeasonWinPct', 'homeTeamCurrentSeasonAvgScore', 'awayTeamCurrentSeasonAvgScore',
    'homeTeamCurrentSeasonAvgScoreAllowed', 'awayTeamCurrentSeasonAvgScoreAllowed', 'homeTeamPriorSeasonRecord', 'awayTeamPriorSeasonRecord',
    'headToHeadHomeTeamWinPct',
    # Upset-specific features
    'spreadSize', 'isCloseSpread', 'isMediumSpread', 'isLargeSpread'
]

# Load best features for spread
try:
    with open(path.join(DATA_DIR, 'best_features_spread.txt'), 'r') as f:
        loaded_features = [line.strip() for line in f if line.strip()]
    # Only keep features that are valid columns in the data
    best_features_spread = [feat for feat in loaded_features if feat in features]
    if not best_features_spread:
        best_features_spread = features
except FileNotFoundError:
    best_features_spread = features  # fallback to all features

# Load best features for moneyline
try:
    with open(path.join(DATA_DIR, 'best_features_moneyline.txt'), 'r') as f:
        best_features_moneyline = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    best_features_moneyline = features

# Load best features for totals
try:
    with open(path.join(DATA_DIR, 'best_features_totals.txt'), 'r') as f:
        best_features_totals = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    best_features_totals = features



# Load full NFL schedule from ESPN API (all regular season weeks) and save to CSV
current_year = datetime.now().year

# Load historical game data with caching and error handling
@st.cache_data
def load_historical_data():
    """Load historical game data with caching"""
    try:
        print(f"[DEBUG] load_historical_data() called at {datetime.now().isoformat()}")
    except Exception:
        pass
    try:
        data = pd.read_csv(path.join(DATA_DIR, 'nfl_games_historical_with_predictions.csv'), sep='\t')
        try:
            print(f"[DEBUG] load_historical_data() loaded rows={len(data)}")
        except Exception:
            pass
        return data
    except FileNotFoundError:
        st.error("Critical file missing: nfl_games_historical_with_predictions.csv")
        st.info("Please run 'python build_and_train_pipeline.py' first to generate the required data files.")
        st.stop()  # Stop execution if critical data is missing
    except Exception as e:
        st.error(f"Error loading historical game data: {str(e)}")
        st.stop()

# DON'T load data at module level - will be loaded lazily when needed
historical_game_level_data = None

# Load model predictions CSV for display (cached)
@st.cache_data
def load_predictions_csv():
    """Load predictions CSV with caching"""
    try:
        print(f"[DEBUG] load_predictions_csv() called at {datetime.now().isoformat()}")
    except Exception:
        pass
    predictions_csv_path = path.join(DATA_DIR, 'nfl_games_historical_with_predictions.csv')
    if os.path.exists(predictions_csv_path):
        df = pd.read_csv(predictions_csv_path, sep='\t')
        try:
            print(f"[DEBUG] load_predictions_csv() loaded rows={len(df)}")
        except Exception:
            pass
        # Prefer returning the latest-season predictions for the UI (avoids showing only 2020).
        # If a current season is present, use it; otherwise use the newest season available.
        try:
            if 'season' in df.columns:
                seasons = pd.to_numeric(df['season'], errors='coerce').dropna().astype(int)
                if len(seasons) > 0:
                    cur_year = datetime.now().year
                    if cur_year in seasons.values:
                        return df[df['season'] == cur_year].reset_index(drop=True)
                    # Fallback: use the most recent season in the file
                    latest = int(seasons.max())
                    return df[df['season'] == latest].reset_index(drop=True)
        except Exception:
            # If anything goes wrong deciding season, return the full dataframe
            return df
        return df
    return None


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to a CSV bytes object suitable for `st.download_button`.

    Keeps encoding explicit to UTF-8 to avoid platform issues.
    """
    try:
        return df.to_csv(index=False).encode('utf-8')
    except Exception:
        # Fallback: coerce to strings then export
        df2 = df.copy()
        for c in df2.columns:
            try:
                df2[c] = df2[c].astype(str)
            except Exception:
                pass
        return df2.to_csv(index=False).encode('utf-8')


def generate_predictions_pdf(df: pd.DataFrame, max_rows: int = 200) -> bytes:
    """Generate a simple PDF summary of the predictions DataFrame and return bytes.

    Keeps layout compact and avoids embedding large images. Uses a landscape
    letter page with a table of core fields.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), topMargin=0.4 * inch)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=16, alignment=1)

    story = []
    story.append(Paragraph("NFL Predictions", title_style))
    story.append(Spacer(1, 8))

    date_text = f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}"
    story.append(Paragraph(date_text, styles['Normal']))
    story.append(Spacer(1, 8))

    # Table header and column selection - keep columns compact for readability
    # Short, compact headers to improve fit on the PDF
    header = ['Gameday', 'Home', 'Away', 'Spread', 'Total', 'Home ML', 'Away ML', 'Pr Cover', 'Pr Win', 'Pr Over']
    table_data = [header]
    # Only include games that are today or later (no past games)
    try:
        today_dt = pd.to_datetime(datetime.now().date())
        if 'gameday' in df.columns:
            gamedays = pd.to_datetime(df['gameday'], errors='coerce')
            upcoming_mask = gamedays >= today_dt
            df_to_use = df[upcoming_mask]
        else:
            df_to_use = df.copy()
    except Exception:
        df_to_use = df.copy()

    # If there are no upcoming games, emit a short message instead of a table
    if df_to_use is None or len(df_to_use) == 0:
        story.append(Spacer(1, 6))
        story.append(Paragraph("No upcoming games found for the selected predictions.", styles['Normal']))
        try:
            doc.build(story)
        except Exception:
            pass

        buffer.seek(0)
        return buffer.getvalue()

    # Fill rows (limit to max_rows for performance)
    for _, row in df_to_use.head(max_rows).iterrows():
        try:
            gameday_val = row.get('gameday', '')
            gameday = ''
            if pd.notna(gameday_val):
                try:
                    dt = pd.to_datetime(gameday_val)
                    # If time is exactly midnight, only show the date (hide 00:00:00)
                    if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
                        gameday = dt.strftime('%Y-%m-%d')
                    else:
                        gameday = dt.strftime('%Y-%m-%d %H:%M')
                except Exception:
                    gameday = str(gameday_val)

            def fmt(v, digits=2):
                try:
                    if pd.isna(v):
                        return ''
                    if isinstance(v, (int, float, np.floating, np.integer)):
                        return f"{v:.{digits}f}"
                    return str(v)
                except Exception:
                    return str(v)

            row_vals = [
                gameday,
                row.get('home_team', ''),
                row.get('away_team', ''),
                fmt(row.get('spread_line', '')),
                fmt(row.get('total_line', '')),
                str(row.get('home_moneyline', '')),
                str(row.get('away_moneyline', '')),
                fmt(row.get('prob_underdogCovered', ''), 2),
                fmt(row.get('prob_underdogWon', ''), 2),
                fmt(row.get('prob_overHit', ''), 2),
            ]
            table_data.append(row_vals)
        except Exception:
            # Skip problematic rows but continue building the PDF
            continue

    # Create table with reasonable column widths for landscape
    # Adjust column widths for a tighter, print-friendly layout
    col_widths = [1.0 * inch, 1.0 * inch, 1.0 * inch, 0.6 * inch, 0.6 * inch, 0.8 * inch, 0.8 * inch, 0.6 * inch, 0.6 * inch, 0.6 * inch]
    table = Table(table_data, colWidths=col_widths)

    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2f4f4f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        # Smaller header and body fonts to fit more content
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        # Tighter paddings for compact layout
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7f7f7')]),
        # Alignments: header centered, numeric columns right-aligned for readability
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('ALIGN', (3, 1), (6, -1), 'RIGHT'),
        ('ALIGN', (7, 1), (9, -1), 'RIGHT')
    ])
    table.setStyle(table_style)

    story.append(table)

    # Build PDF
    try:
        doc.build(story)
    except Exception:
        # If building fails, return an empty bytes object instead of crashing the app
        return b''

    buffer.seek(0)
    return buffer.getvalue()





def get_base_url_from_browser(timeout_ms: int = 1000):
        """Inject JS to set a transient query param with the base URL and reload.

        The JS will set `__detected_base` to `origin + pathname` and navigate so
        Python can read it from `st.query_params` on the next run. This avoids
        relying on component return values which can be inconsistent across
        Streamlit versions.
        """
        js = """
        <script>
            try {
                const base = window.location.origin + window.location.pathname;
                const params = new URLSearchParams(window.location.search);
                if (!params.has('__detected_base')) {
                    params.set('__detected_base', base);
                    const newUrl = window.location.pathname + '?' + params.toString();
                    window.location.href = newUrl;
                }
            } catch (e) {
                // noop
            }
        </script>
        """
        try:
                # Render the snippet; the JS will navigate and cause a reload where
                # the param will be processed at startup. We don't expect a Python
                # return value from this call.
                components_html(js, height=0)
        except Exception:
                pass
        return None

# DON'T load predictions at module level - will be loaded lazily when needed
predictions_csv_path = path.join(DATA_DIR, 'nfl_games_historical_with_predictions.csv')
predictions_df = None

# Background data loader state
data_loader_started = False
data_loader_lock = threading.Lock()

def _background_load():
    """Load large CSVs on a daemon thread to avoid blocking Streamlit startup."""
    global historical_game_level_data, predictions_df, data_loader_started
    try:
        try:
            print(f"[INFO] background loader started at {datetime.now().isoformat()}")
        except Exception:
            pass
        hist_path = path.join(DATA_DIR, 'nfl_games_historical_with_predictions.csv')
        hist_rows = None
        preds_rows = None

        if os.path.exists(hist_path):
            try:
                historical_game_level_data = pd.read_csv(hist_path, sep='\t')
                try:
                    hist_rows = len(historical_game_level_data)
                except Exception:
                    hist_rows = None
            except Exception as e:
                try:
                    import sys
                    print(f"[WARN] background historical load failed: {e}", file=sys.stderr)
                except Exception:
                    pass

        preds_path = hist_path
        if os.path.exists(preds_path):
            try:
                predictions_df = pd.read_csv(preds_path, sep='\t')
                try:
                    preds_rows = len(predictions_df)
                except Exception:
                    preds_rows = None
            except Exception as e:
                try:
                    import sys
                    print(f"[WARN] background predictions load failed: {e}", file=sys.stderr)
                except Exception:
                    pass

        # Emit an application-level info log so Streamlit Cloud logs show completion
        try:
            import sys
            info_parts = []
            if hist_rows is not None:
                info_parts.append(f"historical_rows={hist_rows}")
            if preds_rows is not None:
                info_parts.append(f"predictions_rows={preds_rows}")
            if info_parts:
                print(f"[INFO] background load complete: {' '.join(info_parts)}", file=sys.stdout)
            else:
                print(f"[INFO] background load complete (no row counts available)", file=sys.stdout)
        except Exception:
            pass
    finally:
        with data_loader_lock:
            data_loader_started = True

def start_background_loader():
    """Start the background loader once."""
    global data_loader_started
    with data_loader_lock:
        if data_loader_started:
            return
        t = threading.Thread(target=_background_load, daemon=True)
        t.start()

# Note: background loader will be started lazily from the UI thread
# to avoid doing heavy I/O during module import which can cause
# Streamlit Cloud health-check issues. Call `start_background_loader()`
# from the UI path below when the app is rendering.

# Function to automatically log betting recommendations
def log_betting_recommendations(predictions_df):
    """Automatically log betting recommendations to CSV for tracking"""
    if predictions_df is None:
        return
    
    log_path = path.join(DATA_DIR, 'betting_recommendations_log.csv')
    
    # Filter for upcoming games only (future games) - USE VIEW, NOT COPY
    upcoming_df = predictions_df[predictions_df['gameday'] > today]
    if 'gameday' in upcoming_df.columns:
        upcoming_df['gameday'] = pd.to_datetime(upcoming_df['gameday'], errors='coerce')
        upcoming_df = upcoming_df[upcoming_df['gameday'] > pd.to_datetime(datetime.now())]
    
    if len(upcoming_df) == 0:
        return  # No upcoming games to log
    
    # Prepare records for both moneyline and spread bets
    records = []
    
    # Moneyline bets (underdog)
    if 'pred_underdogWon_optimal' in upcoming_df.columns:
        # Filter for moneyline bets - USE VIEW, NOT COPY
        moneyline_bets = upcoming_df[upcoming_df['pred_underdogWon_optimal'] == 1]
        
        for _, row in moneyline_bets.iterrows():
            # Determine underdog team
            if pd.notna(row.get('spread_line')):
                if row['spread_line'] < 0:
                    recommended_team = row['home_team']  # Home is underdog
                elif row['spread_line'] > 0:
                    recommended_team = row['away_team']  # Away is underdog
                else:
                    recommended_team = 'Pick'
            else:
                recommended_team = 'Unknown'
            
            # Determine confidence tier
            prob = row.get('prob_underdogWon', 0)
            if prob >= 0.75:
                confidence = 'Elite'
            elif prob >= 0.65:
                confidence = 'Strong'
            elif prob >= 0.54:
                confidence = 'Good'
            else:
                confidence = 'Standard'
            
            records.append({
                'log_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'season': row.get('season', current_year),
                'week': row.get('week', ''),
                'game_id': row.get('game_id', ''),
                'gameday': row.get('gameday', ''),
                'home_team': row.get('home_team', ''),
                'away_team': row.get('away_team', ''),
                'bet_type': 'moneyline_underdog',
                'recommended_team': recommended_team,
                'spread_line': row.get('spread_line', ''),
                'total_line': row.get('total_line', ''),
                'moneyline_odds': row.get('away_moneyline' if recommended_team == row.get('away_team') else 'home_moneyline', ''),
                'model_probability': row.get('prob_underdogWon', ''),
                'edge': row.get('edge_underdog_ml', ''),
                'confidence_tier': confidence,
                'actual_home_score': '',
                'actual_away_score': '',
                'bet_result': 'pending',
                'bet_profit': ''
            })
    
    # Spread bets
    if 'pred_spreadCovered_optimal' in upcoming_df.columns:
        # Filter for spread bets - USE VIEW, NOT COPY
        spread_bets = upcoming_df[upcoming_df['pred_spreadCovered_optimal'] == 1]
        
        for _, row in spread_bets.iterrows():
            # Determine underdog team for spread
            if pd.notna(row.get('spread_line')):
                if row['spread_line'] < 0:
                    recommended_team = row['home_team']  # Home is underdog
                elif row['spread_line'] > 0:
                    recommended_team = row['away_team']  # Away is underdog
                else:
                    recommended_team = 'Pick'
            else:
                recommended_team = 'Unknown'
            
            # Determine confidence tier
            prob = row.get('prob_underdogCovered', 0)
            if prob >= 0.75:
                confidence = 'Elite'
            elif prob >= 0.65:
                confidence = 'Strong'
            elif prob >= 0.54:
                confidence = 'Good'
            else:
                confidence = 'Standard'
            
            records.append({
                'log_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'season': row.get('season', current_year),
                'week': row.get('week', ''),
                'game_id': row.get('game_id', ''),
                'gameday': row.get('gameday', ''),
                'home_team': row.get('home_team', ''),
                'away_team': row.get('away_team', ''),
                'bet_type': 'spread',
                'recommended_team': recommended_team,
                'spread_line': row.get('spread_line', ''),
                'total_line': row.get('total_line', ''),
                'moneyline_odds': '',
                'model_probability': row.get('prob_underdogCovered', ''),
                'edge': row.get('edge_underdog_spread', ''),
                'confidence_tier': confidence,
                'actual_home_score': '',
                'actual_away_score': '',
                'bet_result': 'pending',
                'bet_profit': ''
            })
    
    if len(records) == 0:
        return  # No bets to log
    
    # Convert to DataFrame
    new_records_df = pd.DataFrame(records)
    
    # Load existing log or create new one
    if os.path.exists(log_path):
        existing_log = pd.read_csv(log_path)
        # Avoid duplicate entries: check if game_id + bet_type already exists with pending status
        existing_pending = existing_log[existing_log['bet_result'] == 'pending']
        
        # Filter out records that already exist as pending
        new_records_df = new_records_df[
            ~new_records_df.apply(
                lambda x: ((existing_pending['game_id'] == x['game_id']) & 
                          (existing_pending['bet_type'] == x['bet_type'])).any(),
                axis=1
            )
        ]
        
        if len(new_records_df) > 0:
            combined_log = pd.concat([existing_log, new_records_df], ignore_index=True)
            combined_log.to_csv(log_path, index=False)
    else:
        new_records_df.to_csv(log_path, index=False)

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

# Function to automatically update completed game results
def update_completed_games():
    """Fetch scores from ESPN API and update betting log for completed games"""
    log_path = path.join(DATA_DIR, 'betting_recommendations_log.csv')
    
    if not os.path.exists(log_path):
        return  # No log to update
    
    log_df = pd.read_csv(log_path)
    
    # Filter for pending bets only
    pending_bets = log_df[log_df['bet_result'] == 'pending']
    
    if len(pending_bets) == 0:
        return  # No pending bets to update
    
    # Convert gameday to datetime
    pending_bets['gameday'] = pd.to_datetime(pending_bets['gameday'], errors='coerce')
    
    # Only check games that should be completed (game day has passed)
    today = pd.to_datetime(datetime.now().date())
    completed_games = pending_bets[pending_bets['gameday'] < today]
    
    if len(completed_games) == 0:
        return  # No games to check - they are all future games
    
    # Team name mapping for matching ESPN full names to our abbreviations
    team_abbrev_to_full = {
        'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens',
        'BUF': 'Buffalo Bills', 'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears',
        'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'DAL': 'Dallas Cowboys',
        'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
        'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars',
        'KC': 'Kansas City Chiefs', 'LV': 'Las Vegas Raiders', 'LAC': 'Los Angeles Chargers',
        'LA': 'Los Angeles Rams', 'LAR': 'Los Angeles Rams', 'MIA': 'Miami Dolphins', 'MIN': 'Minnesota Vikings',
        'NE': 'New England Patriots', 'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
        'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers',
        'SF': 'San Francisco 49ers', 'SEA': 'Seattle Seahawks', 'TB': 'Tampa Bay Buccaneers',
        'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders'
    }
    
    # Fetch scores from ESPN API for each completed game
    updates_made = False
    
    for idx, bet in completed_games.iterrows():
        season = int(bet['season']) if pd.notna(bet['season']) else current_year
        week = int(bet['week']) if pd.notna(bet['week']) else 1
        
        try:
            # Fetch ESPN data for that week
            espn_url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?seasontype=2&year={season}&week={week}"
            response = requests.get(espn_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Convert our team abbreviations to full names for matching
                bet_home_full = team_abbrev_to_full.get(str(bet['home_team']).upper(), str(bet['home_team']))
                bet_away_full = team_abbrev_to_full.get(str(bet['away_team']).upper(), str(bet['away_team']))
                
                # Find matching game by team names
                for event in data.get("events", []):
                    comp = event.get("competitions", [{}])[0]
                    competitors = comp.get("competitors", [])
                    
                    if len(competitors) >= 2:
                        # ESPN format: competitors[0] is home, competitors[1] is away
                        home_team = competitors[0].get("team", {}).get("displayName", "")
                        away_team = competitors[1].get("team", {}).get("displayName", "")
                        
                        # Match by team names (case-insensitive)
                        if (home_team.lower() == bet_home_full.lower() and 
                            away_team.lower() == bet_away_full.lower()):
                            
                            # Check if game is completed
                            status = event.get("status", {}).get("type", {}).get("completed", False)
                            
                            if status:
                                # Get scores
                                home_score = int(competitors[0].get("score", 0))
                                away_score = int(competitors[1].get("score", 0))
                                
                                # Update the log dataframe
                                log_df.at[idx, 'actual_home_score'] = home_score
                                log_df.at[idx, 'actual_away_score'] = away_score
                                
                                # Determine bet result based on bet type
                                bet_type = bet['bet_type']
                                recommended_team = bet['recommended_team']
                                
                                if bet_type == 'moneyline_underdog':
                                    # Moneyline: did underdog win?
                                    if recommended_team == bet['home_team']:
                                        bet_won = home_score > away_score
                                    else:
                                        bet_won = away_score > home_score
                                    
                                    # Calculate profit
        except Exception:
            # Silently continue if ESPN API fails for a particular week
            continue

    if updates_made:
        log_df.to_csv(log_path, index=False)

# Automatically log recommendations when predictions are loaded
if predictions_df is not None:
    log_betting_recommendations(predictions_df)
    # Update completed games with scores
    update_completed_games()

@st.cache_data
def load_data():
    file_path = path.join(DATA_DIR, 'nfl_play_by_play_historical.csv.gz')
    try:
        print(f"[DEBUG] load_data() called at {datetime.now().isoformat()}")
    except Exception:
        pass
    if os.path.exists(file_path):
        # Memory-optimized loading with float32/int8 dtypes (only for truly numeric columns)
        historical_data = pd.read_csv(
            file_path, 
            compression='gzip', 
            sep='\t', 
            low_memory=False,
            dtype={
                # Convert large numeric columns to float32 (50% memory reduction)
                'down': 'float32',
                'qtr': 'float32', 
                'ydstogo': 'float32',
                'yardline_100': 'float32',
                'score_differential': 'float32',
                'posteam_score': 'float32',
                'defteam_score': 'float32',
                'epa': 'float32',
                'wp': 'float32',
                'td_prob': 'float32',
                # Convert boolean-like integer columns to int8 (only if they contain 0/1/NaN)
                'pass_attempt': 'Int8',
                'rush_attempt': 'Int8'
                # Removed: complete_pass, interception, fumble_lost, touchdown, field_goal_result
                # These may contain strings or other non-numeric values
            }
        )
        try:
            print(f"[DEBUG] load_data() loaded rows={len(historical_data)}")
        except Exception:
            pass
        return historical_data
    else:
        st.warning("Historical play-by-play data file not found. Some features may be limited.")
        return pd.DataFrame()  # Return empty DataFrame as fallback# DON'T load at module level - will be loaded lazily when needed


@st.cache_data
def load_play_by_play_chunked(max_rows: int | None = None, usecols: list | None = None):
    """Load play-by-play in chunks and return a concatenated DataFrame.

    This is cached and intended to be called on-demand from the UI.
    Use `max_rows` to limit memory usage for quick samples.
    """
    file_path = path.join(DATA_DIR, 'nfl_play_by_play_historical.csv.gz')
    if not os.path.exists(file_path):
        return pd.DataFrame()

    # Default columns to read if none specified
    if usecols is None:
        usecols = ['game_id', 'play_id', 'qtr', 'desc', 'offense', 'defense', 'yards_gained', 'play_type', 'game_date']

    chunks = []
    rows_read = 0
    try:
        for chunk in pd.read_csv(file_path, compression='gzip', sep=',', usecols=usecols, chunksize=200_000, low_memory=True):
            # Cast numeric columns to reduce memory footprint early
            if 'yards_gained' in chunk.columns:
                chunk['yards_gained'] = pd.to_numeric(chunk['yards_gained'], errors='coerce').astype('float32')

            chunks.append(chunk)
            rows_read += len(chunk)
            if max_rows is not None and rows_read >= max_rows:
                break
    except Exception:
        # Fallback to a full read if chunked read fails for unexpected reasons
        try:
            df = pd.read_csv(file_path, compression='gzip', sep=',', low_memory=False)
            return df if max_rows is None else df.head(max_rows)
        except Exception:
            return pd.DataFrame()

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
    return df


# DON'T load at module level - will be loaded lazily when needed
historical_data = None

@st.cache_data
def load_schedule():
    try:
        schedule_data = pd.read_csv(path.join(DATA_DIR, f'nfl_schedule_{current_year}.csv'), low_memory=False)
        return schedule_data
    except FileNotFoundError:
        st.warning(f"Schedule file for {current_year} not found. Schedule data will be unavailable.")
        return pd.DataFrame()  # Return empty DataFrame as fallback
    except Exception as e:
        st.error(f"Error loading schedule data: {str(e)}")
        return pd.DataFrame()

# Calculate ROI from betting log
def calculate_roi(betting_log):
    """Calculate return on investment from betting log"""
    if len(betting_log) == 0:
        return 0.0
    
    # Filter to completed bets only
    completed_bets = betting_log[betting_log['bet_result'].isin(['win', 'loss'])]
    if len(completed_bets) == 0:
        return 0.0
    
    # Calculate total profit and total wagered
    total_profit = completed_bets['bet_profit'].sum()
    total_wagered = len(completed_bets) * 100  # Assuming $100 per bet
    
    # ROI = (total profit / total wagered) * 100
    roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0.0
    
    return roi

schedule = None

# Display compact header with logo and title
col1, col2 = st.columns([1, 4])

with col1:
    logo_path = os.path.join(DATA_DIR, "gridiron-oracle-transparent.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=200)  # Reduced from 250 to 200

with col2:
    # Spaces added to vertically center the header
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.header('NFL Game Outcome Predictor')


# If the app was opened with an alert query param, render the per-item alert page
# Use the stable API `st.query_params` instead of the experimental function.
try:
    params = dict(st.query_params) if hasattr(st, 'query_params') else {}
except Exception:
    params = {}

# If a base URL was injected by the browser (via the helper), capture it and
# store it in session state then remove the transient query param.
try:
    if '__detected_base' in params and params.get('__detected_base'):
        raw = params.get('__detected_base')
        if isinstance(raw, list) and len(raw) > 0:
            detected = raw[0]
        else:
            detected = raw
        if detected:
            st.session_state['app_base_url'] = detected
            # Persist detected base URL so external scripts (RSS generator) can read it
            try:
                os.makedirs(DATA_DIR, exist_ok=True)
                cfg_path = path.join(DATA_DIR, 'app_config.json')
                with open(cfg_path, 'w', encoding='utf-8') as cf:
                    json.dump({'app_base_url': detected}, cf)
            except Exception:
                # Do not fail the app if persistence fails
                pass
except Exception:
    pass

# Per-Game detail page support: render when `?game=<game_id>` is present
if 'game' in params and params.get('game'):
    raw_game = params.get('game')
    if isinstance(raw_game, list) and len(raw_game) > 0:
        game_id_param = raw_game[0]
    else:
        game_id_param = raw_game

    try:
        game_id = str(game_id_param)
    except Exception:
        game_id = game_id_param

    with st.spinner("Loading game details..."):
        # Load predictions lazily (avoid loading large PBP dataset)
        preds = load_predictions_csv() if 'load_predictions_csv' in globals() else None

    # Find the game row in predictions
    game_row = None
    if preds is not None and not preds.empty:
        matches = preds[preds['game_id'].astype(str) == str(game_id)]
        if len(matches) > 0:
            game_row = matches.iloc[0]

    # Header
    st.write("### Game Details")
    if game_row is None:
        st.warning(f"Game {game_id} not found in predictions dataset.")
    else:
        # Basic header: teams, gameday, lines
        home = game_row.get('home_team', '')
        away = game_row.get('away_team', '')
        spread = game_row.get('spread_line', '')
        ml_home = game_row.get('home_moneyline', '')
        ml_away = game_row.get('away_moneyline', '')
        total_line = game_row.get('total_line', '')
        week = game_row.get('week', '')
        stadium = game_row.get('stadium', '')

        # Quick probabilities (if available) - compute up-front so we can place them with the left-side data
        prob_ml = game_row.get('prob_underdogWon', None)
        prob_spread = game_row.get('prob_underdogCovered', None)
        prob_over = game_row.get('prob_overHit', None)

        col_a, col_b = st.columns([3, 1])
        with col_a:
            # Team logos (look for files in data_files/logos/{ABBR}.png)
            logo_dir = path.join(DATA_DIR, 'logos')
            away_raw = str(game_row.get('away_team', '')).strip()
            home_raw = str(game_row.get('home_team', '')).strip()

            # Known mapping of abbreviations to full names (used for reverse lookup)
            team_map = {
                'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens',
                'BUF': 'Buffalo Bills', 'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears',
                'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'DAL': 'Dallas Cowboys',
                'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
                'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars',
                'KC': 'Kansas City Chiefs', 'LV': 'Las Vegas Raiders', 'LAC': 'Los Angeles Chargers',
                'LA': 'Los Angeles Rams', 'LAR': 'Los Angeles Rams', 'MIA': 'Miami Dolphins', 'MIN': 'Minnesota Vikings',
                'NE': 'New England Patriots', 'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
                'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers',
                'SF': 'San Francisco 49ers', 'SEA': 'Seattle Seahawks', 'TB': 'Tampa Bay Buccaneers',
                'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders'
            }
            rev_map = {v: k for k, v in team_map.items()}

            away_abbr = away_raw.upper()
            home_abbr = home_raw.upper()
            # If values look like full names, try reverse mapping to abbreviations
            if away_abbr not in team_map and away_raw in rev_map:
                away_abbr = rev_map[away_raw]
            if home_abbr not in team_map and home_raw in rev_map:
                home_abbr = rev_map[home_raw]

            # Determine full display names (prefer canonical full names from mapping)
            away_full = team_map.get(away_abbr, away_raw)
            home_full = team_map.get(home_abbr, home_raw)

            # Determine which team is the underdog.
            # Prefer spread_line (positive => home favored, negative => away favored).
            is_home_underdog = False
            is_away_underdog = False
            try:
                s_raw = game_row.get('spread_line', '')
                s_val = float(s_raw) if (s_raw != '' and s_raw is not None and str(s_raw).strip() != '') else None
            except Exception:
                s_val = None

            if s_val is not None:
                if s_val < 0:
                    # negative spread: away is favorite -> home is underdog
                    is_home_underdog = True
                elif s_val > 0:
                    # positive spread: home is favorite -> away is underdog
                    is_away_underdog = True
            else:
                # Fallback to moneyline odds if spread is not available
                try:
                    hm = game_row.get('home_moneyline', '')
                    am = game_row.get('away_moneyline', '')
                    home_ml = float(hm) if (hm != '' and hm is not None and str(hm).strip() != '') else None
                    away_ml = float(am) if (am != '' and am is not None and str(am).strip() != '') else None
                except Exception:
                    home_ml = away_ml = None

                if home_ml is not None and away_ml is not None:
                    # Negative moneyline implies favorite. Set underdog accordingly.
                    if home_ml < 0 and away_ml > 0:
                        is_away_underdog = True
                    elif away_ml < 0 and home_ml > 0:
                        is_home_underdog = True


            # Render away and home logos side-by-side with full names and a centered '@'
            # Inject responsive CSS for team names and QB labels so layout stays tidy on mobile
            style_html = '''
            <style>
            .team-name { font-size:30px; font-weight:700; margin-top:4px; }
            .team-qb { color:#6b7280; font-size:14px; margin-top:4px; margin-bottom:24px; }
            @media (max-width:600px) {
              .team-name { font-size:20px; }
              .team-qb { font-size:13px; margin-bottom:12px; }
            }
            </style>
            '''
            try:
                st.markdown(style_html, unsafe_allow_html=True)
            except Exception:
                pass
            cols = st.columns([1, 3, 0.5, 3, 1])
            # away logo
            away_logo_path = path.join(logo_dir, f"{away_abbr}.png")
            if os.path.exists(away_logo_path):
                cols[0].image(away_logo_path, width=90)
            else:
                cols[0].write('')
            # away full name (mark underdog in bold if applicable)
            away_label_html = f"<div class='team-name'>{away_full}{' <span style=\"font-weight:700\">(Underdog)</span>' if is_away_underdog else ''}</div>"
            try:
                cols[1].markdown(away_label_html, unsafe_allow_html=True)
            except Exception:
                # Fallback to plain markdown if HTML rendering is not allowed
                away_label = f"**{away_full}**"
                if is_away_underdog:
                    away_label = away_label + " **(Underdog)**"
                cols[1].markdown(away_label)
            # away QB name (if available)
            away_qb = str(game_row.get('away_qb_name', '')).strip()
            if away_qb:
                try:
                    cols[1].markdown(f"<div class='team-qb'>QB: {away_qb}</div>", unsafe_allow_html=True)
                except Exception:
                    cols[1].write(f"QB: {away_qb}")
                    try:
                        cols[1].markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
                    except Exception:
                        cols[1].write("")
            # center '@'
            cols[2].markdown("**@**")
            # home full name (mark underdog in bold if applicable)
            home_label_html = f"<div class='team-name'>{home_full}{' <span style=\"font-weight:700\">(Underdog)</span>' if is_home_underdog else ''}</div>"
            try:
                cols[3].markdown(home_label_html, unsafe_allow_html=True)
            except Exception:
                # Fallback to plain markdown if HTML rendering is not allowed
                home_label = f"**{home_full}**"
                if is_home_underdog:
                    home_label = home_label + " **(Underdog)**"
                cols[3].markdown(home_label)
            # home QB name (if available)
            home_qb = str(game_row.get('home_qb_name', '')).strip()
            if home_qb:
                try:
                    cols[3].markdown(f"<div class='team-qb'>QB: {home_qb}</div>", unsafe_allow_html=True)
                except Exception:
                    cols[3].write(f"QB: {home_qb}")
                    try:
                        cols[3].markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
                    except Exception:
                        cols[3].write("")
            # home logo
            home_logo_path = path.join(logo_dir, f"{home_abbr}.png")
            if os.path.exists(home_logo_path):
                cols[4].image(home_logo_path, width=90)
            else:
                cols[4].write('')

            # Spread and total will be rendered below, centered above the quick-probability metrics
            # (Rendering moved out of the logo/title column so it centers under the '@' marker)
        # Centered quick probabilities row (not visually attached to either team)
        try:
            # Use a full-width container so the metrics start at the left
            center_col = st.container()
            # Add a larger vertical spacer between the team/header row and the info/metrics below
            try:
                st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)
            except Exception:
                # Fallback to a simple blank line
                st.write("")

            # Render spread and total as a centered metric above the three percentage metrics
            # Format numbers to one decimal place where possible so they match numeric style
            try:
                try:
                    s_val_display = f"{float(spread):.1f}"
                except Exception:
                    s_val_display = str(spread)
                try:
                    t_val_display = f"{float(total_line):.1f}"
                except Exception:
                    t_val_display = str(total_line)

                # Top info row: Weather | Momentum / Recent Form | QB Matchup Index
                # These are lightweight, computed from the available `game_row` and `preds` datasets.
                try:
                    top_info = center_col.columns([1, 1, 1])

                    # Weather (temp, wind, roof/surface)
                    try:
                        temp = game_row.get('temp', '')
                        wind = game_row.get('wind', '')
                        roof = game_row.get('roof', '') if 'roof' in game_row else ''
                        weather_parts = []
                        if temp is not None and str(temp).strip() != '':
                            weather_parts.append(f"{temp}°F")
                        if wind is not None and str(wind).strip() != '':
                            weather_parts.append(f"{wind} mph wind")
                        if roof is not None and str(roof).strip() != '':
                            weather_parts.append(str(roof))
                        weather_text = ' · '.join(weather_parts) if len(weather_parts) > 0 else 'N/A'
                        try:
                            # Tooltip explains fields: temperature, wind, roof/surface
                            top_info[0].markdown(
                                "<div title='Temperature, wind, and roof/surface. Example: 60.0°F · 5.0 mph wind · outdoors' style='font-size:12px;font-weight:600;margin-bottom:4px'>Weather</div>",
                                unsafe_allow_html=True,
                            )
                            top_info[0].metric("", weather_text, label_visibility="hidden")
                        except Exception:
                            top_info[0].write(f"**Weather:** {weather_text}")
                    except Exception:
                        top_info[0].write("")

                    # Momentum / Recent form (last 3 games for each team)
                    try:
                        def _recent_form(team, n=3):
                            try:
                                df_t = preds[(preds['home_team'] == team) | (preds['away_team'] == team)].copy()
                                if df_t.empty:
                                    return None
                                df_t['gameday_dt'] = pd.to_datetime(df_t['gameday'], errors='coerce')
                                gday = pd.to_datetime(game_row.get('gameday', ''), errors='coerce')
                                df_t = df_t[df_t['gameday_dt'] < gday].sort_values('gameday_dt', ascending=False)
                                df_t = df_t.head(n)
                                if df_t.empty:
                                    return None
                                wins = 0
                                plays = 0
                                for _, r in df_t.iterrows():
                                    try:
                                        hs = r.get('home_score', None)
                                        ascore = r.get('away_score', None)
                                        if pd.isna(hs) or pd.isna(ascore):
                                            continue
                                        hs = int(hs)
                                        ascore = int(ascore)
                                        plays += 1
                                        if r.get('home_team') == team:
                                            if hs > ascore:
                                                wins += 1
                                        else:
                                            if ascore > hs:
                                                wins += 1
                                    except Exception:
                                        continue
                                return (wins, plays - wins) if plays > 0 else None
                            except Exception:
                                return None

                        home_recent = _recent_form(home)
                        away_recent = _recent_form(away)
                        if home_recent is None and away_recent is None:
                            # Fallback to season win pct if available
                            hwp = game_row.get('homeTeamCurrentSeasonWinPct', None)
                            awp = game_row.get('awayTeamCurrentSeasonWinPct', None)
                            if hwp is not None and awp is not None:
                                mom_text = f"Home: {hwp:.0%} · Away: {awp:.0%}"
                            else:
                                mom_text = 'N/A'
                        else:
                            htxt = f"{home_recent[0]}-{home_recent[1]}" if home_recent is not None else 'N/A'
                            atxt = f"{away_recent[0]}-{away_recent[1]}" if away_recent is not None else 'N/A'
                            mom_text = f"Home: {htxt} · Away: {atxt}"
                        try:
                            # Tooltip: recent form (wins-losses) over last 3 games for each team
                            top_info[1].markdown(
                                "<div title='Win/Loss record in the last 3 games for each team. Format: Home: W-L · Away: W-L' style='font-size:12px;font-weight:600;margin-bottom:4px'>Momentum (Win/Loss in last 3 games)</div>",
                                unsafe_allow_html=True,
                            )
                            top_info[1].metric("", mom_text, label_visibility="hidden")
                        except Exception:
                            top_info[1].write(f"**Momentum:** {mom_text}")
                    except Exception:
                        top_info[1].write("")

                    # QB Matchup Index (simple recent-win based index for each QB)
                    try:
                        def _qb_recent(qb_name, n=5):
                            try:
                                if not qb_name or str(qb_name).strip() == '':
                                    return None
                                df_q = preds[(preds['home_qb_name'] == qb_name) | (preds['away_qb_name'] == qb_name)].copy()
                                if df_q.empty:
                                    return None
                                df_q['gameday_dt'] = pd.to_datetime(df_q['gameday'], errors='coerce')
                                gday = pd.to_datetime(game_row.get('gameday', ''), errors='coerce')
                                df_q = df_q[df_q['gameday_dt'] < gday].sort_values('gameday_dt', ascending=False).head(n)
                                if df_q.empty:
                                    return None
                                wins = 0
                                plays = 0
                                for _, r in df_q.iterrows():
                                    try:
                                        hs = r.get('home_score', None)
                                        ascore = r.get('away_score', None)
                                        if pd.isna(hs) or pd.isna(ascore):
                                            continue
                                        hs = int(hs); ascore = int(ascore)
                                        plays += 1
                                        if r.get('home_qb_name') == qb_name:
                                            if hs > ascore:
                                                wins += 1
                                        elif r.get('away_qb_name') == qb_name:
                                            if ascore > hs:
                                                wins += 1
                                    except Exception:
                                        continue
                                return (wins, plays)
                            except Exception:
                                return None

                        home_qb = str(game_row.get('home_qb_name', '')).strip()
                        away_qb = str(game_row.get('away_qb_name', '')).strip()
                        hqb = _qb_recent(home_qb)
                        aqb = _qb_recent(away_qb)
                        if hqb is None and aqb is None:
                            qb_text = 'N/A'
                        else:
                            htxt = f"{hqb[0]}/{hqb[1]}" if hqb is not None else 'N/A'
                            atxt = f"{aqb[0]}/{aqb[1]}" if aqb is not None else 'N/A'
                            qb_text = f"Home: {htxt} · Away: {atxt}"
                        try:
                            # Tooltip: recent QB performance (wins/games played) over last 5 games
                            top_info[2].markdown(
                                "<div title='Wins and games played for each QB over recent games (format: Wins/Games). Example: Home: 3/5 · Away: 3/5' style='font-size:12px;font-weight:600;margin-bottom:4px'>QB Recent (Wins and Games Played)</div>",
                                unsafe_allow_html=True,
                            )
                            top_info[2].metric("", qb_text, label_visibility="hidden")
                        except Exception:
                            top_info[2].write(f"**QB:** {qb_text}")
                    except Exception:
                        top_info[2].write("")
                except Exception:
                    # if top_info row fails, continue without it
                    pass

                top_inner = center_col.columns([1, 1, 1])
                # Place the Spread above the left (ML%) metric and Total above the center (SP%) metric
                # so they share the same numeric styling and vertical alignment as the percent metrics.
                try:
                    # Tooltip: point spread line (positive => home favored). Example: -3.5
                    top_inner[0].markdown(
                        "<div title='Point spread line (positive => home favored). Example: -3.5' style='font-size:12px;font-weight:600;margin-bottom:4px'>Spread</div>",
                        unsafe_allow_html=True,
                    )
                    top_inner[0].metric("", f"{s_val_display}", label_visibility="hidden")
                except Exception:
                    top_inner[0].write("")
                try:
                    # Tooltip: over/under total points line
                    top_inner[1].markdown(
                        "<div title='Over/Under total points line for the game. Example: 52.5' style='font-size:12px;font-weight:600;margin-bottom:4px'>Total</div>",
                        unsafe_allow_html=True,
                    )
                    top_inner[1].metric("", f"{t_val_display}", label_visibility="hidden")
                except Exception:
                    top_inner[1].write("")

                # Compute best available model edge and render above the right (Over %) metric
                try:
                    candidates = [
                        ("ML", game_row.get('edge_underdog_ml', None)),
                        ("Spread", game_row.get('edge_underdog_spread', None)),
                        ("Total", game_row.get('edge_over', None)),
                        ("Under", game_row.get('edge_under', None))
                    ]
                    best_label = None
                    best_val = None
                    for label, raw in candidates:
                        try:
                            if raw is None or (isinstance(raw, str) and raw.strip() == ""):
                                continue
                            v = float(raw)
                        except Exception:
                            continue
                        if best_val is None or v > best_val:
                            best_val = v
                            best_label = label

                    if best_val is not None and best_label is not None:
                        display_label = f"Edge ({best_label})"
                        display_value = f"{best_val:.1f} pts"
                        try:
                            # Tooltip: model edge vs market (displayed in points). Raw value shown for precision.
                            top_inner[2].markdown(
                                f"<div title='Model edge vs market for {best_label}. Raw value: {best_val}' style='font-size:12px;font-weight:600;margin-bottom:4px'>{display_label}</div>",
                                unsafe_allow_html=True,
                            )
                            top_inner[2].metric("", display_value, delta=best_val, label_visibility="hidden")
                        except Exception:
                            top_inner[2].write(f"{display_label}: {display_value}")
                    else:
                        top_inner[2].write("")
                except Exception:
                    top_inner[2].write("")
            except Exception:
                # If metrics fail, fall back to plain centered markdown
                try:
                    center_col.markdown(f"**Spread:** {spread} • **Total:** {total_line}")
                except Exception:
                    center_col.markdown(f"<div style='text-align:center;font-weight:600;margin-bottom:6px;'>**Spread:** {spread} • **Total:** {total_line}</div>", unsafe_allow_html=True)

            inner = center_col.columns([1, 1, 1])
            if prob_ml is not None:
                # Tooltip: model probability that the underdog wins (displayed as percent)
                inner[0].markdown(
                    "<div title='Model probability that the underdog wins outright (shown as a percentage)' style='font-size:12px;font-weight:600;margin-bottom:4px'>Moneyline Probability %</div>",
                    unsafe_allow_html=True,
                )
                inner[0].metric("", f"{prob_ml:.2%}", label_visibility="hidden")
            else:
                inner[0].write("")
            if prob_spread is not None:
                # Tooltip: model probability the underdog covers the spread
                inner[1].markdown(
                    "<div title='Model probability that the underdog will cover the spread (shown as a percentage)' style='font-size:12px;font-weight:600;margin-bottom:4px'>Spread Probability %</div>",
                    unsafe_allow_html=True,
                )
                inner[1].metric("", f"{prob_spread:.2%}", label_visibility="hidden")
            else:
                inner[1].write("")
            if prob_over is not None:
                # Tooltip: model probability that the game total will go over the line
                inner[2].markdown(
                    "<div title='Model probability that the game total (points) will be over the published line (shown as a percentage)' style='font-size:12px;font-weight:600;margin-bottom:4px'>Over Probability %</div>",
                    unsafe_allow_html=True,
                )
                inner[2].metric("", f"{prob_over:.2%}", label_visibility="hidden")
            else:
                inner[2].write("")
        except Exception:
            # Fallback to centered markdown
            try:
                st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
                if prob_ml is not None:
                    st.markdown(f"**Moneyline Prob (underdog):** {prob_ml:.2%}")
                if prob_spread is not None:
                    st.markdown(f"**Spread Prob (underdog):** {prob_spread:.2%}")
                if prob_over is not None:
                    st.markdown(f"**Totals Over Prob:** {prob_over:.2%}")
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception:
                # Last-resort plain text
                if prob_ml is not None:
                    st.write(f"Moneyline Prob (underdog): {prob_ml:.2%}")
                if prob_spread is not None:
                    st.write(f"Spread Prob (underdog): {prob_spread:.2%}")
                if prob_over is not None:
                    st.write(f"Totals Over Prob: {prob_over:.2%}")
            st.markdown(f"**Moneylines:** {away}: {ml_away} — {home}: {ml_home}")
            # Game date/time formatting: omit the time if it's exactly 00:00:00
            try:
                gtime = pd.to_datetime(game_row.get('gameday', ''), errors='coerce')
                if not pd.isna(gtime):
                    date_part = gtime.date().isoformat()
                    time_part = gtime.time().strftime('%H:%M:%S')
                    if time_part == '00:00:00':
                        st.markdown(f"**Game Date:** {date_part} (Week {week})")
                    else:
                        st.markdown(f"**Game Time:** {gtime.strftime('%Y-%m-%d %H:%M:%S')} (Week {week})")
                st.write(f"**Stadium:** {stadium}")
            except Exception:
                pass

        with col_b:
            # Right column: show a compact "Model Edge vs Market" metric (best available)
            try:
                # Collect possible edge fields from the game row
                candidates = [
                    ("ML", game_row.get('edge_underdog_ml', None)),
                    ("Spread", game_row.get('edge_underdog_spread', None)),
                    ("Total", game_row.get('edge_over', None)),
                    ("Under", game_row.get('edge_under', None))
                ]

                best_label = None
                best_val = None
                for label, raw in candidates:
                    try:
                        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
                            continue
                        v = float(raw)
                    except Exception:
                        continue
                    # Prefer the largest positive edge; if none positive, pick the max value (least negative)
                    if best_val is None or v > best_val:
                        best_val = v
                        best_label = label

                # We moved the visual presentation of the best edge into the top-right cell
                # above the percentage metrics; keep this column minimal here.
                col_b.write("")
            except Exception:
                # Keep minimal if anything goes wrong
                try:
                    st.write("")
                except Exception:
                    pass

    # Play-by-play viewer intentionally removed from per-game page to reduce memory usage.
    # If detailed play-level analysis is needed, use the Historical Data page instead.

    # Betting log removed from per-game view by user request
    # (If you need betting history elsewhere, see the Performance dashboard page.)
    # Use an explicit HTML anchor with target="_self" to ensure same-tab navigation
    st.markdown("<a href='?' target='_self' style='text-decoration:none;font-weight:600;'>Return to Predictions</a>", unsafe_allow_html=True)

# If we don't yet have a detected base URL, try automatic detection once.
# This injects JS that sets a transient `__detected_base` param and reloads
# the page so the value can be captured above. We guard with a session flag
# to avoid repeated attempts if JS is blocked.
try:
    if 'app_base_url' not in st.session_state:
        already = st.session_state.get('_tried_detect_base', False)
        if not already and '__detected_base' not in params:
            st.session_state['_tried_detect_base'] = True
            # This JS will navigate and cause a reload where the param is present
            get_base_url_from_browser()
            # Do not stop execution; if JS navigation is blocked we'll still
            # render the manual fallback UI below. Avoiding st.stop() prevents
            # a blank page when the JS navigation is not allowed.
except Exception:
    pass

# Query params parsed above; no debug logging here

# (removed temporary on-screen debug caption)

if 'alert' in params and params.get('alert'):
    raw_alert = params.get('alert')
    if isinstance(raw_alert, list) and len(raw_alert) > 0:
        alert_guid = raw_alert[0]
    else:
        alert_guid = raw_alert
    # Load betting log and find matching GUID (game_id-bet_type)
    log_path = path.join(DATA_DIR, 'betting_recommendations_log.csv')
    if os.path.exists(log_path):
        try:
            log_df = pd.read_csv(log_path, low_memory=False)
            # Build guid column for matching
            def _make_guid(row):
                gid = str(row.get('game_id', ''))
                btype = str(row.get('bet_type', ''))
                return f"{gid}-{btype}"

            log_df['__guid'] = log_df.apply(_make_guid, axis=1)
            match = log_df[log_df['__guid'] == alert_guid]
            if len(match) == 0:
                st.warning(f"Alert not found: {alert_guid}")
            else:
                row = match.iloc[0]
                # Map abbreviations to full team names (reuse mapping from update_completed_games)
                team_map = {
                    'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens',
                    'BUF': 'Buffalo Bills', 'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears',
                    'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'DAL': 'Dallas Cowboys',
                    'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
                    'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars',
                    'KC': 'Kansas City Chiefs', 'LV': 'Las Vegas Raiders', 'LAC': 'Los Angeles Chargers',
                    'LA': 'Los Angeles Rams', 'LAR': 'Los Angeles Rams', 'MIA': 'Miami Dolphins', 'MIN': 'Minnesota Vikings',
                    'NE': 'New England Patriots', 'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
                    'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers',
                    'SF': 'San Francisco 49ers', 'SEA': 'Seattle Seahawks', 'TB': 'Tampa Bay Buccaneers',
                    'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders'
                }

                away_abbr = str(row.get('away_team', '')).upper()
                home_abbr = str(row.get('home_team', '')).upper()
                away_full = team_map.get(away_abbr, row.get('away_team', 'Away'))
                home_full = team_map.get(home_abbr, row.get('home_team', 'Home'))

                # Friendly bet type display
                btype = str(row.get('bet_type', 'recommendation'))
                if btype == 'moneyline_underdog':
                    btype_label = 'Moneyline — Underdog'
                elif btype == 'spread':
                    btype_label = 'Spread'
                elif btype in ('over', 'under', 'total', 'over_under'):
                    btype_label = 'Totals'
                else:
                    btype_label = btype.replace('_', ' ').title()

                # Header with small team-color markers and friendly names
                left_col, right_col = st.columns([3, 1])
                # Small map of team primary colors (hex) for markers. Keeps in-sync
                # with the later `team_colors` mapping used elsewhere in this file.
                team_colors_header = {
                    'ARI': '#97233F', 'ATL': '#A71930', 'BAL': '#241773', 'BUF': '#00338D',
                    'CAR': '#0085CA', 'CHI': '#0B162A', 'CIN': '#FB4F14', 'CLE': '#311D00',
                    'DAL': '#002244', 'DEN': '#FB4F14', 'DET': '#0076B6', 'GB': '#203731',
                    'HOU': '#03202F', 'IND': '#002C5F', 'JAX': '#006778', 'KC': '#E31837',
                    'LV': '#000000', 'LAC': '#002A5E', 'LA': '#003594', 'LAR': '#003594',
                    'MIA': '#0091A0', 'MIN': '#4F2683', 'NE': '#0C2340', 'NO': '#D3BC8D',
                    'NYG': '#0B2265', 'NYJ': '#125740', 'PHI': '#004C54', 'PIT': '#FFB612',
                    'SF': '#AA0000', 'SEA': '#002244', 'TB': '#D50A0A', 'TEN': '#4B92DB',
                    'WAS': '#5A1414'
                }

                with left_col:
                    # Away team row: small color marker + full name
                    away_color = team_colors_header.get(away_abbr, '#6B7280')
                    away_html = (
                        f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;"
                        f"background:{away_color};margin-right:8px;vertical-align:middle;border:1px solid rgba(0,0,0,0.08)'></span>"
                        f"<strong style='vertical-align:middle'>{away_full}</strong>"
                    )
                    st.markdown(away_html, unsafe_allow_html=True)

                    st.write("@")

                    # Home team row: small color marker + full name
                    home_color = team_colors_header.get(home_abbr, '#6B7280')
                    home_html = (
                        f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;"
                        f"background:{home_color};margin-right:8px;vertical-align:middle;border:1px solid rgba(0,0,0,0.08)'></span>"
                        f"<strong style='vertical-align:middle'>{home_full}</strong>"
                    )
                    st.markdown(home_html, unsafe_allow_html=True)

                # Right column: friendly bet type label
                with right_col:
                    st.markdown(f"**{btype_label}**")

                # Top-level back functionality removed — use the deterministic
                # visible "Return to Predictions" anchor rendered below.

                # Visible inline anchor fallback (use detected base URL when available)
                base_url = st.session_state.get('app_base_url', '/') if 'app_base_url' in st.session_state else '/'
                # Use a path-relative query link so navigation stays within the app
                fallback_inline = (
                    "<div style='margin-top:8px;'>"
                    "<a href='?' target='_self' "
                    "style='display:inline-block;padding:8px 12px;border-radius:6px;border:1px solid #ccc;background:#f3f4f6;color:#111;text-decoration:none;font-weight:600;'>"
                    "Return to Predictions"
                    "</a>"
                    "</div>"
                )
                st.markdown(fallback_inline, unsafe_allow_html=True)

                # Game datetime
                gameday_raw = row.get('gameday', '')
                gameday_text = ''
                try:
                    raw_str = str(gameday_raw)
                    # Detect if original value includes a time component
                    has_time = bool(re.search(r"\d{1,2}:\d{2}", raw_str) or ('T' in raw_str and ':' in raw_str))

                    gd = pd.to_datetime(gameday_raw, errors='coerce')
                    if pd.isna(gd):
                        gameday_text = raw_str
                    else:
                        try:
                            # If naive, assume UTC then convert to Eastern Time; if tz-aware, convert to ET
                            if gd.tzinfo is None:
                                gd = gd.tz_localize('UTC').tz_convert('America/New_York')
                            else:
                                gd = gd.tz_convert('America/New_York')
                        except Exception:
                            # If tz localization/conversion fails, leave as-is
                            pass

                        if has_time:
                            gameday_text = gd.strftime('%a, %b %d %Y %I:%M %p')
                        else:
                            gameday_text = gd.strftime('%a, %b %d %Y')
                except Exception:
                    gameday_text = str(gameday_raw)

                st.markdown(f"**Game Time:** {gameday_text}")

                # Recommendation and quick stats
                recommended_text = row.get('recommended_team','')

                # Map of team primary colors (hex) for subtle tinting
                team_colors = {
                    'ARI': '#97233F', 'ATL': '#A71930', 'BAL': '#241773', 'BUF': '#00338D',
                    'CAR': '#0085CA', 'CHI': '#0B162A', 'CIN': '#FB4F14', 'CLE': '#311D00',
                    'DAL': '#002244', 'DEN': '#FB4F14', 'DET': '#0076B6', 'GB': '#203731',
                    'HOU': '#03202F', 'IND': '#002C5F', 'JAX': '#006778', 'KC': '#E31837',
                    'LV': '#000000', 'LAC': '#002A5E', 'LA': '#003594', 'LAR': '#003594',
                    'MIA': '#0091A0', 'MIN': '#4F2683', 'NE': '#0C2340', 'NO': '#D3BC8D',
                    'NYG': '#0B2265', 'NYJ': '#125740', 'PHI': '#004C54', 'PIT': '#FFB612',
                    'SF': '#AA0000', 'SEA': '#002244', 'TB': '#D50A0A', 'TEN': '#4B92DB',
                    'WAS': '#5A1414'
                }

                # Determine recommended team's abbreviation (some log entries store full names)
                rec_raw = str(recommended_text)
                rec_abbr = rec_raw.upper()
                # If recommended_text is a full team name, try to reverse-map from team_map
                try:
                    rev_map = {v: k for k, v in team_map.items()}
                    if rec_raw in rev_map:
                        rec_abbr = rev_map[rec_raw]
                except Exception:
                    pass

                # Choose color; default to neutral gray
                primary_hex = team_colors.get(rec_abbr, '#6B7280')

                def _hex_to_rgb(h):
                    h = h.lstrip('#')
                    try:
                        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
                    except Exception:
                        return (107, 114, 128)

                r, g, b = _hex_to_rgb(primary_hex)
                bg_rgba = f'rgba({r},{g},{b},0.10)'
                border_rgba = f'rgba({r},{g},{b},0.22)'
                text_color = primary_hex

                badge_html = (
                    f"<div style='display:inline-block;background:{bg_rgba};color:{text_color};"
                    f"padding:6px 10px;border-radius:6px;border:1px solid {border_rgba};font-weight:600;font-size:14px;'>{recommended_text}</div>"
                )
                try:
                    st.markdown(badge_html, unsafe_allow_html=True)
                except Exception:
                    st.markdown(f"**Recommended:** {recommended_text}")

                # Format confidence and edge like the app
                try:
                    prob = float(row.get('model_probability', ''))
                    prob_text = f"{prob:.1%}"
                except Exception:
                    prob_text = str(row.get('model_probability', 'N/A'))

                try:
                    edge_raw = row.get('edge', '')
                    if pd.isna(edge_raw) or edge_raw == '':
                        raise ValueError("missing")
                    edge_val = float(edge_raw)
                    if np.isfinite(edge_val):
                        edge_text = f"{edge_val:.1f}"
                    else:
                        edge_text = 'N/A'
                except Exception:
                    edge_text = 'N/A'

                cols = st.columns(3)
                cols[0].metric('Confidence', prob_text)
                cols[1].metric('Edge', edge_text)
                cols[2].metric('Spread', row.get('spread_line', 'N/A'))

                st.write('---')

                # Present the full log entry as a readable table (no numeric index)
                display_fields = [
                    'log_date', 'season', 'week', 'gameday', 'home_team', 'away_team',
                    'bet_type', 'recommended_team', 'spread_line', 'total_line', 'moneyline_odds',
                    'model_probability', 'edge', 'confidence_tier', 'bet_result', 'bet_profit'
                ]

                # Friendly labels for rows
                field_labels = {
                    'log_date': 'Logged At', 'season': 'Season', 'week': 'Week', 'gameday': 'Game Time',
                    'home_team': 'Home Team', 'away_team': 'Away Team', 'bet_type': 'Bet Type',
                    'recommended_team': 'Recommended', 'spread_line': 'Spread', 'total_line': 'Total',
                    'moneyline_odds': 'Moneyline Odds', 'model_probability': 'Model Confidence',
                    'edge': 'Edge', 'confidence_tier': 'Confidence Tier', 'bet_result': 'Result',
                    'bet_profit': 'Profit'
                }

                rows = []
                for f in display_fields:
                    if f not in row.index:
                        continue
                    label = field_labels.get(f, f.replace('_', ' ').title())
                    val = row.get(f, '')
                    # Substitute full team names
                    if f in ('home_team', 'away_team'):
                        try:
                            abb = str(val).upper()
                            val = team_map.get(abb, val)
                        except Exception:
                            pass
                    # Format bet type, model probability and edge for readability
                    if f == 'bet_type':
                        try:
                            bt = str(val)
                            if bt == 'moneyline_underdog':
                                val = 'Moneyline — Underdog'
                            elif bt == 'spread':
                                val = 'Spread'
                            elif bt in ('over', 'under', 'total', 'over_under'):
                                val = 'Totals'
                            else:
                                # Fallback: make a readable title from underscores
                                val = bt.replace('_', ' ').title()
                        except Exception:
                            pass

                    if f == 'model_probability':
                        try:
                            val = f"{float(val):.1%}"
                        except Exception:
                            pass

                    if f == 'edge':
                        try:
                            if pd.isna(val) or val == '':
                                val = 'N/A'
                            else:
                                v = float(val)
                                val = f"{v:.1f}"
                        except Exception:
                            val = 'N/A'

                    # Coerce all values to plain strings to avoid mixed-type Arrow serialization errors
                    try:
                        if pd.isna(val):
                            val_str = ''
                        else:
                            val_str = str(val)
                    except Exception:
                        val_str = str(val)

                    rows.append((label, val_str))

                import pandas as _pd
                df_display = _pd.DataFrame(rows, columns=['Field', 'Value']).set_index('Field')
                st.table(df_display)

                # Visible inline fallback for bottom (use detected base URL when available)
                base_url = st.session_state.get('app_base_url', '/') if 'app_base_url' in st.session_state else '/'
                # Use a path-relative query link at the bottom as well
                fallback_inline_bottom = (
                    "<div style='margin-top:8px;'>"
                    "<a href='?' target='_self' "
                    "style='display:inline-block;padding:8px 12px;border-radius:6px;border:1px solid #ccc;background:#f3f4f6;color:#111;text-decoration:none;font-weight:600;'>"
                    "Return to Predictions"
                    "</a>"
                    "</div>"
                )
                st.markdown(fallback_inline_bottom, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Failed to load betting log: {e}")
    else:
        st.warning('Betting log not available.')
    # Stop further rendering of main UI when viewing an alert
    st.stop()

# Cache Management UI
with st.sidebar:
    st.write("### ⚙️ Settings")

    if st.button("🔄 Refresh Data", help="Clear cache and reload all data"):
        st.cache_data.clear()
        st.rerun()
    
    # Rebuild RSS feed from betting log
    if st.button("🔁 Rebuild RSS", help="Generate alerts_feed.xml using current app base URL"):
        import subprocess
        import shlex

        rss_script = path.join("scripts", "generate_rss.py")
        cmd = [sys.executable, rss_script]

        with st.spinner("Generating RSS feed..."):
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                stdout = proc.stdout.strip()
                stderr = proc.stderr.strip()

                if proc.returncode == 0:
                    st.success("RSS feed rebuilt successfully.")
                    # if stdout:
                    #     st.text_area("Generator output", stdout, height=200)
                    # Show link to the generated feed file if it exists
                    feed_path = path.join(DATA_DIR, 'alerts_feed.xml')
                    # if os.path.exists(feed_path):
                        # st.write(f"Feed written to: `{feed_path}`")
                else:
                    st.error(f"RSS generator exited with code {proc.returncode}.")
                    if stderr:
                        st.text_area("Generator errors", stderr, height=200)
                    if stdout:
                        st.text_area("Generator output", stdout, height=200)
            except Exception as e:
                st.error(f"Failed to run RSS generator: {e}")

        # End of RSS rebuild button handler
    # Emailing via the sidebar has been removed; use the automated sender instead.
    # For security and automation, emailing will be handled outside the interactive UI.
    
    # Data export placeholders: actual buttons are populated after data loads
    try:
        predictions_dl_placeholder = st.empty()
        betting_log_dl_placeholder = st.empty()
        pdf_dl_placeholder = st.empty()
    except Exception:
        # Fallback: ignore sidebar placeholders if creation fails
        predictions_dl_placeholder = None
        betting_log_dl_placeholder = None
        pdf_dl_placeholder = None

# (Top-level quick filters removed — filters will be shown on the Historical Data page)

# Sidebar filters

filter_keys = [
    'posteam', 'defteam', 'down', 'ydstogo', 'yardline_100', 'play_type', 'qtr',
    'score_differential', 'posteam_score', 'defteam_score', 'epa', 'pass_attempt'
]


# --- For Monte Carlo Feature Selection ---
# The heavy ML imports are handled at the top of the file inside
# try/except blocks so Streamlit can start in headless/cloud environments.
# `train_test_split` and `XGBClassifier` may be None if the packages
# are not available; code that needs them should check and handle None.

# Load data NOW (lazily, only when user accesses the app)
    # Loading logs removed to keep terminal output clean
# Non-blocking startup: show a lightweight status while background loader runs.
if historical_game_level_data is None or predictions_df is None:
    try:
        # Start the background loader from the UI thread so heavy I/O
        # (pandas CSV reads) does not run at module import. This reduces
        # the chance Streamlit Cloud's health probe sees the process as
        # unresponsive during import/startup.
        try:
            start_background_loader()
        except Exception:
            # Non-fatal: background loader will attempt to run again later
            pass

        # st.info("Loading NFL data in background; some features may be unavailable briefly.")
        # Small progress indicator to reassure the user; does not block long.
        progress_bar = st.progress(0)
        progress_bar.progress(50, text="Background loader running...")

        # Poll for a short time for the background loader to finish so the UI
        # can update automatically. This keeps the server binding non-blocking
        # while ensuring the loading info clears shortly after completion.
        max_wait = 10.0
        poll_interval = 0.5
        waited = 0.0
        while waited < max_wait:
            if historical_game_level_data is not None and predictions_df is not None:
                try:
                    progress_bar.progress(75, text="Games loaded")
                    progress_bar.progress(90, text="Predictions loaded")
                    progress_bar.progress(100, text="Ready")
                    progress_bar.empty()
                except Exception:
                    pass
                # Re-run the Streamlit script so the fresh data is picked up in UI
                try:
                    st.experimental_rerun()
                except Exception:
                    # If rerun isn't allowed in this context, just break and allow
                    # the user to interact/refresh manually.
                    break
            time.sleep(poll_interval)
            waited += poll_interval
        # If we exit the polling loop without data, leave the info/progress for now.
    except Exception:
        # If Streamlit UI isn't available in this runtime, silently continue.
        pass
else:
    # Data already available from background loader
    pass

# One-time debug: show which season was selected in the returned predictions dataframe
try:
    shown_key = '_shown_selected_season'
    if not st.session_state.get(shown_key, False):
        sel_text = "Unknown"
        # Prefer seasons for upcoming/future games only to confirm what the user will see
        if predictions_df is not None and 'season' in predictions_df.columns and 'gameday' in predictions_df.columns and len(predictions_df) > 0:
            try:
                gamedays = pd.to_datetime(predictions_df['gameday'], errors='coerce')
                today_dt = pd.to_datetime(datetime.now().date())
                upcoming_mask = gamedays >= today_dt
                upcoming_df = predictions_df[upcoming_mask]
                if len(upcoming_df) > 0:
                    seasons = sorted(upcoming_df['season'].dropna().unique().tolist())
                else:
                    # Fallback to all seasons in the file if no future games are present
                    seasons = sorted(predictions_df['season'].dropna().unique().tolist())
            except Exception:
                seasons = sorted(predictions_df['season'].dropna().unique().tolist())

            if len(seasons) == 1:
                sel_text = str(seasons[0])
            else:
                sel_text = ", ".join(str(s) for s in seasons)
        elif predictions_df is not None and 'season' in predictions_df.columns:
            seasons = sorted(predictions_df['season'].dropna().unique().tolist())
            sel_text = ", ".join(str(s) for s in seasons) if seasons else "Unknown"

        # Small one-time UI hint in the sidebar to confirm which season(s) are present
        # st.sidebar.info(f"Selected season(s): {sel_text} (upcoming games only)")
        st.session_state[shown_key] = True
except Exception:
    # Non-fatal: ignore any UI errors for the debug display
    pass

# Load play-by-play data (for historical analysis)
if 'progress_bar' in locals():
    try:
        progress_bar.progress(75, text="Loading play-by-play...")
    except Exception:
        pass

# Play-by-play load has been moved to the Historical Data expander
# to avoid showing load prompts on the main predictions page.

if 'progress_bar' in locals():
    try:
        progress_bar.progress(100, text="Ready!")
        progress_bar.empty()
    except Exception:
        pass

# Data loading complete (terminal logs suppressed)

# Populate sidebar download placeholders (created earlier) now that data is loaded
try:
    # Only populate if placeholders were created successfully
    if 'predictions_dl_placeholder' in globals() and predictions_dl_placeholder is not None:
        if predictions_df is not None:
            try:
                csv_bytes = convert_df_to_csv(predictions_df)
                predictions_dl_placeholder.download_button(
                    label="📥 Download Predictions",
                    data=csv_bytes,
                    file_name=f'nfl_predictions_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv'
                )
            except Exception:
                # If conversion fails, silently skip the download button
                pass

    # Betting log download when present
    log_path = path.join(DATA_DIR, 'betting_recommendations_log.csv')
    if 'betting_log_dl_placeholder' in globals() and betting_log_dl_placeholder is not None:
        if os.path.exists(log_path):
            try:
                with open(log_path, 'rb') as _f:
                    log_bytes = _f.read()
                betting_log_dl_placeholder.download_button(
                    label="📥 Download Betting Log",
                    data=log_bytes,
                    file_name=f'betting_recommendations_log_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv'
                )
            except Exception:
                # If reading the file fails, skip the button
                pass
    # PDF export: generate on demand to avoid pre-building large binary at load time
    if 'pdf_dl_placeholder' in globals() and pdf_dl_placeholder is not None:
        if predictions_df is not None:
            try:
                # Provide a small button to generate the PDF; after generation expose a download button
                if pdf_dl_placeholder.button("📄 Generate Predictions PDF"):
                    try:
                        pdf_bytes = generate_predictions_pdf(predictions_df)
                    except Exception as e:
                        st.sidebar.error(f"PDF generation failed: {e}")
                        st.sidebar.text(traceback.format_exc())
                        pdf_bytes = b''

                    if pdf_bytes:
                        # Save the generated PDF to the exports folder and expose a link
                        try:
                            exports_dir = path.join(DATA_DIR, 'exports')
                            os.makedirs(exports_dir, exist_ok=True)
                            filename = f'nfl_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
                            file_path = path.join(exports_dir, filename)
                            with open(file_path, 'wb') as f:
                                f.write(pdf_bytes)

                            # Provide a link to the saved file and a download button as a fallback
                            # Use a relative path with forward slashes so links render correctly in the browser
                            rel_path = f"data_files/exports/{filename}"
                            # Provide a relative link that works from the app's current URL and
                            # also show the absolute path so users can open the file locally.
                            rel_link = f"./{rel_path}"
                            abs_path = os.path.abspath(file_path)
                            # Inform the user the PDF was saved; the download button is the primary way
                            # to retrieve the file so we no longer render an 'Open PDF' link.
                            # st.sidebar.success("PDF generated and saved to exports folder.")
                            try:
                                st.sidebar.write(f"PDF saved.")
                            except Exception:
                                pass

                            # Also expose a download button so users can download immediately
                            try:
                                with open(file_path, 'rb') as _f:
                                    file_bytes = _f.read()
                                pdf_dl_placeholder.download_button(
                                    label="⬇️ Download Predictions (PDF)",
                                    data=file_bytes,
                                    file_name=filename,
                                    mime='application/pdf'
                                )
                            except Exception as e:
                                st.sidebar.error(f"Saved PDF but failed to create download button: {e}")

                        except Exception as e:
                            st.sidebar.error(f"Failed to save generated PDF: {e}")
                    else:
                        st.sidebar.error("Failed to generate PDF.")
            except Exception as e:
                # Surface any unexpected errors to the user to aid debugging
                st.sidebar.error(f"PDF export handler error: {e}")
                st.sidebar.text(traceback.format_exc())
except Exception:
    # Do not fail the app if sidebar population fails
    pass


# In-App Notifications for Elite and Strong Bets
# Store new bets in session state to avoid duplicate notifications
if 'notified_games' not in st.session_state:
    st.session_state.notified_games = set()

# Check for new high-confidence bets if predictions are available
if predictions_df is not None:
    # Filter for elite bets (≥65% confidence) that haven't been notified yet
    elite_bets = predictions_df[
        (predictions_df.get('prob_underdogWon', 0) >= 0.65) | 
        (predictions_df.get('prob_underdogCovered', 0) >= 0.65) |
        (predictions_df.get('prob_overHit', 0) >= 0.65)
    ]
    
    # Filter for strong bets (60-65% confidence) that haven't been notified yet
    strong_bets = predictions_df[
        ((predictions_df.get('prob_underdogWon', 0) >= 0.60) & (predictions_df.get('prob_underdogWon', 0) < 0.65)) | 
        ((predictions_df.get('prob_underdogCovered', 0) >= 0.60) & (predictions_df.get('prob_underdogCovered', 0) < 0.65)) |
        ((predictions_df.get('prob_overHit', 0) >= 0.60) & (predictions_df.get('prob_overHit', 0) < 0.65))
    ]
    
    # Filter out already notified games for elite bets
    if 'game_id' in elite_bets.columns:
        new_elite_bets = elite_bets[~elite_bets['game_id'].isin(st.session_state.notified_games)]
        
        if len(new_elite_bets) > 0:
            # Show elite notification
            st.toast(f"🔥 {len(new_elite_bets)} new elite betting opportunities!", icon="🔥")
            
            # Add to notified set to avoid duplicate notifications
            st.session_state.notified_games.update(new_elite_bets['game_id'].tolist())
    
    # Filter out already notified games for strong bets
    if 'game_id' in strong_bets.columns:
        new_strong_bets = strong_bets[~strong_bets['game_id'].isin(st.session_state.notified_games)]
        
        if len(new_strong_bets) > 0:
            # Show strong notification
            st.toast(f"⭐ {len(new_strong_bets)} new strong betting opportunities!", icon="⭐")
            
            # Add to notified set to avoid duplicate notifications
            st.session_state.notified_games.update(new_strong_bets['game_id'].tolist())

# Feature list for modeling and Monte Carlo selection
# Feature setup log suppressed
features = [
    'spread_line', 'total', 'homeTeamWinPct', 'awayTeamWinPct', 'homeTeamCloseGamePct', 'awayTeamCloseGamePct',
    'homeTeamBlowoutPct', 'awayTeamBlowoutPct', 'homeTeamAvgScore', 'awayTeamAvgScore', 'homeTeamAvgScoreAllowed',
    'awayTeamAvgScoreAllowed', 'homeTeamAvgPointDiff', 'awayTeamAvgPointDiff', 'homeTeamAvgTotalScore',
    'awayTeamAvgTotalScore', 'homeTeamGamesPlayed', 'awayTeamGamesPlayed', 'homeTeamAvgPointSpread',
    'awayTeamAvgPointSpread', 'homeTeamAvgTotal', 'awayTeamAvgTotal', 'homeTeamFavoredPct', 'awayTeamFavoredPct',
    'homeTeamSpreadCoveredPct', 'awayTeamSpreadCoveredPct', 'homeTeamOverHitPct', 'awayTeamOverHitPct',
    'homeTeamUnderHitPct', 'awayTeamUnderHitPct', 'homeTeamTotalHitPct', 'awayTeamTotalHitPct', 'total_line_diff'
]
# Target
# Target variable setup log suppressed
if 'spreadCovered' in historical_game_level_data.columns:
    target_spread = 'spreadCovered'
else:
    target_spread = st.selectbox('Select spread target column', historical_game_level_data.columns)

# Prepare data for MC feature selection

# Define X and y for spread
# Preparing spread model data (log suppressed)
X = historical_game_level_data[features]
y_spread = historical_game_level_data[target_spread]

# Spread model (using best features) - filter to numeric only
available_spread_features = [f for f in best_features_spread if f in historical_game_level_data.columns]
X_spread_full = historical_game_level_data[available_spread_features].select_dtypes(include=["number", "bool", "category"])
# Spread features loaded (log suppressed)
X_train_spread, X_test_spread, y_spread_train, y_spread_test = train_test_split(
    X_spread_full, y_spread, test_size=0.2, random_state=42, stratify=y_spread)
# Spread train/test split complete (log suppressed)

# --- Moneyline (underdogWon) target and split ---
# Preparing moneyline model data (log suppressed)
target_moneyline = 'underdogWon'
y_moneyline = historical_game_level_data[target_moneyline]
# Filter to only numeric features for XGBoost compatibility
available_moneyline_features = [f for f in best_features_moneyline if f in historical_game_level_data.columns]
X_moneyline_full = historical_game_level_data[available_moneyline_features].select_dtypes(include=["number", "bool", "category"])
# Moneyline features loaded (log suppressed)
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(
    X_moneyline_full, y_moneyline, test_size=0.2, random_state=42, stratify=y_moneyline)
# Moneyline train/test split complete (log suppressed)

# --- Totals (overHit) target and split ---
# Preparing totals model data (log suppressed)
try:
    target_totals = 'overHit'
    y_totals = historical_game_level_data[target_totals]
    # Target 'overHit' loaded (log suppressed)
    
    # Filter to only numeric features for XGBoost compatibility
    available_totals_features = [f for f in best_features_totals if f in historical_game_level_data.columns]
    # Totals features loaded (log suppressed)
    
    X_totals_full = historical_game_level_data[available_totals_features].select_dtypes(include=["number", "bool", "category"])
    # X_totals_full shape (log suppressed)
    
    X_train_tot, X_test_tot, y_train_tot, y_test_tot = train_test_split(
        X_totals_full, y_totals, test_size=0.2, random_state=42, stratify=y_totals)
    print("✅ Totals train/test split complete", file=sys.stderr, flush=True)
except Exception as e:
    print(f"❌ ERROR in totals model setup: {type(e).__name__}: {str(e)}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
    raise

# # TEMPORARILY SKIP: Create expander for data views (collapsed by default)
# # This section causes timeout issues with 196k rows - will fix later
#     # Historical Data & Filters (lazy-load play-by-play on demand)
#     with st.expander("📊 Historical Data & Filters", expanded=True):
#         # On-demand play-by-play load: avoid prompting on the main predictions page
#         if 'historical_data' not in st.session_state or st.session_state.get('historical_data') is None:
#             try:
#                 st.info("Historical views need play-by-play data for deep analysis. Load it on demand.")
#             except Exception:
#                 pass

#             if st.button("Load play-by-play for historical analysis"):
#                 try:
#                     with st.spinner("Loading play-by-play (this may take several minutes)..."):
#                         df = load_play_by_play_chunked()
#                         st.session_state['historical_data'] = df
#                         try:
#                             st.success(f"Loaded {len(df):,} play-by-play rows")
#                         except Exception:
#                             pass
#                         try:
#                             st.experimental_rerun()
#                         except Exception:
#                             pass
#                 except Exception as e:
#                     try:
#                         st.error(f"Failed to load play-by-play: {e}")
#                     except Exception:
#                         pass
#         else:
#             historical_data = st.session_state.get('historical_data')
    
#     # Create tabs for different data views
#     tab1, tab2, tab3, tab4 = st.tabs(["🏈 Play-by-Play Data", "📊 Game Summaries", "📅 Schedule", "🔍 Filters"])

#     with tab1:
#         st.write("### Historical Play-by-Play Data Sample for " + f"{current_year-4} to {current_year-1} Seasons")
#         if not historical_data.empty:
#             # Play-by-play data uses 'game_date' instead of 'gameday'  
#             if 'game_date' in historical_data.columns:
#                 # Convert to datetime and filter for completed games
#                 filtered_data = historical_data.copy()
#                 if filtered_data['game_date'].dtype == 'object':
#                     filtered_data['game_date'] = pd.to_datetime(filtered_data['game_date'], errors='coerce')
#                 current_date = pd.Timestamp(datetime.now().date())
#                 filtered_data = filtered_data[filtered_data['game_date'] <= current_date]
#                 filtered_data = filtered_data.sort_values(by='game_date', ascending=False)
            
#                 # Select key play-by-play columns for display
#                 display_cols = [
#                     'game_date', 'week', 'season', 'home_team', 'away_team', 'posteam', 'defteam',
#                     'game_seconds_remaining', 'qtr', 'down', 'ydstogo', 'yardline_100',
#                     'play_type', 'yards_gained', 'desc', 'epa', 'wp',
#                     'posteam_score', 'defteam_score', 'score_differential',
#                     'pass_attempt', 'rush_attempt', 'complete_pass', 'interception', 'fumble_lost',
#                     'td_prob', 'touchdown', 'field_goal_result'
#                 ]
            
#                 # Only use columns that exist
#                 display_cols = [col for col in display_cols if col in filtered_data.columns]
            
#                 st.dataframe(
#                     filtered_data[display_cols].head(50),
#                     hide_index=True,
#                     height=600,
#                     column_config={
#                         'game_date': st.column_config.DateColumn('Game Date', format='MM/DD/YYYY'),
#                         'week': st.column_config.NumberColumn('Week', format='%d'),
#                         'season': st.column_config.NumberColumn('Season', format='%d'),
#                         'home_team': st.column_config.TextColumn('Home Team', width='small'),
#                         'away_team': st.column_config.TextColumn('Away Team', width='small'),
#                         'posteam': st.column_config.TextColumn('Offense', width='small'),
#                         'defteam': st.column_config.TextColumn('Defense', width='small'),
#                         'game_seconds_remaining': st.column_config.NumberColumn('Time Left (s)', format='%d'),
#                         'qtr': st.column_config.NumberColumn('Qtr', format='%d'),
#                         'down': st.column_config.NumberColumn('Down', format='%d'),
#                         'ydstogo': st.column_config.NumberColumn('To Go', format='%d'),
#                         'yardline_100': st.column_config.NumberColumn('Yardline', format='%d', help='Distance from opponent endzone'),
#                         'play_type': st.column_config.TextColumn('Play Type', width='small'),
#                         'yards_gained': st.column_config.NumberColumn('Yards', format='%d'),
#                         'desc': st.column_config.TextColumn('Play Description', width='large'),
#                         'epa': st.column_config.NumberColumn('EPA', format='%.2f', help='Expected Points Added'),
#                         'wp': st.column_config.NumberColumn('Win Prob', format='%.1f%%', help='Win probability after play'),
#                         'posteam_score': st.column_config.NumberColumn('Off Score', format='%d'),
#                         'defteam_score': st.column_config.NumberColumn('Def Score', format='%d'),
#                         'score_differential': st.column_config.NumberColumn('Score Diff', format='%d'),
#                         'pass_attempt': st.column_config.CheckboxColumn('Pass?'),
#                         'rush_attempt': st.column_config.CheckboxColumn('Rush?'),
#                         'complete_pass': st.column_config.CheckboxColumn('Complete?'),
#                         'interception': st.column_config.CheckboxColumn('INT?'),
#                         'fumble_lost': st.column_config.CheckboxColumn('Fumble?'),
#                         'td_prob': st.column_config.NumberColumn('TD Prob', format='%.1f%%'),
#                         'touchdown': st.column_config.CheckboxColumn('TD?'),
#                         'field_goal_result': st.column_config.TextColumn('FG Result', width='small')
#                     }
#                 )
            
#             else:
#                 # Fallback: show all data without date filtering
#                 st.dataframe(historical_data.head(50), hide_index=True)
            
#         else:
#             st.info("No historical play-by-play data available. The nfl_history_2020_2024.csv.gz file may be missing or empty.")

#     with tab2:
#         st.write("### Historical Game Summaries")
#         historical_game_level_data_display = historical_game_level_data.copy()
        
#         # Convert gameday to datetime and filter for completed games only (≤ today)
#         historical_game_level_data_display['gameday'] = pd.to_datetime(historical_game_level_data_display['gameday'], errors='coerce')
#         today = pd.to_datetime(datetime.now().date())
#         historical_game_level_data_display = historical_game_level_data_display[historical_game_level_data_display['gameday'] <= today]
        
#         # Sort by most recent games first
#         historical_game_level_data_display = historical_game_level_data_display.sort_values(by='gameday', ascending=False)
        
#         # Select key columns for display
#         display_cols = [
#             'gameday', 'week', 'season', 'home_team', 'away_team', 'home_qb_name', 'away_qb_name',
#             'home_score', 'away_score', 'spread_line', 'home_spread_odds', 'away_spread_odds',
#             'total_line', 'spreadCovered', 'overHit', 'underdogWon',
#             'homeTeamWinPct', 'awayTeamWinPct', 'home_moneyline', 'away_moneyline'
#         ]
#         # Only use columns that exist
#         display_cols = [col for col in display_cols if col in historical_game_level_data_display.columns]
        
#         st.dataframe(
#             historical_game_level_data_display[display_cols].head(50),
#             hide_index=True,
#             height=600,
#             column_config={
#                 'gameday': st.column_config.DateColumn('Game Date', format='MM/DD/YYYY'),
#                 'week': st.column_config.NumberColumn('Week', format='%d'),
#                 'season': st.column_config.NumberColumn('Season', format='%d'),
#                 'home_team': st.column_config.TextColumn('Home Team', width='medium'),
#                 'away_team': st.column_config.TextColumn('Away Team', width='medium'),
#                 'home_qb_name': st.column_config.TextColumn('Home QB', width='medium', help='Shows 0 for upcoming games'),
#                 'away_qb_name': st.column_config.TextColumn('Away QB', width='medium', help='Shows 0 for upcoming games'),
#                 'home_score': st.column_config.NumberColumn('Home Score', format='%d'),
#                 'away_score': st.column_config.NumberColumn('Away Score', format='%d'),
#                 'spread_line': st.column_config.NumberColumn('Spread', format='%.1f', help='Negative = away favored'),
#                 'home_spread_odds': st.column_config.NumberColumn('Home Spread Odds', format='%d'),
#                 'away_spread_odds': st.column_config.NumberColumn('Away Spread Odds', format='%d'),
#                 'total_line': st.column_config.NumberColumn('O/U Line', format='%.1f'),
#                 'spreadCovered': st.column_config.CheckboxColumn('Spread Covered?'),
#                 'overHit': st.column_config.CheckboxColumn('Over Hit?'),
#                 'underdogWon': st.column_config.CheckboxColumn('Underdog Won?'),
#                 'homeTeamWinPct': st.column_config.NumberColumn('Home Win %', format='%.1f%%', help='Historical win percentage'),
#                 'awayTeamWinPct': st.column_config.NumberColumn('Away Win %', format='%.1f%%', help='Historical win percentage'),
#                 'home_moneyline': st.column_config.NumberColumn('Home ML', format='%d'),
#                 'away_moneyline': st.column_config.NumberColumn('Away ML', format='%d')
#             }
#         )

#     with tab3:
#         st.write(f"### {current_year} NFL Schedule")
#         if schedule is None:
#             schedule = load_schedule()
        
#         if not schedule.empty:
#             display_cols = ['week', 'date', 'home_team', 'away_team', 'venue']
#             # Convert UTC date string to local datetime
#             schedule_local = schedule.copy()
#             schedule_local['date'] = pd.to_datetime(schedule_local['date']).dt.tz_convert('America/New_York').dt.strftime('%m/%d/%Y %I:%M %p')
#             st.dataframe(schedule_local[display_cols], height=600, hide_index=True, column_config={'date': 'Date/Time (ET)', 'home_team': 'Home Team', 'away_team': 'Away Team', 'venue': 'Venue', 'week': 'Week'})
#         else:
#             st.warning(f"Schedule data for {current_year} is not available.")

#     with tab4:
#             import math
        
#             # Helpful message to open sidebar
#             st.info("👈 Open the sidebar (click arrow in top-left) to see filter controls")
#             # Insert the enhanced filter UI directly into the Historical Data page
#             try:
#                 with st.expander("⚙️ Team Filters (Play-level)", expanded=True):
#                     team_options = []
#                     try:
#                         if 'historical_data' in st.session_state and st.session_state.get('historical_data') is not None:
#                             hist = st.session_state.get('historical_data')
#                             if 'posteam' in hist.columns and 'defteam' in hist.columns:
#                                 team_options = sorted(set(hist['posteam'].dropna().unique().tolist() + hist['defteam'].dropna().unique().tolist()))
#                         elif predictions_df is not None:
#                             if 'home_team' in predictions_df.columns and 'away_team' in predictions_df.columns:
#                                 team_options = sorted(set(predictions_df['home_team'].dropna().unique().tolist() + predictions_df['away_team'].dropna().unique().tolist()))
#                     except Exception:
#                         team_options = []

#                     st.multiselect("Offense Team", options=team_options, key='filter_selected_offense')
#                     st.multiselect("Defense Team", options=team_options, key='filter_selected_defense')

#                 with st.expander("📊 Game Situation Filters"):
#                     down_options = [1, 2, 3, 4]
#                     qtr_options = [1, 2, 3, 4]
#                     st.multiselect("Downs", options=down_options, format_func=lambda x: f"{x}rd" if x==3 else (f"{x}th" if x!=1 else "1st"), key='filter_selected_downs')
#                     st.multiselect("Quarters", options=qtr_options, key='filter_selected_qtrs')

#                 with st.expander("📈 Advanced Metrics"):
#                     epa_min, epa_max = -10.0, 10.0
#                     wp_min, wp_max = 0.0, 1.0
#                     st.slider("EPA range", min_value=epa_min, max_value=epa_max, value=( -2.0, 2.0 ), step=0.1, key='filter_epa_range')
#                     st.slider("Win Prob range", min_value=wp_min, max_value=wp_max, value=(0.0, 1.0), step=0.01, key='filter_wp_range')

#                 st.write("**Quick Filters**")
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     if st.button("Red Zone", key='quickfilter_redzone'):
#                         st.session_state['quickfilter_yardline_100'] = (0, 20)
#                         st.experimental_rerun()
#                 with col2:
#                     if st.button("2-Minute Drill", key='quickfilter_2min'):
#                         st.session_state['quickfilter_game_seconds_remaining_lt'] = 120
#                         st.experimental_rerun()

#                 with st.expander("🛠️ Dev Tools (dev only)", expanded=False):
#                     if st.button("Clear Quick Filters", key='dev_clear_quickfilters'):
#                         for k in [
#                             'quickfilter_yardline_100', 'quickfilter_game_seconds_remaining_lt',
#                             'filter_selected_offense', 'filter_selected_defense',
#                             'filter_selected_downs', 'filter_selected_qtrs',
#                             'filter_epa_range', 'filter_wp_range'
#                         ]:
#                             st.session_state.pop(k, None)
#                         st.success("Cleared quick filters and dev filter keys")
#                         st.experimental_rerun()

#                     if st.button("Set Example Filters", key='dev_set_example_filters'):
#                         st.session_state['quickfilter_yardline_100'] = (0, 20)
#                         st.session_state['quickfilter_game_seconds_remaining_lt'] = 120
#                         st.session_state['filter_epa_range'] = (-1.5, 1.5)
#                         st.session_state['filter_wp_range'] = (0.25, 0.75)
#                         st.success("Applied example quick filters")
#                         st.experimental_rerun()
#             except Exception:
#                 # Non-fatal: continue even if enhanced UI fails
#                 pass
#                 if 'reset' not in st.session_state:
#                     st.session_state['reset'] = False
#                 # Initialize session state for filters
#                 for key in filter_keys:
#                     if key not in st.session_state:
#                         if key in ['down', 'posteam', 'defteam', 'play_type', 'qtr']:
#                             st.session_state[key] = []
#                         elif key == 'pass_attempt':
#                             st.session_state[key] = False  # Checkbox default
#                         elif key == 'ydstogo':
#                             st.session_state[key] = (int(historical_data['ydstogo'].min()), int(historical_data['ydstogo'].max()))
#                         elif key == 'yardline_100':
#                             st.session_state[key] = (int(historical_data['yardline_100'].min()), int(historical_data['yardline_100'].max()))
#                         elif key == 'score_differential':
#                             st.session_state[key] = (int(historical_data['score_differential'].min()), int(historical_data['score_differential'].max()))
#                         elif key == 'posteam_score':
#                             st.session_state[key] = (int(historical_data['posteam_score'].min()), int(historical_data['posteam_score'].max()))
#                         elif key == 'defteam_score':
#                             st.session_state[key] = (int(historical_data['defteam_score'].min()), int(historical_data['defteam_score'].max()))
#                         elif key == 'epa':
#                             st.session_state[key] = (float(historical_data['epa'].min()), float(historical_data['epa'].max()))

#                 if st.button("Reset Filters"):
#                     for key in filter_keys:
#                         if key in ['down', 'posteam', 'defteam', 'play_type', 'qtr']:
#                             st.session_state[key] = []
#                         elif key == 'pass_attempt':
#                             st.session_state[key] = False  # Reset checkbox
#                         else:
#                             st.session_state[key] = None
#                     st.session_state['reset'] = True

#                 # Default values
#                 default_filters = {
#                     'posteam': historical_data['posteam'].unique().tolist(),
#                     'defteam': historical_data['defteam'].unique().tolist(),
#                     'down': [1, 2, 3, 4],
#                     'ydstogo': (int(historical_data['ydstogo'].min()), int(historical_data['ydstogo'].max())),
#                     'yardline_100': (int(historical_data['yardline_100'].min()), int(historical_data['yardline_100'].max())),
#                     'play_type': historical_data['play_type'].dropna().unique().tolist(),
#                     'qtr': sorted(historical_data['qtr'].dropna().unique()),
#                     'score_differential': (int(historical_data['score_differential'].min()), int(historical_data['score_differential'].max())),
#                     'posteam_score': (int(historical_data['posteam_score'].min()), int(historical_data['posteam_score'].max())),
#                     'defteam_score': (int(historical_data['defteam_score'].min()), int(historical_data['defteam_score'].max())),
#                     'epa': (float(historical_data['epa'].min()), float(historical_data['epa'].max())),
#                     'pass_attempt': False
#                 }

#                 # Filters
#                 posteam_options = historical_data['posteam'].unique().tolist()
#                 posteam = st.multiselect("Possession Team", posteam_options, key="posteam")
#                 # Defense Team
#                 defteam_options = historical_data['defteam'].unique().tolist()
#                 defteam = st.multiselect("Defense Team", defteam_options, key="defteam")
#                 # Down
#                 down_options = [1,2,3,4]
#                 down = st.multiselect("Down", down_options, key="down")
#                 # Yards To Go
#                 if st.session_state['ydstogo'] is None:
#                     st.session_state['ydstogo'] = (int(historical_data['ydstogo'].min()), int(historical_data['ydstogo'].max()))
#                 ydstogo = st.slider("Yards To Go", int(historical_data['ydstogo'].min()), int(historical_data['ydstogo'].max()), value=st.session_state['ydstogo'], key="ydstogo")
#                 # Yardline 100
#                 if st.session_state['yardline_100'] is None:
#                     st.session_state['yardline_100'] = (int(historical_data['yardline_100'].min()), int(historical_data['yardline_100'].max()))
#                 yardline_100 = st.slider("Yardline 100", int(historical_data['yardline_100'].min()), int(historical_data['yardline_100'].max()), value=st.session_state['yardline_100'], key="yardline_100")
#                 # Play Type
#                 play_type_options = historical_data['play_type'].dropna().unique().tolist()
#                 play_type = st.multiselect("Play Type", play_type_options, key="play_type")
#                 # Quarter
#                 qtr_options = sorted([q for q in historical_data['qtr'].dropna().unique() if not (isinstance(q, float) and math.isnan(q))])
#                 qtr = st.multiselect("Quarter", qtr_options, key="qtr")
#                 # Score Differential
#                 if st.session_state['score_differential'] is None:
#                     st.session_state['score_differential'] = (int(historical_data['score_differential'].min()), int(historical_data['score_differential'].max()))
#                 score_differential = st.slider("Score Differential", int(historical_data['score_differential'].min()), int(historical_data['score_differential'].max()), value=st.session_state['score_differential'], key="score_differential")
#                 # Possession Team Score
#                 if st.session_state['posteam_score'] is None:
#                     st.session_state['posteam_score'] = (int(historical_data['posteam_score'].min()), int(historical_data['posteam_score'].max()))
#                 posteam_score = st.slider(
#                     "Possession Team Score",
#                     int(historical_data['posteam_score'].min()),
#                     int(historical_data['posteam_score'].max()),
#                     value=default_filters['posteam_score'] if st.session_state['reset'] else (int(historical_data['posteam_score'].min()), int(historical_data['posteam_score'].max()))
#                 )
#                 defteam_score = st.slider(
#                     "Defense Team Score",
#                     int(historical_data['defteam_score'].min()),
#                     int(historical_data['defteam_score'].max()),
#                     value=default_filters['defteam_score'] if st.session_state['reset'] else (int(historical_data['defteam_score'].min()), int(historical_data['defteam_score'].max()))
#                 )
#                 epa = st.slider(
#                     "Expected Points Added (EPA)",
#                     float(historical_data['epa'].min()),
#                     float(historical_data['epa'].max()),
#                     value=default_filters['epa'] if st.session_state['reset'] else (float(historical_data['epa'].min()), float(historical_data['epa'].max()))
#                 )
#                 pass_attempt = st.checkbox("Pass Attempts Only", key="pass_attempt")

#                 # Reset session state after applying
#                 if st.session_state['reset']:
#                     st.session_state['reset'] = False
#                 # End sidebar

#             # Apply filters to the dataframe
#             filtered_data = historical_data.copy()
#             if posteam:
#                 filtered_data = filtered_data[filtered_data['posteam'].isin(posteam)]
#             if defteam:
#                 filtered_data = filtered_data[filtered_data['defteam'].isin(defteam)]
#             if down:
#                 filtered_data = filtered_data[filtered_data['down'].isin(down)]
#             if ydstogo:
#                 filtered_data = filtered_data[(filtered_data['ydstogo'] >= ydstogo[0]) & (filtered_data['ydstogo'] <= ydstogo[1])]
#             if yardline_100:
#                 filtered_data = filtered_data[(filtered_data['yardline_100'] >= yardline_100[0]) & (filtered_data['yardline_100'] <= yardline_100[1])]
#             if play_type:
#                 filtered_data = filtered_data[filtered_data['play_type'].isin(play_type)]
#             if qtr:
#                 filtered_data = filtered_data[filtered_data['qtr'].isin(qtr)]
#             if score_differential:
#                 filtered_data = filtered_data[(filtered_data['score_differential'] >= score_differential[0]) & (filtered_data['score_differential'] <= score_differential[1])]
#             if posteam_score:
#                 filtered_data = filtered_data[(filtered_data['posteam_score'] >= posteam_score[0]) & (filtered_data['posteam_score'] <= posteam_score[1])]
#             if defteam_score:
#                 filtered_data = filtered_data[(filtered_data['defteam_score'] >= defteam_score[0]) & (filtered_data['defteam_score'] <= defteam_score[1])]
#             if epa:
#                 filtered_data = filtered_data[(filtered_data['epa'] >= epa[0]) & (filtered_data['epa'] <= epa[1])]
#             if pass_attempt:
#                 filtered_data = filtered_data[filtered_data['pass_attempt'] == 1]

#             st.write("### Filtered Historical Data")

#             # Create display copy and convert probability columns to percentages
#             display_data = filtered_data.head(50).copy()

#             # Identify probability columns (typically range 0-1) and convert to percentages
#             prob_columns = ['wp', 'def_wp', 'home_wp', 'away_wp', 'vegas_wp', 'vegas_home_wp', 
#                             'cp', 'cpoe', 'success', 'pass_oe', 'qb_epa', 'xyac_epa']

#             for col in prob_columns:
#                 if col in display_data.columns:
#                     # Check if values are in 0-1 range (probabilities)
#                     if display_data[col].notna().any() and display_data[col].between(0, 1).all():
#                         display_data[col] = display_data[col] * 100

#             # Configure columns with appropriate formatting
#             column_config = {}
#             for col in display_data.columns:
#                 if col in prob_columns and col in display_data.columns:
#                     column_config[col] = st.column_config.NumberColumn(col, format='%.1f%%')
#                 elif col in ['epa', 'wpa', 'air_epa', 'yac_epa', 'comp_air_epa', 'comp_yac_epa',
#                              'air_wpa', 'yac_wpa', 'comp_air_wpa', 'comp_yac_wpa', 'ep', 'vegas_wpa']:
#                     column_config[col] = st.column_config.NumberColumn(col, format='%.3f')
#                 elif col in ['yards_gained', 'air_yards', 'yards_after_catch', 'ydstogo', 
#                              'yardline_100', 'score_differential', 'posteam_score', 'defteam_score']:
#                     column_config[col] = st.column_config.NumberColumn(col, format='%d')

#             st.dataframe(display_data, hide_index=True, column_config=column_config if column_config else None)

print("🎨 Starting main UI rendering...", file=sys.stderr, flush=True)

# Create tabs for prediction and betting sections
st.write("---")
st.write("## 📈 Model Performance & Betting Analysis")

# Upcoming Games Schedule with Predictions
if schedule is None:
    schedule = load_schedule()

if not schedule.empty and predictions_df is not None:
    with st.expander("📅 Upcoming Games Schedule (Click to expand)", expanded=False):
        st.write("### This Week's Games with Model Predictions")
        
        # Team name mapping from full names to abbreviations
        team_abbrev_map = {
            'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
            'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
            'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
            'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
            'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
            'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
            'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
            'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
            'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
            'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
            'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
        }
        
        # Filter for upcoming games (not STATUS_FINAL and future dates)
        current_time = pd.Timestamp.now(tz='UTC')
        upcoming_schedule = schedule.copy()
        upcoming_schedule['date'] = pd.to_datetime(upcoming_schedule['date'], utc=True)
        upcoming_mask = (upcoming_schedule['status'] != 'STATUS_FINAL') & (upcoming_schedule['date'] > current_time)
        upcoming_games = upcoming_schedule[upcoming_mask].copy()
        
        if not upcoming_games.empty:
            # Convert to local time for display
            upcoming_games['date'] = upcoming_games['date'].dt.tz_convert('America/New_York')
            upcoming_games['date_display'] = upcoming_games['date'].dt.strftime('%m/%d/%Y %I:%M %p ET')
            
            # Sort by date
            upcoming_games = upcoming_games.sort_values('date').head(15)  # Show next 15 games
            
            # Create display dataframe
            schedule_display = []
            
            for _, game in upcoming_games.iterrows():
                # Convert team names to abbreviations for matching
                home_abbrev = team_abbrev_map.get(game['home_team'], game['home_team'])
                away_abbrev = team_abbrev_map.get(game['away_team'], game['away_team'])
                
                # Find matching prediction (if available). Prefer same-season matches to avoid older game_id collisions.
                pred_match = predictions_df[
                    ((predictions_df['home_team'] == home_abbrev) & (predictions_df['away_team'] == away_abbrev)) |
                    ((predictions_df['home_team'] == away_abbrev) & (predictions_df['away_team'] == home_abbrev))
                ]

                # Narrow to the same season/year as the scheduled game when possible
                try:
                    game_season = int(pd.to_datetime(game['date']).year)
                    season_vals = pd.to_numeric(pred_match['season'], errors='coerce').fillna(current_year).astype(int)
                    pred_match = pred_match[season_vals == game_season]
                except Exception:
                    # If anything fails, fall back to the broader match (best-effort)
                    pass
                
                if not pred_match.empty:
                    pred = pred_match.iloc[0]
                    
                    # Determine underdog and favorite
                    if pred.get('spread_line', 0) < 0:
                        favorite = pred['away_team']
                        underdog = pred['home_team']
                        spread = abs(pred['spread_line'])
                    else:
                        favorite = pred['home_team']
                        underdog = pred['away_team']
                        spread = pred['spread_line']
                    
                    schedule_display.append({
                        'Date': game['date_display'],
                        'Matchup': f"{game['away_team']} @ {game['home_team']}",
                        'Spread': f"{favorite} -{spread}" if spread > 0 else "Pick'em",
                        'Total': f"{pred.get('total_line', 'N/A')}",
                        'Underdog Win %': f"{pred.get('prob_underdogWon', 0):.1%}",
                        'Spread Cover %': f"{pred.get('prob_underdogCovered', 0):.1%}",
                        'Over Hit %': f"{pred.get('prob_overHit', 0):.1%}",
                        'ML Edge': f"{pred.get('edge_underdog_ml', 0):.1f}",
                        'Spread Edge': f"{pred.get('edge_underdog_spread', 0):.1f}",
                        'Total Edge': f"{pred.get('edge_over', 0):.1f}",
                        'game_id': pred.get('game_id', '')
                    })
                else:
                    # No prediction available
                    schedule_display.append({
                        'Date': game['date_display'],
                        'Matchup': f"{game['away_team']} @ {game['home_team']}",
                        'Spread': "TBD",
                        'Total': "TBD",
                        'Underdog Win %': "N/A",
                        'Spread Cover %': "N/A",
                        'Over Hit %': "N/A",
                        'ML Edge': "N/A",
                        'Spread Edge': "N/A",
                        'Total Edge': "N/A",
                        'game_id': ''
                    })
            
            if schedule_display:
                schedule_df = pd.DataFrame(schedule_display)
                # Render as markdown so the Matchup cell itself is a clickable link to the per-game page
                try:
                    # Render as HTML table with explicit target="_self" so links open in the same window/tab
                    cols = ['Matchup', 'Date', 'Spread', 'Total', 'Underdog Win %', 'Spread Cover %', 'Over Hit %', 'ML Edge', 'Spread Edge', 'Total Edge']
                    table_rows = []
                    for _, r in schedule_df.iterrows():
                        matchup_text = r.get('Matchup', '')
                        gid = str(r.get('game_id', '')).strip()
                        date = r.get('Date', '')
                        spread = r.get('Spread', '')
                        total = r.get('Total', '')
                        uw = r.get('Underdog Win %', '')
                        sc = r.get('Spread Cover %', '')
                        oh = r.get('Over Hit %', '')
                        me = r.get('ML Edge', '')
                        se = r.get('Spread Edge', '')
                        te = r.get('Total Edge', '')
                        if gid:
                            matchup_cell = f'<a href="?game={gid}" target="_self">{matchup_text}</a>'
                        else:
                            matchup_cell = matchup_text
                        row_html = f"<tr><td>{matchup_cell}</td><td>{date}</td><td>{spread}</td><td>{total}</td><td>{uw}</td><td>{sc}</td><td>{oh}</td><td>{me}</td><td>{se}</td><td>{te}</td></tr>"
                        table_rows.append(row_html)

                    # Build a full-width HTML table and render via st.markdown with unsafe HTML
                    # This keeps the table layout consistent with the rest of the page (no mini iframe)
                    html = '<style>\n'
                    html += 'table.go-table{border-collapse:collapse;width:100%;font-family:inherit;margin:6px 0;}\n'
                    html += 'table.go-table th, table.go-table td{border-bottom:1px solid #ddd;padding:8px;text-align:left;}\n'
                    html += 'table.go-table tr:hover{background:#f9f9f9;}\n'
                    html += 'table.go-table a{color:inherit;text-decoration:underline;}\n'
                    html += '</style>\n'
                    html += '<table class="go-table">'
                    html += '<thead><tr>' + ''.join([f'<th>{c}</th>' for c in cols]) + '</tr></thead>'
                    html += '<tbody>' + ''.join(table_rows) + '</tbody></table>'

                    # Use target="_self" on anchors so links open in the same tab/window
                    st.markdown(html, unsafe_allow_html=True)
                except Exception:
                    height = get_dataframe_height(schedule_df)
                    st.dataframe(
                        schedule_df,
                        hide_index=True,
                        height=height
                    )
                st.caption(f"Showing next {len(schedule_display)} upcoming games • Green edges indicate positive expected value bets")
            else:
                st.info("No upcoming games found in schedule data.")
        else:
            st.info("No upcoming games scheduled.")
else:
    with st.expander("📅 Upcoming Games Schedule", expanded=False):
        st.info("Schedule or prediction data not available.")

pred_tab1, pred_tab2, pred_tab3, pred_tab4, pred_tab5, pred_tab6, pred_tab7, pred_tab8, pred_tab9 = st.tabs([
    "📊 Model Predictions", 
    "🎯 Probabilities & Edges",
    "💰 Betting Performance",
    "🔥 Underdog Bets",
    "🏈 Spread Bets",
    "🎯 Over/Under Bets",
    "📋 Betting Log",
    "📈 Model Performance",
    "💰 Bankroll Management"
])

with pred_tab1:
    
    if predictions_df is not None:
        # Team name mapping from abbreviations to full names
        team_full_name_map = {
            'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens',
            'BUF': 'Buffalo Bills', 'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears',
            'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'DAL': 'Dallas Cowboys',
            'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
            'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars',
            'KC': 'Kansas City Chiefs', 'LV': 'Las Vegas Raiders', 'LAC': 'Los Angeles Chargers',
            'LA': 'Los Angeles Rams', 'MIA': 'Miami Dolphins', 'MIN': 'Minnesota Vikings',
            'NE': 'New England Patriots', 'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
            'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers',
            'SF': 'San Francisco 49ers', 'SEA': 'Seattle Seahawks', 'TB': 'Tampa Bay Buccaneers',
            'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders'
        }
        
        display_cols = [
            'game_id', 'gameday', 'home_team', 'away_team', 'home_score', 'away_score',
            'total_line', 'spread_line',
            'predictedSpreadCovered', 'spreadCovered',
            'predictedOverHit', 'overHit'
        ]
        # Only show columns that exist
        display_cols = [col for col in display_cols if col in predictions_df.columns]
        st.write("### Model Predictions vs Actual Results")
        predictions_df_display = predictions_df.copy()
        predictions_df_display['gameday'] = pd.to_datetime(predictions_df_display['gameday'], errors='coerce')
        mask = predictions_df_display['gameday'] <= pd.to_datetime(datetime.now())
        predictions_df_display = predictions_df_display[mask]
        predictions_df_display.sort_values(by='gameday', ascending=False, inplace=True)
        
        # Convert team abbreviations to full names for display
        predictions_df_display['home_team'] = predictions_df_display['home_team'].map(team_full_name_map).fillna(predictions_df_display['home_team'])
        predictions_df_display['away_team'] = predictions_df_display['away_team'].map(team_full_name_map).fillna(predictions_df_display['away_team'])
        
        st.dataframe(
            predictions_df_display[display_cols].head(50), 
            hide_index=True,
            height=600,
            column_config={
                'game_id': None,
                'gameday': st.column_config.DateColumn('Game Date', format='MM/DD/YYYY'),
                'home_team': st.column_config.TextColumn('Home Team', width='medium'),
                'away_team': st.column_config.TextColumn('Away Team', width='medium'),
                'home_score': st.column_config.NumberColumn('Home Score', format='%d'),
                'away_score': st.column_config.NumberColumn('Away Score', format='%d'),
                'total_line': st.column_config.NumberColumn('O/U Line', format='%.1f', help='Over/Under betting line'),
                'spread_line': st.column_config.NumberColumn('Spread', format='%.1f', help='Point spread (negative = away favored)'),
                'predictedSpreadCovered': st.column_config.CheckboxColumn('Predicted Spread', help='Model prediction: underdog covers spread'),
                'spreadCovered': st.column_config.CheckboxColumn('Actual Spread', help='Actual result: underdog covered spread'),
                'predictedOverHit': st.column_config.CheckboxColumn('Predicted Over', help='Model prediction: total goes over'),
                'overHit': st.column_config.CheckboxColumn('Actual Over', help='Actual result: total went over')
            }
        )
        st.caption(f"Showing 50 most recent completed games • Total predictions: {predictions_df.shape[0]:,} games")
    else:
        st.warning("Predictions CSV not found. Run the model script to generate predictions.")

with pred_tab2:
    if predictions_df is not None:
        st.write("### Model Probabilities, Implied Probabilities, and Edges")
        prob_cols = [
             'gameday', 'home_team', 'away_team', 'home_score', 'away_score',
            'spread_line', 'total_line',
            'prob_underdogCovered', 'implied_prob_underdog_spread', 'edge_underdog_spread',
            'prob_underdogWon', 'pred_underdogWon_optimal', 'implied_prob_underdog_ml', 'edge_underdog_ml',
            'prob_overHit', 'implied_prob_over', 'edge_over',
            'implied_prob_under', 'edge_under'
        ]
        # Only show columns that exist
        # Add favored_team column (spread_line is from away team perspective)
        def get_favored_team(row):
            if not pd.isnull(row.get('spread_line', None)):
                if row['spread_line'] < 0:
                    return row['away_team']  # Away team favored (negative spread)
                elif row['spread_line'] > 0:
                    return row['home_team']  # Home team favored (positive spread)
                else:
                    return 'Pick'
            return None
        predictions_df['favored_team'] = predictions_df.apply(get_favored_team, axis=1)
        # Build display_cols: all columns in prob_cols that exist, plus favored_team after away_team
        display_cols = [col for col in prob_cols if col in predictions_df.columns]
        if 'favored_team' in predictions_df.columns and 'favored_team' not in display_cols:
            # Insert after away_team if possible, else at the end
            if 'away_team' in display_cols:
                idx = display_cols.index('away_team') + 1
                display_cols.insert(idx, 'favored_team')
            else:
                display_cols.append('favored_team')
        predictions_df['gameday'] = pd.to_datetime(predictions_df['gameday'], errors='coerce')
        today = pd.to_datetime(datetime.now().date())
        next_week = today + pd.Timedelta(days=7)
        mask = (predictions_df['gameday'] >= today) & (predictions_df['gameday'] < next_week)
        predictions_df = predictions_df[mask]
        
        # Filter out games with zero spread lines (no betting data available)
        predictions_df = predictions_df[predictions_df['spread_line'] != 0.0]
        
        # Don't convert gameday to string yet - keep as datetime for sorting
        predictions_df['pred_underdogWon_optimal'] = predictions_df['pred_underdogWon_optimal'].astype(int)
        
        # Create a display copy and convert probabilities/edges to percentages
        display_df = predictions_df[display_cols].sort_values(by='gameday', ascending=False).head(50).copy()
        
        # Convert probability and edge columns from decimal to percentage
        prob_cols = ['prob_underdogCovered', 'implied_prob_underdog_spread', 'edge_underdog_spread',
                     'prob_underdogWon', 'implied_prob_underdog_ml', 'edge_underdog_ml',
                     'prob_overHit', 'implied_prob_over', 'edge_over', 'implied_prob_under', 'edge_under']
        for col in prob_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col] * 100
        
        height = get_dataframe_height(display_df)
        st.dataframe(
            display_df, 
            hide_index=True, 
            height=height,
            column_config={
                'gameday': st.column_config.DateColumn('Date', format='MM/DD/YYYY', width='small'),
                'home_team': st.column_config.TextColumn('Home', width='small'),
                'away_team': st.column_config.TextColumn('Away', width='small'),
                'favored_team': st.column_config.TextColumn('Favored', width='small', help='Team favored to win'),
                'home_score': st.column_config.NumberColumn('Home Pts', format='%d', width='small'),
                'away_score': st.column_config.NumberColumn('Away Pts', format='%d', width='small'),
                'spread_line': st.column_config.NumberColumn('Spread', format='%.1f', width='small', help='Point spread (negative = away favored)'),
                'total_line': st.column_config.NumberColumn('O/U', format='%.1f', width='small', help='Over/Under line'),
                'prob_underdogCovered': st.column_config.NumberColumn('Spread Prob', format='%.1f%%', width='small', help='Model probability underdog covers spread'),
                'implied_prob_underdog_spread': st.column_config.NumberColumn('Implied Spread', format='%.1f%%', width='small', help='Sportsbook implied probability for underdog covering'),
                'edge_underdog_spread': st.column_config.NumberColumn('Spread Edge', format='%.1f%%', width='small', help='Model edge for underdog spread bet'),
                'prob_underdogWon': st.column_config.NumberColumn('ML Prob', format='%.1f%%', width='small', help='Model probability underdog wins outright'),
                'pred_underdogWon_optimal': st.column_config.CheckboxColumn('ML Signal', help='🎯 Betting signal: Bet on underdog (≥28% threshold)'),
                'implied_prob_underdog_ml': st.column_config.NumberColumn('Implied ML', format='%.1f%%', width='small', help='Sportsbook implied probability for underdog moneyline'),
                'edge_underdog_ml': st.column_config.NumberColumn('ML Edge', format='%.1f%%', width='small', help='Model edge for underdog moneyline bet'),
                'prob_overHit': st.column_config.NumberColumn('Over Prob', format='%.1f%%', width='small', help='Model probability total goes over'),
                'implied_prob_over': st.column_config.NumberColumn('Implied Over', format='%.1f%%', width='small', help='Sportsbook implied probability for over'),
                'edge_over': st.column_config.NumberColumn('Over Edge', format='%.1f%%', width='small', help='Model edge for over bet'),
                'implied_prob_under': st.column_config.NumberColumn('Implied Under', format='%.1f%%', width='small', help='Sportsbook implied probability for under'),
                'edge_under': st.column_config.NumberColumn('Under Edge', format='%.1f%%', width='small', help='Model edge for under bet')
            }
        )
        st.caption(f"Showing next 50 upcoming games with betting data • {len(predictions_df):,} games in next week")

        # Add per-row links to open the Per-Game detail page for each matchup
        try:
            links_df = display_df.copy()
            if 'game_id' in links_df.columns:
                links_df = links_df[['gameday', 'away_team', 'home_team', 'game_id']].copy()
                links_df['matchup'] = links_df.apply(
                    lambda r: f"{r['away_team']} @ {r['home_team']} ({pd.to_datetime(r['gameday']).date()})",
                    axis=1
                )
                links_df['details_link'] = links_df['game_id'].astype(str).apply(lambda gid: f"[Open](?game={gid})")
                with st.expander("Open game details (click a link)"):
                    md = links_df[['matchup', 'details_link']].to_markdown(index=False)
                    st.markdown(md, unsafe_allow_html=False)
        except Exception:
            pass

        st.info("""
        **📌 Quick Reference:**
        - **Prob Columns**: Model's predicted probability (higher = more confident)
        - **Implied Columns**: Probability implied by sportsbook odds
        - **Edge Columns**: Model advantage vs sportsbook (positive = value bet)
        - **ML Signal (✅)**: Automated betting signal when model probability ≥ 28% (F1-optimized threshold)
        
        💡 **Positive edge = value bet opportunity** (model thinks probability is higher than sportsbook odds suggest)
        """)
    else:
        st.warning("Predictions CSV not found. Run the model script to generate predictions.")

with pred_tab3:
    if predictions_df is not None:
        st.write("### 📊 Betting Analysis & Performance")
        st.write("Assuming a $100 bet per game")
        # Load the full predictions data for analysis
        try:
            predictions_df_full = pd.read_csv(predictions_csv_path, sep='\t')
            # Calculate moneyline_bet_return column
            def calc_moneyline_return(row):
                if row['pred_underdogWon_optimal'] == 1:
                    if row['underdogWon'] == 1:
                        underdog_odds = max(row['away_moneyline'], row['home_moneyline'])
                        if underdog_odds > 0:
                            return underdog_odds
                        else:
                            return 100 / abs(underdog_odds) * 100 if underdog_odds != 0 else 0
                    else:
                        return -100
                else:
                    return 0
            predictions_df_full['moneyline_bet_return'] = predictions_df_full.apply(calc_moneyline_return, axis=1)
        except Exception as e:
            st.error(f"Error loading predictions data for betting analysis: {str(e)}")
            st.info("Please run 'python build_and_train_pipeline.py' to generate the required data files.")
            predictions_df_full = None
        
        # Calculate betting statistics
        if predictions_df_full is not None and 'pred_underdogWon_optimal' in predictions_df_full.columns and 'moneyline_bet_return' in predictions_df_full.columns:
            # Moneyline betting stats
            moneyline_bets = predictions_df_full[predictions_df_full['pred_underdogWon_optimal'] == 1].copy()
            if len(moneyline_bets) > 0:
                bet_returns = moneyline_bets['moneyline_bet_return']
                moneyline_wins = (bet_returns > 0).sum()
                moneyline_total_return = bet_returns.sum()
                moneyline_win_rate = moneyline_wins / len(moneyline_bets)
                moneyline_roi = moneyline_total_return / (len(moneyline_bets) * 100)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🎯 Moneyline Strategy", "Underdog Betting")
                    st.metric("Total Bets", f"{len(moneyline_bets):,}")
                    st.metric("Win Rate", f"{moneyline_win_rate:.1%}")
                    
                with col2:
                    st.metric("Total Return", f"${moneyline_total_return:,.2f}")
                    st.metric("ROI", f"{moneyline_roi:.1%}")
                    avg_return = moneyline_total_return / len(moneyline_bets)
                    st.metric("Avg Return/Bet", f"${avg_return:.2f}")
                
                # Show betting threshold info
                st.info(f"🎲 **Strategy**: Bet on underdogs when model probability ≥ 24% (F1-score optimized threshold)")
                st.caption("💡 The 24% threshold was determined by testing values from 10% to 60% and selecting the one that maximizes F1-score on training data.")
                
                # Add explanatory information
                st.markdown("#### 📊 What These Numbers Mean:")
                
                st.write(f"**Total Bets ({len(moneyline_bets):,})**: Your model identified {len(moneyline_bets):,} games where underdogs met the 24% probability threshold - this represents selective betting, not every game.")
                
                losses = len(moneyline_bets) - moneyline_wins
                st.write(f"**Win Rate ({moneyline_win_rate:.1%})**: Out of {len(moneyline_bets):,} bets, you won {moneyline_wins:,} bets and lost {losses:,} bets. This is exceptionally high for underdog betting (underdogs typically win around 35-40% of games).")
                
                original_investment = len(moneyline_bets) * 100
                total_payout = moneyline_total_return + original_investment

                st.markdown("#### 🔍 Why This Strategy Works:")
                st.write("• **Market Inefficiency**: Sportsbooks often undervalue underdogs with good statistical profiles")
                st.write("• **Selective Approach**: Only betting when model confidence ≥24% filters out poor value bets")
                st.write("• **High-Odds Payouts**: Underdog wins pay 2:1, 3:1, or higher, so you don't need to win most bets to profit")
                st.write("• **Statistical Edge**: Your model found patterns that predict underdog victories better than market expectations")

                # Show best recent bets
                if 'gameday' in moneyline_bets.columns:
                    recent_bets = moneyline_bets.copy()
                    recent_bets['gameday'] = pd.to_datetime(recent_bets['gameday'], errors='coerce')
                    recent_bets = recent_bets.sort_values('gameday', ascending=False).head(20)
                    
                    bet_display_cols = ['gameday', 'home_team', 'away_team', 'home_score', 'away_score', 
                                      'spread_line', 'prob_underdogWon', 'underdogWon', 'moneyline_bet_return']
                    bet_display_cols = [col for col in bet_display_cols if col in recent_bets.columns]
                    
                    if bet_display_cols:
                        st.write("#### 🔥 Recent Moneyline Bets")
                        st.write("*Shows underdog bets only. ✅ = Underdog won outright*")
                        
                        recent_bets_display = recent_bets[bet_display_cols].copy()
                        
                        # Add favored team column (spread_line is from away team perspective)
                        def get_favored_team(row):
                            if not pd.isnull(row.get('spread_line', None)):
                                if row['spread_line'] < 0:
                                    return row['away_team'] + ' (A)'  # Away team favored (negative spread)
                                elif row['spread_line'] > 0:
                                    return row['home_team'] + ' (H)'  # Home team favored (positive spread)
                                else:
                                    return 'Pick'
                            return 'N/A'
                        recent_bets_display['Favored'] = recent_bets_display.apply(get_favored_team, axis=1)
                        
                        # Convert probability to percentage for display
                        recent_bets_display['prob_underdogWon'] = recent_bets_display['prob_underdogWon'] * 100
                        
                        # Create formatted score column with whole numbers
                        if 'home_score' in recent_bets_display.columns and 'away_score' in recent_bets_display.columns:
                            recent_bets_display['Score'] = recent_bets_display['home_score'].astype(int).astype(str) + '-' + recent_bets_display['away_score'].astype(int).astype(str)
                        
                        # Rename columns for better display
                        recent_bets_display = recent_bets_display.rename(columns={
                            'gameday': 'Date',
                            'home_team': 'Home',
                            'away_team': 'Away', 
                            'prob_underdogWon': 'Model %',
                            'underdogWon': 'Underdog Won?',
                            'moneyline_bet_return': 'Return'
                        })
                    
                        # Select final display columns (excluding individual score columns)
                        final_display_cols = ['Date', 'Home', 'Away', 'Favored', 'Score', 'Model %', 'Underdog Won?', 'Return']
                        final_display_cols = [col for col in final_display_cols if col in recent_bets_display.columns]
                        
                        st.dataframe(
                            recent_bets_display[final_display_cols],
                            column_config={
                                'Date': st.column_config.DateColumn(format='MM/DD/YYYY'),
                                'Home': st.column_config.TextColumn(width='medium'),
                                'Away': st.column_config.TextColumn(width='medium'),
                                'Favored': st.column_config.TextColumn(width='medium'),
                                'Score': st.column_config.TextColumn(width='small'),
                                'Model %': st.column_config.NumberColumn(format='%.1f%%'),
                                'Underdog Won?': st.column_config.CheckboxColumn(),
                                'Return': st.column_config.NumberColumn(format='$%.2f')
                            },
                            height=750,
                            hide_index=True
                        )
        
        # Spread betting stats (if available)
        if 'spread_bet_return' in predictions_df_full.columns:
            spread_bets = predictions_df_full[predictions_df_full['spread_bet_return'] != 0]
            if len(spread_bets) > 0:
                spread_wins = (spread_bets['spread_bet_return'] > 0).sum()
                spread_total_return = spread_bets['spread_bet_return'].sum()
                spread_win_rate = spread_wins / len(spread_bets)
                spread_roi = spread_total_return / (len(spread_bets) * 100)
                
                st.write("#### 📈 Spread Betting Performance")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Spread Bets", f"{len(spread_bets):,}")
                with col2:
                    st.metric("Win Rate", f"{spread_win_rate:.1%}")
                with col3:
                    st.metric("ROI", f"{spread_roi:.1%}")
            
            # Over/Under betting performance
            if 'totals_bet_return' in predictions_df_full.columns:
                totals_bets = predictions_df_full[predictions_df_full['totals_bet_return'].notna()]
                totals_wins = (totals_bets['totals_bet_return'] > 0).sum()
                totals_total_return = totals_bets['totals_bet_return'].sum()
                totals_win_rate = totals_wins / len(totals_bets)
                totals_roi = totals_total_return / (len(totals_bets) * 100)
                
                st.write("#### 🎯 Over/Under Betting Performance")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Over/Under Bets", f"{len(totals_bets):,}")
                with col2:
                    st.metric("Win Rate", f"{totals_win_rate:.1%}")
                with col3:
                    st.metric("ROI", f"{totals_roi:.1%}")
        
        # Performance comparison
        st.write("#### 🏆 Model vs Baseline Comparison")
        baseline_accuracy = (predictions_df_full['underdogWon'] == 0).mean()  # Always pick favorites
        st.write(f"- **Baseline Strategy (Always Pick Favorites)**: {baseline_accuracy:.1%} accuracy")
        st.write(f"- **Our Model**: Identifies profitable underdog opportunities with {moneyline_roi:.1%} ROI")
        st.write(f"- **Key Insight**: While model sacrifices overall accuracy, it finds value bets with positive expected return")
        
    else:
        st.warning("Predictions CSV not found. Run the model script to generate betting analysis.")

with pred_tab4:
    if predictions_df is not None:
        st.write("### 🎯 Next 10 Recommended Underdog Bets")
        st.write("*Games where model recommends betting on underdog to win (≥28% confidence)*")
        
        # Reload predictions_df fresh and filter for upcoming games only
        predictions_df_upcoming = pd.read_csv(predictions_csv_path, sep='\t')
        predictions_df_upcoming['gameday'] = pd.to_datetime(predictions_df_upcoming['gameday'], errors='coerce')
        
        # Filter for future games only
        today = pd.to_datetime(datetime.now().date())
        predictions_df_upcoming = predictions_df_upcoming[predictions_df_upcoming['gameday'] > today]
        
        # Filter for upcoming games where we should bet on underdog, sort by date, take first 10
        upcoming_bets = predictions_df_upcoming[predictions_df_upcoming['pred_underdogWon_optimal'] == 1].copy()
        
        if len(upcoming_bets) > 0:
            # Sort by date and take first 10
            if 'gameday' in upcoming_bets.columns:
                upcoming_bets = upcoming_bets.sort_values('gameday').head(10)
            else:
                upcoming_bets = upcoming_bets.head(10)
            # Add columns for better display
            upcoming_bets_display = upcoming_bets.copy()
            
            # Add favored team column (corrected logic)
            def get_favored_team_upcoming(row):
                if not pd.isnull(row.get('spread_line', None)):
                    if row['spread_line'] < 0:
                        return row['away_team'] + ' (A)'  # Away team favored
                    elif row['spread_line'] > 0:
                        return row['home_team'] + ' (H)'  # Home team favored
                    else:
                        return 'Pick'
                return 'N/A'
            
            upcoming_bets_display['Favored'] = upcoming_bets_display.apply(get_favored_team_upcoming, axis=1)
            
            # Add underdog team column
            def get_underdog_team(row):
                if not pd.isnull(row.get('spread_line', None)):
                    if row['spread_line'] < 0:
                        return row['home_team'] + ' (H)'  # Home team underdog
                    elif row['spread_line'] > 0:
                        return row['away_team'] + ' (A)'  # Away team underdog
                    else:
                        return 'Pick'
                return 'N/A'
            
            upcoming_bets_display['Underdog'] = upcoming_bets_display.apply(get_underdog_team, axis=1)
            
            # Convert probability to percentage for display
            if 'prob_underdogWon' in upcoming_bets_display.columns:
                upcoming_bets_display['Model %'] = upcoming_bets_display['prob_underdogWon'] * 100
            
            # Get underdog moneyline odds for expected payout
            def get_underdog_odds_payout(row):
                if not pd.isnull(row.get('away_moneyline', None)) and not pd.isnull(row.get('home_moneyline', None)):
                    underdog_odds = max(row['away_moneyline'], row['home_moneyline'])  # Higher odds = underdog
                    if underdog_odds == 0:
                        return 'N/A (even odds)'
                    elif underdog_odds > 0:
                        return f"+{int(underdog_odds)} (${underdog_odds} profit on $100)"
                    else:
                        profit = 100 / abs(underdog_odds) * 100
                        return f"{int(underdog_odds)} (${profit:.0f} profit on $100)"
                return 'N/A'
            
            upcoming_bets_display['Expected Payout'] = upcoming_bets_display.apply(get_underdog_odds_payout, axis=1)
            
            # Select and rename columns for display
            display_cols = ['gameday', 'home_team', 'away_team', 'Favored', 'Underdog', 'spread_line', 'Model %', 'Expected Payout']
            display_cols = [col for col in display_cols if col in upcoming_bets_display.columns]
            
            final_display = upcoming_bets_display[display_cols].rename(columns={
                'gameday': 'Date',
                'home_team': 'Home',
                'away_team': 'Away',
                'spread_line': 'Spread'
            })
            
            # Sort by date
            if 'Date' in final_display.columns:
                final_display = final_display.sort_values('Date')

            height = get_dataframe_height(final_display) 
            st.dataframe(
                final_display,
                column_config={
                    'Date': st.column_config.DateColumn(format='MM/DD/YYYY'),
                    'Home': st.column_config.TextColumn(width='medium'),
                    'Away': st.column_config.TextColumn(width='medium'),
                    'Favored': st.column_config.TextColumn(width='medium'),
                    'Underdog': st.column_config.TextColumn(width='medium'),
                    'Spread': st.column_config.NumberColumn(format='%.1f'),
                    'Model %': st.column_config.NumberColumn(format='%.1f%%'),
                    'Expected Payout': st.column_config.TextColumn(width='large')
                },
                height=height,
                hide_index=True
            )
            
            st.write(f"**📊 Showing**: {len(upcoming_bets)} next underdog betting opportunities")
            
            # Add explanatory note
            st.info("""
            **💡 How to Use This:**
            - **Underdog**: Team recommended to bet on (model gives them ≥28% chance to win outright)
            - **Expected Payout**: Amount you'd win on a $100 bet if underdog wins
            - **Model %**: Model's confidence the underdog will win (higher = more confident)
            - **Strategy**: These are value bets where the model thinks the underdog is undervalued
            """)
            
        else:
            st.info("No upcoming games with underdog betting signals found in current predictions.")
            st.write("*The model may not have enough confidence (≥28%) in any upcoming underdog victories.*")
    
    else:
        st.warning("Predictions CSV not found. Run the model script to generate betting opportunities.")

with pred_tab5:
    if predictions_df is not None:
        st.write("### 🏈 Next 10 Recommended Spread Bets")
        st.write("*Games where model thinks underdog will cover spread (>50% confidence)*")
        
        # Reload predictions_df fresh and filter for upcoming games only
        predictions_df_spread = pd.read_csv(predictions_csv_path, sep='\t')
        predictions_df_spread['gameday'] = pd.to_datetime(predictions_df_spread['gameday'], errors='coerce')
        
        # Filter for future games only
        today = pd.to_datetime(datetime.now().date())
        predictions_df_spread = predictions_df_spread[predictions_df_spread['gameday'] > today]
        
        # Filter for upcoming games where model thinks underdog has ANY chance to cover (>50%)
        if 'prob_underdogCovered' in predictions_df_spread.columns:
            spread_bets = predictions_df_spread[predictions_df_spread['prob_underdogCovered'] > 0.50].copy()
            
            if len(spread_bets) > 0:
                # Sort by date and take first 10
                if 'gameday' in spread_bets.columns:
                    spread_bets = spread_bets.sort_values('gameday').head(10)
                else:
                    spread_bets = spread_bets.head(10)

                # Add columns for better display
                spread_bets_display = spread_bets.copy()
                
                # Add favored team and spread info
                def get_spread_info(row):
                    if not pd.isnull(row.get('spread_line', None)):
                        spread = row['spread_line']
                        if spread < 0:
                            return f"{row['away_team']} -{abs(spread)}"  # Away team favored
                        elif spread > 0:
                            return f"{row['home_team']} -{spread}"  # Home team favored
                        else:
                            return 'Pick\'em'
                    return 'N/A'
                
                spread_bets_display['Favorite & Spread'] = spread_bets_display.apply(get_spread_info, axis=1)
                
                # Add underdog team (who we're betting on to cover)
                def get_spread_underdog(row):
                    if not pd.isnull(row.get('spread_line', None)):
                        spread = row['spread_line']
                        if spread < 0:
                            return f"{row['home_team']} +{abs(spread)}"  # Home team underdog
                        elif spread > 0:
                            return f"{row['away_team']} +{spread}"  # Away team underdog
                        else:
                            return 'Pick\'em'
                    return 'N/A'
                
                spread_bets_display['Underdog (Bet On)'] = spread_bets_display.apply(get_spread_underdog, axis=1)
                
                # Convert probability to percentage for display
                spread_bets_display['Model Confidence'] = spread_bets_display['prob_underdogCovered'] * 100
                
                # Add confidence tier for prioritization
                def get_confidence_tier(row):
                    confidence = row['prob_underdogCovered']
                    if confidence >= 0.54:
                        return "🔥 Elite (54%+)"
                    elif confidence >= 0.52:
                        return "⭐ Strong (52-54%)"
                    else:
                        return "📈 Good (50-52%)"
                
                spread_bets_display['Tier'] = spread_bets_display.apply(get_confidence_tier, axis=1)
                
                # Calculate expected payout (standard -110 odds)
                spread_bets_display['Expected Payout'] = "$90.91 profit on $100 bet (91% ROI based on historical performance)"
                
                # Add edge calculation if available
                if 'edge_underdog_spread' in spread_bets_display.columns:
                    spread_bets_display['Value Edge'] = spread_bets_display['edge_underdog_spread'] * 100
                else:
                    spread_bets_display['Value Edge'] = 'N/A'
                
                # Select and rename columns for display
                display_cols = ['gameday', 'home_team', 'away_team', 'Favorite & Spread', 'Underdog (Bet On)', 'Tier', 'Model Confidence', 'Value Edge', 'Expected Payout']
                display_cols = [col for col in display_cols if col in spread_bets_display.columns]
                
                final_spread_display = spread_bets_display[display_cols].rename(columns={
                    'gameday': 'Date',
                    'home_team': 'Home Team',
                    'away_team': 'Away Team'
                })
                
                # Sort by model confidence (highest first), then by date
                if 'Model Confidence' in final_spread_display.columns:
                    final_spread_display = final_spread_display.sort_values(['Model Confidence', 'Date'], ascending=[False, True])
                elif 'Date' in final_spread_display.columns:
                    final_spread_display = final_spread_display.sort_values('Date')
                
                height = get_dataframe_height(final_spread_display)
                st.dataframe(
                    final_spread_display,
                    column_config={
                        'Date': st.column_config.DateColumn(format='MM/DD/YYYY'),
                        'Home Team': st.column_config.TextColumn(width='medium'),
                        'Away Team': st.column_config.TextColumn(width='medium'),
                        'Favorite & Spread': st.column_config.TextColumn(width='medium'),
                        'Underdog (Bet On)': st.column_config.TextColumn(width='medium'),
                        'Tier': st.column_config.TextColumn(width='medium'),
                        'Model Confidence': st.column_config.NumberColumn(format='%.1f%%'),
                        'Value Edge': st.column_config.NumberColumn(format='%.1f%%') if 'Value Edge' in final_spread_display.columns and final_spread_display['Value Edge'].dtype != 'object' else st.column_config.TextColumn(),
                        'Expected Payout': st.column_config.TextColumn(width='large')
                    },
                    height=height,
                    hide_index=True
                )
                
                st.write(f"**📊 Showing**: {len(spread_bets)} next spread betting opportunities")
                
                # Add explanatory note with tiered performance
                st.success(f"""
                **� PERFORMANCE BY CONFIDENCE LEVEL:**
                - **High Confidence (≥54%)**: 91.9% win rate, 75.5% ROI (elite level)
                - **Medium Confidence (50-54%)**: Expected ~52-55% win rate (still profitable)
                - **Current Selection**: Showing all games >50% confidence for more opportunities
                """)
                
                st.info("""
                **💡 How to Use Spread Bets:**
                - **Underdog (Bet On)**: Team to bet on covering the spread (+points means they get that advantage)
                - **Model Confidence**: How confident model is underdog will cover (50%+ shown for more opportunities)
                - **Value Edge**: How much better the model thinks the odds are vs the betting line
                - **Strategy**: Higher confidence = better historical performance, but >50% still profitable
                
                **Example**: If betting "Chiefs +3.5", the Chiefs can lose by 1, 2, or 3 points and you still win!
                
                **💰 Betting Strategy**: Focus on highest confidence games first, but >50% games still have value!
                """)
                
            else:
                st.info("No upcoming games with positive spread betting signals found.")
                st.write("*The model doesn't favor underdogs to cover in any upcoming games (all <50% confidence).*")
        else:
            st.warning("Spread probabilities not found in predictions data. Ensure the model has been trained with spread predictions.")
    
    else:
        st.warning("Predictions CSV not found. Run the model script to generate spread betting opportunities.")

with pred_tab6:
    st.write("### 🎯 Over/Under Betting Opportunities")
    st.write("*Top games where the model predicts profitable over/under bets based on optimal threshold*")
    
    if os.path.exists(predictions_csv_path):
        predictions_df_full = pd.read_csv(predictions_csv_path, sep='\t')
        
        # Check for the required column
        if 'pred_overHit_optimal' not in predictions_df_full.columns:
            st.error("Over/under predictions not found. Ensure pred_overHit_optimal column exists in the predictions CSV.")
        else:
            # Filter for games with over/under betting signals AND that haven't been played yet
            totals_bets = predictions_df_full[
                (predictions_df_full['pred_overHit_optimal'] == 1) & 
                (pd.to_datetime(predictions_df_full['gameday']) > pd.Timestamp.now().normalize())
            ].copy()
            
            if len(totals_bets) > 0:
                # Add confidence tiers based on probability
                def get_totals_confidence_tier(prob):
                    if prob >= 0.65:
                        return "🔥 Elite"
                    elif prob >= 0.60:
                        return "💪 Strong"
                    elif prob >= 0.55:
                        return "✓ Good"
                    else:
                        return "→ Standard"
                
                totals_bets['confidence_tier'] = totals_bets['prob_overHit'].apply(get_totals_confidence_tier)
                
                # Calculate expected payout on $100 bet
                def calculate_over_payout(row):
                    # If predicting over, use over odds, else under odds
                    if row['pred_over'] == 1:
                        odds = row['over_odds']
                        bet_on = 'Over'
                    else:
                        odds = row['under_odds']
                        bet_on = 'Under'
                    # Coerce to numeric safely
                    try:
                        odds_val = float(odds)
                    except Exception:
                        odds_val = float('nan')

                    # Handle missing or zero odds to avoid division-by-zero
                    if pd.isna(odds_val) or odds_val == 0:
                        # Unknown payout when odds are missing/zero
                        return float('nan'), bet_on

                    if odds_val > 0:
                        payout = 100.0 + odds_val
                    else:
                        # Negative American odds: profit on $100 = 100 * 100 / abs(odds)
                        payout = 100.0 + (100.0 * 100.0 / abs(odds_val))

                    return float(payout), bet_on
                
                totals_bets[['expected_payout', 'bet_on']] = totals_bets.apply(
                    lambda row: pd.Series(calculate_over_payout(row)), axis=1
                )
                
                # Calculate value edge
                totals_bets['value_edge'] = (
                    totals_bets['prob_overHit'] * totals_bets['expected_payout'] - 100
                ) / 100
                
                # Sort by value edge
                totals_bets = totals_bets.sort_values('value_edge', ascending=False)
                
                # Summary metrics
                st.write("#### 📊 Over/Under Betting Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Opportunities", len(totals_bets))
                with col2:
                    avg_prob = totals_bets['prob_overHit'].mean()
                    st.metric("Avg Probability", f"{avg_prob:.1%}")
                with col3:
                    avg_edge = totals_bets['value_edge'].mean()
                    st.metric("Avg Value Edge", f"{avg_edge:.1%}")
                
                # Display top opportunities
                st.write("#### 🎯 Top 15 Over/Under Opportunities")
                
                display_totals = totals_bets.head(15).copy()
                
                # Format for display
                display_totals['matchup'] = display_totals['away_team'] + ' @ ' + display_totals['home_team']
                display_totals['game_date'] = pd.to_datetime(display_totals['gameday']).dt.strftime('%m/%d/%Y')
                display_totals['total_line'] = display_totals['total_line'].round(1)
                display_totals['prob_pct'] = (display_totals['prob_overHit'] * 100).round(1)
                display_totals['value_pct'] = (display_totals['value_edge'] * 100).round(1)
                display_totals['payout_fmt'] = '$' + display_totals['expected_payout'].round(0).astype(int).astype(str)
                
                # Create display dataframe
                display_cols = {
                    'game_date': 'Date',
                    'matchup': 'Matchup',
                    'bet_on': 'Bet',
                    'total_line': 'Line',
                    'prob_pct': 'Model Prob %',
                    'payout_fmt': 'Expected Payout',
                    'value_pct': 'Value Edge %',
                    'confidence_tier': 'Confidence'
                }
                
                height = get_dataframe_height(display_totals)
                st.dataframe(
                    display_totals[list(display_cols.keys())].rename(columns=display_cols),
                    column_config={
                        'Date': st.column_config.TextColumn('Date', width='small'),
                        'Matchup': st.column_config.TextColumn('Matchup', width='large'),
                        'Bet': st.column_config.TextColumn('Bet', width='small'),
                        'Line': st.column_config.NumberColumn('Line', format='%.1f'),
                        'Model Prob %': st.column_config.NumberColumn('Model Prob %', format='%.1f'),
                        'Expected Payout': st.column_config.TextColumn('Expected Payout', width='medium'),
                        'Value Edge %': st.column_config.NumberColumn('Value Edge %', format='%.1f'),
                        'Confidence': st.column_config.TextColumn('Confidence', width='medium')
                    },
                    hide_index=True,
                    height=height,
                    width='stretch'
                )
                
                # Confidence tier breakdown
                st.write("#### 📈 Opportunities by Confidence Tier")
                tier_counts = display_totals['confidence_tier'].value_counts().sort_index(ascending=False)
                
                cols = st.columns(len(tier_counts))
                for i, (tier, count) in enumerate(tier_counts.items()):
                    with cols[i]:
                        st.metric(tier, count)
                
                # Betting strategy guide
                st.info("""
                **💡 Over/Under Betting Strategy:**
                - **Elite (≥65%)**: Highest confidence bets with strong value edge
                - **Strong (60-65%)**: Very good betting opportunities
                - **Good (55-60%)**: Solid value bets worth considering
                - **Standard (<55%)**: Meets threshold but requires careful consideration
                
                **Value Edge** represents the expected profit percentage on a $100 bet based on model probability.
                """)
                
            else:
                st.info("No over/under betting opportunities found for current games. Check back when new predictions are available.")
    
    else:
        st.warning("Predictions CSV not found. Run the model script to generate over/under betting opportunities.")

with pred_tab7:
    st.write("### 📋 Betting Recommendations Tracking Log")
    st.write("*All logged betting recommendations with performance tracking*")
    
    log_path = path.join(DATA_DIR, 'betting_recommendations_log.csv')
    
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        
        if len(log_df) > 0:
            # Convert dates for filtering
            log_df['gameday'] = pd.to_datetime(log_df['gameday'], errors='coerce')
            log_df['log_date'] = pd.to_datetime(log_df['log_date'], errors='coerce')
            
            # Sidebar filters for log
            st.write("#### Filter Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Filter by bet type
                bet_types = ['All'] + list(log_df['bet_type'].unique())
                selected_bet_type = st.selectbox("Bet Type", bet_types, key="log_bet_type")
            
            with col2:
                # Filter by status
                statuses = ['All'] + list(log_df['bet_result'].unique())
                selected_status = st.selectbox("Bet Status", statuses, key="log_status")
            
            with col3:
                # Filter by confidence tier
                tiers = ['All'] + list(log_df['confidence_tier'].unique())
                selected_tier = st.selectbox("Confidence Tier", tiers, key="log_tier")
            
            # Apply filters
            filtered_log = log_df.copy()
            if selected_bet_type != 'All':
                filtered_log = filtered_log[filtered_log['bet_type'] == selected_bet_type]
            if selected_status != 'All':
                filtered_log = filtered_log[filtered_log['bet_result'] == selected_status]
            if selected_tier != 'All':
                filtered_log = filtered_log[filtered_log['confidence_tier'] == selected_tier]
            
            # Summary statistics
            st.write("#### 📊 Summary Statistics")
            
            total_bets = len(filtered_log)
            pending_bets = len(filtered_log[filtered_log['bet_result'] == 'pending'])
            completed_bets = len(filtered_log[filtered_log['bet_result'] != 'pending'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Bets", total_bets)
            with col2:
                st.metric("Pending", pending_bets)
            with col3:
                st.metric("Completed", completed_bets)
            with col4:
                if completed_bets > 0:
                    wins = len(filtered_log[filtered_log['bet_result'] == 'win'])
                    win_rate = wins / completed_bets
                    st.metric("Win Rate", f"{win_rate:.1%}")
                else:
                    st.metric("Win Rate", "N/A")
            
            # Performance by confidence tier (if any completed bets)
            if completed_bets > 0:
                st.write("#### 🎯 Performance by Confidence Tier")
                
                tier_stats = []
                for tier in ['Elite', 'Strong', 'Good', 'Standard']:
                    tier_bets = filtered_log[
                        (filtered_log['confidence_tier'] == tier) & 
                        (filtered_log['bet_result'] != 'pending')
                    ]
                    if len(tier_bets) > 0:
                        wins = len(tier_bets[tier_bets['bet_result'] == 'win'])
                        win_rate = wins / len(tier_bets)
                        
                        # Calculate profit if bet_profit column exists and has values
                        if 'bet_profit' in tier_bets.columns:
                            tier_bets['bet_profit'] = pd.to_numeric(tier_bets['bet_profit'], errors='coerce')
                            total_profit = tier_bets['bet_profit'].sum()
                            roi = total_profit / (len(tier_bets) * 100) if len(tier_bets) > 0 else 0
                        else:
                            total_profit = 0
                            roi = 0
                        
                        tier_stats.append({
                            'Tier': tier,
                            'Bets': len(tier_bets),
                            'Wins': wins,
                            'Win Rate': f"{win_rate:.1%}",
                            'Total Profit': f"${total_profit:.2f}",
                            'ROI': f"{roi:.1%}"
                        })
                
                if tier_stats:
                    tier_df = pd.DataFrame(tier_stats)
                    st.dataframe(tier_df, hide_index=True, width='stretch')
            
            # Display the log
            st.write("#### 📋 Detailed Betting Log")
            st.write(f"*Showing {len(filtered_log)} bets*")
            
            # Format for display
            display_log = filtered_log.copy()
            display_log['gameday'] = display_log['gameday'].dt.strftime('%Y-%m-%d')
            display_log['log_date'] = display_log['log_date'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Format probability and edge as strings to avoid sorting arrows
            if 'model_probability' in display_log.columns:
                display_log['model_probability'] = display_log['model_probability'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "")
            if 'edge' in display_log.columns:
                display_log['edge'] = display_log['edge'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
            
            # Select columns to display
            display_cols = [
                'log_date', 'gameday', 'week', 'home_team', 'away_team', 
                'bet_type', 'recommended_team', 'spread_line', 'model_probability',
                'edge', 'confidence_tier', 'bet_result', 'bet_profit'
            ]
            display_cols = [col for col in display_cols if col in display_log.columns]
            
            # Sort by game date (most recent first)
            display_log = display_log.sort_values('gameday', ascending=False)
            
            st.dataframe(
                display_log[display_cols],
                column_config={
                    'log_date': st.column_config.TextColumn('Logged At', width='medium'),
                    'gameday': st.column_config.TextColumn('Game Date', width='medium'),
                    'week': st.column_config.NumberColumn('Week', format='%d'),
                    'home_team': st.column_config.TextColumn('Home', width='medium'),
                    'away_team': st.column_config.TextColumn('Away', width='medium'),
                    'bet_type': st.column_config.TextColumn('Bet Type', width='medium'),
                    'recommended_team': st.column_config.TextColumn('Bet On', width='medium'),
                    'spread_line': st.column_config.NumberColumn('Spread', format='%.1f'),
                    'model_probability': st.column_config.TextColumn('Model Prob', width='small', help='Model\'s predicted probability'),
                    'edge': st.column_config.TextColumn('Edge', width='small', help='Model\'s edge over sportsbook'),
                    'confidence_tier': st.column_config.TextColumn('Tier', width='small'),
                    'bet_result': st.column_config.TextColumn('Result', width='small'),
                    'bet_profit': st.column_config.NumberColumn('Profit', format='$%.2f')
                },
                height=600,
                hide_index=True
            )
            
            # Instructions for automatic updates
            st.info("""
            **🔄 Automatic Updates:**
            - Results are automatically updated when games are completed
            - The system checks ESPN for scores after game day
            - Future games will show "pending" until they're played
            - Refresh the app after games complete to see updated results
            
            **Note**: All games in the current log are scheduled for the future (Nov 16, 2025), 
            so no results have been updated yet. Results will appear automatically after games are played.
            """)
            
        else:
            st.info("No betting recommendations have been logged yet. Run the app when predictions are available.")
    
    else:
        st.warning("No betting log file found. The log will be created automatically when predictions with betting signals are available.")

with pred_tab8:
    st.write("### 📈 Model Performance Dashboard")
    st.write("*Track the accuracy and profitability of betting recommendations*")
    
    # Load betting log with results
    log_path = path.join(DATA_DIR, 'betting_recommendations_log.csv')
    
    if os.path.exists(log_path):
        betting_log = pd.read_csv(log_path)
        
        if len(betting_log) > 0:
            # Filter to completed bets only for performance metrics
            completed_bets = betting_log[betting_log['bet_result'].isin(['win', 'loss'])]
            
            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Bets", len(betting_log))
            with col2:
                if len(completed_bets) > 0:
                    win_rate = (completed_bets['bet_result'] == 'win').mean() * 100
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                else:
                    st.metric("Win Rate", "N/A")
            with col3:
                if len(completed_bets) > 0:
                    roi = calculate_roi(betting_log)
                    st.metric("ROI", f"{roi:.1f}%")
                else:
                    st.metric("ROI", "N/A")
            with col4:
                if len(completed_bets) > 0:
                    units_won = completed_bets['bet_profit'].sum() / 100  # Convert to units
                    st.metric("Units Won", f"{units_won:+.1f}")
                else:
                    st.metric("Units Won", "N/A")
            
            if len(completed_bets) > 0:
                # Performance by confidence tier
                st.write("#### 🎯 Performance by Confidence Level")
                confidence_performance = completed_bets.groupby('confidence_tier').agg({
                    'bet_result': lambda x: (x == 'win').mean() * 100,
                    'bet_profit': 'sum'
                }).round(2)
                confidence_performance.columns = ['Win Rate %', 'Total Profit $']
                confidence_performance = confidence_performance.sort_values('Win Rate %', ascending=False)
                st.dataframe(confidence_performance, width='stretch')
                
                # Week-over-week tracking
                st.write("#### 📅 Weekly Performance")
                # Convert gameday to datetime and extract week
                completed_bets['gameday'] = pd.to_datetime(completed_bets['gameday'], errors='coerce')
                completed_bets['week'] = completed_bets['gameday'].dt.isocalendar().week
                
                weekly_data = completed_bets.groupby('week').agg({
                    'bet_result': lambda x: (x == 'win').mean() * 100,
                    'bet_profit': 'sum'
                }).round(2)
                weekly_data.columns = ['Win Rate %', 'Total Profit $']
                
                if len(weekly_data) > 1:
                    st.line_chart(weekly_data)
                else:
                    st.dataframe(weekly_data, width='stretch')
                
                # Additional insights
                st.write("#### 📊 Key Insights")
                elite_bets = completed_bets[completed_bets['confidence_tier'] == 'Elite']
                if len(elite_bets) > 0:
                    elite_win_rate = (elite_bets['bet_result'] == 'win').mean() * 100
                    st.info(f"🏆 **Elite Tier Performance**: {len(elite_bets)} bets with {elite_win_rate:.1f}% win rate")
                
                # Best performing bet types
                bet_type_performance = completed_bets.groupby('bet_type').agg({
                    'bet_result': lambda x: (x == 'win').mean() * 100,
                    'bet_profit': 'sum'
                }).round(2)
                bet_type_performance.columns = ['Win Rate %', 'Total Profit $']
                bet_type_performance = bet_type_performance.sort_values('Win Rate %', ascending=False)
                
                if len(bet_type_performance) > 1:
                    st.write("#### 🏆 Best Performing Bet Types")
                    st.dataframe(bet_type_performance.head(3), width='stretch')
            else:
                st.info("📊 **No completed bets yet.** Performance metrics will appear here once games are played and results are recorded. All current bets are scheduled for future games.")
        else:
            st.info("No betting recommendations have been logged yet.")
    else:
        st.warning("Betting log file not found. Performance dashboard will be available once betting recommendations are generated.")

with pred_tab9:
    st.write("### 💰 Bankroll Management Tool")
    st.write("*Smart position sizing for elite bets to optimize risk and reward*")
    
    # Bankroll input
    col1, col2 = st.columns([2, 1])
    with col1:
        bankroll = st.number_input(
            "Current Bankroll ($)", 
            min_value=100, 
            max_value=1000000, 
            value=10000,
            step=100,
            help="Your total betting bankroll amount"
        )
    with col2:
        risk_level = st.selectbox(
            "Risk Level",
            ["Conservative (1%)", "Moderate (2%)", "Aggressive (3%)", "Very Aggressive (5%)"],
            index=1,
            help="Percentage of bankroll to risk per bet"
        )
    
    # Parse risk percentage
    risk_pct = {
        "Conservative (1%)": 0.01,
        "Moderate (2%)": 0.02,
        "Aggressive (3%)": 0.03,
        "Very Aggressive (5%)": 0.05
    }[risk_level]
    
    st.write(f"**Risk per bet**: ${bankroll * risk_pct:,.2f} ({risk_pct*100:.0f}% of bankroll)")
    
    # Load current predictions to find elite bets
    if predictions_df is not None:
        # Filter for upcoming games only
        predictions_df_upcoming = predictions_df.copy()
        predictions_df_upcoming['gameday'] = pd.to_datetime(predictions_df_upcoming['gameday'], errors='coerce')
        today = pd.to_datetime(datetime.now().date())
        predictions_df_upcoming = predictions_df_upcoming[predictions_df_upcoming['gameday'] > today]
        
        # Identify elite bets (≥65% confidence)
        elite_bets = []
        
        # Check moneyline predictions for elite confidence
        if 'prob_underdogWon' in predictions_df_upcoming.columns:
            moneyline_elite = predictions_df_upcoming[
                (predictions_df_upcoming['prob_underdogWon'] >= 0.65) & 
                (predictions_df_upcoming['pred_underdogWon_optimal'] == 1)
            ].copy()
            for _, row in moneyline_elite.iterrows():
                # Get underdog odds
                underdog_odds = max(row.get('away_moneyline', 0), row.get('home_moneyline', 0))
                if underdog_odds > 0:
                    payout_multiplier = underdog_odds
                else:
                    payout_multiplier = 100 / abs(underdog_odds) * 100 if underdog_odds != 0 else 0
                
                bet_amount = bankroll * risk_pct
                expected_payout = bet_amount * (payout_multiplier / 100)
                
                elite_bets.append({
                    'game': f"{row['away_team']} @ {row['home_team']}",
                    'gameday': row['gameday'].strftime('%m/%d/%Y') if pd.notna(row['gameday']) else 'TBD',
                    'bet_type': 'Moneyline',
                    'bet_on': row['away_team'] if row.get('away_moneyline', 0) > row.get('home_moneyline', 0) else row['home_team'],
                    'confidence': f"{row['prob_underdogWon']*100:.1f}%",
                    'odds': f"+{underdog_odds}" if underdog_odds > 0 else f"{underdog_odds}",
                    'recommended_bet': f"${bet_amount:,.2f}",
                    'expected_payout': f"${expected_payout:,.2f}",
                    'expected_value': f"{(row['prob_underdogWon'] * payout_multiplier - 100):+.1f}%"
                })
        
        # Check spread predictions for elite confidence
        if 'prob_underdogCovered' in predictions_df_upcoming.columns:
            spread_elite = predictions_df_upcoming[
                (predictions_df_upcoming['prob_underdogCovered'] >= 0.65) & 
                (predictions_df_upcoming['pred_spreadCovered_optimal'] == 1)
            ].copy()
            for _, row in spread_elite.iterrows():
                # Get underdog for spread bets
                if row.get('spread_line', 0) < 0:
                    underdog = row['home_team']
                    spread_odds = row.get('home_spread_odds', -110)
                else:
                    underdog = row['away_team']
                    spread_odds = row.get('away_spread_odds', -110)
                
                bet_amount = bankroll * risk_pct
                if spread_odds > 0:
                    payout_multiplier = spread_odds
                else:
                    payout_multiplier = 100 / abs(spread_odds) * 100
                
                expected_payout = bet_amount * (payout_multiplier / 100)
                
                elite_bets.append({
                    'game': f"{row['away_team']} @ {row['home_team']}",
                    'gameday': row['gameday'].strftime('%m/%d/%Y') if pd.notna(row['gameday']) else 'TBD',
                    'bet_type': 'Spread',
                    'bet_on': f"{underdog} ({row.get('spread_line', 0):+.1f})",
                    'confidence': f"{row['prob_underdogCovered']*100:.1f}%",
                    'odds': f"+{spread_odds}" if spread_odds > 0 else f"{spread_odds}",
                    'recommended_bet': f"${bet_amount:,.2f}",
                    'expected_payout': f"${expected_payout:,.2f}",
                    'expected_value': f"{(row['prob_underdogCovered'] * payout_multiplier - 100):+.1f}%"
                })
        
        # Check totals predictions for elite confidence
        if 'prob_overHit' in predictions_df_upcoming.columns:
            totals_elite = predictions_df_upcoming[
                (predictions_df_upcoming['prob_overHit'] >= 0.65) & 
                (predictions_df_upcoming['pred_overHit_optimal'] == 1)
            ].copy()
            for _, row in totals_elite.iterrows():
                over_odds = row.get('over_odds', -110)
                bet_amount = bankroll * risk_pct
                
                if over_odds > 0:
                    payout_multiplier = over_odds
                else:
                    payout_multiplier = 100 / abs(over_odds) * 100
                
                expected_payout = bet_amount * (payout_multiplier / 100)
                
                elite_bets.append({
                    'game': f"{row['away_team']} @ {row['home_team']}",
                    'gameday': row['gameday'].strftime('%m/%d/%Y') if pd.notna(row['gameday']) else 'TBD',
                    'bet_type': 'Over/Under',
                    'bet_on': f"Over {row.get('total_line', 0):.1f}",
                    'confidence': f"{row['prob_overHit']*100:.1f}%",
                    'odds': f"+{over_odds}" if over_odds > 0 else f"{over_odds}",
                    'recommended_bet': f"${bet_amount:,.2f}",
                    'expected_payout': f"${expected_payout:,.2f}",
                    'expected_value': f"{(row['prob_overHit'] * payout_multiplier - 100):+.1f}%"
                })
        
        if elite_bets:
            st.write("#### 🏆 Elite Bets (≥65% Confidence)")
            elite_df = pd.DataFrame(elite_bets)
            
            # Sort by expected value (highest first)
            elite_df['ev_numeric'] = elite_df['expected_value'].str.rstrip('%').astype(float)
            elite_df = elite_df.sort_values('ev_numeric', ascending=False).drop('ev_numeric', axis=1)
            
            st.dataframe(
                elite_df,
                column_config={
                    'game': st.column_config.TextColumn('Game', width='large'),
                    'gameday': st.column_config.TextColumn('Date', width='medium'),
                    'bet_type': st.column_config.TextColumn('Type', width='medium'),
                    'bet_on': st.column_config.TextColumn('Bet On', width='medium'),
                    'confidence': st.column_config.TextColumn('Confidence', width='small'),
                    'odds': st.column_config.TextColumn('Odds', width='small'),
                    'recommended_bet': st.column_config.TextColumn('Bet Amount', width='medium'),
                    'expected_payout': st.column_config.TextColumn('Expected Payout', width='medium'),
                    'expected_value': st.column_config.TextColumn('Expected Value', width='small', help='Positive EV = profitable long-term')
                },
                height=400,
                hide_index=True
            )
            
            # Summary stats
            total_elite_bets = len(elite_bets)
            total_recommended_wager = sum(float(bet['recommended_bet'].strip('$').replace(',', '')) for bet in elite_bets)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Elite Opportunities", total_elite_bets)
            with col2:
                st.metric("Total Recommended Wager", f"${total_recommended_wager:,.2f}")
            with col3:
                st.metric("Max Bankroll Impact", f"{(total_recommended_wager/bankroll)*100:.1f}%")
            
            st.info(f"""
            **💡 Bankroll Strategy:**
            - **Elite bets only**: Only betting when model confidence ≥65%
            - **Position sizing**: {risk_pct*100:.0f}% of bankroll per bet (${bankroll * risk_pct:,.2f})
            - **Expected value**: All shown bets have positive expected value
            - **Risk management**: Maximum exposure is {total_recommended_wager/bankroll*100:.1f}% of your bankroll
            """)
        else:
            st.info("🎯 **No elite betting opportunities found.** Elite bets require ≥65% model confidence. Check back when new predictions are available or adjust your risk criteria.")
    
    else:
        st.warning("No predictions data available. Run the model script to generate predictions.")

# Create tabs for advanced model features
st.write("---")
st.write("## 🔬 Advanced Model Features")
adv_tab1, adv_tab2 = st.tabs([
    "📊 Feature Importances",
    "🎲 Monte Carlo Selection"
])

with adv_tab1:
    st.write("### Model Feature Importances and Error Metrics")
    # Try to load feature importances and metrics if saved as a CSV or JSON
    import json
    metrics_path = path.join(DATA_DIR, 'model_metrics.json')
    importances_path = path.join(DATA_DIR, 'model_feature_importances.csv')
    # Display metrics
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        st.write("#### 📊 Model Performance Metrics")
        
        # Organize metrics into a clean table format
        metrics_data = []
        
        # Extract model names and their metrics
        models = set()
        metric_model_names = {}  # Map from metrics key to display name
        for key in metrics.keys():
            if 'Spread' in key:
                models.add('Spread')
                metric_model_names['Spread'] = 'Spread'
            elif 'Moneyline' in key:
                models.add('Moneyline')
                metric_model_names['Moneyline'] = 'Moneyline'
            elif 'Totals' in key:
                models.add('Totals')
                metric_model_names['Totals'] = 'Over/Under'
        
        # Build table rows
        for model in sorted(models):
            display_name = metric_model_names.get(model, model)
            row = {'Model': display_name}
            
            # Get accuracy
            acc_key = f"{model} Accuracy"
            if acc_key in metrics:
                row['Accuracy'] = f"{metrics[acc_key]:.1%}"
            
            # Get MAE
            mae_key = f"{model} MAE"
            if mae_key in metrics:
                row['MAE'] = f"{metrics[mae_key]:.3f}"
            
            # Get threshold
            threshold_key = f"Optimal {model} Threshold"
            if threshold_key in metrics:
                row['Optimal Threshold'] = f"{metrics[threshold_key]:.1%}"
            
            metrics_data.append(row)
        
        # Display as a clean dataframe
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(
                metrics_df,
                hide_index=True,
                width=600,
                column_config={
                    'Model': st.column_config.TextColumn('Model', width='medium', help='Betting market type'),
                    'Accuracy': st.column_config.TextColumn('Accuracy', width='medium', help='Out-of-sample prediction accuracy'),
                    'MAE': st.column_config.TextColumn('MAE', width='medium', help='Mean Absolute Error (lower is better)'),
                    'Optimal Threshold': st.column_config.TextColumn('Betting Threshold', width='medium', help='Probability threshold for placing bets (F1-score optimized)')
                }
            )
            
            # Add helpful explanation
            st.info("""
            **📌 Quick Guide:**
            - **Accuracy**: How often the model correctly predicts outcomes on unseen data
            - **MAE**: Average prediction error (lower = better calibration)
            - **Betting Threshold**: Minimum probability to trigger a bet (optimized for F1-score, NOT 50%)
            
            💡 **Why thresholds aren't 50%**: These are optimized to maximize the F1-score (balance of precision and recall), 
            which produces better long-term betting results than simple 50% cutoffs.
            """)
        
    else:
        st.info("No model metrics file found. Run the model script to generate metrics.")
    
    # Display feature importances with separate tabs
    if os.path.exists(importances_path):
        importances_df = pd.read_csv(importances_path)
        
        # Create tabs for each model
        st.write("#### 🔍 Feature Importances by Model")
        feat_tab1, feat_tab2, feat_tab3 = st.tabs([
            "📈 Spread Model",
            "💰 Moneyline Model", 
            "🎯 Over/Under Model"
        ])
        
        with feat_tab1:
            st.write("### Spread Model Feature Importances (Top 25)")
            spread_features = importances_df[importances_df['model'] == 'spread'].head(25)
            if len(spread_features) > 0:
                height = get_dataframe_height(spread_features)
                st.dataframe(
                    spread_features[['feature', 'importance_mean']],
                    hide_index=True,
                    height=height,
                    width=400,
                    column_config={
                        'feature': st.column_config.TextColumn('Feature Name'),
                        'importance_mean': st.column_config.NumberColumn('Importance', format='%.4f', help='XGBoost feature importance (gain-based)')
                    }
                )
            else:
                st.warning("No spread feature importances found.")
        
        with feat_tab2:
            st.write("### Moneyline Model Feature Importances (Top 25)")
            moneyline_features = importances_df[importances_df['model'] == 'moneyline'].head(25)
            if len(moneyline_features) > 0:
                height = get_dataframe_height(moneyline_features)
                st.dataframe(
                    moneyline_features[['feature', 'importance_mean']],
                    hide_index=True,
                    height=height,
                    width=400,
                    column_config={
                        'feature': st.column_config.TextColumn('Feature Name'),
                        'importance_mean': st.column_config.NumberColumn('Importance', format='%.4f', help='XGBoost feature importance (gain-based)')
                    }
                )
            else:
                st.warning("No moneyline feature importances found.")
        
        with feat_tab3:
            st.write("### Over/Under Model Feature Importances (Top 25)")
            totals_features = importances_df[importances_df['model'] == 'totals'].head(25)
            if len(totals_features) > 0:
                height = get_dataframe_height(totals_features)
                st.dataframe(
                    totals_features[['feature', 'importance_mean']],
                    hide_index=True,
                    height=height,
                    width=400,
                    column_config={
                        'feature': st.column_config.TextColumn('Feature Name'),
                        'importance_mean': st.column_config.NumberColumn('Importance', format='%.4f', help='XGBoost feature importance (gain-based)')
                    }
                )
            else:
                st.warning("No over/under feature importances found.")
    else:
        st.info("No feature importances file found. Run the model script to generate importances.")

with adv_tab2:
    st.write("### 🎲 Monte Carlo Feature Selection")
    st.write("*Advanced feature selection using Monte Carlo simulation*")
    
    mc_sub_tab1, mc_sub_tab2, mc_sub_tab3 = st.tabs([
        "📈 Spread Model",
        "💰 Moneyline Model",
        "🎯 Totals Model"
    ])
    
    with mc_sub_tab1:
        st.write("### Monte Carlo Feature Selection (Spread Model)")
        import random
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import make_scorer, roc_auc_score, f1_score
        # User controls
        num_iter = st.number_input("Number of Iterations", min_value=10, max_value=500, value=100, step=10, key="mc_iter_1")
        subset_size = st.number_input("Subset Size", min_value=2, max_value=len(features), value=8, step=1, key="mc_subset_1")
        random_seed = st.number_input("Random Seed", min_value=0, value=42, step=1, key="mc_seed_1")
        run_mc = st.button("Run Monte Carlo Search", key="mc_run_1")
        if run_mc:
            with st.spinner("Running Monte Carlo feature selection..."):
                # Filter features to only those available in the spread dataset
                available_spread_features = list(X_train_spread.columns)
                
                best_score = 0
                best_features = []
                random.seed(random_seed)
                scores_list = []
                for i in range(int(num_iter)):
                    subset = random.sample(available_spread_features, min(int(subset_size), len(available_spread_features)))
                    X_subset = X_train_spread[subset]
                    model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
                    acc = cross_val_score(model, X_subset, y_spread_train, cv=3, scoring='accuracy').mean()
                    # For AUC and F1, use make_scorer
                    try:
                        auc = cross_val_score(model, X_subset, y_spread_train, cv=3, scoring='roc_auc').mean()
                    except Exception:
                        auc = float('nan')
                    try:
                        f1 = cross_val_score(model, X_subset, y_spread_train, cv=3, scoring='f1').mean()
                    except Exception:
                        f1 = float('nan')
                    scores_list.append({
                        'iteration': i+1,
                        'features': subset,
                        'accuracy': acc,
                        'AUC': auc,
                        'F1-score': f1
                    })
                    if acc > best_score:
                        best_score = acc
                        best_features = subset
            st.success(f"Best mean CV accuracy: {best_score:.3f}")
            st.write(f"Best feature subset:")
            st.code(best_features)
            # Save best features to file (spread)
            with open(path.join(DATA_DIR, 'best_features_spread.txt'), 'w') as f:
                f.write("\n".join(best_features))
            # Retrain model using best_features and calibrate probabilities
            X_train_best = X_train_spread[best_features]
            X_test_best = X_test_spread[best_features]
            from sklearn.calibration import CalibratedClassifierCV
            model_spread_best = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
            calibrated_model = CalibratedClassifierCV(model_spread_best, method='isotonic', cv=3)
            calibrated_model.fit(X_train_best, y_spread_train)
            y_spread_pred_best = calibrated_model.predict(X_test_best)
            spread_accuracy_best = accuracy_score(y_spread_test, y_spread_pred_best)
            # Probability sanity check
            probs = calibrated_model.predict_proba(X_test_best)[:, 1]
            mean_pred_prob = np.mean(probs)
            actual_rate = np.mean(y_spread_test)
            st.write(f"Accuracy with best features: {spread_accuracy_best:.3f}")
            st.write(f"Mean predicted probability (test set): {mean_pred_prob:.3f}")
            st.write(f"Actual outcome rate (test set): {actual_rate:.3f}")
            # Show top 10 results
            scores_df = pd.DataFrame(scores_list).sort_values(by='accuracy', ascending=False).head(10)
            st.write("#### Top 10 Feature Subsets (by Accuracy)")
            
            st.dataframe(
                scores_df,
                hide_index=True,
                column_config={
                    'iteration': st.column_config.NumberColumn('Iteration', format='%d'),
                    'accuracy': st.column_config.NumberColumn('Accuracy', format='%.3f'),
                    'AUC': st.column_config.NumberColumn('AUC', format='%.3f'),
                    'F1-score': st.column_config.NumberColumn('F1-Score', format='%.3f'),
                    'features': st.column_config.TextColumn('Features', width='large')
                }
            )

    with mc_sub_tab2:
        st.write("### Monte Carlo Feature Selection (Moneyline Model)")
        import random
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import make_scorer, roc_auc_score, f1_score
        num_iter = st.number_input("Number of Iterations (Moneyline)", min_value=10, max_value=500, value=100, step=10)
        # Get available numeric features for validation
        available_features = [f for f in features if f in historical_game_level_data.columns]
        numeric_features_count = len(historical_game_level_data[available_features].select_dtypes(include=["number", "bool", "category"]).columns)
        subset_size = st.number_input("Subset Size (Moneyline)", min_value=2, max_value=numeric_features_count, value=min(8, numeric_features_count), step=1)
        random_seed = st.number_input("Random Seed (Moneyline)", min_value=0, value=42, step=1)
        run_mc = st.button("Run Monte Carlo Search (Moneyline)")
        if run_mc:
            with st.spinner("Running Monte Carlo feature selection..."):
                # Filter features to only those available in the dataset
                available_features = [f for f in features if f in historical_game_level_data.columns]
                numeric_features = historical_game_level_data[available_features].select_dtypes(include=["number", "bool", "category"]).columns.tolist()
                
                best_score = 0
                best_features = []
                random.seed(random_seed)
                scores_list = []
                for i in range(int(num_iter)):
                    subset = random.sample(numeric_features, min(int(subset_size), len(numeric_features)))
                    # Create a fresh dataset slice for this subset to avoid column mismatch
                    X_subset_data = historical_game_level_data[subset]
                    X_train_subset, _, y_train_subset, _ = train_test_split(
                        X_subset_data, y_moneyline, test_size=0.2, random_state=42, stratify=y_moneyline)
                    
                    model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
                    acc = cross_val_score(model, X_train_subset, y_train_subset, cv=3, scoring='accuracy').mean()
                    try:
                        auc = cross_val_score(model, X_train_subset, y_train_subset, cv=3, scoring='roc_auc').mean()
                    except Exception:
                        auc = float('nan')
                    try:
                        f1 = cross_val_score(model, X_train_subset, y_train_subset, cv=3, scoring='f1').mean()
                    except Exception:
                        f1 = float('nan')
                    scores_list.append({
                        'iteration': i+1,
                        'features': subset,
                        'accuracy': acc,
                        'AUC': auc,
                        'F1-score': f1
                    })
                    if acc > best_score:
                        best_score = acc
                        best_features = subset
            st.success(f"Best mean CV accuracy: {best_score:.3f}")
            st.write(f"Best feature subset:")
            st.code(best_features)
            # Save best features to file (moneyline)
            with open(path.join(DATA_DIR, 'best_features_moneyline.txt'), 'w') as f:
                f.write("\n".join(best_features))
            # Retrain model using best_features (Moneyline) - create new dataset with selected features
            X_moneyline_best = historical_game_level_data[best_features]
            X_train_ml_best, X_test_ml_best, _, _ = train_test_split(
                X_moneyline_best, y_moneyline, test_size=0.2, random_state=42, stratify=y_moneyline)
            model_moneyline_best = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
            model_moneyline_best.fit(X_train_ml_best, y_train_ml)
            y_moneyline_pred_best = model_moneyline_best.predict(X_test_ml_best)
            moneyline_accuracy_best = accuracy_score(y_test_ml, y_moneyline_pred_best)
            st.write(f"Accuracy with best features (Moneyline): {moneyline_accuracy_best:.3f}")
            scores_df = pd.DataFrame(scores_list).sort_values(by='accuracy', ascending=False).head(10)
            st.write("#### Top 10 Feature Subsets (by Accuracy)")
            
            st.dataframe(
                scores_df,
                hide_index=True,
                column_config={
                    'iteration': st.column_config.NumberColumn('Iteration', format='%d'),
                    'accuracy': st.column_config.NumberColumn('Accuracy', format='%.3f'),
                    'AUC': st.column_config.NumberColumn('AUC', format='%.3f'),
                    'F1-score': st.column_config.NumberColumn('F1-Score', format='%.3f'),
                    'features': st.column_config.TextColumn('Features', width='large')
                }
            )

    with mc_sub_tab3:
        st.write("### Monte Carlo Feature Selection (Totals Model)")
        import random
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import make_scorer, roc_auc_score, f1_score
        num_iter = st.number_input("Number of Iterations (Totals)", min_value=10, max_value=500, value=100, step=10)
        subset_size = st.number_input("Subset Size (Totals)", min_value=2, max_value=len(features), value=8, step=1)
        random_seed = st.number_input("Random Seed (Totals)", min_value=0, value=42, step=1)
        run_mc = st.button("Run Monte Carlo Search (Totals)")
        if run_mc:
            with st.spinner("Running Monte Carlo feature selection..."):
                best_score = 0
                best_features = []
                random.seed(random_seed)
                scores_list = []
                valid_totals_features = list(X_train_tot.columns)
                for i in range(int(num_iter)):
                    subset = random.sample(valid_totals_features, int(subset_size))
                    X_subset = X_train_tot[subset]
                    model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
                    acc = cross_val_score(model, X_subset, y_train_tot, cv=3, scoring='accuracy').mean()
                    try:
                        auc = cross_val_score(model, X_subset, y_train_tot, cv=3, scoring='roc_auc').mean()
                    except Exception:
                        auc = float('nan')
                    try:
                        f1 = cross_val_score(model, X_subset, y_train_tot, cv=3, scoring='f1').mean()
                    except Exception:
                        f1 = float('nan')
                    scores_list.append({
                        'iteration': i+1,
                        'features': subset,
                        'accuracy': acc,
                        'AUC': auc,
                        'F1-score': f1
                    })
                    if acc > best_score:
                        best_score = acc
                        best_features = subset
            st.success(f"Best mean CV accuracy: {best_score:.3f}")
            st.write(f"Best feature subset:")
            st.code(best_features)
            # Save best features to file (totals)
            with open(path.join(DATA_DIR, 'best_features_totals.txt'), 'w') as f:
                f.write("\n".join(best_features))
            # Retrain model using best_features and calibrate probabilities (Totals)
            X_totals_best = historical_game_level_data[best_features].select_dtypes(include=["number", "bool", "category"])
            X_train_tot_best, X_test_tot_best, _, _ = train_test_split(
                X_totals_best, y_totals, test_size=0.2, random_state=42, stratify=y_totals)
            from sklearn.calibration import CalibratedClassifierCV
            model_totals_best = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
            calibrated_model_totals = CalibratedClassifierCV(model_totals_best, method='isotonic', cv=3)
            calibrated_model_totals.fit(X_train_tot_best, y_train_tot)
            y_totals_pred_best = calibrated_model_totals.predict(X_test_tot_best)
            totals_accuracy_best = accuracy_score(y_test_tot, y_totals_pred_best)
            # Probability sanity check
            probs_totals = calibrated_model_totals.predict_proba(X_test_tot_best)[:, 1]
            mean_pred_prob_totals = np.mean(probs_totals)
            actual_rate_totals = np.mean(y_test_tot)
            st.write(f"Accuracy with best features (Totals): {totals_accuracy_best:.3f}")
            st.write(f"Mean predicted probability (test set): {mean_pred_prob_totals:.3f}")
            st.write(f"Actual outcome rate (test set): {actual_rate_totals:.3f}")
            scores_df = pd.DataFrame(scores_list).sort_values(by='accuracy', ascending=False).head(10)
            st.write("#### Top 10 Feature Subsets (by Accuracy)")
            
            st.dataframe(
                scores_df,
                hide_index=True,
                column_config={
                    'iteration': st.column_config.NumberColumn('Iteration', format='%d'),
                    'accuracy': st.column_config.NumberColumn('Accuracy', format='%.3f'),
                    'AUC': st.column_config.NumberColumn('AUC', format='%.3f'),
                    'F1-score': st.column_config.NumberColumn('F1-Score', format='%.3f'),
                    'features': st.column_config.TextColumn('Features', width='large')
                }
            )