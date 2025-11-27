#!/usr/bin/env python3
"""Send a rich HTML predictions email now using project files and env vars.

This script is a one-off helper: it reads `.env`, loads predictions CSV,
filters for upcoming games, builds a rich HTML body (with embedded logos),
and sends the email via `emailer.send_predictions_email`.

It prints only success/failure messages and never echoes secrets.
"""
import os
import sys
import io
from datetime import datetime

# Load .env if python-dotenv is available, otherwise try a minimal loader
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    try:
        with open(env_path, 'r', encoding='utf-8') as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                k = k.strip(); v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass

# Make repo root importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import pandas as pd
    from emailer import build_rich_html_email, send_predictions_email
except Exception as e:
    print('Missing dependency or import error:', e)
    sys.exit(2)

# Read env vars
smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
smtp_port = int(os.getenv('SMTP_PORT', '587'))
email_from = os.getenv('EMAIL_FROM')
email_to = os.getenv('EMAIL_TO')
email_pass = os.getenv('EMAIL_PASSWORD')

missing = [k for k in ('EMAIL_FROM', 'EMAIL_TO', 'EMAIL_PASSWORD') if not os.getenv(k)]
if missing:
    print('Missing required environment variables:', ', '.join(missing))
    sys.exit(2)

# Load predictions CSV
csv_path = os.path.join(ROOT, 'data_files', 'nfl_games_historical_with_predictions.csv')
if not os.path.exists(csv_path):
    print('Predictions CSV not found at', csv_path)
    sys.exit(2)

try:
    preds = pd.read_csv(csv_path, sep='\t')
except Exception as e:
    print('Failed to read predictions CSV:', e)
    sys.exit(2)

# Filter upcoming games
try:
    preds['gameday_dt'] = pd.to_datetime(preds.get('gameday', None), errors='coerce')
    now_dt = pd.to_datetime(datetime.now())
    upcoming = preds[preds['gameday_dt'] >= now_dt].sort_values('gameday_dt', ascending=True)
    if upcoming.empty:
        # fallback to latest rows
        send_df = preds.head(50)
    else:
        send_df = upcoming.head(50)
except Exception:
    send_df = preds.head(50)

# Build rich HTML body
logos_dir = os.path.join(ROOT, 'data_files', 'logos') if os.path.exists(os.path.join(ROOT, 'data_files', 'logos')) else None
html_body = build_rich_html_email(send_df, logos_dir=logos_dir, max_rows=50)

# Prepare CSV attachment bytes
csv_bytes = send_df.to_csv(index=False).encode('utf-8')

recipients = [a.strip() for a in email_to.split(',') if a.strip()]
subject = f"NFL Predictions — {datetime.now().strftime('%Y-%m-%d')}"

try:
    send_predictions_email(
        smtp_server=smtp_server,
        smtp_port=smtp_port,
        username=email_from,
        password=email_pass,
        from_addr=email_from,
        to_addrs=recipients,
        subject=subject,
        html_body=html_body,
        attachment_bytes=csv_bytes,
        attachment_name='predictions_upcoming.csv',
    )
    print('Rich email sent successfully to:', ', '.join(recipients))
    sys.exit(0)
except Exception as e:
    print('Error sending rich email:', e)
    sys.exit(3)
