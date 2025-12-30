#!/usr/bin/env python3
"""Preview the email HTML format without sending.

Generates the HTML email body and saves it to a file for browser preview.
"""
import os
import sys
from datetime import datetime

# Make repo root importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import pandas as pd
    from emailer import build_rich_html_email
except Exception as e:
    print('Missing dependency or import error:', e)
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
        send_df = preds.tail(10)
        print("No upcoming games found, showing last 10 games as preview")
    else:
        send_df = upcoming.head(20)
        print(f"Found {len(upcoming)} upcoming games, showing top 20")
except Exception:
    send_df = preds.head(20)
    print("Error filtering upcoming games, showing first 20 rows")

# Build rich HTML body
logos_dir = os.path.join(ROOT, 'data_files', 'logos') if os.path.exists(os.path.join(ROOT, 'data_files', 'logos')) else None
html_body = build_rich_html_email(send_df, logos_dir=logos_dir, max_rows=20)

# Wrap in full HTML document for browser preview
full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NFL Predictions Email Preview</title>
    <style>
        body {{
            font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
            max-width: 1000px;
            margin: 20px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .preview-header {{
            background: #fff;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
        }}
        .email-container {{
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="preview-header">
        <h2 style="margin:0 0 10px 0">📧 Email Preview</h2>
        <p style="margin:0;color:#666">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p style="margin:5px 0 0 0;color:#666">Games shown: {len(send_df)}</p>
    </div>
    <div class="email-container">
        {html_body}
    </div>
</body>
</html>
"""

# Save to file
output_path = os.path.join(ROOT, 'data_files', 'email_preview.html')
try:
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"\n✅ Preview saved to: {output_path}")
    print(f"   Open this file in your browser to see the email format\n")
    
    # Try to open in default browser (optional)
    try:
        import webbrowser
        webbrowser.open(f'file://{output_path}')
        print("   Opened in default browser")
    except Exception:
        pass
        
except Exception as e:
    print(f'Failed to save preview: {e}')
    sys.exit(3)
