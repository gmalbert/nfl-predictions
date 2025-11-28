import os
import sys
from io import BytesIO
from datetime import datetime

import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_files')
CSV_PATH = os.path.join(DATA_DIR, 'nfl_games_historical_with_predictions.csv')
EXPORTS_DIR = os.path.join(DATA_DIR, 'exports')


def generate_predictions_pdf(df: pd.DataFrame, max_rows: int = 200) -> bytes:
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

    header = ['Gameday', 'Home', 'Away', 'Spread', 'Total', 'Home ML', 'Away ML', 'P(Cover)', 'P(Win)', 'P(Over)']
    table_data = [header]

    for _, row in df.head(max_rows).iterrows():
        try:
            gameday_val = row.get('gameday', '')
            gameday = ''
            if pd.notna(gameday_val):
                try:
                    dt = pd.to_datetime(gameday_val)
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
            continue

    col_widths = [1.6 * inch, 1.6 * inch, 1.6 * inch, 0.7 * inch, 0.7 * inch, 0.9 * inch, 0.9 * inch, 0.7 * inch, 0.7 * inch, 0.7 * inch]
    table = Table(table_data, colWidths=col_widths)

    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2f4f4f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7f7f7')])
    ])
    table.setStyle(table_style)

    story.append(table)

    try:
        doc.build(story)
    except Exception as e:
        print('PDF generation failed:', e)
        return b''

    buffer.seek(0)
    return buffer.getvalue()


def main():
    if not os.path.exists(CSV_PATH):
        print('Predictions CSV not found at', CSV_PATH)
        sys.exit(2)

    df = pd.read_csv(CSV_PATH, sep='\t')
    if df is None or df.empty:
        print('Predictions CSV is empty')
        sys.exit(3)

    os.makedirs(EXPORTS_DIR, exist_ok=True)
    pdf_bytes = generate_predictions_pdf(df)
    if not pdf_bytes:
        print('PDF generation returned no data')
        sys.exit(4)

    out_path = os.path.join(EXPORTS_DIR, f'smoke_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
    with open(out_path, 'wb') as f:
        f.write(pdf_bytes)

    print('PDF written to', out_path)


if __name__ == '__main__':
    main()
