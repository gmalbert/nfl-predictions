import os
import smtplib
from email.message import EmailMessage
from typing import List, Optional


def send_predictions_email(
    smtp_server: str,
    smtp_port: int,
    username: str,
    password: str,
    from_addr: str,
    to_addrs: List[str],
    subject: str,
    html_body: str,
    attachment_bytes: Optional[bytes] = None,
    attachment_name: str = "predictions.csv",
):
    """Send an email with HTML body and optional CSV attachment using SMTP.

    Uses STARTTLS by default (port 587) which works with Gmail App Passwords.
    """
    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)
    msg["Subject"] = subject
    # Provide a useful plain-text fallback generated from the HTML so clients
    # that hide HTML by default still display readable content without needing
    # a "view entire message" click. Keep HTML alternative for rich clients.
    def _html_to_text(h: str) -> str:
        try:
            import re
            text = h
            # Replace tags that imply breaks with newlines
            text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
            text = re.sub(r"</tr>|</div>|</p>|</table>|</h[1-6]>", "\n", text, flags=re.I)
            text = re.sub(r"</t[dh]>", "\t", text, flags=re.I)
            # Remove remaining tags
            text = re.sub(r"<[^>]+>", "", text)
            # Unescape common HTML entities
            text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            # Collapse multiple blank lines
            text = re.sub(r"\n\s*\n+", "\n\n", text)
            return text.strip()
        except Exception:
            return "Predictions attached as CSV."

    plain = _html_to_text(html_body)
    msg.set_content(plain)
    msg.add_alternative(html_body, subtype="html")

    if attachment_bytes is not None:
        msg.add_attachment(attachment_bytes, maintype="text", subtype="csv", filename=attachment_name)

    # Connect and send
    with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(username, password)
        server.send_message(msg)


def build_simple_html_table(df, max_rows: int = 20) -> str:
    """Build a simple HTML table for the top `max_rows` of a DataFrame.

    Keeps the produced HTML compact to work in email clients.
    """
    try:
        import pandas as pd
    except Exception:
        return "<p>Predictions attached as CSV.</p>"


def build_rich_html_email(df, logos_dir=None, max_rows: int = 20) -> str:
    """Build a richer HTML email body with inline logos, colored confidence badges, and compact stats.

    - `df` should contain `home_team` and `away_team` abbreviations (e.g., 'KC', 'NYJ').
    - Looks for logo files at `logos_dir/{ABBR}.png` if provided and embeds them as data URIs.
    - Formats probability columns (`prob_underdogWon`, `prob_underdogCovered`, `prob_overHit`) as percentages.
    """
    try:
        import base64
        import io
        import pandas as pd
    except Exception:
        return build_simple_html_table(df, max_rows=max_rows)

    if df is None or df.empty:
        return "<p>No predictions to display.</p>"

    rows = []
    dfp = df.head(max_rows).copy()
    # Ensure gameday string
    if 'gameday' in dfp.columns:
        try:
            dfp['gameday_dt'] = pd.to_datetime(dfp['gameday'], errors='coerce')
            # Format dates: drop time when it's exactly midnight (00:00:00)
            def _fmt_gameday(dt):
                try:
                    if pd.isna(dt):
                        return ''
                    # dt may be a pandas Timestamp
                    h = getattr(dt, 'hour', None)
                    m = getattr(dt, 'minute', None)
                    s = getattr(dt, 'second', None)
                    if h == 0 and (m == 0 or m is None) and (s == 0 or s is None):
                        return dt.strftime('%Y-%m-%d')
                    return dt.strftime('%Y-%m-%d %H:%M')
                except Exception:
                    return str(dt)

            dfp['gameday_str'] = dfp['gameday_dt'].apply(_fmt_gameday)
        except Exception:
            dfp['gameday_str'] = dfp['gameday'].astype(str)
    else:
        dfp['gameday_str'] = ''

    def badge_html(prob):
        try:
            p = float(prob)
        except Exception:
            return '<span style="background:#ddd;color:#000;padding:2px 6px;border-radius:4px;font-size:12px">N/A</span>'
        # Tier colors
        if p >= 0.65:
            color = '#ff6b00'  # orange/red for Elite
            label = 'Elite'
        elif p >= 0.60:
            color = '#9b59b6'  # purple for Strong
            label = 'Strong'
        elif p >= 0.55:
            color = '#3498db'  # blue for Good
            label = 'Good'
        else:
            color = '#95a5a6'  # gray for Standard
            label = 'Standard'
        return f'<span style="background:{color};color:#fff;padding:4px 8px;border-radius:6px;font-weight:600;font-size:12px">{label}</span>'

    def embed_logo(abbr):
        # Logos are intentionally disabled for email compatibility and to
        # keep message size small. Return empty string to avoid embedding.
        return ''

    # Build rows
    for _, r in dfp.iterrows():
        gameday = r.get('gameday_str', '')
        home = str(r.get('home_team', '')).upper()
        away = str(r.get('away_team', '')).upper()
        home_logo = embed_logo(home)
        away_logo = embed_logo(away)
        spread = r.get('spread_line', '')
        total = r.get('total_line', '')
        prob_ml = r.get('prob_underdogWon', None)
        prob_sp = r.get('prob_underdogCovered', None)
        prob_ov = r.get('prob_overHit', None)

        prob_ml_s = f"{prob_ml:.2%}" if pd.notna(prob_ml) and isinstance(prob_ml, (int, float)) else ''
        prob_sp_s = f"{prob_sp:.2%}" if pd.notna(prob_sp) and isinstance(prob_sp, (int, float)) else ''
        prob_ov_s = f"{prob_ov:.2%}" if pd.notna(prob_ov) and isinstance(prob_ov, (int, float)) else ''

        # Choose a confidence metric (use max of the three probabilities if present)
        conf_val = None
        for candidate in (prob_ml, prob_sp, prob_ov):
            try:
                if candidate is not None:
                    v = float(candidate)
                    if conf_val is None or v > conf_val:
                        conf_val = v
            except Exception:
                continue

        badge = badge_html(conf_val if conf_val is not None else 0)

        # Small colored marker spans for home/away (keep logos disabled)
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

        home_color = team_colors.get(home, '#6B7280')
        away_color = team_colors.get(away, '#6B7280')
        home_marker = (
            f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;"
            f"background:{home_color};margin-right:8px;vertical-align:middle;border:1px solid rgba(0,0,0,0.08)'></span>"
        )
        away_marker = (
            f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;"
            f"background:{away_color};margin-right:8px;vertical-align:middle;border:1px solid rgba(0,0,0,0.08)'></span>"
        )

        row_html = f"""
        <tr style="border-bottom:1px solid #eee;">
          <td style="padding:8px 6px;vertical-align:middle;white-space:nowrap">{gameday}</td>
          <td style="padding:8px 6px;vertical-align:middle;text-align:left">
            {home_marker}
            <strong>{home}</strong>
          </td>
          <td style="padding:8px 6px;vertical-align:middle;text-align:center">@</td>
          <td style="padding:8px 6px;vertical-align:middle;text-align:left">
            {away_marker}
            <strong>{away}</strong>
          </td>
          <td style="padding:8px 6px;vertical-align:middle;text-align:right">Spread: {spread}</td>
          <td style="padding:8px 6px;vertical-align:middle;text-align:right">Total: {total}</td>
          <td style="padding:8px 6px;vertical-align:middle;text-align:right">ML: {prob_ml_s}<br>SP: {prob_sp_s}<br>OV: {prob_ov_s}</td>
          <td style="padding:8px 6px;vertical-align:middle;text-align:right">{badge}</td>
        </tr>
        """

        rows.append(row_html)

    # Assemble final HTML
    table_html = """
    <div style="font-family:system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; color:#111">
        <h3 style="margin-bottom:6px">NFL Predictions</h3>
        <table style="width:100%;border-collapse:collapse">{rows_html}</table>
        <p style="font-size:12px;color:#666;margin-top:8px">Generated by GridIron Oracle — predictions are model outputs, not financial advice.</p>
    </div>
    """.format(rows_html='\n'.join(rows))

    return table_html

    if df is None or df.empty:
        return "<p>No predictions to display.</p>"

    try:
        pdf = df.head(max_rows).copy()
        # Reduce column set for compactness if too many columns
        if pdf.shape[1] > 10:
            pdf = pdf.iloc[:, :10]
        html_table = pdf.to_html(index=False, justify='center', border=0)
        return f"<div><p>Top {min(len(df), max_rows)} predictions:</p>{html_table}</div>"
    except Exception:
        return "<p>Predictions attached as CSV.</p>"
