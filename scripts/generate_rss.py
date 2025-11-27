#!/usr/bin/env python3
"""
Simple RSS feed generator for NFL Predictions
Reads `data_files/betting_recommendations_log.csv` (or predictions file) and
writes `data_files/alerts_feed.xml` (RSS 2.0) containing upcoming/pending recommendations.

Usage:
    python scripts/generate_rss.py

This script is intentionally lightweight and does not publish the feed; it only
writes the XML file to `data_files/alerts_feed.xml` so you can validate and
choose a hosting method later (GitHub Pages, S3, app route, etc.).

"""
import os
import sys
import datetime
from xml.sax.saxutils import escape
from urllib.parse import urlparse, urljoin

try:
    import pandas as pd
except Exception as e:
    print("ERROR: pandas is required to run this script. Install with `pip install pandas`.")
    raise

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data_files")
LOG_PATH = os.path.join(DATA_DIR, "betting_recommendations_log.csv")
OUT_PATH = os.path.join(DATA_DIR, "alerts_feed.xml")
# If you later publish to GitHub Pages, set SITE_URL to your GH pages URL
# SITE_URL default; can be overridden by ALERTS_SITE_URL env var or app_config.json
SITE_URL = os.environ.get("ALERTS_SITE_URL", "http://localhost:8501/")

# If the Streamlit app saved a detected base URL to `data_files/app_config.json`, prefer that
try:
    cfg_path = os.path.join(DATA_DIR, 'app_config.json')
    if os.path.exists(cfg_path):
        import json as _json
        with open(cfg_path, 'r', encoding='utf-8') as cf:
            cfg = _json.load(cf)
            if isinstance(cfg, dict) and cfg.get('app_base_url'):
                SITE_URL = cfg.get('app_base_url')
except Exception:
    pass
# Path under SITE_URL where individual alert pages will live. Example: '/alerts/'
ALERTS_ITEM_PATH = os.environ.get("ALERTS_ITEM_PATH", "/alerts/")


def _normalize_base_url(site_url):
    """Return a safe absolute http(s) base URL.

    If the provided site_url has no http/https scheme, attempt to fall back
    to environment `ALERTS_SITE_URL` or to a localhost default.
    """
    try:
        parsed = urlparse(site_url or "")
        if parsed.scheme in ("http", "https") and parsed.netloc:
            return site_url.rstrip('/') + '/'
    except Exception:
        pass

    # Try environment override
    env = os.environ.get("ALERTS_SITE_URL")
    if env:
        p = urlparse(env)
        if p.scheme in ("http", "https") and p.netloc:
            return env.rstrip('/') + '/'

    # Fallback to localhost with port 8501 (Streamlit default)
    return "http://localhost:8501/"


def _choose_confidence(row):
    """Pick the most relevant confidence/probability value from available columns."""
    for col in ("confidence", "model_probability", "prob_underdogWon", "prob_underdogCovered", "prob_overHit"):
        if col in row and pd.notna(row[col]):
            try:
                return float(row[col])
            except Exception:
                # could be string like '0.65' or '65%'
                try:
                    return float(str(row[col]).replace('%', ''))
                except Exception:
                    continue
    return None


def _format_confidence(conf):
    """Format confidence like the app: one decimal percent (e.g. 25.8%)."""
    if conf is None:
        return "N/A"
    try:
        # If value looks like a fraction between 0 and 1, format as percent
        f = float(conf)
        if 0 <= f <= 1:
            return f"{f:.1%}"
        # If value looks like 0-100 scale already, format with one decimal and %
        return f"{f:.1f}%"
    except Exception:
        return str(conf)


def _format_edge(edge):
    """Format edge with one decimal like the app (e.g. 1.2 or -0.3)."""
    if edge is None or edge == "":
        return "N/A"
    try:
        return f"{float(edge):.1f}"
    except Exception:
        return str(edge)


def build_rss_items(df, max_items=100):
    items = []
    for _, r in df.head(max_items).iterrows():
        away = r.get("away_team", "Away")
        home = r.get("home_team", "Home")
        bet_type = r.get("bet_type", "recommendation")
        title = f"{away} @ {home} — {bet_type}"
        title = escape(title)

        game_id = r.get("game_id", "")
        guid = escape(f"{game_id}-{bet_type}") if game_id != "" else escape(str(hash(title)))

        # pubDate: use gameday if available, else now
        pub = None
        if "gameday" in r and pd.notna(r.get("gameday")):
            try:
                pub_dt = pd.to_datetime(r.get("gameday"))
                pub = pub_dt.strftime("%a, %d %b %Y %H:%M:%S GMT")
            except Exception:
                pub = None
        if pub is None:
            pub = datetime.datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")

        # Description: include confidence + short summary
        confidence = _choose_confidence(r)
        conf_text = f"Confidence: {_format_confidence(confidence)}" if confidence is not None else ""
        extra = []
        if "spread_line" in r and pd.notna(r.get("spread_line")):
            extra.append(f"Spread: {r.get('spread_line')}")
        if "total_line" in r and pd.notna(r.get("total_line")):
            extra.append(f"Total: {r.get('total_line')}")
        if "edge" in r and pd.notna(r.get("edge")):
            extra.append(f"Edge: {_format_edge(r.get('edge'))}")

        description = " | ".join([conf_text] + extra).strip(" |")
        description = escape(description)

        # Use query-param links so the Streamlit app can read ?alert=<guid>
        base = _normalize_base_url(SITE_URL)
        # ALERTS_ITEM_PATH may be something like '/alerts/' or '/'
        path = ALERTS_ITEM_PATH or '/'
        if not path.startswith('/'):
            path = '/' + path
        # Build absolute URL for the alert page and append query param
        alert_base = urljoin(base, path)
        if not alert_base.endswith('/'):
            alert_base = alert_base + '/'
        item_link = f"{alert_base}?alert={guid}"

        item = (
            f"<item>"
            f"<title>{title}</title>"
            f"<link>{escape(item_link)}</link>"
            f"<guid isPermaLink=\"false\">{guid}</guid>"
            f"<pubDate>{pub}</pubDate>"
            f"<description>{description}</description>"
            f"</item>"
        )
        items.append(item)
    return "\n".join(items)


def generate_feed():
    if not os.path.exists(LOG_PATH):
        print(f"No betting log found at {LOG_PATH}. Make sure predictions/log exist.")
        return 1

    try:
        df = pd.read_csv(LOG_PATH, low_memory=False)
    except Exception as e:
        print(f"Failed to read {LOG_PATH}: {e}")
        return 2

    # Prefer pending/upcoming bets if column exists
    if "bet_result" in df.columns:
        df = df[df["bet_result"].isin(["pending", "recommended"]) | df["bet_result"].isna()]

    # Optionally filter for only high-confidence recommendations (uncomment to enable)
    # df = df[df.apply(lambda r: (_choose_confidence(r) or 0) >= 0.6, axis=1)]

    # Sort by gameday if present, else keep original order
    if "gameday" in df.columns:
        try:
            df = df.sort_values(by="gameday")
        except Exception:
            pass

    items_xml = build_rss_items(df, max_items=200)

    now = datetime.datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
    rss = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<title>NFL Predictions Alerts</title>
<link>{escape(SITE_URL)}</link>
<description>Latest high-confidence betting recommendations from NFL Predictions</description>
<lastBuildDate>{now}</lastBuildDate>
{items_xml}
</channel></rss>
"""

    try:
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            f.write(rss)
        print(f"Wrote RSS feed to: {OUT_PATH}")
        return 0
    except Exception as e:
        print(f"Failed to write feed to {OUT_PATH}: {e}")
        return 3


if __name__ == "__main__":
    rc = generate_feed()
    sys.exit(rc)
