#!/usr/bin/env python3
"""
Render `data_files/alerts_feed.xml` as a simple, styled HTML page for quick local preview.

Usage:
    python scripts/preview_rss.py

This writes `data_files/alerts_feed_preview.html` and prints its path. Open that file
in your browser to see how the feed would look in a reader.
"""
from pathlib import Path
import sys
import html
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_files"
FEED_PATH = DATA_DIR / "alerts_feed.xml"
OUT_PATH = DATA_DIR / "alerts_feed_preview.html"


def parse_feed(path: Path):
    if not path.exists():
        print(f"Feed not found at: {path}")
        return None
    try:
        tree = ET.parse(path)
        return tree.getroot()
    except Exception as e:
        print(f"Failed to parse XML: {e}")
        return None


def elt_text(elt, tag):
    child = elt.find(tag)
    return child.text if child is not None and child.text is not None else ""


def build_html(root: ET.Element):
    channel = root.find('channel')
    title = elt_text(channel, 'title')
    link = elt_text(channel, 'link')
    description = elt_text(channel, 'description')
    last_build = elt_text(channel, 'lastBuildDate')

    items = []
    for item in channel.findall('item'):
        items.append({
            'title': elt_text(item, 'title'),
            'link': elt_text(item, 'link'),
            'guid': elt_text(item, 'guid'),
            'pubDate': elt_text(item, 'pubDate'),
            'description': elt_text(item, 'description'),
        })

    safe = html.escape

    css = """
    body{font-family:Segoe UI, Roboto, Arial; background:#f7f9fc; color:#111;}
    .container{max-width:900px;margin:32px auto;padding:18px;background:#fff;border-radius:8px;box-shadow:0 8px 24px rgba(17,24,39,0.08);}
    h1{margin:0 0 6px;font-size:20px}
    .meta{color:#556; font-size:13px; margin-bottom:14px}
    .item{padding:12px 16px;border-top:1px solid #eef2f7}
    .item:first-of-type{border-top:0}
    .item h2{margin:0;font-size:16px}
    .item .desc{color:#333;margin-top:6px}
    .item .meta{color:#667; font-size:12px}
    .link{color:#0b69ff;text-decoration:none}
    """

    parts = []
    parts.append(f"<html><head><meta charset=\"utf-8\"><title>{safe(title or 'Feed Preview')}</title><style>{css}</style></head><body>")
    parts.append(f"<div class=\"container\"><h1>{safe(title)}</h1><div class=\"meta\">{safe(description)}")
    if link:
        parts.append(f" — <a class=\"link\" href=\"{safe(link)}\">Source</a>")
    if last_build:
        parts.append(f"<div class=\"meta\">Last build: {safe(last_build)}</div>")
    parts.append("</div>")

    if not items:
        parts.append("<p>No items found in feed.</p>")
    else:
        for it in items:
            parts.append('<div class="item">')
            parts.append(f"<h2><a class=\"link\" href=\"{safe(it['link']) or '#'}\">{safe(it['title'])}</a></h2>")
            parts.append(f"<div class=\"meta\">{safe(it['pubDate'])} — GUID: {safe(it['guid'])}</div>")
            if it['description']:
                parts.append(f"<div class=\"desc\">{safe(it['description'])}</div>")
            parts.append('</div>')

    parts.append('</div></body></html>')
    return '\n'.join(parts)


def main():
    root = parse_feed(FEED_PATH)
    if root is None:
        sys.exit(1)
    html_text = build_html(root)
    try:
        with open(OUT_PATH, 'w', encoding='utf-8') as f:
            f.write(html_text)
        print(f"Wrote preview to: {OUT_PATH}")
        print("Open this file in your browser to view the feed preview.")
    except Exception as e:
        print(f"Failed to write preview: {e}")
        sys.exit(2)


if __name__ == '__main__':
    main()
