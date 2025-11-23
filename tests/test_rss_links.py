import os
import xml.etree.ElementTree as ET
import requests
try:
    import pytest
except Exception:
    pytest = None
from pathlib import Path
from urllib.parse import urlparse
import sys


def get_rss_file_path():
    # repo root is two levels up from this tests file
    repo_root = Path(__file__).resolve().parents[1]
    rss_path = repo_root / "data_files" / "alerts_feed.xml"
    return rss_path


def load_links(rss_path):
    if not rss_path.exists():
        if pytest:
            pytest.skip(f"RSS feed not found at: {rss_path}")
        else:
            print(f"SKIPPED: RSS feed not found at: {rss_path}")
            sys.exit(0)

    tree = ET.parse(rss_path)
    root = tree.getroot()

    # Try common RSS structures: <rss><channel><item><link>
    links = []
    for item in root.findall('.//item'):
        link = item.find('link')
        if link is not None and link.text:
            links.append(link.text.strip())

    # Fallback: any <link> elements
    if not links:
        for link in root.findall('.//link'):
            if link.text:
                links.append(link.text.strip())

    return links


def is_absolute_http(url):
    p = urlparse(url)
    return p.scheme in ("http", "https") and p.netloc


def test_rss_links_resolve():
    """Parse the RSS feed and verify each link returns a 2xx/3xx HTTP status.

    If the target app is not running (connection error), this test will be skipped.
    """
    rss_path = get_rss_file_path()
    links = load_links(rss_path)

    if not links:
        if pytest:
            pytest.skip("No links found in RSS feed; skipping link resolution test.")
        else:
            print("SKIPPED: No links found in RSS feed; skipping link resolution test.")
            sys.exit(0)

    failed = []
    for link in links:
        if not is_absolute_http(link):
            if pytest:
                raise AssertionError(f"Link is not an absolute http(s) URL: {link}")
            else:
                print(f"FAILED: Link is not an absolute http(s) URL: {link}")
                sys.exit(2)

        try:
            resp = requests.head(link, allow_redirects=True, timeout=5)
        except requests.exceptions.RequestException as e:
            if pytest:
                pytest.skip(f"Network/connectivity issue while checking '{link}': {e}")
            else:
                print(f"SKIPPED: Network/connectivity issue while checking '{link}': {e}")
                sys.exit(0)

        # Some servers don't allow HEAD and return 405; retry with GET in that case
        status = resp.status_code
        if status == 405:
            try:
                resp2 = requests.get(link, allow_redirects=True, timeout=7)
                status = resp2.status_code
            except requests.exceptions.RequestException as e:
                if pytest:
                    pytest.skip(f"Network/connectivity issue while checking '{link}' (GET retry): {e}")
                else:
                    print(f"SKIPPED: Network/connectivity issue while checking '{link}' (GET retry): {e}")
                    sys.exit(0)

        if status >= 400:
            failed.append((link, status))

    if failed:
        msgs = [f"{u} -> {code}" for u, code in failed]
        if pytest:
            pytest.fail("Some RSS links returned error status: " + "; ".join(msgs))
        else:
            print("FAILED: Some RSS links returned error status: " + "; ".join(msgs))
            sys.exit(2)


if __name__ == '__main__':
    # Allow running directly: print results and exit with non-zero on failure
    try:
        test_rss_links_resolve()
    except SystemExit as s:
        # test function uses sys.exit for skip/fail cases
        code = s.code if isinstance(s.code, int) else 1
        if code == 0:
            print("SKIPPED or OK: exiting with code 0")
        elif code == 2:
            print("FAILED: one or more links invalid or returned error")
        else:
            print(f"Exited with code {code}")
        sys.exit(code)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(2)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(3)
    print("All RSS links resolved.")
    sys.exit(0)
