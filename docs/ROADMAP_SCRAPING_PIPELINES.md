# Roadmap: Scraping Pipelines

> Inspired by [thadhutch/sports-quant](https://github.com/thadhutch/sports-quant) PFF and PFR scrapers.

## Overview

The sports-quant project uses two Selenium-based scrapers to collect NFL data not available through public APIs:

1. **PFF Scraper**: Authenticated scraping of PFF Premium team grades (requires subscription)
2. **PFR Scraper**: Proxy-rotated scraping of Pro Football Reference boxscore data (free, rate-limited)

This document provides production-ready code adapted for our project structure.

> **Important**: Scraping should comply with each site's Terms of Service. PFF requires a paid subscription. PFR is publicly available but rate-limited. Always use respectful delays between requests.

---

## 1. PFR Boxscore URL Collector

### Purpose
Collect all boxscore URLs from Pro Football Reference for specified seasons. These URLs are then used by the PFR game data scraper.

### Implementation

Create `scripts/scrapers/pfr_urls.py`:

```python
"""Gather NFL boxscore URLs from Pro Football Reference."""

import logging
import random
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# PFR weekly schedule page pattern
BASE_URL = "https://www.pro-football-reference.com/years/{}/week_{}.htm"

# Output file
OUTPUT_FILE = Path("data_files/pfr_boxscore_urls.txt")


def get_boxscores_for_week(
    year: int,
    week: int,
    proxies: list[dict] | None = None,
    timeout: int = 10,
) -> list[str]:
    """Extract boxscore URLs for a given year and week.

    Returns list of full boxscore URLs like:
    https://www.pro-football-reference.com/boxscores/202409050kan.htm
    """
    url = BASE_URL.format(year, week)
    kwargs = {"timeout": timeout}
    if proxies:
        kwargs["proxies"] = random.choice(proxies)

    try:
        response = requests.get(url, **kwargs)
        if response.status_code != 200:
            logger.warning("HTTP %d for %s", response.status_code, url)
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        boxscores = []

        # PFR game links are in <td class="right gamelink"> elements
        for gamelink in soup.find_all("td", class_="right gamelink"):
            a_tag = gamelink.find("a")
            if a_tag and "href" in a_tag.attrs:
                boxscore_url = "https://www.pro-football-reference.com" + a_tag["href"]
                boxscores.append(boxscore_url)

        logger.info("Year %d Week %d: found %d boxscores", year, week, len(boxscores))
        return boxscores

    except requests.RequestException as e:
        logger.error("Error fetching %s: %s", url, e)
        return []


def collect_all_boxscore_urls(
    start_year: int = 2020,
    end_year: int = 2025,
    max_week_current: int = 18,
    delay: float = 2.0,
    proxies: list[dict] | None = None,
) -> list[str]:
    """Collect boxscore URLs for all seasons and weeks.

    Args:
        start_year: First season to scrape
        end_year: Last season to scrape
        max_week_current: Max week for current/last season (regular season = 18)
        delay: Seconds between requests (be respectful)
        proxies: Optional list of proxy dicts for requests
    """
    all_urls = []

    for year in range(start_year, end_year + 1):
        max_weeks = max_week_current if year == end_year else 22  # 18 reg + 4 post
        for week in range(1, max_weeks + 1):
            urls = get_boxscores_for_week(year, week, proxies)
            all_urls.extend(urls)
            time.sleep(delay)  # Respectful delay

    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for url in all_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    logger.info("Total unique boxscore URLs: %d", len(unique_urls))
    return unique_urls


def save_urls(urls: list[str], output_path: Path = OUTPUT_FILE) -> None:
    """Save URLs to a text file, one per line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for url in urls:
            f.write(url + "\n")
    logger.info("Saved %d URLs to %s", len(urls), output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    urls = collect_all_boxscore_urls(start_year=2020, end_year=2025)
    save_urls(urls)
```

---

## 2. PFR Game Data Scraper (Proxy-Rotated Selenium)

### Purpose
Scrape game metadata (Roof, Surface, Vegas Line, Over/Under) from each PFR boxscore page.

### Why Selenium?
PFR uses JavaScript rendering and anti-bot measures. Simple `requests` calls may miss content or get blocked. The sports-quant approach uses:
- `selenium-wire` for proxy support
- Random user-agent rotation
- WebDriver property spoofing
- Auto-scrolling for lazy-loaded content

### Dependencies

```
# Add to requirements.txt
selenium>=4.15.0
selenium-wire>=5.1.0
beautifulsoup4>=4.12.0
webdriver-manager>=4.0.0
```

### Proxy Configuration

Create `data_files/proxies.csv`:

```csv
proxy_url,proxy_auth
us-proxy1.example.com:8080,username:password
us-proxy2.example.com:8080,username:password
```

> **Proxy sources**: Services like BrightData, Oxylabs, or SmartProxy offer residential proxies. Free proxies are unreliable and often blocked. Budget ~$10-20/month for a small proxy pool.

### Implementation

Create `scripts/scrapers/pfr_scraper.py`:

```python
"""Scrape Pro Football Reference boxscores for game metadata using proxied Selenium."""

import csv
import logging
import random
import time
from pathlib import Path

from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logger = logging.getLogger(__name__)

# Output paths
PROXY_FILE = Path("data_files/proxies.csv")
URLS_FILE = Path("data_files/pfr_boxscore_urls.txt")
OUTPUT_FILE = Path("data_files/pfr_game_data.csv")

# User agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]


def load_proxies(proxy_file: Path = PROXY_FILE) -> list[tuple[str, str]]:
    """Load proxies from CSV. Returns list of (proxy_url, proxy_auth) tuples."""
    proxies = []
    with open(proxy_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            proxies.append((row["proxy_url"], row["proxy_auth"]))
    logger.info("Loaded %d proxies", len(proxies))
    return proxies


def configure_driver(proxy_url: str | None = None, proxy_auth: str | None = None):
    """Create a Chrome WebDriver with optional proxy and anti-detection measures.

    Uses selenium-wire if proxy is provided, otherwise plain selenium.
    """
    options = Options()
    options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("window-size=1280x800")
    options.add_argument("--disable-gpu")
    options.add_argument("--headless=new")  # Run headless for automation

    if proxy_url and proxy_auth:
        # Use selenium-wire for proxy with auth
        from seleniumwire import webdriver as sw_webdriver

        proxy_user, proxy_pass = proxy_auth.split(":", 1)
        proxy_options = {
            "proxy": {
                "http": f"http://{proxy_user}:{proxy_pass}@{proxy_url}",
                "https": f"https://{proxy_user}:{proxy_pass}@{proxy_url}",
                "no_proxy": "localhost,127.0.0.1",
            },
            "verify_ssl": False,
        }
        driver = sw_webdriver.Chrome(seleniumwire_options=proxy_options, options=options)
    else:
        from selenium import webdriver
        driver = webdriver.Chrome(options=options)

    # Spoof WebDriver property to avoid detection
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )

    return driver


def auto_scroll(driver, pause: float = 1.0) -> None:
    """Scroll to page bottom to trigger lazy-loaded content."""
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


def scrape_game_info(url: str, proxies: list[tuple[str, str]]) -> dict | None:
    """Extract game info from a single PFR boxscore URL.

    Returns dict with keys: title, roof, surface, vegas_line, over_under
    """
    proxy_url, proxy_auth = random.choice(proxies) if proxies else (None, None)
    driver = None

    try:
        driver = configure_driver(proxy_url, proxy_auth)
        driver.get(url)
        auto_scroll(driver)

        # Wait for content
        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, "content"))
        )
        content_div = driver.find_element(By.ID, "content")

        # Game title (e.g., "Kansas City Chiefs at Baltimore Ravens")
        h1_tag = WebDriverWait(content_div, 10).until(
            EC.visibility_of_element_located((By.TAG_NAME, "h1"))
        )
        game_title = h1_tag.text.strip() if h1_tag else "N/A"

        # Game info table
        game_info_table = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, "game_info"))
        )

        rows = game_info_table.find_elements(By.TAG_NAME, "tr")
        game_info = {}
        required_keys = {"Roof", "Surface", "Vegas Line", "Over/Under"}

        for row in rows:
            try:
                key = row.find_element(By.TAG_NAME, "th").text.strip()
                value = row.find_element(By.TAG_NAME, "td").text.strip()
                if key in required_keys:
                    game_info[key] = value
            except Exception:
                continue

        return {
            "title": game_title,
            "roof": game_info.get("Roof", "N/A"),
            "surface": game_info.get("Surface", "N/A"),
            "vegas_line": game_info.get("Vegas Line", "N/A"),
            "over_under": game_info.get("Over/Under", "N/A"),
        }

    except Exception as e:
        logger.error("Error scraping %s: %s", url, e)
        return None

    finally:
        if driver:
            driver.quit()


def scrape_all_games(
    max_retries: int = 3,
    delay: float = 2.0,
) -> None:
    """Scrape all boxscore URLs with retry logic."""
    proxies = load_proxies() if PROXY_FILE.exists() else []

    with open(URLS_FILE) as f:
        urls = [line.strip() for line in f if line.strip()]

    all_data = []
    failed_urls = []

    for i, url in enumerate(urls):
        logger.info("[%d/%d] Scraping %s", i + 1, len(urls), url)
        success = False

        for attempt in range(max_retries):
            result = scrape_game_info(url, proxies)
            if result:
                all_data.append(result)
                success = True
                break
            logger.warning("Retry %d/%d for %s", attempt + 1, max_retries, url)
            time.sleep(delay * (attempt + 1))  # Exponential backoff

        if not success:
            failed_urls.append(url)

        # Save progress periodically
        if (i + 1) % 50 == 0:
            _save_results(all_data)
            logger.info("Progress saved: %d/%d complete", i + 1, len(urls))

        time.sleep(delay)

    _save_results(all_data)

    if failed_urls:
        failed_path = Path("data_files/pfr_failed_urls.txt")
        with open(failed_path, "w") as f:
            f.write("\n".join(failed_urls))
        logger.warning("%d failed URLs saved to %s", len(failed_urls), failed_path)

    logger.info("Done: %d scraped, %d failed", len(all_data), len(failed_urls))


def _save_results(data: list[dict]) -> None:
    """Save scraped data to CSV."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "roof", "surface", "vegas_line", "over_under"])
        writer.writeheader()
        for row in data:
            writer.writerow({
                "title": row["title"],
                "roof": row["roof"],
                "surface": row["surface"],
                "vegas_line": row["vegas_line"],
                "over_under": row["over_under"],
            })


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    scrape_all_games()
```

---

## 3. PFF Team Grades Scraper (Authenticated Selenium)

### Prerequisites
- PFF Premium subscription (~$35/month during season)
- Chrome browser installed
- Manual login the first time (cookies cached after)

### Authentication Module

Create `scripts/scrapers/pff_auth.py`:

```python
"""PFF authentication via Selenium with cookie caching."""

import json
import logging
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

logger = logging.getLogger(__name__)

COOKIE_FILE = Path("data_files/.pff_cookies.json")
PFF_LOGIN_URL = "https://premium.pff.com/login"


def save_cookies(driver, path: Path = COOKIE_FILE) -> None:
    """Save browser cookies to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cookies = driver.get_cookies()
    with open(path, "w") as f:
        json.dump(cookies, f)
    logger.info("Saved %d cookies to %s", len(cookies), path)


def load_cookies(driver, path: Path = COOKIE_FILE) -> bool:
    """Load cookies from JSON file into driver. Returns True if successful."""
    if not path.exists():
        return False
    try:
        with open(path) as f:
            cookies = json.load(f)
        for cookie in cookies:
            # Selenium requires specific cookie format
            cookie.pop("sameSite", None)
            try:
                driver.add_cookie(cookie)
            except Exception:
                pass
        logger.info("Loaded %d cookies from %s", len(cookies), path)
        return True
    except Exception as e:
        logger.error("Failed to load cookies: %s", e)
        return False


def login_to_pff(headless: bool = False) -> webdriver.Chrome:
    """Log in to PFF Premium. Uses cached cookies if available.

    If cookies are expired/invalid, opens a visible browser for manual login.
    The user must complete login (including any 2FA) within 120 seconds.
    After successful login, cookies are cached for future use.
    """
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("window-size=1280x900")

    driver = webdriver.Chrome(options=options)

    # Try cached cookies first
    driver.get("https://premium.pff.com")
    time.sleep(2)

    if load_cookies(driver):
        driver.refresh()
        time.sleep(3)
        # Check if we're logged in (no redirect to login page)
        if "login" not in driver.current_url.lower():
            logger.info("Logged in via cached cookies")
            return driver

    # Manual login required
    logger.info("Manual login required. Please log in within 120 seconds.")
    driver.get(PFF_LOGIN_URL)

    # Wait for user to complete login
    timeout = 120
    start = time.time()
    while time.time() - start < timeout:
        if "login" not in driver.current_url.lower():
            logger.info("Login detected!")
            save_cookies(driver)
            return driver
        time.sleep(2)

    raise TimeoutError("PFF login timed out after 120 seconds")


def navigate_and_sign_in(driver, url: str) -> None:
    """Navigate to URL, re-authenticate if redirected to login."""
    driver.get(url)
    time.sleep(2)
    if "login" in driver.current_url.lower():
        logger.warning("Session expired, reloading cookies...")
        if load_cookies(driver):
            driver.get(url)
            time.sleep(3)
```

### PFF Scraper

Create `scripts/scrapers/pff_scraper.py`:

```python
"""Scrape PFF team grades from premium.pff.com."""

import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Output file
OUTPUT_FILE = Path("data_files/pff_team_grades.csv")

# PFF URL-safe team names
URL_TEAMS = [
    "arizona-cardinals", "atlanta-falcons", "baltimore-ravens",
    "buffalo-bills", "carolina-panthers", "chicago-bears",
    "cincinnati-bengals", "cleveland-browns", "dallas-cowboys",
    "denver-broncos", "detroit-lions", "green-bay-packers",
    "houston-texans", "indianapolis-colts", "jacksonville-jaguars",
    "kansas-city-chiefs", "las-vegas-raiders", "los-angeles-chargers",
    "los-angeles-rams", "miami-dolphins", "minnesota-vikings",
    "new-england-patriots", "new-orleans-saints", "new-york-giants",
    "new-york-jets", "philadelphia-eagles", "pittsburgh-steelers",
    "san-francisco-49ers", "seattle-seahawks", "tampa-bay-buccaneers",
    "tennessee-titans", "washington-commanders",
]

# Historical team name mappings
TEAM_RENAMES = {
    "washington-commanders": {
        "pre_2020": "washington-redskins",
        "2020_2021": "washington-football-team",
    },
    "las-vegas-raiders": {"pre_2020": "oakland-raiders"},
    "los-angeles-rams": {"pre_2016": "st-louis-rams"},
    "los-angeles-chargers": {"pre_2017": "san-diego-chargers"},
}

# PFF grade categories scraped from the team schedule page
# Column indices in the PFF schedule table
HOME_STAT_COLS = {
    9: "home-off", 10: "home-pass", 11: "home-pblk",
    12: "home-recv", 13: "home-run", 14: "home-rblk",
    15: "home-def", 16: "home-rdef", 17: "home-tack",
    18: "home-prsh", 19: "home-cov",
}
AWAY_STAT_COLS = {
    9: "away-off", 10: "away-pass", 11: "away-pblk",
    12: "away-recv", 13: "away-run", 14: "away-rblk",
    15: "away-def", 16: "away-rdef", 17: "away-tack",
    18: "away-prsh", 19: "away-cov",
}


def resolve_team_url(url_team: str, season: str) -> str:
    """Resolve historical team name changes."""
    szn = int(season)
    if url_team == "washington-commanders":
        if szn < 2020:
            return "washington-redskins"
        elif szn <= 2021:
            return "washington-football-team"
    elif url_team == "las-vegas-raiders" and szn < 2020:
        return "oakland-raiders"
    elif url_team == "los-angeles-rams" and szn < 2016:
        return "st-louis-rams"
    elif url_team == "los-angeles-chargers" and szn < 2017:
        return "san-diego-chargers"
    return url_team


def scrape_pff_seasons(
    seasons: list[str],
    delay: float = 5.0,
) -> pd.DataFrame:
    """Scrape PFF team grades for all teams across specified seasons.

    This requires a PFF Premium login. The first run will open a browser
    for manual authentication; subsequent runs use cached cookies.

    Returns DataFrame with one row per game, columns:
    Formatted Date, season, home_team, away_team, home-off..home-cov, away-off..away-cov
    """
    from scripts.scrapers.pff_auth import login_to_pff, navigate_and_sign_in

    driver = login_to_pff()
    time.sleep(delay)

    games: dict[str, dict] = {}

    for szn in seasons:
        logger.info("=== Season %s ===", szn)

        for team_idx, url_team in enumerate(URL_TEAMS, 1):
            resolved = resolve_team_url(url_team, szn)
            logger.info("  [%d/%d] %s", team_idx, len(URL_TEAMS), resolved)

            url = f"https://premium.pff.com/nfl/teams/{szn}/REGPO/{resolved}/schedule"
            navigate_and_sign_in(driver, url)
            time.sleep(delay)

            # Scrape table rows (up to 22 games including playoffs)
            rows = driver.find_elements("class name", "kyber-table-body__row")

            for row_num in range(1, 22):
                try:
                    # Check if row exists and has data
                    away_indicator = driver.find_element(
                        "xpath",
                        f'//*[@id="react-root"]/div/div[2]/div/div/div[3]/div/div/div[2]/div/div[1]/div/div[1]/div/div[{row_num}]/div[2]',
                    ).text

                    # Check if the score cell is empty (no game played)
                    score_cell = driver.find_element(
                        "xpath",
                        f'//*[@id="react-root"]/div/div[2]/div/div/div[3]/div/div/div[2]/div/div[1]/div/div[2]/div/div[{row_num}]/div[4]',
                    ).text
                    if score_cell in ("-", ""):
                        break

                    # Build game ID from opponent and date
                    opp_cell = driver.find_element(
                        "xpath",
                        f'//*[@id="react-root"]/div/div[2]/div/div/div[3]/div/div/div[2]/div/div[1]/div/div[2]/div/div[{row_num}]/div[1]',
                    ).text
                    date_cell = driver.find_element(
                        "xpath",
                        f'//*[@id="react-root"]/div/div[2]/div/div/div[3]/div/div/div[2]/div/div[1]/div/div[2]/div/div[{row_num}]/div[2]',
                    ).text

                    if not date_cell:
                        break

                    is_away = away_indicator == "@"

                    # Create unique game ID
                    if is_away:
                        game_id = f"{resolved}-{opp_cell}-{date_cell}/{szn}"
                    else:
                        game_id = f"{opp_cell}-{resolved}-{date_cell}/{szn}"

                    if game_id not in games:
                        games[game_id] = {col: "" for col in
                            list(HOME_STAT_COLS.values()) + list(AWAY_STAT_COLS.values())
                        }
                        games[game_id]["season"] = szn
                        games[game_id]["Formatted Date"] = date_cell

                    # Scrape grade columns
                    stat_cols = AWAY_STAT_COLS if is_away else HOME_STAT_COLS
                    for col_idx, col_name in stat_cols.items():
                        try:
                            cell = driver.find_element(
                                "xpath",
                                f'//*[@id="react-root"]/div/div[2]/div/div/div[3]/div/div/div[2]/div/div[1]/div/div[2]/div/div[{row_num}]/div[{col_idx}]',
                            ).text
                            games[game_id][col_name] = cell
                        except Exception:
                            pass

                except Exception:
                    continue

    driver.quit()
    logger.info("Scraping complete: %d games", len(games))

    df = pd.DataFrame.from_dict(games, orient="index")
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info("Saved to %s", OUTPUT_FILE)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    seasons = [str(y) for y in range(2020, 2026)]
    scrape_pff_seasons(seasons)
```

---

## 4. Proxy Management

### Setup

Create `scripts/scrapers/proxy_manager.py`:

```python
"""Proxy loading and management for web scraping."""

import csv
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)


def load_proxies_selenium(proxy_file: str | Path) -> list[tuple[str, str]]:
    """Load proxies for selenium-wire. Returns list of (url, auth) tuples."""
    proxies = []
    with open(proxy_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            proxies.append((row["proxy_url"], row["proxy_auth"]))
    logger.info("Loaded %d Selenium proxies", len(proxies))
    return proxies


def load_proxies_requests(proxy_file: str | Path) -> list[dict]:
    """Load proxies for requests library. Returns list of proxy dicts."""
    proxies = []
    with open(proxy_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row["proxy_url"]
            auth = row["proxy_auth"]
            user, pwd = auth.split(":", 1)
            proxy_url = f"http://{user}:{pwd}@{url}"
            proxies.append({"http": proxy_url, "https": proxy_url})
    logger.info("Loaded %d requests proxies", len(proxies))
    return proxies


def get_random_proxy(proxies: list) -> any:
    """Return a random proxy from the list."""
    return random.choice(proxies)
```

### Proxy CSV Format

```csv
proxy_url,proxy_auth
us-dc.example.com:10001,user123:pass456
us-ny.example.com:10001,user123:pass456
us-la.example.com:10001,user123:pass456
```

---

## 5. Anti-Detection Measures

The sports-quant project implements several anti-detection techniques. Here's a consolidated approach:

```python
"""Anti-detection utilities for Selenium-based scraping."""

import random
import time


# Diverse user-agent pool
USER_AGENTS = [
    # Chrome (Windows)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Firefox (Windows)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    # Safari (macOS)
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    # Chrome (macOS)
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Chrome (Linux)
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]


def apply_stealth(driver) -> None:
    """Apply anti-detection JavaScript patches to the WebDriver."""
    # Remove navigator.webdriver flag
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )


def random_delay(min_sec: float = 1.0, max_sec: float = 3.0) -> None:
    """Sleep for a random duration to mimic human behavior."""
    time.sleep(random.uniform(min_sec, max_sec))


def configure_stealth_options(options) -> None:
    """Apply stealth Chrome options."""
    options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("window-size=1280x800")
    options.add_argument("--disable-webgl")
    options.add_argument("--disable-gpu")
```

---

## 6. Pipeline Integration

### Scraping Orchestrator

Create `scripts/scrapers/run_scrapers.py`:

```python
"""Orchestrate all scraping pipelines."""

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_pfr_pipeline(start_year: int = 2020, end_year: int = 2025) -> None:
    """Run the complete PFR scraping pipeline."""
    from scripts.scrapers.pfr_urls import collect_all_boxscore_urls, save_urls
    from scripts.scrapers.pfr_scraper import scrape_all_games

    urls_file = Path("data_files/pfr_boxscore_urls.txt")

    # Step 1: Collect URLs (if not already done)
    if not urls_file.exists():
        logger.info("Step 1: Collecting boxscore URLs...")
        urls = collect_all_boxscore_urls(start_year=start_year, end_year=end_year)
        save_urls(urls)
    else:
        logger.info("Using existing URLs file: %s", urls_file)

    # Step 2: Scrape game data
    logger.info("Step 2: Scraping game data...")
    scrape_all_games()


def run_pff_pipeline(seasons: list[str] | None = None) -> None:
    """Run the PFF scraping pipeline (requires Premium subscription)."""
    from scripts.scrapers.pff_scraper import scrape_pff_seasons

    if seasons is None:
        seasons = [str(y) for y in range(2020, 2026)]

    logger.info("Scraping PFF grades for seasons: %s", seasons)
    scrape_pff_seasons(seasons)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run scraping pipelines")
    parser.add_argument("--pfr", action="store_true", help="Run PFR scraper")
    parser.add_argument("--pff", action="store_true", help="Run PFF scraper (requires subscription)")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    if args.pfr:
        run_pfr_pipeline(args.start_year, args.end_year)
    if args.pff:
        run_pff_pipeline()
    if not args.pfr and not args.pff:
        print("Specify --pfr, --pff, or both")
```

### Usage

```powershell
# Collect PFR boxscore URLs
python scripts/scrapers/pfr_urls.py

# Scrape PFR game data (needs proxies)
python scripts/scrapers/pfr_scraper.py

# Scrape PFF grades (needs subscription + manual login first time)
python scripts/scrapers/pff_scraper.py

# Or run everything via orchestrator
python scripts/scrapers/run_scrapers.py --pfr --pff
```

---

## 7. Directory Structure

```
scripts/
  scrapers/
    __init__.py
    pfr_urls.py           # Collect boxscore URLs from PFR
    pfr_scraper.py        # Scrape game metadata from PFR boxscores
    pff_auth.py           # PFF login + cookie management
    pff_scraper.py        # Scrape PFF team grades
    proxy_manager.py      # Proxy loading utilities
    run_scrapers.py       # Orchestrator script
data_files/
    proxies.csv           # Proxy credentials (git-ignored)
    .pff_cookies.json     # Cached PFF auth cookies (git-ignored)
    pfr_boxscore_urls.txt # Collected PFR URLs
    pfr_game_data.csv     # Scraped PFR data
    pff_team_grades.csv   # Scraped PFF data
```

### Git Ignore Additions

Add to `.gitignore`:

```
# Scraper credentials / auth
data_files/proxies.csv
data_files/.pff_cookies.json
data_files/pfr_failed_urls.txt
```

---

## 8. Cost & Effort Summary

| Component | Cost | Effort | Notes |
|-----------|------|--------|-------|
| PFR URL collector | Free | Low | BeautifulSoup + requests, ~10 min for 5 seasons |
| PFR game scraper | ~$10-20/mo (proxies) | Medium | Selenium + proxies, ~2-4 hours for 5 seasons |
| PFF scraper | ~$35/mo (subscription) | Medium | Selenium + auth, ~1-2 hours per season |
| Proxy setup | ~$10-20/mo | Low | BrightData/Oxylabs residential proxy |

### Alternative: Check nfl_data_py First

Before building scrapers, verify what's already available:

```python
import nfl_data_py as nfl

# Vegas lines might already be in schedule data
sched = nfl.import_schedules(years=[2024])
print([c for c in sched.columns if 'spread' in c.lower() or 'over' in c.lower() or 'line' in c.lower()])

# Check scoring lines
try:
    lines = nfl.import_sc_lines(years=[2024])
    print(lines.columns.tolist())
except:
    print("Scoring lines not available")
```

If `nfl_data_py` provides Vegas spread and O/U lines, the PFR scraper becomes optional (only needed for Roof/Surface data).
