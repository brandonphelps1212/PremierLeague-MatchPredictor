
"""
scrape.py — Scrape Premier League match data from FBref and save to matches.csv

Usage:
    python scrape.py --start 2017 --end 2025 --out matches.csv

Notes:
- Requires: requests, beautifulsoup4, pandas
- FBref occasionally rate-limits; consider adding sleep between requests if needed.
"""
import argparse
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE = "https://fbref.com"

def get_season_urls(start_year: int, end_year: int):
    # seasons are like 2017-2018, 2018-2019, ... with the "end" year - 1 as last start
    for y in range(start_year, end_year):
        yield f"{BASE}/en/comps/9/{y}-{y+1}/schedule/{y}-{y+1}-Premier-League-Scores-and-Fixtures"

def scrape_season(url: str) -> pd.DataFrame:
    res = requests.get(url)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")

    # Main scores & fixtures table
    table = soup.select_one("#sched_ks_3232_1, table.stats_table") or soup.select_one("table.stats_table")
    if table is None:
        return pd.DataFrame()
    rows = table.select("tbody tr")
    out = []
    for r in rows:
        # Skip separators
        if "thead" in r.get("class", []) or r.get("class") == ["spacer"]:
            continue
        date = (r.select_one("td[data-stat='date']") or {}).get_text(strip=True) if r.select_one("td[data-stat='date']") else ""
        time_str = (r.select_one("td[data-stat='time']") or {}).get_text(strip=True) if r.select_one("td[data-stat='time']") else ""
        home = (r.select_one("td[data-stat='home_team']") or {}).get_text(strip=True) if r.select_one("td[data-stat='home_team']") else ""
        away = (r.select_one("td[data-stat='away_team']") or {}).get_text(strip=True) if r.select_one("td[data-stat='away_team']") else ""
        score = (r.select_one("td[data-stat='score']") or {}).get_text(strip=True) if r.select_one("td[data-stat='score']") else ""
        venue = (r.select_one("td[data-stat='venue']") or {}).get_text(strip=True) if r.select_one("td[data-stat='venue']") else ""
        # Some pages include match links with advanced boxscore tables
        match_a = r.select_one("td[data-stat='match_report'] a")
        match_url = BASE + match_a["href"] if match_a and match_a.get("href") else ""
        out.append({
            "date": date, "time": time_str, "home": home, "away": away,
            "score": score, "venue": venue, "match_url": match_url
        })
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True, help="Season start year, e.g., 2017")
    ap.add_argument("--end", type=int, required=True, help="Season end year (exclusive), e.g., 2025")
    ap.add_argument("--out", type=str, default="matches.csv")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds to sleep between season requests")
    args = ap.parse_args()

    frames = []
    for url in get_season_urls(args.start, args.end):
        try:
            df = scrape_season(url)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"[warn] failed to scrape {url}: {e}")
        time.sleep(args.delay)

    if not frames:
        print("No data scraped.")
        return

    all_df = pd.concat(frames, ignore_index=True)

    # Basic cleanup to match tutorial expectations
    # Parse date/hour/day
    all_df["date"] = pd.to_datetime(all_df["date"], errors="coerce")
    all_df["hour"] = pd.to_datetime(all_df["time"], format="%H:%M", errors="coerce").dt.hour
    all_df["day_code"] = all_df["date"].dt.dayofweek

    # Outcome from score (H/A win/draw)
    def get_outcome(s):
        if not isinstance(s, str) or "-" not in s:
            return None
        try:
            h, a = s.split("-")
            h, a = int(h.strip()), int(a.strip())
            if h > a: return "win_home"
            if h < a: return "win_away"
            return "draw"
        except:
            return None

    all_df["outcome"] = all_df["score"].apply(get_outcome)

    all_df.to_csv(args.out, index=False)
    print(f"Saved {len(all_df)} rows to {args.out}")

if __name__ == "__main__":
    main()
