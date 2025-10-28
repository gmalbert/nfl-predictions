import requests
import pandas as pd

all_scores = []
for year in range(2020, 2025):
    for week in range(1, 19):  # NFL regular season weeks 1-18
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?seasontype=2&year={year}&week={week}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                for event in data.get("events", []):
                    comp = event.get("competitions", [{}])[0]
                    competitors = comp.get("competitors", [{}]*2)
                    # ESPN sometimes puts home/away in either order, so check
                    home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
                    away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
                    home_team = home.get("team", {}).get("displayName", "")
                    away_team = away.get("team", {}).get("displayName", "")
                    home_score = int(home.get("score", 0))
                    away_score = int(away.get("score", 0))
                    date = event.get("date", "")
                    venue = comp.get("venue", {}).get("fullName", "")
                    status = event.get("status", {}).get("type", {}).get("name", "")
                    all_scores.append({
                        "year": year,
                        "week": week,
                        "date": date,
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_score": home_score,
                        "away_score": away_score,
                        "venue": venue,
                        "status": status
                    })
            else:
                print(f"Failed to fetch {year} week {week}: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error fetching {year} week {week}: {e}")

scores_df = pd.DataFrame(all_scores)
scores_df.to_csv("espn_nfl_scores_2020_2024.csv", index=False)
print("Saved all scores to espn_nfl_scores_2020_2024.csv")
