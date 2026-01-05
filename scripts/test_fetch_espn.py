import requests

for week in (19,20,21,22):
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?seasontype=3&year=2024&week={week}"
        r = requests.get(url, timeout=10)
        events = r.json().get('events', []) if r.status_code == 200 else []
        print('week', week, 'status', r.status_code, 'events', len(events))
    except Exception as e:
        print('week', week, 'error', e)
