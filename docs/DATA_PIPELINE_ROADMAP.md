# 🔧 Data Pipeline & Infrastructure Roadmap

**Automation, Data Quality, & Production Systems**  
**Focus**: Scalability, reliability, real-time updates

---

## 🔄 Data Collection Automation

### 30. Automated Daily Data Updates

**Impact**: CRITICAL | **Effort**: 8 hours | **Benefit**: Always current data

```python
# Create new file: automation/daily_data_update.py

import schedule
import time
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    filename='data_files/logs/data_updates.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DailyDataUpdater:
    """Automated daily data collection and model updates"""
    
    def __init__(self):
        self.last_update = None
        
    def update_game_scores(self):
        """Fetch latest game scores from ESPN API"""
        try:
            import requests
            
            # Get current week
            current_week = self.get_current_nfl_week()
            
            # Fetch scores
            url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
            params = {'week': current_week}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            scores = response.json()
            
            # Update historical data
            self.process_completed_games(scores)
            
            logging.info(f"✅ Updated game scores for week {current_week}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to update scores: {e}")
            return False
    
    def update_injury_reports(self):
        """Scrape latest injury reports"""
        try:
            # Import injury scraper
            from scripts.fetch_injury_data import scrape_injury_reports
            
            injury_df = scrape_injury_reports()
            injury_df.to_csv('data_files/current_injuries.csv', index=False)
            
            logging.info("✅ Updated injury reports")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to update injuries: {e}")
            return False
    
    def update_betting_lines(self):
        """Fetch latest betting lines from odds API"""
        try:
            import requests
            import os
            
            api_key = os.getenv('ODDS_API_KEY')
            
            url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
            params = {
                'apiKey': api_key,
                'regions': 'us',
                'markets': 'spreads,totals,h2h',
                'oddsFormat': 'american'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            odds_data = response.json()
            
            # Process and save
            self.process_odds_data(odds_data)
            
            logging.info("✅ Updated betting lines")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to update lines: {e}")
            return False
    
    def update_weather_forecasts(self):
        """Get weather forecasts for upcoming games"""
        try:
            import requests
            
            # Get upcoming games
            upcoming_games = pd.read_csv('data_files/nfl_schedule_2025.csv')
            upcoming_games = upcoming_games[upcoming_games['game_type'] == 'REG']
            
            # For each outdoor stadium, get forecast
            weather_api_key = os.getenv('WEATHER_API_KEY')
            
            for idx, game in upcoming_games.iterrows():
                if game['roof'] == 'outdoors':
                    # Get stadium coordinates
                    lat, lon = self.get_stadium_coordinates(game['stadium'])
                    
                    # Fetch forecast
                    url = f"https://api.openweathermap.org/data/2.5/forecast"
                    params = {
                        'lat': lat,
                        'lon': lon,
                        'appid': weather_api_key,
                        'units': 'imperial'
                    }
                    
                    response = requests.get(url, params=params)
                    forecast = response.json()
                    
                    # Extract game time weather
                    game_weather = self.extract_game_time_forecast(forecast, game['gametime'])
                    
                    # Update game data
                    upcoming_games.at[idx, 'temp'] = game_weather['temp']
                    upcoming_games.at[idx, 'wind'] = game_weather['wind']
                    upcoming_games.at[idx, 'precipitation'] = game_weather['precipitation']
            
            upcoming_games.to_csv('data_files/nfl_schedule_2025.csv', index=False)
            
            logging.info("✅ Updated weather forecasts")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to update weather: {e}")
            return False
    
    def retrain_models(self):
        """Retrain models with latest data"""
        try:
            # Run training pipeline
            import subprocess
            
            result = subprocess.run(
                ['python', 'nfl-gather-data.py'],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                logging.info("✅ Models retrained successfully")
                return True
            else:
                logging.error(f"❌ Model retraining failed: {result.stderr}")
                return False
                
        except Exception as e:
            logging.error(f"❌ Failed to retrain models: {e}")
            return False
    
    def daily_update_workflow(self):
        """Run complete daily update workflow"""
        logging.info("🔄 Starting daily data update workflow...")
        
        success_count = 0
        total_tasks = 5
        
        # Update game scores
        if self.update_game_scores():
            success_count += 1
        
        # Update injury reports
        if self.update_injury_reports():
            success_count += 1
        
        # Update betting lines
        if self.update_betting_lines():
            success_count += 1
        
        # Update weather forecasts
        if self.update_weather_forecasts():
            success_count += 1
        
        # Retrain models if all data updated successfully
        if success_count == total_tasks - 1:
            if self.retrain_models():
                success_count += 1
        
        logging.info(f"✅ Daily update complete: {success_count}/{total_tasks} tasks successful")
        
        self.last_update = datetime.now()
        
        # Send notification
        self.send_update_notification(success_count, total_tasks)
    
    def send_update_notification(self, success, total):
        """Send email notification about update status"""
        from emailer import send_email
        
        status = "✅ SUCCESS" if success == total else "⚠️ PARTIAL SUCCESS"
        
        subject = f"NFL Predictions - Daily Update {status}"
        body = f"""
        Daily data update completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Tasks completed: {success}/{total}
        
        {'All systems operational ✅' if success == total else 'Check logs for errors ⚠️'}
        """
        
        send_email(subject, body)

# Schedule daily updates
updater = DailyDataUpdater()

# Run at 3 AM ET every day
schedule.every().day.at("03:00").do(updater.daily_update_workflow)

# Also run every hour during game days (Sunday/Monday/Thursday)
schedule.every().hour.do(lambda: updater.update_game_scores() if datetime.now().weekday() in [6, 0, 3] else None)

# Keep script running
if __name__ == "__main__":
    print("📅 NFL Data Updater started...")
    print("   Daily updates: 3:00 AM ET")
    print("   Hourly score checks on game days")
    
    while True:
        schedule.run_pending()
        time.sleep(60)
```

---

### 31. Real-Time Data Streaming

**Impact**: HIGH | **Effort**: 16 hours | **Benefit**: Live predictions

```python
# Create new file: automation/live_data_stream.py

import websocket
import json
import threading
from queue import Queue

class LiveNFLDataStream:
    """Real-time streaming of NFL game data"""
    
    def __init__(self):
        self.ws = None
        self.data_queue = Queue()
        self.callbacks = []
        
    def connect_to_espn_stream(self):
        """Connect to ESPN's WebSocket for live data"""
        
        def on_message(ws, message):
            """Handle incoming data"""
            try:
                data = json.loads(message)
                
                # Process play-by-play updates
                if data['type'] == 'play':
                    self.process_play_update(data)
                
                # Process score updates
                elif data['type'] == 'score':
                    self.process_score_update(data)
                
                # Trigger callbacks
                for callback in self.callbacks:
                    callback(data)
                    
            except Exception as e:
                print(f"Error processing message: {e}")
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print("WebSocket connection closed")
            # Reconnect after 5 seconds
            threading.Timer(5.0, self.connect_to_espn_stream).start()
        
        def on_open(ws):
            print("WebSocket connection established")
            
            # Subscribe to all NFL games
            subscribe_msg = {
                'type': 'subscribe',
                'sport': 'football',
                'league': 'nfl'
            }
            ws.send(json.dumps(subscribe_msg))
        
        # WebSocket URL (example - actual ESPN WebSocket may differ)
        ws_url = "wss://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard/stream"
        
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # Run WebSocket in separate thread
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
    
    def process_play_update(self, play_data):
        """Process individual play updates"""
        
        game_id = play_data['gameId']
        
        # Extract play details
        play_info = {
            'game_id': game_id,
            'quarter': play_data['period'],
            'time_remaining': play_data['clock'],
            'down': play_data['down'],
            'distance': play_data['distance'],
            'yard_line': play_data['yardLine'],
            'play_type': play_data['playType'],
            'yards_gained': play_data['yardsGained'],
            'score_home': play_data['homeScore'],
            'score_away': play_data['awayScore']
        }
        
        # Update live win probability
        from models.live_win_probability import LiveWinProbability
        
        live_model = LiveWinProbability()
        
        game_state = {
            'score_diff': play_info['score_home'] - play_info['score_away'],
            'time_remaining_seconds': self.convert_clock_to_seconds(
                play_info['quarter'], 
                play_info['time_remaining']
            ),
            'possession': play_data['possession'],
            'field_position': play_info['yard_line'],
            'down': play_info['down'],
            'distance': play_info['distance'],
            'timeouts_home': play_data['homeTimeouts'],
            'timeouts_away': play_data['awayTimeouts']
        }
        
        win_prob = live_model.predict_live_win_prob(game_state)
        
        # Check for betting opportunities
        self.check_live_betting_edge(game_id, win_prob)
        
        # Store in queue for processing
        self.data_queue.put(play_info)
    
    def check_live_betting_edge(self, game_id, our_win_prob):
        """Check if current live odds offer value"""
        
        # Fetch current live odds
        live_odds = self.get_live_odds(game_id)
        
        if live_odds:
            market_prob = abs(live_odds['home_ml']) / (abs(live_odds['home_ml']) + 100)
            edge = our_win_prob - market_prob
            
            if edge > 0.05:
                # Send alert
                alert_msg = f"""
                🚨 LIVE BETTING ALERT 🚨
                
                Game: {game_id}
                Our Win Prob: {our_win_prob*100:.1f}%
                Market Prob: {market_prob*100:.1f}%
                Edge: {edge*100:.1f}%
                
                Recommended Bet: Home ML at {live_odds['home_ml']}
                """
                
                self.send_push_notification(alert_msg)
                logging.info(alert_msg)
    
    def send_push_notification(self, message):
        """Send push notification for betting alerts"""
        try:
            # Use Pushover, Telegram, or similar service
            import requests
            
            pushover_token = os.getenv('PUSHOVER_TOKEN')
            pushover_user = os.getenv('PUSHOVER_USER')
            
            requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token": pushover_token,
                    "user": pushover_user,
                    "message": message,
                    "priority": 1  # High priority
                }
            )
        except Exception as e:
            logging.error(f"Failed to send push notification: {e}")
    
    def register_callback(self, callback_func):
        """Register callback for live data updates"""
        self.callbacks.append(callback_func)

# Usage
live_stream = LiveNFLDataStream()
live_stream.connect_to_espn_stream()

# Register custom callback
def my_callback(data):
    print(f"Live update: {data['type']}")

live_stream.register_callback(my_callback)
```

---

## 🗄️ Data Warehouse & Storage

### 32. PostgreSQL Database Migration

**Impact**: HIGH | **Effort**: 12 hours | **Benefit**: Better data management

```python
# Create new file: database/migrate_to_postgres.py

import psycopg2
from psycopg2 import sql
import pandas as pd

class PostgreSQLMigration:
    """Migrate from CSV files to PostgreSQL database"""
    
    def __init__(self, db_config):
        self.conn = psycopg2.connect(
            host=db_config['host'],
            database=db_config['database'],
            user=db_config['user'],
            password=db_config['password']
        )
        self.cursor = self.conn.cursor()
    
    def create_schema(self):
        """Create database schema"""
        
        # Games table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS games (
                game_id VARCHAR(50) PRIMARY KEY,
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                game_type VARCHAR(10),
                gameday DATE,
                home_team VARCHAR(10),
                away_team VARCHAR(10),
                home_score INTEGER,
                away_score INTEGER,
                spread_line DECIMAL(5,2),
                total_line DECIMAL(5,2),
                home_moneyline INTEGER,
                away_moneyline INTEGER,
                temp DECIMAL(5,2),
                wind DECIMAL(5,2),
                roof VARCHAR(20),
                surface VARCHAR(20),
                stadium VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_games_season_week ON games(season, week);
            CREATE INDEX IF NOT EXISTS idx_games_teams ON games(home_team, away_team);
        """)
        
        # Predictions table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id SERIAL PRIMARY KEY,
                game_id VARCHAR(50) REFERENCES games(game_id),
                model_version VARCHAR(20),
                prob_spread_covered DECIMAL(5,4),
                prob_underdog_won DECIMAL(5,4),
                prob_over_hit DECIMAL(5,4),
                ev_spread DECIMAL(10,2),
                ev_moneyline DECIMAL(10,2),
                ev_totals DECIMAL(10,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_predictions_game ON predictions(game_id);
            CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_version);
        """)
        
        # Player stats table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_stats (
                stat_id SERIAL PRIMARY KEY,
                player_id VARCHAR(50),
                game_id VARCHAR(50) REFERENCES games(game_id),
                position VARCHAR(10),
                passing_yards INTEGER,
                rushing_yards INTEGER,
                receiving_yards INTEGER,
                touchdowns INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_player_stats_player ON player_stats(player_id);
            CREATE INDEX IF NOT EXISTS idx_player_stats_game ON player_stats(game_id);
        """)
        
        # Betting results table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS betting_results (
                bet_id SERIAL PRIMARY KEY,
                game_id VARCHAR(50) REFERENCES games(game_id),
                bet_type VARCHAR(20),
                bet_amount DECIMAL(10,2),
                odds INTEGER,
                predicted_prob DECIMAL(5,4),
                won BOOLEAN,
                profit DECIMAL(10,2),
                roi DECIMAL(10,4),
                placed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_betting_results_date ON betting_results(placed_at);
            CREATE INDEX IF NOT EXISTS idx_betting_results_type ON betting_results(bet_type);
        """)
        
        self.conn.commit()
        print("✅ Database schema created")
    
    def migrate_historical_data(self):
        """Migrate CSV data to PostgreSQL"""
        
        # Migrate games
        games_df = pd.read_csv('data_files/nfl_games_historical.csv', sep='\t')
        
        for idx, row in games_df.iterrows():
            self.cursor.execute("""
                INSERT INTO games (
                    game_id, season, week, game_type, gameday,
                    home_team, away_team, home_score, away_score,
                    spread_line, total_line, home_moneyline, away_moneyline,
                    temp, wind, roof, surface, stadium
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (game_id) DO NOTHING
            """, (
                row['game_id'], row['season'], row['week'], row['game_type'],
                row['gameday'], row['home_team'], row['away_team'],
                row['home_score'], row['away_score'], row['spread_line'],
                row['total_line'], row['home_moneyline'], row['away_moneyline'],
                row.get('temp'), row.get('wind'), row.get('roof'),
                row.get('surface'), row.get('stadium')
            ))
        
        self.conn.commit()
        print(f"✅ Migrated {len(games_df)} games")
        
        # Migrate predictions
        predictions_df = pd.read_csv('data_files/nfl_games_historical_with_predictions.csv', sep='\t')
        
        for idx, row in predictions_df.iterrows():
            self.cursor.execute("""
                INSERT INTO predictions (
                    game_id, model_version,
                    prob_spread_covered, prob_underdog_won, prob_over_hit,
                    ev_spread, ev_moneyline, ev_totals
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                row['game_id'], 'v1.0',
                row.get('prob_underdogCovered'),
                row.get('prob_underdogWon'),
                row.get('prob_overHit'),
                row.get('ev_spread'),
                row.get('ev_moneyline'),
                row.get('ev_totals')
            ))
        
        self.conn.commit()
        print(f"✅ Migrated {len(predictions_df)} predictions")
    
    def create_materialized_views(self):
        """Create materialized views for common queries"""
        
        # View: Team season statistics
        self.cursor.execute("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS team_season_stats AS
            SELECT 
                season,
                home_team AS team,
                COUNT(*) AS games_played,
                SUM(CASE WHEN home_score > away_score THEN 1 ELSE 0 END) AS wins,
                AVG(home_score) AS avg_points_scored,
                AVG(away_score) AS avg_points_allowed,
                AVG(spread_line) AS avg_spread
            FROM games
            GROUP BY season, home_team
            
            UNION ALL
            
            SELECT 
                season,
                away_team AS team,
                COUNT(*) AS games_played,
                SUM(CASE WHEN away_score > home_score THEN 1 ELSE 0 END) AS wins,
                AVG(away_score) AS avg_points_scored,
                AVG(home_score) AS avg_points_allowed,
                AVG(-spread_line) AS avg_spread
            FROM games
            GROUP BY season, away_team;
            
            CREATE INDEX IF NOT EXISTS idx_team_season_stats ON team_season_stats(season, team);
        """)
        
        # View: Model performance by week
        self.cursor.execute("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS model_performance_weekly AS
            SELECT 
                g.season,
                g.week,
                p.model_version,
                COUNT(*) AS predictions_made,
                AVG(CASE WHEN (g.home_score > g.away_score AND p.prob_spread_covered > 0.5)
                          OR (g.away_score > g.home_score AND p.prob_spread_covered <= 0.5)
                     THEN 1 ELSE 0 END) AS accuracy,
                SUM(br.profit) AS total_profit,
                AVG(br.roi) AS avg_roi
            FROM games g
            JOIN predictions p ON g.game_id = p.game_id
            LEFT JOIN betting_results br ON g.game_id = br.game_id
            GROUP BY g.season, g.week, p.model_version;
        """)
        
        self.conn.commit()
        print("✅ Created materialized views")
    
    def close(self):
        """Close database connection"""
        self.cursor.close()
        self.conn.close()

# Usage
db_config = {
    'host': 'localhost',
    'database': 'nfl_predictions',
    'user': 'nfl_user',
    'password': os.getenv('DB_PASSWORD')
}

migrator = PostgreSQLMigration(db_config)
migrator.create_schema()
migrator.migrate_historical_data()
migrator.create_materialized_views()
migrator.close()
```

---

### 33. Data Quality Monitoring

**Impact**: MEDIUM | **Effort**: 6 hours | **Benefit**: Prevent bad predictions

```python
# Create new file: data_quality/monitoring.py

class DataQualityMonitor:
    """Monitor data quality and alert on issues"""
    
    def __init__(self):
        self.checks = []
        self.failed_checks = []
        
    def check_data_freshness(self, df, date_column, max_age_days=7):
        """Ensure data is recent"""
        latest_date = pd.to_datetime(df[date_column]).max()
        age_days = (datetime.now() - latest_date).days
        
        if age_days > max_age_days:
            self.failed_checks.append({
                'check': 'data_freshness',
                'status': 'FAIL',
                'message': f"Data is {age_days} days old (max: {max_age_days})"
            })
            return False
        
        return True
    
    def check_missing_values(self, df, critical_columns, max_missing_pct=0.05):
        """Check for excessive missing values"""
        for col in critical_columns:
            missing_pct = df[col].isna().sum() / len(df)
            
            if missing_pct > max_missing_pct:
                self.failed_checks.append({
                    'check': 'missing_values',
                    'column': col,
                    'status': 'FAIL',
                    'message': f"{col} has {missing_pct*100:.1f}% missing (max: {max_missing_pct*100:.1f}%)"
                })
                return False
        
        return True
    
    def check_value_ranges(self, df, column, min_val, max_val):
        """Check values are within expected range"""
        out_of_range = df[(df[column] < min_val) | (df[column] > max_val)]
        
        if len(out_of_range) > 0:
            self.failed_checks.append({
                'check': 'value_ranges',
                'column': column,
                'status': 'FAIL',
                'message': f"{len(out_of_range)} values in {column} outside range [{min_val}, {max_val}]"
            })
            return False
        
        return True
    
    def check_probability_calibration(self, y_true, y_pred, max_error=0.15):
        """Check model calibration hasn't degraded"""
        from sklearn.calibration import calibration_curve
        
        fraction_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=5)
        calibration_error = np.abs(fraction_pos - mean_pred).mean()
        
        if calibration_error > max_error:
            self.failed_checks.append({
                'check': 'calibration',
                'status': 'FAIL',
                'message': f"Calibration error {calibration_error:.3f} exceeds threshold {max_error}"
            })
            return False
        
        return True
    
    def run_all_checks(self, data_dict):
        """Run all quality checks"""
        self.failed_checks = []
        
        # Check game data
        games_df = data_dict['games']
        self.check_data_freshness(games_df, 'gameday', max_age_days=7)
        self.check_missing_values(games_df, ['spread_line', 'total_line'], max_missing_pct=0.01)
        self.check_value_ranges(games_df, 'home_score', min_val=0, max_val=100)
        self.check_value_ranges(games_df, 'spread_line', min_val=-30, max_val=30)
        
        # Check predictions
        predictions_df = data_dict['predictions']
        self.check_value_ranges(predictions_df, 'prob_underdogCovered', min_val=0, max_val=1)
        
        # Generate report
        if len(self.failed_checks) > 0:
            self.send_alert()
            return False
        
        return True
    
    def send_alert(self):
        """Send alert about data quality issues"""
        alert_msg = "⚠️ DATA QUALITY ALERT ⚠️\n\n"
        alert_msg += f"Found {len(self.failed_checks)} failed checks:\n\n"
        
        for check in self.failed_checks:
            alert_msg += f"- {check['check']}: {check['message']}\n"
        
        # Send email
        from emailer import send_email
        send_email("NFL Predictions - Data Quality Alert", alert_msg)
        
        logging.error(alert_msg)

# Usage - run before making predictions
monitor = DataQualityMonitor()

data_to_check = {
    'games': pd.read_csv('data_files/nfl_games_historical.csv', sep='\t'),
    'predictions': pd.read_csv('data_files/nfl_games_historical_with_predictions.csv', sep='\t')
}

if not monitor.run_all_checks(data_to_check):
    print("❌ Data quality checks failed - fix issues before proceeding")
    sys.exit(1)
else:
    print("✅ All data quality checks passed")
```

---

## 🚀 Production Deployment

### 34. Docker Containerization

**Impact**: HIGH | **Effort**: 4 hours | **Benefit**: Portable, scalable deployment

```dockerfile
# Create Dockerfile

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data_files/logs data_files/exports

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TZ=America/New_York

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "predictions.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```yaml
# Create docker-compose.yml

version: '3.8'

services:
  nfl-predictions-app:
    build: .
    container_name: nfl-predictions
    ports:
      - "8501:8501"
    environment:
      - ODDS_API_KEY=${ODDS_API_KEY}
      - WEATHER_API_KEY=${WEATHER_API_KEY}
      - DB_PASSWORD=${DB_PASSWORD}
    volumes:
      - ./data_files:/app/data_files
      - ./logs:/app/logs
    restart: unless-stopped
    depends_on:
      - postgres
      - redis
  
  postgres:
    image: postgres:15-alpine
    container_name: nfl-postgres
    environment:
      - POSTGRES_DB=nfl_predictions
      - POSTGRES_USER=nfl_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped
  
  redis:
    image: redis:7-alpine
    container_name: nfl-redis
    ports:
      - "6379:6379"
    restart: unless-stopped
  
  data-updater:
    build: .
    container_name: nfl-data-updater
    command: python automation/daily_data_update.py
    environment:
      - ODDS_API_KEY=${ODDS_API_KEY}
      - WEATHER_API_KEY=${WEATHER_API_KEY}
    volumes:
      - ./data_files:/app/data_files
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

volumes:
  postgres-data:
```

---

### 35. CI/CD Pipeline

**Impact**: MEDIUM | **Effort**: 6 hours | **Benefit**: Automated testing & deployment

```yaml
# Create .github/workflows/deploy.yml

name: NFL Predictions CI/CD

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    # Run daily during NFL season
    - cron: '0 3 * 9-2 *'

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python 3.12
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    
    - name: Cache pip packages
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run data quality checks
      run: python data_quality/monitoring.py
    
    - name: Run unit tests
      run: pytest tests/ --cov=. --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml
  
  update-predictions:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule' || github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python 3.12
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    
    - name: Install dependencies
      run: pip install -r requirements.txt
    
    - name: Update data and retrain models
      env:
        ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
        WEATHER_API_KEY: ${{ secrets.WEATHER_API_KEY }}
      run: |
        python automation/daily_data_update.py
        python build_and_train_pipeline.py
    
    - name: Commit updated predictions
      run: |
        git config --local user.email "action@github.com"
        git config --local user.name "GitHub Action"
        git add data_files/nfl_games_historical_with_predictions.csv
        git add data_files/model_metrics.json
        git commit -m "🤖 Auto-update predictions [skip ci]" || echo "No changes"
        git push
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to Streamlit Cloud
      env:
        STREAMLIT_TOKEN: ${{ secrets.STREAMLIT_TOKEN }}
      run: |
        # Trigger Streamlit Cloud deployment
        curl -X POST https://share.streamlit.io/api/v1/apps/deploy \
          -H "Authorization: Bearer $STREAMLIT_TOKEN" \
          -d '{"repo": "${{ github.repository }}", "branch": "main"}'
```

---

## 📊 Summary: Infrastructure Priorities

**Week 1**: Daily automation (#30), Data quality monitoring (#33)  
**Week 2**: PostgreSQL migration (#32), Docker deployment (#34)  
**Week 3**: Real-time streaming (#31), CI/CD pipeline (#35)  

**Expected Impact**: 
- 95%+ system uptime
- <5 minute data latency
- Automated model retraining
- Scalable to 1000+ concurrent users
