PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS source_run (
    source_run_id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    source_uri TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    available_at TEXT NOT NULL,
    content_sha256 TEXT,
    row_count INTEGER,
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS game (
    game_id TEXT PRIMARY KEY,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_type TEXT,
    kickoff_at TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    neutral_site INTEGER NOT NULL DEFAULT 0 CHECK (neutral_site IN (0, 1)),
    stadium_id TEXT,
    roof TEXT,
    surface TEXT,
    home_score INTEGER,
    away_score INTEGER,
    result_available_at TEXT,
    source_run_id TEXT REFERENCES source_run(source_run_id)
);

CREATE TABLE IF NOT EXISTS team_game (
    game_id TEXT NOT NULL REFERENCES game(game_id),
    team TEXT NOT NULL,
    opponent TEXT NOT NULL,
    is_home INTEGER NOT NULL CHECK (is_home IN (0, 1)),
    points_for INTEGER,
    points_against INTEGER,
    plays INTEGER,
    epa_per_play REAL,
    success_rate REAL,
    pass_rate REAL,
    PRIMARY KEY (game_id, team)
);

CREATE TABLE IF NOT EXISTS market_snapshot (
    market_snapshot_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL REFERENCES game(game_id),
    book TEXT NOT NULL,
    market TEXT NOT NULL CHECK (market IN ('moneyline', 'spread', 'total', 'player_prop')),
    participant_id TEXT NOT NULL DEFAULT '',
    side TEXT NOT NULL,
    line REAL NOT NULL DEFAULT 0.0,
    price_american REAL NOT NULL CHECK (price_american != 0),
    observed_at TEXT NOT NULL,
    available_at TEXT NOT NULL,
    source_run_id TEXT REFERENCES source_run(source_run_id),
    UNIQUE (game_id, book, market, participant_id, side, line, observed_at)
);

CREATE TABLE IF NOT EXISTS availability_snapshot (
    availability_snapshot_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL REFERENCES game(game_id),
    player_id TEXT NOT NULL,
    team TEXT NOT NULL,
    injury_status TEXT,
    practice_status TEXT,
    active_probability REAL CHECK (active_probability BETWEEN 0 AND 1),
    expected_snap_share REAL CHECK (expected_snap_share BETWEEN 0 AND 1),
    observed_at TEXT NOT NULL,
    available_at TEXT NOT NULL,
    source_run_id TEXT REFERENCES source_run(source_run_id),
    UNIQUE (game_id, player_id, observed_at)
);

CREATE TABLE IF NOT EXISTS player_game (
    game_id TEXT NOT NULL REFERENCES game(game_id),
    player_id TEXT NOT NULL,
    team TEXT NOT NULL,
    position TEXT,
    snaps INTEGER,
    routes INTEGER,
    targets INTEGER,
    carries INTEGER,
    dropbacks INTEGER,
    result_available_at TEXT NOT NULL,
    PRIMARY KEY (game_id, player_id)
);

CREATE TABLE IF NOT EXISTS feature_snapshot (
    entity_type TEXT NOT NULL CHECK (entity_type IN ('game', 'team', 'player')),
    entity_id TEXT NOT NULL,
    game_id TEXT NOT NULL REFERENCES game(game_id),
    feature_set_version TEXT NOT NULL,
    cutoff_at TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    feature_value REAL,
    is_missing INTEGER NOT NULL DEFAULT 0 CHECK (is_missing IN (0, 1)),
    source_run_id TEXT REFERENCES source_run(source_run_id),
    PRIMARY KEY (entity_type, entity_id, game_id, feature_set_version, cutoff_at, feature_name)
);

CREATE TABLE IF NOT EXISTS model_run (
    model_run_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    feature_set_version TEXT NOT NULL,
    target_name TEXT NOT NULL,
    train_start_at TEXT NOT NULL,
    train_end_at TEXT NOT NULL,
    calibration_start_at TEXT,
    calibration_end_at TEXT,
    code_commit TEXT,
    parameters_json TEXT NOT NULL,
    metrics_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prediction_snapshot (
    prediction_id TEXT PRIMARY KEY,
    model_run_id TEXT NOT NULL REFERENCES model_run(model_run_id),
    game_id TEXT NOT NULL REFERENCES game(game_id),
    market TEXT NOT NULL,
    participant_id TEXT NOT NULL DEFAULT '',
    side TEXT NOT NULL,
    line REAL NOT NULL DEFAULT 0.0,
    model_probability REAL NOT NULL CHECK (model_probability BETWEEN 0 AND 1),
    market_probability REAL CHECK (market_probability BETWEEN 0 AND 1),
    calibrated_probability REAL CHECK (calibrated_probability BETWEEN 0 AND 1),
    cutoff_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (model_run_id, game_id, market, participant_id, side, line, cutoff_at)
);

CREATE TABLE IF NOT EXISTS bet_decision (
    bet_decision_id TEXT PRIMARY KEY,
    prediction_id TEXT NOT NULL REFERENCES prediction_snapshot(prediction_id),
    market_snapshot_id TEXT NOT NULL REFERENCES market_snapshot(market_snapshot_id),
    decision TEXT NOT NULL CHECK (decision IN ('bet', 'pass', 'shadow')),
    edge REAL NOT NULL,
    expected_profit_per_unit REAL NOT NULL,
    stake_fraction REAL NOT NULL CHECK (stake_fraction BETWEEN 0 AND 1),
    decided_at TEXT NOT NULL,
    policy_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bet_settlement (
    bet_decision_id TEXT PRIMARY KEY REFERENCES bet_decision(bet_decision_id),
    result TEXT NOT NULL CHECK (result IN ('win', 'loss', 'push', 'void')),
    stake REAL NOT NULL CHECK (stake >= 0),
    profit REAL NOT NULL,
    closing_line REAL,
    closing_price_american REAL,
    clv REAL,
    settled_at TEXT NOT NULL,
    settlement_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS data_quality_event (
    data_quality_event_id TEXT PRIMARY KEY,
    source_run_id TEXT REFERENCES source_run(source_run_id),
    table_name TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'error')),
    rule_name TEXT NOT NULL,
    affected_rows INTEGER NOT NULL,
    examples_json TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_game_kickoff ON game(kickoff_at);
CREATE INDEX IF NOT EXISTS idx_market_game_time ON market_snapshot(game_id, market, observed_at);
CREATE INDEX IF NOT EXISTS idx_availability_game_time ON availability_snapshot(game_id, observed_at);
CREATE INDEX IF NOT EXISTS idx_feature_game_cutoff ON feature_snapshot(game_id, cutoff_at);
CREATE INDEX IF NOT EXISTS idx_prediction_game_cutoff ON prediction_snapshot(game_id, cutoff_at);
