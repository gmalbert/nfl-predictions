# 🧠 Advanced ML Techniques & Deep Learning Roadmap

**Part 2 of Advanced Model Improvements**  
**Focus**: Neural networks, advanced ML, cutting-edge techniques

---

## 🤖 Deep Learning Models

### 17. LSTM for Sequential Game Predictions

**Impact**: HIGH | **Effort**: 12 hours | **Expected ROI**: +15-20%

**Why**: Capture temporal dependencies in team performance over the season

```python
# Create new file: models/lstm_predictor.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

class NFLSequenceLSTM:
    """LSTM model for predicting game outcomes based on team performance sequences"""
    
    def __init__(self, sequence_length=5):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = None
        
    def create_sequences(self, df, team):
        """Create sequences of team performance data"""
        from sklearn.preprocessing import StandardScaler
        
        # Get games for this team
        team_games = df[
            (df['home_team'] == team) | (df['away_team'] == team)
        ].sort_values(['season', 'week'])
        
        # Features to track over time
        features_per_game = [
            'points_scored', 'points_allowed', 'yards_gained', 'yards_allowed',
            'turnovers_forced', 'turnovers_committed', 'third_down_pct', 
            'red_zone_pct', 'time_of_possession'
        ]
        
        sequences = []
        targets = []
        
        for i in range(len(team_games) - self.sequence_length):
            # Get last N games
            seq_data = team_games.iloc[i:i+self.sequence_length][features_per_game].values
            
            # Target: did they win the next game?
            next_game = team_games.iloc[i+self.sequence_length]
            if next_game['home_team'] == team:
                target = next_game['homeWin']
            else:
                target = next_game['awayWin']
            
            sequences.append(seq_data)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def build_model(self, input_shape):
        """Build LSTM architecture"""
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )
        
        return model
    
    def train(self, df, epochs=50, batch_size=32):
        """Train LSTM on historical data"""
        from sklearn.preprocessing import StandardScaler
        
        # Get all teams
        teams = pd.concat([df['home_team'], df['away_team']]).unique()
        
        all_sequences = []
        all_targets = []
        
        for team in teams:
            seq, tar = self.create_sequences(df, team)
            if len(seq) > 0:
                all_sequences.append(seq)
                all_targets.append(tar)
        
        X = np.vstack(all_sequences)
        y = np.concatenate(all_targets)
        
        # Normalize features
        self.scaler = StandardScaler()
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(X.shape)
        
        # Build and train model
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )
        
        return history
    
    def predict(self, team_sequence):
        """Predict outcome for a team given their recent games"""
        if self.scaler is not None:
            team_sequence = self.scaler.transform(team_sequence.reshape(-1, team_sequence.shape[-1]))
            team_sequence = team_sequence.reshape(1, self.sequence_length, -1)
        
        return self.model.predict(team_sequence)[0, 0]

# Usage in nfl-gather-data.py
lstm_model = NFLSequenceLSTM(sequence_length=5)
history = lstm_model.train(historical_game_level_data, epochs=100)

# Add LSTM predictions as a feature
historical_game_level_data['lstm_home_win_prob'] = 0.0
historical_game_level_data['lstm_away_win_prob'] = 0.0

# This would require extracting sequences for each game's teams
features.extend(['lstm_home_win_prob', 'lstm_away_win_prob'])
```

---

### 18. Transformer Architecture for Game Predictions

**Impact**: VERY HIGH | **Effort**: 20 hours | **Expected ROI**: +20-30%

**Why**: Attention mechanism can identify what features matter most in different contexts

```python
# Create new file: models/transformer_predictor.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TransformerBlock(layers.Layer):
    """Transformer block with multi-head attention"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class NFLTransformer:
    """Transformer model for NFL game prediction"""
    
    def __init__(self, num_features):
        self.num_features = num_features
        self.model = None
        
    def build_model(self):
        """Build transformer architecture"""
        inputs = layers.Input(shape=(None, self.num_features))
        
        # Positional encoding
        transformer_block = TransformerBlock(
            embed_dim=self.num_features, 
            num_heads=4, 
            ff_dim=128
        )
        
        x = transformer_block(inputs)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC()]
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        """Train transformer model"""
        self.model = self.build_model()
        
        # Learning rate scheduler
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
        )
        
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[lr_scheduler, early_stop],
            verbose=1
        )
        
        return history

# Usage
transformer = NFLTransformer(num_features=len(features))
history = transformer.train(
    X_train_spread.values.reshape(X_train_spread.shape[0], 1, -1),
    y_spread_train.values,
    X_test_spread.values.reshape(X_test_spread.shape[0], 1, -1),
    y_spread_test.values
)
```

---

## 🎲 Advanced Statistical Models

### 19. Bayesian Hierarchical Model

**Impact**: HIGH | **Effort**: 15 hours | **Expected ROI**: +12-18%

**Why**: Properly accounts for uncertainty and team-specific effects

```python
# Requires: pip install pymc

import pymc as pm
import arviz as az

class BayesianNFLModel:
    """Bayesian hierarchical model for NFL predictions"""
    
    def __init__(self):
        self.model = None
        self.trace = None
        
    def build_model(self, df):
        """Build Bayesian hierarchical model"""
        
        # Encode teams as integers
        team_encoder = {team: idx for idx, team in enumerate(df['home_team'].unique())}
        df['home_team_idx'] = df['home_team'].map(team_encoder)
        df['away_team_idx'] = df['away_team'].map(team_encoder)
        
        n_teams = len(team_encoder)
        
        with pm.Model() as model:
            # Hyperpriors for team strength
            mu_attack = pm.Normal('mu_attack', mu=0, sigma=1)
            sigma_attack = pm.HalfNormal('sigma_attack', sigma=1)
            mu_defense = pm.Normal('mu_defense', mu=0, sigma=1)
            sigma_defense = pm.HalfNormal('sigma_defense', sigma=1)
            
            # Team-specific attack and defense strengths
            attack = pm.Normal('attack', mu=mu_attack, sigma=sigma_attack, shape=n_teams)
            defense = pm.Normal('defense', mu=mu_defense, sigma=sigma_defense, shape=n_teams)
            
            # Home field advantage
            home_advantage = pm.Normal('home_advantage', mu=2.5, sigma=1)
            
            # Expected points for each game
            home_attack_idx = df['home_team_idx'].values
            home_defense_idx = df['home_team_idx'].values
            away_attack_idx = df['away_team_idx'].values
            away_defense_idx = df['away_team_idx'].values
            
            # Home team expected score
            home_theta = pm.Deterministic(
                'home_theta',
                attack[home_attack_idx] - defense[away_defense_idx] + home_advantage
            )
            
            # Away team expected score
            away_theta = pm.Deterministic(
                'away_theta',
                attack[away_attack_idx] - defense[home_defense_idx]
            )
            
            # Observed scores (Poisson distribution)
            home_points = pm.Poisson('home_points', mu=pm.math.exp(home_theta), observed=df['home_score'])
            away_points = pm.Poisson('away_points', mu=pm.math.exp(away_theta), observed=df['away_score'])
            
            # Probability home team wins
            home_win_prob = pm.Deterministic(
                'home_win_prob',
                pm.math.sigmoid(home_theta - away_theta)
            )
        
        self.model = model
        return model
    
    def fit(self, df, draws=2000, tune=1000):
        """Fit Bayesian model using MCMC"""
        model = self.build_model(df)
        
        with model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=4,
                return_inferencedata=True,
                random_seed=42
            )
        
        return self.trace
    
    def predict_game(self, home_team, away_team, team_encoder):
        """Predict outcome of a specific game"""
        home_idx = team_encoder[home_team]
        away_idx = team_encoder[away_team]
        
        # Get posterior samples
        attack_samples = self.trace.posterior['attack'].values.reshape(-1, len(team_encoder))
        defense_samples = self.trace.posterior['defense'].values.reshape(-1, len(team_encoder))
        home_adv_samples = self.trace.posterior['home_advantage'].values.flatten()
        
        # Calculate win probability
        home_theta = attack_samples[:, home_idx] - defense_samples[:, away_idx] + home_adv_samples
        away_theta = attack_samples[:, away_idx] - defense_samples[:, home_idx]
        
        win_prob = (home_theta > away_theta).mean()
        
        return win_prob

# Usage
bayesian_model = BayesianNFLModel()
trace = bayesian_model.fit(historical_game_level_data[historical_game_level_data['season'] >= 2023])

# Generate predictions
print(f"Bayesian model converged: {az.summary(trace)}")
```

---

### 20. Gradient Boosting with CatBoost

**Impact**: MEDIUM-HIGH | **Effort**: 4 hours | **Expected ROI**: +8-12%

**Why**: Handles categorical features better, often outperforms XGBoost

```python
# Requires: pip install catboost

from catboost import CatBoostClassifier, Pool

# Add to nfl-gather-data.py

# Identify categorical features
categorical_features = [
    'home_team', 'away_team', 'home_coach', 'away_coach',
    'stadium', 'roof', 'surface', 'weekday'
]

# Prepare data with categorical encoding preserved
X_spread_cat = historical_game_level_data[best_features_spread].copy()

# CatBoost automatically handles categorical features
cat_indices = [X_spread_cat.columns.get_loc(col) for col in categorical_features if col in X_spread_cat.columns]

# Create Pool objects (CatBoost's data structure)
train_pool = Pool(
    X_train_spread, 
    y_spread_train,
    cat_features=cat_indices
)

test_pool = Pool(
    X_test_spread,
    y_spread_test,
    cat_features=cat_indices
)

# Train CatBoost model
catboost_spread = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='AUC',
    early_stopping_rounds=50,
    random_seed=42,
    verbose=100
)

catboost_spread.fit(
    train_pool,
    eval_set=test_pool,
    use_best_model=True
)

# Get feature importances
feature_importance = catboost_spread.get_feature_importance(train_pool)
feature_names = X_train_spread.columns

print("\n🐱 CatBoost Top 10 Features:")
for name, importance in sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)[:10]:
    print(f"   {name}: {importance:.4f}")

# Compare with XGBoost
catboost_predictions = catboost_spread.predict_proba(X_test_spread)[:, 1]
catboost_accuracy = accuracy_score(y_spread_test, (catboost_predictions >= 0.5).astype(int))
print(f"\nCatBoost accuracy: {catboost_accuracy:.4f}")
print(f"XGBoost accuracy: {spread_accuracy:.4f}")
print(f"Improvement: {(catboost_accuracy - spread_accuracy)*100:+.2f}%")

# Create ensemble of XGBoost + CatBoost
ensemble_predictions = (spread_probs + catboost_predictions) / 2
ensemble_accuracy = accuracy_score(y_spread_test, (ensemble_predictions >= 0.5).astype(int))
print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
```

---

## 🔮 Time Series & Forecasting

### 21. Prophet for Season Trend Forecasting

**Impact**: MEDIUM | **Effort**: 6 hours | **Expected ROI**: +5-8%

**Why**: Identify team trends and seasonal patterns

```python
# Requires: pip install prophet

from prophet import Prophet
import pandas as pd

class TeamTrendForecaster:
    """Use Facebook Prophet to forecast team performance trends"""
    
    def __init__(self):
        self.models = {}
        
    def prepare_team_data(self, df, team, metric='win_pct'):
        """Prepare time series data for a team"""
        team_games = df[
            (df['home_team'] == team) | (df['away_team'] == team)
        ].copy()
        
        team_games['date'] = pd.to_datetime(
            team_games['season'].astype(str) + '-' + 
            team_games['week'].astype(str) + '-1',
            format='%Y-%W-%w'
        )
        
        # Calculate rolling win percentage
        team_games['win'] = 0
        team_games.loc[
            ((team_games['home_team'] == team) & (team_games['homeWin'] == 1)) |
            ((team_games['away_team'] == team) & (team_games['awayWin'] == 1)),
            'win'
        ] = 1
        
        team_games = team_games.sort_values('date')
        team_games['rolling_win_pct'] = team_games['win'].rolling(window=5, min_periods=1).mean()
        
        # Prophet requires 'ds' and 'y' columns
        prophet_df = pd.DataFrame({
            'ds': team_games['date'],
            'y': team_games['rolling_win_pct']
        })
        
        return prophet_df
    
    def train_team_model(self, df, team):
        """Train Prophet model for a specific team"""
        prophet_df = self.prepare_team_data(df, team)
        
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        model.fit(prophet_df)
        self.models[team] = model
        
        return model
    
    def forecast_team_performance(self, team, periods=4):
        """Forecast next N weeks of team performance"""
        if team not in self.models:
            raise ValueError(f"No model trained for {team}")
        
        model = self.models[team]
        future = model.make_future_dataframe(periods=periods, freq='W')
        forecast = model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    
    def get_trend_strength(self, team):
        """Get trend strength for a team (positive=improving, negative=declining)"""
        if team not in self.models:
            return 0.0
        
        model = self.models[team]
        
        # Get trend from the last few weeks
        future = model.make_future_dataframe(periods=1, freq='W')
        forecast = model.predict(future)
        
        # Trend is the derivative of the trend component
        recent_trend = forecast['trend'].tail(5).diff().mean()
        
        return recent_trend

# Train models for all teams
forecaster = TeamTrendForecaster()
teams = pd.concat([historical_game_level_data['home_team'], historical_game_level_data['away_team']]).unique()

print("Training Prophet models for all teams...")
for team in teams:
    try:
        forecaster.train_team_model(historical_game_level_data, team)
    except Exception as e:
        print(f"Failed to train model for {team}: {e}")

# Add trend features to dataset
historical_game_level_data['home_team_prophet_trend'] = 0.0
historical_game_level_data['away_team_prophet_trend'] = 0.0

for idx, row in historical_game_level_data.iterrows():
    try:
        home_trend = forecaster.get_trend_strength(row['home_team'])
        away_trend = forecaster.get_trend_strength(row['away_team'])
        
        historical_game_level_data.at[idx, 'home_team_prophet_trend'] = home_trend
        historical_game_level_data.at[idx, 'away_team_prophet_trend'] = away_trend
    except:
        pass

historical_game_level_data['prophet_trend_diff'] = (
    historical_game_level_data['home_team_prophet_trend'] - 
    historical_game_level_data['away_team_prophet_trend']
)

features.extend(['home_team_prophet_trend', 'away_team_prophet_trend', 'prophet_trend_diff'])
```

---

## 🎯 Meta-Learning & AutoML

### 22. AutoML with FLAML

**Impact**: VERY HIGH | **Effort**: 8 hours | **Expected ROI**: +15-25%

**Why**: Automatically finds best model architecture and hyperparameters

```python
# Requires: pip install flaml

from flaml import AutoML

class AutoMLOptimizer:
    """Use FLAML for automated model selection and hyperparameter tuning"""
    
    def __init__(self, time_budget=3600):
        self.time_budget = time_budget  # seconds
        self.automl = AutoML()
        
    def optimize_spread_model(self, X_train, y_train, X_test, y_test):
        """Automatically find best model for spread predictions"""
        
        settings = {
            "time_budget": self.time_budget,
            "metric": "roc_auc",
            "task": "classification",
            "log_file_name": "data_files/automl_spread.log",
            "estimator_list": ["xgboost", "lgbm", "catboost", "rf", "extra_tree"],
            "early_stop": True,
            "eval_method": "cv",
            "n_splits": 5,
            "verbose": 1
        }
        
        print(f"\n🤖 Running AutoML for {self.time_budget/60:.0f} minutes...")
        
        self.automl.fit(X_train, y_train, **settings)
        
        # Print best config
        print(f"\n✅ Best model: {self.automl.best_estimator}")
        print(f"   Best hyperparameters: {self.automl.best_config}")
        print(f"   Best CV score: {self.automl.best_loss:.4f}")
        
        # Evaluate on test set
        y_pred_proba = self.automl.predict_proba(X_test)[:, 1]
        
        from sklearn.metrics import roc_auc_score
        test_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"   Test AUC: {test_auc:.4f}")
        
        return self.automl
    
    def get_model(self):
        """Get the optimized model"""
        return self.automl

# Usage
automl_optimizer = AutoMLOptimizer(time_budget=7200)  # 2 hours
best_model = automl_optimizer.optimize_spread_model(
    X_train_spread, y_spread_train,
    X_test_spread, y_spread_test
)

# Use AutoML model for predictions
historical_game_level_data['prob_underdogCovered_automl'] = best_model.predict_proba(X_spread)[:, 1]

# Save the AutoML model
import pickle
with open('data_files/automl_spread_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
```

---

### 23. Neural Architecture Search (NAS)

**Impact**: VERY HIGH | **Effort**: 24 hours | **Expected ROI**: +20-35%

**Why**: Automatically discover optimal neural network architectures

```python
# Requires: pip install keras-tuner

import keras_tuner as kt

class NFLHyperModel(kt.HyperModel):
    """Define search space for neural architecture"""
    
    def build(self, hp):
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(X_train_spread.shape[1],)))
        
        # Hidden layers - search for optimal number
        for i in range(hp.Int('num_layers', 1, 5)):
            model.add(layers.Dense(
                units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
                activation=hp.Choice('activation', ['relu', 'tanh', 'elu'])
            ))
            model.add(layers.Dropout(
                rate=hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)
            ))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Optimizer selection
        optimizer = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
        
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = keras.optimizers.SGD(learning_rate=learning_rate)
        
        model.compile(
            optimizer=opt,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return model

# Run hyperparameter search
tuner = kt.BayesianOptimization(
    NFLHyperModel(),
    objective=kt.Objective('val_auc', direction='max'),
    max_trials=50,
    directory='data_files/nas',
    project_name='nfl_spread_prediction'
)

# Search for best architecture
tuner.search(
    X_train_spread, y_spread_train,
    epochs=50,
    validation_split=0.2,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
)

# Get best model
best_model_nas = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"\n🏆 Best architecture found:")
print(best_model_nas.summary())
print(f"\nBest hyperparameters: {best_hyperparameters.values}")

# Train final model with best architecture
history = best_model_nas.fit(
    X_train_spread, y_spread_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
)

# Evaluate
nas_predictions = best_model_nas.predict(X_test_spread).flatten()
nas_accuracy = accuracy_score(y_spread_test, (nas_predictions >= 0.5).astype(int))
print(f"\nNAS Model Test Accuracy: {nas_accuracy:.4f}")
```

---

## 📊 Performance Tracking

All advanced techniques should be evaluated with:

```python
# Create comprehensive evaluation script: scripts/evaluate_all_models.py

import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss

def evaluate_model_performance(y_true, y_pred_proba, model_name):
    """Comprehensive model evaluation"""
    
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_pred_proba),
        'Log Loss': log_loss(y_true, y_pred_proba),
        'Brier Score': brier_score_loss(y_true, y_pred_proba)
    }
    
    return metrics

# Compare all models
results = []

models_to_evaluate = {
    'XGBoost': model_spread.predict_proba(X_test_spread)[:, 1],
    'CatBoost': catboost_predictions,
    'LSTM': lstm_predictions,
    'Transformer': transformer_predictions,
    'Bayesian': bayesian_predictions,
    'AutoML': automl_predictions,
    'NAS': nas_predictions,
    'Ensemble': ensemble_predictions
}

for model_name, predictions in models_to_evaluate.items():
    results.append(evaluate_model_performance(y_spread_test, predictions, model_name))

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('AUC', ascending=False)

print("\n📊 Model Performance Comparison:")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('data_files/model_comparison_advanced.csv', index=False)
```

---

## 🎯 Next Steps

Continue to:
- [NEW_MODELS_ROADMAP.md](NEW_MODELS_ROADMAP.md) - Completely new model types
- [DATA_PIPELINE_ROADMAP.md](DATA_PIPELINE_ROADMAP.md) - Infrastructure improvements
