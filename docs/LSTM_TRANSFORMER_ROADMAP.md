# LSTM / Transformer Modeling Roadmap

This roadmap describes what is required to build advanced sequence models for NFL prediction tasks in this repo.
It is intentionally detailed and oriented to the current project structure.

## 1. Objective

Build a separate deep-learning model pipeline for one or more of the following targets:
- game-level spread/moneyline/totals
- player prop outcomes (passing yards, rushing yards, receiving yards, TDs)
- live in-game win probability or game-state prediction

These models should be treated as a new track, not a patch on the current XGBoost/LightGBM pipelines.

## 2. Scope selection

Choose a narrow first target before scaling.
Recommended first target:
- **spread prediction** or **player prop prediction** for a single stat line.

Why:
- existing tabular baseline is already stable
- sequence models are easier to validate on a single output
- training compute remains manageable

### Proposed first experiment
- `game-level spread prediction` using team history sequences
- or `player passing yards` using QB game history sequences

## 3. New repo structure

Add a dedicated models folder and training file.
Suggested files:
- `models/lstm_transformer_train.py`
- `models/lstm_transformer_data.py`
- `models/lstm_transformer_model.py`
- `models/lstm_transformer_utils.py`
- `models/configs/lstm_transformer_config.yaml`

If you want to keep docs scope separate, add:
- `docs/LSTM_TRANSFORMER_ROADMAP.md`

## 4. Dependencies

Add deep learning libraries and tooling to `requirements.txt`:
- `torch>=2.0` or `tensorflow>=2.13`
- `torchvision>=0.15` (if using PyTorch)
- `scikit-learn>=1.3`
- `pandas>=2.0`
- `numpy>=1.25`
- `pyyaml>=6.0`
- `tqdm>=4.0`
- `wandb>=0.15` or `mlflow>=2.5` for experiment tracking

Example block for `requirements.txt`:
```text
torch>=2.0
torchvision>=0.15
scikit-learn>=1.3
pyyaml>=6.0
tqdm>=4.0
wandb>=0.15
```

## 5. Data engineering

### 5.1 Build sequence datasets

Create new sequence datasets with ordered historical events.
Two likely formats:
- `team_seq_{target}.parquet` for game-level team sequences
- `player_seq_{target}.parquet` for player prop sequences

### 5.2 Example sequence design

For team/game-level models:
- each sample is a vector sequence of the last `N` games for a team
- include recent game features, matchup features, and situational context
- `X.shape = [batch, seq_len, feature_dim]`
- `y.shape = [batch]`

For player props:
- each sample is a player's prior `N` games
- include age, position, usage stats, matchup defense rank, rest, injury flags

### 5.3 Avoid data leakage

Use only pre-game information for the current prediction:
- no current-game scores or outcomes in features
- rolling sequence window must end at the previous game
- if predicting week `W`, only use games before `W`

### 5.4 Feature candidates

#### Shared features
- team/opponent IDs (embedded)
- home/away indicator
- rest days
- stadium type / surface / roof
- weather flags
- spread line / total line / moneyline
- injury count or impact indicator

#### Sequence features
- prior game score differential
- offensive/defensive yardage
- third-down conversion rate
- red zone success rate
- last 3/5/10 game averages
- target share / snap share (for player props)

### 5.5 Serialization

Save prepared sequences to disk for reproducible training.
Example:
- `data_files/lstm/team_spread_sequences.parquet`
- `data_files/lstm/player_passing_sequences.parquet`

## 6. Model architectures

### 6.1 LSTM prototype

Use a lightweight architecture first.
Example with PyTorch:
```python
import torch
import torch.nn as nn

class GameLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        return self.fc(h_last).squeeze(-1)
```

### 6.2 Transformer prototype

A transformer can model longer-range context.
Example with PyTorch:
```python
class GameTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.mean(dim=1)
        return self.fc(x).squeeze(-1)
```

### 6.3 Embeddings for categorical fields

Use trainable embeddings for IDs:
- team
- opponent
- stadium
- surface
- coach
- player position

Example:
```python
self.team_embedding = nn.Embedding(num_teams, emb_dim)
```

Concatenate embeddings with numeric sequence features.

## 7. Training pipeline

### 7.1 New scripts

Suggested files:
- `models/lstm_transformer_data.py` → sequence dataset builder
- `models/lstm_transformer_model.py` → model definitions
- `models/lstm_transformer_train.py` → training loop
- `models/lstm_transformer_utils.py` → metrics, config, logging

### 7.2 Config-driven training

Use YAML or JSON config for hyperparameters:
```yaml
model:
  type: lstm
  input_dim: 32
  hidden_dim: 128
  num_layers: 2
  dropout: 0.2
training:
  batch_size: 64
  epochs: 30
  lr: 1e-3
  weight_decay: 1e-5
  device: cuda
```

### 7.3 Loss and metrics

- binary classification loss: `BCEWithLogitsLoss`
- evaluation metrics:
  - ROC-AUC
  - precision / recall / F1
  - calibration (e.g. reliability curve)
  - betting ROI / EV on validation set

### 7.4 Sample training loop

```python
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        x, y = batch['features'].to(device), batch['target'].to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)
```

### 7.5 Validation loop

```python
def eval_epoch(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch['features'].to(device), batch['target'].to(device)
            preds = model(x)
            all_preds.append(preds.sigmoid().cpu())
            all_labels.append(y.cpu())
    return torch.cat(all_preds), torch.cat(all_labels)
```

## 8. Experiment tracking

Track experiments with a lightweight system.
Suggested options:
- `wandb` for online logs
- `mlflow` for offline tracking
- simple CSV/JSON logs if you want minimal dependency

Record:
- model config
- learning curves
- validation metrics
- training time
- checkpoint path

## 9. Evaluation strategy

### 9.1 Walk-forward validation

Use a rolling training window and test on subsequent seasons.
Example:
- train on 2020-2022, validate on 2023
- then train on 2020-2023, validate on 2024

### 9.2 Baseline comparison

Compare deep models to existing XGBoost/LightGBM baseline.
Measure:
- ROC-AUC
- F1
- calibration error
- simulated betting ROI

### 9.3 Calibration

Use probability calibration checks.
- reliability curve
- Brier score
- isotonic calibration or Platt scaling if needed

## 10. Deployment and integration

### 10.1 Save model artifacts

Save model checkpoints alongside existing artifacts:
- `player_props/models/lstm_spread.pt`
- `player_props/models/transformer_spread.pt`
- `data_files/lstm/validation_metrics.json`

### 10.2 Inference pipeline

Add a new inference wrapper that:
- loads sequence features for upcoming games
- preprocesses to the same shape as training data
- returns probability outputs

Example entrypoint:
- `models/lstm_transformer_predict.py`

### 10.3 Pipeline integration

If performance justifies it, integrate into the main build pipeline:
- `build_and_train_pipeline.py` can call `models/lstm_transformer_train.py`
- `predictions.py` can optionally load the new sequence model outputs

## 11. Practical first tasks

1. **Choose target and architecture**
   - initial target: spread or player passing yards
   - initial architecture: LSTM

2. **Build sequence dataset**
   - implement `models/lstm_transformer_data.py`
   - serialize to Parquet

3. **Prototype model**
   - implement `models/lstm_transformer_model.py`
   - train one experiment locally

4. **Validate**
   - compare against current baseline
   - measure ROC-AUC and ROI

5. **Iterate**
   - tune sequence length, hidden size, embedding dimensions
   - try Transformer only after LSTM baseline proves useful

## 12. Code suggestions

### 12.1 Data builder skeleton

```python
import pandas as pd
import numpy as np

from pathlib import Path

DATA_DIR = Path('data_files')


def build_game_sequences(df, seq_len=10):
    sequences = []
    targets = []
    for team in df['home_team'].append(df['away_team']).unique():
        team_games = df[(df['home_team'] == team) | (df['away_team'] == team)].sort_values(['season', 'week'])
        for idx in range(seq_len, len(team_games)):
            window = team_games.iloc[idx-seq_len:idx]
            seq_features = window[FEATURE_COLUMNS].values.astype(np.float32)
            sequences.append(seq_features)
            targets.append(team_games.iloc[idx]['spreadCovered'])
    return np.asarray(sequences), np.asarray(targets)
```

### 12.2 Simple PyTorch dataset

```python
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.sequences[idx], dtype=torch.float32),
            'target': torch.tensor(self.targets[idx], dtype=torch.float32),
        }
```

### 12.3 Save checkpoints

```python
torch.save({
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'config': config,
}, checkpoint_path)
```

## 13. Why this should be off-season work

- deep sequence models require new dataset design and validation
- training is more expensive than current tree-based models
- stable production integration should wait until performance is proven
- this should be done when you have time to iterate rather than during season operations

## 14. Recommended next step

Start with a single focused proof-of-concept:
- build one LSTM for one target
- compare directly with the current XGBoost baseline
- if it helps, then expand to Transformer and additional targets
