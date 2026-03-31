"""
Module: 07_deep_learning_regression_with_l0_gates

Description:
This script implements a Deep Learning regression pipeline using a Multi-Layer Perceptron (MLP) 
augmented with a Hard-Concrete Feature Gate (L0 Regularization). 
The architecture performs simultaneous age prediction and automated feature selection. 
Includes a robust resume mechanism to handle large hyperparameter grid searches 
across multiple server sessions.
"""

import os
import sys
import json
import time
import math
import random
from datetime import datetime
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. GLOBAL CONFIGURATION
# ==========================================
DATA_CSV    = "/home/dsi/levieli8/15kfeatures_to_prediction/train_combined_matrix_pruned_95.csv"
TARGET      = "Age"
DROP_COLS   = ["sample name", "Biological Sex"]

RUN_ROOT    = Path("deep_learning_runs_gates")
(RUN_ROOT/"logs").mkdir(parents=True, exist_ok=True)
(RUN_ROOT/"outputs").mkdir(parents=True, exist_ok=True)
LOGFILE     = RUN_ROOT/"logs/run.log"

N_SPLITS    = 5
SEED        = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS      = 128
PATIENCE    = 30
WARMUP_E    = 10  # Warmup epochs before cosine annealing

# Hyperparameter search space configuration
MAX_TRIALS  = 1200
FIXED_RESIDUAL  = True
FIXED_SCHEDULER = "cosine"

# L0 Gate Parameters
GATE_INIT_P = 0.5
HC_TEMP     = 0.33
HC_LOW      = -0.1
HC_HIGH     = 1.1

SPACE = {
    "n_hidden_layers": [2, 3],
    "hidden_width":    [256, 512],
    "activation":      ["relu", "gelu"],
    "dropout":         [0.0, 0.1],
    "batch_size":      [1024, 2048],
    "lr":              [0.003, 0.005, 0.01, 0.02],
    "weight_decay":    [0.0, 1e-4, 5e-4],
    "l0_lambda":       [0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
}

# Parameters used for resume matching
PARAM_COLS = ["n_hidden_layers", "hidden_width", "activation", "dropout", 
              "batch_size", "lr", "weight_decay", "l0_lambda"]

# ==========================================
# 2. UTILITIES & DATA HANDLING
# ==========================================
def log(msg):
    ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a") as f:
        f.write(line + "\n")

def canon_str(v):
    """Canonical string formatting for robust resume matching."""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.12g}" if math.isfinite(v) else str(v)
    return str(v)

class TabDataset(Dataset):
    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ==========================================
# 3. MODEL ARCHITECTURE (L0 GATES & MLP)
# ==========================================
class FeatureGate(nn.Module):
    """Hard-Concrete Feature Gate for differentiable feature selection."""
    def __init__(self, in_dim, init_p=0.5, temp=0.33, low=-0.1, high=1.1, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.temp, self.low, self.high = temp, low, high
        logit = math.log(init_p) - math.log(1 - init_p)
        self.log_alpha = nn.Parameter(torch.full((in_dim,), logit))

    def forward(self, x):
        if not self.enabled: return x
        if self.training:
            u = torch.rand(x.shape, device=x.device)
            s = torch.sigmoid((self.log_alpha + torch.log(u) - torch.log1p(-u)) / self.temp)
        else:
            s = torch.sigmoid(self.log_alpha)
        
        s = s * (self.high - self.low) + self.low
        z = torch.clamp(s, 0.0, 1.0)
        return x * z

    def expected_l0(self):
        """Returns the expected number of active features."""
        return torch.sigmoid(self.log_alpha - self.temp * math.log(-self.low / self.high))

class MLPBlock(nn.Module):
    def __init__(self, width, activation="relu", dropout=0.0, residual=False):
        super().__init__()
        act = nn.ReLU if activation == "relu" else nn.GELU
        self.fc = nn.Linear(width, width)
        self.bn = nn.BatchNorm1d(width)
        self.act = act()
        self.do = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, x):
        z = self.do(self.act(self.bn(self.fc(x))))
        return x + z if self.residual else z

class MLP(nn.Module):
    def __init__(self, in_dim, n_layers, width, activation, dropout, residual, l0_enabled):
        super().__init__()
        self.gate = FeatureGate(in_dim, enabled=l0_enabled)
        act = nn.ReLU if activation == "relu" else nn.GELU
        
        self.head = nn.Sequential(nn.Linear(in_dim, width), nn.BatchNorm1d(width), act())
        self.blocks = nn.ModuleList([MLPBlock(width, activation, dropout, residual) for _ in range(n_layers-1)])
        self.out = nn.Linear(width, 1)

    def forward(self, x):
        x = self.head(self.gate(x))
        for block in self.blocks: x = block(x)
        return self.out(x)

# ==========================================
# 4. TRAINING & CROSS-VALIDATION
# ==========================================
def train_one_fold(X_tr, y_tr, X_va, y_va, params):
    dl_tr = DataLoader(TabDataset(X_tr, y_tr), batch_size=params["batch_size"], shuffle=True)
    dl_va = DataLoader(TabDataset(X_va, y_va), batch_size=params["batch_size"])

    model = MLP(X_tr.shape[1], params["n_hidden_layers"], params["hidden_width"], 
                params["activation"], params["dropout"], FIXED_RESIDUAL, 
                params["l0_lambda"] > 0).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-WARMUP_E) if FIXED_SCHEDULER == "cosine" else None
    
    loss_fn = nn.MSELoss()
    best_mae, best_epoch, epochs_no_improve = float("inf"), -1, 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            mse = loss_fn(pred, yb)
            l0_pen = model.gate.expected_l0().sum() * params["l0_lambda"] if model.gate.enabled else 0
            (mse + l0_pen).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        if scheduler and epoch >= WARMUP_E: scheduler.step()

        # Validation
        model.eval()
        preds_va = []
        with torch.no_grad():
            for xb, _ in dl_va: preds_va.append(model(xb.to(DEVICE)).cpu().numpy())
        
        mae = mean_absolute_error(y_va, np.concatenate(preds_va))
        if mae < best_mae:
            best_mae, best_epoch, best_state = mae, epoch, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE: break

    model.load_state_dict(best_state)
    model.eval()
    final_preds = []
    with torch.no_grad():
        for xb, _ in dl_va: final_preds.append(model(xb.to(DEVICE)).cpu().numpy())
    
    y_pred = np.concatenate(final_preds).ravel()
    active_feats = (model.gate.deterministic_gate() >= 0.5).sum().item() if model.gate.enabled else X_tr.shape[1]
    
    return mean_absolute_error(y_va, y_pred), math.sqrt(mean_squared_error(y_va, y_pred)), r2_score(y_va, y_pred), best_epoch, active_feats

# ==========================================
# 5. MAIN EXECUTION (RESUME LOGIC)
# ==========================================
def main():
    log(f"Starting Phase 07: Deep Learning Regression (L0 Gates)")
    df = pd.read_csv(DATA_CSV)
    X = df.drop(columns=set(DROP_COLS + [TARGET])).apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)

    results_path = RUN_ROOT/"outputs/grid_results_gates.csv"
    done_keys = set()
    if results_path.exists():
        prev = pd.read_csv(results_path)
        for _, row in prev.iterrows():
            done_keys.add(tuple(canon_str(row[c]) for c in PARAM_COLS))

    combos = [dict(zip(SPACE.keys(), v)) for v in itertools.product(*SPACE.values())]
    random.Random(SEED).shuffle(combos)
    combos = combos[:MAX_TRIALS]
    
    log(f"Resume: {len(done_keys)} trials already completed. Remaining: {len(combos) - len(done_keys)}")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for i, p in enumerate(combos, 1):
        tupkey = tuple(canon_str(p[c]) for c in PARAM_COLS)
        if tupkey in done_keys: continue

        log(f"Trial {i}/{len(combos)}: {json.dumps(p)}")
        fold_results = []
        for fold, (tr, va) in enumerate(kf.split(X), 1):
            fold_results.append(train_one_fold(X[tr], y[tr], X[va], y[va], p))

        # Aggregate Metrics
        cv_mae = np.mean([r[0] for r in fold_results])
        cv_rmse = np.mean([r[1] for r in fold_results])
        cv_r2 = np.mean([r[2] for r in fold_results])
        avg_feats = np.mean([r[4] for r in fold_results])

        # Write to CSV immediately (Crash-safe)
        header = not results_path.exists()
        pd.DataFrame([list(p.values()) + [cv_mae, cv_rmse, cv_r2, avg_feats]], 
                     columns=list(p.keys()) + ["cv_mae", "cv_rmse", "cv_r2", "active_feats"]).to_csv(results_path, mode='a', index=False, header=header)
        log(f"Result: MAE={cv_mae:.3f}, Features={avg_feats:.1f}")

if __name__ == "__main__":
    main()