"""
Module: 08_evaluate_model_on_holdout_test

Description:
Evaluates the final, optimized Multi-Layer Perceptron (MLP) model on the held-out TEST cohort.
This script implements the winning architecture identified during hyperparameter optimization 
(3 layers, GELU activation, residual connections). 

The pipeline performs:
1. Strict feature alignment between TRAIN and TEST sets to prevent data leakage.
2. Training on the full training set with an internal 10% validation split for early stopping.
3. Prediction and residual analysis on the unseen test matrix.
4. Performance stratification by biological sex to assess model bias.
"""

import os
import time
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
TRAIN_CSV = "/home/dsi/levieli8/15kfeatures_to_prediction/train_combined_matrix_pruned_95.csv"
TEST_CSV  = "/home/dsi/levieli8/15kfeatures_to_prediction/test_combined_matrix_pruned_95.csv"
TARGET    = "Age"

# Canonical metadata names for consistency
CANON_META = {
    "sample name": "sample name",
    "age": "Age",
    "biological sex": "Biological Sex",
}
META_SET = set(CANON_META.values())

OUT_ROOT = Path("deepmlp_eval_final")
(OUT_ROOT/"outputs").mkdir(parents=True, exist_ok=True)
LOGFILE  = OUT_ROOT/"outputs/final_eval.log"

SEED     = 42
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# Training Regimen
EPOCHS   = 128
PATIENCE = 30
WARMUP_E = 10
VAL_FRAC = 0.10

# Final Optimized Hyperparameters (Derived from 5-fold CV on Training Set)
# Optimal l0_lambda was determined to be 0.0, indicating distributed immune signaling.
BEST_PARAMS = {
    "n_hidden_layers": 3, 
    "hidden_width": 256, 
    "activation": "gelu", 
    "dropout": 0.1, 
    "batch_size": 2048, 
    "lr": 0.02, 
    "weight_decay": 1e-4, 
    "residual": True, 
    "scheduler": "cosine"
}

# ==========================================
# 2. UTILITIES
# ==========================================
def log(msg):
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a") as f: f.write(line + "\n")

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def standardize_meta_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.shape[1] == 0: return df
    rename = {c: CANON_META[c.strip().lower()] for c in df.columns if c.strip().lower() in CANON_META}
    if rename:
        df = df.rename(columns=rename)
        df = df.loc[:, ~df.columns.duplicated()]
    return df

def split_features(df: pd.DataFrame):
    meta_cols = [c for c in df.columns if c in META_SET]
    feat_cols = [c for c in df.columns if c not in META_SET]
    return meta_cols, feat_cols

def coerce_numeric(df: pd.DataFrame, cols):
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    for c in cols:
        if X[c].isna().any(): X[c] = X[c].fillna(X[c].median())
    return X.astype(np.float32)

class TabDataset(Dataset):
    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
class MLPBlock(nn.Module):
    """Residual block with Batch Normalization and Dropout."""
    def __init__(self, width, activation="gelu", dropout=0.1, residual=True):
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
    """Final MLP architecture for Age Regression."""
    def __init__(self, in_dim, n_layers, width, activation, dropout, residual):
        super().__init__()
        act = nn.ReLU if activation == "relu" else nn.GELU
        head = [nn.Linear(in_dim, width), nn.BatchNorm1d(width), act()]
        if dropout > 0: head.append(nn.Dropout(dropout))
        self.head = nn.Sequential(*head)
        
        self.blocks = nn.Sequential(*[
            MLPBlock(width, activation, dropout, residual) for _ in range(max(0, n_layers - 1))
        ])
        self.out = nn.Linear(width, 1)
        
    def forward(self, x):
        return self.out(self.blocks(self.head(x)))

# ==========================================
# 4. TRAINING & PREDICTION
# ==========================================
def train_final_model(X_df, y_arr, params):
    set_seed(SEED)
    ds_full = TabDataset(X_df.values, y_arr)
    model = MLP(
        in_dim=X_df.shape[1], n_hidden_layers=params["n_hidden_layers"],
        hidden_width=params["hidden_width"], activation=params["activation"],
        dropout=params["dropout"], residual=params["residual"],
    ).to(DEVICE)
    
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS-WARMUP_E) if params["scheduler"] == "cosine" else None
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(params["lr"] <= 0.02))

    val_size = max(1, int(VAL_FRAC * len(ds_full)))
    train_size = len(ds_full) - val_size
    ds_tr, ds_va = random_split(ds_full, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
    
    dl_tr = DataLoader(ds_tr, batch_size=params["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=params["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    best_mae, best_state, best_epoch = float("inf"), None, -1
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                pred = model(xb)
                loss = loss_fn(pred, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            
        if sch and epoch >= WARMUP_E: sch.step()

        model.eval()
        vs = []
        with torch.no_grad():
            for xb, _ in dl_va: vs.append(model(xb.to(DEVICE)).float().cpu().numpy().reshape(-1))
        
        y_va_true = ds_va.dataset.y[ds_va.indices].reshape(-1)
        mae = mean_absolute_error(y_va_true, np.concatenate(vs))
        
        if mae + 1e-6 < best_mae:
            best_mae, best_epoch, epochs_no_improve = mae, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE: break

    if best_state: model.load_state_dict(best_state)
    log(f"[TRAIN] Internal validation optimized. Best Epoch: {best_epoch} | Val MAE: {best_mae:.4f}")
    return model

def predict(model, X_df, batch_size):
    dl = DataLoader(TabDataset(X_df.values, np.zeros((len(X_df),), dtype=np.float32)),
                    batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    model.eval()
    vs = []
    with torch.no_grad():
        for xb, _ in dl: vs.append(model(xb.to(DEVICE)).float().cpu().numpy().reshape(-1))
    return np.concatenate(vs)

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    set_seed(SEED)
    log(f"--- Initiating Final Holdout Test Evaluation ---")

    # Load and clean datasets
    train_raw = standardize_meta_cols(pd.read_csv(TRAIN_CSV))
    test_raw  = standardize_meta_cols(pd.read_csv(TEST_CSV))

    train_meta, train_feats = split_features(train_raw)
    _, test_feats  = split_features(test_raw)
    
    # Align features (Strict train-set order)
    for m in [f for f in train_feats if f not in test_feats]: test_raw[m] = 0.0
    test_raw = test_raw[[c for c in train_meta if c in test_raw.columns] + train_feats]

    # Matrix preparation
    Xtr = coerce_numeric(train_raw, train_feats)
    ytr = pd.to_numeric(train_raw[TARGET], errors="coerce").to_numpy(np.float32)
    mask_tr = np.isfinite(ytr)
    Xtr, ytr = Xtr.loc[mask_tr].reset_index(drop=True), ytr[mask_tr]

    Xte = coerce_numeric(test_raw, train_feats)
    yte = pd.to_numeric(test_raw[TARGET], errors="coerce").to_numpy(np.float32)
    mask_te = np.isfinite(yte)
    Xte, yte = Xte.loc[mask_te].reset_index(drop=True), yte[mask_te]
    test_meta_aligned = test_raw.loc[mask_te].reset_index(drop=True)

    # Execution
    model = train_final_model(Xtr, ytr, BEST_PARAMS)
    y_pred = predict(model, Xte, batch_size=BEST_PARAMS["batch_size"])

    # Global Metrics
    mae = mean_absolute_error(yte, y_pred)
    rmse = math.sqrt(mean_squared_error(yte, y_pred))
    r2 = r2_score(yte, y_pred)
    log(f"[FINAL TEST RESULTS] MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")

    # Residuals & Stratification
    df_out = pd.DataFrame({
        "sample name": test_meta_aligned.get("sample name", np.nan),
        "Biological Sex": test_meta_aligned.get("Biological Sex", np.nan),
        "Age": yte, "y_pred": y_pred, "residual": yte - y_pred
    })
    
    bysex_metrics = []
    if "Biological Sex" in df_out.columns:
        for sex in ["Female", "Male"]:
            sub = df_out[df_out["Biological Sex"] == sex]
            if not sub.empty:
                bysex_metrics.append({
                    "sex": sex, "n": len(sub),
                    "mae": mean_absolute_error(sub["Age"], sub["y_pred"]),
                    "r2": r2_score(sub["Age"], sub["y_pred"])
                })

    # Save Outputs
    df_out.to_csv(OUT_ROOT/"outputs/test_preds.csv", index=False)
    pd.DataFrame(bysex_metrics).to_csv(OUT_ROOT/"outputs/test_metrics_by_sex.csv", index=False)
    log(f"Predictions and stratified metrics saved to {OUT_ROOT}/outputs/")

if __name__ == "__main__":
    main()