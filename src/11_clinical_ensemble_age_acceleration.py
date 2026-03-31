"""
Module: 11_clinical_ensemble_age_acceleration

Description:
Trains an ensemble of Deep Learning models strictly on a cohort of healthy individuals 
to establish a pure immunological aging baseline. The ensemble utilizes 5-Fold Cross 
Validation to generate robust predictions. 

Once trained, the ensemble projects these predictions onto a separate cohort of patients 
with various clinical conditions to calculate "Age Acceleration" (age_accel), defined as 
the difference between the model's predicted immunological age and the patient's 
chronological age.
"""

import os
import time
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
TRAIN_CSV = "/home/dsi/levieli8/15kfeatures_to_prediction/train_rearranged_healthy.csv"
TEST_CSV  = "/home/dsi/levieli8/15kfeatures_to_prediction/test_rearranged_conditions.csv"
TARGET    = "Age"

# Canonical metadata names to prevent feature leakage
CANON_META = {
    "sample name": "sample name",
    "age": "Age",
    "biological sex": "Biological Sex",
}
META_SET = set(CANON_META.values())

OUT_ROOT = Path("clinical_case_study_final")
(OUT_ROOT/"outputs").mkdir(parents=True, exist_ok=True)
LOGFILE  = OUT_ROOT/"outputs/case_study.log"

SEED     = 42
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS  = 5

# Training Regimen
EPOCHS   = 128
PATIENCE = 30
WARMUP_E = 10
VAL_FRAC = 0.10

# Winning Hyperparameters for the Healthy Baseline
PARAMS = {
    "n_hidden_layers": 3, 
    "hidden_width": 256, 
    "activation": "gelu", 
    "dropout": 0.1, 
    "batch_size": 2048, 
    "lr": 0.005,
    "weight_decay": 1e-4, 
    "residual": True, 
    "scheduler": "cosine"
}

# ==========================================
# 2. UTILITIES & DATA PREPARATION
# ==========================================
def log(msg):
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    try:
        with open(LOGFILE, "a") as f: f.write(line + "\n")
    except Exception: pass

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
    meta_cols = [c for c in df.columns if c in META_SET or c in ['sample_id_clean', 'sample_name']]
    feat_cols = [c for c in df.columns if c not in meta_cols and c not in ['y_pred', 'residual', 'age_accel']]
    return meta_cols, feat_cols

def coerce_numeric(df: pd.DataFrame, cols):
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    # Impute missing TCR sequences/K-mers with 0.0
    X = X.fillna(0.0)
    return X.astype(np.float32)

class TabDataset(Dataset):
    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ==========================================
# 3. NEURAL NETWORK ARCHITECTURE
# ==========================================
class MLPBlock(nn.Module):
    def __init__(self, width, activation="gelu", dropout=0.1, residual=True):
        super().__init__()
        act = nn.ReLU if activation == "relu" else nn.GELU
        self.fc = nn.Linear(width, width)
        self.bn = nn.BatchNorm1d(width)
        self.act = act()
        self.do = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = residual
    def forward(self, x):
        z = self.do(self.act(self.bn(self.fc(x))))
        return x + z if self.residual else z

class MLP(nn.Module):
    def __init__(self, in_dim, n_hidden_layers, hidden_width, activation, dropout, residual):
        super().__init__()
        act = nn.ReLU if activation == "relu" else nn.GELU
        head = [nn.Linear(in_dim, hidden_width), nn.BatchNorm1d(hidden_width), act()]
        if dropout > 0: head.append(nn.Dropout(dropout))
        self.head = nn.Sequential(*head)
        
        blocks = [MLPBlock(hidden_width, activation, dropout, residual) for _ in range(max(0, n_hidden_layers - 1))]
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Linear(hidden_width, 1)
    def forward(self, x):
        return self.out(self.blocks(self.head(x)))

# ==========================================
# 4. TRAINING & PREDICTION PIPELINE
# ==========================================
def train_with_optional_val(X_df, y_arr, params):
    set_seed(SEED)
    ds_full = TabDataset(X_df.values, y_arr)
    model = MLP(X_df.shape[1], params["n_hidden_layers"], params["hidden_width"], 
                params["activation"], params["dropout"], params["residual"]).to(DEVICE)
    
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS - WARMUP_E)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    val_size = max(1, int(VAL_FRAC * len(ds_full)))
    tr_size = len(ds_full) - val_size
    ds_tr, ds_va = random_split(ds_full, [tr_size, val_size], generator=torch.Generator().manual_seed(SEED))
    
    dl_tr = DataLoader(ds_tr, batch_size=params["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=params["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    best_mae, best_state, epochs_no_improve = float("inf"), None, 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                pred = model(xb)
                loss = loss_fn(pred, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt); scaler.update()
        
        if sch and epoch >= WARMUP_E: sch.step()

        # Validation
        model.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                out = model(xb.to(DEVICE))
                v_preds.append(out.cpu().numpy().reshape(-1))
                v_targets.append(yb.numpy().reshape(-1))
        
        mae = mean_absolute_error(np.concatenate(v_targets), np.concatenate(v_preds))
        if mae < best_mae:
            best_mae, best_epoch, epochs_no_improve = mae, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE: break

    if best_state: model.load_state_dict(best_state)
    return model

def predict(model, X_df, batch_size):
    dl = DataLoader(TabDataset(X_df.values, np.zeros(len(X_df))), batch_size=batch_size, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in dl:
            preds.append(model(xb.to(DEVICE)).cpu().numpy().reshape(-1))
    return np.concatenate(preds)

# ==========================================
# 5. MAIN ENSEMBLE EXECUTION
# ==========================================
def main():
    set_seed(SEED)
    log(f"[START] Training on HEALTHY: {TRAIN_CSV}")
    log(f"[START] Predicting on CONDITIONS: {TEST_CSV}")

    # Load and clean datasets
    train_raw = standardize_meta_cols(pd.read_csv(TRAIN_CSV))
    test_raw  = standardize_meta_cols(pd.read_csv(TEST_CSV))
    
    _, feat_cols = split_features(train_raw)
    
    # Prepare matrices
    X_train = coerce_numeric(train_raw, feat_cols)
    y_train = pd.to_numeric(train_raw[TARGET], errors="coerce").fillna(train_raw[TARGET].median()).values
    
    X_test = coerce_numeric(test_raw, feat_cols)
    
    # K-Fold Ensemble Training
    log(f"Initiating {N_FOLDS}-Fold Ensemble Training...")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    test_preds_accum = np.zeros(len(X_test))

    for fold_i, (tr_idx, _) in enumerate(kf.split(X_train), 1):
        log(f"--> Training Ensemble Model {fold_i}/{N_FOLDS}...")
        X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
        
        model = train_with_optional_val(X_tr, y_tr, PARAMS)
        
        # Predict on the clinical conditions cohort and accumulate
        test_preds_accum += predict(model, X_test, PARAMS["batch_size"]) / N_FOLDS

    # Save final ensemble results
    test_raw["y_pred"] = test_preds_accum
    test_raw["age_accel"] = test_raw["y_pred"] - test_raw["Age"]
    
    out_path = OUT_ROOT / "outputs/clinical_test_predictions.csv"
    test_raw.to_csv(out_path, index=False)
    
    log(f"[FINISH] Mean Age Acceleration across clinical cohort: {test_raw['age_accel'].mean():+.2f} years")
    log(f"Results successfully saved to {out_path}")

if __name__ == "__main__":
    main()