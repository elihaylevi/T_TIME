"""
Module: 10_transfer_learning_covid_to_emerson

Description:
Implements a Transfer Learning pipeline to validate the immunological age model on an 
independent external dataset (Emerson cohort).

Protocol:
1. Pre-training: The optimal MLP architecture is trained on the COVID-19 discovery cohort.
2. Feature Alignment: The external Emerson dataset is strictly aligned to the discovery feature space.
3. Fine-tuning: The model is fine-tuned on 60% of the Emerson data using a progressive 
   unfreezing strategy (freezing the backbone for the first 10 epochs).
4. External Validation: Final performance metrics and residuals are calculated strictly 
   on a held-out 25% test subset of the Emerson cohort to prevent data leakage.
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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
TRAIN_CSV = "/home/dsi/levieli8/15kfeatures_to_prediction/train_combined_matrix_pruned_95.csv"   # COVID
TEST_CSV  = "/home/dsi/levieli8/15kfeatures_to_prediction/emerson_combined_matrix_pruned_95.csv" # Emerson

TARGET = "Age"
OUT_ROOT = Path("deepmlp_transfer_final")
(OUT_ROOT / "outputs").mkdir(parents=True, exist_ok=True)
LOGFILE = OUT_ROOT / "outputs/transfer_learning.log"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Training Regimen
EPOCHS_PRETRAIN = 128
EPOCHS_FINETUNE = 128
PATIENCE = 30
FREEZE_EPOCHS = 10

# Final Optimized Hyperparameters
PARAMS = {
    "n_hidden_layers": 3, 
    "hidden_width": 256, 
    "activation": "gelu", 
    "dropout": 0.1, 
    "batch_size": 2048, 
    "lr": 0.02, 
    "weight_decay": 1e-4, 
    "residual": True
}
FINE_TUNE_LR = 0.001

# ==========================================
# 2. UTILITIES & DATA HANDLING
# ==========================================
def log(msg):
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a") as f: f.write(line + "\n")

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def split_features(df):
    meta_cols = [c for c in df.columns if c.strip().lower() in ["sample name", "age", "biological sex"]]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    return meta_cols, feat_cols

def coerce_numeric(df, cols):
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)

class TabDataset(Dataset):
    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ==========================================
# 3. MODEL ARCHITECTURE
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
# 4. TRAINING ROUTINE
# ==========================================
def train_model(model, dl_tr, dl_va, lr, epochs, patience, phase_name):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=PARAMS["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(lr <= 0.02))

    best_mae, best_state, best_epoch = float("inf"), None, 0

    for epoch in range(epochs):
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
            
        sch.step()

        # Validation
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                preds.append(model(xb.to(DEVICE)).cpu().numpy())
                ys.append(yb.numpy())
        
        mae = mean_absolute_error(np.concatenate(ys), np.concatenate(preds))
        
        if mae < best_mae:
            best_mae, best_epoch = mae, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            log(f"[{phase_name}] Early stopping triggered at epoch {epoch}")
            break

    if best_state is not None: model.load_state_dict(best_state)
    log(f"[{phase_name}] Finished. Best Epoch: {best_epoch} | Val MAE: {best_mae:.4f}")
    return model

# ==========================================
# 5. MAIN PIPELINE
# ==========================================
def main():
    set_seed(SEED)
    log("=== Initiating Transfer Learning Pipeline (COVID -> Emerson) ===")

    # ------------------------------------------
    # STAGE 1: PRE-TRAIN ON COVID
    # ------------------------------------------
    log("[STAGE 1] Pre-training on COVID discovery cohort...")
    df_covid = pd.read_csv(TRAIN_CSV)
    df_covid = df_covid.dropna(subset=[TARGET]).reset_index(drop=True)
    
    meta_cv, feats_cv = split_features(df_covid)
    X_cv = coerce_numeric(df_cv, feats_cv).values
    y_cv = df_cv[TARGET].values.astype(np.float32)

    # 85/15 split for pre-training early stopping
    indices_cv = np.arange(len(df_covid))
    np.random.shuffle(indices_cv)
    split_idx = int(0.85 * len(indices_cv))
    
    dl_cv_tr = DataLoader(TabDataset(X_cv[indices_cv[:split_idx]], y_cv[indices_cv[:split_idx]]), batch_size=PARAMS["batch_size"], shuffle=True)
    dl_cv_va = DataLoader(TabDataset(X_cv[indices_cv[split_idx:]], y_cv[indices_cv[split_idx:]]), batch_size=PARAMS["batch_size"])

    model = MLP(len(feats_cv), PARAMS["n_hidden_layers"], PARAMS["hidden_width"], 
                PARAMS["activation"], PARAMS["dropout"], PARAMS["residual"]).to(DEVICE)
    
    model = train_model(model, dl_cv_tr, dl_cv_va, PARAMS["lr"], EPOCHS_PRETRAIN, PATIENCE, "Pre-train (COVID)")

    # ------------------------------------------
    # STAGE 2: DATA PREPARATION & ALIGNMENT (EMERSON)
    # ------------------------------------------
    log("\n[STAGE 2] Aligning external Emerson cohort...")
    df_em = pd.read_csv(TEST_CSV)
    df_em = df_em.rename(columns={c: c.strip() for c in df_em.columns})
    df_em = df_em.dropna(subset=[TARGET]).reset_index(drop=True)

    meta_em, feat_cols_em = split_features(df_em)
    
    # Strict Feature Alignment to COVID
    for m in [f for f in feats_cv if f not in feat_cols_em]: df_em[m] = 0.0
    df_em = df_em[meta_em + feats_cv]  # Enforce order and drop extras
    
    X_em = coerce_numeric(df_em, feats_cv).values
    y_em = df_em[TARGET].values.astype(np.float32)

    # Emerson Split: 60% Train, 15% Val, 25% Pure Test
    n_em = len(df_em)
    indices_em = np.arange(n_em)
    np.random.shuffle(indices_em)
    
    tr_end = int(0.60 * n_em)
    va_end = int(0.75 * n_em)
    
    idx_tr, idx_va, idx_te = indices_em[:tr_end], indices_em[tr_end:va_end], indices_em[va_end:]
    log(f"Emerson Split -> Fine-tune Train: {len(idx_tr)} | Fine-tune Val: {len(idx_va)} | Held-out Test: {len(idx_te)}")

    dl_em_tr = DataLoader(TabDataset(X_em[idx_tr], y_em[idx_tr]), batch_size=PARAMS["batch_size"], shuffle=True)
    dl_em_va = DataLoader(TabDataset(X_em[idx_va], y_em[idx_va]), batch_size=PARAMS["batch_size"])

    # ------------------------------------------
    # STAGE 3: FINE-TUNE ON EMERSON
    # ------------------------------------------
    log("\n[STAGE 3] Fine-tuning on Emerson (Progressive Unfreezing)...")
    
    # Freeze backbone, train only head
    for name, param in model.named_parameters(): param.requires_grad = ("out" in name)
    model = train_model(model, dl_em_tr, dl_em_va, FINE_TUNE_LR, FREEZE_EPOCHS, PATIENCE, "Fine-tune (Frozen Head)")

    # Unfreeze entire network
    for param in model.parameters(): param.requires_grad = True
    model = train_model(model, dl_em_tr, dl_em_va, FINE_TUNE_LR, EPOCHS_FINETUNE, PATIENCE, "Fine-tune (Unfrozen Full)")

    # ------------------------------------------
    # STAGE 4: FINAL EXTERNAL EVALUATION
    # ------------------------------------------
    log("\n[STAGE 4] Evaluating on ZERO-LEAKAGE Emerson Held-out Test Set...")
    model.eval()
    
    X_test, y_test = X_em[idx_te], y_em[idx_te]
    dl_te = DataLoader(TabDataset(X_test, y_test), batch_size=PARAMS["batch_size"])
    
    preds = []
    with torch.no_grad():
        for xb, _ in dl_te: preds.append(model(xb.to(DEVICE)).cpu().numpy())
    y_pred = np.concatenate(preds).ravel()

    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    log(f"--- EXTERNAL VALIDATION METRICS ---")
    log(f"MAE:  {mae:.4f}")
    log(f"RMSE: {rmse:.4f}")
    log(f"R2:   {r2:.4f}")

    # Map back to metadata using the preserved indices
    df_test_meta = df_em.iloc[idx_te].copy()
    df_out = pd.DataFrame({
        "sample name": df_test_meta.get("sample name", np.nan),
        "Biological Sex": df_test_meta.get("Biological Sex", np.nan),
        "Age": y_test,
        "y_pred": y_pred,
        "residual": y_test - y_pred
    })

    out_csv = OUT_ROOT / "outputs/emerson_external_test_preds.csv"
    df_out.to_csv(out_csv, index=False)
    log(f"[DONE] Final predictions and metadata saved to: {out_csv}")

if __name__ == "__main__":
    main()