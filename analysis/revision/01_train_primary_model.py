"""
01_train_primary_model.py

Trains the verified MLP (MAE ~ 7.475, matches archive) on the FULL
train_combined_matrix_pruned_95.csv, predicts on the full held-out
test_combined_matrix_pruned_95.csv (818 samples), and saves per-sample
predictions. This is Model M1 - used for:
  - Figure panel a (Primary cohort scatter, MAE/R2)
  - Sex-gap analysis (panels d, e) via 03_sex_gap_analysis.py

No leakage handling here: this matches the archived/manuscript Primary
cohort exactly (n=818). Clinical-patient exclusion (for the AAR
pipeline) is a SEPARATE model, trained in 04_clinical_aar_pipeline.py.

Output: results/revision/test_preds.csv
  columns: sample name, Biological Sex, Age, y_pred, residual
  NOTE: residual = Age - y_pred (NOT y_pred - Age). Downstream scripts
  that need the y_pred-Age convention flip the sign explicitly - this
  is documented at each point it happens, do not "fix" it here.
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
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


REPO = Path(__file__).resolve().parents[2]


def _find(name):
    _repo = REPO
    # results/revision comes BEFORE data/: test_preds.csv exists in both, and the
    # revision pipeline must consume its own retrained predictions, not the legacy ones.
    for d in [Path("."), _repo / "results" / "revision", Path("data"), _repo, _repo / "data",
              _repo / "data" / "external", _repo / "data" / "external" / "reference_db",
              Path(__file__).resolve().parent,
              Path("/content"), Path("/content/data"), Path("/content/outputs"),
              Path("/content/deepmlp_eval_final/outputs")]:
        p = d / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(name)


TRAIN_CSV = _find("train_combined_matrix_pruned_95.csv")
TEST_CSV = _find("test_combined_matrix_pruned_95.csv")
print("TRAIN_CSV =", TRAIN_CSV)
print("TEST_CSV  =", TEST_CSV)
TARGET = "Age"

CANON_META = {"sample name": "sample name", "age": "Age", "biological sex": "Biological Sex"}
META_SET = set(CANON_META.values())

# Outputs land in results/revision/ (repo-anchored, not cwd-relative). Before
# 2026-09-08 this wrote to the same directory that
# 08_evaluate_model_on_holdout_test.py used to write to (now archived out of the repo),
# silently clobbered the other. 01 is the canonical primary model, so it moved; 08
# keeps the old path and is historical. Override with TTIME_REVISION_OUT.
OUT_ROOT = Path(os.environ.get("TTIME_REVISION_OUT", REPO / "results" / "revision"))
OUT_ROOT.mkdir(parents=True, exist_ok=True)
LOGFILE = OUT_ROOT / "final_eval.log"
for _f in ("test_preds.csv", "test_metrics_by_sex.csv"):
    if (OUT_ROOT / _f).exists():
        print(f"[!] overwriting {OUT_ROOT / _f}")

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, PATIENCE, WARMUP_E, VAL_FRAC = 128, 30, 10, 0.10
BEST_PARAMS = {"n_hidden_layers": 3, "hidden_width": 256, "activation": "gelu",
               "dropout": 0.1, "batch_size": 2048, "lr": 0.02, "weight_decay": 1e-4,
               "residual": True, "scheduler": "cosine"}


def log(msg):
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a") as f:
        f.write(line + "\n")


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def standardize_meta_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.shape[1] == 0:
        return df
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
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    return X.astype(np.float32)


class TabDataset(Dataset):
    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLPBlock(nn.Module):
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
    def __init__(self, in_dim, n_layers, width, activation, dropout, residual):
        super().__init__()
        act = nn.ReLU if activation == "relu" else nn.GELU
        head = [nn.Linear(in_dim, width), nn.BatchNorm1d(width), act()]
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        self.head = nn.Sequential(*head)
        self.blocks = nn.Sequential(*[MLPBlock(width, activation, dropout, residual)
                                       for _ in range(max(0, n_layers - 1))])
        self.out = nn.Linear(width, 1)

    def forward(self, x):
        return self.out(self.blocks(self.head(x)))


def train_final_model(X_df, y_arr, params):
    set_seed(SEED)
    ds_full = TabDataset(X_df.values, y_arr)
    model = MLP(in_dim=X_df.shape[1], n_layers=params["n_hidden_layers"],
                width=params["hidden_width"], activation=params["activation"],
                dropout=params["dropout"], residual=params["residual"]).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS - WARMUP_E) \
        if params["scheduler"] == "cosine" else None
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(params["lr"] <= 0.02))

    val_size = max(1, int(VAL_FRAC * len(ds_full)))
    train_size = len(ds_full) - val_size
    ds_tr, ds_va = random_split(ds_full, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

    dl_tr = DataLoader(ds_tr, batch_size=params["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=params["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

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

        if sch and epoch >= WARMUP_E:
            sch.step()

        model.eval()
        vs = []
        with torch.no_grad():
            for xb, _ in dl_va:
                vs.append(model(xb.to(DEVICE)).float().cpu().numpy().reshape(-1))

        y_va_true = ds_va.dataset.y[ds_va.indices].reshape(-1)
        mae = mean_absolute_error(y_va_true, np.concatenate(vs))

        if mae + 1e-6 < best_mae:
            best_mae, best_epoch, epochs_no_improve = mae, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    log(f"[TRAIN] Internal validation optimized. Best Epoch: {best_epoch} | Val MAE: {best_mae:.4f}")
    return model


def predict(model, X_df, batch_size):
    dl = DataLoader(TabDataset(X_df.values, np.zeros((len(X_df),), dtype=np.float32)),
                     batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model.eval()
    vs = []
    with torch.no_grad():
        for xb, _ in dl:
            vs.append(model(xb.to(DEVICE)).float().cpu().numpy().reshape(-1))
    return np.concatenate(vs)


def main():
    set_seed(SEED)
    log("--- Initiating Final Holdout Test Evaluation ---")

    train_raw = standardize_meta_cols(pd.read_csv(TRAIN_CSV))
    test_raw = standardize_meta_cols(pd.read_csv(TEST_CSV))

    train_meta, train_feats = split_features(train_raw)
    _, test_feats = split_features(test_raw)

    for m in [f for f in train_feats if f not in test_feats]:
        test_raw[m] = 0.0
    test_raw = test_raw[[c for c in train_meta if c in test_raw.columns] + train_feats]

    Xtr = coerce_numeric(train_raw, train_feats)
    ytr = pd.to_numeric(train_raw[TARGET], errors="coerce").to_numpy(np.float32)
    mask_tr = np.isfinite(ytr)
    Xtr, ytr = Xtr.loc[mask_tr].reset_index(drop=True), ytr[mask_tr]

    Xte = coerce_numeric(test_raw, train_feats)
    yte = pd.to_numeric(test_raw[TARGET], errors="coerce").to_numpy(np.float32)
    mask_te = np.isfinite(yte)
    Xte, yte = Xte.loc[mask_te].reset_index(drop=True), yte[mask_te]
    test_meta_aligned = test_raw.loc[mask_te].reset_index(drop=True)

    print(f"[*] n_features = {Xtr.shape[1]}  (archive numbers.json says 5,459)")

    model = train_final_model(Xtr, ytr, BEST_PARAMS)
    y_pred = predict(model, Xte, batch_size=BEST_PARAMS["batch_size"])

    mae = mean_absolute_error(yte, y_pred)
    rmse = math.sqrt(mean_squared_error(yte, y_pred))
    r2 = r2_score(yte, y_pred)
    log(f"[FINAL TEST RESULTS] MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")

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

    df_out.to_csv(OUT_ROOT / "test_preds.csv", index=False)
    pd.DataFrame(bysex_metrics).to_csv(OUT_ROOT / "test_metrics_by_sex.csv", index=False)
    log(f"Predictions and stratified metrics saved to {OUT_ROOT}")


if __name__ == "__main__":
    main()
