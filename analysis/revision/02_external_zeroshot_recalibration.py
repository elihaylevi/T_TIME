"""
02_external_zeroshot_recalibration.py

Trains Model M3 (same architecture/data as M1 in script 01, but a
SEPARATE training run - by design, not a bug: torch training is not
bitwise-reproducible across runs even with the same seed) on the full
training matrix, then evaluates zero-shot on the external Emerson
cohort and fits a two-parameter affine recalibration (20% calibration
slice, 80% evaluation, seed 42).

Used for:
  - Figure panel b (external zero-shot scatter)
  - Figure panel c (external + recalibration scatter)
  - 03_sex_gap_analysis.py, part 2 (external sex gap)

NOTE on EMER file: Sol's original script pointed at
emerson_combined_matrix.csv (unpruned). If that file isn't found, this
falls back to emerson_combined_matrix_pruned_95.csv - which is known
to give a *better* zero-shot MAE (~10) than the archived manuscript
value (13.5). This discrepancy is unresolved; flagged in
the archived RECONCILIATION_REPORT.md (outside this repo,
in ~/Downloads/ttime_archive/docs_internal/). Do not treat panel b's numbers as final
without locating the unpruned matrix.

Outputs (written to the current directory, like scripts 03-06):
  - emerson_zeroshot_preds.csv       (all 495, for panel b)
  - emerson_recalibrated_persample.csv (396 eval-slice, for panel c/f)
  - external_validation.json
"""
import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

CANON_META = {"sample name": "sample name", "age": "Age", "biological sex": "Biological Sex"}
META_SET = set(CANON_META.values())
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, PATIENCE, WARMUP_E, VAL_FRAC = 128, 30, 10, 0.10
BEST_PARAMS = {"n_hidden_layers": 3, "hidden_width": 256, "activation": "gelu",
               "dropout": 0.1, "batch_size": 2048, "lr": 0.02, "weight_decay": 1e-4,
               "residual": True, "scheduler": "cosine"}


def set_seed(seed=SEED):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def standardize_meta_cols(df):
    if df is None or df.shape[1] == 0:
        return df
    rename = {c: CANON_META[c.strip().lower()] for c in df.columns if c.strip().lower() in CANON_META}
    if rename:
        df = df.rename(columns=rename)
        df = df.loc[:, ~df.columns.duplicated()]
    return df


def split_features(df):
    meta_cols = [c for c in df.columns if c in META_SET]
    feat_cols = [c for c in df.columns if c not in META_SET]
    return meta_cols, feat_cols


def coerce_numeric(df, cols):
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
    model = MLP(X_df.shape[1], params["n_hidden_layers"], params["hidden_width"],
                params["activation"], params["dropout"], params["residual"]).to(DEVICE)
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
    best_mae, best_state, best_epoch, no_imp = float("inf"), None, -1, 0
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                loss = loss_fn(model(xb), yb)
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
            best_mae, best_epoch, no_imp = mae, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    print(f"[TRAIN] Best epoch {best_epoch} | Val MAE {best_mae:.4f}")
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


def _find(name):
    _repo = Path(__file__).resolve().parents[2]
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


def metrics(y, p):
    return dict(n=int(len(y)), MAE=round(float(mean_absolute_error(y, p)), 3),
                RMSE=round(float(math.sqrt(mean_squared_error(y, p))), 3),
                R2=round(float(r2_score(y, p)), 3),
                Pearson=round(float(pearsonr(y, p)[0]), 3))


def main():
    TRAIN = _find("train_combined_matrix_pruned_95.csv")
    try:
        EMER = _find("emerson_combined_matrix.csv")
        print("[*] Using emerson_combined_matrix.csv (Sol's original filename)")
    except FileNotFoundError:
        EMER = _find("emerson_combined_matrix_pruned_95.csv")
        print("[!] emerson_combined_matrix.csv not found - using emerson_combined_matrix_pruned_95.csv")
        print("[!] KNOWN ISSUE: this gives a lower zero-shot MAE than the archived manuscript value (~10 vs 13.5).")
    print("TRAIN =", TRAIN)
    print("EMER  =", EMER)

    tr = standardize_meta_cols(pd.read_csv(TRAIN))
    em = standardize_meta_cols(pd.read_csv(EMER))
    _, train_feats = split_features(tr)
    for f in [c for c in train_feats if c not in em.columns]:
        em[f] = 0.0
    Xtr = coerce_numeric(tr, train_feats)
    ytr = pd.to_numeric(tr['Age'], errors='coerce').to_numpy(np.float32)
    mtr = np.isfinite(ytr)
    Xtr, ytr = Xtr.loc[mtr].reset_index(drop=True), ytr[mtr]
    em = em[np.isfinite(pd.to_numeric(em['Age'], errors='coerce'))].reset_index(drop=True)
    Xem = coerce_numeric(em, train_feats)
    yem = pd.to_numeric(em['Age'], errors='coerce').to_numpy(np.float32)

    print(f"[*] n_features = {Xtr.shape[1]}  n_train = {len(Xtr)}  n_emerson = {len(Xem)}")

    model = train_final_model(Xtr, ytr, BEST_PARAMS)
    pem = predict(model, Xem, batch_size=BEST_PARAMS['batch_size'])
    zs = metrics(yem, pem)

    lr = LinearRegression().fit(pem.reshape(-1, 1), yem)
    slope, intercept = float(lr.coef_[0]), float(lr.intercept_)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(yem))
    ncal = int(0.2 * len(yem))
    cal, ev = idx[:ncal], idx[ncal:]
    a = LinearRegression().fit(pem[cal].reshape(-1, 1), yem[cal])
    pem_recal = a.predict(pem[ev].reshape(-1, 1))
    recal = metrics(yem[ev], pem_recal)
    zs_ev = metrics(yem[ev], pem[ev])

    out = dict(
        zero_shot_full=zs,
        calibration_line={'slope': round(slope, 3), 'intercept': round(intercept, 2)},
        affine_recalibration={'calib_slice_n': ncal, 'eval_n': len(ev),
                               'zero_shot_on_eval': zs_ev, 'recalibrated_on_eval': recal,
                               'recal_a': round(float(a.coef_[0]), 3), 'recal_b': round(float(a.intercept_), 2)},
        emerson_age_range=[float(np.min(yem)), float(np.max(yem))],
    )
    print(json.dumps(out, indent=2))

    # C9 (2026-09-08): write into the CURRENT directory, not ./outputs/.
    # Scripts 03-06 write to '.', so the old split left a fresh run's artefacts in two
    # places; _find() searches '.' first and the repo second, which meant the later
    # scripts silently fell back to the SHIPPED files instead of the fresh ones.
    pd.DataFrame({'sample name': em['sample name'], 'Age': yem, 'y_pred_zeroshot': pem}).to_csv(
        "emerson_zeroshot_preds.csv", index=False)
    Path("external_validation.json").write_text(json.dumps(out, indent=2))

    sex_col = em['Biological Sex'] if 'Biological Sex' in em.columns else pd.Series([np.nan] * len(em))
    df_recal = pd.DataFrame({
        'sample name': em['sample name'].iloc[ev].values,
        'Sex': sex_col.iloc[ev].values,
        'Age': yem[ev],
        'y_pred_zeroshot': pem[ev],
        'y_pred_recal': pem_recal,
        'residual_recal': pem_recal - yem[ev],
    })
    df_recal.to_csv("emerson_recalibrated_persample.csv", index=False)
    print(f"[*] Saved emerson_recalibrated_persample.csv (n={len(df_recal)})")


if __name__ == "__main__":
    main()
