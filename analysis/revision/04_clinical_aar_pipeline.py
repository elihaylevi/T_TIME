"""
04_clinical_aar_pipeline.py

Full clinical AAR (Age Acceleration Residual) pipeline, leakage-safe:

  1. Identifies the 244 clinical-condition patients (from
     FINAL_Clinical_Case_Study_Table.csv, metadata only - NOT its
     feature columns, which may not match train/test exactly) inside
     train_combined_matrix_pruned_95.csv and test_combined_matrix_pruned_95.csv
     by normalized sample name.
  2. Removes them from BOTH train and test -> train_clean, test_clean.
  3. Trains Model M2 on train_clean only (never sees clinical patients).
  4. Builds the "Robust Baseline" regression on test_clean (healthy
     Primary samples), top-5% trim, per Methods 4.11.
  5. Predicts on the clinical cohort (features taken from train/test
     rows directly - NOT from the clinical CSV's own feature columns,
     to guarantee identical featurization) and computes AAR relative
     to the baseline regression line. No outlier trimming on the
     clinical AAR itself (see prior discussion: trimming the outcome
     group biases toward the null).
  6. Flags each patient for hypertension / autoimmune / cancer, using
     word-list + boolean-flag definitions that are IDENTICAL to
     05_all_categories_exploration.py's core three categories (this
     was a real inconsistency in earlier drafts - cancer previously
     used the boolean flag ONLY here, giving n=17, vs n=22 when text
     words were included as in script 05. Both scripts now use the
     same definition; expect n~66/22/22, not 17/13/8).
  7. Runs a 5,000-permutation test for each category against the
     test_clean baseline pool, with Benjamini-Hochberg correction.

REQUIRES:
  - train_combined_matrix_pruned_95.csv
  - test_combined_matrix_pruned_95.csv
  - FINAL_Clinical_Case_Study_Table.csv

OUTPUTS (everything downstream scripts need - no in-memory/session
dependency from this point on):
  - clinical_aar_persample_NEW.csv   (244 rows: Age, y_pred_new,
    AAR_new, hypertension, autoimmune, cancer, + original metadata
    columns incl. age_accel from the old script-11 ensemble, for
    comparison)
  - baseline_reference.csv           (test_clean pool: Age, y_pred -
    the healthy-baseline population the permutation test compares
    against; also what the panel g/h/i gray background is drawn from)
  - baseline_params.json             (slope, intercept, test MAE)
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_absolute_error


# ============================================================
# 0. File location + sample-name normalization
# ============================================================
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


def norm_id(s):
    return str(s).lower().replace('copy of ', '').strip().replace('_tcrb', '')


# ============================================================
# 1. Load clinical metadata (labels + text columns only, not
#    the clinical CSV's own feature columns)
# ============================================================
LABEL_COLS_WANTED = ['sample name', 'y_pred', 'age_accel',
                      'has_chronic_hypertension', 'has_cancer', 'cancer_diagnosed',
                      'uses_autoimmune_medications', 'uses_ace_inhibitor', 'uses_arb',
                      'current_medications', 'diseases', 'selected_autoimmune_diagnoses',
                      'selected_other_diagnoses', 'describe_other_diagnoses',
                      'describe_immunosupressants', 'describe_cancers',
                      'describe_autoimmune_medications', 'describe_autoimmune_diagnoses',
                      'cancer_type', 'nsaid_type']

CANDIDATE_TEXT_COLS = ['current_medications', 'diseases', 'selected_autoimmune_diagnoses',
                        'selected_other_diagnoses', 'describe_other_diagnoses',
                        'describe_immunosupressants', 'describe_cancers',
                        'describe_autoimmune_medications', 'describe_autoimmune_diagnoses',
                        'cancer_type', 'nsaid_type']

# Canonical category definitions - MUST match 05_all_categories_exploration.py exactly
CLINICAL_CATEGORY_DEFS = {
    'hypertension': dict(
        words=['lisinopril', 'amlodipine', 'losartan', 'candesartan', 'olmesartan', 'hctz',
               'hydrochlorothiazide', 'atenolol', 'hypertension', 'high blood pressure'],
        flags=['has_chronic_hypertension', 'uses_ace_inhibitor', 'uses_arb']),
    'autoimmune': dict(
        words=['hashimoto', 'rheumatoid', 'crohn', 'psoriasis', 'ulcerative colitis', 't1d',
               'lupus', 'arthritis', 'stelara', 'humira', 'enbrel', 'dupixent'],
        flags=['uses_autoimmune_medications']),
    'cancer': dict(
        words=['cancer', 'malignancy', 'leukemia', 'lymphoma', 'melanoma', 'carcinoma', 'chemotherapy'],
        flags=['has_cancer']),
}


def flag_words(df, words, text_cols):
    if not words or not text_cols:
        return pd.Series(False, index=df.index)
    pattern = r'\b(' + '|'.join(re.escape(w) for w in words) + r')\b'
    return df[text_cols].fillna('').astype(str).apply(
        lambda x: x.str.contains(pattern, case=False, na=False, regex=True)).any(axis=1)


def flag_bool_cols(df, cols):
    out = pd.Series(False, index=df.index)
    for c in cols:
        if c in df.columns:
            out = out | df[c].apply(lambda x: str(x).lower() in ['1', '1.0', 'true'])
    return out


def build_category_flags(df):
    """Returns dict[category_name] -> boolean Series, using CLINICAL_CATEGORY_DEFS."""
    text_cols = [c for c in CANDIDATE_TEXT_COLS if c in df.columns]
    flags = {}
    for cat_name, cfg in CLINICAL_CATEGORY_DEFS.items():
        flags[cat_name] = flag_words(df, cfg['words'], text_cols) | flag_bool_cols(df, cfg['flags'])
    return flags


# ============================================================
# 2. Model (same architecture as scripts 01/02 - verified, MAE~7.5)
# ============================================================
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
    return [c for c in df.columns if c in META_SET], [c for c in df.columns if c not in META_SET]


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


# ============================================================
# 3. Permutation test + BH
# ============================================================
RNG = np.random.default_rng(42)


def perm_test(cond_aar, pool_aar, nperm=5000):
    obs = cond_aar.mean() - pool_aar.mean()
    combined = np.concatenate([cond_aar, pool_aar])
    n_cond = len(cond_aar)
    null = np.empty(nperm)
    for i in range(nperm):
        idx = RNG.permutation(len(combined))
        null[i] = combined[idx[:n_cond]].mean() - combined[idx[n_cond:]].mean()
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (nperm + 1)
    return float(obs), float(p)


def bh(pvals):
    p = np.array(pvals)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        prev = min(prev, p[order[i]] * n / rank)
        adj[order[i]] = prev
    return adj


def main():
    # ------------------------------------------------------------
    # Load clinical labels (metadata only) + train/test
    # ------------------------------------------------------------
    clin_full = pd.read_csv(_find("FINAL_Clinical_Case_Study_Table.csv"), low_memory=False,
                             usecols=lambda c: c in LABEL_COLS_WANTED)
    clin_full['norm_id'] = clin_full['sample name'].apply(norm_id)
    print(f"[*] Clinical labels loaded (metadata only): {clin_full.shape}")

    train_full = pd.read_csv(_find("train_combined_matrix_pruned_95.csv"))
    test_full = pd.read_csv(_find("test_combined_matrix_pruned_95.csv"))
    train_full['norm_id'] = train_full['sample name'].apply(norm_id)
    test_full['norm_id'] = test_full['sample name'].apply(norm_id)

    clin_ids = set(clin_full['norm_id'])
    in_train_mask = train_full['norm_id'].isin(clin_ids)
    in_test_mask = test_full['norm_id'].isin(clin_ids)
    print(f"[*] Clinical patients found in TRAIN: {in_train_mask.sum()}")
    print(f"[*] Clinical patients found in TEST:  {in_test_mask.sum()}")

    train_clean = train_full[~in_train_mask].drop(columns=['norm_id']).reset_index(drop=True)
    test_clean = test_full[~in_test_mask].drop(columns=['norm_id']).reset_index(drop=True)
    print(f"[*] train_clean: {len(train_clean)}  (was {len(train_full)})")
    print(f"[*] test_clean:  {len(test_clean)}  (was {len(test_full)})")

    # ------------------------------------------------------------
    # Reconstruct clinical feature rows FROM train/test (guarantees
    # identical featurization to the model - do not use the
    # clinical CSV's own feature columns)
    # ------------------------------------------------------------
    clin_from_train = train_full[in_train_mask].copy()
    clin_from_test = test_full[in_test_mask].copy()
    clin_features = pd.concat([clin_from_train, clin_from_test], ignore_index=True)
    print(f"\n[*] Clinical feature rows reconstructed from train+test: {len(clin_features)}")

    found_ids = set(clin_features['norm_id'])
    missing_ids = clin_ids - found_ids
    print(f"[*] Clinical patients NOT found in train or test: {len(missing_ids)} / {len(clin_ids)}")
    if missing_ids:
        print(f"    sample missing IDs: {list(missing_ids)[:5]}")

    clin = clin_features.merge(clin_full.drop(columns=['sample name']), on='norm_id', how='inner')
    clin = clin.drop(columns=['norm_id'])
    print(f"[*] Final clinical cohort (features from train/test + labels): n={len(clin)}")

    # ------------------------------------------------------------
    # Train model M2 on train_clean
    # ------------------------------------------------------------
    train_clean = standardize_meta_cols(train_clean)
    test_clean = standardize_meta_cols(test_clean)
    clin = standardize_meta_cols(clin)

    _, train_feats = split_features(train_clean)
    for f in [c for c in train_feats if c not in test_clean.columns]:
        test_clean[f] = 0.0
    for f in [c for c in train_feats if c not in clin.columns]:
        clin[f] = 0.0

    Xtr = coerce_numeric(train_clean, train_feats)
    ytr = pd.to_numeric(train_clean['Age'], errors='coerce').to_numpy(np.float32)
    mtr = np.isfinite(ytr)
    Xtr, ytr = Xtr.loc[mtr].reset_index(drop=True), ytr[mtr]

    print(f"\n[*] n_features = {len(train_feats)}  n_train_clean = {len(Xtr)}")
    model = train_final_model(Xtr, ytr, BEST_PARAMS)

    # ------------------------------------------------------------
    # Robust baseline regression on test_clean (Methods 4.11: top-5% trim)
    # ------------------------------------------------------------
    Xte = coerce_numeric(test_clean, train_feats)
    yte = pd.to_numeric(test_clean['Age'], errors='coerce').to_numpy(np.float32)
    mte = np.isfinite(yte)
    Xte, yte = Xte.loc[mte].reset_index(drop=True), yte[mte]

    p_test = predict(model, Xte, BEST_PARAMS['batch_size'])
    test_mae = mean_absolute_error(yte, p_test)
    print(f"[*] test_clean MAE (sanity, expect ~7.5): {test_mae:.3f}")

    abs_res = np.abs(p_test - yte)
    thresh95 = np.quantile(abs_res, 0.95)
    keep = abs_res <= thresh95
    slope, intercept = np.polyfit(yte[keep], p_test[keep], 1)
    print(f"[*] Robust baseline: slope={slope:.4f}, intercept={intercept:.4f} (n={keep.sum()}/{len(yte)})")

    aar_healthy_pool = p_test - (slope * yte + intercept)
    print(f"[*] test_clean AAR pool: mean={aar_healthy_pool.mean():.3f}, n={len(aar_healthy_pool)}")

    # Save baseline to disk - THIS IS THE FIX for the stale-variable
    # bug class: downstream scripts read this, never rely on notebook
    # memory.
    pd.DataFrame({'Age': yte, 'y_pred': p_test}).to_csv("baseline_reference.csv", index=False)
    Path("baseline_params.json").write_text(json.dumps({
        'slope': float(slope), 'intercept': float(intercept),
        'test_clean_mae': float(test_mae), 'n_baseline_pool': int(len(yte))
    }, indent=2))
    print("[*] Saved baseline_reference.csv and baseline_params.json")

    # ------------------------------------------------------------
    # Predict on clinical cohort - no trimming
    # ------------------------------------------------------------
    Xcl = coerce_numeric(clin, train_feats)
    ycl = pd.to_numeric(clin['Age'], errors='coerce').to_numpy(np.float32)
    mcl = np.isfinite(ycl)
    Xcl, ycl = Xcl.loc[mcl].reset_index(drop=True), ycl[mcl]
    clin_valid = clin.loc[mcl].reset_index(drop=True)

    p_clin = predict(model, Xcl, BEST_PARAMS['batch_size'])
    aar_clin = p_clin - (slope * ycl + intercept)
    clin_valid['y_pred_new'] = p_clin
    clin_valid['AAR_new'] = aar_clin
    print(f"\n[*] Clinical patients with valid AAR: {len(clin_valid)} / {len(clin)}")

    extreme = clin_valid[clin_valid['AAR_new'].abs() > 20][['sample name', 'Age', 'y_pred_new', 'AAR_new']]
    print(f"\n[*] Extreme |AAR|>20: {len(extreme)}/{len(clin_valid)}")
    if len(extreme) > 0:
        print(extreme.sort_values('AAR_new', ascending=False).to_string())

    # ------------------------------------------------------------
    # Category flags (canonical definition - see build_category_flags)
    # ------------------------------------------------------------
    cat_flags = build_category_flags(clin_valid)
    for name in ['hypertension', 'autoimmune', 'cancer']:
        clin_valid[name] = cat_flags[name].astype(int)

    print("\n[*] Category n's (canonical definition, matches script 05):")
    for name in ['hypertension', 'autoimmune', 'cancer']:
        print(f"    {name}: {clin_valid[name].sum()}")

    # ------------------------------------------------------------
    # Permutation tests + BH
    # ------------------------------------------------------------
    results = {}
    pvals, keys = [], []
    for name in ['hypertension', 'autoimmune', 'cancer']:
        mask = clin_valid[name] == 1
        n = mask.sum()
        if n < 2:
            print(f"\n[!] {name}: n={n}, too few for permutation test")
            continue
        obs, p = perm_test(clin_valid.loc[mask, 'AAR_new'].values, aar_healthy_pool, nperm=5000)
        results[name] = dict(n=int(n), AAR=round(obs, 3), perm_p=round(p, 4))
        pvals.append(p)
        keys.append(name)

    if pvals:
        adj = bh(pvals)
        for k, a in zip(keys, adj):
            results[k]['perm_p_BH'] = round(float(a), 4)

    print(f"\n{'=' * 70}\nRESULTS vs MANUSCRIPT / Sol's old notes\n{'=' * 70}")
    manuscript = {'hypertension': (-3.4, 0.005, 17), 'autoimmune': (-0.46, 0.047, 13), 'cancer': (0.1, 0.593, 8)}
    sol_old_n = {'hypertension': 65, 'autoimmune': 31, 'cancer': 24}
    for name, r in results.items():
        ms = manuscript[name]
        print(f"{name:14s}: n={r['n']:3d} (ms n={ms[2]}, Sol old n={sol_old_n[name]})  "
              f"AAR={r['AAR']:+.3f} (ms {ms[0]:+.2f})  perm_p={r['perm_p']:.4f} (ms {ms[1]:.3f})  "
              f"BH={r.get('perm_p_BH', '-')}")

    if 'age_accel' in clin_valid.columns:
        print(f"\n{'=' * 70}\nOLD (script-11 ensemble) vs NEW (verified model) - by group\n{'=' * 70}")
        none_flag = ~((clin_valid['hypertension'] == 1) | (clin_valid['autoimmune'] == 1) |
                       (clin_valid['cancer'] == 1))
        old_ref_mean = clin_valid.loc[none_flag, 'age_accel'].mean()
        print(f"(no condition) n={none_flag.sum()}  OLD mean age_accel={old_ref_mean:.3f}")
        for name in ['hypertension', 'autoimmune', 'cancer']:
            mask = clin_valid[name] == 1
            if mask.sum() == 0:
                continue
            old_mean = clin_valid.loc[mask, 'age_accel'].mean()
            print(f"{name:14s} n={mask.sum():3d}  OLD={old_mean:+.3f}  NEW={results[name]['AAR']:+.3f}  "
                  f"perm_p={results[name]['perm_p']:.4f}")
        print(f"\n[*] Overall correlation (old age_accel vs new AAR): "
              f"r={clin_valid[['AAR_new', 'age_accel']].corr().iloc[0, 1]:.3f}")

    clin_valid.to_csv("clinical_aar_persample_NEW.csv", index=False)
    print(f"\n[*] Saved clinical_aar_persample_NEW.csv (n={len(clin_valid)})")


if __name__ == "__main__":
    main()
