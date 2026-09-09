"""
Interpretable baselines evaluated against the neural network on the same split,
with bootstrap confidence intervals:
  (a) repertoire diversity (Shannon, inverse-Simpson, clonality, Gini, richness)
  (b) linear models (Ridge, ElasticNet) on the full feature matrix
"""
import sys, json
import numpy as np, pandas as pd
from multiprocessing import Pool
from pathlib import Path
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score
sys.path.insert(0, str(Path(__file__).parent))
import config as C

META_SET = {'sample name', 'Age', 'Biological Sex'}
def norm(s): return str(s).lower().replace('copy of ', '').strip().replace('_tcrb', '')

def diversity(fp):
    try:
        df = pd.read_csv(fp, sep='\t', usecols=['amino_acid', 'templates'])
        t = df.groupby('amino_acid')['templates'].sum().values.astype(float)
        p = t / t.sum(); rich = len(t)
        shannon = -np.sum(p * np.log(p))
        simpson = np.sum(p**2); invsimpson = 1.0 / simpson
        clonality = 1 - shannon / np.log(rich) if rich > 1 else 0.0
        s = np.sort(p); n = len(s); gini = (2*np.sum((np.arange(1, n+1))*s)/(n*np.sum(s))) - (n+1)/n
        return norm(fp.stem), dict(richness=rich, shannon=shannon, inv_simpson=invsimpson,
                                   clonality=clonality, gini=gini)
    except Exception:
        return norm(fp.stem), None

def build_div(d):
    rows = {}
    with Pool(16) as pool:
        for k, v in pool.imap_unordered(diversity, sorted(d.glob('*.tsv')), chunksize=8):
            if v: rows[k] = v
    return pd.DataFrame.from_dict(rows, orient='index')

def boot(y, p, n=2000):
    rng = np.random.default_rng(42); idx = np.arange(len(y)); m, r = [], []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        m.append(mean_absolute_error(y[b], p[b])); r.append(r2_score(y[b], p[b]))
    ci = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
    return ci(m), ci(r)

def evalm(name, model, Xtr, ytr, Xte, yte, res):
    model.fit(Xtr, ytr); p = model.predict(Xte)
    mae, r2 = mean_absolute_error(yte, p), r2_score(yte, p)
    mci, rci = boot(yte, p)
    res[name] = dict(MAE=round(float(mae), 3), MAE_CI=mci, R2=round(float(r2), 3), R2_CI=rci)
    print(f"[{name}] MAE={mae:.3f} {mci} R2={r2:.3f} {rci}", flush=True)

if __name__ == '__main__':
    tr = pd.read_csv(C.DATA / 'train_combined_matrix_pruned_95.csv')
    te = pd.read_csv(C.DATA / 'test_combined_matrix_pruned_95.csv')
    tr = tr.loc[:, ~tr.columns.duplicated()]; te = te.loc[:, ~te.columns.duplicated()]
    feats = [c for c in tr.columns if c not in META_SET]
    for f in [c for c in feats if c not in te.columns]: te[f] = 0.0
    ytr = tr['Age'].to_numpy(float); yte = te['Age'].to_numpy(float)
    res = {}

    # (b) linear on features
    Xtr, Xte = tr[feats].fillna(0).to_numpy(float), te[feats].fillna(0).to_numpy(float)
    evalm('ridge_features', make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=10.0)), Xtr, ytr, Xte, yte, res)
    evalm('elasticnet_features', make_pipeline(StandardScaler(with_mean=False), ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)), Xtr, ytr, Xte, yte, res)

    # (a) diversity
    # (a) diversity - Tier B: needs the raw repertoire TSVs at C.TRAIN_DIR / C.TEST_DIR.
    # C14 (2026-09-08): those are absent in a data-only checkout, build_div() then returns
    # an empty frame and dtr[dcols] raised a KeyError - which killed the script BEFORE it
    # wrote baselines.json, so the Ridge/ElasticNet results computed above were lost too.
    # Skip the block instead, and let figures_3_4.py omit those two bars (C3).
    dcols = ['richness', 'shannon', 'inv_simpson', 'clonality', 'gini']
    if not any(C.TRAIN_DIR.glob('*.tsv')) or not any(C.TEST_DIR.glob('*.tsv')):
        print(f'[!] no repertoire TSVs under {C.TRAIN_DIR} / {C.TEST_DIR} - '
              'skipping diversity_ridge / diversity_gbm (Tier B)', flush=True)
    else:
        print('computing diversity...', flush=True)
        dtr = build_div(C.TRAIN_DIR); dte = build_div(C.TEST_DIR)
        trk = tr.assign(key=tr['sample name'].map(norm)).set_index('key')
        tek = te.assign(key=te['sample name'].map(norm)).set_index('key')
        dtr = dtr.join(trk['Age']).dropna(); dte = dte.join(tek['Age']).dropna()
        Xdtr, ydtr = dtr[dcols].to_numpy(float), dtr['Age'].to_numpy(float)
        Xdte, ydte = dte[dcols].to_numpy(float), dte['Age'].to_numpy(float)
        evalm('diversity_ridge', make_pipeline(StandardScaler(), Ridge(alpha=1.0)), Xdtr, ydtr, Xdte, ydte, res)
        evalm('diversity_gbm', GradientBoostingRegressor(random_state=42), Xdtr, ydtr, Xdte, ydte, res)

    res['_mlp'] = dict(MAE=7.484, R2=0.784, note='neural network, same split')
    res['_n'] = dict(train=len(tr), test=len(te))
    print(json.dumps(res, indent=2), flush=True)
    (C.OUTPUTS / 'baselines.json').write_text(json.dumps(res, indent=2))
