"""
Stage 2: public-TCR identification (5% prevalence), per-clonotype age aggregation,
and signed-Wasserstein scoring with two-component GMM significance.

Run with 'train' (default) to derive age-associated clonotypes from training donors
only, or 'combined' to use all donors.
"""
import sys, ast, json
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from multiprocessing import Pool
from scipy.stats import wasserstein_distance
from sklearn.mixture import GaussianMixture
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import config as C

def norm(s): return str(s).lower().replace('copy of ', '').strip().replace('_tcrb', '')

def unique_tcrs(fp):
    try:
        df = pd.read_csv(fp, sep='\t', usecols=['amino_acid'])
        return set(df['amino_acid'].dropna().unique())
    except Exception:
        return set()

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'train'   # 'train' (leak-free) or 'combined' (reproduce reference)
    if mode == 'combined':
        train_files = sorted(list(C.TRAIN_DIR.glob('*.tsv')) + list(C.TEST_DIR.glob('*.tsv')))
        SUFFIX = 'combined'
    else:
        train_files = sorted(C.TRAIN_DIR.glob('*.tsv'))
        SUFFIX = 'trainonly'
    n = len(train_files)
    thr = int(n * C.PREVALENCE)
    print(f"train samples: {n}  5% prevalence threshold: >= {thr} samples", flush=True)

    # --- src/02: public TCR publicity across TRAIN ---
    pub = Counter()
    with Pool(64) as pool:
        for s in pool.imap_unordered(unique_tcrs, train_files, chunksize=8):
            pub.update(s)
    public = {t for t, c in pub.items() if c >= thr}
    print(f"unique TCRs: {len(pub)}  public (>=5%): {len(public)}", flush=True)
    pd.DataFrame({'amino_acid': sorted(public)}).to_csv(C.OUTPUTS / f'public_tcrs_{SUFFIX}.csv', index=False)

    # --- src/04: aggregate ages per public TCR across TRAIN only ---
    meta = pd.read_csv(C.METADATA); meta['key'] = meta['key'].map(norm)
    sample_to_age = dict(zip(meta['key'], meta['age']))
    tcr_ages = defaultdict(list)
    for fp in train_files:
        k = norm(fp.stem)
        age = sample_to_age.get(k)
        if age is None or (isinstance(age, float) and np.isnan(age)):
            continue
        s = unique_tcrs(fp)
        for t in s & public:
            tcr_ages[t].append(age)
    print(f"public TCRs with age support: {len(tcr_ages)}", flush=True)

    # --- src/05: signed Wasserstein + GMM significance ---
    df = pd.DataFrame([{'TCR': t, 'Ages': a} for t, a in tcr_ages.items()])
    global_ages = np.concatenate(df['Ages'].values)
    g_mean = global_ages.mean()
    g_u, g_c = np.unique(global_ages, return_counts=True)

    def sw(ages):
        if not len(ages): return 0.0
        u, c = np.unique(ages, return_counts=True)
        d = wasserstein_distance(u, g_u, u_weights=c, v_weights=g_c)
        return np.sign(np.mean(ages) - g_mean) * d
    df['signed_wasserstein'] = df['Ages'].apply(sw)

    gmm = GaussianMixture(n_components=2, random_state=42)
    v = df['signed_wasserstein'].values.reshape(-1, 1)
    df['gmm_component'] = gmm.fit_predict(v)
    probs = gmm.predict_proba(v); means = gmm.means_.flatten()
    df['old_prob'] = probs[:, int(np.argmax(means))]
    df['young_prob'] = probs[:, int(np.argmin(means))]
    grp = df.groupby('gmm_component')['signed_wasserstein']
    df['component_zscore'] = (df['signed_wasserstein'] - grp.transform('mean')) / grp.transform('std')
    df['signed_wasserstein_significant'] = df['component_zscore'].abs() > 1.96
    df['Ages'] = df['Ages'].apply(lambda a: [int(x) for x in a])
    df.to_csv(C.OUTPUTS / f'{SUFFIX}_significant_tcrs_signed_wasserstein.csv', index=False)
    sig = int(df['signed_wasserstein_significant'].sum())
    print(f"scored {len(df)} public TCRs; significant (|Z|>1.96): {sig}", flush=True)

    report = {
        'mode': mode,
        'donors_used': n,
        'prevalence_threshold_donors': thr,
        'public_tcrs': len(public),
        'scored_public_tcrs': len(df),
        'significant_clonotypes': sig,
    }
    print(json.dumps(report, indent=2), flush=True)
    (C.OUTPUTS / f'stage02_report_{SUFFIX}.json').write_text(json.dumps(report, indent=2))
