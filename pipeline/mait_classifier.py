"""
Young-vs-old CDR3 physicochemical classifier (Atchley factors, center-padded),
evaluated with and without depletion of putative MAIT clonotypes, and reporting
which physicochemical properties drive the separation.
"""
import sys, re, json
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
sys.path.insert(0, str(Path(__file__).parent))
import config as C

ATCHLEY = {
 'A':[-0.591,-1.302,-0.733,1.570,-0.146],'C':[-1.343,0.465,-0.862,-1.020,-0.255],
 'D':[1.050,0.302,-3.656,-0.259,-3.242],'E':[1.357,-1.453,1.477,0.113,-0.837],
 'F':[-1.006,-0.590,1.891,-0.397,0.412],'G':[-0.384,1.652,1.330,1.045,2.064],
 'H':[0.336,-0.417,-1.673,-1.474,-0.078],'I':[-1.239,-0.547,2.131,0.393,0.816],
 'K':[1.831,-0.561,0.533,-0.277,1.648],'L':[-1.019,-0.987,-1.505,1.266,-0.912],
 'M':[-0.663,-1.524,2.219,-1.005,1.212],'N':[0.945,0.828,1.299,-0.169,0.933],
 'P':[0.189,2.081,-1.628,0.421,-1.392],'Q':[0.931,-0.179,-3.005,-0.503,-1.853],
 'R':[1.538,-0.055,1.502,0.440,2.897],'S':[-0.228,1.399,-4.760,0.670,-2.647],
 'T':[-0.032,0.326,2.213,0.908,1.313],'V':[-1.337,-0.279,-0.544,1.242,-1.262],
 'W':[-0.595,0.009,0.672,-2.128,-0.184],'Y':[0.260,0.830,3.097,-0.838,1.512]}
MAXLEN = 20

def encode(seq):
    v = np.zeros((MAXLEN, 5))
    s = seq[:MAXLEN]
    off = (MAXLEN - len(s)) // 2   # center padding
    for i, a in enumerate(s):
        if a in ATCHLEY: v[off+i] = ATCHLEY[a]
    return v.flatten()

def classify(df):
    X = np.vstack([encode(s) for s in df['TCR']])
    y = df['is_old'].values.astype(int)
    clf = GradientBoostingClassifier(random_state=42)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    proba = cross_val_predict(clf, X, y, cv=skf, method='predict_proba')[:, 1]
    return roc_auc_score(y, proba), accuracy_score(y, proba > 0.5)

def balance(df):
    n = df['is_old'].value_counts().min()
    return pd.concat([g.sample(n, random_state=42) for _, g in df.groupby('is_old')]).reset_index(drop=True)

if __name__ == '__main__':
    d = pd.read_csv(C.DATA / 'significant_tcrs_signed_wasserstein.csv')
    sig = d[d['signed_wasserstein_significant'] == True].copy() if 'signed_wasserstein_significant' in d else d.copy()
    sig['is_old'] = (sig['signed_wasserstein'] > 0).astype(int)
    sig['len'] = sig['TCR'].str.len()
    sig['mait'] = sig['TCR'].str.match(r'^CASS[DE]') & (sig['len'] >= 15)

    youth = sig[sig['is_old'] == 0]
    mait_frac_youth = float(sig.loc[sig['is_old'] == 0, 'mait'].mean())

    full = balance(sig)
    auc_full, acc_full = classify(full)

    depl = balance(sig[~sig['mait']])
    auc_depl, acc_depl = classify(depl)

    # R1 Major 5: which physicochemical properties drive young-vs-old separation?
    from sklearn.ensemble import GradientBoostingClassifier as _GBC
    Xf = np.vstack([encode(s) for s in full['TCR']]); yf = full['is_old'].values.astype(int)
    clf = _GBC(random_state=42).fit(Xf, yf)
    imp = clf.feature_importances_.reshape(MAXLEN, 5)   # positions x Atchley factors
    factor_names = ['ATC1_polarity/charge', 'ATC2_secondary_struct', 'ATC3_size/volume', 'ATC4_refractivity', 'ATC5_charge/heat']
    by_factor = {factor_names[k]: round(float(imp[:, k].sum()), 4) for k in range(5)}
    top_positions = {f'pos{int(p)+1}': round(float(imp[p].sum()), 4) for p in np.argsort(imp.sum(1))[::-1][:5]}
    drivers = dict(importance_by_atchley_factor=by_factor, top_cdr3_positions=top_positions)

    out = dict(
        physicochemical_drivers=drivers,
        n_significant=int(len(sig)), n_young=int((sig.is_old == 0).sum()), n_old=int((sig.is_old == 1).sum()),
        n_mait=int(sig['mait'].sum()), mait_frac_of_youth=round(mait_frac_youth, 4),
        full=dict(n=int(len(full)), AUC=round(auc_full, 3), acc=round(acc_full, 3)),
        mait_depleted=dict(n=int(len(depl)), AUC=round(auc_depl, 3), acc=round(acc_depl, 3)),
        delta_AUC=round(auc_full - auc_depl, 3),
    )
    print(json.dumps(out, indent=2), flush=True)
    (C.OUTPUTS / 'mait_classifier.json').write_text(json.dumps(out, indent=2))
