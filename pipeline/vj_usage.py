"""
V/J gene usage of young- vs old-associated CDR3 sets, from the raw repertoires
(which carry V/J calls), including MAIT-consistent V-family enrichment.
"""
import sys, glob, os, json, re
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from scipy.stats import chi2_contingency, fisher_exact
sys.path.insert(0, str(Path(__file__).parent))
import config as C

COVID = str(C.RAW_COVID)
VO = str(C.RAW_VO)
def norm(s): return str(s).lower().replace('copy of ', '').strip().replace('_tcrb', '')

sig = pd.read_csv(C.DATA / 'significant_tcrs_signed_wasserstein.csv')
if 'signed_wasserstein_significant' in sig: sig = sig[sig['signed_wasserstein_significant'] == True]
SIGSET = set(sig['TCR'])
YOUNG = set(sig.loc[sig['signed_wasserstein'] < 0, 'TCR'])
OLD = set(sig.loc[sig['signed_wasserstein'] > 0, 'TCR'])

def scan(fp):
    try:
        cols = pd.read_csv(fp, sep='\t', nrows=0).columns
        vcol = 'v_gene' if 'v_gene' in cols else ('v_resolved' if 'v_resolved' in cols else None)
        jcol = 'j_gene' if 'j_gene' in cols else ('j_resolved' if 'j_resolved' in cols else None)
        if vcol is None: return None
        df = pd.read_csv(fp, sep='\t', usecols=['amino_acid', vcol, jcol])
        df = df[df['amino_acid'].isin(SIGSET)]
        out = []
        for _, r in df.iterrows():
            out.append((r['amino_acid'], str(r[vcol]), str(r[jcol])))
        return out
    except Exception:
        return None

if __name__ == '__main__':
    train_keys = set(pd.read_csv(C.OUTPUTS / 'split_train.csv')['key'])
    fmap = {}
    for p in glob.glob(f'{COVID}/*.tsv') + glob.glob(f'{VO}/*.tsv'):
        k = norm(os.path.basename(p).replace('.tsv', ''))
        if k in train_keys: fmap[k] = p
    files = sorted(fmap.values())
    # sample up to 500 donors for the V/J distribution
    rng = np.random.default_rng(42)
    if len(files) > 500: files = list(rng.choice(files, 500, replace=False))
    print(f"scanning {len(files)} raw train donors for V/J of {len(SIGSET)} age-assoc CDR3s", flush=True)

    cdr3_v = defaultdict(Counter); cdr3_j = defaultdict(Counter)
    with Pool(24) as pool:
        for i, res in enumerate(pool.imap_unordered(scan, files, chunksize=4)):
            if res:
                for aa, v, j in res:
                    cdr3_v[aa][v] += 1; cdr3_j[aa][j] += 1
            if (i+1) % 50 == 0: print(f"  {i+1}/{len(files)}", flush=True)

    def domV(aa):
        c = cdr3_v.get(aa); return c.most_common(1)[0][0] if c else None
    def vfam(v):
        m = re.search(r'(TCRBV\d+|TRBV\d+)', str(v)); return m.group(1) if m else str(v)

    rows = []
    for aa in SIGSET:
        v = domV(aa)
        if v and v != 'nan':
            rows.append({'TCR': aa, 'V': v, 'Vfam': vfam(v),
                         'grp': 'young' if aa in YOUNG else 'old',
                         'mait_motif': bool(re.match(r'^CASS[DE]', aa)) and len(aa) >= 15})
    vj = pd.DataFrame(rows)
    # MAIT-consistent V families
    MAIT_V = ['TCRBV06', 'TCRBV20', 'TCRBV04', 'TRBV6', 'TRBV20', 'TRBV4']
    def is_mait_v(vf): return any(vf.startswith(m) for m in MAIT_V)
    vj['mait_V'] = vj['Vfam'].apply(is_mait_v)

    # young vs old: MAIT-V enrichment
    tab = pd.crosstab(vj['grp'], vj['mait_V'])
    chi2, p, _, _ = chi2_contingency(tab)
    young_maitV = float(vj[vj.grp == 'young']['mait_V'].mean())
    old_maitV = float(vj[vj.grp == 'old']['mait_V'].mean())

    # top V families by group
    topV = {g: vj[vj.grp == g]['Vfam'].value_counts().head(8).to_dict() for g in ['young', 'old']}

    out = dict(
        n_cdr3_with_V=int(len(vj)),
        young_MAIT_V_fraction=round(young_maitV, 4), old_MAIT_V_fraction=round(old_maitV, 4),
        chi2_MAITV_young_vs_old_p=float(p),
        mait_motif_MAIT_V_fraction=round(float(vj[vj.mait_motif]['mait_V'].mean()), 4) if vj['mait_motif'].any() else None,
        nonmotif_MAIT_V_fraction=round(float(vj[~vj.mait_motif]['mait_V'].mean()), 4),
        top_V_families=topV,
    )
    vj.to_csv(C.OUTPUTS / 'p1a_vj_percdr3.csv', index=False)
    print(json.dumps(out, indent=2), flush=True)
    (C.OUTPUTS / 'vj_usage.json').write_text(json.dumps(out, indent=2))
