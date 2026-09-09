"""
Build the feature matrix for out-of-training clinical samples, aligned to the
training feature space, with age/sex and clinical condition labels.
"""
import sys, hashlib, glob, os
import numpy as np, pandas as pd
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import config as C

COVID_DIR = str(C.RAW_COVID)
TRAIN_MATRIX = C.DATA / 'train_combined_matrix_pruned_95.csv'
LABELS = C.OUTPUTS / 'clinical_sample_labels.csv'
META_SET = {'sample name', 'Age', 'Biological Sex'}

_cols = pd.read_csv(TRAIN_MATRIX, nrows=0).columns.tolist()
FEATS = [c for c in _cols if c not in META_SET]
KMERS = [c for c in FEATS if len(c) == 3 and c.isupper() and c.isalpha()]
TCRS  = [c for c in FEATS if c not in KMERS]
KSET, TSET = set(KMERS), set(TCRS)

def norm(s): return str(s).lower().replace('copy of ', '').strip().replace('_tcrb', '')
def stable_seed(name): return int(hashlib.sha1(name.encode()).hexdigest(), 16) % (10**8)

def process(args):
    key, fp = args
    try:
        df = pd.read_csv(fp, sep='\t', usecols=['amino_acid', 'templates'])
        df = df.dropna(subset=['amino_acid', 'templates'])
        df = df[~df['amino_acid'].str.contains(r'\*', na=False)]
        df['templates'] = df['templates'].astype(int)
        total = int(df['templates'].sum())
        if total < C.TARGET_DEPTH:
            return {'_status': 'EXCLUDED', 'key': key}
        rng = np.random.default_rng(stable_seed(os.path.basename(fp)))
        counts = rng.multinomial(C.TARGET_DEPTH, df['templates'].values / total)
        m = counts > 0
        aa = df['amino_acid'].values[m]; tm = counts[m]
        tcr_val = Counter()
        for a, t in zip(aa, tm):
            if a in TSET: tcr_val[a] += int(t)
        kmer_val = Counter()
        for a in set(aa):
            for i in range(len(a) - 2):
                km = a[i:i+3]
                if km in KSET: kmer_val[km] += 1
        row = {'key': key, '_status': 'SUCCESS'}
        row.update({k: kmer_val.get(k, 0) for k in KMERS})
        row.update({t: tcr_val.get(t, 0) for t in TCRS})
        return row
    except Exception as e:
        return {'_status': f'ERROR:{e}', 'key': key}

if __name__ == '__main__':
    lab = pd.read_csv(LABELS)
    covid_files = {norm(os.path.basename(p).replace('.tsv', '')): p for p in glob.glob(f'{COVID_DIR}/*.tsv')}
    work = [(k, covid_files[k]) for k in lab['key'] if k in covid_files]
    print(f"clinical samples to build: {len(work)}", flush=True)
    rows = []
    with Pool(48) as pool:
        for i, r in enumerate(pool.imap_unordered(process, work, chunksize=2)):
            rows.append(r)
            if (i+1) % 50 == 0: print(f"  {i+1}/{len(work)}", flush=True)
    df = pd.DataFrame(rows)
    print("status:", df['_status'].str.split(':').str[0].value_counts().to_dict(), flush=True)
    ok = df[df['_status'] == 'SUCCESS'].drop(columns='_status').merge(lab, on='key', how='left')
    ok = ok.rename(columns={'age': 'Age'})
    ok['sample name'] = ok['key']
    ok = ok[['sample name', 'Age', 'Biological Sex'] + KMERS + TCRS +
            ['hypertension', 'autoimmune', 'healthy', 'cancer']]
    ok.to_csv(C.OUTPUTS / 'clinical_combined_matrix.csv', index=False)
    print(f"SUCCESS {len(ok)} -> clinical_combined_matrix.csv {ok.shape}", flush=True)
