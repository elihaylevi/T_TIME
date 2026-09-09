"""
Build the external-cohort (Emerson 2017) feature matrix: parse age/sex from each
repertoire's sample tags, apply the same QC and downsampling, and compute the
k-mer and CDR3 features aligned to the training feature space.
"""
import sys, re, hashlib
import numpy as np, pandas as pd
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import config as C

EMERSON_DIR = C.RAW_EXTERNAL
TRAIN_MATRIX = C.DATA / 'train_combined_matrix_pruned_95.csv'
META_SET = {'sample name', 'Age', 'Biological Sex'}

# reference feature columns (order preserved), split into kmer vs TCR
_cols = pd.read_csv(TRAIN_MATRIX, nrows=0).columns.tolist()
FEATS = [c for c in _cols if c not in META_SET]
KMERS = [c for c in FEATS if len(c) == 3 and c.isupper() and c.isalpha()]   # includes 'AGE' the A-G-E kmer
TCRS  = [c for c in FEATS if c not in KMERS]
KSET, TSET = set(KMERS), set(TCRS)

def stable_seed(name): return int(hashlib.sha1(name.encode()).hexdigest(), 16) % (10**8)

def parse_tags(tags):
    age = sex = None
    if isinstance(tags, str):
        m = re.search(r'Age:(\d+)\s*Years', tags)
        if m: age = int(m.group(1))
        m = re.search(r'Biological Sex:(Male|Female)', tags)
        if m: sex = m.group(1)
    return age, sex

def process(fp):
    try:
        # header meta from first data row
        hdr = pd.read_csv(fp, sep='\t', nrows=1, usecols=lambda c: c in ('sample_name', 'sample_tags'))
        sname = str(hdr['sample_name'].iloc[0]) if 'sample_name' in hdr else fp.stem
        age, sex = parse_tags(hdr['sample_tags'].iloc[0] if 'sample_tags' in hdr else None)

        df = pd.read_csv(fp, sep='\t', usecols=['amino_acid', 'templates'])
        df = df.dropna(subset=['amino_acid', 'templates'])
        df = df[~df['amino_acid'].str.contains(r'\*', na=False)]
        df['templates'] = df['templates'].astype(int)
        total = int(df['templates'].sum())
        if total < C.TARGET_DEPTH:
            return {'_status': 'EXCLUDED', 'sample name': sname}
        # downsample to 200k
        rng = np.random.default_rng(stable_seed(fp.name))
        counts = rng.multinomial(C.TARGET_DEPTH, df['templates'].values / total)
        m = counts > 0
        aa = df['amino_acid'].values[m]; tm = counts[m]

        # TCR features: summed templates per CDR3 (restricted to reference TCR set)
        tcr_val = Counter()
        for a, t in zip(aa, tm):
            if a in TSET: tcr_val[a] += int(t)
        # k-mer features: number of unique sequences containing each 3-mer (restricted to reference kmer set)
        kmer_val = Counter()
        for a in set(aa):
            for i in range(len(a) - 2):
                km = a[i:i+3]
                if km in KSET: kmer_val[km] += 1

        row = {'sample name': sname, 'Age': age, 'Biological Sex': sex, '_status': 'SUCCESS'}
        row.update({k: kmer_val.get(k, 0) for k in KMERS})
        row.update({t: tcr_val.get(t, 0) for t in TCRS})
        return row
    except Exception as e:
        return {'_status': f'ERROR:{e}', 'sample name': fp.stem}

if __name__ == '__main__':
    files = sorted(EMERSON_DIR.glob('*.tsv'))
    print(f"Emerson files: {len(files)}  reference feats: {len(FEATS)} (kmers {len(KMERS)}, TCR {len(TCRS)})", flush=True)
    rows = []
    with Pool(48) as pool:
        for i, r in enumerate(pool.imap_unordered(process, files, chunksize=2)):
            rows.append(r)
            if (i+1) % 50 == 0: print(f"  {i+1}/{len(files)}", flush=True)
    df = pd.DataFrame(rows)
    status = df['_status'].str.split(':').str[0].value_counts().to_dict()
    print(f"status: {status}", flush=True)
    ok = df[df['_status'] == 'SUCCESS'].drop(columns='_status')
    # column order: meta + reference feature order
    ok = ok[['sample name', 'Age', 'Biological Sex'] + KMERS + TCRS]
    ok.to_csv(C.OUTPUTS / 'emerson_combined_matrix.csv', index=False)
    df[['sample name', '_status']].to_csv(C.OUTPUTS / 'emerson_build_log.csv', index=False)
    with_age = ok['Age'].notna().sum()
    print(f"SUCCESS {len(ok)}  with_age {with_age}  -> emerson_combined_matrix.csv shape {ok.shape}", flush=True)
    print(f"age range: {ok['Age'].min()} - {ok['Age'].max()}", flush=True)
