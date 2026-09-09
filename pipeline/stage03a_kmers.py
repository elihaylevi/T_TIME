"""
Stage 3a: unique K-mer (K=3) feature extraction per repertoire, merged with sample
metadata, for the train and test splits.
"""
import sys
import pandas as pd
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import config as C

def norm(s): return str(s).lower().replace('copy of ', '').strip().replace('_tcrb', '')

def kmers(seq, k):
    return [seq[i:i+k] for i in range(len(seq)-k+1)]

def process(fp):
    try:
        df = pd.read_csv(fp, sep='\t', usecols=['amino_acid']).dropna(subset=['amino_acid'])
        df = df[~df['amino_acid'].str.contains(r'\*', na=False)]
        c = Counter()
        for seq in df['amino_acid']:
            c.update(set(kmers(seq, C.K)))
        return fp.stem, dict(c)
    except Exception as e:
        return fp.stem, None

def build(files):
    res = {}
    with Pool(48) as pool:
        for name, d in pool.imap_unordered(process, files, chunksize=8):
            if d is not None:
                res[name] = d
    return pd.DataFrame.from_dict(res, orient='index').fillna(0)

if __name__ == '__main__':
    meta = pd.read_csv(C.METADATA); meta['key'] = meta['key'].map(norm)
    meta_small = meta[['key', 'age', 'sex']].rename(columns={'age': 'Age', 'sex': 'Biological Sex'})

    for split, d in [('train', C.TRAIN_DIR), ('test', C.TEST_DIR)]:
        files = sorted(d.glob('*.tsv'))
        print(f"{split}: {len(files)} files -> extracting kmers", flush=True)
        km = build(files)
        km.index.name = 'sample name'
        km = km.reset_index()
        km['key'] = km['sample name'].map(norm)
        km = km.merge(meta_small, on='key', how='left').drop(columns='key')
        out = C.OUTPUTS / f'kmers_{split}.csv'
        km.to_csv(out, index=False)
        print(f"  wrote {out}  shape={km.shape}", flush=True)
