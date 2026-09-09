"""
Stage 1: repertoire QC (>= 200,000 templates) and multinomial downsampling to a
uniform sequencing depth. Uses a stable per-file RNG seed so the downsampling is
reproducible across runs (requires PYTHONHASHSEED=0).
"""
import sys, hashlib
import numpy as np, pandas as pd
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import config as C

def stable_seed(name: str) -> int:
    return int(hashlib.sha1(name.encode()).hexdigest(), 16) % (10**8)

def process(file_path: Path):
    try:
        df = pd.read_csv(file_path, sep='\t', usecols=['amino_acid', 'templates'])
        df = df.dropna(subset=['amino_acid', 'templates'])
        df = df[~df['amino_acid'].str.contains(r'\*', na=False)]
        df['templates'] = df['templates'].astype(int)
        total = df['templates'].sum()
        if total < C.TARGET_DEPTH:
            return ("EXCLUDED", file_path.name, int(total))
        probs = df['templates'].values / total
        rng = np.random.default_rng(seed=stable_seed(file_path.name))
        counts = rng.multinomial(C.TARGET_DEPTH, probs)
        mask = counts > 0
        out = pd.DataFrame({'amino_acid': df['amino_acid'].values[mask],
                            'templates': counts[mask]})
        out.to_csv(C.DOWNSAMPLED / file_path.name, sep='\t', index=False)
        return ("SUCCESS", file_path.name, int(total))
    except Exception as e:
        return ("ERROR", file_path.name, str(e))

if __name__ == '__main__':
    raw = sorted(C.RAW_ALL.glob('*.tsv'))
    print(f"Found {len(raw)} raw files. Target depth {C.TARGET_DEPTH}.", flush=True)
    rows = []
    with Pool(processes=64) as pool:
        for i, r in enumerate(pool.imap_unordered(process, raw, chunksize=4)):
            rows.append(r)
            if (i+1) % 250 == 0:
                print(f"  processed {i+1}/{len(raw)}", flush=True)
    log = pd.DataFrame(rows, columns=['status', 'sample_file', 'detail'])
    log.to_csv(C.OUTPUTS / 'stage01_qc_log.csv', index=False)
    vc = log['status'].value_counts().to_dict()
    print(f"QC done. {vc}", flush=True)
    print(f"SUCCESS files written to {C.DOWNSAMPLED}", flush=True)
