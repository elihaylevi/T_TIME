"""
Materialise train/ and test/ directories of downsampled repertoires from the
split key lists, so the train-only feature scripts can glob them.
"""
import sys, os
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import config as C

def norm(s): return str(s).lower().replace('copy of ', '').strip().replace('_tcrb', '')

which = sys.argv[1] if len(sys.argv) > 1 else 'matrix'   # 'matrix' or 'fresh'
train_keys = set(pd.read_csv(C.OUTPUTS / f'split_{which}_train.csv')['key'])
test_keys  = set(pd.read_csv(C.OUTPUTS / f'split_{which}_test.csv')['key'])

# map normalized key -> actual downsampled filename
by_key = {}
for p in C.DOWNSAMPLED.glob('*.tsv'):
    by_key[norm(p.stem)] = p

for keys, dst in [(train_keys, C.TRAIN_DIR), (test_keys, C.TEST_DIR)]:
    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.glob('*.tsv'):
        old.unlink()
    n = 0
    for k in keys:
        if k in by_key:
            os.symlink(by_key[k], dst / by_key[k].name); n += 1
    print(f"{dst.name}: linked {n}/{len(keys)}", flush=True)
