"""
Stage 1b: define the modelling cohort (QC-passing repertoires with age metadata)
and produce the 75/25 train/test split.
"""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import config as C

def norm(s):
    return str(s).lower().replace('copy of ', '').strip().replace('_tcrb', '')

if __name__ == '__main__':
    meta = pd.read_csv(C.METADATA)
    meta['key'] = meta['key'].map(norm)
    age_ok = set(meta.loc[meta['age'].notna(), 'key'])

    qc = pd.read_csv(C.OUTPUTS / 'stage01_qc_log.csv')
    passed = qc.loc[qc['status'] == 'SUCCESS', 'sample_file'].map(lambda f: norm(Path(f).stem))
    passed = set(passed)
    print(f"QC-passing: {len(passed)}  with-age: {len(passed & age_ok)}", flush=True)

    cohort = sorted(passed & age_ok)

    # Split defined by the feature matrices (this is the split used for the reported
    # results, so downstream stages are directly comparable).
    tr = pd.read_csv(C.DATA / 'train_combined_matrix_pruned_95.csv', usecols=['sample name'])
    te = pd.read_csv(C.DATA / 'test_combined_matrix_pruned_95.csv', usecols=['sample name'])
    matrix_train = set(tr['sample name'].map(norm)) & set(cohort)
    matrix_test  = set(te['sample name'].map(norm)) & set(cohort)

    # Independently derived 75/25 split (sort, then seeded shuffle), for sensitivity analysis.
    files = sorted(cohort)
    np.random.seed(C.RANDOM_SEED); np.random.shuffle(files)
    k = int(C.TRAIN_RATIO * len(files))
    fresh_train, fresh_test = set(files[:k]), set(files[k:])

    report = {
        'qc_passing': len(passed),
        'cohort_with_age': len(cohort),
        'split_train_n': len(matrix_train), 'split_test_n': len(matrix_test),
        'fresh_train_n': len(fresh_train), 'fresh_test_n': len(fresh_test),
    }
    print(json.dumps(report, indent=2), flush=True)
    (C.OUTPUTS / 'stage01b_split_report.json').write_text(json.dumps(report, indent=2))

    C.TRAIN_DIR.mkdir(parents=True, exist_ok=True); C.TEST_DIR.mkdir(parents=True, exist_ok=True)
    pd.Series(sorted(matrix_train)).to_csv(C.OUTPUTS / 'split_train.csv', index=False, header=['key'])
    pd.Series(sorted(matrix_test)).to_csv(C.OUTPUTS / 'split_test.csv', index=False, header=['key'])
    pd.Series(sorted(fresh_train)).to_csv(C.OUTPUTS / 'split_fresh_train.csv', index=False, header=['key'])
    pd.Series(sorted(fresh_test)).to_csv(C.OUTPUTS / 'split_fresh_test.csv', index=False, header=['key'])
    print("Wrote split files and report.", flush=True)
