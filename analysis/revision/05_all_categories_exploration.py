"""
05_all_categories_exploration.py

Standalone exploration of ALL plausible clinical categories (not just
the three reported in the manuscript), for comparison. Reads
everything from disk - no dependency on notebook/session memory, so
it can be run in a fresh session at any time after script 04.

REQUIRES (run script 04 first):
  - clinical_aar_persample_NEW.csv
  - baseline_reference.csv
  - baseline_params.json

The three core categories (hypertension/autoimmune/cancer) use the
EXACT SAME word/flag definitions as 04_clinical_aar_pipeline.py's
CLINICAL_CATEGORY_DEFS - kept in sync manually since these run as
separate scripts. If you change one, change the other.

Outputs:
  - all_categories_summary.csv
  - all_categories_grid.png / .pdf
"""
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _find(name):
    _repo = Path(__file__).resolve().parents[2]
    # results/revision comes BEFORE data/: test_preds.csv exists in both, and the
    # revision pipeline must consume its own retrained predictions, not the legacy ones.
    for d in [Path("."), _repo / "results" / "revision", Path("data"), _repo, _repo / "data",
              _repo / "data" / "external", _repo / "data" / "external" / "reference_db",
              Path(__file__).resolve().parent,
              Path("/content"), Path("/content/data"), Path("/content/outputs"),
              Path("/content/deepmlp_eval_final/outputs")]:
        p = d / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"Could not find {name} - run scripts 01-04 first.")


plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'sans-serif',
                      'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                      'font.size': 7, 'axes.linewidth': 0.5})

CANDIDATE_TEXT_COLS = ['current_medications', 'diseases', 'selected_autoimmune_diagnoses',
                        'selected_other_diagnoses', 'describe_other_diagnoses',
                        'describe_immunosupressants', 'describe_cancers',
                        'describe_autoimmune_medications', 'describe_autoimmune_diagnoses',
                        'cancer_type', 'nsaid_type']

# KEEP IN SYNC with 04_clinical_aar_pipeline.py's CLINICAL_CATEGORY_DEFS
# for the first three entries.
ALL_CATEGORIES = {
    'Hypertension': dict(
        words=['lisinopril', 'amlodipine', 'losartan', 'candesartan', 'olmesartan', 'hctz',
               'hydrochlorothiazide', 'atenolol', 'hypertension', 'high blood pressure'],
        flags=['has_chronic_hypertension', 'uses_ace_inhibitor', 'uses_arb']),
    'Autoimmune': dict(
        words=['hashimoto', 'rheumatoid', 'crohn', 'psoriasis', 'ulcerative colitis', 't1d',
               'lupus', 'arthritis', 'stelara', 'humira', 'enbrel', 'dupixent'],
        flags=['uses_autoimmune_medications']),
    'Cancer': dict(
        words=['cancer', 'malignancy', 'leukemia', 'lymphoma', 'melanoma', 'carcinoma', 'chemotherapy'],
        flags=['has_cancer']),
    'Immune Suppressed': dict(
        words=['immunosuppressant', 'transplant', 'immunodeficiency', 'orencia'],
        flags=['is_immunocompromised', 'uses_immunosuppressant']),
    'Psychiatric & Neuro': dict(
        words=['depression', 'anxiety', 'stress', 'escitalopram', 'citalopram', 'wellbutrin',
               'xanax', 'alprazolam', 'lithium', 'lamictal', 'bipolar', 'epilepsy', 'seizure'],
        flags=[]),
    'Diabetes T2D': dict(
        words=['t2d', 'type 2 diabetes', 'metformin', 'insulin', 'glucophage', 'diabetes mellitus'],
        flags=[]),
    'Cholesterol (Statins)': dict(
        words=['atorvastatin', 'lipitor', 'pravastatin', 'simvastatin', 'hyperlipidemia', 'cholesterol'],
        flags=[]),
    'Chronic Respiratory (COPD)': dict(
        words=['copd', 'emphysema', 'bronchitis'],
        flags=[]),
    'Asthma': dict(
        words=['asthma', 'albuterol', 'inhaler', 'singulair', 'ventolin'],
        flags=['uses_asthma_quick_relief', 'uses_corticosteroids_for_asthma']),
    'Allergy & Atopy': dict(
        words=['zyrtec', 'claritin', 'fexofenadine', 'xyzal', 'allergy', 'rhinitis', 'sudafed'],
        flags=[]),
    'Thyroid Dysfunction': dict(
        words=['hypothyroidism', 'hyperthyroidism', 'thyroid', 'levothyroxine', 'synthroid'],
        flags=[]),
}


def flag_words(df, words, text_cols):
    if not words or not text_cols:
        return pd.Series(False, index=df.index)
    pattern = r'\b(' + '|'.join(re.escape(w) for w in words) + r')\b'
    return df[text_cols].fillna('').astype(str).apply(
        lambda x: x.str.contains(pattern, case=False, na=False, regex=True)).any(axis=1)


def flag_bool_cols(df, cols):
    out = pd.Series(False, index=df.index)
    for c in cols:
        if c in df.columns:
            out = out | df[c].apply(lambda x: str(x).lower() in ['1', '1.0', 'true'])
    return out


def perm_test(cond_aar, pool_aar, nperm=5000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    obs = cond_aar.mean() - pool_aar.mean()
    combined = np.concatenate([cond_aar, pool_aar])
    n_cond = len(cond_aar)
    null = np.empty(nperm)
    for i in range(nperm):
        idx = rng.permutation(len(combined))
        null[i] = combined[idx[:n_cond]].mean() - combined[idx[n_cond:]].mean()
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (nperm + 1)
    return float(obs), float(p)


def main():
    clin_valid = pd.read_csv(_find("clinical_aar_persample_NEW.csv"), low_memory=False)
    baseline_ref = pd.read_csv(_find("baseline_reference.csv"))
    baseline_params = json.loads(Path(_find("baseline_params.json")).read_text())
    slope, intercept = baseline_params['slope'], baseline_params['intercept']

    aar_healthy_pool = baseline_ref['y_pred'].values - (slope * baseline_ref['Age'].values + intercept)
    yte = baseline_ref['Age'].values
    p_test = baseline_ref['y_pred'].values
    ycl = clin_valid['Age'].values
    p_clin = clin_valid['y_pred_new'].values

    text_cols = [c for c in CANDIDATE_TEXT_COLS if c in clin_valid.columns]

    results_summary = []
    for cat_name, cfg in ALL_CATEGORIES.items():
        m = flag_words(clin_valid, cfg['words'], text_cols) | flag_bool_cols(clin_valid, cfg['flags'])
        n = int(m.sum())
        if n >= 2:
            obs, p = perm_test(clin_valid.loc[m, 'AAR_new'].values, aar_healthy_pool)
        else:
            obs, p = np.nan, np.nan
        results_summary.append(dict(category=cat_name, n=n, AAR=obs, perm_p=p, mask=m))

    summary_df = pd.DataFrame(results_summary).sort_values('perm_p')
    print(summary_df[['category', 'n', 'AAR', 'perm_p']].to_string(index=False))
    summary_df[['category', 'n', 'AAR', 'perm_p']].to_csv("all_categories_summary.csv", index=False)

    # ------------------------------------------------------------
    # Grid visualization
    # ------------------------------------------------------------
    plot_cats = [r for r in results_summary if r['n'] >= 2]
    n_cols = 4
    n_rows = int(np.ceil(len(plot_cats) / n_cols))

    fig = plt.figure(figsize=(3.2 * n_cols, 3.0 * n_rows), dpi=200)
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.55, wspace=0.4)

    lim_min = min(ycl.min(), p_clin.min(), yte.min()) - 5
    lim_max = max(ycl.max(), p_clin.max(), yte.max()) + 5
    x_line = np.array([lim_min, lim_max])

    for i, r in enumerate(plot_cats):
        row, col = divmod(i, n_cols)
        ax = fig.add_subplot(gs[row, col])

        ax.scatter(yte, p_test, c='silver', alpha=0.3, s=6, edgecolors='none', zorder=1, rasterized=True)
        ax.plot(x_line, slope * x_line + intercept, 'k--', lw=0.8, alpha=0.7, zorder=2)

        m = r['mask']
        sub_age = ycl[m.values]
        sub_pred = p_clin[m.values]
        ax.scatter(sub_age, sub_pred, c='crimson', s=15, edgecolors='black', linewidth=0.4,
                   alpha=0.8, zorder=3)
        for a, p_ in zip(sub_age, sub_pred):
            expected = slope * a + intercept
            ax.plot([a, a], [expected, p_], color='crimson', alpha=0.35, lw=1, zorder=2)

        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_title(r['category'], fontsize=7)
        p_str = "P<0.001" if r['perm_p'] < 0.001 else f"P={r['perm_p']:.3f}"
        ax.text(0.05, 0.92, f"n={r['n']}\nAAR={r['AAR']:+.1f} yr\n{p_str}",
                transform=ax.transAxes, fontsize=6, va='top')
        ax.set_xlabel('Chronological Age', fontsize=6)
        ax.set_ylabel('Predicted Age', fontsize=6)
        ax.tick_params(labelsize=5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("all_categories_grid.pdf", dpi=200, transparent=True)
    plt.savefig("all_categories_grid.png", dpi=200)
    plt.show()
    print("\n[*] Saved all_categories_grid.pdf/.png and all_categories_summary.csv")


if __name__ == "__main__":
    main()
