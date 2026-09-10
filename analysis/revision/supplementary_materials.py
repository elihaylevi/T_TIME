"""
supplementary_materials.py

Every producer of a manuscript **supplementary** deliverable, in one file:
Supplementary Figures S1-S4 and Supplementary Tables 1-4.

    python analysis/revision/supplementary_materials.py              # all eight
    python analysis/revision/supplementary_materials.py table4       # just one
    python analysis/revision/supplementary_materials.py suppfig1 table1
    python analysis/revision/supplementary_materials.py --list

Consolidated on 2026-09-09 from eight separate numbered scripts, which were deleted in
the same change:

    09_supp_fig1_fairness.py             -> generate_suppfig1()
    10_supp_table1_cohort.py             -> generate_table1()
    11_supp_table2_cv_folds.py           -> generate_table2()
    12_supp_table3_ablation.py           -> generate_table3()
    13_supp_fig2_wasserstein_logcount.py -> generate_suppfig2()
    14_supp_fig3_feature_weights.py      -> generate_suppfig3()
    15_supp_fig4_mae_comparison.py       -> generate_suppfig4()
    16_supp_table4_cohort_summary.py     -> generate_table4()

Each function keeps its original script's docstring verbatim - source notebook cell,
what was verified against the shipped artefact, and why. Nothing about the analyses
changed. The consolidated file was verified end to end against the eight separate
scripts before they were removed: all four CSVs byte-identical (SHA-256), all four PDFs
identical on rendered text (PDFs are never byte-reproducible here - matplotlib stamps a
creation date - so figures are compared the way docs/REPRODUCIBILITY.md prescribes, on
the text extracted from their content streams).

WHAT THE MERGE HAD TO CHANGE, AND WHY IT IS BEHAVIOUR-PRESERVING

  1. ONE `_find()`. Scripts 10-16 shared an identical resolver; script 09 had a shorter
     one. The 10-16 version is used throughout. Its first four search directories are
     exactly script 09's four, in the same order, so nothing 09 resolved can now resolve
     differently - the extra entries (`data/external`, `/content`, `/content/data`) sit
     strictly after them, and `TTIME_REVISION_OUT` defaults to the same
     `results/revision` that 09 hardcoded.

  2. `matplotlib.use("Agg")` MODULE-WIDE. Scripts 13/14/15 already forced it; 09 ran on
     whatever interactive backend was present. The backend is irrelevant to a
     `savefig(...pdf)`, which goes through the PDF backend either way - and S1's
     rendered text was verified unchanged after the switch. Script 09's trailing
     `plt.show()`, a no-op without a display, became an explicit `plt.close(fig)` so
     that running all eight in one process does not accumulate open figures.

  3. `plt.rcdefaults()` AT THE TOP OF EVERY FIGURE FUNCTION. This is the only change
     that is actually load-bearing. As separate processes each figure started from
     matplotlib's defaults; in one process script 09's module-level rcParams
     (`font.size` 7, tick widths, `lines.linewidth`) would have leaked into S2, whose
     own setup only overrides four keys. Resetting first restores the
     one-process-per-figure semantics and makes the functions independent in any order.

  4. NAMESPACING. Module-level names that collided across the eight files were moved
     inside the function that owns them or given an owning prefix - `OUT` (seven
     different values), `MODELS` (Table 3 vs Fig S4 use different display names), and
     the two different `set_nature_style()` bodies.

  Everything else - every constant, every seed, every string that identifies a figure -
  is carried over character for character.

OUTPUTS (all written to the working directory)

    Supplementary_Fig1_Fairness_Stability.pdf     Supp_Table_1_cohort_characteristics.csv
    Supp_Fig_S2_wasserstein_logcount.pdf          Supp_Table_2_cv_folds.csv
    Supp_Fig_S3_feature_weights.pdf               Supp_Table_3_ablation.csv
    Supp_Fig_S4_mae_comparison.pdf                Supp_Table_4_cohort_summary.csv

  The published copies live in `results/supplementary/`. Tables 1-3 and Figures S1-S4
  reproduce their original-submission counterparts; Table 4 *replaces* its counterpart -
  see `generate_table4` and the archived internal changelog §12
  (`~/Downloads/ttime_archive/docs_internal/CHANGELOG_RESTRUCTURE.md`).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # headless; see note 2 in the module docstring
import matplotlib.gridspec as gridspec     # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402
import seaborn as sns                      # noqa: E402
import statsmodels.formula.api as smf      # noqa: E402
from scipy import stats                    # noqa: E402

REPO = Path(__file__).resolve().parents[2]


# ============================================================================
# Shared helpers
# ============================================================================

def _find(name):
    """Locate a shipped input. Search order is the one scripts 10-16 shared."""
    _rev = Path(os.environ.get("TTIME_REVISION_OUT", REPO / "results" / "revision"))
    for d in [Path("."), _rev, REPO / "data", REPO, REPO / "data" / "external",
              Path(__file__).resolve().parent, Path("/content"), Path("/content/data")]:
        p = d / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(name)


def _canon_sex(s):
    return (s.astype(str).str.strip().str.capitalize()
            .replace({"1": "Male", "1.0": "Male", "0": "Female", "0.0": "Female"}))


def _reset_style():
    """Undo any rcParams a previously-run figure set. See note 3 in the module docstring."""
    plt.rcdefaults()


# ============================================================================
# Supplementary Figure S1
# ============================================================================

def generate_suppfig1():
    """
    Supplementary Figure S1 - model fairness and lifespan stability of the sex offset.

    C2 (2026-09-08): the input was the hardcoded 'test_preds_covid.csv', a filename that
    does not exist anywhere in the repository, so this script could not run. It now resolves
    results/revision/test_preds.csv (the canonical primary predictions from
    analysis/revision/01_train_primary_model.py) through the same _find() search list the
    analysis/revision/* scripts use. Nothing else was changed.
    """
    out = "Supplementary_Fig1_Fairness_Stability.pdf"
    _reset_style()

    # ==========================================
    # 1. Nature Style Settings (Vector & 180mm)
    # ==========================================
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 7
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['lines.linewidth'] = 1.0

    # Colors from previous figures for consistency
    colors = {'Male': '#D55E00', 'Female': '#0072B2'}

    def add_panel_letter(ax, letter, x=-0.1, y=1.05):
        ax.text(x, y, letter, transform=ax.transAxes,
                fontsize=8, fontweight='bold', va='bottom', ha='right')

    # ==========================================
    # 2. Data Loading & Robust Cleaning
    # ==========================================
    df = pd.read_csv(_find('test_preds.csv'), low_memory=False)

    sex_col = next((c for c in df.columns
                    if c.lower() in ['biological sex', 'sex', 'biological_sex']), 'Sex')
    df.rename(columns={sex_col: 'Sex', 'Age': 'y_true'}, inplace=True)
    df['Sex'] = (df['Sex'].astype(str).str.strip().str.capitalize()
                 .replace({'1': 'Male', '1.0': 'Male', '0': 'Female', '0.0': 'Female'}))
    df = df[df['Sex'].isin(['Male', 'Female'])]

    df['residual'] = df['y_pred'] - df['y_true']
    df['abs_residual'] = df['residual'].abs()

    # 90th Percentile Filter (Robust Analysis)
    cut90 = df['abs_residual'].quantile(0.90)
    df_clean = df[df['abs_residual'] <= cut90].copy()

    # ==========================================
    # 3. Statistical Computations
    # ==========================================
    females_abs = df_clean[df_clean['Sex'] == 'Female']['abs_residual']
    males_abs = df_clean[df_clean['Sex'] == 'Male']['abs_residual']

    stat_ks, p_ks = stats.ks_2samp(females_abs, males_abs)
    stat_levene, p_levene = stats.levene(females_abs, males_abs)

    # Interaction Model (Age x Sex)
    model = smf.ols('residual ~ y_true * C(Sex)', data=df_clean).fit()
    interaction_pval = model.pvalues.get('y_true:C(Sex)[T.Male]', np.nan)

    # ==========================================
    # 4. Figure Construction
    # ==========================================
    # Width: 180mm (7.08 inches), Height: 3 inches (compact for supplement)
    fig = plt.figure(figsize=(7.08, 3), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.3,
                           left=0.08, right=0.95, top=0.88, bottom=0.18)

    # --- Panel A: ECDF of Absolute Residuals ---
    ax_a = fig.add_subplot(gs[0])
    sns.ecdfplot(data=df_clean, x='abs_residual', hue='Sex', palette=colors, ax=ax_a, lw=1.2)
    add_panel_letter(ax_a, 'a')

    ax_a.set_title('Error Distribution Symmetry', loc='center', pad=10,
                   fontsize=7, fontweight='bold')
    ax_a.set_xlabel('Absolute Prediction Error (Years)')
    ax_a.set_ylabel('Cumulative Probability')

    # Annotation box for stats
    stats_text = f"KS test $P$ = {p_ks:.2f}\nLevene $P$ = {p_levene:.2f}"
    ax_a.text(0.95, 0.05, stats_text, transform=ax_a.transAxes, fontsize=6,
              ha='right', va='bottom',
              bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax_a.legend(frameon=False, fontsize=6, loc='center right')

    # --- Panel B: Bias Stability across Decades ---
    ax_b = fig.add_subplot(gs[0, 1])

    bins = np.arange(20, 90, 10)  # 20s to 80s
    df_clean['Age_Decade'] = pd.cut(df_clean['y_true'], bins=bins, right=False)
    decades = sorted(df_clean['Age_Decade'].dropna().unique())

    diffs, cis, labels = [], [], []

    for d in decades:
        sub = df_clean[df_clean['Age_Decade'] == d]
        f_sub = sub[sub['Sex'] == 'Female']['residual']
        m_sub = sub[sub['Sex'] == 'Male']['residual']

        if len(f_sub) >= 5 and len(m_sub) >= 5:
            diff = m_sub.mean() - f_sub.mean()
            se = np.sqrt(f_sub.var() / len(f_sub) + m_sub.var() / len(m_sub))
            diffs.append(diff)
            cis.append(1.96 * se)
            labels.append(f"{int(d.left)}s")

    x_pos = np.arange(len(labels))
    ax_b.errorbar(x_pos, diffs, yerr=cis, fmt='o', color='black', markersize=3,
                  capsize=2, elinewidth=0.8, markeredgewidth=0.8)

    ax_b.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    # Global mean bias line
    mean_m = df_clean[df_clean['Sex'] == 'Male']['residual'].mean()
    mean_f = df_clean[df_clean['Sex'] == 'Female']['residual'].mean()
    ax_b.axhline(mean_m - mean_f, color='#27AE60', linestyle=':', linewidth=0.8,
                 alpha=0.8, label='Global Mean Bias')

    add_panel_letter(ax_b, 'b')
    ax_b.set_title('Bias Stability Across Lifespan', loc='center', pad=10,
                   fontsize=7, fontweight='bold')
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(labels)
    ax_b.set_xlabel('Age Decade')
    ax_b.set_ylabel('Bias (Male - Female Residuals) [yr]')

    # Interaction stat text
    interaction_text = f"Age $\\times$ Sex Interaction\n$P$ = {interaction_pval:.2f}"
    ax_b.text(0.05, 0.95, interaction_text, transform=ax_b.transAxes, fontsize=6,
              ha='left', va='top',
              bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    sns.despine(fig)

    # ==========================================
    # 5. Save Final Figure
    # ==========================================
    plt.savefig(out, format='pdf', transparent=True)
    plt.close(fig)

    print("[OK] Supplementary Figure S1 generated as vector PDF.")
    return out


# ============================================================================
# Supplementary Table 1
# ============================================================================

_T1_VO_LABEL = "Longitudinal Vo’, Italy"


def _table1_stats(data):
    n = len(data)
    if n == 0:
        return pd.Series({"N (Samples)": 0, "Age: Median [IQR]": "-",
                          "Age: Mean (±SD)": "-", "Sex: Female (%)": "-"})
    med, q1, q3 = data["Age"].median(), data["Age"].quantile(0.25), data["Age"].quantile(0.75)
    fem = int((data["Biological Sex"] == "Female").sum())
    return pd.Series({
        "N (Samples)": int(n),
        "Age: Median [IQR]": f"{med:.1f} [{q1:.1f} – {q3:.1f}]",
        "Age: Mean (±SD)": f"{data['Age'].mean():.1f} (±{data['Age'].std():.1f})",
        "Sex: Female (%)": f"{fem} ({fem / n * 100:.1f}%)",
    })


def generate_table1():
    """
    Regenerates **Supplementary Table 1** — cohort demographic characteristics, broken
    down by cohort (Vo' Italy vs ImmuneCODE) and by split (training vs internal test).

    VERIFIED (2026-09-08). Its output matches the shipped
    `results/supplementary/tables/Supplementary Table 1.xlsx` value for value:

        N               3271 / 2534 / 737 / 2453 / 818
        Age median      53.0 / 51.0 / 59.0 / 53.0 / 52.0
        Age mean (SD)   51.2 (21.0) / 49.1 (21.2) / 58.4 (18.6) / 51.2 (21.0) / 51.0 (21.0)
        Female          1708 (52.2%) / 1306 (51.5%) / 402 (54.5%) / 1284 (52.3%) / 424 (51.8%)

    Extracted verbatim from cell 1 of the former `sup.ipynb`; only the input paths were
    routed through `_find()` and the console text translated. An earlier draft (cell 0)
    produced a three-column version without the cohort split and does NOT match the
    shipped table — it was not carried over.

    COHORT ASSIGNMENT
      A sample counts as Vo' Italy when any `Sample_ID` from `TCR_shared_ELIHAY_full.csv`
      appears as a substring of its `sample name`; everything else is ImmuneCODE. This
      substring rule is the notebook's own and is kept unchanged, because changing it
      would change the published table.

    INPUTS  (all shipped)
      - data/train_combined_matrix_pruned_95.csv
      - data/test_combined_matrix_pruned_95.csv
      - data/TCR_shared_ELIHAY_full.csv

    OUTPUT
      - Supp_Table_1_cohort_characteristics.csv, in the working directory
    """
    out = "Supp_Table_1_cohort_characteristics.csv"

    train = pd.read_csv(_find("train_combined_matrix_pruned_95.csv"))
    test = pd.read_csv(_find("test_combined_matrix_pruned_95.csv"))
    vo_ids = pd.read_csv(_find("TCR_shared_ELIHAY_full.csv"))["Sample_ID"].astype(str).unique()

    train["Set"] = "Training Set"
    test["Set"] = "Internal Test Set"
    df = pd.concat([train, test], axis=0)

    print(f"[*] identifying cohorts against {len(vo_ids)} Vo' sample ids ...")
    vo = {v for v in vo_ids}

    def is_vo(name):
        s = str(name).lower()
        return any(v in s for v in vo)

    df["Cohort"] = df["sample name"].apply(lambda x: _T1_VO_LABEL if is_vo(x) else "ImmuneCODE")

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Biological Sex"] = _canon_sex(df["Biological Sex"])

    table = pd.DataFrame()
    table["Total Cohort"] = _table1_stats(df)
    table = pd.concat([table,
                       df.groupby("Cohort").apply(_table1_stats).T,
                       df.groupby("Set").apply(_table1_stats).T], axis=1)
    order = ["Total Cohort", _T1_VO_LABEL, "ImmuneCODE", "Training Set", "Internal Test Set"]
    table = table[[c for c in order if c in table.columns]]

    print("\n=== Supplementary Table 1: cohort characteristics ===")
    print(table.to_string())
    table.to_csv(out)
    print(f"\n[*] wrote {out}")
    return out


# ============================================================================
# Supplementary Table 2
# ============================================================================

_T2_PARAM_COLS = ["n_hidden_layers", "hidden_width", "activation", "dropout", "batch_size",
                  "lr", "weight_decay", "residual", "scheduler", "l0_lambda"]


def generate_table2():
    """
    Regenerates **Supplementary Table 2** — the 5-fold cross-validation results for the
    selected hyper-parameter configuration (MAE, RMSE, R² per fold plus a mean ± SD row).

    VERIFIED (2026-09-08). Output is value-for-value identical to the shipped
    `results/supplementary/tables/Supplementary Table 2.xlsx`:

        fold 1   7.602115   10.54345989227295   0.7640355668670872
        fold 2   7.591525    9.7723970413208    0.7912807615753307
        fold 3   7.178809    9.497020721435549  0.7886579676203116
        fold 4   7.1311603   9.141667366027832  0.8025633302386934
        fold 5   7.2768145   9.607603073120115  0.7853466278396115
        Mean     7.356 (±0.226)  9.712 (±0.519)  0.786 (±0.014)

    MODEL SELECTION
      `per_fold_gates.csv` holds every fold of every hyper-parameter configuration from the
      L0-gate sweep. The reported configuration is the one with the lowest mean MAE across
      its folds; its five folds are then tabulated. That selection rule is the notebook's
      own and is kept unchanged - altering it would change the published table.

    INPUT   (shipped)
      - per_fold_gates.csv   (8,241 rows: the full sweep)

    OUTPUT
      - Supp_Table_2_cv_folds.csv, in the working directory

    Extracted verbatim from cell 4 of the former `sup.ipynb` (archived at
    `~/Downloads/ttime_archive/sup.ipynb`); only the input path was routed through
    `_find()` and the console text translated.
    """
    out = "Supp_Table_2_cv_folds.csv"

    df = pd.read_csv(_find("per_fold_gates.csv"), index_col=False)
    for col in ("mae", "rmse", "r2"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    clean = df.dropna(subset=["mae", "rmse", "r2"]).copy()
    print(f"[*] sweep rows: {len(df)}  usable: {len(clean)}")

    best = clean.groupby(_T2_PARAM_COLS)["mae"].mean().sort_values().index[0]
    best_df = clean.set_index(_T2_PARAM_COLS).loc[[best]].reset_index()
    print(f"[*] selected configuration (lowest mean MAE), {len(best_df)} folds")

    table = best_df[["fold", "mae", "rmse", "r2"]].copy()
    table.columns = ["Fold", "MAE (Years)", "RMSE (Years)", "R2 Score"]
    summary = pd.DataFrame({
        "Fold": ["Mean (±SD)"],
        "MAE (Years)": [f"{table['MAE (Years)'].mean():.3f} (±{table['MAE (Years)'].std():.3f})"],
        "RMSE (Years)": [f"{table['RMSE (Years)'].mean():.3f} (±{table['RMSE (Years)'].std():.3f})"],
        "R2 Score": [f"{table['R2 Score'].mean():.3f} (±{table['R2 Score'].std():.3f})"],
    })
    res = pd.concat([table.astype(str), summary], ignore_index=True)

    print("\n=== Supplementary Table 2: cross-validation results (best model) ===")
    print(res.to_string(index=False))
    res.to_csv(out, index=False)
    print(f"\n[*] wrote {out}")
    return out


# ============================================================================
# Supplementary Table 3
# ============================================================================

_T3_MODELS = {
    "Pred_Combined": "Combined 5k\n(K-mers + TCR)",
    "Pred_Kmers": "K-mers Only",
    "Pred_TCR": "TCR Only",
}
_T3_N_BOOT, _T3_SEED = 1000, 42


def _table3_bootstrap(df, n_iterations=_T3_N_BOOT, seed=_T3_SEED):
    np.random.seed(seed)
    y = df["Actual_Age"].values
    n = len(df)
    acc = {m: [] for m in _T3_MODELS}
    print(f"[*] {n_iterations} bootstrap iterations on {n} samples ...")
    for _ in range(n_iterations):
        idx = np.random.randint(0, n, n)          # shared across models -> paired
        yt = y[idx]
        for m in _T3_MODELS:
            acc[m].append(np.mean(np.abs(yt - df[m].values[idx])))
    rows = []
    for m, name in _T3_MODELS.items():
        a = np.array(acc[m])
        # Mean_MAE is the MEAN OF THE BOOTSTRAP DISTRIBUTION, not the point estimate on
        # the full sample - that is what the shipped table records (7.490731, not 7.499).
        rows.append(dict(Model_Col=m, Display_Name=name,
                         Mean_MAE=float(np.mean(a)),
                         Lower_CI=float(np.percentile(a, 2.5)),
                         Upper_CI=float(np.percentile(a, 97.5))))
    return pd.DataFrame(rows)


def _table3_pvalues(df):
    y = df["Actual_Age"].values
    ec = np.abs(y - df["Pred_Combined"].values)
    return {m: stats.wilcoxon(ec, np.abs(y - df[m].values), alternative="less")[1]
            for m in ("Pred_Kmers", "Pred_TCR")}


def generate_table3():
    """
    Regenerates **Supplementary Table 3** — the feature-ablation comparison: mean absolute
    error of the combined model against k-mer-only and TCR-only variants, with bootstrap
    95% confidence intervals and a Wilcoxon test of each variant against the combined model.

    VERIFIED (2026-09-08). Output is value-for-value identical to the shipped
    `results/supplementary/tables/Supplementary Table 3.xlsx`:

        Pred_Combined  7.490731   [7.061724, 7.933096]
        Pred_Kmers    10.102281   [9.548410, 10.675197]
        Pred_TCR      11.848007   [11.173615, 12.563967]

    All 15 numeric cells (Mean_MAE, Lower_CI, Upper_CI, Error_Lower, Error_Upper x 3 models)
    match to within 1e-6, and the two p-values match as well: 5.871638572069639e-19 and
    1.0314558329076263e-27.

    METHOD (unchanged from the notebook)
      1,000 bootstrap resamples of the 818 held-out samples, `np.random.seed(42)`, shared
      indices across the three models so the comparison is paired; 2.5/97.5 percentiles for
      the CI. p-values are a one-sided Wilcoxon signed-rank test on absolute errors
      (combined < variant).

    INPUT   (shipped)
      - all_model_predictions.csv   (818 rows: Actual_Age, Pred_Combined, Pred_Kmers, Pred_TCR)

    OUTPUT
      - Supp_Table_3_ablation.csv, in the working directory

    Extracted from cell 10 of the former `sup.ipynb` (archived at
    `~/Downloads/ttime_archive/sup.ipynb`). Cell 6 computes the same statistics but writes
    only the figure, no table; cell 10 supersedes it. The plotting half of cell 10 was left
    behind - the shipped Supplementary Figure S4 does not match its rendering (see
    docs/REPRODUCIBILITY.md, Tier C).
    """
    out = "Supp_Table_3_ablation.csv"

    df = pd.read_csv(_find("all_model_predictions.csv"))
    res = _table3_bootstrap(df)
    res["Error_Lower"] = res["Mean_MAE"] - res["Lower_CI"]
    res["Error_Upper"] = res["Upper_CI"] - res["Mean_MAE"]
    res["p_value_vs_Combined"] = res["Model_Col"].map(_table3_pvalues(df))
    res["Display_Name_Clean"] = res["Display_Name"].str.replace("\n", " ", regex=False)

    print("\n=== Supplementary Table 3: feature ablation (95% CI) ===")
    for _, r in res.iterrows():
        p = "" if pd.isna(r.p_value_vs_Combined) else f"   p vs combined = {r.p_value_vs_Combined:.3g}"
        print(f"   {r.Display_Name_Clean:28s} {r.Mean_MAE:6.3f}  "
              f"[{r.Lower_CI:.3f}, {r.Upper_CI:.3f}]{p}")
    res.to_csv(out, index=False)
    print(f"\n[*] wrote {out}")
    return out


# ============================================================================
# Supplementary Figure S2
# ============================================================================

_S2_FEATURE = "mean_age"
_S2_PALETTE = {
    "Significant (Old)": "#D55E00",
    "Significant (Young)": "#0072B2",
    "Not Significant": "#B0B0B0",
}
_S2_TRUE_SET = {True, 1, "True", "true", "Yes", "yes"}


def _s2_find_age_table():
    for n in ("updated_tcr_age_lists_with_all_significance.csv.gz",
              "updated_tcr_age_lists_with_all_significance.csv"):
        try:
            return _find(n)
        except FileNotFoundError:
            continue
    raise FileNotFoundError("updated_tcr_age_lists_with_all_significance.csv[.gz]")


def _s2_determine_group(row):
    if row["signed_wasserstein_significant"] in _S2_TRUE_SET:
        return "Significant (Old)" if row["signed_wasserstein"] > 0 else "Significant (Young)"
    return "Not Significant"


def generate_suppfig2():
    """
    Regenerates **Supplementary Figure S2** — the distribution of per-clonotype mean donor
    age, split by signed-Wasserstein significance group, on a log count scale.

    VERIFIED (2026-09-08). Every text string rendered into the shipped
    `results/supplementary/figures/Supplementary Figure S2.pdf` is reproduced **exactly**:

        x tick labels   40 45 50 55 60
        x label         "mean_age Value"
        y tick labels   10^0 ... 10^4
        y label         "Number of TCRs (Log Scale)"
        title           "mean_age by Significance (Log Count Scale)"
        legend          "Significance_Group" / Not Significant / Significant (Old) /
                        Significant (Young)

    The rendered PDF is not byte-identical (42 KB here vs 22 KB shipped) because matplotlib
    and font versions differ; the figure content is the same.

    HOW THIS ONE WAS IDENTIFIED
      Five other notebook cells draw a similar distribution, and all five were run and
      rejected: they set polished axis labels ("Mean Age of Sample List", "Number of TCRs",
      "TCR Significance Across Age") which do **not** appear in the shipped PDF. The shipped
      figure keeps the raw column names, which is what this cell's f-strings produce. The
      y-label and title above are unique to it among all 21 cells.

    INPUT   (shipped, gzipped)
      - data/updated_tcr_age_lists_with_all_significance.csv.gz

      The notebook read the uncompressed `.csv`; this script accepts either, since the file
      was gzipped in the repository on 2026-09-08. pandas infers the codec from the suffix.

    OUTPUT
      - Supp_Fig_S2_wasserstein_logcount.pdf, in the working directory

    Extracted from cell 9 of the former `sup.ipynb` (archived at
    `~/Downloads/ttime_archive/sup.ipynb`). Only the input path, the output filename and the
    console text were changed.
    """
    out = "Supp_Fig_S2_wasserstein_logcount.pdf"

    src = _s2_find_age_table()
    print(f"[*] reading {src}")
    df = pd.read_csv(src)
    df["Significance_Group"] = df.apply(_s2_determine_group, axis=1)
    print("[*] group sizes:")
    for g, c in df["Significance_Group"].value_counts().items():
        print(f"      {g:22s} {c:,}")

    _reset_style()
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]
    plt.rcParams["axes.linewidth"] = 1.0

    plt.figure(figsize=(8, 5), dpi=300)
    sns.histplot(data=df, x=_S2_FEATURE, hue="Significance_Group", palette=_S2_PALETTE,
                 element="step", stat="count", common_norm=False, bins=40,
                 alpha=0.3, linewidth=2)
    plt.yscale("log")
    # These three strings are what identify the figure - do not "tidy" them.
    plt.title(f"{_S2_FEATURE} by Significance (Log Count Scale)",
              fontsize=14, fontweight="bold", pad=15)
    plt.xlabel(f"{_S2_FEATURE} Value", fontsize=12)
    plt.ylabel("Number of TCRs (Log Scale)", fontsize=12, fontweight="bold")
    sns.despine()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"\n[*] wrote {out}")
    return out


# ============================================================================
# Supplementary Figure S3
# ============================================================================

_S3_COLOR_PALETTE = {"K-mer": "#0072B2", "TCR-Wasserstein": "#D55E00"}


def _s3_drop_legend(ax):
    """seaborn <0.13 has no `legend` kwarg; strip the auto-legend instead."""
    lg = ax.get_legend()
    if lg is not None:
        lg.remove()


def _s3_set_nature_style():
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })


def generate_suppfig3():
    """
    Regenerates **Supplementary Figure S3** — a three-panel summary of the trained model's
    input-layer weights: (a) the 25 highest-weighted features, (b) the weight distribution
    by feature type, (c) each type's share of total importance as a donut.

    VERIFIED (2026-09-08). Every text string rendered into the shipped
    `results/supplementary/figures/Supplementary Figure S3.pdf` is reproduced **exactly**
    (1,032 characters, identical): the three panel titles `a  Top 25 Most Important
    Features`, `b  Weight Distribution`, `c  Total Contribution`; the x-label `Absolute
    Weight`; the donut percentages 86.7% / 13.3%; the shared legend `K-mer Frequency
    Features` / `TCR-Wasserstein Distance Features`; and all 25 CDR3 tick labels.

    The `Absolute Weight` x-label is what identifies this figure: the five other cells that
    plot the same weights all use `Absolute Input Layer Weight`.

    The rendered PDF is not byte-identical (55 KB here vs 29 KB shipped) - matplotlib and
    font versions differ - but the content matches.

    SEABORN COMPATIBILITY (the one change from the notebook)
      The original passed `legend=False` to `sns.barplot` and `sns.violinplot`. That keyword
      arrived in seaborn 0.13; under 0.11/0.12 it is forwarded to matplotlib and raises
      `AttributeError: 'Rectangle' object has no property 'legend'`. Here the keyword is
      dropped and the auto-generated legend removed afterwards, which is exactly what
      `legend=False` does. Output verified identical to the shipped figure, so the shim is
      behaviour-preserving and the script runs on both old and new seaborn.

    INPUT   (shipped)
      - weights.csv   (5,459 rows: Feature, Weight, Type)

    OUTPUT
      - Supp_Fig_S3_feature_weights.pdf, in the working directory

    Extracted from cell 18 of the former `sup.ipynb` (archived at
    `~/Downloads/ttime_archive/sup.ipynb`).
    """
    out = "Supp_Fig_S3_feature_weights.pdf"

    src = _find("weights.csv")
    print(f"[*] reading {src}")
    df = pd.read_csv(src)
    _reset_style()
    _s3_set_nature_style()

    fig = plt.figure(figsize=(7.08, 6.3), dpi=300)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                           hspace=0.25, wspace=0.35)

    # (a) top 25 features
    ax_a = fig.add_subplot(gs[:, 0])
    top_25 = df.sort_values(by="Weight", ascending=False).head(25)
    sns.barplot(data=top_25, x="Weight", y="Feature", hue="Type",
                palette=_S3_COLOR_PALETTE, dodge=False, ax=ax_a, alpha=1.0)
    _s3_drop_legend(ax_a)
    ax_a.set_title("a  Top 25 Most Important Features", loc="left", fontweight="bold")
    ax_a.set_xlabel("Absolute Weight")
    ax_a.set_ylabel("")
    ax_a.set_xlim(0, top_25["Weight"].max() * 1.1)
    sns.despine(ax=ax_a)

    # (b) weight distribution
    ax_b = fig.add_subplot(gs[0, 1])
    sns.violinplot(data=df, x="Type", y="Weight", hue="Type",
                   palette=_S3_COLOR_PALETTE, inner=None, alpha=0.9, ax=ax_b)
    _s3_drop_legend(ax_b)
    sns.boxplot(data=df, x="Type", y="Weight", width=0.12, color="white",
                ax=ax_b, showfliers=False, boxprops={"linewidth": 0.7})
    ax_b.set_title("b  Weight Distribution", loc="left", fontweight="bold")
    ax_b.set_xlabel("")
    ax_b.set_ylabel("Weight")
    sns.despine(ax=ax_b)

    # (c) total contribution donut
    ax_c = fig.add_subplot(gs[1, 1])
    sum_weights = df.groupby("Type")["Weight"].sum()
    ax_c.pie(sum_weights, labels=None, autopct="%1.1f%%",
             colors=[_S3_COLOR_PALETTE[t] for t in sum_weights.index], startangle=90,
             wedgeprops={"edgecolor": "white", "linewidth": 1.2},
             textprops={"fontsize": 7, "fontweight": "bold", "color": "white"})
    ax_c.add_artist(plt.Circle((0, 0), 0.72, fc="white"))
    ax_c.set_title("c  Total Contribution", loc="left", fontweight="bold")

    handles = [plt.Rectangle((0, 0), 1, 1, color=_S3_COLOR_PALETTE["K-mer"]),
               plt.Rectangle((0, 0), 1, 1, color=_S3_COLOR_PALETTE["TCR-Wasserstein"])]
    fig.legend(handles, ["K-mer Frequency Features", "TCR-Wasserstein Distance Features"],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.02))
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(out, bbox_inches="tight")
    plt.close()

    pct = sum_weights / sum_weights.sum() * 100
    print(f"[*] total modelling features: {len(df):,}")
    for t, p in pct.items():
        print(f"      {t:18s} {p:.1f}%")
    top = top_25.iloc[0]
    print(f"[*] highest-weighted feature: {top['Feature']} ({top['Type']}) = {top['Weight']:.5f}")
    print(f"\n[*] wrote {out}")
    return out


# ============================================================================
# Supplementary Figure S4
# ============================================================================

_S4_MODELS = {"Pred_Combined": "Combined", "Pred_Kmers": "K-mers Only", "Pred_TCR": "TCR Only"}
_S4_COLORS = {"Pred_Combined": "#006400", "Pred_Kmers": "#0072B2", "Pred_TCR": "#D55E00"}
_S4_N_BOOT, _S4_SEED = 1000, 42


def _s4_set_nature_style():
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "legend.fontsize": 6, "axes.titlesize": 8,
        "axes.linewidth": 0.5, "grid.linewidth": 0.3, "lines.linewidth": 1.2,
    })


def generate_suppfig4():
    """
    Regenerates **Supplementary Figure S4** — the feature-ablation bar chart: mean absolute
    error of the combined model against the k-mer-only and TCR-only variants, with bootstrap
    error bars.

    VERIFIED (2026-09-08). Every text string rendered into the shipped
    `results/supplementary/figures/Supplementary Figure S4.pdf` is reproduced **exactly**
    (208 characters, identical): the bar labels `Combined` / `K-mers Only` / `TCR Only`, the
    y-label `Mean Absolute Error (Years)`, the title `Model Accuracy Comparison`, the bar
    annotations `7.50` / `10.11` / `11.87`, and the y tick sequence `0 2 4 6 8 10 12 14`.

    HOW THIS ONE WAS IDENTIFIED
      Three notebook cells draw this same chart with the same title and the same three bar
      values; they are told apart **only by the y-limit**, which sets the tick sequence:

          cell 11   no ylim (matplotlib auto)   ticks 0 .. 12          rejected
          cell 19   ylim = 1.6  x max_reach     ticks 0.0 .. 20.0      rejected
          cell 13   ylim = 1.25 x max_reach     ticks 0 .. 14          MATCHES

      Cell 10, which computes the same ablation statistics for Supplementary Table 3, was also
      rejected: its plotting half titles the figure `Age Prediction Accuracy by Feature
      Cohort`, which does not appear in the shipped PDF. Cell 14 is a byte-identical duplicate
      of cell 13.

    RELATIONSHIP TO SUPPLEMENTARY TABLE 3
      Same data, different statistic - do not expect the numbers to line up digit for digit.
      The table reports the **mean of the bootstrap distribution** (7.490731) with 2.5/97.5
      percentile bounds; this figure annotates the **point estimate on the full sample**
      (7.4998 -> `7.50`) with +/- 1.96 x SD of the bootstrap means. Both are as published.

    The rendered PDF is not byte-identical (28 KB here vs 17 KB shipped) - matplotlib and
    font versions differ - but the content matches.

    INPUT   (shipped)
      - all_model_predictions.csv   (818 rows: Actual_Age, Pred_Combined, Pred_Kmers, Pred_TCR)

    OUTPUT
      - Supp_Fig_S4_mae_comparison.pdf, in the working directory

    Extracted from cell 13 of the former `sup.ipynb` (archived at
    `~/Downloads/ttime_archive/sup.ipynb`). Only the input path, the output filename and the
    console text were changed.
    """
    out = "Supp_Fig_S4_mae_comparison.pdf"

    src = _find("all_model_predictions.csv")
    print(f"[*] reading {src}")
    df = pd.read_csv(src)
    _reset_style()
    _s4_set_nature_style()

    np.random.seed(_S4_SEED)
    rows = []
    for col, name in _S4_MODELS.items():
        errors = np.abs(df["Actual_Age"] - df[col])
        boot = [errors.sample(frac=1, replace=True).mean() for _ in range(_S4_N_BOOT)]
        rows.append({"Name": name, "MAE": float(np.mean(errors)),
                     "err": float(1.96 * np.std(boot)), "col": col})
    res = pd.DataFrame(rows)

    plt.figure(figsize=(3.5, 4), dpi=300)
    x = np.arange(len(res))
    bars = plt.bar(x, res["MAE"], yerr=res["err"], capsize=4,
                   color=[_S4_COLORS[c] for c in res["col"]],
                   edgecolor="black", linewidth=0.6, alpha=1.0)
    plt.xticks(x, res["Name"])
    plt.ylabel("Mean Absolute Error (Years)")
    plt.title("Model Accuracy Comparison")
    # The 1.25 factor is what fixes the y ticks at 0..14 and identifies this figure.
    plt.ylim(0, (res["MAE"] + res["err"]).max() * 1.25)
    for bar in bars:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, y + 0.3, f"{y:.2f}",
                 ha="center", va="bottom", fontsize=6, fontweight="bold")
    sns.despine()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

    print(f"\n=== Supplementary Figure S4: ablation MAE ({_S4_N_BOOT} bootstrap, 95% CI) ===")
    for _, r in res.iterrows():
        print(f"   {r.Name:14s} {r.MAE:6.2f}  (+/- {r.err:.2f})")
    print(f"\n[*] wrote {out}")
    return out


# ============================================================================
# Supplementary Table 4
# ============================================================================

# Canonical n's, from results/numbers.json -> age_acceleration. Asserted, not assumed:
# if script 04 is ever re-run with different definitions this must fail loudly rather
# than quietly ship a table that disagrees with Section 2.5.
_T4_EXPECTED_N = {"Healthy Reference": 762, "Hypertension": 66,
                  "Autoimmune": 22, "Cancer": 22, "All Clinical Cases": 244}


def _t4_norm_id(s):
    """Same normalisation script 04 uses to match clinical patients into train/test."""
    return str(s).lower().replace("copy of ", "").strip().replace("_tcrb", "")


def _table4_summarize(df, label):
    """N / median age [IQR] / female n (%) for one group."""
    n = len(df)
    age = pd.to_numeric(df["Age"], errors="coerce").dropna()
    if len(age) != n:
        print(f"    ! {label}: {n - len(age)} row(s) with unparseable Age, "
              f"excluded from the age statistics only")
    med, q1, q3 = age.median(), age.quantile(0.25), age.quantile(0.75)

    sex = _canon_sex(df["Biological Sex"])
    n_known = int(sex.isin(["Male", "Female"]).sum())
    n_fem = int((sex == "Female").sum())
    # Denominator is the sex-known count, so an unrecorded sex cannot silently
    # deflate the percentage - it was exactly this kind of row that cost the
    # superseded table its 17th hypertension patient.
    pct = 100.0 * n_fem / n_known if n_known else float("nan")
    if n_known != n:
        print(f"    ! {label}: sex unrecorded for {n - n_known} of {n}; "
              f"% female is over the {n_known} with a recorded sex")

    return pd.Series({
        "N": n,
        "Age: Median [IQR]": f"{med:.1f} [{q1:.1f}-{q3:.1f}]",
        "Age: Mean (SD)": f"{age.mean():.1f} ({age.std():.1f})",
        "Sex: Female n (%)": f"{n_fem} ({pct:.1f}%)",
    }, name=label)


def generate_table4():
    """
    Regenerates **Supplementary Table 4** - the demographic summary of the clinical
    sub-cohorts against the healthy reference population, i.e. the cohort table that
    backs the AAR analysis of Section 2.5.

      !! THIS FUNCTION DOES NOT REPRODUCE THE ORIGINAL-SUBMISSION
      !! `Supplementary Table 4.xlsx`. IT REPLACES IT.
      !!
      !! As of 2026-09-09 its output IS
      !! `results/supplementary/tables/Supplementary Table 4.xlsx`. The original-submission
      !! file was retired outside this repository, to
      !! `~/Downloads/ttime_archive/docs_internal/archive/supplementary_table4_original_submission.xlsx`.

    WHY THE ORIGINAL TABLE IS SUPERSEDED
      The original-submission XLSX reports

          Healthy Reference   Hypertension   Immune System   Neuro-Psych
          N                762             16              14             6

      The `Healthy Reference` column is correct and is carried over unchanged (see
      below). The three clinical columns are not: they disagree with the n = 66 / 22 / 22
      that Section 2.5, `results/numbers.json` and `04_clinical_aar_pipeline.py` report
      for the *same* analysis - the manuscript caption says the table is "used to assess
      the impact of clinical status on biological age acceleration", which is exactly the
      Section 2.5 AAR test.

      The shipped numbers were traced (2026-09-09) to
      `~/Downloads/ttime_archive/as_submitted/scripts/superseded/Figure2_Age_Prediction_Clinical.py`
      and reproduced from it exactly. Two independent pre-revision defects stack:

      (1) COHORT RESTRICTION - the dominant one. The superseded script tabulated only the
          clinical patients that happen to fall in the internal test split. Of the 244
          clinical patients, 56 are in `test_preds.csv` (818 - 56 = 762, which is where
          the correct `Healthy Reference` n comes from); one of those 56 has a missing
          `Biological Sex` and was dropped by that script's `Sex.isin(['Male','Female'])`
          filter, leaving 55. Categorising those 55 gives 16 / 14 / 6.
          The Section 2.5 AAR analysis uses all 244 clinical patients - they were removed
          from train *and* test precisely so that the model never sees them, so
          restricting the table to the test split is not meaningful here.

      (2) SUPERSEDED CATEGORY TAXONOMY. The old definitions used the categories
          `Hypertension` / `Immune System` / `Neuro-Psych` (no cancer group at all), and
          mis-specified them: the boolean flag column names `uses_ace_inhibitor`,
          `uses_arb` and `uses_autoimmune_medications` were listed under `words` instead
          of under `flags`, so they were regex-searched inside free-text fields rather
          than read as flags, and never matched. The canonical definitions in
          `04_clinical_aar_pipeline.py` / `05_all_categories_exploration.py` put them in
          `flags`, and use `Hypertension` / `Autoimmune` / `Cancer`. This is the same
          category-definition defect already recorded in script 04's docstring
          (cancer n=17 under flags-only vs n=22 canonical) and in discussion D5.

      For the record, the *other* candidate producer - cell 3 of the archived `sup.ipynb`
      - is NOT the source of the shipped table either. It uses text + flags (not
      flags-only) with the same superseded taxonomy but over all 244 patients, and prints
      `All Clinical Cases` n = 244 / 65 / 31 / 24. Those 65 / 31 / 24 are the numbers
      script 04 carries as `sol_old_n`. Cell 3 emits no `Healthy Reference` column.

    WHAT THIS FUNCTION DOES INSTEAD
      Reads the two artefacts that already define the Section 2.5 cohorts and tabulates
      them directly. No re-derivation of the category flags happens here - the flags are
      taken as written by script 04, so this table cannot drift from the AAR analysis.

          Healthy Reference  <- baseline_reference.csv, n = 762
          Hypertension       <- clinical_aar_persample_NEW.csv, hypertension == 1, n = 66
          Autoimmune         <- clinical_aar_persample_NEW.csv, autoimmune   == 1, n = 22
          Cancer             <- clinical_aar_persample_NEW.csv, cancer       == 1, n = 22
          All Clinical Cases <- clinical_aar_persample_NEW.csv, n = 244

      The three clinical categories overlap (a patient may be in more than one) and do
      not partition the 244 - that is why `All Clinical Cases` is reported alongside them
      rather than as their sum.

    ON n = 762 vs n = 723
      762 is correct and is what the manuscript states. `baseline_reference.csv` holds the
      full untrimmed test_clean pool (n = 762); `baseline_params.json` records
      `n_baseline_pool = 762`. The Methods 4.11 top-5% trim is applied *only* when fitting
      the baseline regression line - 723 of the 762 residuals survive
      `abs_res <= quantile(abs_res, 0.95)` and set the slope/intercept (39 trimmed). The
      permutation test and the panel g/h/i background then run against all 762. So 723 is
      a fit-internal number, never a cohort n, and must not appear in this table.

      `baseline_reference.csv` carries only Age and y_pred, so `Biological Sex` for the
      healthy reference is recovered from `test_preds.csv` by removing the 244 clinical
      sample names. That recovery is checked, not assumed: it asserts the recovered pool
      is n = 762 and that its sorted Age vector matches `baseline_reference.csv` exactly,
      and aborts otherwise.

    INPUTS  (all shipped, all products of 04_clinical_aar_pipeline.py / 01)
      - results/revision/clinical_aar_persample_NEW.csv
      - results/revision/baseline_reference.csv
      - results/revision/baseline_params.json
      - results/revision/test_preds.csv

    OUTPUT
      - Supp_Table_4_cohort_summary.csv, in the working directory

      The published copy of that same table, refreshed from this code on 2026-09-09:
      - results/supplementary/tables/Supplementary Table 4.xlsx

      XLSX only, matching Supplementary Tables 1-3. A `.csv` sidecar shipped alongside it
      until 2026-09-10 and was removed as a duplicate: identical data, no reference
      anywhere, and no sibling table had one. The CSV this function writes to the working
      directory is unaffected - that is the script's own output, not a shipped artefact.

      A separate mirror exists outside the repository at
      ~/Desktop/T_TIME/supplementary/tables/ and is maintained by hand; it still carries
      both formats.
    """
    out = "Supp_Table_4_cohort_summary.csv"

    clin = pd.read_csv(_find("clinical_aar_persample_NEW.csv"), low_memory=False,
                       usecols=["sample name", "Biological Sex", "Age",
                                "hypertension", "autoimmune", "cancer"])
    base = pd.read_csv(_find("baseline_reference.csv"))
    params = json.loads(Path(_find("baseline_params.json")).read_text())
    test = pd.read_csv(_find("test_preds.csv"), low_memory=False)

    print(f"[*] clinical cohort: {len(clin)} patients")
    print(f"[*] baseline pool:   {len(base)} (baseline_params n_baseline_pool="
          f"{params['n_baseline_pool']})")

    # --- recover Biological Sex for the healthy reference -------------------
    # baseline_reference.csv is Age + y_pred only. Rebuild the same pool out of
    # test_preds.csv by removing the clinical patients, then verify it really is
    # the same 762 rows before using its sex column.
    clin_ids = set(clin["sample name"].map(_t4_norm_id))
    healthy = test[~test["sample name"].map(_t4_norm_id).isin(clin_ids)].copy()

    n_in_test = len(test) - len(healthy)
    print(f"[*] of {len(clin)} clinical patients, {n_in_test} fall in the "
          f"{len(test)}-sample internal test split -> healthy reference "
          f"{len(test)} - {n_in_test} = {len(healthy)}")

    if len(healthy) != len(base):
        sys.exit(f"[!] recovered healthy pool is n={len(healthy)}, but "
                 f"baseline_reference.csv is n={len(base)}. Refusing to guess.")
    a_rec = np.sort(pd.to_numeric(healthy["Age"], errors="coerce").to_numpy(float))
    a_ref = np.sort(pd.to_numeric(base["Age"], errors="coerce").to_numpy(float))
    if not np.allclose(a_rec, a_ref, equal_nan=True):
        sys.exit("[!] recovered healthy pool has a different Age distribution than "
                 "baseline_reference.csv. Refusing to guess.")
    print("[*] healthy-reference sex recovered from test_preds.csv "
          "(n and Age distribution both match baseline_reference.csv)")

    # --- assemble ----------------------------------------------------------
    groups = [
        _table4_summarize(healthy, "Healthy Reference"),
        _table4_summarize(clin[clin["hypertension"] == 1], "Hypertension"),
        _table4_summarize(clin[clin["autoimmune"] == 1], "Autoimmune"),
        _table4_summarize(clin[clin["cancer"] == 1], "Cancer"),
        _table4_summarize(clin, "All Clinical Cases"),
    ]
    table = pd.concat(groups, axis=1)

    bad = {c: int(table.loc["N", c]) for c in table.columns
           if int(table.loc["N", c]) != _T4_EXPECTED_N[c]}
    if bad:
        sys.exit(f"[!] group sizes disagree with results/numbers.json: {bad}. "
                 f"Expected {_T4_EXPECTED_N}. Re-run 04_clinical_aar_pipeline.py, or "
                 f"update _T4_EXPECTED_N here if the category definitions changed on "
                 f"purpose - and update Section 2.5 with them.")

    print("\n=== Supplementary Table 4: clinical and healthy reference cohorts ===")
    print(table.to_string())
    print("\n    The three clinical categories overlap and do not partition the 244.")
    table.to_csv(out)
    print(f"\n[*] wrote {out}")
    print("[*] NOTE: this IS the current 'Supplementary Table 4'. The original-submission "
          "version (n=762/16/14/6) was retired to "
          "~/Downloads/ttime_archive/docs_internal/archive/"
          "supplementary_table4_original_submission.xlsx. If anything here "
          "changes, refresh results/supplementary/tables/ and the Desktop mirror, and "
          "re-embed the XLSX in the manuscript.")
    return out


# ============================================================================
# Registry + CLI
# ============================================================================

ITEMS = {
    "suppfig1": (generate_suppfig1, "Supplementary Figure S1 - fairness / lifespan stability"),
    "table1":   (generate_table1,   "Supplementary Table 1  - cohort demographics"),
    "table2":   (generate_table2,   "Supplementary Table 2  - 5-fold cross-validation"),
    "table3":   (generate_table3,   "Supplementary Table 3  - feature ablation, bootstrap CIs"),
    "suppfig2": (generate_suppfig2, "Supplementary Figure S2 - Wasserstein significance, log count"),
    "suppfig3": (generate_suppfig3, "Supplementary Figure S3 - input-layer feature weights"),
    "suppfig4": (generate_suppfig4, "Supplementary Figure S4 - ablation MAE bar chart"),
    "table4":   (generate_table4,   "Supplementary Table 4  - clinical vs healthy reference"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate the manuscript supplementary figures and tables.",
        epilog="With no arguments, every item is generated in the order listed by --list.")
    ap.add_argument("items", nargs="*", metavar="ITEM",
                    help="one or more of: " + ", ".join(ITEMS))
    ap.add_argument("--list", action="store_true", help="list the items and exit")
    args = ap.parse_args(argv)

    if args.list:
        for k, (_, desc) in ITEMS.items():
            print(f"  {k:9s} {desc}")
        return 0

    wanted = args.items or list(ITEMS)
    unknown = [i for i in wanted if i not in ITEMS]
    if unknown:
        ap.error(f"unknown item(s): {', '.join(unknown)}. Known: {', '.join(ITEMS)}")

    written = []
    for name in wanted:
        fn, desc = ITEMS[name]
        print(f"\n{'=' * 78}\n[{name}] {desc}\n{'=' * 78}")
        written.append(fn())

    print(f"\n{'=' * 78}")
    print(f"[*] {len(written)} artefact(s) written to {Path('.').resolve()}:")
    for w in written:
        print(f"      {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
