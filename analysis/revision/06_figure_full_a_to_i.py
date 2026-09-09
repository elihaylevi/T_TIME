"""
06_figure_full_a_to_i.py

Final combined figure, panels a-i, matching the manuscript's Figure 2
layout. STANDALONE: reads every input from disk, never from notebook
session memory. This is the fix for the "stale variable" bug class we
hit repeatedly (panels drawn from an old model run because a live
variable wasn't refreshed) - run this any time, in any session, as
long as scripts 01-04 have been run at least once and their outputs
exist on disk.

REQUIRES (run scripts 01, 02, 03, 04 first, in that order):
  - deepmlp_eval_final/outputs/test_preds.csv     (01, panel a)
  - outputs/emerson_zeroshot_preds.csv            (02, panel b)
  - outputs/emerson_recalibrated_persample.csv    (02, panel c)
  - gap1_primary_full.csv                         (03, panel d)
  - gap3_covid_broad.csv, gap3b_healthy_broad.csv (03, panel e)
  - gap2_external_recalibrated.csv                (03, panel f)
  - clinical_aar_persample_NEW.csv                (04, panels g/h/i)
  - baseline_reference.csv, baseline_params.json  (04, panels g/h/i)

Output: Figure2_full_reproduced_a-i.pdf / .png
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr, ttest_ind

plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'sans-serif',
                      'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                      'font.size': 7, 'axes.linewidth': 0.5,
                      'xtick.major.width': 0.5, 'ytick.major.width': 0.5})

C_MAIN = '#E69F00'
C_RECAL = '#009E73'
C_MALE = '#D55E00'
C_FEMALE = '#0072B2'
CLINICAL_COLORS = {'hypertension': '#4C72B0', 'autoimmune': '#C44E52', 'cancer': '#DD8452'}
CLINICAL_TITLES = {'hypertension': 'Hypertension', 'autoimmune': 'Autoimmune', 'cancer': 'Cancer'}


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


def panel_letter(ax, letter):
    ax.text(-0.18, 1.08, letter, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')


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
    # ------------------------------------------------------------
    # Load everything fresh from disk
    # ------------------------------------------------------------
    primary = pd.read_csv(_find("test_preds.csv"))
    zeroshot = pd.read_csv(_find("emerson_zeroshot_preds.csv"))
    recal = pd.read_csv(_find("emerson_recalibrated_persample.csv"))
    gap1 = pd.read_csv(_find("gap1_primary_full.csv"))
    gap2 = pd.read_csv(_find("gap2_external_recalibrated.csv"))
    gap3_covid = pd.read_csv(_find("gap3_covid_broad.csv"))
    gap3b_healthy = pd.read_csv(_find("gap3b_healthy_broad.csv"))
    clin_valid = pd.read_csv(_find("clinical_aar_persample_NEW.csv"), low_memory=False)
    baseline_ref = pd.read_csv(_find("baseline_reference.csv"))
    baseline_params = json.loads(Path(_find("baseline_params.json")).read_text())
    slope, intercept = baseline_params['slope'], baseline_params['intercept']

    # ------------------------------------------------------------
    # Panel a: Primary cohort (held-out) - Model M1
    # ------------------------------------------------------------
    y_true_a, y_pred_a = primary['Age'].values, primary['y_pred'].values
    mae_a = mean_absolute_error(y_true_a, y_pred_a)
    r2_a = r2_score(y_true_a, y_pred_a)

    # ------------------------------------------------------------
    # Panel b: External zero-shot (frozen) - Model M3
    # ------------------------------------------------------------
    y_true_b, y_pred_b = zeroshot['Age'].values, zeroshot['y_pred_zeroshot'].values
    mae_b = mean_absolute_error(y_true_b, y_pred_b)
    r2_b = r2_score(y_true_b, y_pred_b)
    r_b = pearsonr(y_true_b, y_pred_b)[0]

    # ------------------------------------------------------------
    # Panel c: External + recalibration
    # ------------------------------------------------------------
    y_true_c, y_pred_c = recal['Age'].values, recal['y_pred_recal'].values
    mae_c = mean_absolute_error(y_true_c, y_pred_c)
    r2_c = r2_score(y_true_c, y_pred_c)

    # ------------------------------------------------------------
    # Panel d: Sex offset (Primary Cohort)
    # ------------------------------------------------------------
    sex_col_d = 'Biological Sex' if 'Biological Sex' in gap1.columns else 'Sex'
    M_d = gap1[gap1[sex_col_d] == 'Male']['centered'].values
    F_d = gap1[gap1[sex_col_d] == 'Female']['centered'].values
    gap_d = M_d.mean() - F_d.mean()
    p_d = ttest_ind(M_d, F_d, equal_var=False).pvalue

    # ------------------------------------------------------------
    # Panel e: Convergence (primary) - Healthy vs COVID
    # ------------------------------------------------------------
    sex_col_e = 'Biological Sex' if 'Biological Sex' in gap3b_healthy.columns else 'Sex'
    Mh = gap3b_healthy[gap3b_healthy[sex_col_e] == 'Male']['centered']
    Fh = gap3b_healthy[gap3b_healthy[sex_col_e] == 'Female']['centered']
    Mc = gap3_covid[gap3_covid[sex_col_e] == 'Male']['centered']
    Fc = gap3_covid[gap3_covid[sex_col_e] == 'Female']['centered']
    means = [Fh.mean(), Fc.mean()]
    sems = [Fh.sem(), Fc.sem()]
    meansM = [Mh.mean(), Mc.mean()]
    semsM = [Mh.sem(), Mc.sem()]
    p_h = ttest_ind(Mh, Fh, equal_var=False).pvalue
    p_c = ttest_ind(Mc, Fc, equal_var=False).pvalue

    # ------------------------------------------------------------
    # Panel f: external dataset (sex)
    # ------------------------------------------------------------
    M_f = gap2[gap2['Sex'] == 'Male']['residual_recal']
    F_f = gap2[gap2['Sex'] == 'Female']['residual_recal']
    gap_f = M_f.mean() - F_f.mean()
    p_f = ttest_ind(M_f, F_f, equal_var=False).pvalue

    # ------------------------------------------------------------
    # Panels g/h/i: clinical categories - fresh AAR from disk,
    # NEVER from a live model/session variable.
    # ------------------------------------------------------------
    yte = baseline_ref['Age'].values
    p_test = baseline_ref['y_pred'].values
    aar_healthy_pool = p_test - (slope * yte + intercept)

    # ------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(10.5, 10.2), dpi=200)
    gs = gridspec.GridSpec(3, 3, hspace=0.6, wspace=0.4, left=0.06, right=0.98, top=0.96, bottom=0.05)
    lim = [0, 95]

    # --- a ---
    ax = fig.add_subplot(gs[0, 0]); panel_letter(ax, 'a')
    ax.scatter(y_true_a, y_pred_a, c=C_MAIN, alpha=0.6, s=8, edgecolors='none', rasterized=True)
    ax.plot(lim, lim, 'k--', lw=0.8)
    ax.set_title('Primary cohort (held-out)')
    ax.set_xlabel('Chronological age (yr)'); ax.set_ylabel('Predicted age (yr)')
    ax.text(0.05, 0.85, f"n = {len(y_true_a)}\nMAE = {mae_a:.2f} yr\n$R^2$ = {r2_a:.2f}",
            transform=ax.transAxes, fontsize=6, va='top')
    ax.set_xlim(lim); ax.set_ylim(lim)

    # --- b ---
    ax = fig.add_subplot(gs[0, 1]); panel_letter(ax, 'b')
    ax.scatter(y_true_b, y_pred_b, c=C_MAIN, alpha=0.6, s=8, edgecolors='none', rasterized=True)
    ax.plot(lim, lim, 'k--', lw=0.8)
    ax.set_title('External: zero-shot (frozen)')
    ax.set_xlabel('Chronological age (yr)'); ax.set_ylabel('Predicted age (yr)')
    ax.text(0.05, 0.85, f"n = {len(y_true_b)}\nMAE = {mae_b:.1f} yr\n$R^2$ = {r2_b:.2f}\nr = {r_b:.2f}",
            transform=ax.transAxes, fontsize=6, va='top')
    ax.set_xlim(lim); ax.set_ylim(lim)

    # --- c ---
    ax = fig.add_subplot(gs[0, 2]); panel_letter(ax, 'c')
    ax.scatter(y_true_c, y_pred_c, c=C_RECAL, alpha=0.6, s=8, edgecolors='none', rasterized=True)
    ax.plot(lim, lim, 'k--', lw=0.8)
    ax.set_title('External: + recalibration (2-param)')
    ax.set_xlabel('Chronological age (yr)'); ax.set_ylabel('Predicted age (yr)')
    ax.text(0.05, 0.85, f"n = {len(y_true_c)}\nMAE = {mae_c:.1f} yr\n$R^2$ = {r2_c:.2f}",
            transform=ax.transAxes, fontsize=6, va='top')
    ax.set_xlim(lim); ax.set_ylim(lim)

    # --- d ---
    ax = fig.add_subplot(gs[1, 0]); panel_letter(ax, 'd')
    parts = ax.violinplot([F_d, M_d], positions=[0, 1], vert=False, showextrema=False, widths=0.8)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(C_FEMALE if i == 0 else C_MALE); pc.set_alpha(0.5)
    bp = ax.boxplot([F_d, M_d], positions=[0, 1], vert=False, widths=0.15, patch_artist=True, showfliers=False)
    for patch, col in zip(bp['boxes'], [C_FEMALE, C_MALE]):
        patch.set_facecolor(col)
    plt.setp(bp['medians'], color='black')
    ax.set_yticks([0, 1]); ax.set_yticklabels(['F', 'M'])
    ax.axvline(0, color='grey', ls='--', lw=0.8, alpha=0.5)
    ax.set_title('Sex offset (Primary Cohort)')
    ax.set_xlabel('Residual (centered, yr)')
    p_str_d = "P<0.001" if p_d < 0.001 else f"P = {p_d:.3f}"
    ax.text(0.03, 0.95, f"{p_str_d}\nGap = {gap_d:.2f} yr", transform=ax.transAxes, fontsize=6, va='top')

    # --- e ---
    ax = fig.add_subplot(gs[1, 1]); panel_letter(ax, 'e')
    x = np.arange(2); width = 0.35
    ax.bar(x - width / 2, means, width, yerr=sems, capsize=3, color=C_FEMALE, label='Female')
    ax.bar(x + width / 2, meansM, width, yerr=semsM, capsize=3, color=C_MALE, label='Male')
    ax.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(['Healthy', 'COVID'])
    ax.set_title('Convergence (primary)')
    ax.set_ylabel('Mean residual (yr)')
    ax.legend(fontsize=6, frameon=False, loc='lower left')

    y_top = max(means[0] + sems[0], meansM[0] + semsM[0],
                means[1] + sems[1], meansM[1] + semsM[1])
    ax.set_ylim(top=y_top * 1.35)

    star_h = y_top * 1.08
    ax.text(0, star_h, "*" if p_h < 0.05 else "ns", ha='center', fontsize=7)
    ax.text(1, star_h, "*" if p_c < 0.05 else "ns", ha='center', fontsize=7)

    # --- f ---
    ax = fig.add_subplot(gs[1, 2]); panel_letter(ax, 'f')
    mean_f, sem_f = F_f.mean(), F_f.sem()
    mean_m, sem_m = M_f.mean(), M_f.sem()
    ax.bar([0], [mean_f], 0.5, yerr=[sem_f * 1.96], capsize=3, color=C_FEMALE)
    ax.bar([1], [mean_m], 0.5, yerr=[sem_m * 1.96], capsize=3, color=C_MALE)
    ax.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['F', 'M'])
    ax.set_title('external dataset (sex)')
    ax.set_ylabel('Mean residual (yr)')
    ax.text(0.5, 0.92, f"P = {p_f:.3f}\nGap = {gap_f:.2f} yr",
            transform=ax.transAxes, ha='center', va='top', fontsize=6)

    # --- g, h, i ---
    lim_min_clin = min(yte.min(), p_test.min(), clin_valid['Age'].min(), clin_valid['y_pred_new'].min()) - 5
    lim_max_clin = max(yte.max(), p_test.max(), clin_valid['Age'].max(), clin_valid['y_pred_new'].max()) + 5
    x_line = np.array([lim_min_clin, lim_max_clin])

    for idx, cat_key in enumerate(['hypertension', 'autoimmune', 'cancer']):
        letter = ['g', 'h', 'i'][idx]
        ax = fig.add_subplot(gs[2, idx]); panel_letter(ax, letter)

        ax.scatter(yte, p_test, c='silver', alpha=0.3, s=6, edgecolors='none', zorder=1, rasterized=True)
        ax.plot(x_line, slope * x_line + intercept, 'k--', lw=0.8, alpha=0.7, zorder=2)

        mask = clin_valid[cat_key] == 1
        n = int(mask.sum())
        sub_age = clin_valid.loc[mask, 'Age'].values
        sub_pred = clin_valid.loc[mask, 'y_pred_new'].values
        aar_vals = sub_pred - (slope * sub_age + intercept)  # recomputed fresh, not read from a stored column

        color = CLINICAL_COLORS[cat_key]
        ax.scatter(sub_age, sub_pred, c=color, s=22, edgecolors='black', linewidth=0.4, alpha=0.85, zorder=3)
        for a, p_ in zip(sub_age, sub_pred):
            ax.plot([a, a], [slope * a + intercept, p_], color=color, alpha=0.4, lw=1, zorder=2)

        ax.set_xlim(lim_min_clin, lim_max_clin); ax.set_ylim(lim_min_clin, lim_max_clin)
        ax.set_title(CLINICAL_TITLES[cat_key])
        ax.set_xlabel('Chronological age (yr)')
        if idx == 0:
            ax.set_ylabel('Predicted age (yr)')

        if n >= 2:
            obs, p_val = perm_test(aar_vals, aar_healthy_pool)
            p_str = "P < 0.001" if p_val < 0.001 else f"P = {p_val:.3f}"
            ax.text(0.05, 0.90, f"n = {n}\nAAR = {obs:+.1f} yr\nperm {p_str}",
                    transform=ax.transAxes, fontsize=6.5, va='top')

    for ax in fig.axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.savefig("Figure2_full_reproduced_a-i.pdf", dpi=200, transparent=True)
    plt.savefig("Figure2_full_reproduced_a-i.png", dpi=200)
    plt.show()

    print(f"\n{'=' * 65}\nREPRODUCED vs MANUSCRIPT\n{'=' * 65}")
    print(f"a  Primary:      n={len(y_true_a)}, MAE={mae_a:.2f}, R2={r2_a:.2f}   | ms: n=818, MAE=7.48, R2=0.78")
    print(f"b  Zero-shot:    n={len(y_true_b)}, MAE={mae_b:.1f}, R2={r2_b:.2f}   | ms: n=494, MAE=13.5, R2=-0.32")
    print(f"c  Recalibrated: n={len(y_true_c)}, MAE={mae_c:.1f}, R2={r2_c:.2f}   | ms: n=396, MAE=7.4,  R2=0.56")
    print(f"d  Sex Primary:  Gap={gap_d:.2f}, P={p_d:.4f}                       | ms: Gap=1.95, P=0.003")
    print(f"e  Healthy F/M:  {means[0]:+.2f}/{meansM[0]:+.2f}  P={p_h:.3f}")
    print(f"   COVID   F/M:  {means[1]:+.2f}/{meansM[1]:+.2f}  P={p_c:.3f}")
    print(f"f  External sex: Gap={gap_f:.2f}, P={p_f:.4f}                       | canonical: Gap=0.760, P=0.340")


if __name__ == "__main__":
    main()
