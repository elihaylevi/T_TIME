"""
03_sex_gap_analysis.py

Sex-gap analysis on the outputs of scripts 01 and 02: Primary (all),
External (recalibrated), COVID vs Healthy split (broad definition:
non-Vo' samples default to COVID, Vo' samples use the Infected flag).

REQUIRES (run scripts 01 and 02 first):
  - deepmlp_eval_final/outputs/test_preds.csv   (from 01)
  - outputs/emerson_recalibrated_persample.csv  (from 02)
  - TCR_shared_ELIHAY_full.csv                  (uploaded separately)

Outputs (for the final figure, script 06):
  - gap1_primary_full.csv       (panel d)
  - gap2_external_recalibrated.csv (panel f)
  - gap3_covid_broad.csv        (panel e, COVID bar)
  - gap3b_healthy_broad.csv     (panel e, Healthy bar)

NOTE: this script does NOT exclude the 244 clinical-condition patients
from the Primary cohort before computing the sex gap. That means some
hypertension/autoimmune/cancer patients are included within the 818
Primary samples used here. This is a design choice inherited from the
manuscript's own approach (no evidence they excluded clinical patients
from the sex-gap panels either) - flagged for awareness, not changed
silently. If you want the "clean" cohort for this analysis too, filter
test_preds.csv against clinical_aar_persample_NEW.csv's sample names
before running this script.
"""
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats


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



def analyze(name, M, F, label_extra=""):
    if len(M) < 2 or len(F) < 2:
        print(f"\n{'=' * 55}\n{name}{label_extra}\n{'=' * 55}")
        print(f"n_M={len(M)}  n_F={len(F)}  -- too few for a meaningful test")
        return None, None
    gap = M.mean() - F.mean()
    p = stats.ttest_ind(M, F, equal_var=False).pvalue
    print(f"\n{'=' * 55}\n{name}{label_extra}\n{'=' * 55}")
    print(f"n_M={len(M)}  n_F={len(F)}")
    print(f"Gap (M - F) = {gap:.3f} years   Welch P = {p:.5f}")
    return gap, p


def norm_id(s):
    """From stage01b_split.py:11-12 - links test_preds sample names to ELIHAY IDs."""
    return str(s).lower().replace('copy of ', '').strip().replace('_tcrb', '')


def main():
    # ============================================================
    # Part 1: PRIMARY (full cohort)
    # ============================================================
    primary = pd.read_csv(_find("test_preds.csv"))
    primary = primary[primary['Biological Sex'].isin(['Male', 'Female'])].copy()
    primary['residual'] = -primary['residual']  # flip: Age-y_pred -> y_pred-Age

    primary['abs_res'] = primary['residual'].abs()
    thresh = primary['abs_res'].quantile(0.90)
    primary_robust = primary[primary['abs_res'] <= thresh].copy()
    primary_robust['centered'] = primary_robust['residual'] - primary_robust['residual'].mean()

    M1 = primary_robust[primary_robust['Biological Sex'] == 'Male']['centered']
    F1 = primary_robust[primary_robust['Biological Sex'] == 'Female']['centered']
    gap1, p1 = analyze("1. PRIMARY (full cohort)", M1, F1,
                        f"  [n_total={len(primary_robust)}/{len(primary)} after 10% filter]")
    print("(canonical 2026-09-08: Gap=1.374 yr, P=0.0103, n=735)")
    primary_robust.to_csv("gap1_primary_full.csv", index=False)

    # ============================================================
    # Part 2: EXTERNAL (Emerson, recalibrated, top-10% filter)
    # ============================================================
    ext = pd.read_csv(_find("emerson_recalibrated_persample.csv"))
    ext = ext[ext['Sex'].isin(['Male', 'Female'])].copy()

    ext['abs_res'] = ext['residual_recal'].abs()
    thresh_e = ext['abs_res'].quantile(0.90)
    ext_robust = ext[ext['abs_res'] <= thresh_e].copy()

    M2 = ext_robust[ext_robust['Sex'] == 'Male']['residual_recal']
    F2 = ext_robust[ext_robust['Sex'] == 'Female']['residual_recal']
    gap2, p2 = analyze("2. EXTERNAL (recalibrated, robust 90%)", M2, F2,
                        f"  [n_total={len(ext_robust)}/{len(ext)} after 10% filter]")
    print("(canonical 2026-09-08: Gap=0.760 yr, P=0.340, n=356 - not significant)")
    ext_robust.to_csv("gap2_external_recalibrated.csv", index=False)

    # ============================================================
    # Part 3: COVID vs Healthy - broad definition
    # rule: not found in ELIHAY (= not Vo') -> COVID by default
    #       found in ELIHAY (= Vo')         -> Infected flag decides
    # ============================================================
    elihay = pd.read_csv(_find("TCR_shared_ELIHAY_full.csv"))
    elihay['norm_id'] = elihay['Sample_ID'].apply(norm_id)

    primary_robust['norm_id_full'] = primary_robust['sample name'].apply(norm_id)
    primary_robust['norm_id_trimmed'] = primary_robust['norm_id_full'].str[4:]

    merged = primary_robust.merge(elihay[['norm_id', 'Infected']],
                                   left_on='norm_id_trimmed', right_on='norm_id', how='left')

    coverage = merged['norm_id'].notna().sum()
    print(f"\n[*] ELIHAY (Vo') match coverage: {coverage}/{len(merged)} "
          f"({coverage / len(merged) * 100:.1f}%) - the rest are non-Vo' cohorts, classified COVID by default")

    merged['cohort_group'] = np.where(
        merged['norm_id'].isna(), 'COVID',
        np.where(merged['Infected'] == 1.0, 'COVID', 'Healthy'))

    print(f"\n[*] Broad cohort split (n={len(merged)}):")
    print(merged['cohort_group'].value_counts())

    covid_broad = merged[merged['cohort_group'] == 'COVID'].copy()
    healthy_broad = merged[merged['cohort_group'] == 'Healthy'].copy()

    Mc = covid_broad[covid_broad['Biological Sex'] == 'Male']['centered']
    Fc = covid_broad[covid_broad['Biological Sex'] == 'Female']['centered']
    gap3, p3 = analyze("3. COVID (broad definition)", Mc, Fc, f"  [n={len(covid_broad)}]")

    Mh = healthy_broad[healthy_broad['Biological Sex'] == 'Male']['centered']
    Fh = healthy_broad[healthy_broad['Biological Sex'] == 'Female']['centered']
    gap3h, p3h = analyze("3b. Healthy (broad definition)", Mh, Fh, f"  [n={len(healthy_broad)}]")

    covid_broad.to_csv("gap3_covid_broad.csv", index=False)
    healthy_broad.to_csv("gap3b_healthy_broad.csv", index=False)

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n\n{'#' * 55}\nSUMMARY\n{'#' * 55}")
    print(f"1. Primary:            Gap={gap1:.3f}  P={p1:.5f}   (canonical: 1.374 / 0.0103)")
    print(f"2. External (recalib): Gap={gap2:.3f}  P={p2:.5f}   (canonical: 0.760 / 0.340)")
    if gap3 is not None:
        print(f"3. COVID (broad):       Gap={gap3:.3f}  P={p3:.5f}   (ms panel e: -0.14 to +1.0)")
    if gap3h is not None:
        print(f"3b. Healthy (broad):    Gap={gap3h:.3f}  P={p3h:.5f}   (ms: 1.77, P=0.0031)")


if __name__ == "__main__":
    main()
