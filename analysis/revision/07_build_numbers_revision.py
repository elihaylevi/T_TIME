"""
07_build_numbers_revision.py

Assembles results/numbers.json from artefacts that ALREADY EXIST on
disk. It trains nothing, loads no model, and touches no raw repertoire: every
value is either read verbatim from a shipped CSV/JSON, or computed from one
with standard statistics (MAE/R2/Welch/permutation).

The superseded hand-transcribed file is archived, not deleted. The two files
describe two different lineages and are meant to coexist until one is chosen:

  archived copy          <- superseded lineage (hand-transcribed; its scripts are
                            archived OUTSIDE this repo - see
                            ~/Downloads/ttime_archive/docs_internal/RECONCILIATION_REPORT.md 4.4)
  results/numbers.json   <- analysis/revision/01-04 lineage (this script)

Canonical choices, per the 2026-09-08 decision:
  - external_validation comes from revision/02 (zero-shot MAE ~9.99,
    R2 ~ +0.13), superseding the deprecated 13.521 / -0.324.
  - antigen_age_bias comes from species_age_bias_spectrum_verified.csv
    (9 species, T1D included, Cancer/Self and TB old-skewed), superseding
    results/tables/antigen_age_bias.csv.
  - cmv_seropositivity IS included (2026-09-08): the serostatus source
    data/emerson_metadata_expanded.csv was located, the sample-name join is
    exact (100%), and the contrast is reproducible. It is computed by
    analysis/revision/08_cmv_seropositivity.py, which must run first.

Fields whose source is not shipped are emitted as null with a "source" string
that names exactly what must be re-run. Nothing is guessed.

Usage:  python analysis/revision/07_build_numbers_revision.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
# C16 (2026-09-08): REV and OUT follow TTIME_REVISION_OUT so a verification run reads the
# FRESH artefacts and writes beside them, instead of silently reading the shipped ones and
# overwriting the repository copy. ANN and TAB stay repo-anchored - they are shipped inputs.
REV = Path(os.environ.get("TTIME_REVISION_OUT", REPO / "results" / "revision"))
ANN = REPO / "results" / "annotation_reproduction"
TAB = REPO / "results" / "tables"
# Stage 4 (2026-09-08): this file IS results/numbers.json now. The hand-transcribed
# predecessor is archived OUTSIDE this repo, at
# ~/Downloads/ttime_archive/docs_internal/archive/numbers_original_submission.json.
OUT = (Path(os.environ["TTIME_REVISION_OUT"]) / "numbers.json"
       if "TTIME_REVISION_OUT" in os.environ
       else REPO / "results" / "numbers.json")

R4 = lambda x: round(float(x), 4)
R3 = lambda x: round(float(x), 3)


def need(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"required source missing: {p}")
    return p


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    denom = ((y - y.mean()) ** 2).sum()
    if len(y) < 2 or denom == 0:          # undefined, e.g. a single-sample group
        return None
    return float(1 - ((y - p) ** 2).sum() / denom)


def welch_gap(df, value_col, sex_col):
    """male-minus-female difference in `value_col`, Welch t-test."""
    s = df[sex_col].astype(str).str.strip().str.capitalize()
    m = df.loc[s == "Male", value_col].astype(float).values
    f = df.loc[s == "Female", value_col].astype(float).values
    t = stats.ttest_ind(m, f, equal_var=False)
    return dict(gap_yr=R4(np.mean(m) - np.mean(f)), welch_p=R4(t.pvalue),
                n_male=int(len(m)), n_female=int(len(f)))


# ---------------------------------------------------------------- 1. internal
def internal_holdout():
    f = need(REV / "test_preds.csv")
    d = pd.read_csv(f)
    y, p = d["Age"].values, d["y_pred"].values
    sex = d["Biological Sex"].astype(str).str.strip().str.capitalize()
    by_sex = {}
    for s in ("Female", "Male"):
        g = d[sex == s]
        v = r2(g["Age"], g["y_pred"])
        by_sex[s] = dict(n=int(len(g)), mae=R3(mae(g["Age"], g["y_pred"])),
                         r2=None if v is None else R3(v))
    unknown = d[~sex.isin(["Female", "Male"])]
    return dict(
        source="results/revision/test_preds.csv (analysis/revision/01_train_primary_model.py, model M1)",
        n_test=int(len(d)), MAE=R3(mae(y, p)), RMSE=R3(rmse(y, p)), R2=R3(r2(y, p)),
        by_sex=by_sex,
        n_sex_unknown=int(len(unknown)),
        sex_unknown_note=(None if len(unknown) == 0 else
                          "excluded from by_sex: %s (Biological Sex is blank in the source CSV)"
                          % ", ".join(map(str, unknown["sample name"].tolist()))))


# ---------------------------------------------------------------- 2. external
def external_validation():
    fz = need(REV / "emerson_zeroshot_preds.csv")
    fr = need(REV / "emerson_recalibrated_persample.csv")
    z = pd.read_csv(fz).dropna(subset=["Age", "y_pred_zeroshot"])
    r = pd.read_csv(fr)
    yz, pz = z["Age"].values, z["y_pred_zeroshot"].values
    yr, pr = r["Age"].values, r["y_pred_recal"].values

    # y_recal is an exact affine image of y_pred_zeroshot -> a, b are recoverable.
    a, b = np.polyfit(r["y_pred_zeroshot"].values, pr, 1)
    resid = float(np.abs(pr - (a * r["y_pred_zeroshot"].values + b)).max())
    slope, intercept = np.polyfit(pz, yz, 1)

    return dict(
        source=("results/revision/emerson_zeroshot_preds.csv + "
                "emerson_recalibrated_persample.csv (analysis/revision/02, model M3)"),
        cohort="Emerson 2017",
        n=int(len(z)),
        age_range=[int(yz.min()), int(yz.max())],
        zero_shot_frozen=dict(MAE=R3(mae(yz, pz)), RMSE=R3(rmse(yz, pz)),
                              R2=R3(r2(yz, pz)), Pearson=R3(stats.pearsonr(yz, pz)[0])),
        calibration_line=dict(
            slope=R3(slope), intercept=R3(intercept),
            source="OLS of true Age on zero-shot prediction over all %d samples (diagnostic)" % len(z)),
        affine_recalibration=dict(
            eval_n=int(len(r)),
            calibration_n=int(len(z) - len(r)),
            calibration_n_note=("INFERRED as n_zeroshot - n_eval; the calibration slice "
                                "itself is not shipped, only the disjoint evaluation slice"),
            a=R4(a), b=R4(b),
            ab_note="recovered by exact affine fit of y_pred_recal on y_pred_zeroshot "
                    "(max residual %.1e); a/b are not stored on disk" % resid,
            MAE=R3(mae(yr, pr)), R2=R3(r2(yr, pr)),
            Pearson=R3(stats.pearsonr(yr, pr)[0])),
        note=("Supersedes the deprecated zero-shot figures (MAE 13.521, R2 -0.324) that "
              "the archive carries from emerson_zeroshot.py, a script archived outside this repository."))


# ---------------------------------------------------------------- 3. sex
def sex_difference():
    g1 = pd.read_csv(need(REV / "gap1_primary_full.csv"))
    g2 = pd.read_csv(need(REV / "gap2_external_recalibrated.csv"))
    g3 = pd.read_csv(need(REV / "gap3_covid_broad.csv"))
    g3b = pd.read_csv(need(REV / "gap3b_healthy_broad.csv"))

    # zero-shot gap on the evaluation slice (residual on the uncalibrated scale)
    ez = pd.read_csv(need(REV / "emerson_recalibrated_persample.csv")).copy()
    ez["residual_zeroshot"] = ez["y_pred_zeroshot"] - ez["Age"]

    # diagnostics on the primary robust set
    s = g1["Biological Sex"].astype(str).str.capitalize()
    m, f = g1.loc[s == "Male", "centered"].values, g1.loc[s == "Female", "centered"].values
    ks = stats.ks_2samp(m, f)
    lev = stats.levene(m, f)
    X = np.column_stack([np.ones(len(g1)), g1["Age"].values,
                         (s == "Male").astype(float).values,
                         g1["Age"].values * (s == "Male").astype(float).values])
    beta, *_ = np.linalg.lstsq(X, g1["residual"].values, rcond=None)
    res = g1["residual"].values - X @ beta
    dof = len(g1) - X.shape[1]
    cov = (res @ res) / dof * np.linalg.inv(X.T @ X)
    p_int = float(2 * stats.t.sf(abs(beta[3] / np.sqrt(cov[3, 3])), dof))

    return dict(
        source="results/revision/gap{1,2,3,3b}*.csv (analysis/revision/03_sex_gap_analysis.py)",
        convention="male minus female, years; all sets already have the top-10% |residual| filter applied",
        primary_all=dict(**welch_gap(g1, "centered", "Biological Sex"),
                         n=int(len(g1)), source="gap1_primary_full.csv (centered residual)"),
        primary_healthy=dict(**welch_gap(g3b, "centered", "Biological Sex"),
                             n=int(len(g3b)), source="gap3b_healthy_broad.csv (centered residual)"),
        primary_covid=dict(**welch_gap(g3, "centered", "Biological Sex"),
                           n=int(len(g3)), source="gap3_covid_broad.csv (centered residual)"),
        external_recalibrated=dict(**welch_gap(g2, "residual_recal", "Sex"),
                                   n=int(len(g2)),
                                   source="gap2_external_recalibrated.csv (recalibrated residual)"),
        external_zeroshot=dict(**welch_gap(ez, "residual_zeroshot", "Sex"),
                               n=int(len(ez)),
                               source=("emerson_recalibrated_persample.csv, "
                                       "y_pred_zeroshot - Age on the %d-sample evaluation "
                                       "slice only (not all %s)" % (len(ez), "zero-shot samples"))),
        diagnostics=dict(KS_p=R4(ks.pvalue), Levene_p=R4(lev.pvalue),
                         age_x_sex_interaction_p=R4(p_int),
                         source="computed on gap1_primary_full.csv"),
        note=("The deprecated lineage reported combined_offset 0.86 and an external "
              "ZERO-SHOT gap of -0.1; this lineage reports the primary set as a whole "
              "and an external gap on the recalibrated scale. The quantities differ - "
              "do not compare the two files field by field."))


# ---------------------------------------------------------------- 4. clinical
def age_acceleration():
    fc = need(REV / "clinical_aar_persample_NEW.csv")
    fb = need(REV / "baseline_reference.csv")
    fp = need(REV / "baseline_params.json")
    clin = pd.read_csv(fc, low_memory=False,
                       usecols=["sample name", "Biological Sex", "Age", "y_pred_new",
                                "AAR_new", "hypertension", "autoimmune", "cancer"])
    base = pd.read_csv(fb)
    par = json.loads(fp.read_text(encoding="utf-8"))

    # identical to 04_clinical_aar_pipeline.py: AAR = y_pred - (slope*Age + intercept)
    pool = base["y_pred"].values - (par["slope"] * base["Age"].values + par["intercept"])

    # perm_test / bh replicated verbatim from 04 (single RNG(42), same call order)
    RNG = np.random.default_rng(42)

    def perm_test(cond, pool_, nperm=5000):
        obs = cond.mean() - pool_.mean()
        comb = np.concatenate([cond, pool_])
        nc = len(cond)
        null = np.empty(nperm)
        for i in range(nperm):
            idx = RNG.permutation(len(comb))
            null[i] = comb[idx[:nc]].mean() - comb[idx[nc:]].mean()
        return float(obs), float((np.sum(np.abs(null) >= abs(obs)) + 1) / (nperm + 1))

    def bh(pv):
        p = np.array(pv); n = len(p); order = np.argsort(p)
        adj = np.empty(n); prev = 1.0
        for i in range(n - 1, -1, -1):
            prev = min(prev, p[order[i]] * n / (i + 1))
            adj[order[i]] = prev
        return adj

    out, pvals, keys = {}, [], []
    for name in ["hypertension", "autoimmune", "cancer"]:      # order matters: shared RNG
        mask = clin[name] == 1
        obs, p = perm_test(clin.loc[mask, "AAR_new"].values, pool, nperm=5000)
        out[name] = dict(n=int(mask.sum()), AAR=R3(obs), perm_p=R4(p))
        pvals.append(p); keys.append(name)
    for k, a in zip(keys, bh(pvals)):
        out[k]["perm_p_BH"] = R4(a)

    out.update(
        source=("results/revision/{clinical_aar_persample_NEW.csv, baseline_reference.csv, "
                "baseline_params.json} (analysis/revision/04_clinical_aar_pipeline.py)"),
        design=("AAR = predicted age - robust baseline line fitted on the leakage-free "
                "test_clean pool (n=%d, slope=%.4f, intercept=%.4f); 5,000-permutation "
                "null with RNG(42); Benjamini-Hochberg across the three categories"
                % (par["n_baseline_pool"], par["slope"], par["intercept"])),
        n_clinical_cohort=int(len(clin)),
        age_sex_adjusted=None,
        age_sex_adjusted_source=("requires re-run: script 04 does not compute an age/sex-adjusted "
                                 "AAR. The -3.376 / -0.464 / 0.103 values in numbers.json come from "
                                 "clinical_aar.py - archived outside this repository - a different model and a "
                                 "different category definition"),
        category_definition_note=("word-list + boolean-flag definition shared with script 05; "
                                  "yields n~66/22/22. numbers.json's 17/13/8 uses the older "
                                  "boolean-flag-only definition - see 04's docstring, step 6"),
        cmv_seropositivity_note=("moved to its own top-level key; the serostatus source "
                                 "data/emerson_metadata_expanded.csv was located on 2026-09-08 "
                                 "and the contrast is now reproducible - see "
                                 "analysis/revision/08_cmv_seropositivity.py"))
    return out


# ------------------------------------------------------------------- 4b. cmv
def cmv_seropositivity():
    """Read the output of 08_cmv_seropositivity.py; null if it has not run."""
    f = REV / "cmv_seropositivity.json"
    if not f.exists():
        return dict(value=None,
                    source=("requires re-run: analysis/revision/08_cmv_seropositivity.py "
                            "-> results/revision/cmv_seropositivity.json. Inputs "
                            "(data/emerson_metadata_expanded.csv and script 02's "
                            "predictions) ARE shipped, so this is cheap to regenerate."))
    d = json.loads(f.read_text(encoding="utf-8"))
    p = d["primary"]
    return dict(
        source=d["source"],
        generator=d["generator"],
        cohort=d["cohort"],
        specification=d["specification"],
        n=p["n"], n_pos=p["n_pos"], n_neg=p["n_neg"],
        age_sex_adjusted_yr=p["age_sex_adjusted_yr"],
        age_sex_adjusted_p=p["age_sex_adjusted_p"],
        unadjusted_gap_yr=p["unadjusted_gap_yr"],
        unadjusted_welch_p=p["unadjusted_welch_p"],
        zero_shot_full_set=d["zero_shot_full_set"],
        caveat=d["caveat"],
        note_vs_published=d["note_vs_published"])


# ---------------------------------------------------------------- 5. antigen
def antigen_age_bias():
    f = need(ANN / "species_age_bias_spectrum_verified.csv")
    d = pd.read_csv(f).sort_values("median_signed_wasserstein")
    return dict(
        source="results/annotation_reproduction/species_age_bias_spectrum_verified.csv",
        reference="per annotated clonotype row of the Figure 4 merged table (VDJdb + McPAS + TRAIT)",
        median_signed_wasserstein={r.species: R4(r.median_signed_wasserstein)
                                   for r in d.itertuples()},
        n_by_species={r.species: int(r.n) for r in d.itertuples()},
        note=("Supersedes results/tables/antigen_age_bias.csv (7 categories, per unique "
              "clonotype, all negative). This lineage has 9 species, includes T1D, and "
              "puts Cancer/Self and TB on the old-skewed side."))


# ---------------------------------------------------------------- 6. carried
def cohort_qc():
    f = need(TAB / "qc_per_cohort.csv")
    d = pd.read_csv(f).set_index("cohort")
    a = d.loc["All"]
    return dict(source="results/tables/qc_per_cohort.csv",
                raw_repertoires=int(a["All"]), passed_qc_ge_200k_templates=int(a["SUCCESS"]),
                excluded_low_depth=int(a["EXCLUDED"]), malformed=int(a["ERROR"]),
                per_cohort={c: {k: int(v) for k, v in d.loc[c].items()}
                            for c in d.index if c != "All"},
                train=None, test=None,
                train_test_source=("requires re-run: split_train/test counts come from "
                                   "work/outputs/stage01b_split_report.json (not shipped). "
                                   "results/tables/data_provenance.csv records 2453/818."))


def hla_restriction():
    f = need(TAB / "hla_restriction.csv")
    d = pd.read_csv(f)
    return dict(source="results/tables/hla_restriction.csv",
                median_signed_wasserstein_by_restriction={r.hla: R3(r.median) for r in d.itertuples()},
                n_by_restriction={r.hla: int(r.count) for r in d.itertuples()},
                note="Age association is largely consistent across common HLA-A restrictions.")


def baselines():
    f = need(ANN / "mlp_vs_baselines_paired_bootstrap.json")
    b = json.loads(f.read_text(encoding="utf-8"))
    pm = b["per_model"]
    return dict(
        source="results/annotation_reproduction/mlp_vs_baselines_paired_bootstrap.json",
        bootstrap=b["bootstrap"],
        neural_network=dict(**{k: v for k, v in pm["MLP (fig2new/test_preds.csv)"].items()},
                            source="MLP (fig2new/test_preds.csv) == results/revision/test_preds.csv"),
        elasticnet_features=pm["elasticnet_features"],
        ridge_features=pm["ridge_features"],
        paired_vs_mlp=b["paired"],
        diversity_ridge=None,
        diversity_gbm=None,
        diversity_source=("requires re-run: pipeline/baselines.py needs the raw repertoire TSVs at "
                          "config.TRAIN_DIR / config.TEST_DIR to compute Shannon / inverse-Simpson / "
                          "clonality / Gini / richness. Not shipped."))


def not_reproducible():
    return {
        "age_association_scoring": dict(
            value=None,
            source=("requires re-run: pipeline/stage02_wasserstein.py -> "
                    "work/outputs/stage02_report_{train,combined}.json. Needs the downsampled "
                    "repertoire directories; not shipped.")),
        "vj_usage": dict(
            value=None,
            source=("requires re-run: pipeline/vj_usage.py -> work/outputs/vj_usage.json. "
                    "Needs the raw repertoire TSVs (V/J calls); not shipped.")),
        "physicochemical_classifier": dict(
            value=None,
            source=("requires re-run: pipeline/mait_classifier.py -> "
                    "work/outputs/mait_classifier.json. Its only input, "
                    "data/significant_tcrs_signed_wasserstein.csv, IS shipped, so this one is "
                    "cheap to regenerate - it just was not run in this pass.")),
    }


def main():
    doc = {
        "_description": ("Headline results rebuilt from artefacts already on disk. "
                         "Every key carries its own 'source'. No model was trained and no "
                         "raw repertoire was read to produce this file."),
        "_generator": "analysis/revision/07_build_numbers_revision.py",
        "_lineage": "analysis/revision/01-04 (models M1/M3, leakage-safe clinical AAR)",
        "_supersedes": ("results/numbers.json for external_validation, antigen_age_bias, "
                        "age_acceleration and sex_difference. That file is left in place and "
                        "archived outside this repository, at ~/Downloads/ttime_archive/"
                        "docs_internal/archive/numbers_original_submission.json."),
        "_training_sensitivity": (
            "The values below come from the shipped GPU run of scripts 01/02/04 and are "
            "NOT replaced by any later re-run. Torch training is not bitwise reproducible "
            "across backends: a CPU re-run on 2026-09-08 moved primary MAE 7.475 -> 7.534, "
            "the primary sex gap 1.374 -> 1.530, the CMV effect +2.762 -> +3.576, and the "
            "hypertension AAR -3.249 -> -2.393 with its BH-adjusted P going 0.017 -> 0.091. "
            "Treat every trained-model figure here as one specific run, not a value a "
            "re-run will land on. Discussed as a limitation in the manuscript, Methods 4.8; "
            "measured in the archived internal changelog, addendum G4 "
            "(~/Downloads/ttime_archive/docs_internal/CHANGELOG_RESTRUCTURE.md)."),
        "internal_holdout": internal_holdout(),
        "cohort_qc": cohort_qc(),
        "external_validation": external_validation(),
        "baselines": baselines(),
        "hla_restriction": hla_restriction(),
        "antigen_age_bias": antigen_age_bias(),
        "sex_difference": sex_difference(),
        "age_acceleration": age_acceleration(),
        "cmv_seropositivity": cmv_seropositivity(),
    }
    doc.update(not_reproducible())

    OUT.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    try:
        _shown = OUT.relative_to(REPO)
    except ValueError:      # redirected outside the repo by TTIME_REVISION_OUT
        _shown = OUT
    print(f"[*] wrote {_shown}  ({OUT.stat().st_size:,} bytes)")
    print("[*] superseded lineage: archived outside this repo, at "
          "~/Downloads/ttime_archive/docs_internal/archive/numbers_original_submission.json")


if __name__ == "__main__":
    main()
