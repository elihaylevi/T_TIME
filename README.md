# T-Time

**A T-cell receptor repertoire clock for immune aging.**

T-Time predicts immunological age directly from T-cell receptor β-chain (TCRβ) repertoires. Age-associated clonotypes are identified with a signed Wasserstein score, combined with k-mer sequence composition, and used to train a neural regressor that produces a continuous, per-individual estimate of immunological age.

This repository contains the complete analysis pipeline, from raw repertoire files through model training, external validation, and the downstream immunological analyses.

---

## Repository layout

| Path | Contents |
|---|---|
| `pipeline/` | The core pipeline — QC/downsampling, Wasserstein scoring, training, baselines |
| `pipeline/slurm/` | Slurm job wrappers (`*.sbatch`) for the above |
| `analysis/` | Post-submission analyses, by round (see the three rows below) |
| `analysis/revision/` | Revision-round analyses: retraining, zero-shot recalibration, sex gap, clinical AAR |
| `analysis/annotation/` | Epitope harmonisation (13), age alignment (14), Figure 4 landscape |
| `results/supplementary/` | Manuscript Supplementary Figures S1–S4 and Tables 1–4 |
| `analysis/annotation/15_merged_overlap_extraction.py` | Rebuilds `data/external/merged_overlap_tcrs_wasserstein.csv` (lineage B, the MHC-carrying epitope join). Replaces the former `notebooks/tcr_correlations.ipynb`, archived outside this repository on 2026-09-09 |
| `data/` | Processed feature matrices and age-associated clonotype scores (see below) |
| `data/external/` | Inputs not produced by `pipeline/` — reference DBs, annotation intermediates |
| `metadata/` | Sample metadata (sample → age / sex / cohort) |
| `results/` | Headline numbers (`numbers.json`), figures, tables |
| `results/revision/` | Outputs of `analysis/revision/` (predictions, gap tables, baselines) |
| `results/sex_correction/` | Superseded sex-offset outputs, kept for provenance; the code that made them is archived outside this repository |
| `results/annotation_reproduction/` | Re-run checks of published numbers (baselines, paired bootstrap, sex-gap grid, species spectrum) |
| `docs/REPRODUCIBILITY.md` | **The only documentation shipped, and self-contained** — tiers, one-command run, known limits. The internal consolidation record (plan, changelog, reconciliation audit) and the two superseded artefacts were archived outside this repository on 2026-09-09; see its *Full history* section |
| `run_all.sh` | One-command Tier A reproduction |

The `data/` directory ships the processed inputs needed to reproduce the model and all downstream analyses without re-running the heavy raw-repertoire stages:

- `train_combined_matrix_pruned_95.csv`, `test_combined_matrix_pruned_95.csv` — the 75/25 feature matrices (k-mer + CDR3 features)
- `emerson_combined_matrix_pruned_95.csv` — the external-cohort matrix
- `significant_tcrs_signed_wasserstein.csv` — the age-associated clonotypes with signed-Wasserstein scores
- `updated_tcr_age_lists_with_all_significance.csv.gz` — the full per-clonotype age table.
  Shipped **gzipped** (77 MB; 240 MB uncompressed): the plain CSV exceeds GitHub's 100 MB
  per-file limit. All 26 columns are intact — nothing was dropped. `pandas.read_csv()`
  decompresses it transparently from the `.gz` suffix, so no code change is needed to read it.
- `test_preds.csv` — held-out predictions from the original-submission model
- `adv_unique_nojoker_plus_signed_wasserstein_nonan_simplified_merged.csv` — TCR→epitope
  annotations joined to signed-Wasserstein scores; the input to Figure 4. Notebook-derived,
  **not** the output of `analysis/annotation/14` (see the annotation-chain section below)
- `TCR_shared_ELIHAY_full.csv` — per-sample shared-clonotype table with infection status,
  read by `analysis/revision/03_sex_gap_analysis.py`
- `emerson_metadata_expanded.csv` — external-cohort sample metadata including CMV serostatus
  (`Virus Diseases`), read by `analysis/revision/08_cmv_seropositivity.py`. Joins to the
  prediction files on `Sample Name` minus its `.tsv` suffix — an exact, 100% match

These are derived summaries of public immuneACCESS data (no raw sequences or personal data). `config.py` finds them automatically from a fresh clone.

**Repository size: 325 MB** (`data/` is 299 MB of it). Every file is under GitHub's 100 MB
per-file limit; the largest are `updated_tcr_age_lists_with_all_significance.csv.gz` (77 MB),
`train_combined_matrix_pruned_95.csv` (58 MB) and `data/external/merged_overlap_tcrs_wasserstein.csv`
(51 MB). Cloning pulls all of it — there is no LFS and no partial-clone configuration.

> **Note.** `test_preds.csv` exists twice on purpose: `data/test_preds.csv` is the
> original-submission model's output, `results/revision/test_preds.csv` is the retrained
> revision model's. They are different runs (MAE 7.499 vs 7.475) — the resolvers in
> `analysis/revision/` deliberately prefer the revision copy.

### Superseded scripts (archived outside this repository)

Five superseded scripts are **archived outside this repository** (see below). They are retained there so the
numbers in the archived reconciliation report remain traceable, but they are **not** part of any
current pipeline and are **not shipped here** — they live in
`~/Downloads/ttime_archive/post_submission/deprecated_pipeline/`:

| Deprecated | Superseded by |
|---|---|
| `sex_analysis.py` | `analysis/revision/03_sex_gap_analysis.py` |
| `clinical_aar.py` | `analysis/revision/04_clinical_aar_pipeline.py` |
| `emerson_zeroshot.py` | `analysis/revision/02_external_zeroshot_recalibration.py` |
| `figure2.py` | `analysis/revision/06_figure_full_a_to_i.py` |
| `stage03_train_eval.py` | `analysis/revision/01_train_primary_model.py` |

## Data

All repertoire data is public, from Adaptive Biotechnologies immuneACCESS:

- **ImmuneCODE** (SARS-CoV-2): https://clients.adaptivebiotech.com/pub/covid-2020
- **Vo' Italy longitudinal** (Gittelman et al.): https://clients.adaptivebiotech.com/pub/gittelman-2022-jci
- **Emerson et al. 2017** (external cohort; HLA- and CMV-typed): https://clients.adaptivebiotech.com/pub/emerson-2017-natgen

Epitope annotation uses **VDJdb** and **McPAS-TCR**. No raw sequencing data is redistributed here.

## Configuration

All paths live in `pipeline/config.py` and are set through environment variables — there are no paths baked into the code.

```bash
export TTIME_WORK=/path/to/workspace        # required
export TTIME_RAW_COVID=/path/to/immunecode  # raw repertoire directories
export TTIME_RAW_VO=/path/to/vo
export TTIME_RAW_EXTERNAL=/path/to/emerson
export PYTHONHASHSEED=0                     # required for reproducible downsampling
```

Install dependencies with `pip install -r requirements.txt`.

> The epitope databases ship in `data/external/reference_db/` and `analysis/annotation/13`
> resolves them itself, so no environment variable points at them. Four unused entries
> (`TTIME_VDJDB`, `TTIME_MCPAS`, `TTIME_COVID_TAGS`, `TTIME_EXTERNAL_CMV`) were removed from
> `config.py` on 2026-09-08.

## Pipeline

Everything is sorted into three tiers by what it needs to run. **`bash run_all.sh` reproduces
all of Tier A from a clean clone** — see **[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)**
for the step-by-step table, the tier boundaries and the known limits.

| Tier | Needs | Contents |
|---|---|---|
| **A** | `pip install -r requirements.txt` and what `data/` ships | `analysis/revision/01`–`08`, `analysis/annotation/13`–`14` + Figure 4, `pipeline/{mait_classifier,baselines,figures_3_4}.py`. Produces Figures 2 and 4, panels 3e–f, Supplementary Fig. S1 and every number in `results/numbers.json`. |
| **B** | raw immuneACCESS repertoires + a cluster (`TTIME_RAW_*`) | `pipeline/stage01`–`stage03a`, `emerson_build`, `clinical_build`, `vj_usage` and the diversity half of `baselines.py`. Regenerates the matrices already in `data/` — normally skipped. |
| **C** | not reproducible here | Figure 3 panels a–d (H2O AutoML; the script is archived outside this repository, output verified at AUC 0.919 / accuracy 0.838), the diversity baselines and V/J usage. Documented, not silently missing. |

> **Before quoting any number**, read *Known limits* in `docs/REPRODUCIBILITY.md`: run-to-run
> training variance exceeds several reported effects (the hypertension result changes
> significance between runs), the split is at sample level rather than donor level, the CMV
> effect exists only after age/sex adjustment, and the +1.67 yr sex offset in the discussion is
> a cited literature value, not a T-Time output.

### Core model — the training chain (Tier B, plus the legacy trainer)
| Script | Step |
|---|---|
| `stage01_qc_downsample.py` | QC (≥200,000 templates) and multinomial downsampling to uniform depth |
| `stage01b_split.py` | Cohort definition and train/test split |
| `populate_split_dirs.py` | Materialise the train/test repertoire directories |
| `stage02_wasserstein.py` | Public-TCR identification, signed-Wasserstein age scoring, GMM significance |
| `stage03a_kmers.py` | K-mer (K=3) feature extraction |
| `analysis/revision/01_train_primary_model.py` | Feature integration, regressor training and held-out evaluation — **the canonical model behind the reported numbers** |

Age-associated clonotypes are derived from **training donors only** (`stage02_wasserstein.py`, default mode), so no test-set information enters feature definition. Pass `combined` to score across all donors instead.

### Validation and downstream analyses
| Script | Analysis |
|---|---|
| `analysis/revision/01_train_primary_model.py` | Retrain the primary MLP (model M1) on the full train matrix; writes the revision `test_preds.csv` |
| `emerson_build.py` | Build the external-cohort feature matrix, aligned to the training feature space |
| `analysis/revision/02_external_zeroshot_recalibration.py` | Zero-shot external evaluation and per-cohort affine recalibration |
| `clinical_build.py`, `analysis/revision/04_clinical_aar_pipeline.py` | Age acceleration by clinical condition, with permutation null, BH correction and age/sex adjustment |
| `mait_classifier.py` | Physicochemical (Atchley) young-vs-old classifier, with and without MAIT depletion |
| `analysis/revision/03_sex_gap_analysis.py` | Male-vs-female residual offset, primary and recalibrated-external, with a 10% outlier filter |
| `analysis/revision/05_all_categories_exploration.py` | Exploratory sweep of every clinical category (not a manuscript figure) |
| `vj_usage.py` | V/J gene usage of age-associated clonotypes |
| `baselines.py` | Repertoire-diversity and linear baselines |
| `figures_3_4.py`, `analysis/revision/06_figure_full_a_to_i.py` | Figure generation |
| `analysis/revision/08_cmv_seropositivity.py` | CMV serostatus vs predicted-age residual in the external cohort (ANCOVA, age- and sex-adjusted) |
| `analysis/revision/supplementary_materials.py` | **All eight supplementary deliverables** — Supplementary Figures S1–S4 and Tables 1–4, one function each (`generate_suppfig1()`, `generate_table1()`, …). Run it with no arguments for all eight, or name items to run a subset (`… supplementary_materials.py table4 suppfig3`); `--list` shows the names. Tables 1–3 and Figures S1–S4 reproduce their original-submission counterparts value-for-value; Table 4 (n = 762/66/22/22) **replaced** its counterpart, whose clinical columns (n = 16/14/6) came from a superseded test-split-only categorisation — since 2026-09-09 its output *is* `results/supplementary/tables/Supplementary Table 4.xlsx`, and the original was retired to the external archive. Consolidated on 2026-09-09 from the eight numbered scripts `09`–`16`, which it replaces byte-for-byte |
| `analysis/revision/07_build_numbers_revision.py` | Assembles `results/numbers.json` from artefacts already on disk — trains nothing |

> **`figures_3_4.py` Figure 4b crashed on every run until 2026-09-08.** Line 18 indexed
> `sp['sp']`, but `results/tables/antigen_age_bias.csv` has columns `cat,median,count` — there
> has never been an `sp` column, so the block died with `KeyError: 'sp'` before reaching the
> Figure 3 additions. Fixed to `sp['cat']`; see the archived changelog §B1 for the
> traceback and the rebuilt panel. The script writes to `config.FIGURES`
> (`workspace/results/figures/`), not to `results/figures/` — so it never overwrites a shipped
> figure. A stale `results/figures/Figure4b.{pdf,png}` pair from a different generator was
> deleted on 2026-09-09; Figure 4b now lives only where the script puts it.
>
> The Figure 3 half of that script still needs `baselines.json` and `mait_classifier.json` in
> `config.OUTPUTS`, which require the raw repertoire TSVs — a missing workspace artefact, not a
> bug.

### Epitope annotation chain

`analysis/annotation/` runs end to end from the reference databases shipped in
`data/external/reference_db/`. The scripts use bare filenames, so run them with the working
directory set to a directory holding their inputs:

```
13_epitope_database_harmonization.py   vdjdb.slim.txt + McPAS-TCR.csv + 20250312-TRAIT_search_download.xlsx
  -> adv_unique_nojoker.csv
14_tcr_epitope_age_alignment.py        + updated_tcr_age_lists_with_all_significance.csv.gz
  -> adv_unique_nojoker_plus_signed_wasserstein_nonan_simplified_merged.csv
Figure4_Landscape_Analysis.py
  -> Figure4_Landscape_Final_Nature_V2.pdf
```

> **The published Figure 4 is not this chain's output.** Scripts 13/14 are a simplified
> reconstruction of the original notebook; the shipped
> `data/adv_unique_nojoker_plus_signed_wasserstein_nonan_simplified_merged.csv` came from the
> notebook and carries a richer species vocabulary. Both files now run through
> `Figure4_Landscape_Analysis.py`, but they give different top-9 species (the reconstruction
> surfaces `YFV`/`HCV` where the notebook file has `Cancer/Self antigen`/`T1D`). See
> `data/external/README.md` for the two annotation lineages.

**What had to be fixed to make this chain run** (details and verification in the archived
changelog, addendum B — see `docs/REPRODUCIBILITY.md` → *Full history*):

- **`13`: the `HomoSapiens` spelling was never harmonised.** VDJdb and TRAIT write the
  unannotated human self-antigen bucket as `HomoSapiens` (no space; TRAIT also has
  `Homo Sapiens` / `Homo sapiens`). `SPECIES_ALIASES` had no entry for it, so it passed
  straight through `13` and `14` and never matched `Figure4`'s Human filter, which tests
  against the reference vocabulary's `"Human (Homo sapiens)"`. At 34,869 of VDJdb's TRB rows
  it is the single largest species, so it entered the top-9 and killed the figure with
  `KeyError: 'HomoSapiens'`. A single alias now maps all three spellings onto
  `Human (Homo sapiens)` (§B2).
- **`Figure4`: the Human filter and `category_map` are now total.** The filter matches a set of
  spellings rather than one string, and `category_map` gained `HomoSapiens`, `YFV` and `HCV`
  (§B3). Both changes are strictly additive: rendered from the shipped merged file, the
  Figure 4 PDF is byte-identical before and after.
- **`13` → `14` contract** (earlier, §1.2): `13` used to *rename* `Epitope_gene`/
  `Epitope_species` to `*_adv`, which `14` then could not find (`KeyError: 'Epitope_species'`).
  The rename is now additive, so both column families survive.

---

## Results

Full values in `results/numbers.json`.

**Age prediction.** On the held-out test set (n = 818): **MAE 7.48 years, RMSE 9.74, R² 0.784**, with essentially identical performance in females (7.48) and males (7.48).

**External validation.** Applied with frozen weights to an independent cohort, the model preserves age ranking (Pearson r = 0.73) with **zero-shot MAE 9.99, R² +0.13**; a single two-parameter per-cohort affine recalibration gives **MAE 7.60, R² 0.52** — i.e. the clock transfers up to a per-cohort linear calibration.

> **Which external matrix this is computed on.** The figure above uses `emerson_combined_matrix_pruned_95.csv`, whose feature space is exactly the 5,460 features the model was trained on (`train_combined_matrix_pruned_95.csv`, identical set *and* order) — the only feature space a trained model can be evaluated in. That alignment is not an arbitrary filter: `pipeline/emerson_build.py` applies the same one when it builds `emerson_combined_matrix.csv`, reading the reference columns straight from the training matrix (lines 19–23) and emitting them in training order (line 87). `analysis/revision/02` additionally re-aligns whatever matrix it is given. The archived **MAE 13.5 / R² −0.32** came from `emerson_combined_matrix.csv`, which is a Tier B output and is not shipped here. Because both matrices carry the identical feature set, the gap between the two figures **cannot** be a feature-alignment difference; it must lie in the underlying build (QC/downsampling run or sample set), and that could not be determined without the file. `02` still prints a warning about the substitution.

**Model choice.** The neural regressor (R² 0.784) outperforms linear models on the same features (ElasticNet R² 0.555) and repertoire-diversity baselines (R² ≈ 0.38), with non-overlapping bootstrap confidence intervals.

**Sequence architecture.** Young- and old-associated clonotypes are separable on physicochemical composition alone (AUC 0.890), driven mainly by refractivity and polarity/charge at central CDR3 positions. Putative MAIT clonotypes account for 7% of the youth-associated set and their removal leaves the separation essentially unchanged (AUC 0.888); V-gene usage nevertheless shows a broad MAIT-consistent bias in youth (34% TRBV6/20/4-3 vs 10% in the old-associated set).

**Immunological signal.** CMV-seropositive donors show **+2.76 years of accelerated predicted immune age** (age- and sex-adjusted; n = 395 of the recalibrated external slice; P = 1.2 × 10⁻⁴), recovering the best-established driver of immunosenescence. The effect requires the age/sex adjustment — the unadjusted contrast on that slice is null (+0.16 yr, P = 0.87), because the affine recalibration removes the cohort offset the raw contrast rode on. On the full 494-sample zero-shot set the adjusted effect is +3.66 yr. Reproduce with `analysis/revision/08_cmv_seropositivity.py`. Females show an immunologically younger profile: males carry a **+1.37-year offset** in the primary cohort (Welch P = 0.010; n = 735 after the top-10% residual filter). In the external cohort the same contrast is +0.76 years and **not significant** (P = 0.34, n = 356), and it is abolished during acute COVID (+0.21, P = 0.85). Reproduce with `analysis/revision/03_sex_gap_analysis.py`. (The +1.67-year figure quoted in the manuscript's discussion is the published RFU-framework value, cited as independent convergence — it is not a T-Time output and is not reproducible from this repository.) Treated hypertension is associated with decelerated aging that survives age/sex adjustment (−3.38 years); an apparent acceleration in autoimmune conditions does not survive adjustment and is reported as an age-distribution confound.

**HLA.** Age association is largely consistent across the common HLA-A restrictions, indicating the signal is not an artefact of HLA background.

> **Figure 3 provenance.** Panels a–d were produced with H2O AutoML by a script that is now **archived outside this repository** (`~/Downloads/ttime_archive/as_submitted/src/12_tcr_sequence_classification.py`; the submission pinned `h2o==3.46.0.10`). The output has been independently verified against archived headline numbers (AUC = 0.919, accuracy = 0.838). The shipped figure is retained as-is and **cannot be regenerated from this repository**. Panels e–f are regenerated by `pipeline/figures_3_4.py` from `pipeline/baselines.py` and `pipeline/mait_classifier.py`.

## Reproducibility

**Full guide: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)** — tier tables, the one-command
Tier A run, and the known limits that must be read before quoting any figure.

- `bash run_all.sh` reproduces Tier A from a clean clone. Set `TTIME_REVISION_OUT` to a scratch
  directory to compare a fresh run against the shipped artefacts without overwriting them.
- `results/numbers.json` is **generated** by `analysis/revision/07_build_numbers_revision.py`;
  every key carries its own `source`, and fields whose inputs are not shipped are `null` with a
  `"requires re-run: …"` note rather than a guessed value. The hand-transcribed predecessor is
  archived **outside this repository**, at
  `~/Downloads/ttime_archive/docs_internal/archive/numbers_original_submission.json`.
- ⚠️ **Trained-model figures move between runs.** Torch is not bitwise reproducible across
  backends; a CPU re-run shifted MAE by 0.06 yr, the sex gap by 0.16 yr and the clinical AAR by
  0.86 yr, and the hypertension result stopped being significant (BH P 0.017 → 0.091). Treat the
  shipped values as one citable run. Discussed in the manuscript, Methods §4.8.
- All randomness is seeded (`RANDOM_SEED` in `config.py`); set `PYTHONHASHSEED=0` so repertoire downsampling is reproducible.
- Paths are environment-driven; no absolute paths are embedded in the code.
- `analysis/revision/01_train_primary_model.py` is the canonical model behind the reported
  held-out numbers; it writes the trained weights, per-sample predictions and metrics.
  (`stage03_train_eval.py`, an earlier superseded trainer, is archived outside this repository.)

## Citation

Levi E, Zilberberg A, Efroni S. *A T Cell Receptor Repertoire Clock for Immune Aging.*
