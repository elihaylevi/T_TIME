# Reproducing T-Time

Everything in this repository is sorted into three tiers by what it needs to run. **Tier A runs
from a clean clone with nothing but `pip install -r requirements.txt`.** Tier B needs the raw
repertoire files and a cluster. Tier C cannot be reproduced here at all, and says so.

```bash
git clone <repo> && cd ttime
pip install -r requirements.txt
bash run_all.sh
```

That single command runs all of Tier A in order and writes to `results/revision/`. To check a
fresh run against the shipped artefacts without overwriting them:

```bash
TTIME_REVISION_OUT=/tmp/ttime_check bash run_all.sh
```

Every script honours `TTIME_REVISION_OUT`, so a verification run reads the fresh artefacts and
writes beside them. Nothing under `data/` or `results/` is modified.

---

## Tier A — runs from a clean clone

Trains three small MLPs on CPU; budget roughly half an hour.

| Step | Script | Produces | Manuscript |
|---|---|---|---|
| A1 | `analysis/revision/01_train_primary_model.py` | `test_preds.csv`, `test_metrics_by_sex.csv` | MAE 7.475 / R² 0.782 · Fig 2a |
| A2 | `analysis/revision/02_external_zeroshot_recalibration.py` | `emerson_zeroshot_preds.csv`, `emerson_recalibrated_persample.csv`, `external_validation.json` | zero-shot 9.99 / R² +0.13; recalibrated 7.60 / 0.52 · Fig 2b–c |
| A3 | `analysis/revision/03_sex_gap_analysis.py` | `gap1`–`gap3b*.csv` | sex gaps · Fig 2d–f |
| A4 | `analysis/revision/04_clinical_aar_pipeline.py` | `clinical_aar_persample_NEW.csv`, `baseline_reference.csv`, `baseline_params.json` | clinical AAR · Fig 2g–i |
| A5 | `analysis/revision/05_all_categories_exploration.py` | `all_categories_summary.csv`, `all_categories_grid.*` | exploratory only |
| A6 | `analysis/revision/06_figure_full_a_to_i.py` | `Figure2_full_reproduced_a-i.*` in `results/revision/`; its PDF is also **the shipped `results/figures/Figure2.pdf`** | **Figure 2** |
| A7 | `analysis/revision/08_cmv_seropositivity.py` | `cmv_seropositivity.json` | CMV +2.76 yr |
| A8 | `pipeline/mait_classifier.py` | `mait_classifier.json` | AUC 0.890 / 0.888 · Fig 3f |
| A9 | `pipeline/baselines.py` | `baselines.json` (Ridge + ElasticNet) | Fig 3e |
| A10 | `pipeline/figures_3_4.py` | `Figure3_additions.*`, `Figure4b.*` — into `config.FIGURES` (`workspace/results/figures/`), **not** `results/figures/` | Fig 3e–f, Fig 4b |
| A11 | `analysis/annotation/13_epitope_database_harmonization.py` | `adv_unique_nojoker.csv` | intermediate |
| A12 | `analysis/annotation/14_tcr_epitope_age_alignment.py` | `..._simplified_merged.csv` | intermediate |
| A13 | `analysis/annotation/Figure4_Landscape_Analysis.py` | `Figure4_Landscape_Final_Nature_V2.pdf` | **Figure 4** |
| A14 | `analysis/revision/07_build_numbers_revision.py` | `results/numbers.json` | **all headline numbers** |

> **Figure 2 provenance.** `results/figures/Figure2.pdf` is a byte-identical copy of A6's
> output, `results/revision/Figure2_full_reproduced_a-i.pdf`
> (227,430 B, `sha256 24649aea…85ba1f97`), placed there on 2026-09-09. It replaced a stale
> 159,201-byte render (`sha256 dd7b54cf…3ad5cdd9`, matplotlib 3.9.4, created 2026-07-29) that
> predated the revision retrain and was the output of no script in this repository. That file
> was deleted, not archived; it exists nowhere in `~/Downloads/ttime_archive/` and was distinct
> from both the original-submission Figure 2 there (`as_submitted/results/Figure2.pdf`,
> 278,250 B, md5 `3ea8699f…`) and the reviewer-round revision
> (`manuscript/reviewer_round/Figure2_revised (1).pdf`, 190,885 B). Re-running A6 reproduces
> the shipped file exactly — unlike Figures 1 and 3, Figure 2 has a producer in this
> repository, which is why it is Tier A and not Tier C.

One more Tier A script sits outside `run_all.sh` because it writes to the working
directory: **`analysis/revision/supplementary_materials.py`**, which produces all eight
supplementary deliverables. Run it bare for all of them, or name items for a subset
(`--list` prints the names):

| Item | Produces | Status |
|---|---|---|
| `suppfig1` | **Supplementary Figure S1** | runs, 61,973-byte PDF |
| `table1` | **Supplementary Table 1** | runs; output matches the shipped table **20/20 values** |
| `table2` | **Supplementary Table 2** | runs; matches the shipped table **value for value** |
| `table3` | **Supplementary Table 3** | runs; matches **15/15 numeric cells** and both p-values |
| `suppfig2` | **Supplementary Figure S2** | runs; every text string in the rendered PDF matches the shipped figure |
| `suppfig3` | **Supplementary Figure S3** | runs; every text string in the rendered PDF matches the shipped figure |
| `suppfig4` | **Supplementary Figure S4** | runs; every text string in the rendered PDF matches the shipped figure |
| `table4` | **Supplementary Table 4** | runs; **replaced** the original-submission table rather than reproducing it (2026-09-09) - see Tier C |

> Until 2026-09-09 these were eight separate scripts, `analysis/revision/09`-`16`. They were
> consolidated into the single module and deleted; the merge was verified first, against the
> eight scripts' own output - all four CSVs byte-identical, and all four PDFs identical in
> **every decompressed content stream**, not merely in their rendered text. Selective runs
> and out-of-order runs were checked too, and reproduce the same artefacts. The full
> evidence table is in the internal changelog, **archived outside this repository** at
> `~/Downloads/ttime_archive/docs_internal/CHANGELOG_RESTRUCTURE.md` §13.

A second script also sits outside `run_all.sh`:
**`analysis/annotation/15_merged_overlap_extraction.py`**, which rebuilds
`data/external/merged_overlap_tcrs_wasserstein.csv` (lineage B — the epitope join that
carries `MHC_A`/`MHC_B`/`component_zscore`). It reproduces the shipped file **byte for
byte** (`sha256 b5794c13…`, 17,982 x 38, 53,751,400 B); `--verify` asserts it.

> **Provenance.** Extracted on 2026-09-09 from `notebooks/tcr_correlations.ipynb`
> (389 cells) — cells 183 → 186 → 187 → 189. The notebook was archived outside this
> repository in the same change, alongside its 117 MB with-outputs twin; `notebooks/`
> no longer exists here.
>
> **One trap worth knowing.** The notebook writes a 76.5 MB `merged_tcr_vdjdb.csv` and
> immediately reads it back, which looks like removable dead weight. It is not: pandas'
> CSV reader does not round-trip floats to nearest, so skipping the write/read leaves
> 342 of the 17,982 rows one ULP off and the file no longer matches. The script performs
> the round-trip by default and deletes the intermediate afterwards; `--no-round-trip`
> skips it and is documented to fail `--verify`.

The published copies of Supplementary Figures S1-S4 and Tables 1-4 are in
`results/supplementary/`. All eight regenerate. Tables 1-3 and Figures S1-S4 reproduce
their original-submission counterparts; **Table 4 is the one exception - it *replaces*
its counterpart**, whose clinical columns were wrong (see Tier C). The original-submission
Table 4 is **archived outside this repository**, at
`~/Downloads/ttime_archive/docs_internal/archive/supplementary_table4_original_submission.xlsx`.

The XLSX written by `table4` is not byte-reproducible either - openpyxl stamps a creation
time into the archive - so verify it on the `.csv` sidecar shipped beside it, or on the
`N` row, not on the XLSX hash.

None of the four figures is byte-reproducible: matplotlib and font versions differ, so the
PDFs differ in size. Identity was established on the *rendered text* - every string the
shipped PDF draws, extracted from its content streams, against the same extraction of the
regenerated PDF. That is what separated each figure from its near-identical siblings in
the notebook: S2 from six candidates (raw column names in the axis labels), S3 from six
(`Absolute Weight`, not `Absolute Input Layer Weight`), and S4 from three (the y-limit
factor, which sets the tick sequence to `0 .. 14`).

## Tier B — needs the raw repertoire workspace

These regenerate the matrices already shipped in `data/`. **Skipping them is the normal path.**

```bash
export TTIME_WORK=/path/to/workspace
export TTIME_RAW_COVID=... TTIME_RAW_VO=... TTIME_RAW_EXTERNAL=...
export PYTHONHASHSEED=0     # required for reproducible downsampling
sbatch pipeline/slurm/stage01.sbatch     # etc.
```

`stage01_qc_downsample` → `stage01b_split` → `populate_split_dirs` → `stage02_wasserstein` →
`stage03a_kmers`; plus `emerson_build`, `clinical_build`, `vj_usage`, the diversity half of
`baselines.py`.

Raw data is public, from Adaptive Biotechnologies immuneACCESS — see the README's *Data*
section for the three cohort URLs.

## Tier C — not reproducible here

| Item | Why | Status |
|---|---|---|
| Figure 3, panels a–d | Produced with H2O AutoML by a script now **archived outside this repository** (`~/Downloads/ttime_archive/as_submitted/src/12_tcr_sequence_classification.py`; the submission pinned `h2o==3.46.0.10`). | Output **independently verified** against archived headline numbers (AUC = 0.919, accuracy = 0.838). The shipped figure is retained as-is and cannot be regenerated here. |
| `results/figures/Figure1.pdf` | The original-submission Figure 1 (schematic). No producer in the repository and none expected — it was drawn, not computed. | Copied in from `~/Downloads/ttime_archive/as_submitted/results/Figure1.pdf` on 2026-09-09 so that the manuscript's four main figures all ship from one directory. Byte-identical to the archived original (`sha256 081eaa52…a62373a`). |
| `results/figures/Figure3.pdf` | No producer in the repository — assembled outside it from the panels (a–d from the archived H2O script, e–f from `pipeline/figures_3_4.py`). | **Replaced on 2026-09-09** by the merged a–f version, previously sitting at the repository root as `Figure3_merged_a_to_f.pdf` (321,601 B, `sha256 268635bd…37c558fd8`). The 19,894-byte file it replaced (`sha256 7f1ef7c7…8f57cabed`) was deleted, not archived; a larger, distinct original-submission Figure 3 survives at `~/Downloads/ttime_archive/as_submitted/results/Figure3.pdf` (69,854 B). |
| diversity baselines (`diversity_ridge`, `diversity_gbm`) | Need the raw TSVs (Tier B). | `baselines.json` omits them; Figure 3e draws three bars instead of five. Values in the archive are carried from the original run. |
| `vj_usage` | Needs V/J calls from the raw TSVs. | Values carried from the original run. |
| `notebooks/tcr_correlations.ipynb` | 389 exploratory cells; only the four that build `merged_overlap_tcrs_wasserstein.csv` produced a shipped artefact. | **Extracted and archived (2026-09-09).** Those four are now `analysis/annotation/15_merged_overlap_extraction.py` (Tier A, byte-verified); the notebook moved to `~/Downloads/ttime_archive/`. The other 385 cells are exploratory plotting over that same file and produced nothing that ships. |
| External MAE 13.5 / R² −0.32 | Computed on `emerson_combined_matrix.csv`, a Tier B output that is not shipped. | See *Known limits* below. |
| Supplementary Table 4 | **Source found and superseded (2026-09-09).** The shipped table (n=762/16/14/6) was traced to the archived `superseded/Figure2_Age_Prediction_Clinical.py` and reproduced from it exactly: it tabulates only the 56 clinical patients inside the internal test split (55 after that script's sex filter) under the pre-revision `Hypertension`/`Immune System`/`Neuro-Psych` taxonomy, whose flag columns were mis-listed as free-text words. Section 2.5 tests all 244 clinical patients under the canonical taxonomy: n = 66/22/22. | **Regenerated** by `analysis/revision/supplementary_materials.py` (`table4`). The `Healthy Reference` column (n=762, 52.0 [35.0-67.0], 50.9% female) is unchanged; the clinical columns are replaced. Since 2026-09-09 `results/supplementary/tables/Supplementary Table 4.xlsx` **is** this script's output (762/66/22/22); the original-submission file was retired **outside this repository**, to `~/Downloads/ttime_archive/docs_internal/archive/supplementary_table4_original_submission.xlsx`. |

---

## Known limits — read before quoting any number

### 1. Run-to-run variance is larger than several reported effects

Torch training is not bitwise reproducible across backends. The shipped artefacts come from a
GPU run; a CPU re-run on 2026-09-08 gave:

| quantity | shipped | re-run | delta |
|---|---|---|---|
| primary MAE | 7.475 | 7.534 | +0.059 |
| primary R² | 0.782 | 0.777 | −0.005 |
| sex gap, primary | 1.374 | 1.530 | +0.156 |
| CMV adjusted (yr) | 2.762 | 3.576 | +0.814 |
| hypertension AAR | −3.249 | −2.393 | +0.856 |
| **hypertension BH-adjusted P** | **0.017** | **0.091** | **+0.074** |

The two fully deterministic keys — `antigen_age_bias` and `hla_restriction` — are identical
across runs, confirming the variance comes from model training alone.

**The hypertension result crosses the significance threshold between runs.** Treat
`results/numbers.json` as one specific, citable run rather than a value a re-run will land on.
Discussed in the manuscript, Methods §4.8.

### 2. The external figure is computed on the pruned matrix

`analysis/revision/02` uses `emerson_combined_matrix_pruned_95.csv`, whose feature space is
exactly the 5,460 features the model was trained on (identical set *and* order) — the only
feature space a trained model can be evaluated in. `pipeline/emerson_build.py` applies the same
alignment when it builds `emerson_combined_matrix.csv` (lines 19–23 and 87), so **both matrices
carry the identical feature set** and the archived 13.5 figure cannot differ from 9.99 by
feature alignment. The cause lies in the underlying build and could not be determined without
that unshipped file. `02` prints a warning about the substitution when it runs.

### 3. The split is at sample level, not donor level

Donor identity is recoverable for 702 of 818 test samples (85.8%), including 100% of the Vo'
cohort. Within that subset, 19 donors — 20 samples, 2.4%, all ImmuneCODE — also appear in
training. Removing them changes MAE by 0.003 yr and R² by 0.001. The remaining 116 samples carry
no recoverable donor id, so this is a lower bound.

### 4. CMV must be quoted as adjusted

The +2.76 yr effect is the CMV coefficient of `residual ~ Age + Sex + CMV`. **Unadjusted, the
contrast on the same slice is null** (+0.16 yr, P = 0.87) — the affine recalibration removes the
offset the raw contrast rode on. Re-running without the adjustment yields nothing, and that is
expected.

### 5. One number in the manuscript is not ours

The **+1.67 yr** male offset quoted in the discussion is the published RFU-framework value
(reference 35), cited as independent convergence. It is not a T-Time output and is not
reproducible from this repository. T-Time's own figures are +1.374 yr (primary, P = 0.010) and
+0.760 yr (external, **not significant**, P = 0.340).

---

## Where the numbers live

`results/numbers.json` is **generated** by `analysis/revision/07_build_numbers_revision.py` from
artefacts already on disk — it trains nothing and every key carries its own `source`. Fields
whose inputs are not shipped are `null` with a `"requires re-run: …"` string rather than a
guessed value.

Its hand-transcribed predecessor is **archived outside this repository**, at
`~/Downloads/ttime_archive/docs_internal/archive/numbers_original_submission.json`. That
predecessor was **not produced by any script** — it was transcribed by hand, which is why it
was superseded. It still carries the superseded external 13.521 / −0.324 and the CMV
+2.75 / P = 0.0038 / n = 493 triplet, so those remain traceable.

## Environment

`torch 2.8.0+cpu`, `pandas 1.4.4`, `numpy 1.21.5`, `scikit-learn 1.0.2`, `scipy 1.9.1`,
`matplotlib 3.5.2`, `seaborn 0.11.2`, `statsmodels 0.13.2` on Python 3.9.

`run_all.sh` sets `MPLBACKEND=Agg` — three scripts call `plt.show()`, which blocks forever under
an interactive backend.

## Full history — archived outside this repository

This file is now the **only** documentation the repository ships, and it is self-contained:
how to run everything (Tier A above), what cannot be run and why (Tier C), and every caveat
that bears on a quoted number (*Known limits*). Nothing below is needed to reproduce a result.

The internal working documents — the consolidation audit trail — were moved out on
2026-09-09 to `~/Downloads/ttime_archive/docs_internal/` (see that archive's
`ARCHIVE_README.md`). **None of it is in this repository or in its git history.** It is kept
so historical numbers can still be explained if anyone asks:

| Archived file | What it holds |
|---|---|
| `docs_internal/CONSOLIDATION_PLAN.md` | The consolidation plan and its decision gate (D1–D5). |
| `docs_internal/CHANGELOG_RESTRUCTURE.md` | Every change made during the consolidation, each with its verification evidence — including §12 (Supplementary Table 4's superseded producer) and §13 (the eight-script merge). |
| `docs_internal/RECONCILIATION_REPORT.md` | The earlier audit that started it. |
| `docs_internal/archive/` | `numbers_original_submission.json` and `supplementary_table4_original_submission.xlsx` — the two superseded artefacts, kept for provenance. |
