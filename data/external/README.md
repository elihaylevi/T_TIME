# `data/external/` — externally sourced inputs and epitope-annotation intermediates

Files the analysis code reads but that were **not** produced by `pipeline/`. Before the
restructure these lived scattered in `~/Downloads/` and `~/Desktop/TEE TIME/`, which meant
several scripts in this repo could not run from a clean clone. They are collected here with
their original modification times preserved.

## Contents

| File | Size | Origin | Consumed by |
|---|---|---|---|
| `reference_db/vdjdb.slim.txt` | 19.8 MB | VDJdb release 2025-07-30 | `analysis/annotation/13` |
| `reference_db/McPAS-TCR.csv` | 8.2 MB | McPAS-TCR public download | `analysis/annotation/13` |
| `reference_db/20250312-TRAIT_search_download.xlsx` | 6.5 MB | TRAIT export, 2025-03-12 | `analysis/annotation/13` |
| `adv_unique_nojoker.csv` | 7.6 MB | output of script `13` | `analysis/annotation/14` |
| `vdjdb_trait_onlyB.csv` | 19.8 MB | TRB-only VDJdb+TRAIT export | `notebooks/tcr_correlations.ipynb` |
| `merged_overlap_tcrs_wasserstein.csv` | 51.3 MB | notebook output (see below) | downstream correlation analyses |
| `tcrb_union_TRAIT_VDJdb_McPAS.csv` | 4.3 MB | notebook output | exploratory only |
| `FINAL_Clinical_Case_Study_Table.csv` | 5.9 MB | clinical metadata table | `analysis/revision/04_clinical_aar_pipeline.py` |

## Two separate annotation lineages — do not confuse them

There are **two independent** TCR→epitope annotation chains in this project. They produce
different files and only one of them carries MHC.

**Lineage A — the scripted pipeline (no MHC).**

```
reference_db/{vdjdb.slim.txt, McPAS-TCR.csv, 20250312-TRAIT_search_download.xlsx}
  └─ analysis/annotation/13_epitope_database_harmonization.py
      └─ adv_unique_nojoker.csv
          └─ analysis/annotation/14_tcr_epitope_age_alignment.py
              (+ data/updated_tcr_age_lists_with_all_significance.csv.gz)
              └─ data/adv_unique_nojoker_plus_signed_wasserstein_nonan_simplified_merged.csv
                  └─ analysis/annotation/Figure4_Landscape_Analysis.py  →  Figure 4
```

Script `13` reads only `cdr3`, `antigen.gene` and `antigen.species` from VDJdb, so **MHC
never enters this lineage**. The final merged file has 5 columns and no MHC.

Lineage A now runs end to end (verified 2026-09-08). Two crashes had to be cleared first —
`13` never harmonised VDJdb/TRAIT's `HomoSapiens` spelling, and `Figure4_Landscape_Analysis.py`
filtered only the reference vocabulary's `Human (Homo sapiens)`; see
the archived internal changelog, addendum B
(`~/Downloads/ttime_archive/docs_internal/CHANGELOG_RESTRUCTURE.md`).

**The shipped `adv_unique_nojoker.csv` is still the notebook's, not `13`'s output**, and the two
differ in both size and vocabulary. Re-running `13` gives 128,789 entries against the shipped
file's 116,999, and it harmonises *more* aggressively, not less — one `M. tuberculosis` bucket
(16,681) where the shipped file keeps `M.Tuberculosis` / `M. tuberculosis` / `M.tuberculosis` /
`Mtb` apart, one `CMV` (32,795) where the shipped file also carries `Cytomegalovirus (CMV)`.

The shipped **merged** file is a further step removed: its `Epitope_species_norm` values
(`Cytomegalovirus (CMV)`, `Cancer/Self antigen`, `T1D`, `COVID-19 / SARS-CoV-2 / SARS-CoV`,
`Human (Homo sapiens)`) are produced by neither `13` nor `14` — they come from an additional
notebook mapping layer. Running the scripted chain therefore reproduces Figure 4's *structure*,
not the published panel: its top-9 species are `CMV, TB, Influenza, EBV, SARS-CoV-2, HIV,
Neoantigen, YFV, HCV` versus the published `CMV, Influenza, TB, EBV, Cancer/Self, SARS-CoV-2,
HIV, T1D, Neoantigen`, because the scripts leave `Diabetes Type 1`, `Merkel cell carcinoma` and
the other per-tumour labels ungrouped. The published figure comes from the shipped merged file,
which still renders byte-identically.

**Lineage B — the notebook (carries MHC).**

```
data/updated_tcr_age_lists_with_all_significance.csv.gz
  ×  vdjdb_trait_onlyB.csv            (CDR3b + Species/Epitope/TRBV/TRBJ/MHC_A/MHC_B/MHC_class/PMID/Category/Source)
  └─ notebooks/tcr_correlations.ipynb  cell 183   (rename CDR3b→TCR, inner merge on TCR)
      └─ merged_tcr_vdjdb.csv          [NOT SHIPPED — regenerable, see below]
          └─ cells 186 → 187 → 190     (read, drop_duplicates, to_csv)
              └─ merged_overlap_tcrs_wasserstein.csv
```

If you need `MHC_A` / `MHC_B` / `component_zscore`, they come from lineage B, **not** from a
missing variant of script `13`. No MHC-preserving version of `13` has ever existed.

## Regenerating `merged_tcr_vdjdb.csv`

Deliberately not shipped: it is a 76.5 MB pure intermediate, fully reproducible in one step.

```python
import pandas as pd

df_main = pd.read_csv("data/updated_tcr_age_lists_with_all_significance.csv.gz")
df_vdj  = pd.read_csv("data/external/vdjdb_trait_onlyB.csv")
df_vdj  = df_vdj.rename(columns={"CDR3b": "TCR"})

merged = pd.merge(df_main, df_vdj, on="TCR", how="inner")
merged.to_csv("merged_tcr_vdjdb.csv", index=False)
# expect 25,638 rows x 38 columns
```

`merged_overlap_tcrs_wasserstein.csv` is then just `merged.drop_duplicates()` —
25,638 → 17,983 rows, same 38 columns.

> **Note on the notebook.** `notebooks/tcr_correlations.ipynb` also contains an
> *alternative* recipe for `merged_overlap_tcrs_wasserstein.csv` (the cell right after the
> "DEAD END" warning). It never produced the shipped file and is kept only for provenance.
> It is marked in place; do not run it.

## Provenance of these copies

Copied verbatim (MD5-verified, mtimes preserved) from:

| File | Original location |
|---|---|
| `vdjdb_trait_onlyB.csv`, `merged_overlap_tcrs_wasserstein.csv`, `tcrb_union_TRAIT_VDJdb_McPAS.csv`, `adv_unique_nojoker.csv`, `reference_db/McPAS-TCR.csv` | `~/Downloads/` |
| `reference_db/vdjdb.slim.txt` | `~/Downloads/vdjdb-2025-07-30/` |
| `reference_db/20250312-TRAIT_search_download.xlsx` | `~/Downloads/Interactive_TCR-pMHC_Pairs.zip_20250312/` |
| `FINAL_Clinical_Case_Study_Table.csv` | `~/Desktop/TEE TIME/` |

The originals were left in place; nothing was moved out of those directories.
