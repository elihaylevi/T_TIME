# T-TIME: TCR-based Immunological Age Prediction

**A Deep-Learning Framework for Quantifying Immune Aging from Repertoires**

T-TIME is a sequencing-first computational framework designed to predict immunological age directly from T-cell receptor β-chain (TCRβ) repertoires. By integrating Wasserstein distance (Earth Mover's Distance) scoring with overlapping sequence motifs, the pipeline maps the complex biological remodeling of immunosenescence into a continuous, scalable, and interpretable biological clock.

## 📂 Repository Structure

```text
.
├── README.md
├── requirements.txt
│
├── data/                             # Processed datasets for modeling
│   ├── significant_tcrs_signed_wasserstein.csv
│   ├── test_combined_matrix_pruned_95.csv
│   ├── test_preds.csv
│   └── train_combined_matrix_pruned_95.csv
│
├── results/                          # Final vector PDFs of primary figures
│   ├── Figure1.pdf
│   ├── Figure2.pdf
│   ├── Figure3.pdf
│   └── Figure4.pdf
│
├── scripts/                          # Plotting code for Nature-style visualizations
│   ├── Fig3_TCR_Age_Pipeline_Metrics.py
│   ├── Figure2_Age_Prediction_Clinical.py
│   ├── Figure4_Landscape_Analysis.py
│   └── SuppFig1_Model_Fairness_Stability.py
│
├── src/                              # Sequential Analysis Pipeline (01-14)
│   ├── 01_cohort_normalization_and_splitting.py
│   ├── 02_identify_public_tcrs.py
│   ├── 03_extract_kmer_features.py
│   ├── 04_aggregate_tcr_age_distributions.py
│   ├── 05_calculate_wasserstein_significance.py
│   ├── 06_feature_integration_and_pruning.py
│   ├── 07_deep_learning_regression_with_l0_gates.py
│   ├── 08_evaluate_model_on_holdout_test.py
│   ├── 09_prepare_emerson_external_validation.py
│   ├── 10_transfer_learning_covid_to_emerson.py
│   ├── 11_clinical_ensemble_age_acceleration.py
│   ├── 12_tcr_sequence_classification.py
│   ├── 13_epitope_database_harmonization.py
│   └── 14_tcr_epitope_age_alignment.py
│
└── supplementary/                    # Supplementary materials
    ├── figures/
    │   ├── Supplementary Figure S1.pdf
    │   ├── Supplementary Figure S2.pdf
    │   ├── Supplementary Figure S3.pdf
    │   └── Supplementary Figure S4.pdf
    └── tables/
        ├── Supplementary Table 1.xlsx
        ├── Supplementary Table 2.xlsx
        ├── Supplementary Table 3.xlsx
        └── Supplementary Table 4.xlsx
```

## 🔬 Core Pipeline Modules (`src/`)

The analysis is driven by 14 sequential Python scripts, structured into functional blocks:

### Phase 1: Data Preparation & Feature Extraction (01-03)
* **`01`**: Performs rigorous QC, discarding samples below 200,000 templates. Applies multinomial downsampling to standardize depth, followed by a randomized 75/25 training/testing split.
* **`02`**: Evaluates TCR prevalence across the training cohort, isolating "Public TCRs" via a strict 5% frequency threshold to filter out sample-specific noise.
* **`03`**: Utilizes parallel processing to extract unique K-mer features from repertoires and integrates them with patient metadata.

### Phase 2: Age Scoring & Feature Integration (04-06)
* **`04`**: Maps highly prevalent public TCRs to patient age distributions to create the baseline for downstream scoring.
* **`05`**: Computes age-association using an optimized, signed Wasserstein distance metric. Fits a two-component Gaussian Mixture Model (GMM) to calculate component-specific Z-scores, isolating highly significant aging clonotypes.
* **`06`**: Merges Wasserstein and K-mer matrices. Applies rigorous multi-stage filtering: prevalence pruning (<5%), strict correlation pruning (|r| >= 0.90), variance filtering, and H2O AutoML importance selection.

### Phase 3: Deep Learning Regression (07-08)
* **`07`**: Implements the core Multi-Layer Perceptron (MLP) augmented with Hard-Concrete Feature Gates (L0 Regularization) for simultaneous age prediction and interpretable feature selection. 
* **`08`**: Evaluates the optimized MLP architecture on the strict 25% holdout TEST cohort. Assesses final performance metrics and biological sex stratification to ensure fairness and prevent data leakage.

### Phase 4: External Validation & Clinical Application (09-11)
* **`09`**: Aligns the feature space of the external Emerson cohort to the discovery dataset, handling missing features via zero-imputation.
* **`10`**: Executes a transfer learning pipeline. Fine-tunes the optimal pre-trained MLP on the Emerson dataset using a progressive unfreezing strategy.
* **`11`**: Establishes a pure immunological aging baseline by training an ensemble on healthy individuals. Projects predictions onto clinical condition cohorts to calculate "Age Acceleration".

### Phase 5: Sequence Classification & Epitope Alignment (12-14)
* **`12`**: Encodes TCR sequences using Atchley Factors. Uses H2O AutoML on strictly balanced cohorts to validate physical-chemical separation between age-biased sequences.
* **`13`**: Constructs the definitive reference database by merging VDJdb, McPAS-TCR, and TRAIT, harmonizing highly variable antigen nomenclature.
* **`14`**: Bridges the model's age predictions with epitope specificities by merging Signed Wasserstein scores with the unified reference database.

## 🚀 Getting Started

### Prerequisites & Installation
* Python 3.8+
* High-memory computing environment (Required for massive repertoire matrices)
* PyTorch (for Deep Learning modules)

```bash
git clone https://github.com/elihaylevi/T_TIME.git
cd T_TIME
pip install -r requirements.txt
```

### Reproducing the Machine Learning Core
The pre-processed, fully pruned feature matrices representing 95% cumulative importance are available in the `data/` directory. **Ensure all commands are run from the root directory of the repository.**

**1. Train the L0-gated MLP Model**
```bash
python src/07_deep_learning_regression_with_l0_gates.py
```

**2. Evaluate the Holdout Test Set**
```bash
python src/08_evaluate_model_on_holdout_test.py
```

## 📊 Data & Supplementary Materials
* **Model Datasets:** Cleaned matrices (`train_combined_matrix_pruned_95.csv`, `test_combined_matrix_pruned_95.csv`), output predictions (`test_preds.csv`), and significant TCR Wasserstein scores are located in the `data/` directory.
* **Visualizations:** The underlying code for generating the manuscript figures is available in the `scripts/` directory, and the final vectorized PDFs are located in `results/`.
* **Supplementary Information:** All supplementary figures (S1-S4) and tables (1-4) accompanying the manuscript are accessible in the `supplementary/` directory. *(Note: Raw sequencing files are available in public repositories as outlined in the manuscript's Data Availability statement).*

## ✍️ Author
**Elihai Levi**
PhD Candidate in Computational Biology
Efroni Lab, Bar-Ilan University