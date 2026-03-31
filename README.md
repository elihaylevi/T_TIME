# T-TIME: TCR-based Immunological Age Prediction

### *Continuous Framework for Quantifying Immune Aging from Repertoires*

[](https://www.python.org/downloads/)
[](https://opensource.org/licenses/MIT)

**T-TIME** (TCR-based Temporal Immune MEtric) is a computational framework designed to predict immunological age and detect accelerated aging (AAR) using high-dimensional TCR repertoire data. By leveraging a novel "TCR Weight" metric based on Wasserstein distances and Earth Mover's Distance (EMD), the pipeline maps stochastic immune signals into a continuous biological clock.

-----

## 🔬 Core Methodology

Unlike traditional discrete classification, T-TIME implements a **continuous aging pipeline**:

1.  **Repertoire Weighting:** Quantifying the "age-signal" of individual TCR sequences using Signed Wasserstein scores.
2.  **GMM Selection:** Statistical filtering of high-confidence aging biomarkers.
3.  **L0-Regularized Regression:** Deep Learning architecture with L0 gates for sparse, interpretable feature selection.
4.  **Clinical Validation:** Detection of accelerated aging in COVID-19, Hypertension, and Autoimmune cohorts.

-----

## 📂 Repository Structure

The project is organized into a modular pipeline for maximum reproducibility:

```text
T_TIME/
├── src/                      # Sequential Analysis Pipeline
│   ├── 01-03_Data_Prep/      # Normalization, K-mer extraction & Public TCRs
│   ├── 04-06_Scoring/        # Wasserstein scoring and TCR-Weight integration
│   ├── 07-09_Modeling/       # Deep Learning training (L0) and Validation
│   └── 10-14_Clinical/       # COVID transfer learning & Epitope alignment
│
├── scripts/                  # Visualization (Nature-ready Figures)
│   ├── Figure2_Clinical.py   # Model performance & Accelerated Aging
│   ├── Figure3_Pipeline.py   # Continuous framework & GMM Selection
│   └── Figure4_Landscape.py  # Antigenic resolution & MDS analysis
│
├── data/                     # Data directory (Raw CSVs excluded via .gitignore)
└── results/                  # Final vector PDFs and statistical exports
```

-----

## 🚀 Getting Started

### Prerequisites

  * Python 3.8 or higher
  * PyTorch (for Deep Learning modules)
  * High-memory environment (recommended for large repertoire matrices)

### Installation

```bash
git clone https://github.com/elihaylevi/T_TIME.git
cd T_TIME
pip install -r requirements.txt
```

### Reproducing Figures

To generate the finalized manuscript figures (stored in `/results`), run:

```bash
python scripts/Fig3_TCR_Age_Pipeline_Metrics.py
```

-----

## 📊 Key Findings

  * **Systematic Accuracy:** T-TIME achieves high correlation with chronological age across independent cohorts (Primary & Emerson/Adaptive).
  * **Clinical Biomarkers:** Identification of specific TCR clusters associated with accelerated aging in COVID-19 patients and individuals with hypertension.
  * **Antigenic Resolution:** Mapping age-biased TCRs to specific viral (CMV, EBV, SARS-CoV-2) and self-antigenic genes.

-----

## ✍️ Author

**Elihai Levi** *PhD Candidate in Computational Biology* *Bar-Ilan University* 
