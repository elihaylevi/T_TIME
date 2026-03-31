"""
Module: 14_tcr_epitope_age_alignment

Description:
This script integrates the statistical results of the immune age model (Signed Wasserstein scores)
with TCR-epitope specificity metadata. 

Processing Steps:
1. Load the model's output (scores for each unique TCR).
2. Load the TCR-Epitope reference database (VDJdb/McPAS derived).
3. Merge on the CDR3 beta sequence.
4. Data Cleaning: Remove sequences with missing age scores (nonan).
5. Species Simplification: Normalize complex species names and group related antigens 
   (e.g., grouping SARS-CoV-2 variants) to ensure statistical power for visualization.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. PATHS & CONFIGURATION
# ==========================================
# Input: Model results
AGE_SCORES_PATH = Path("updated_tcr_age_lists_with_all_significance.csv")
# Input: Epitope mapping database
EPITOPE_DB_PATH = Path("adv_unique_nojoker.csv")

# Output: Final merged file for Figure 4
OUTPUT_PATH = Path("adv_unique_nojoker_plus_signed_wasserstein_nonan_simplified_merged.csv")

# ==========================================
# 2. DATA LOADING & MERGING
# ==========================================
def main():
    print("Loading datasets...")
    # Load model results (Score per TCR)
    df_scores = pd.read_csv(AGE_SCORES_PATH)
    # Load Epitope DB (Metadata per TCR)
    df_epitopes = pd.read_csv(EPITOPE_DB_PATH)

    # Standardize column names for merging
    # Model uses 'TCR', Epitope DB uses 'cdr3b'
    df_scores = df_scores.rename(columns={'TCR': 'cdr3b'})

    print(f"Merging model scores with epitope metadata...")
    # Inner join to keep only TCRs that have both a score and a known epitope
    merged = pd.merge(df_epitopes, df_scores[['cdr3b', 'signed_wasserstein']], 
                      on='cdr3b', how='inner')

    # ==========================================
    # 3. CLEANING & SIMPLIFICATION (The "nonan_simplified" part)
    # ==========================================
    print("Cleaning and simplifying species names...")

    # A. Remove NaNs in the metric column
    merged = merged.dropna(subset=['signed_wasserstein'])

    # B. Species Normalization Mapping
    # Grouping strains and cleaning up long nomenclature
    species_map = {
        'Cytomegalovirus (CMV)': 'CMV',
        'InfluenzaA': 'Influenza',
        'Influenza A': 'Influenza',
        'EBV': 'EBV',
        'Epstein-Barr virus (EBV)': 'EBV',
        'COVID-19': 'SARS-CoV-2',
        'SARS-CoV-2': 'SARS-CoV-2',
        'Severe acute respiratory syndrome coronavirus 2': 'SARS-CoV-2',
        'Mycobacterium tuberculosis': 'M. tuberculosis',
        'Human immunodeficiency virus 1': 'HIV',
        'HIV-1': 'HIV'
    }

    # Apply mapping to a new normalized column
    merged['Epitope_species_norm'] = merged['Epitope_species'].replace(species_map)

    # C. Simplify Gene Names (The "merged" part of the filename)
    # Many databases have redundant gene names (e.g., 'pp65', 'UL83')
    # We ensure consistent naming for the intra-species analysis (Panel E)
    gene_map = {
        'pp65': 'pp65 (UL83)',
        'UL83': 'pp65 (UL83)',
        'M1': 'M1 (Matrix)',
        'NP': 'NP (Nucleoprotein)'
    }
    merged['Epitope_gene_merged'] = merged['Epitope_gene'].replace(gene_map)

    # ==========================================
    # 4. FINAL EXPORT
    # ==========================================
    print(f"Final dataset size: {merged.shape[0]} sequences.")
    print(f"Unique species covered: {merged['Epitope_species_norm'].nunique()}")
    
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Successfully saved merged landscape data to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()