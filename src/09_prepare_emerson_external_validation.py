"""
Module: 09_prepare_emerson_external_validation

Description:
Prepares the Emerson external validation dataset by aligning its feature space 
with the COVID-19 discovery cohort. Merges Emerson TCR (Wasserstein) and K-mer counts, 
standardizes metadata, and enforces strict feature ordering. Missing features 
are imputed as zero to maintain matrix compatibility for transfer learning.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# PATHS
# =========================
TRAIN_REF_PATH = Path("/home/dsi/levieli8/15kfeatures_to_prediction/train_combined_matrix_pruned_95.csv")
OUT_PATH       = Path("/home/dsi/levieli8/15kfeatures_to_prediction/emerson_combined_matrix_aligned.csv")

# Raw Emerson files from scratch
EMERSON_TCR_FILES  = [
    "/home/dsi/levieli8/scratch/emerson/merged_emerson_train_significant_tcrs_signed_wasserstein.csv",
    "/home/dsi/levieli8/scratch/emerson/merged_emerson_test_significant_tcrs_signed_wasserstein.csv"
]
EMERSON_KMER_FILES = [
    "/home/dsi/levieli8/scratch/emerson/processed/merged_unique_kmers_train.csv",
    "/home/dsi/levieli8/scratch/emerson/processed/merged_unique_kmers_test.csv"
]

def normalize_sample_col(df):
    for col in df.columns:
        if col.strip().lower() in ["sample", "sample name", "sample_name"]:
            return df.rename(columns={col: "sample name"})
    return df

def main():
    print("Loading COVID train reference for feature alignment...")
    ref_df = pd.read_csv(TRAIN_REF_PATH)
    meta_cols = ["sample name", "Age", "Biological Sex"]
    target_features = [c for c in ref_df.columns if c not in meta_cols]

    print("Merging Emerson cohort files...")
    tcr_df  = pd.concat([pd.read_csv(f) for f in EMERSON_TCR_FILES], ignore_index=True)
    kmer_df = pd.concat([pd.read_csv(f) for f in EMERSON_KMER_FILES], ignore_index=True)
    
    tcr_df, kmer_df = normalize_sample_col(tcr_df), normalize_sample_col(kmer_df)
    merged = pd.merge(tcr_df, kmer_df, on="sample name", how="outer")

    # Add missing metadata placeholders
    for col in ["Age", "Biological Sex"]:
        if col not in merged.columns: merged[col] = np.nan

    # --- Strict Feature Alignment ---
    print(f"Aligning to {len(target_features)} features...")
    present_feats = [f for f in target_features if f in merged.columns]
    missing_feats = [f for f in target_features if f not in merged.columns]
    
    # Add missing as 0, drop extras not in original model
    for f in missing_feats: merged[f] = 0.0
    merged = merged[meta_cols + target_features]

    # Clean numeric types
    for c in target_features:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0).astype(np.float32)

    print(f"Final Emerson Matrix: {merged.shape[0]} samples | {len(target_features)} features")
    merged.to_csv(OUT_PATH, index=False)
    print(f"Saved aligned matrix to: {OUT_PATH}")

if __name__ == "__main__":
    main()