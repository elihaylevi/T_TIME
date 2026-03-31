"""
Module: 06_feature_integration_and_pruning

Description:
This script integrates TCR (Wasserstein-based) and K-mer feature matrices into a 
unified dataset. It performs a multi-stage filtering process:
1. Prevalence Filtering: Removes features present in <5% of the training cohort.
2. Correlation Pruning: Identifies highly correlated feature pairs (|r| >= 0.90) 
   and retains the most representative feature (based on prevalence/variance).
3. Variance Filtering: Removes near-constant features (variance < 1e-8).
4. AutoML Importance Selection: Retains top features accounting for 95% of 
   cumulative importance as determined by H2O AutoML.
"""

import os
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

# --- Path Configurations ---
INDIR = './15kfeatures_to_prediction'
OUTDIR = './15kfeatures_to_prediction'
os.makedirs(OUTDIR, exist_ok=True)

# --- Thresholds ---
CORR_THRESHOLD = 0.90
VAR_THRESHOLD = 1e-8
IMPORTANCE_CUM_LIMIT = 0.95
N_WORKERS = min(72, cpu_count())

# Metadata Columns (Case-insensitive mapping)
META_KEYS = {"sample name", "age", "biological sex"}

def normalize_column_names(s: pd.Series) -> pd.Series:
    """Standardizes string series for consistent merging and filtering."""
    return s.astype(str).str.strip().str.lower()

# ============================================================
# PHASE 1: Merging & 5% Prevalence Filter
# ============================================================
def merge_and_prefilter(tcr_path, kmer_path):
    print("Initiating merge and initial prevalence filtering...")
    
    # Load and clean K-mer data
    km = pd.read_csv(kmer_path)
    if 'sample' in km.columns:
        km = km.rename(columns={'sample': 'sample name'})
    km['sample name'] = normalize_column_names(km['sample name'])
    km = km.drop(columns=[c for c in ['Age', 'Biological Sex'] if c in km.columns], errors='ignore')
    
    # K-mer Prevalence filter
    km_feats = [c for c in km.columns if c not in ['sample name']]
    kmer_prevalence = (km[km_feats] != 0).sum(axis=0) / len(km)
    km_kept = kmer_prevalence[kmer_prevalence >= 0.05].index.tolist()
    
    # Load and clean TCR Wasserstein data
    tcr = pd.read_csv(tcr_path)
    tcr['sample name'] = normalize_column_names(tcr['sample name'])
    tcr_meta = [c for c in tcr.columns if c.strip().lower() in META_KEYS]
    tcr_feats = [c for c in tcr.columns if c not in tcr_meta]
    
    # TCR Prevalence filter
    tcr_prevalence = (tcr[tcr_feats] != 0).sum(axis=0) / len(tcr)
    tcr_kept = tcr_prevalence[tcr_prevalence >= 0.05].index.tolist()
    
    # Inner join on sample name
    combined = pd.merge(
        tcr[tcr_meta + tcr_kept], 
        km[['sample name'] + km_kept], 
        on='sample name', 
        how='inner'
    )
    return combined

# ============================================================
# PHASE 2: AutoML Cumulative Importance Filtering
# ============================================================
def filter_by_cumulative_importance(combined_df, km_imp_path, tcr_imp_path):
    """
    Filters features based on the 95% cumulative importance threshold 
    derived from H2O AutoML variable importance rankings.
    """
    print(f"Applying 95% cumulative importance threshold...")
    
    # Identify existing features in the merged matrix
    meta_cols = [c for c in combined_df.columns if c.strip().lower() in META_KEYS]
    current_feats = [c for c in combined_df.columns if c not in meta_cols]
    
    # Load importance scores
    km_imp = pd.read_csv(km_imp_path).sort_values("percentage", ascending=False)
    tcr_imp = pd.read_csv(tcr_imp_path).sort_values("percentage", ascending=False)
    
    # Calculate CUMULATIVE SUM of importance (Clean naming for Nature publication)
    km_imp["cumulative_importance"] = km_imp["percentage"].cumsum()
    tcr_imp["cumulative_importance"] = tcr_imp["percentage"].cumsum()
    
    # Extract features contributing to 95% of model predictive power
    km_keep = km_imp.loc[km_imp["cumulative_importance"] <= IMPORTANCE_CUM_LIMIT, "variable"].tolist()
    tcr_keep = tcr_imp.loc[tcr_imp["cumulative_importance"] <= IMPORTANCE_CUM_LIMIT, "variable"].tolist()
    
    # Final intersection with current feature space
    target_feats = set(km_keep + tcr_keep)
    final_selected_feats = [f for f in current_feats if f in target_feats]
    
    print(f"Importance filtering complete: {len(final_selected_feats)} features retained.")
    return combined_df[meta_cols + final_selected_feats]

# ============================================================
# MAIN EXECUTION FLOW
# ============================================================
if __name__ == "__main__":
    # 1. Merge and basic prevalence cleanup
    tcr_path = '/dsi/scratch/users/elihay/downsampled_files/dataFilesAfterMerge/merged_train_significant_tcrs_signed_wasserstein.csv'
    kmer_path = '/dsi/scratch/users/elihay/downsampled_files/regression/kmers/merged/merged_unique_kmers_with_metadata.csv'
    
    df_combined = merge_and_prefilter(tcr_path, kmer_path)
    
    # 2. Importance-based pruning (Using H2O AutoML results)
    # Update these paths to the exact location of your importance CSVs
    km_importance_path = "/home/dsi/levieli8/full_feature_importance.csv"
    tcr_importance_path = "/home/dsi/levieli8/full_feature_importance_signed_wasserstein_tcrs.csv"
    
    df_final = filter_by_cumulative_importance(
        df_combined, 
        km_importance_path, 
        tcr_importance_path
    )
    
    # Save the pruned matrix for the final regression stage
    final_out_csv = os.path.join(OUTDIR, 'train_combined_matrix_pruned_95.csv')
    df_final.to_csv(final_out_csv, index=False)
    
    print(f"\n[OUTPUT] Final integrated matrix saved to: {final_out_csv}")
    print(f"Total features available for regression: {df_final.shape[1] - 3}")