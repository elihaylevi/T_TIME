"""
Module: Signed Wasserstein Distance & GMM Significance Scoring

Description:
Calculates the age-association of public TCRs using an optimized, weighted, signed Wasserstein 
distance metric against a global age baseline. Fits a two-component Gaussian Mixture Model (GMM) 
to stratify young- and old-associated TCR populations, and computes component-specific Z-scores 
to isolate statistically significant clonotypes for downstream modeling.
"""

import pandas as pd
import numpy as np
import ast
from scipy.stats import wasserstein_distance
from sklearn.mixture import GaussianMixture

# ==========================================
# STEP 1: Data Ingestion & Global Age Profiling
# ==========================================
print("Ingesting TCR age distributions...")
df = pd.read_csv("updated_tcr_age_lists_with_scores.csv")
df["Ages"] = df["Ages"].apply(ast.literal_eval)

print("Deriving global age distribution baseline...")
global_ages = np.concatenate(df["Ages"].values)
global_mean = np.mean(global_ages)
global_unique_ages, global_age_counts = np.unique(global_ages, return_counts=True)

# ==========================================
# STEP 2: Optimized Signed Wasserstein Computation
# ==========================================
print("Computing signed Wasserstein distances via weighted probability mass functions...")
def fast_signed_wasserstein(ages):
    if len(ages) == 0:
        return 0
    
    local_mean = np.mean(ages)
    local_unique_ages, local_age_counts = np.unique(ages, return_counts=True)
    
    # Calculate Earth Mover's Distance using weighted frequencies for $O(1)$ scaling
    dist = wasserstein_distance(
        u_values=local_unique_ages, 
        v_values=global_unique_ages, 
        u_weights=local_age_counts, 
        v_weights=global_age_counts
    )
    
    # Apply directionality: negative = young-associated, positive = old-associated
    sign = np.sign(local_mean - global_mean)
    return sign * dist

df["signed_wasserstein"] = df["Ages"].apply(fast_signed_wasserstein)

# ==========================================
# STEP 3: Gaussian Mixture Model (GMM) Clustering
# ==========================================
print("Fitting 2-component Gaussian Mixture Model to Wasserstein scores...")
gmm = GaussianMixture(n_components=2, random_state=42)
signed_vals = df["signed_wasserstein"].values.reshape(-1, 1)

# Fit model and assign cluster components
df["gmm_component"] = gmm.fit_predict(signed_vals)

# Extract component probabilities and resolve age-association vectors
probs = gmm.predict_proba(signed_vals)
means = gmm.means_.flatten()
old_idx = np.argmax(means)
young_idx = np.argmin(means)

df["old_prob"] = probs[:, old_idx]
df["young_prob"] = probs[:, young_idx]

# ==========================================
# STEP 4: Vectorized Z-Score Computation & Thresholding
# ==========================================
print("Calculating component-specific Z-scores via vectorized transformations...")

# Map mean and standard deviation directly to the DataFrame space
grouped = df.groupby("gmm_component")["signed_wasserstein"]
df["component_mean"] = grouped.transform("mean")
df["component_std"] = grouped.transform("std")

# Safely compute Z-scores preventing division by zero
df["component_zscore"] = np.where(
    df["component_std"] > 0,
    (df["signed_wasserstein"] - df["component_mean"]) / df["component_std"],
    0
)

# Apply significance threshold (alpha ~ 0.05 equivalent via Z > 1.96)
df["significant"] = df["component_zscore"].abs() > 1.96

sig_count = df["significant"].sum()
print(f"Identified {sig_count} statistically significant TCRs (|Z| > 1.96).")

# ==========================================
# STEP 5: Data Cleansing & Matrix Export
# ==========================================
# Remove intermediary computation columns
df.drop(columns=["component_mean", "component_std"], inplace=True)

output_file = "updated_tcr_age_lists_with_all_significance.csv"
df.to_csv(output_file, index=False)
print(f"Pipeline complete. Significant TCR feature matrix exported to: {output_file}")