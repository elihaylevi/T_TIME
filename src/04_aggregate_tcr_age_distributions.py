"""
Module: TCR Age Distribution Aggregator

Description:
Aggregates patient ages for highly prevalent (public) TCRs across the training cohort.
Generates an optimized mapping of each TCR to a list of ages (representing its occurrence distribution),
which serves as the input for downstream Wasserstein distance scoring.
"""

import os
import pandas as pd
from collections import defaultdict
from pathlib import Path

# ==========================================
# 1. Path Configurations
# ==========================================
# Filtered public TCRs file (e.g., >5% prevalence across samples):
tcr_filtered_file = '/dsi/scratch/home/dsi/elihay/downsampled_files/outputs/tcr_counts_filtered.csv'

# Directory containing training samples:
train_dir = Path('/dsi/scratch/home/dsi/elihay/downsampled_files/train')

# Patient metadata file:
metadata_file = '/dsi/scratch/home/dsi/elihay/Matched_File_Data.xlsx'

# Output file expected by the downstream Wasserstein scoring script:
output_file = 'updated_tcr_age_lists_with_scores.csv' 


# ==========================================
# 2. Load Base Data & Initialize Dictionaries
# ==========================================
print("Loading public TCRs...")
# Load only relevant TCRs to optimize memory and processing speed
public_tcrs_df = pd.read_csv(tcr_filtered_file)
valid_tcrs = set(public_tcrs_df['amino_acid'].dropna())
print(f"Loaded {len(valid_tcrs)} valid public TCRs.")

print("Loading patient metadata...")
meta_df = pd.read_excel(metadata_file)
# Clean '.tsv' extensions from sample names to ensure proper merging
meta_df['sample name'] = meta_df['sample name'].astype(str).str.replace('.tsv', '', regex=False)

# Create a fast O(1) lookup dictionary mapping sample names to patient ages
sample_to_age = dict(zip(meta_df['sample name'], meta_df['Age']))


# ==========================================
# 3. Map Ages to Each Public TCR
# ==========================================
# Automatically initializes an empty list for new TCR entries
tcr_to_ages = defaultdict(list)

print("Gathering ages for public TCRs across all samples...")
# Iterate through files (fast execution via single-column loading)
for file_path in train_dir.glob('*.tsv'):
    sample_name = file_path.stem
    
    # Skip samples lacking age metadata
    if sample_name not in sample_to_age or pd.isna(sample_to_age[sample_name]):
        continue
        
    sample_age = sample_to_age[sample_name]
    
    # Load only the amino acid sequence column from the current sample
    df = pd.read_csv(file_path, sep='\t', usecols=['amino_acid'])
    sample_tcrs = set(df['amino_acid'].dropna())
    
    # Optimized intersection: Keep only public TCRs present in this specific sample
    relevant_tcrs = sample_tcrs.intersection(valid_tcrs)
    
    # Append the patient's age to the respective TCR distribution lists
    for tcr in relevant_tcrs:
        tcr_to_ages[tcr].append(sample_age)


# ==========================================
# 4. Construct Final DataFrame and Export
# ==========================================
print("Building final dataframe...")
# Convert dictionary to DataFrame with the exact column names required downstream
final_df = pd.DataFrame([
    {'TCR': tcr, 'Ages': ages} 
    for tcr, ages in tcr_to_ages.items()
])

final_df.to_csv(output_file, index=False)
print(f"Done! Output saved to {output_file}.")
print("You can now safely run the Wasserstein scoring script on this file!")