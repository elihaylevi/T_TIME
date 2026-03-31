"""
Module: Public TCR Identification and Prevalence Filtering

Description:
Analyzes TCR publicity (prevalence) across the entire training cohort. 
Identifies "Public TCRs"—sequences shared across multiple individuals—and 
applies a 5% frequency threshold to filter out rare or sample-specific 
clonotypes. This ensures that downstream feature extraction (e.g., Wasserstein 
scoring) is performed only on biologically relevant, shared immune signatures.
"""

import os
import pandas as pd
from collections import Counter
import concurrent.futures

# --- Configuration ---
# Path to the downsampled training samples
train_path = '/dsi/scratch/home/dsi/elihay/downsampled_files/train/'
# Path for global project outputs
output_dir = '/dsi/scratch/home/dsi/elihay/downsampled_files/outputs/'
os.makedirs(output_dir, exist_ok=True)

# Parallel processing configuration
num_processes = 64

# List all TCR files passing the initial QC
downsampled_files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith('.tsv')]
num_samples = len(downsampled_files)

# Dynamically calculate the 5% prevalence threshold based on cohort size
min_threshold = int(num_samples * 0.05)
print(f"Total samples detected: {num_samples}")
print(f"Applying 5% prevalence threshold: TCRs must appear in at least {min_threshold} samples.")

def get_unique_tcrs(file_path):
    """
    Reads a repertoire file and returns a set of unique amino acid sequences.
    This identifies the presence (not abundance) of clones for publicity analysis.
    """
    try:
        # Load only the amino_acid column to optimize I/O and memory
        df = pd.read_csv(file_path, delimiter="\t", usecols=['amino_acid'])
        # Return unique sequences as a set for O(1) membership testing and fast Counter updates
        return set(df['amino_acid'].dropna().unique())
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return set()

# Main execution for publicity extraction
if __name__ == '__main__':
    # Initialize Counter to track TCR occurrences across the cohort
    tcr_publicity = Counter()
    
    print("Extracting TCR publicity across all samples in parallel...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Update the global counter with sets from each sample
        # Since each sample provides a set, each TCR is incremented by exactly 1 per sample
        for unique_tcrs_in_sample in executor.map(get_unique_tcrs, downsampled_files):
            tcr_publicity.update(unique_tcrs_in_sample)
            
    # Filter TCRs based on the dynamic prevalence threshold
    public_tcrs = {tcr: count for tcr, count in tcr_publicity.items() if count >= min_threshold}
    
    print(f"Initial unique TCR count: {len(tcr_publicity)}")
    print(f"Filtered Public TCR count: {len(public_tcrs)} (survived >= 5% threshold)")
    
    # Export the identified public TCRs for downstream feature extraction
    filtered_df = pd.DataFrame(public_tcrs.items(), columns=['amino_acid', 'sample_appearance_count'])
    output_file = os.path.join(output_dir, 'tcr_counts_filtered.csv')
    filtered_df.to_csv(output_file, index=False)
    
    print(f"Success! Filtered Public TCR master list saved to: {output_file}")