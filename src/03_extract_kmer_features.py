"""
Module: Unique K-mer Feature Extraction and Metadata Integration

Description:
Extracts unique K-mer representations from TCR repertoires using parallel processing.
Integrates the resulting feature matrix with patient metadata for downstream regression modeling.
"""

import os
import pandas as pd
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

# --- Configuration ---
K = 3
# Path to the training samples directory
data_dir = Path('/dsi/scratch/home/dsi/elihay/downsampled_files/train')
# Path to the metadata file
metadata_file = '/dsi/scratch/home/dsi/elihay/Matched_File_Data.xlsx'

# Exact output directory required by the downstream pipeline
output_dir = Path('/dsi/scratch/users/elihay/downsampled_files/regression/kmers/merged/')
output_dir.mkdir(parents=True, exist_ok=True)

# Function to extract k-mers from a sequence
def extract_kmers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

# Function to process a single file (Only Unique K-mers)
def process_file(file_path):
    try:
        # Read only the relevant amino acid column
        df = pd.read_csv(file_path, sep='\t', usecols=['amino_acid'])
        
        # Filter out missing values and invalid sequences (containing asterisks)
        df = df.dropna(subset=['amino_acid'])
        df = df[~df['amino_acid'].str.contains(r'\*', na=False)]
        
        unique_kmer_counts = Counter()

        # Fast iteration over sequences
        for seq in df['amino_acid']:
            k_mers = extract_kmers(seq, K)
            # Unique counting: Each K-mer is counted only once per TCR sequence
            unique_kmer_counts.update(set(k_mers))
                
        # Return the sample name and the populated dictionary
        sample_name = file_path.stem
        return sample_name, dict(unique_kmer_counts)
    
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return None

if __name__ == '__main__':
    print(f"Starting Unique K-mer (K={K}) extraction process...")
    files = list(data_dir.glob('*.tsv'))
    
    results_unique = {}

    # Safe and fast parallel processing
    with Pool(processes=60) as pool:
        for result in pool.imap_unordered(process_file, files):
            if result:
                sample_name, unique_counts = result
                results_unique[sample_name] = unique_counts
                
    print(f"Processed {len(results_unique)} samples. Building DataFrame...")

    # Build DataFrame directly from dictionary (Samples as rows, K-mers as columns)
    df_unique = pd.DataFrame.from_dict(results_unique, orient='index').fillna(0)

    # --- Metadata Integration ---
    print("Merging with metadata...")
    df_metadata = pd.read_excel(metadata_file)
    # Ensure sample names in metadata are clean of file extensions
    df_metadata['sample name'] = df_metadata['sample name'].astype(str).str.replace('.tsv', '', regex=False)

    # Merge based on index (sample name)
    merged_unique = df_unique.merge(df_metadata, left_index=True, right_on='sample name', how='inner')

    # Save the final matrix with the specific name expected by the pipeline
    output_file = output_dir / 'merged_unique_kmers_with_metadata.csv'
    merged_unique.to_csv(output_file, index=False)

    print(f"Success! Final matrix saved to: {output_file}")