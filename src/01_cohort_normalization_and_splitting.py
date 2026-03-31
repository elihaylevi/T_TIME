"""
Module: Data Preparation - QC, Downsampling, and Train/Test Split

Description:
Phase 1: Performs quality control on raw TCR repertoire files, excluding samples below 
the target depth (200,000 templates). Applies multinomial downsampling to standardize depth.
Phase 2: Randomly shuffles the normalized cohort and splits it into training (75%) and 
testing (25%) sets, moving the files to their respective dedicated directories.
"""

import os
import shutil
import pandas as pd
import numpy as np
from multiprocessing import Pool
from pathlib import Path

# ==========================================
# 1. Configuration & Paths
# ==========================================
# Raw data input directory
RAW_DATA_DIR = Path('/dsi/scratch/home/dsi/elihay/DATA')

# Intermediate directory for successfully downsampled files
DOWNSAMPLED_DIR = Path('/dsi/scratch/home/dsi/elihay/downsampled_files')
DOWNSAMPLED_DIR.mkdir(parents=True, exist_ok=True)

# Final output directories for the split
TRAIN_DIR = Path('/dsi/scratch/home/dsi/elihay/train')
TEST_DIR = Path('/dsi/scratch/home/dsi/elihay/test')
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
TARGET_DEPTH = 200000
TRAIN_RATIO = 0.75
RANDOM_SEED = 42

# ==========================================
# 2. Phase 1: Quality Control & Downsampling
# ==========================================
def process_and_downsample(file_path):
    """
    Reads a raw repertoire, enforces depth threshold, and performs multinomial downsampling.
    """
    try:
        df = pd.read_csv(file_path, sep='\t', usecols=['amino_acid', 'templates'])
        df = df.dropna(subset=['amino_acid', 'templates'])
        df = df[~df['amino_acid'].str.contains(r'\*', na=False)]
        
        df['templates'] = df['templates'].astype(int)
        total_templates = df['templates'].sum()
        
        # QC Filter
        if total_templates < TARGET_DEPTH:
            return f"EXCLUDED", file_path.name
            
        # Fast Multinomial Downsampling
        probabilities = df['templates'].values / total_templates
        
        # Local random state to avoid multiprocessing seed conflicts
        local_rng = np.random.default_rng(seed=abs(hash(file_path.name)) % (10**8))
        downsampled_counts = local_rng.multinomial(TARGET_DEPTH, probabilities)
        
        survival_mask = downsampled_counts > 0
        normalized_df = pd.DataFrame({
            'amino_acid': df['amino_acid'].values[survival_mask],
            'templates': downsampled_counts[survival_mask]
        })
        
        # Save to intermediate downsampled directory
        out_path = DOWNSAMPLED_DIR / file_path.name
        normalized_df.to_csv(out_path, sep='\t', index=False)
        
        return "SUCCESS", file_path.name
        
    except Exception as e:
        return "ERROR", file_path.name


# ==========================================
# 3. Main Execution Pipeline
# ==========================================
if __name__ == '__main__':
    print(f"--- PHASE 1: Repertoire QC & Downsampling (Target: {TARGET_DEPTH}) ---")
    
    raw_files = list(RAW_DATA_DIR.glob('*.tsv'))
    print(f"Found {len(raw_files)} raw files in {RAW_DATA_DIR}.")
    
    valid_files = []
    
    # Process in parallel
    with Pool(processes=60) as pool:
        for status, filename in pool.imap_unordered(process_and_downsample, raw_files):
            if status == "SUCCESS":
                valid_files.append(filename)
                
    print(f"QC Complete. {len(valid_files)} samples successfully normalized.")
    
    # ==========================================
    # 4. Phase 2: Train-Test Split
    # ==========================================
    if len(valid_files) == 0:
        print("No files passed the QC threshold. Exiting.")
        exit()

    print(f"\n--- PHASE 2: Train/Test Split ({TRAIN_RATIO*100:.0f}% / {(1-TRAIN_RATIO)*100:.0f}%) ---")
    
    # Sort files for deterministic behavior before shuffling
    valid_files.sort()
    
    # Set global seed and shuffle
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(valid_files)
    
    split_index = int(TRAIN_RATIO * len(valid_files))
    train_files = valid_files[:split_index]
    test_files = valid_files[split_index:]
    
    print("Moving files to Train and Test directories...")
    
    for file_name in train_files:
        src = DOWNSAMPLED_DIR / file_name
        dest = TRAIN_DIR / file_name
        shutil.move(src, dest)
        
    for file_name in test_files:
        src = DOWNSAMPLED_DIR / file_name
        dest = TEST_DIR / file_name
        shutil.move(src, dest)
        
    print("\n" + "="*50)
    print("PIPELINE 01 COMPLETE")
    print("="*50)
    print(f"Training set: {len(train_files)} samples -> {TRAIN_DIR}")
    print(f"Testing set:  {len(test_files)} samples -> {TEST_DIR}")
    print("="*50)