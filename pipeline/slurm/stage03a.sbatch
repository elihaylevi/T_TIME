#!/bin/bash
#SBATCH --job-name=ttime_kmers
#SBATCH --output=logs/ttime_kmers_%j.out
#SBATCH --error=logs/ttime_kmers_%j.err
#SBATCH --partition=cpu1T-24h
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G

# TTIME_WORK must point at the analysis workspace (see pipeline/config.py)
: "${TTIME_WORK:?set TTIME_WORK to the workspace root}"
export PYTHONHASHSEED=0          # required for reproducible downsampling
mkdir -p logs
python3 "$(dirname "$0")"/../stage03a_kmers.py
