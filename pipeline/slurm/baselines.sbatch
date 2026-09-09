#!/bin/bash
#SBATCH --job-name=ttime_baselines
#SBATCH --output=logs/ttime_baselines_%j.out
#SBATCH --error=logs/ttime_baselines_%j.err
#SBATCH --partition=cpu1T-24h
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# TTIME_WORK must point at the analysis workspace (see pipeline/config.py)
: "${TTIME_WORK:?set TTIME_WORK to the workspace root}"
export PYTHONHASHSEED=0          # required for reproducible downsampling
mkdir -p logs
python3 "$(dirname "$0")"/../baselines.py
