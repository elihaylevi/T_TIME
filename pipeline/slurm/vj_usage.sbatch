#!/bin/bash
#SBATCH --job-name=ttime_vj
#SBATCH --output=logs/ttime_vj_%j.out
#SBATCH --error=logs/ttime_vj_%j.err
#SBATCH --partition=cpu1T-24h
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G

# TTIME_WORK must point at the analysis workspace (see pipeline/config.py)
: "${TTIME_WORK:?set TTIME_WORK to the workspace root}"
export PYTHONHASHSEED=0          # required for reproducible downsampling
mkdir -p logs
python3 "$(dirname "$0")"/../vj_usage.py
