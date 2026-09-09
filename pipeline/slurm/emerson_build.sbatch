#!/bin/bash
#SBATCH --job-name=ttime_ext_build
#SBATCH --output=logs/ttime_ext_build_%j.out
#SBATCH --error=logs/ttime_ext_build_%j.err
#SBATCH --partition=cpu1T-24h
#SBATCH --cpus-per-task=48
#SBATCH --mem=250G

# TTIME_WORK must point at the analysis workspace (see pipeline/config.py)
: "${TTIME_WORK:?set TTIME_WORK to the workspace root}"
export PYTHONHASHSEED=0          # required for reproducible downsampling
mkdir -p logs
python3 "$(dirname "$0")"/../emerson_build.py
