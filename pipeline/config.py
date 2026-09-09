"""
Central configuration: workspace paths and analysis parameters.

Set TTIME_WORK to the workspace root before running any stage, e.g.

    export TTIME_WORK=/path/to/workspace
"""
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent   # repository root

# --- workspace roots ---
WORK      = Path(os.environ["TTIME_WORK"]) if "TTIME_WORK" in os.environ else Path.cwd() / "workspace"
RAW_ALL   = WORK / "raw_staging" / "all"    # raw repertoire TSVs (or symlinks to them)
# Processed feature matrices / clonotype scores ship in the repo's data/ (override with TTIME_DATA).
DATA      = Path(os.environ.get("TTIME_DATA", _REPO / "data"))
METADATA  = Path(os.environ.get("TTIME_METADATA", _REPO / "metadata" / "sample_metadata.csv"))

# --- cohort data (override via environment; see README for sources) ---
RAW_COVID    = Path(os.environ.get("TTIME_RAW_COVID",    WORK / "raw" / "covid"))
RAW_VO       = Path(os.environ.get("TTIME_RAW_VO",       WORK / "raw" / "vo"))
RAW_EXTERNAL = Path(os.environ.get("TTIME_RAW_EXTERNAL", WORK / "raw" / "external"))
VO_METADATA  = Path(os.environ.get("TTIME_VO_METADATA",  WORK / "raw" / "vo_metadata.txt"))
# C6 (2026-09-08): COVID_TAGS, EXTERNAL_CMV, VDJDB and MCPAS were removed - no script
# ever read them. The epitope databases ship at data/external/reference_db/ and
# analysis/annotation/13 resolves them itself; CMV serostatus is in
# data/emerson_metadata_expanded.csv, read by analysis/revision/08.

# --- pipeline stage dirs ---
DOWNSAMPLED = WORK / "work" / "downsampled"   # QC-passing, depth-normalised repertoires
TRAIN_DIR   = WORK / "work" / "train"         # training split
TEST_DIR    = WORK / "work" / "test"          # held-out test split
OUTPUTS     = WORK / "work" / "outputs"       # feature/scoring outputs and results
FIGURES     = WORK / "results" / "figures"
TABLES      = WORK / "results" / "tables"
for d in (DOWNSAMPLED, OUTPUTS, FIGURES, TABLES):
    d.mkdir(parents=True, exist_ok=True)

# --- analysis parameters ---
TARGET_DEPTH = 200_000    # templates per repertoire after downsampling
TRAIN_RATIO  = 0.75
RANDOM_SEED  = 42
K            = 3          # k-mer length
PREVALENCE   = 0.05       # public-TCR / feature prevalence threshold
