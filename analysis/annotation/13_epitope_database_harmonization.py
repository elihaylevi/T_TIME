"""
Module: 13_epitope_database_harmonization

Description:
This script builds the definitive, unified TCR-Epitope reference database for the study.
It processes three major public repositories (VDJdb, McPAS-TCR, TRAIT), focusing strictly 
on TCR-beta chains. 

Processing Steps:
1. Data Ingestion: Loads and standardizes column schemas across all three databases.
2. TRB Filtering: Isolates TCR-beta chains and drops missing values.
3. "No-Joker" CDR3 Cleaning: Removes non-functional sequences, resolving stop codons (*) 
   and unknown amino acids (X), keeping only standard uppercase sequences.
4. Annotation Harmonization: Applies a rigorous clustering dictionary to normalize highly 
   variable gene and species nomenclature (e.g., collapsing SARS-CoV-2 variants, mapping 
   "polymerase acidic" to "Pa").
5. Deduplication: Removes exact redundancies across the merged dataset to produce the 
   final 'adv_unique_nojoker.csv' reference file.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def _find(name):
    """C5 (2026-09-08): resolve inputs from anywhere, like analysis/revision/*.

    Inputs are searched for; OUTPUTS stay relative to the current directory, so a
    verification run in a scratch dir cannot write into the repository.
    """
    for d in [Path('.'), REPO / 'data' / 'external' / 'reference_db',
              REPO / 'data' / 'external', REPO / 'data', REPO,
              Path(__file__).resolve().parent,
              Path('/content'), Path('/content/data')]:
        p = d / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(name)

# ==========================================
# 1. PATHS & CONFIGURATION
# ==========================================
TRAIT_XLSX = _find('20250312-TRAIT_search_download.xlsx')
VDJDB_TSV  = _find('vdjdb.slim.txt')
MCPAS_CSV  = _find('McPAS-TCR.csv')

OUTPUT_CSV = 'adv_unique_nojoker.csv'

# ==========================================
# 2. HARMONIZATION DICTIONARIES
# ==========================================
def norm_compact(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

SPECIES_ALIASES = {
    'CMV': ["cmv", "cytomegaloviruscmv", "cytomegalovirus"],
    'InfluenzaA': ["influenzaa", "influenza"],
    'Influenza B': ["influenzab", "influenzabvirus"],
    'EBV': ["epsteinbarrvirusebv", "ebv", "epsteinbarrvirus"],
    'SARS-CoV-2': ["sarscov2", "covid19"],
    'M. tuberculosis': ["mtuberculosis", "mycobacteriumtuberculosis"],
    'HIV': ["hiv1", "hiv", "humanimmunodeficiencyvirushiv"],
    'HCV': ["hepatitiscvirus", "hepatitiscvirushcv", "hcv"],
    'YFV': ["yfv", "yellowfevervirus"],
    'Cancer/Self': ["tumorassociatedantigentaa", "tumor", "melanoma"],
    # VDJdb and TRAIT both spell the human self-antigen bucket "HomoSapiens" (no space);
    # TRAIT additionally carries "Homo Sapiens" / "Homo sapiens". All three normalise to
    # 'homosapiens'. Collapsing them onto the reference vocabulary's
    # "Human (Homo sapiens)" is what lets Figure4_Landscape_Analysis.py's Human filter
    # see them; without this they survive into the top-9 species and raise
    # KeyError: 'HomoSapiens' on category_map.
    'Human (Homo sapiens)': ["homosapiens", "humanhomosapiens"]
}

GENE_ALIASES = {
    'M': ["m", "matrixproteinm1", "m1", "matrix"],
    'Gag': ["gag", "gagpolyproteinrq13", "gagp24", "gagpolyprotein", "gagpolpolyprotein", "gagprotein"],
    'NP': ["np177", "np", "nucleoprotein"],
    'Nef': ["nef", "proteinnef", "rf10proteinnef"],
    'MART-1': ["melanamart1", "mart1"],
    'Tax-1': ["tax1", "tax"],
    'Vpr': ["vpr", "proteinvpr"],
    'Pa': ["pa", "polymeraseacidic", "polymeraseacidicprotein"],
    'Imp2/a': ["imp2a", "imp2", "lmp2a", "lmp2"]
}

def apply_mapping(token, mapping_dict):
    if pd.isna(token): return token
    compact_token = norm_compact(token)
    for canonical, variants in mapping_dict.items():
        if compact_token in variants:
            return canonical
    # Special regex rules from the notebook
    if re.search(r'polymerase\s*acidic', str(token), re.I): return 'Pa'
    return token

# ==========================================
# 3. DATABASE INGESTION & FILTERING
# ==========================================
def load_and_standardize_vdjdb():
    print("Processing VDJdb...")
    df = pd.read_csv(VDJDB_TSV, sep='\t', dtype=str, low_memory=False)
    # Keep TRB only
    df = df[df['gene'].astype(str).str.upper().str.startswith('TRB')].copy()
    out = df[['cdr3', 'antigen.gene', 'antigen.species']].rename(
        columns={'cdr3': 'cdr3b', 'antigen.gene': 'Epitope_gene', 'antigen.species': 'Epitope_species'}
    )
    out['source'] = 'VDJdb'
    return out.dropna(subset=['cdr3b'])

def load_and_standardize_mcpas():
    print("Processing McPAS-TCR...")
    df = pd.read_csv(MCPAS_CSV, dtype=str, low_memory=False)
    out = df[['CDR3.beta.aa', 'Antigen.protein', 'Pathology']].rename(
        columns={'CDR3.beta.aa': 'cdr3b', 'Antigen.protein': 'Epitope_gene', 'Pathology': 'Epitope_species'}
    )
    out['source'] = 'McPAS'
    return out.dropna(subset=['cdr3b'])

def load_and_standardize_trait():
    print("Processing TRAIT...")
    df = pd.read_excel(TRAIT_XLSX, dtype=str)
    # TRAIT has various CDR3b column names
    cdr3_col = next(c for c in ['CDR3β', 'CDR3b', 'CDR3_beta', 'CDR3B'] if c in df.columns)
    out = df[[cdr3_col, 'Epitope_gene', 'Epitope_species']].rename(
        columns={cdr3_col: 'cdr3b'}
    )
    out['source'] = 'TRAIT'
    return out.dropna(subset=['cdr3b'])

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def main():
    # 1. Merge datasets
    df_v = load_and_standardize_vdjdb()
    df_m = load_and_standardize_mcpas()
    df_t = load_and_standardize_trait()
    
    merged = pd.concat([df_v, df_m, df_t], ignore_index=True)
    print(f"\nTotal raw TRB sequences merged: {len(merged):,}")

    # 2. "No Joker" Cleaning (Functional CDR3s only)
    print("Applying 'No-Joker' filter (removing non-standard AA, stops, and gaps)...")
    # Clean spaces and force uppercase
    merged['cdr3b'] = merged['cdr3b'].astype(str).str.strip().str.upper()
    # Drop rows containing *, X, or any non-standard amino acid letter
    joker_mask = merged['cdr3b'].str.contains(r'[^ACDEFGHIKLMNPQRSTVWY]')
    merged = merged[~joker_mask].copy()
    print(f"Sequences remaining after No-Joker filter: {len(merged):,}")

    # 3. Annotation Harmonization
    print("Harmonizing Gene and Species nomenclature...")
    merged['Epitope_species'] = merged['Epitope_species'].apply(lambda x: apply_mapping(x, SPECIES_ALIASES))
    merged['Epitope_gene'] = merged['Epitope_gene'].apply(lambda x: apply_mapping(x, GENE_ALIASES))

    # Clean multi-value separator formatting (e.g. "A,B" -> "A | B")
    for col in ['Epitope_species', 'Epitope_gene']:
        merged[col] = merged[col].astype(str).str.replace(r'\s*[,;]\s*', ' | ', regex=True)
        # Convert "nan" strings back to real NaNs
        merged.loc[merged[col].str.lower().isin(['nan', 'none', '']), col] = np.nan

    # 4. Deduplication
    print("Performing NA-aware deduplication...")
    # Drop exact complete duplicates
    dedup = merged.drop_duplicates(subset=['cdr3b', 'Epitope_gene', 'Epitope_species'], keep='first')
    
    # Add the '_adv' aliases to signify these columns passed the advanced fixing.
    # NOTE: these are ADDED, not renamed. Script 14 reads the plain 'Epitope_gene' /
    # 'Epitope_species' names, and the reference adv_unique_nojoker.csv on disk carries
    # both families. Renaming here (the previous behaviour) broke the 13 -> 14 chain with
    # KeyError: 'Epitope_species'.
    dedup = dedup.copy()
    dedup['Epitope_gene_adv'] = dedup['Epitope_gene']
    dedup['Epitope_species_adv'] = dedup['Epitope_species']
    
    # Save the final pristine database
    dedup.to_csv(OUTPUT_CSV, index=False)
    print(f"\nPipeline complete! Unified reference database saved to: {OUTPUT_CSV}")
    print(f"Final dataset size: {len(dedup):,} unique entries.")

if __name__ == "__main__":
    main()