"""
15_merged_overlap_extraction.py

Regenerates **`data/external/merged_overlap_tcrs_wasserstein.csv`** — the age-scored
clonotype table joined to VDJdb/TRAIT epitope annotations, carrying the MHC columns.

VERIFIED (2026-09-09). Byte-for-byte identical to the shipped file:

    17,982 rows x 38 columns   53,751,400 bytes
    sha256 b5794c13e018a8085a36608f4e0871a5b90fd50a1346a80b5df2b71d5f149c44

Extracted from `notebooks/tcr_correlations.ipynb`, which was archived out of this
repository in the same change (see *Provenance* below). The notebook is 389 cells; this
chain is four of them, and the four are trivial:

    cell 183   read both inputs, rename CDR3b -> TCR, inner-merge on TCR,
               write merged_tcr_vdjdb.csv          (a 76.5 MB intermediate)
    cell 186   read merged_tcr_vdjdb.csv back
    cell 187   df = df.drop_duplicates()
    cell 189   write merged_overlap_tcrs_wasserstein.csv

Cells 184, 185, 188 and 190 sit inside that range but are bare display expressions
(`df_vdj`, `merged`, `df`, `df`) — Jupyter echo, no effect. Nothing else in the notebook
feeds this file: every other cell that names it only *reads* it (31 plotting cells,
191-229) or belongs to the dead-end recipe below.

THE INTERMEDIATE ROUND-TRIP IS LOAD-BEARING — DO NOT "OPTIMISE" IT AWAY
  It looks like dead weight: cell 183 writes a 76.5 MB `merged_tcr_vdjdb.csv`, cell 186
  reads it straight back, and nothing touches it in between. Keeping the frame in memory
  instead gives the same 17,982 x 38 table with the same values to ~15 significant
  digits — and a **different file**: 342 of the 17,982 rows differ in the last digit of
  one or more float columns, e.g.

      in memory   2.9569444341586397    -1.9608232643200172    9.388716004272949e-11
      round-trip  2.95694443415864      -1.9608232643200167    9.388716004272948e-11

  The cause is pandas' CSV reader, not the writer: its C parser does not round-trip
  floats to nearest. Writing a float and reading it back can therefore land one ULP away
  from where it started, and the shipped file carries those perturbed values because the
  notebook went through disk. Reproducing the artefact means reproducing the round-trip.

  So this script performs it by default and deletes the intermediate afterwards (it is
  deliberately not shipped — `data/external/README.md`, "Regenerating
  merged_tcr_vdjdb.csv"). `--keep-intermediate` leaves it on disk; `--no-round-trip`
  skips it, which is faster and mathematically no worse, but produces a file that is
  **not** byte-identical to the shipped one and will fail `--verify`.

NOT THE PRODUCER: the dead-end recipe
  The notebook holds a second recipe for this same filename (cell 239, immediately after
  a markdown "DEAD END — do not run" warning). It never produced the shipped file:

    - it merges `left_on='TCR', right_on='CDR3b'` without renaming, so it would emit a
      `CDR3b` column; the shipped file has none.
    - it reads the 9-column `significant_tcrs_signed_wasserstein.csv`, whereas the
      shipped file carries all 26 columns of
      `updated_tcr_age_lists_with_all_significance.csv.gz` on the left-hand side.

  38 = 26 (left) + 13 (right) - 1 (CDR3b renamed to TCR and consumed as the join key).
  That arithmetic only works for the 183-chain, which is what settles it.

  (The in-notebook warning cites the chain as "183 -> 186 -> 187 -> 190". The write is
  cell **189**; 190 is the bare `df` after it. Off-by-one in that note only.)

LINEAGE B, NOT LINEAGE A
  This is the *notebook* lineage, and it is the only one that carries `MHC_A`, `MHC_B`
  and `component_zscore`. It is not a variant of `13`/`14`: those build lineage A, which
  merges the same age scores against `adv_unique_nojoker.csv` (TRAIT + VDJdb + McPAS,
  harmonised) and drops MHC. Both lineages are real and are meant to coexist —
  `data/external/README.md` documents the split. No MHC-preserving version of `13` has
  ever existed.

INPUTS  (both shipped)
  - data/updated_tcr_age_lists_with_all_significance.csv.gz   (26 cols; gzipped 2026-09-08)
  - data/external/vdjdb_trait_onlyB.csv                       (13 cols, key `CDR3b`)

OUTPUT
  - merged_overlap_tcrs_wasserstein.csv, in the working directory
    (the shipped copy lives at data/external/merged_overlap_tcrs_wasserstein.csv)

Usage:
    python analysis/annotation/15_merged_overlap_extraction.py
    python analysis/annotation/15_merged_overlap_extraction.py --verify
    python analysis/annotation/15_merged_overlap_extraction.py --keep-intermediate

PROVENANCE
  The source notebook, `tcr_correlations.ipynb` (389 cells, outputs stripped), was moved
  to `~/Downloads/ttime_archive/` on 2026-09-09, alongside `sup.ipynb`. The archive also
  holds `tcr_correlations_WITH_OUTPUTS.ipynb` (117 MB, same 389 cells with outputs
  retained). See that archive's `ARCHIVE_README.md`.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

OUT = "merged_overlap_tcrs_wasserstein.csv"
INTERMEDIATE = "merged_tcr_vdjdb.csv"

# What the shipped file is. Asserted, not assumed - if an input is ever rebuilt, this
# must fail loudly rather than quietly ship a different table.
EXPECTED_ROWS = 17982
EXPECTED_COLS = 38
EXPECTED_SHA256 = "b5794c13e018a8085a36608f4e0871a5b90fd50a1346a80b5df2b71d5f149c44"


def _to_csv_lf(df, path):
    """Write CSV with LF line endings, on any platform and either pandas API.

    The notebook ran on Colab (Linux), so the shipped file is LF. pandas opens the
    output in text mode, so on Windows the default terminator is CRLF - which changes
    every line and makes the result differ from the shipped file byte for byte while
    being identical as data. Forcing LF is what makes this script reproduce the
    artefact rather than merely re-derive it.

    The keyword was renamed in pandas 1.5 (`line_terminator` -> `lineterminator`);
    the pinned environment has 1.4.4, so both spellings are supported.
    """
    kw = ("lineterminator" if pd.__version__ >= "1.5" else "line_terminator")
    try:
        df.to_csv(path, index=False, **{kw: "\n"})
    except TypeError:                       # version sniffing was wrong - try the other
        other = "line_terminator" if kw == "lineterminator" else "lineterminator"
        df.to_csv(path, index=False, **{other: "\n"})


def _find(name):
    """Locate a shipped input. Same search order as the other analysis/ scripts."""
    _rev = Path(os.environ.get("TTIME_REVISION_OUT", REPO / "results" / "revision"))
    for d in [Path("."), _rev, REPO / "data", REPO, REPO / "data" / "external",
              Path(__file__).resolve().parent, Path("/content"), Path("/content/data")]:
        p = d / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(name)


def build(round_trip=True, keep_intermediate=False):
    """cells 183 -> 186 -> 187 -> 189.

    `round_trip` reproduces cell 183's write and cell 186's read-back. It is on by
    default because it changes the output - see the module docstring.
    """
    main_path = _find("updated_tcr_age_lists_with_all_significance.csv.gz")
    vdj_path = _find("vdjdb_trait_onlyB.csv")
    print(f"[*] age scores : {main_path}")
    print(f"[*] epitope db : {vdj_path}")

    # --- cell 183 ---------------------------------------------------------
    df_main = pd.read_csv(main_path)
    df_vdj = pd.read_csv(vdj_path)
    print(f"[*] left  {len(df_main):,} rows x {len(df_main.columns)} cols")
    print(f"[*] right {len(df_vdj):,} rows x {len(df_vdj.columns)} cols")

    df_vdj = df_vdj.rename(columns={"CDR3b": "TCR"})
    merged = pd.merge(df_main, df_vdj, on="TCR", how="inner")
    print(f"[*] inner merge on TCR -> {len(merged):,} rows x {len(merged.columns)} cols")

    if round_trip:
        # Cell 183's write + cell 186's read-back. This perturbs 342 rows in the last
        # float ULP and is required for byte-identity - see the module docstring.
        _to_csv_lf(merged, INTERMEDIATE)
        print(f"[*] wrote {INTERMEDIATE} ({Path(INTERMEDIATE).stat().st_size:,} bytes)")
        merged = pd.read_csv(INTERMEDIATE, low_memory=False)      # cell 186
        print(f"[*] read it back: {len(merged):,} rows")
        if not keep_intermediate:
            Path(INTERMEDIATE).unlink()
            print(f"[*] removed {INTERMEDIATE} (not a shipped artefact; "
                  f"--keep-intermediate to retain)")
    else:
        print("[!] --no-round-trip: skipping the intermediate. The output will differ "
              "from the shipped file in the last float digit of ~342 rows.")

    # --- cell 187 ---------------------------------------------------------
    df = merged.drop_duplicates()
    print(f"[*] drop_duplicates -> {len(df):,} rows")

    # --- cell 189 ---------------------------------------------------------
    _to_csv_lf(df, OUT)
    print(f"[*] wrote {OUT} ({Path(OUT).stat().st_size:,} bytes)")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep-intermediate", action="store_true",
                    help=f"leave {INTERMEDIATE} (76.5 MB) on disk instead of deleting it")
    ap.add_argument("--no-round-trip", action="store_true",
                    help="skip the intermediate write/read entirely. Faster, same data, "
                         "but NOT byte-identical to the shipped file (see docstring)")
    ap.add_argument("--verify", action="store_true",
                    help="compare the result byte-for-byte against the shipped copy")
    args = ap.parse_args(argv)

    df = build(round_trip=not args.no_round_trip,
               keep_intermediate=args.keep_intermediate)

    if len(df) != EXPECTED_ROWS or len(df.columns) != EXPECTED_COLS:
        sys.exit(f"[!] expected {EXPECTED_ROWS:,} x {EXPECTED_COLS}, "
                 f"got {len(df):,} x {len(df.columns)}. Inputs may have changed; "
                 f"do not ship this output.")
    if "CDR3b" in df.columns:
        sys.exit("[!] output carries a CDR3b column - that is the dead-end recipe "
                 "(notebook cell 239), not this chain. Refusing to continue.")

    digest = hashlib.sha256(Path(OUT).read_bytes()).hexdigest()
    print(f"[*] sha256 {digest}")

    if args.verify:
        shipped = REPO / "data" / "external" / OUT
        if not shipped.exists():
            sys.exit(f"[!] --verify: shipped copy not found at {shipped}")
        ref = hashlib.sha256(shipped.read_bytes()).hexdigest()
        print(f"[*] shipped {ref}")
        if digest == ref:
            print("[*] VERIFY PASS - byte-for-byte identical to the shipped file")
        else:
            sys.exit("[!] VERIFY FAIL - output differs from the shipped file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
