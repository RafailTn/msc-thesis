#!/usr/bin/env python3
"""Append IntaRNA ensemble-energy columns to a v7 TSV.

For every (MRE, miRNA) pair in the file this runs IntaRNA in
partition-function / ensemble mode (``--model=P --mode=M``) and appends two
columns:

    Eall      ensemble energy of the miRNA-MRE interaction ensemble
              (-RT ln Zall over all considered interactions)
    Eall_MRE  ensemble energy of all *intramolecular* structures of the MRE
              alone (IntaRNA's ``Eall1``)

The miRNA (query) is assumed by default to have NO intramolecular structure
(``--qacc N`` -> IntaRNA ``--qAcc=N``), which zeroes its accessibility penalty
(ED2) so the interaction is scored as if the miRNA is fully single-stranded
(as it is when loaded in AGO2).

v7 schema: column ``gene`` = MRE/target (seq1), ``noncodingRNA`` = miRNA/query
(seq2). Column names are overridable with ``--target-col`` / ``--query-col``.
Pairs with no reportable interaction get empty values.

Extended MRE coordinates
------------------------
If you widen the MRE window (longer ``gene`` sequences with genomic flanks),
set a *local* accessibility window instead of folding the whole target:

    --tacc-w 150 --tacc-l 100     # RNAplfold-style sliding window (recommended)

The defaults ``--tacc-w 0 --tacc-l 0`` fold the full target (fine for short
~50 nt MREs, but O(L^3) and slow for long extended targets).

To keep ``Eall`` comparable to the non-extended run, anchor the *interaction*
to the original MRE inside the extended target with ``--tregion-col`` pointing
at the ``mre_region`` column that ``extend_mre.py`` writes (a 1-based
``--tRegion`` spec). This restricts only where the miRNA may pair; ``Eall_MRE``
still reflects the full extended-context fold.

Speed / robustness
------------------
IntaRNA is invoked once per chunk in ``--outPairwise`` mode (target[i] paired
with query[i]) with internal threading (``--threads``), amortising the
Vienna-RNA init a per-pair subprocess would repeat. Rows are streamed and the
output is flushed per chunk, so memory stays bounded regardless of input size.
The run is *resumable*: re-running the same command continues from wherever the
output left off (any partial trailing line is validated away first). With
``--max-seconds`` the process stops cleanly after a wall-clock budget and exits
with code 2 (re-run to continue); exit 0 means fully complete.

Usage
-----
    python add_ensemble_energy.py INPUT.tsv OUTPUT.tsv \\
        [--threads 12] [--chunk 25000] [--tacc-w 0 --tacc-l 0] \\
        [--intarna /path/to/IntaRNA] [--max-seconds 0]
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def sanitize(seq: str) -> str:
    """Uppercase and DNA->RNA (T->U); IntaRNA wants the RNA alphabet."""
    return seq.strip().upper().replace("T", "U")


def run_chunk(rows, intarna, threads, gene_col, mir_col, tacc_w, tacc_l, qacc,
              tregion=None):
    """Run IntaRNA pairwise on a chunk of rows.

    rows     : list of the raw split-line lists for this chunk.
    tregion  : optional 1-based inclusive ``--tRegion`` spec (e.g. "101-150")
               applied to every target in this call; restricts the *interaction*
               to that sub-region without changing the accessibility/Eall1 fold.
    Returns dict {local_id -> (Eall, Eall_MRE)} for rows that produced output.
    """
    with tempfile.NamedTemporaryFile("w", suffix="_t.fa", delete=False) as tf, \
         tempfile.NamedTemporaryFile("w", suffix="_q.fa", delete=False) as qf:
        t_path, q_path = tf.name, qf.name
        for i, r in enumerate(rows):
            tgt = sanitize(r[gene_col]) if gene_col < len(r) else ""
            qry = sanitize(r[mir_col]) if mir_col < len(r) else ""
            if not tgt or not qry:      # empty seq would abort the batch
                continue
            tf.write(f">{i}\n{tgt}\n")
            qf.write(f">{i}\n{qry}\n")

    cmd = [
        intarna,
        "-t", t_path,
        "-q", q_path,
        "--outPairwise",              # target[i] vs query[i]
        f"--qAcc={qacc}",             # N => miRNA has no intramolecular structure
        "--model=P",                  # partition function / ensemble
        "--mode=M",
        f"--tAccW={tacc_w}",          # target accessibility window (0 = full length)
        f"--tAccL={tacc_l}",          # max base-pair span within that window
        "--outMaxE=999",              # report even weak interactions (fewer empties)
        "--noSeed",                   # ensemble over all interactions, not seed-only
        "--outNoLP",
        "--threads", str(threads),
        "--outMode=C",
        "--outCsvCols=id1,Eall,Eall1",
    ]
    if tregion:                       # anchor the interaction to the MRE region
        cmd.append(f"--tRegion={tregion}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    out = {}
    lines = [l for l in result.stdout.splitlines() if l and not l.startswith("#")]
    if len(lines) >= 2:
        reader = csv.DictReader(lines, delimiter=";")
        for row in reader:
            try:
                out[int(row["id1"])] = (row.get("Eall", ""), row.get("Eall1", ""))
            except (ValueError, KeyError, TypeError):
                continue

    if not out and result.stderr:
        err = [l for l in result.stderr.splitlines()
               if "error" in l.lower() and not l.startswith("#")]
        if err:
            print(f"\n[warn] IntaRNA reported errors: {err[:3]}", file=sys.stderr)

    Path(t_path).unlink(missing_ok=True)
    Path(q_path).unlink(missing_ok=True)
    return out


def score_chunk(chunk, regions, intarna, threads, gene_col, mir_col,
                tacc_w, tacc_l, qacc):
    """Score a chunk, returning {chunk_index -> (Eall, Eall_MRE)}.

    If *regions* is None, one IntaRNA call scores the whole chunk. Otherwise
    *regions* is a per-row list of ``--tRegion`` specs (empty/None = no
    constraint); rows are bucketed by spec so each distinct region needs one
    call, and results are mapped back to their original chunk positions.
    """
    if regions is None:
        return run_chunk(chunk, intarna, threads, gene_col, mir_col,
                         tacc_w, tacc_l, qacc)

    buckets: dict = {}
    for idx, reg in enumerate(regions):
        buckets.setdefault(reg or None, []).append(idx)

    out = {}
    for reg, idxs in buckets.items():
        sub = [chunk[i] for i in idxs]
        res = run_chunk(sub, intarna, threads, gene_col, mir_col,
                        tacc_w, tacc_l, qacc, tregion=reg)
        for sub_i, chunk_i in enumerate(idxs):
            if sub_i in res:
                out[chunk_i] = res[sub_i]
    return out


def resume_offset(path: str, out_header: str, expected_ncols: int) -> int:
    """Prepare *path* for resuming and return #valid data rows already written.

    Returns -1 if the file is absent or its header doesn't match (caller starts
    fresh). Otherwise scans forward keeping only complete, well-formed data
    lines (correct field count, terminated by '\\n'), truncates the file at the
    end of the last valid line to discard any partial tail left by a hard kill,
    and returns the number of valid data rows.
    """
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return -1
    with open(path, "r+", newline="") as f:
        first = f.readline()
        if first != out_header:
            return -1                     # different/legacy file -> overwrite
        good_offset = f.tell()            # byte position after the header
        count = 0
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.endswith("\n") and line.count("\t") + 1 == expected_ncols:
                count += 1
                good_offset = pos + len(line)
            else:
                break                     # partial / malformed tail
        f.truncate(good_offset)
    return count


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--threads", type=int, default=12,
                    help="IntaRNA threads (range 0-20; 0 = all CPUs). "
                         "Memory scales with this.")
    ap.add_argument("--chunk", type=int, default=25_000,
                    help="pairs per IntaRNA invocation")
    ap.add_argument("--tacc-w", type=int, default=0,
                    help="target accessibility window (IntaRNA --tAccW); "
                         "0 = full length. Use e.g. 150 for extended MREs.")
    ap.add_argument("--tacc-l", type=int, default=0,
                    help="max bp span in the accessibility window (--tAccL); "
                         "0 = full length. Use e.g. 100 for extended MREs.")
    ap.add_argument("--qacc", default="N", choices=["N", "C"],
                    help="miRNA accessibility: N = no intramolecular structure "
                         "(default), C = fold the miRNA too")
    ap.add_argument("--target-col", default="gene",
                    help="TSV column holding the MRE/target sequence")
    ap.add_argument("--query-col", default="noncodingRNA",
                    help="TSV column holding the miRNA/query sequence")
    ap.add_argument("--tregion-col", default="",
                    help="TSV column with a 1-based --tRegion spec (e.g. the "
                         "'mre_region' column written by extend_mre.py). When "
                         "set, the interaction is anchored to that sub-region of "
                         "each (extended) target; Eall_MRE is unaffected.")
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="stop cleanly after this wall-clock budget (0 = no limit); "
                         "exit code 2 means partial, re-run to resume")
    ap.add_argument("--intarna", default=shutil.which("IntaRNA") or "IntaRNA",
                    help="path to the IntaRNA binary (default: found on PATH)")
    args = ap.parse_args()

    if not (shutil.which(args.intarna) or Path(args.intarna).is_file()):
        print(f"ERROR: IntaRNA not found ({args.intarna!r}). "
              f"Run inside the pixi env, e.g. `pixi run python ...`.", file=sys.stderr)
        return 1

    # --- read input header, locate the two sequence columns -----------------
    fin = open(args.input)
    header = fin.readline().rstrip("\n").split("\t")
    try:
        gene_col = header.index(args.target_col)
        mir_col = header.index(args.query_col)
    except ValueError:
        print(f"ERROR: input must have '{args.target_col}' and "
              f"'{args.query_col}' columns", file=sys.stderr)
        return 1
    treg_col = -1
    if args.tregion_col:
        try:
            treg_col = header.index(args.tregion_col)
        except ValueError:
            print(f"ERROR: --tregion-col {args.tregion_col!r} not in input",
                  file=sys.stderr)
            return 1
    out_header = "\t".join(header) + "\tEall\tEall_MRE\n"
    expected_ncols = len(header) + 2

    # --- resume: count valid rows already written, drop any partial tail -----
    already = resume_offset(args.output, out_header, expected_ncols)
    if already < 0:                       # no/invalid existing output -> start fresh
        fout = open(args.output, "w", newline="")
        fout.write(out_header)
        already = 0
    else:
        fout = open(args.output, "r+", newline="")
        fout.seek(0, 2)                   # append position (file already truncated)
        for _ in range(already):          # skip the input rows we've already scored
            fin.readline()
        print(f"[resume] {already:,} rows already done; continuing", file=sys.stderr)

    chunk, total, t0 = [], already, time.time()
    complete = True

    def flush():
        nonlocal total
        regions = None
        if treg_col >= 0:
            regions = [r[treg_col] if treg_col < len(r) else "" for r in chunk]
        res = score_chunk(chunk, regions, args.intarna, args.threads,
                          gene_col, mir_col, args.tacc_w, args.tacc_l, args.qacc)
        for i, r in enumerate(chunk):
            eall, eall_mre = res.get(i, ("", ""))
            fout.write("\t".join(r) + f"\t{eall}\t{eall_mre}\n")
        fout.flush()
        total += len(chunk)
        dt = time.time() - t0
        print(f"\r{total:,} rows  (+{total-already:,} this run, "
              f"{(total-already)/dt:,.0f}/s)  "
              f"covered={len(res)}/{len(chunk)} last chunk  "
              f"elapsed={dt/60:.1f}m",
              end="", file=sys.stderr, flush=True)

    for line in fin:
        chunk.append(line.rstrip("\n").split("\t"))
        if len(chunk) >= args.chunk:
            flush()
            chunk = []
            if args.max_seconds and (time.time() - t0) > args.max_seconds:
                complete = False
                break
    if complete and chunk:
        flush()

    fin.close()
    fout.close()

    if complete:
        print(f"\nCOMPLETE: {total:,} rows -> {args.output}", file=sys.stderr)
        return 0
    print(f"\nPARTIAL: {total:,} rows so far -> {args.output} (re-run to resume)",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
