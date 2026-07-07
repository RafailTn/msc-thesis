#!/usr/bin/env python3
"""Append IntaRNA ensemble-energy columns to a v7 TSV.

For every (MRE, miRNA) pair in the file this runs IntaRNA in
partition-function / ensemble mode (``--model=P --mode=M``) and appends two
columns:

    Eall      ensemble energy of the miRNA-MRE interaction ensemble
              (-RT ln Zall over all considered interactions)
    Eall_MRE  ensemble energy of all *intramolecular* structures of the MRE
              alone (IntaRNA's ``Eall1``)

The miRNA (query) is assumed to have NO intramolecular structure: ``--qAcc=N``
zeroes its accessibility penalty (ED2), so the interaction is scored as if the
miRNA is fully single-stranded (as it is when loaded in AGO2).

v7 schema: column ``gene`` = MRE/target (seq1), ``noncodingRNA`` = miRNA/query
(seq2). Pairs with no reportable interaction get empty values.

Speed: IntaRNA is invoked once per chunk in ``--outPairwise`` mode (target[i]
paired with query[i]) with internal threading (``--threads``), which amortises
the Vienna-RNA init that a per-pair subprocess would repeat. Rows are streamed,
so memory stays bounded to one chunk regardless of input size (works on the
2GB train file). Missing IntaRNA rows are filled with empty strings.

Usage
-----
    python add_ensemble_energy.py INPUT.tsv OUTPUT.tsv \\
        [--threads 20] [--chunk 50000] [--intarna /path/to/IntaRNA]
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_INTARNA = (
    "/home/rafail/.cache/rattler/cache/cached-envs-v0/"
    "intarna-cdfac3b03d2c238e/bin/IntaRNA"
)


def sanitize(seq: str) -> str:
    """Uppercase and DNA->RNA (T->U); IntaRNA wants the RNA alphabet."""
    return seq.strip().upper().replace("T", "U")


def run_chunk(rows, intarna, threads, gene_col, mir_col):
    """Run IntaRNA pairwise on a chunk of rows.

    rows : list of the raw split-line lists for this chunk.
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
        "--outPairwise",       # target[i] vs query[i]
        "--qAcc=N",            # miRNA assumed to have no intramolecular structure
        "--model=P",           # partition function / ensemble
        "--mode=M",
        "--tAccW=0", "--tAccL=0",
        "--outMaxE=999",       # report even weak interactions (fewer empties)
        "--noSeed",            # ensemble over all interactions, not seed-only
        "--outNoLP",
        "--threads", str(threads),
        "--outMode=C",
        "--outCsvCols=id1,Eall,Eall1",
    ]

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
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=25_000)
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="stop cleanly after this wall-clock budget (0 = no limit); "
                         "exit code 2 means partial, re-run to resume")
    ap.add_argument("--intarna", default=DEFAULT_INTARNA)
    args = ap.parse_args()

    # --- read input header, locate the two sequence columns -----------------
    fin = open(args.input)
    header = fin.readline().rstrip("\n").split("\t")
    try:
        gene_col = header.index("gene")
        mir_col = header.index("noncodingRNA")
    except ValueError:
        print("ERROR: input must have 'gene' and 'noncodingRNA' columns",
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
        # skip the input rows we've already scored
        for _ in range(already):
            fin.readline()
        print(f"[resume] {already:,} rows already done; continuing",
              file=sys.stderr)

    chunk, total, t0 = [], already, time.time()
    complete = True

    def flush():
        nonlocal total
        res = run_chunk(chunk, args.intarna, args.threads, gene_col, mir_col)
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
