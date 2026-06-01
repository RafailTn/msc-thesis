#!/usr/bin/env python3
"""
Prepare a single CSV file ready for cnn_branches.py from an existing TSV.

Input TSV required columns
---------------------------
    mre_sequence    – nucleotide string for the target / MRE (up to 50 nt)
    mirna_sequence  – nucleotide string for the query / miRNA (~22 nt)

Optional columns used for conservation
---------------------------------------
    chr, start, end, strand
        1-based inclusive genomic coordinates of the MRE.
        Required if --bigwig is passed; absent → conservation_vector zero-filled.

All other columns (e.g. label, target_id, query_id) are passed through unchanged.

What this script produces per row
----------------------------------
  Phase 1 – IntaRNA ensemble (parallel, --threads):
      Eall, P_E      – scalar energy features (rank-1 partition-function interaction)
      tspot_probs    – per-base interaction probability vector (MRE_LEN values)

  Phase 2 – phastCons conservation (sequential, one BigWig handle):
      conservation_vector – per-base phastCons scores (MRE_LEN values)

  Phase 3 – AGO2-eCLIP model inference (batched, GPU/CPU):
      eclip_probs    – per-base softmax probability vector (MRE_LEN values)
                       from the TwoComponentEclip model in mirna_eqtl.

Output CSV columns
------------------
    <all original columns>  plus:
    conservation_vector  – comma-separated floats (MRE_LEN values)
    tspot_probs          – comma-separated floats (MRE_LEN values)
    eclip_probs          – comma-separated floats (MRE_LEN values)
    Eall, P_E            – floats
    intarna_status       – ok | no_interaction | failed

Usage
-----
    python prepare_cnn_input.py \\
        --input   samples.tsv \\
        --bigwig  hg38.phastCons470way.bw \\
        --eclip-checkpoint /path/to/eclip_two_component.pt \\
        --eclip-src-dir    /path/to/mirna_eqtl/src \\
        --output  cnn_input.csv \\
        --threads 8
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import pyBigWig
    HAS_PYBIGWIG = True
except ImportError:
    HAS_PYBIGWIG = False

MRE_LEN = 50   # must match cnn_branches.py


# ---------------------------------------------------------------------------
# Phase 2 helpers – Conservation (BigWig)
# ---------------------------------------------------------------------------

def _normalise_chrom(name: object) -> str:
    n = str(name).strip()
    if n.endswith(".0"):
        n = n[:-2]
    if not n.startswith("chr"):
        n = "chr" + n
    return "chrM" if n == "chrMT" else n


def _fetch_conservation(bw, chrom: str, start, end, strand: str,
                        length: int) -> List[float]:
    """Fetch phastCons scores for a region and resize to *length* values."""
    chrom = _normalise_chrom(chrom)
    try:
        scores = bw.values(chrom, int(start) - 1, int(end))   # 0-based half-open
        scores = np.nan_to_num(np.array(scores, dtype=np.float32), nan=0.0)
        if strand == "-":
            scores = scores[::-1]
    except (RuntimeError, ValueError, TypeError):
        scores = np.zeros(length, dtype=np.float32)
    out = np.zeros(length, dtype=np.float32)
    n = min(len(scores), length)
    out[:n] = scores[:n]
    return out.tolist()


# ---------------------------------------------------------------------------
# Phase 1 helpers – IntaRNA ensemble + tSpotProb
# ---------------------------------------------------------------------------

def _parse_spot_prob_file(filepath: str, target_len: int) -> List[float]:
    """Parse an IntaRNA tSpotProb CSV into a fixed-length float list.

    Expected format (comment / header lines skipped):
        idx1;spotProb
        1;0.12345
        ...
    Positions are 1-based. Falls back to zeros on any error.
    """
    probs = [0.0] * target_len
    try:
        with open(filepath) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(";")
                if len(parts) < 2:
                    continue
                try:
                    pos = int(parts[0]) - 1        # 1-based → 0-based
                    val = float(parts[1])
                    if 0 <= pos < target_len:
                        probs[pos] = val
                except ValueError:
                    continue
    except OSError:
        pass
    return probs


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _run_intarna(target_seq: str, query_seq: str,
                 target_id: str, query_id: str,
                 debug: bool = False) -> tuple[float, float, List[float], str]:
    """Run IntaRNA ensemble mode for one pair.

    Returns (Eall, P_E, tspot_probs, status).
    """
    target_len = len(target_seq)

    tmp = tempfile.NamedTemporaryFile(suffix="_tspot.csv", delete=False)
    spot_path = tmp.name
    tmp.close()

    cmd = [
        "IntaRNA",
        "--target", target_seq,
        "--query",  query_seq,
        "--tId=" + target_id,
        "--qId=" + query_id,
        "--model=P",
        "--mode=M",
        "--tAccW=0",
        "--tAccL=0",
        "--qAcc=N",
        "--outMode=C",
        "--outCsvCols=id1,id2,Eall,P_E",
        "-n", "1",
        "--noSeed",
        "--outNoLP",
        "--outMaxE=100",
        "--out=tSpotProb:" + spot_path,
    ]

    if debug:
        print("[DEBUG]", " ".join(cmd), file=sys.stderr)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        _safe_unlink(spot_path)
        return 0.0, 0.0, [0.0] * target_len, "failed"
    except FileNotFoundError:
        _safe_unlink(spot_path)
        return 0.0, 0.0, [0.0] * target_len, "failed"

    tspot = _parse_spot_prob_file(spot_path, target_len)
    _safe_unlink(spot_path)

    stderr = result.stderr or ""
    if any("error" in l.lower() for l in stderr.splitlines()
           if not l.startswith("#")):
        return 0.0, 0.0, tspot, "failed"

    lines = [l for l in (result.stdout or "").strip().splitlines()
             if not l.startswith("#")]
    if len(lines) < 2:
        return 0.0, 0.0, tspot, "no_interaction"

    try:
        row = next(csv.DictReader(lines, delimiter=";"))
        eall = float(row.get("Eall", 0) or 0)
        p_e  = float(row.get("P_E",  0) or 0)
    except (StopIteration, ValueError, KeyError):
        return 0.0, 0.0, tspot, "no_interaction"

    return eall, p_e, tspot, "ok"


@dataclass
class _IntarnaResult:
    row_index:   int
    eall:        float       = 0.0
    p_e:         float       = 0.0
    tspot_probs: List[float] = field(default_factory=list)
    status:      str         = "ok"


def _process_intarna_row(row_index: int, mre_seq: str, mirna_seq: str,
                         target_id: str, query_id: str,
                         debug: bool) -> _IntarnaResult:
    eall, p_e, tspot, status = _run_intarna(
        mre_seq, mirna_seq, target_id, query_id, debug)
    return _IntarnaResult(row_index=row_index, eall=eall, p_e=p_e,
                          tspot_probs=tspot, status=status)


# ---------------------------------------------------------------------------
# Phase 3 helpers – eCLIP model inference
# ---------------------------------------------------------------------------

# One-hot encoder for DNA sequences (U treated as T for compatibility with
# mirna_eqtl models trained on DNA).
_ECLIP_BASE_IDX = np.full(256, -1, dtype=np.int8)
for _ch, _i in (("A", 0), ("C", 1), ("G", 2), ("T", 3), ("U", 3),
                ("a", 0), ("c", 1), ("g", 2), ("t", 3), ("u", 3)):
    _ECLIP_BASE_IDX[ord(_ch)] = _i


def _encode_sequence(seq: str) -> np.ndarray:
    """(4, L) float32 one-hot.  N / ambiguous → all-zero column."""
    arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    idx = _ECLIP_BASE_IDX[arr]
    out = np.zeros((4, len(arr)), dtype=np.float32)
    valid = idx >= 0
    out[idx[valid], np.nonzero(valid)[0]] = 1.0
    return out


def _load_eclip_model(checkpoint_path: str, src_dir: str, device):
    """Load a TwoComponentEclip checkpoint.

    Temporarily adds *src_dir* (mirna_eqtl/src) to sys.path so the relative
    imports inside load_model resolve correctly, then removes it again.
    """
    import torch
    abs_src = os.path.abspath(src_dir)
    sys.path.insert(0, abs_src)
    try:
        from predict_bigwig_two_component import load_model
        model = load_model(checkpoint_path, device)
    finally:
        sys.path.remove(abs_src)
    model.eval()
    return model


def _run_eclip_inference(model, seqs: List[str], device,
                         batch_size: int = 256) -> np.ndarray:
    """Run eCLIP model on MRE sequences.

    Sequences are padded / cropped to MRE_LEN and converted to DNA (U→T)
    before encoding. Returns an (N, MRE_LEN) float32 array of per-base
    softmax probabilities.
    """
    import torch

    def _prep(s: str) -> str:
        s = s.upper().replace("U", "T")
        if len(s) >= MRE_LEN:
            return s[:MRE_LEN]
        return s + "N" * (MRE_LEN - len(s))

    normed = [_prep(s) for s in seqs]
    n = len(normed)
    results = np.zeros((n, MRE_LEN), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = normed[start:start + batch_size]
            x = np.stack([_encode_sequence(s) for s in batch])
            x_t = torch.from_numpy(x).to(device)
            mask = torch.ones(x_t.shape[0], MRE_LEN, device=device)
            dist_logits, _ = model(x_t, mask)
            probs = torch.softmax(dist_logits, dim=-1).cpu().numpy()
            results[start:start + len(batch)] = probs

            done = min(start + batch_size, n)
            print(f"\r  eCLIP inference: {done}/{n}", end="", file=sys.stderr)

    print("", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# Serialise a float list to a compact comma-separated string
# ---------------------------------------------------------------------------

def _vec_to_str(vals: List[float], fmt: str = "{:.6f}") -> str:
    if not vals:
        return ",".join(["0.000000"] * MRE_LEN)
    return ",".join(fmt.format(v) for v in vals)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Build a cnn_branches.py-ready CSV from an existing TSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",   required=True,
                   help="Input TSV with mre_sequence and mirna_sequence columns")
    p.add_argument("--output",  required=True,
                   help="Output CSV for cnn_branches.py")
    p.add_argument("--bigwig",  default=None,
                   help="phastCons470way BigWig file. "
                        "Omit to zero-fill conservation_vector.")
    p.add_argument("--eclip-checkpoint", default=None, dest="eclip_checkpoint",
                   help="Path to TwoComponentEclip .pt checkpoint. "
                        "Omit to zero-fill eclip_probs.")
    p.add_argument("--eclip-src-dir", default=None, dest="eclip_src_dir",
                   help="Path to mirna_eqtl/src (needed to import the model). "
                        "Required when --eclip-checkpoint is given.")
    p.add_argument("--eclip-batch-size", type=int, default=256,
                   dest="eclip_batch_size",
                   help="Batch size for eCLIP model inference.")
    p.add_argument("--eclip-device", default=None, dest="eclip_device",
                   help="Device for eCLIP inference (cuda / cpu). "
                        "Defaults to cuda if available.")
    p.add_argument("--threads", type=int, default=4,
                   help="Parallel threads for IntaRNA")
    p.add_argument("--sep",     default="\t",
                   help="Separator of the input file")
    p.add_argument("--id-col",  default=None, dest="id_col",
                   help="Column to use as target_id passed to IntaRNA "
                        "(default: row index)")
    p.add_argument("--debug",   action="store_true",
                   help="Print IntaRNA commands to stderr")
    args = p.parse_args()

    # Validate eCLIP arguments
    if args.eclip_checkpoint and not args.eclip_src_dir:
        sys.exit("ERROR: --eclip-src-dir is required when --eclip-checkpoint is given.")
    if args.eclip_checkpoint and not os.path.isfile(args.eclip_checkpoint):
        sys.exit(f"ERROR: eCLIP checkpoint not found: {args.eclip_checkpoint}")

    # ── Load input ────────────────────────────────────────────────────────────
    print(f"Reading {args.input} ...", file=sys.stderr)
    df = pd.read_csv(args.input, sep=args.sep, dtype=str)
    n  = len(df)
    print(f"  {n} rows", file=sys.stderr)

    for col in ("mre_sequence", "mirna_sequence"):
        if col not in df.columns:
            sys.exit(f"ERROR: required column '{col}' not found in input.")

    # ── Phase 1: IntaRNA (parallel) ───────────────────────────────────────────
    print(f"\nPhase 1 — IntaRNA ensemble ({args.threads} threads) ...",
          file=sys.stderr)

    intarna_results: dict[int, _IntarnaResult] = {}
    lock = Lock()
    done_count = [0]

    def _tick(res: _IntarnaResult) -> None:
        with lock:
            done_count[0] += 1
            pct = done_count[0] / n * 100
            print(f"\r  {done_count[0]}/{n}  ({pct:.1f}%)",
                  end="", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {}
        for i, row in df.iterrows():
            mre   = str(row["mre_sequence"]).upper().replace("T", "U")
            mirna = str(row["mirna_sequence"]).upper().replace("T", "U")
            tid   = str(row[args.id_col]) if args.id_col else f"target_{i}"
            qid   = f"mirna_{i}"
            futures[pool.submit(
                _process_intarna_row, i, mre, mirna, tid, qid, args.debug
            )] = i

        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as exc:
                idx = futures[fut]
                res = _IntarnaResult(row_index=idx, status="failed")
                print(f"\n  row {idx} error: {exc}", file=sys.stderr)
            intarna_results[res.row_index] = res
            _tick(res)

    print("", file=sys.stderr)
    ok     = sum(1 for r in intarna_results.values() if r.status == "ok")
    no_int = sum(1 for r in intarna_results.values() if r.status == "no_interaction")
    failed = sum(1 for r in intarna_results.values() if r.status == "failed")
    print(f"  ok={ok}  no_interaction={no_int}  failed={failed}", file=sys.stderr)

    # ── Phase 2: Conservation (sequential — BigWig is not thread-safe) ────────
    has_coords = all(c in df.columns for c in ("chr", "start", "end", "strand"))
    cons_vectors: list[List[float]] = [[0.0] * MRE_LEN] * n

    print("\nPhase 2 — Conservation vectors ...", file=sys.stderr)
    if args.bigwig:
        if not HAS_PYBIGWIG:
            print("  WARNING: pyBigWig not installed — zero-filled.", file=sys.stderr)
        elif not has_coords:
            print("  WARNING: chr/start/end/strand absent — zero-filled.",
                  file=sys.stderr)
        else:
            bw = pyBigWig.open(args.bigwig)
            cons_vectors = []
            for i, row in df.iterrows():
                cons_vectors.append(_fetch_conservation(
                    bw, row["chr"], row["start"], row["end"], row["strand"], MRE_LEN))
                if (i + 1) % 500 == 0 or (i + 1) == n:
                    print(f"\r  {i + 1}/{n}", end="", file=sys.stderr)
            bw.close()
            print("", file=sys.stderr)
    else:
        print("  No --bigwig supplied — zero-filled.", file=sys.stderr)

    # ── Phase 3: eCLIP inference (batched) ────────────────────────────────────
    eclip_probs_mat: Optional[np.ndarray] = None

    print("\nPhase 3 — eCLIP inference ...", file=sys.stderr)
    if args.eclip_checkpoint:
        import torch
        if args.eclip_device:
            device = torch.device(args.eclip_device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  device: {device}", file=sys.stderr)
        print(f"  loading checkpoint: {args.eclip_checkpoint}", file=sys.stderr)
        try:
            model = _load_eclip_model(args.eclip_checkpoint, args.eclip_src_dir, device)
            mre_seqs = df["mre_sequence"].astype(str).tolist()
            eclip_probs_mat = _run_eclip_inference(
                model, mre_seqs, device, args.eclip_batch_size)
            print(f"  done — shape {eclip_probs_mat.shape}", file=sys.stderr)
        except Exception as exc:
            print(f"  WARNING: eCLIP inference failed ({exc}) — zero-filled.",
                  file=sys.stderr)
            eclip_probs_mat = None
    else:
        print("  No --eclip-checkpoint supplied — zero-filled.", file=sys.stderr)

    # ── Phase 4: Assemble and write ───────────────────────────────────────────
    print(f"\nWriting {args.output} ...", file=sys.stderr)

    tspot_col  = []
    eall_col   = []
    pe_col     = []
    status_col = []
    eclip_col  = []
    cons_col   = [_vec_to_str(v) for v in cons_vectors]

    for i in range(n):
        res = intarna_results.get(i, _IntarnaResult(row_index=i, status="failed"))
        tspot_col.append(_vec_to_str(res.tspot_probs))
        eall_col.append(res.eall)
        pe_col.append(res.p_e)
        status_col.append(res.status)

        if eclip_probs_mat is not None:
            eclip_col.append(_vec_to_str(eclip_probs_mat[i].tolist()))
        else:
            eclip_col.append(_vec_to_str([]))

    out_df = df.copy()
    out_df["conservation_vector"] = cons_col
    out_df["tspot_probs"]         = tspot_col
    out_df["eclip_probs"]         = eclip_col
    out_df["Eall"]                = eall_col
    out_df["P_E"]                 = pe_col
    out_df["intarna_status"]      = status_col

    out_df.to_csv(args.output, index=False)
    print(f"Done — {n} rows × {out_df.shape[1]} columns → {args.output}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
