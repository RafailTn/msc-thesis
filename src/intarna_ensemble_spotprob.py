#!/usr/bin/env python3
"""
IntaRNA ensemble runner — extracts Eall, P_E and the per-base tSpotProb
vector for every miRNA-MRE pair.

For each pair this script runs IntaRNA once in partition-function mode
(--model=P).  Two outputs are captured from that single call:

  Eall, P_E   — from the rank-1 interaction in the CSV output
  tspot_probs — per-base probability that position i of the target is
                covered by any interaction in the ensemble (Zi / Z),
                produced by --out=tSpotProb.  Length = len(target_seq).

Because tSpotProb is derived from the same partition function that
produces Eall / P_E, there is no extra thermodynamic computation.

Output TSV columns
------------------
    pair_index   target_id   query_id
    target_sequence   query_sequence
    Eall   P_E
    tspot_probs       (comma-separated floats, one per target position)
    status            (ok | no_interaction | failed)

The tspot_probs column is ready to be used directly as the ``tspot_probs``
input column of cnn_branches.py.

Usage
-----
    python intarna_ensemble_spotprob.py \\
        mre.fasta mirna.fasta \\
        -o ensemble_spotprob.tsv \\
        --threads 8
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import Iterator, List, Optional


# ---------------------------------------------------------------------------
# FASTA parsing (self-contained so this script has no local imports)
# ---------------------------------------------------------------------------

@dataclass
class FastaRecord:
    header: str
    sequence: str

    @property
    def id(self) -> str:
        return self.header.lstrip(">").split()[0]


def parse_fasta(filepath: str) -> Iterator[FastaRecord]:
    header = None
    seq_lines: list[str] = []
    with open(filepath) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield FastaRecord(header, "".join(seq_lines))
                header = line
                seq_lines = []
            else:
                seq_lines.append(line.upper().replace("T", "U"))
        if header is not None:
            yield FastaRecord(header, "".join(seq_lines))


# ---------------------------------------------------------------------------
# IntaRNA call
# ---------------------------------------------------------------------------

def _parse_spot_prob_file(filepath: str, target_len: int) -> List[float]:
    """Parse an IntaRNA tSpotProb CSV (semicolon-delimited, 1-based positions).

    Expected format (comment lines and optional header skipped):
        idx1;spotProb
        1;0.12345
        2;0.00000
        ...

    Returns a list of length *target_len* (zero-padded / cropped as needed).
    Falls back to all-zeros if the file is absent or unreadable.
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
                    pos = int(parts[0]) - 1       # 1-based → 0-based
                    val = float(parts[1])
                    if 0 <= pos < target_len:
                        probs[pos] = val
                except ValueError:
                    continue                       # skip header / malformed
    except OSError:
        pass
    return probs


def run_ensemble_pair(
    target_seq: str,
    query_seq: str,
    target_id: str,
    query_id: str,
    debug: bool = False,
) -> tuple[float, float, List[float], str]:
    """Run IntaRNA ensemble mode for one pair.

    Returns
    -------
    eall        : Eall of the rank-1 interaction (0.0 on failure)
    p_e         : P_E  of the rank-1 interaction (0.0 on failure)
    tspot_probs : per-base spot-probability vector (len = len(target_seq))
    status      : "ok" | "no_interaction" | "failed"
    """
    target_len = len(target_seq)

    # Temp file for tSpotProb output — each thread gets its own file
    tmp = tempfile.NamedTemporaryFile(suffix="_tspot.csv", delete=False)
    spot_path = tmp.name
    tmp.close()

    cmd = [
        "IntaRNA",
        "--target",   target_seq,
        "--query",    query_seq,
        "--tId=" + target_id,
        "--qId=" + query_id,
        "--model=P",          # partition function / ensemble
        "--mode=M",
        "--tAccW=0",
        "--tAccL=0",
        "--qAcc=N",
        "--outMode=C",
        "--outCsvCols=id1,id2,Eall,P_E",
        "-n", "1",            # only rank-1 needed for energy scalars
        "--noSeed",
        "--outNoLP",
        "--outMaxE=100",
        "--out=tSpotProb:" + spot_path,   # per-base interaction probability
    ]

    if debug:
        print(f"[DEBUG] {' '.join(cmd)}", file=sys.stderr)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
    except subprocess.TimeoutExpired:
        os.unlink(spot_path)
        return 0.0, 0.0, [0.0] * target_len, "failed"
    except FileNotFoundError:
        os.unlink(spot_path)
        return 0.0, 0.0, [0.0] * target_len, "failed"

    # Parse tSpotProb file before any early return
    tspot = _parse_spot_prob_file(spot_path, target_len)
    try:
        os.unlink(spot_path)
    except OSError:
        pass

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    # Check for hard errors
    error_lines = [l for l in stderr.splitlines()
                   if "error" in l.lower() and not l.startswith("#")]
    if error_lines:
        if debug:
            print(f"[DEBUG] IntaRNA errors: {error_lines}", file=sys.stderr)
        return 0.0, 0.0, tspot, "failed"

    # Parse energy scalars from CSV
    lines = [l for l in stdout.strip().splitlines() if not l.startswith("#")]
    if len(lines) < 2:
        return 0.0, 0.0, tspot, "no_interaction"

    try:
        reader = csv.DictReader(lines, delimiter=";")
        row = next(reader)
        eall = float(row.get("Eall", 0) or 0)
        p_e  = float(row.get("P_E",  0) or 0)
    except (StopIteration, ValueError, KeyError):
        return 0.0, 0.0, tspot, "no_interaction"

    return eall, p_e, tspot, "ok"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

@dataclass
class PairEnsembleResult:
    pair_index:   int
    target_id:    str
    query_id:     str
    target_seq:   str
    query_seq:    str
    eall:         float       = 0.0
    p_e:          float       = 0.0
    tspot_probs:  List[float] = field(default_factory=list)
    status:       str         = "ok"


def process_pair(
    pair_index: int,
    target: FastaRecord,
    query:  FastaRecord,
    debug:  bool,
) -> PairEnsembleResult:
    eall, p_e, tspot, status = run_ensemble_pair(
        target.sequence, query.sequence, target.id, query.id, debug
    )
    return PairEnsembleResult(
        pair_index  = pair_index,
        target_id   = target.id,
        query_id    = query.id,
        target_seq  = target.sequence,
        query_seq   = query.sequence,
        eall        = eall,
        p_e         = p_e,
        tspot_probs = tspot,
        status      = status,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

HEADERS = [
    "pair_index", "target_id", "query_id",
    "target_sequence", "query_sequence",
    "Eall", "P_E",
    "tspot_probs",
    "status",
]


def write_tsv(results: list[PairEnsembleResult], output_file: str) -> None:
    with open(output_file, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(HEADERS)
        for r in results:
            tspot_str = ",".join(f"{v:.6f}" for v in r.tspot_probs) if r.tspot_probs else ""
            writer.writerow([
                r.pair_index,
                r.target_id,
                r.query_id,
                r.target_seq,
                r.query_seq,
                f"{r.eall:.4f}",
                f"{r.p_e:.6f}",
                tspot_str,
                r.status,
            ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

class _ProgressTracker:
    def __init__(self, total: int) -> None:
        self.total = total
        self.done  = 0
        self._lock = Lock()

    def tick(self, r: PairEnsembleResult) -> None:
        with self._lock:
            self.done += 1
            pct = self.done / self.total * 100
            print(
                f"\r[{self.done}/{self.total}  {pct:.1f}%]  "
                f"pair {r.pair_index}: {r.target_id} vs {r.query_id}  "
                f"status={r.status}  Eall={r.eall:.2f}  P_E={r.p_e:.4f}",
                end="", file=sys.stderr,
            )


def main() -> int:
    p = argparse.ArgumentParser(
        description="IntaRNA ensemble: extract Eall, P_E and tSpotProb per pair.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("target_fasta", help="FASTA file with target / MRE sequences")
    p.add_argument("query_fasta",  help="FASTA file with query / miRNA sequences")
    p.add_argument("-o", "--output", required=True,
                   help="Output TSV file")
    p.add_argument("-t", "--threads", type=int, default=4,
                   help="Parallel threads")
    p.add_argument("--debug", action="store_true",
                   help="Print IntaRNA commands to stderr")
    args = p.parse_args()

    targets = list(parse_fasta(args.target_fasta))
    queries = list(parse_fasta(args.query_fasta))
    print(f"Loaded {len(targets)} targets, {len(queries)} queries", file=sys.stderr)

    if len(targets) != len(queries):
        print(
            f"Warning: counts differ — processing {min(len(targets), len(queries))} pairs.",
            file=sys.stderr,
        )

    pairs = list(zip(targets, queries))
    tracker = _ProgressTracker(len(pairs))
    results_dict: dict[int, PairEnsembleResult] = {}

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        future_to_idx = {
            pool.submit(process_pair, i, tgt, qry, args.debug): i
            for i, (tgt, qry) in enumerate(pairs, 1)
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                res = fut.result()
            except Exception as exc:
                tgt, qry = pairs[idx - 1]
                res = PairEnsembleResult(
                    pair_index=idx, target_id=tgt.id, query_id=qry.id,
                    target_seq=tgt.sequence, query_seq=qry.sequence,
                    status="failed",
                )
                print(f"\nPair {idx} raised: {exc}", file=sys.stderr)
            results_dict[idx] = res
            tracker.tick(res)

    print("", file=sys.stderr)   # newline after progress bar

    results = [results_dict[i] for i in sorted(results_dict)]
    write_tsv(results, args.output)

    ok      = sum(1 for r in results if r.status == "ok")
    no_int  = sum(1 for r in results if r.status == "no_interaction")
    failed  = sum(1 for r in results if r.status == "failed")
    print(f"\nDone.  ok={ok}  no_interaction={no_int}  failed={failed}", file=sys.stderr)
    print(f"Output written to: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
