# IntaRNA ensemble-energy annotation

Self-contained tool to append two IntaRNA ensemble-energy columns to a v7 TSV:

| column     | meaning |
|------------|---------|
| `Eall`     | ensemble energy of the miRNA–MRE **interaction** (`-RT·ln Zall` over all interactions) |
| `Eall_MRE` | ensemble energy of the MRE's own **intramolecular** structure (IntaRNA's `Eall1`) |

The miRNA is treated as having **no intramolecular structure** (`--qAcc=N`, i.e.
ED2 = 0), matching the assumption that it is single-stranded while loaded in AGO2.

This lives outside the main project env on purpose — IntaRNA is not in
`dependencies/pixi.toml`, and this keeps the server footprint tiny.

## Setup (once, on the server)

```bash
cd intarna_ensemble
pixi install          # fetches IntaRNA 3.4.1 (+ ViennaRNA) from bioconda
pixi run version      # sanity check: prints the IntaRNA version
```

`pixi.lock` (committed next to `pixi.toml`) pins exact package builds, so the
server resolves the identical environment. If the server is not `linux-64`, add
its platform to `platforms` in `pixi.toml` and re-run `pixi install`.

## Run

```bash
./run_ensemble.sh INPUT.tsv OUTPUT.tsv --threads 12
```

The input must be a tab-separated file with a header containing a target column
(`gene` by default) and a query column (`noncodingRNA` by default). Two new
columns `Eall` and `Eall_MRE` are appended; the originals are untouched.

### Extended MRE coordinates (two steps)

The bare `gene` in a v7 TSV is only the ~50 nt MRE. To fold the MRE **in
context**, first widen it with `extend_mre.py`, then score the widened file.

**Step 1 — widen `gene`** (needs the genome + GENCODE GTF; picks the flank
source the same way `cnn/compute_accessibility.py` does — MANE mature-transcript
context for exonic sites, strand-aware genomic context otherwise):

```bash
pixi run python extend_mre.py \
    --input  in_v7.tsv \
    --output in_v7_extended.tsv \
    --genome /path/GRCh38.primary_assembly.genome.fa \
    --gtf    /path/gencode.v47.primary_assembly.annotation.gtf.gz \
    --flank  100                       # nt added on EACH side (50 nt -> ~250 nt)
```

This leaves `chr/start/end/strand` untouched (the locus doesn't move) and adds:

| column | meaning |
|--------|---------|
| `mre_offset` | 0-based start of the original 50 nt MRE inside the new `gene` (so `gene[mre_offset:mre_offset+50]` is the real MRE) |
| `mre_region` | the same span as a 1-based inclusive IntaRNA `--tRegion` string (e.g. `101-150`) — feed it to the scorer to anchor the interaction |
| `acc_mode`   | context used: `mane` (spliced/exonic), `genomic`, or `unmapped` (chrom/coords missing — `gene` left as-is) |

**Step 2 — score with a local accessibility window** so the now-long target is
not folded whole (full folding is `O(L^3)`), and anchor the interaction to the
original MRE with `--tregion-col mre_region`:

```bash
./run_ensemble.sh in_v7_extended.tsv out.tsv --threads 16 \
    --tacc-w 150 --tacc-l 100 --tregion-col mre_region
```

- `--tacc-w` = sliding window length (IntaRNA `--tAccW`), RNAplfold-style.
- `--tacc-l` = max base-pair span within that window (`--tAccL`).
- Defaults `0/0` = fold the full target (fine for short ~50 nt MREs, the setting
  used for the original non-extended run).
- `--tregion-col mre_region` restricts *where the miRNA may pair* to the original
  MRE, so the miRNA can't latch onto a spurious stronger site out in the flanks.
  Rows are bucketed by region, so this adds essentially no runtime (almost all
  rows share the same `--flank`-derived region).

**What each column captures, and comparability to the non-extended run** (all
verified empirically):

- `Eall_MRE` (the target's own structure) is **unaffected** by `--tregion-col` —
  it always reflects the full extended-context fold. It drops a lot versus the
  bare 50 nt MRE (e.g. ~−6 kcal/mol at 50 nt → ~−70 at 250 nt): more sequence,
  more intramolecular structure. So extended and non-extended `Eall_MRE` are
  **not** on the same scale — that longer-context structure is the point of
  extending, but don't compare the two directly.
- `Eall` (the interaction) **with** `--tregion-col` stays close to the
  non-extended value (same MRE pairing region), so it *is* comparable across the
  extended/non-extended runs. **Without** it, `Eall` drifts more negative because
  the miRNA can pair anywhere in the long window.

Re-annotate **all** splits with the same `--flank` / `--tacc-w` / `--tacc-l`
(and the same `--tregion-col` choice) if you intend to compare them.

## Robustness / resuming

- **Streamed + chunked**: memory stays bounded no matter how big the input is.
- **Resumable**: if the job dies, re-run the *same* command — it validates the
  existing output, drops any half-written trailing line, and continues.
- **Time-boxed (optional)**: `--max-seconds 540` stops cleanly after a budget
  and exits code `2`; loop it if your scheduler caps wall-clock per job:

  ```bash
  while ! ./run_ensemble.sh in.tsv out.tsv --threads 12 --max-seconds 540; do
      echo "resuming..."      # exits 0 when fully complete
  done
  ```

On a normal server you usually don't need `--max-seconds` at all — just run once
(optionally under `nohup`/`tmux`).

## Key options

| flag | default | notes |
|------|---------|-------|
| `--threads` | 12 | IntaRNA threads (0–20; 0 = all CPUs). Memory scales with it. |
| `--chunk` | 25000 | pairs per IntaRNA invocation |
| `--tacc-w` / `--tacc-l` | 0 / 0 | target accessibility window (see above) |
| `--tregion-col` | (off) | column with a 1-based `--tRegion` spec (use `mre_region` from `extend_mre.py`) to anchor the interaction to the original MRE |
| `--qacc` | N | `N` = miRNA unstructured; `C` = fold the miRNA too |
| `--target-col` / `--query-col` | gene / noncodingRNA | sequence column names |
| `--intarna` | (PATH) | path to the `IntaRNA` binary; found automatically in the pixi env |

## Throughput

~500–550 pairs/s at 12 threads on short (~50 nt) MREs (partition-function mode).
Longer extended targets are slower — the local accessibility window keeps it
manageable. All (miRNA, MRE) pairs are scored independently (no dedup), so cost
scales linearly with row count.
