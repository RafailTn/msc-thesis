#!/usr/bin/env python3
"""Small shared helpers for the genomic-track analyses (BigWig / BED overlap scripts).

Extracted from the now-removed ``rbp_enrichment_fn_vs_tp.py`` so ``rnaseq_fn_vs_tp.py``
does not depend on a deleted module.
"""
from __future__ import annotations

import numpy as np


def tsv_chrom_to_fa(chrom: str) -> str:
    """Map the v7 TSV chromosome label to a GENCODE/UCSC contig name (chr...)."""
    c = str(chrom)
    if c in ("MT", "M", "chrMT"):
        return "chrM"
    return c if c.startswith("chr") else "chr" + c


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0, 1)
    return q
