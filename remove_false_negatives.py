import polars as pl
import numpy as np
import pyranges as pr

# Load files — Polars is significantly faster here for large files
df_sites = pl.read_csv("Homo_sapiens_cleaned.tsv", separator="\t")
df_main = pl.read_csv("AGO2eCLIPManakov2022leftout.tsv", separator="\t")

# Normalize chromosomes
df_sites = df_sites.with_columns(
    pl.col("chromosome").str.replace("chr", "").alias("chr_norm")
)
df_main = df_main.with_columns(
    pl.col("chr").cast(pl.Utf8).alias("chr_norm"),
    pl.arange(0, pl.count()).alias("_orig_idx"),  # preserve row identity
)

# --- PyRanges overlap step (requires brief pandas conversion) ---

# Prepare sites for PyRanges
sites_pd = (
    df_sites.select(
        (pl.col("mirna_name") + "__" + pl.col("chr_norm")).alias("Chromosome"),
        pl.col("start").alias("Start"),
        pl.col("end").alias("End"),
    )
    .to_pandas()
)

# Only convert label-0 rows
label0 = df_main.filter(pl.col("label") == 0)
label0_pd = (
    label0.select(
        (pl.col("noncodingRNA_name") + "__" + pl.col("chr_norm")).alias("Chromosome"),
        pl.col("start").alias("Start"),
        pl.col("end").alias("End"),
        pl.col("_orig_idx"),
    )
    .to_pandas()
)

# Find overlaps
gr_sites = pr.PyRanges(sites_pd)
gr_label0 = pr.PyRanges(label0_pd)
overlaps = gr_label0.join(gr_sites, how=None)

# Get indices of overlapping label-0 rows — back to Polars immediately
overlap_indices = pl.Series("_orig_idx", overlaps.df["_orig_idx"].unique())

# --- Back to pure Polars for the rest ---

# Tag rows to remove (label-0 overlapping)
remove_0 = df_main.filter(pl.col("_orig_idx").is_in(overlap_indices))

# Count removals per miRNA
remove_counts = (
    remove_0
    .group_by("noncodingRNA_name")
    .agg(pl.count().alias("n_remove"))
)

# For each miRNA, sample the same number of label-1 rows to remove
rng = np.random.default_rng(42)
label1_remove_indices = []

for row in remove_counts.iter_rows(named=True):
    mirna = row["noncodingRNA_name"]
    count = row["n_remove"]

    label1_pool = df_main.filter(
        (pl.col("noncodingRNA_name") == mirna) & (pl.col("label") == 1)
    )

    n_available = label1_pool.height
    n_remove = min(count, n_available)

    if n_remove < count:
        print(f"Warning: {mirna} has only {n_available} label-1 rows but {count} need removal.")

    # Random sample using numpy, then extract polars indices
    sampled_positions = rng.choice(n_available, size=n_remove, replace=False)
    sampled_idx = label1_pool["_orig_idx"].gather(sampled_positions.tolist())
    label1_remove_indices.extend(sampled_idx.to_list())

label1_remove = pl.Series("_orig_idx", label1_remove_indices)

# Combine all indices to remove and filter
all_remove = pl.concat([overlap_indices, label1_remove])
df_filtered = (
    df_main
    .filter(~pl.col("_orig_idx").is_in(all_remove))
    .drop(["chr_norm", "_orig_idx"])
)

print(f"Removed {overlap_indices.len()} label-0 rows and {len(label1_remove_indices)} label-1 rows")
print(f"Remaining: {df_filtered.height} rows")

df_filtered.write_csv("AGO2eCLIPManakov2022leftoutimproved.tsv", separator="\t")

