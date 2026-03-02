import pandas as pd
import requests
import time

def fetch_gene_coords_ensembl(gene_name, assembly="GRCh38"):
    """Fetch gene coordinates from Ensembl REST API for hg38."""
    url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene_name}"
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return {
                "chr": str(data["seq_region_name"]),
                "start": int(data["start"]),
                "end": int(data["end"]),
            }
        elif r.status_code == 429:  # rate limit
            time.sleep(1)
            return fetch_gene_coords_ensembl(gene_name, assembly)
        else:
            print(f"  Ensembl lookup failed for {gene_name}: HTTP {r.status_code}")
            return None
    except Exception as e:
        print(f"  Error fetching {gene_name}: {e}")
        return None


def clean_file1(path, output_path=None):
    df = pd.read_csv(path, sep="\t")

    # Identify rows with missing coordinates
    missing_mask = df["start"].isna() | df["end"].isna()
    df_ok = df[~missing_mask].copy()
    df_missing = df[missing_mask].copy()

    if len(df_missing) == 0:
        print("No missing coordinates found.")
        return df

    print(f"Found {len(df_missing)} rows with missing coordinates.")

    rescued = []
    to_fetch = []

    for idx, row in df_missing.iterrows():
        mirna = row["mirna_name"]
        gene = row["gene_name"]

        # Check if a complete duplicate exists (same mirna + same gene with valid coords)
        match = df_ok[
            (df_ok["mirna_name"] == mirna) & (df_ok["gene_name"] == gene)
        ]

        if len(match) > 0:
            print(f"  Dropping {mirna} / {gene}: duplicate with coords exists ({len(match)} rows)")
            # Safe to drop — don't add to rescued or to_fetch
        else:
            to_fetch.append((idx, row))

    # Fetch missing coordinates from Ensembl
    if to_fetch:
        # Cache to avoid re-fetching the same gene
        gene_cache = {}
        print(f"\nFetching coordinates for {len(to_fetch)} rows from Ensembl (hg38)...")

        for idx, row in to_fetch:
            gene = row["gene_name"]

            if gene not in gene_cache:
                print(f"  Looking up {gene}...")
                gene_cache[gene] = fetch_gene_coords_ensembl(gene)
                time.sleep(0.15)  # respect Ensembl rate limits (~6 req/sec)

            coords = gene_cache[gene]
            if coords is not None:
                row_fixed = row.copy()
                row_fixed["chromosome"] = "chr" + coords["chr"]
                row_fixed["start"] = coords["start"]
                row_fixed["end"] = coords["end"]
                rescued.append(row_fixed)
                print(f"  ✓ {gene}: chr{coords['chr']}:{coords['start']}-{coords['end']}")
            else:
                print(f"  ✗ {gene}: could not fetch, dropping row")

    # Combine
    if rescued:
        df_rescued = pd.DataFrame(rescued)
        df_final = pd.concat([df_ok, df_rescued], ignore_index=True)
    else:
        df_final = df_ok.reset_index(drop=True)

    print(f"\nResult: {len(df_final)} rows ({len(df)} original, "
          f"{len(df) - len(df_final)} dropped, {len(rescued)} rescued via Ensembl)")

    if output_path:
        df_final.to_csv(output_path, sep="\t", index=False)

    return df_final


# Usage
df_sites = clean_file1("Homo_sapiens.tsv", output_path="Homo_sapiens_cleaned.tsv")
