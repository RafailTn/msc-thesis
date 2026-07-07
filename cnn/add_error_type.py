#!/usr/bin/env python3
"""Add an `error_type` column (TP/TN/FP/FN) to an OOF predictions TSV.

Ground truth is the `label` column, prediction is the last column
(`interaction_probability`), thresholded at 0.5. Streams line-by-line so the
2GB file never needs to fit in memory.
"""
import sys

IN = sys.argv[1] if len(sys.argv) > 1 else "data/AGO2_eCLIP_Manakov2022_train_v7_oof.tsv"
OUT = sys.argv[2] if len(sys.argv) > 2 else IN.replace(".tsv", "_errtype.tsv")
THRESH = 0.5

with open(IN) as fin, open(OUT, "w") as fout:
    header = fin.readline().rstrip("\n").split("\t")
    label_idx = header.index("label")
    prob_idx = len(header) - 1  # interaction_probability is the last column
    fout.write("\t".join(header) + "\terror_type\n")

    n = 0
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for line in fin:
        row = line.rstrip("\n").split("\t")
        label = int(float(row[label_idx]))
        pred = 1 if float(row[prob_idx]) >= THRESH else 0
        if label == 1:
            et = "TP" if pred == 1 else "FN"
        else:
            et = "FP" if pred == 1 else "TN"
        counts[et] += 1
        fout.write(line.rstrip("\n") + "\t" + et + "\n")
        n += 1

print(f"rows: {n}")
for k in ("TP", "TN", "FP", "FN"):
    print(f"{k}: {counts[k]}")
