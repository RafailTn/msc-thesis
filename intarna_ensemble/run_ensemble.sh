#!/usr/bin/env bash
# Run the IntaRNA ensemble-energy annotation inside this folder's pixi env.
#
#   ./run_ensemble.sh INPUT.tsv OUTPUT.tsv [extra add_ensemble_energy.py args]
#
# Examples:
#   ./run_ensemble.sh in.tsv out.tsv --threads 12
#   ./run_ensemble.sh in.tsv out.tsv --threads 16 --tacc-w 150 --tacc-l 100
#
# For a very long run on a login node, wrap it so it survives disconnects:
#   nohup ./run_ensemble.sh in.tsv out.tsv --threads 16 > ensemble.log 2>&1 &
#
# If the job is killed for any reason, just run the SAME command again — it
# resumes from where the output file left off.
set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -lt 2 ]]; then
    echo "usage: $0 INPUT.tsv OUTPUT.tsv [extra args]" >&2
    exit 64
fi

pixi run python add_ensemble_energy.py "$@"
