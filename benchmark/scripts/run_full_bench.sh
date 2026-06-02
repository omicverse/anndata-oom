#!/bin/bash
# Run the full 4-config x 7-dataset bench matrix.
# Order: small -> large dataset; within each, light -> heavy config so
# the more memory-hungry runs get a fresh start and do not compete
# with leftover RSS from a prior run. Skips a cell if its result JSON
# already exists and is non-empty (lets you resume after a crash).
#
# Run from the benchmark/ directory:
#   bash scripts/run_full_bench.sh
# Override the python binary or log location with env vars:
#   PY=python3.10 LOG=/var/log/bench.log bash scripts/run_full_bench.sh
set -uo pipefail
# Move to benchmark/ root (this script lives in benchmark/scripts/).
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
PY="${PY:-python}"
LOG="${LOG:-/tmp/full_bench.log}"
echo "=== full bench start $(date) ===" > "$LOG"

# Small → large; within each, run the cheap configs first so the more
# memory-hungry ones get a fresh start and don't compete with leftover
# RSS from a prior run.
DATASETS=(ts_5k ts_10k ts_vasculature ts_stromal ts_epithelial ts_immune ts_1M)
CONFIGS=(ov-oom ov-anndata-implicit scanpy-backed ov-anndata)

for ds in "${DATASETS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    out=results/${cfg}__${ds}.json
    if [[ -s "$out" ]]; then
      echo "[skip] $out exists" | tee -a "$LOG"
      continue
    fi
    if [[ ! -f data/${ds}.h5ad ]]; then
      echo "[skip] data/${ds}.h5ad missing" | tee -a "$LOG"
      continue
    fi
    echo "" | tee -a "$LOG"
    echo ">>> $ds / $cfg @ $(date '+%H:%M:%S')" | tee -a "$LOG"
    # Hard timeout 90 min per job (longest ov-oom on 1M was 57 min).
    timeout 5400 "$PY" scripts/bench.py \
      --input data/${ds}.h5ad \
      --config $cfg \
      --out $out 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if [[ $rc -eq 124 ]]; then
      echo "  [TIMEOUT after 90 min, marking incomplete]" | tee -a "$LOG"
    elif [[ $rc -ne 0 ]]; then
      echo "  [exit $rc — likely OOM-killed, marking incomplete]" | tee -a "$LOG"
    fi
  done
done
echo "" | tee -a "$LOG"
echo "=== full bench done $(date) ===" | tee -a "$LOG"
