#!/bin/bash
# Full clean re-run of the entire benchmark matrix on ONE consistent stack
# (anndataoom 0.1.8 HVG-aware PCA + current omicverse). Replaces the earlier
# results/ that mixed omicverse versions across configs. Skip-if-exists, so
# it resumes after a crash. OOM-killed cells simply produce no JSON.
#
#   DATA=/path/to/ts_data bash scripts/run_all_bench.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE/.."
PY="${PY:-python}"
DATA="${DATA:-/scratch/users/steorra/analysis/anndataoom_benchmark/data}"
LOG="${LOG:-/tmp/all_bench.log}"
echo "=== full re-run start $(date) === DATA=$DATA stack=0.1.8" > "$LOG"

DATASETS=(ts_5k ts_10k ts_vasculature ts_stromal ts_epithelial ts_immune ts_1M)
# light -> heavy within each dataset so the memory-hungry in-memory configs
# get a fresh start; OOM/mixed need the GPU node.
CONFIGS=(ov-oom ov-oom-mixed ov-anndata-implicit scanpy-backed ov-anndata ov-anndata-mixed)

run_cell () {
  local cfg="$1" ds="$2" out="results/${1}__${2}.json"
  [[ -s "$out" ]] && { echo "[skip] $out" | tee -a "$LOG"; return; }
  [[ -f "$DATA/${ds}.h5ad" ]] || { echo "[skip] $DATA/${ds}.h5ad missing" | tee -a "$LOG"; return; }
  echo "" | tee -a "$LOG"; echo ">>> $cfg / $ds @ $(date '+%H:%M:%S')" | tee -a "$LOG"
  timeout 7200 "$PY" scripts/bench.py --input "$DATA/${ds}.h5ad" --config "$cfg" --out "$out" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [[ $rc -eq 124 ]] && echo "  [TIMEOUT]" | tee -a "$LOG"
  [[ $rc -ne 0 && $rc -ne 124 ]] && echo "  [exit $rc — likely OOM-killed]" | tee -a "$LOG"
}

for ds in "${DATASETS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do run_cell "$cfg" "$ds"; done
done

echo "" | tee -a "$LOG"; echo ">>> compat matrix (ts_5k)" | tee -a "$LOG"
[[ -s results/compat_oom_ts5k.json ]] || \
  timeout 3600 "$PY" scripts/compat_matrix.py --input "$DATA/ts_5k.h5ad" --backends oom \
    --out results/compat_oom_ts5k.json 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"; echo "=== full re-run done $(date) ===" | tee -a "$LOG"
