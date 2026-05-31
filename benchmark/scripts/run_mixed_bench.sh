#!/bin/bash
# Run the cpu-gpu-mixed half of the bench matrix and compare against the
# existing CPU baselines (ov-oom / ov-anndata) already in results/.
#
# Two mode-paired comparisons:
#   ov-oom        (cpu)  vs  ov-oom-mixed        — out-of-core path
#   ov-anndata    (cpu)  vs  ov-anndata-mixed    — in-memory dense path
#
# The `-mixed` configs call ov.settings.cpu_gpu_mixed_init() so PCA /
# neighbors / UMAP route to torch-GPU. Everything else is identical to
# run_full_bench.sh. Skips a cell whose JSON already exists.
#
# Data lives outside the repo (too large for git). Override with DATA=...
#   DATA=/path/to/ts_data bash scripts/run_mixed_bench.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
PY="${PY:-python}"
DATA="${DATA:-/scratch/users/steorra/analysis/anndataoom_benchmark/data}"
LOG="${LOG:-/tmp/mixed_bench.log}"
echo "=== mixed bench start $(date) === DATA=$DATA" > "$LOG"

# OOM path: all 7 scales (matches the ov-oom cpu baselines).
OOM_DS=(ts_5k ts_10k ts_vasculature ts_stromal ts_epithelial ts_immune ts_1M)
# In-memory dense path: only the scales that have an ov-anndata cpu baseline
# (immune/1M OOM-killed the dense path under the paper's 256 GB cap).
ANN_DS=(ts_5k ts_10k ts_vasculature ts_stromal ts_epithelial)

run_cell () {
  local cfg="$1" ds="$2"
  local out="results/${cfg}__${ds}.json"
  if [[ -s "$out" ]]; then echo "[skip] $out exists" | tee -a "$LOG"; return; fi
  if [[ ! -f "$DATA/${ds}.h5ad" ]]; then echo "[skip] $DATA/${ds}.h5ad missing" | tee -a "$LOG"; return; fi
  echo "" | tee -a "$LOG"; echo ">>> $cfg / $ds @ $(date '+%H:%M:%S')" | tee -a "$LOG"
  timeout 7200 "$PY" scripts/bench.py --input "$DATA/${ds}.h5ad" --config "$cfg" --out "$out" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  if [[ $rc -eq 124 ]]; then echo "  [TIMEOUT, marking incomplete]" | tee -a "$LOG"
  elif [[ $rc -ne 0 ]]; then echo "  [exit $rc — likely OOM-killed]" | tee -a "$LOG"; fi
}

# Small -> large so a crash on the big cells still leaves the small results.
for ds in "${OOM_DS[@]}"; do run_cell ov-oom-mixed "$ds"; done
for ds in "${ANN_DS[@]}"; do run_cell ov-anndata-mixed "$ds"; done

echo "" | tee -a "$LOG"; echo "=== mixed bench done $(date) ===" | tee -a "$LOG"
