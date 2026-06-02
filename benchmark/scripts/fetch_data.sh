#!/bin/bash
# Assemble the seven benchmark datasets under benchmark/../data/.
#
# The raw inputs are Tabula Sapiens compartment slices from the cellxgene
# collection (~57 GB total) — too large to ship in git. This script
# documents their provenance and scripts the two derived steps
# (subsampling + 1M concat). The three compartment downloads are manual
# because cellxgene serves per-version presigned URLs that expire; grab
# them from the collection page and drop the .h5ad files into data/.
#
# Collection:
#   https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5
#
# Required downloads → data/ :
#   ts_vasculature.h5ad   TS Vasculature           (42,650 cells)
#   ts_stromal.h5ad       TS Stromal compartment   (232,684)
#   ts_epithelial.h5ad    TS Epithelial compartment(228,032)
#   ts_immune.h5ad        TS Immune compartment    (592,317)
# All on the shared 60,606-gene panel; .X may be log-normalised with raw
# counts in layers['decontXcounts'] (bench.py / the helpers promote it).
#
# Derived (scripted below):
#   ts_5k.h5ad, ts_10k.h5ad   seed-0 subsamples of ts_vasculature
#   ts_1M.h5ad                row-concat of stromal+epithelial+immune
#
# Run from the benchmark/ directory:  bash scripts/fetch_data.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE/.."
PY="${PY:-python}"
mkdir -p data

need=(ts_vasculature ts_stromal ts_epithelial ts_immune)
missing=()
for d in "${need[@]}"; do [[ -f data/$d.h5ad ]] || missing+=("$d.h5ad"); done
if (( ${#missing[@]} )); then
  echo "Missing source h5ads in data/ : ${missing[*]}"
  echo "Download them from the cellxgene collection (URL above) and re-run."
  exit 1
fi

echo ">>> subsampling ts_5k / ts_10k (seed 0)"
"$PY" scripts/subsample.py

echo ">>> building ts_1M (stromal + epithelial + immune)"
"$PY" scripts/concat_1M.py

echo "=== data ready ==="
ls -lh data/ts_*.h5ad
