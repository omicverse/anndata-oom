# anndataoom benchmark

End-to-end benchmark of the omicverse pipeline (`qc → preprocess →
scale → pca`) under four storage / scaling configurations, on seven
real Tabula Sapiens slices from cellxgene spanning $5\,000$ to
$1\,053\,033$ cells. Reproduces the technical report at
[`paper/main.pdf`](paper/main.pdf).

## What lives where

```
benchmark/
├── scripts/
│   ├── bench.py                 — per-cell measurement harness
│   │                              (one config × one dataset → one JSON)
│   ├── plot.py                  — JSON → figures + LaTeX tables + macros
│   ├── concat_1M.py             — build ts_1M.h5ad from three TS tissues
│   ├── run_full_bench.sh        — drive the full 4 × 7 matrix sequentially
│   ├── run_mixed_bench.sh       — drive the cpu-gpu-mixed half of the matrix
│   ├── compat_matrix.py         — omicverse-function × {cpu,mixed} compatibility
│   │                              probe on an AnnDataOOM backend
│   └── test_implicit_scale.py   — standalone validator for CenteredSparseArray
│                                  (per-gene / per-cell / matmul vs textbook ref)
├── results/                     — one JSON per (config, dataset) cell;
│                                  compat_*.json for the compatibility matrix
├── figures/                     — fig_totaltime, fig_stages, fig_mixed (.pdf/.png)
└── paper/
    ├── main.tex / main.pdf
    ├── numbers.tex              — hand-curated macros (v0.1.6 PCA story)
    ├── numbers_auto.tex         — written by plot.py from the JSONs
    ├── numbers_mixed.tex        — written by plot.py (cpu-gpu-mixed + compat)
    └── table_*.tex              — written by plot.py (incl. table_mixed,
                                   table_compat)
```

The 23 JSONs in `results/` are the metrics from the run reported in the
paper (`anndataoom v0.1.7`, `omicverse master` at PRs #802, #804, #805
merged, single CPU node with a 256 GB per-process RSS cap). Five (config,
dataset) cells were OOM-killed by that cap and have no JSON; the missing
files are flagged as `OOM` in the headline table.

## The four configurations

| config | input load | scale-step backend |
|---|---|---|
| `ov-anndata` | `anndata.read_h5ad(path)` | dense materialisation |
| `ov-anndata-implicit` | `anndata.read_h5ad(path)` | `ov.pp.scale(use_implicit_centering=True)` — `CenteredSparseArray` wrapper |
| `scanpy-backed` | `anndata.read_h5ad(path, backed='r')` | `sc.pp.scale` after `adata.to_memory()` |
| `ov-oom` | `anndataoom.read(path)` | lazy `ScaledBackedArray` |

All four go through the same logical stages (`qc → preprocess → scale →
pca`). The `scanpy-backed` config uses scanpy's own pipeline functions
because backed-`'r'` `_CSRDataset` does not dispatch through the same
omicverse paths.

## CPU–GPU mixed mode (v0.1.7 supplementary)

Two extra configs run the **identical** pipeline under
`ov.settings.cpu_gpu_mixed_init()`, which routes PCA / neighbors / UMAP /
clustering to torch-GPU:

| config | backend | mode |
|---|---|---|
| `ov-oom-mixed`     | `anndataoom.read(path)`   | `cpu-gpu-mixed` |
| `ov-anndata-mixed` | `anndata.read_h5ad(path)` | `cpu-gpu-mixed` |

Run with `bash scripts/run_mixed_bench.sh` (needs a CUDA/MPS torch); each
mixed JSON pairs with its `cpu` twin already in `results/`. `bench.py`
records per-stage `gpu_peak_mb` and the device name.

**Headline finding.** On the OOM backend the `qc → preprocess → scale →
pca` pipeline is essentially **mode-invariant** — same wall-clock, same
RSS, **zero GPU memory** — because anndataoom's chunked operators are
pure-CPU and omicverse routes the OOM PCA through `anndataoom.chunked_pca`
rather than the torch-GPU solver. The GPU only engages for the downstream
graph/embedding steps that operate on the small `(n_obs × 50)` PCA matrix
(`ov.pp.neighbors` ~8× faster on TS-5k). `compat_matrix.py` records the
full `ov.pp.*` compatibility table (`results/compat_*.json` →
`paper/table_compat.tex`).

```bash
python scripts/compat_matrix.py --input data/ts_5k.h5ad \
    --backends oom --out results/compat_oom_ts5k.json
```

## Reproducing

### 1. Environment

```bash
# Python deps
pip install anndataoom==0.1.7 omicverse  scanpy>=1.10  scikit-learn  numpy  scipy  pandas  numba  psutil  anndata
```

`omicverse` should be at master after PRs #802, #804, #805 to get the
HVG-aware OOM PCA dispatch, the `use_implicit_centering` kwarg, and the
fused normalize↔HVG precompute. Older releases will silently fall back
to behaviour reported as `v0.1.6 baseline` in the paper.

### 2. Datasets

The seven inputs are Tabula Sapiens slices retrieved from cellxgene's
collection
(<https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5>).
Place them at `data/` next to the `benchmark/` parent (the bench
scripts use relative paths from that working directory). The
`layers['decontXcounts']` raw counts are auto-promoted to `.X` by
`bench.py` if `.X` looks log-normalised (the cellxgene default).

| key | n_obs | source |
|---|---|---|
| `ts_vasculature` | 42,650 | TS Vasculature (single tissue from cellxgene) |
| `ts_5k` | 5,000 | random subsample of TS-Vasculature, seed 0 |
| `ts_10k` | 10,000 | random subsample of TS-Vasculature, seed 0 |
| `ts_stromal` | 232,684 | TS Stromal compartment |
| `ts_epithelial` | 228,032 | TS Epithelial compartment |
| `ts_immune` | 592,317 | TS Immune compartment |
| `ts_1M` | 1,053,033 | row-concat of the three compartments (see `scripts/concat_1M.py`) |

### 3. Run

```bash
# All 28 cells (skip-if-exists; will quietly resume after a crash).
bash benchmark/scripts/run_full_bench.sh
```

Wall-clock on a 17-core / 1.5 TB Sherlock node was ~4.5 hours total;
five (config, dataset) cells hit the 256 GB per-process cap and are
recorded as missing files.

### 4. Re-render figures + tables + paper

```bash
python benchmark/scripts/plot.py
cd benchmark/paper
tectonic main.tex
```

`plot.py` reads all `results/*.json`, writes `figures/fig_*.{pdf,png}`
+ `paper/table_{datasets,headline}.tex` + `paper/numbers_auto.tex`.
`tectonic` then compiles `paper/main.tex` (which `\input{}`s the
generated tables and macros) to `paper/main.pdf`.

## Validation harness

`scripts/test_implicit_scale.py` is a standalone numerical
equivalence + memory check for the `CenteredSparseArray` wrapper
introduced in v0.1.7:

```bash
python benchmark/scripts/test_implicit_scale.py path/to/ts_10k.h5ad
```

Reports max-absolute-error of (`[:, gene_idx]`, `[i, :]`, `@ W`) vs
the textbook `np.clip((X-μ)/σ, -inf, max_value)` reference, plus the
wrapper storage size next to the dense buffer it replaces.

## Citing

If you use this benchmark or anndataoom in your work, please cite the
project repository <https://github.com/omicverse/anndata-oom>.
