# anndata-oom

[![PyPI](https://img.shields.io/pypi/v/anndataoom.svg)](https://pypi.org/project/anndataoom/)
[![Python](https://img.shields.io/pypi/pyversions/anndataoom.svg)](https://pypi.org/project/anndataoom/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/omicverse/anndata-oom/actions/workflows/ci.yml/badge.svg)](https://github.com/omicverse/anndata-oom/actions/workflows/ci.yml)

**Out-of-memory `AnnData` powered by Rust** — a drop-in replacement for
`anndata.AnnData` that keeps the expression matrix on disk and runs entire
preprocessing pipelines (normalize, log1p, scale, PCA, neighbors, UMAP,
Leiden) as **lazy transforms** or **chunked operations**. The full matrix
is **never** loaded into memory.

Built on top of [scverse/anndata-rs](https://github.com/scverse/anndata-rs),
the Rust implementation of AnnData.

---

## Why?

Standard `anndata.AnnData` loads the entire expression matrix into RAM.
For a million-cell atlas this can mean **100+ GB of memory** — beyond
what most workstations have.

`anndataoom` keeps X on disk (HDF5) and streams it through the
preprocessing pipeline in chunks. Peak RAM is independent of dataset size.

### Memory comparison

| Dataset                | `anndata.AnnData` | `anndataoom`      | Savings |
|------------------------|------------------:|------------------:|--------:|
| PBMC 8k (7.7k × 21k)   | 1.5 GB            | **54 MB**         | 27.8x   |
| 100k cells × 30k genes | ~12 GB            | **~700 MB**       | 17x     |
| 1M cells × 30k genes   | ~120 GB (OOM)     | **~700 MB**       | 170x    |

> Measured end-to-end on a Tabula Sapiens series (5k–1,053,033 cells × 60,606
> genes, full `qc → preprocess → scale → PCA` pipeline, 256 GB cap): peak RSS
> stays ≈ 0.9–5.0 GB while in-memory and `backed` configurations are
> OOM-killed beyond ~228k cells. The 1.05M-cell run finishes in **44.8 min at
> 5.0 GB peak** — the only configuration that completes it. See
> [`benchmark/`](benchmark/).

### How?

Each preprocessing step adds a small "transform descriptor" (a vector or
flag) to a lazy computation chain. Data is computed **on-the-fly during
chunked reads** from the HDF5 file:

```
X (HDF5 on disk, Rust I/O via anndata-rs)
  → TransformedBackedArray      (normalize: ÷ per-cell size factors)
    → TransformedBackedArray    (log1p: on-the-fly)
      → _SubsetBackedArray      (HVG: select 2,000 gene columns)
        → ScaledBackedArray     (z-score: stores only mean/std vectors)
          → Randomized SVD      (chunked matrix products)
            → X_pca             (n_obs × 50, in memory)
              → Neighbors / UMAP / Leiden (operate on X_pca only)
```

| Step                   | What's stored             | Peak memory     |
|------------------------|--------------------------|-----------------|
| Read                   | File handle              | ~0              |
| Normalize              | Per-cell factor vector   | n_obs × 8 B     |
| log1p                  | Flag only                | 0               |
| HVG subset             | Column index             | ~8 KB           |
| Scale                  | Mean + std vectors       | ~32 KB          |
| PCA (working set)      | Y, Q matrices (k=60)     | n_obs × 60 × 8 B|
| X_pca                  | Final embedding          | n_obs × 50 × 4 B|

> For a deeper look at the lazy operator model behind this — how nodes
> compose, how data flows through the chain, and why subsetting stays O(1) —
> see [docs/transform-chain.md](docs/transform-chain.md).

---

## Documentation

- [The Transform Chain](docs/transform-chain.md) — design walkthrough with
  ASCII diagrams of the lazy operator model (`BackedArray`,
  `_SubsetBackedArray`, `TransformedBackedArray`, `ScaledBackedArray`) and
  how chunked reads flow through it.

---

## Installation

### Prebuilt wheels (recommended)

```bash
pip install anndataoom
```

Platform support (verified end-to-end by the
[CI matrix](.github/workflows/ci.yml) — builds the Rust extension from
source and runs the full test suite on each cell):

| Platform                | Python      | Status |
|-------------------------|-------------|--------|
| Linux x86_64            | 3.10, 3.12  | ✅ tested + wheels |
| macOS arm64 (Apple Si)  | 3.10, 3.12  | ✅ tested + wheels |
| macOS x86_64 (Intel)    | 3.10, 3.12  | ✅ tested + wheels |
| Windows x86_64          | 3.10, 3.12  | ✅ tested + wheels (since 0.1.8) |

> **Windows support landed in 0.1.8.** anndata-rs is now vendored in-tree
> (`vendor/anndata-rs/`) and its `anndata-hdf5` gates the HDF5 `threadsafe`
> feature off on Windows — HDF5's CMake refuses thread-safety with a *static*
> library there, which previously broke the build. The vendored static HDF5
> now compiles and the full test suite passes on Windows in CI. Linux aarch64
> wheels are still pending (the `ring` crate fails to cross-compile).

**Wheels bundle a statically-linked HDF5** — on the supported platforms no
system dependencies are needed and no Rust toolchain is required.

### Build from source

If no prebuilt wheel matches your system, `pip` falls back to source.
You'll need a Rust toolchain:

```bash
# Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Build + install
pip install anndataoom
```

Or for development:

```bash
git clone https://github.com/Starlitnightly/anndata-oom
cd anndata-oom
pip install maturin
maturin develop --release
```

---

## Quick start

```python
import anndataoom as oom

# Read an h5ad file — matrix stays on disk
adata = oom.read("large_dataset.h5ad")
print(adata)
```

```
AnnDataOOM                                 [Rust · out-of-memory · backed]
Dimensions:  n_obs: 100,000    n_vars: 30,000

┌───────────┬──────────────────────────────────────┐
│ File      │ large_dataset.h5ad  (1.2 GB on disk) │
│ X         │ csr_matrix · float32 · 5.3% density   │
│ Chunk I/O │ ~20 MB per 1,000-row chunk            │
└───────────┴──────────────────────────────────────┘

▸ obs     (8)    batch · cell_type · n_counts · ...
▸ var     (3)    gene_name · highly_variable · ...
▸ obsm    (–)
▸ layers  (–)
▸ raw     (–)
```

### Chunked operations

```python
# Sum over all cells — streams the matrix in 1000-row chunks
row_sums = adata.X.sum(axis=1)       # ndarray of shape (n_obs,)

# Per-gene means — one-pass chunked Welford's
mean, var = oom.chunked_mean_var(adata)

# Iterate chunks manually
for start, end, chunk in adata.X.chunked(5000):
    # chunk is a csr_matrix (or ndarray) of shape (≤5000, n_vars)
    ...
```

### Subsetting

```python
# All forms of indexing work
sub = adata[0:1000]                          # first 1000 cells
sub = adata[adata.obs["cell_type"] == "B"]   # boolean mask
sub = adata[:, ["GENE1", "GENE2"]]           # by gene name
sub = adata[:, adata.var["highly_variable"]] # after HVG selection

# Returns a new AnnDataOOM — still lazy
print(sub.shape)   # e.g. (17003, 2000)
```

### Single-gene access

```python
# obs_vector reads exactly one column from disk
expr = adata.obs_vector("CD3D")   # ndarray of shape (n_obs,)
```

---

## Integration with omicverse

`omicverse` automatically detects `anndataoom` and uses it as the backend
for `ov.read(..., backend="rust")`:

```python
import omicverse as ov

# Read — returns AnnDataOOM if anndataoom is installed
adata = ov.read("data.h5ad", backend="rust")

# Full preprocessing pipeline — all chunked/lazy
adata = ov.pp.qc(adata,
                 tresh={"mito_perc": 0.2, "nUMIs": 500, "detected_genes": 250},
                 doublets=False)
adata = ov.pp.preprocess(adata, mode="shiftlog|pearson",
                         n_HVGs=2000, target_sum=50 * 1e4)

# HVG subset — returns a new AnnDataOOM
adata.raw = adata
adata = adata[:, adata.var.highly_variable_features]

# Scale + PCA — lazy z-score + chunked randomized SVD
ov.pp.scale(adata)
ov.pp.pca(adata, layer="scaled", n_pcs=50)

# Neighbors / UMAP / Leiden — operate on obsm['X_pca'], no matrix touch
ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50,
                use_rep="scaled|original|X_pca")
ov.pp.umap(adata)
ov.pp.leiden(adata, resolution=1)

# Plotting — all ov.pl.* functions work directly, incl. use_raw=True
ov.pl.embedding(adata, basis="X_umap", color="leiden")
ov.pl.dotplot(adata, marker_genes, groupby="leiden")
ov.pl.violin(adata, keys="CD3D", groupby="leiden", use_raw=True)
```

### CPU–GPU mixed mode

`anndataoom` (storage backend) is orthogonal to `omicverse`'s execution
mode (compute), so you can flip on GPU acceleration with an OOM-backed
`AnnData` and pay no penalty:

```python
import omicverse as ov
ov.settings.cpu_gpu_mixed_init()       # route PCA/neighbors/UMAP to torch-GPU
adata = oom.read("data.h5ad")          # still out-of-memory
# … identical qc → preprocess → scale → pca → neighbors → umap pipeline …
```

What changes (measured on the Tabula Sapiens benchmark, H100):

- **Memory-bound preprocessing is mode-invariant.** `qc → preprocess →
  scale → pca` run at the *same* wall-clock and RSS in `cpu` and
  `cpu-gpu-mixed`, with **zero** GPU memory used. This is by design:
  anndataoom's chunked operators are pure-CPU, and omicverse routes the
  OOM PCA through `anndataoom.chunked_pca` (never the torch-GPU solver).
  Peak RSS stays flat and bounded by chunk size regardless of mode.
- **Downstream embedding gets the GPU.** Steps that operate on the small
  `(n_obs × 50)` PCA embedding *do* offload: `ov.pp.neighbors` runs on a
  CUDA PyG-kNN backend (device memory allocated) and `ov.pp.umap`,
  `ov.pp.mde` run on the GPU. The kNN itself is sub-second on a warm GPU;
  the exact wall-clock gain at small scale is measurement-sensitive, so
  we just note the offload is real. PCA results are bit-identical between
  modes (|cos| = 1.0).
- **In-memory backend, by contrast, gets a big PCA win.** With a plain
  `anndata.AnnData` (not OOM), `ov.pp.pca` dispatches to the GPU
  `torch_pca` solver: PCA dropped **13×** (13.9 s → 1.1 s on TS-5k),
  ~1.6× on the whole pipeline. anndataoom forgoes this by design — it
  routes through CPU `chunked_pca` to keep peak RSS flat.

### omicverse function compatibility

Compatibility of `ov.pp.*` against an `AnnDataOOM` backend — **22 of 24
probed functions** run on the OOM path (probed in both `cpu` and
`cpu-gpu-mixed`; `ᴳ` = offloads to GPU in mixed mode, `◐` = bounded
materialisation of only the HVG subset):

| function | OOM | notes |
|---|:---:|---|
| `qc`, `preprocess`, `normalize_total`, `log1p`, `identify_robust_genes` | ✅ | core pipeline |
| `scale`, `pca` | ✅ | lazy / chunked, CPU |
| `neighbors`ᴳ, `umap`ᴳ, `leiden`, `louvain` | ✅ | operate on `X_pca` only |
| `tsne`, `mde`ᴳ, `sude` | ✅ | embeddings on `X_pca` (`sude` errors in mixed) |
| `filter_cells`, `filter_genes` | ✅ | chunked stat pass + lazy `_inplace_subset_*` |
| `highly_variable_genes` | ✅ | chunked; `seurat`/`cell_ranger` (dispersion) + `pearson` |
| `highly_variable_features` | ✅ | chunked pegasus (mean/var pass + LOESS) |
| `normalize_pearson_residuals` | ✅ | lazy `PearsonResidualBackedArray` (analytic, Lause 2021) |
| `regress` | ✅ | lazy `RegressedBackedArray`; one chunked OLS pass; honours custom `keys` |
| `score_genes_cell_cycle` | ✅ | per-cell gene-set means over small column reads |
| `scrublet` | ◐ | materialises only the HVG subset (`n_obs × n_HVG`), not the full matrix |
| `anndata_to_GPU` / `anndata_to_CPU` | ➖ | require optional `rapids_singlecell` |

Failures raise a clear exception at the call site — they never silently
mis-compute. Reproduce with
[`benchmark/scripts/compat_matrix.py`](benchmark/scripts/compat_matrix.py).

---

## Full API reference

### Top-level

| Function / Class                    | Description                                          |
|-------------------------------------|------------------------------------------------------|
| `oom.read(path, backed='r')`        | Read an .h5ad file → `AnnDataOOM`                    |
| `oom.AnnDataOOM`                    | Out-of-memory AnnData (full `anndata.AnnData` API)   |
| `oom.BackedArray`                   | Lazy row-chunked wrapper over anndata-rs X           |
| `oom.TransformedBackedArray`        | Lazy normalize / log1p transform chain node          |
| `oom.ScaledBackedArray`             | Lazy z-score transform                               |
| `oom.PearsonResidualBackedArray`    | Lazy analytic Pearson-residual transform             |
| `oom.RegressedBackedArray`          | Lazy covariate regress-out transform                 |
| `oom.is_oom(obj)`                   | Check if `obj` is an `AnnDataOOM`                    |
| `oom.oom_guard(...)`                | Decorator: auto-materialise for in-memory functions  |
| `oom.concat(adatas)`                | Concatenate multiple AnnData                         |

### Chunked preprocessing

| Function                                    | Description                                        |
|---------------------------------------------|----------------------------------------------------|
| `chunked_qc_metrics(adata)`                 | nUMIs, detected_genes, n_cells per gene            |
| `chunked_gene_group_pct(adata, mask)`       | Per-cell fraction of counts in a gene group        |
| `chunked_normalize_total(adata, target_sum)`| Lazy normalize-total                               |
| `chunked_log1p(adata)`                      | Lazy log1p                                         |
| `chunked_mean_var(adata)`                   | Welford's mean + var per gene                      |
| `chunked_identify_robust_genes(adata)`      | Filter low-expression genes                        |
| `chunked_highly_variable_genes_pearson(...)`| Pearson residuals HVG selection (2 passes)         |
| `chunked_scale(adata)`                      | Lazy z-score                                       |
| `chunked_pca(adata)`                        | Randomized SVD, 3-path (materialise / implicit / Halko) |
| `chunked_highly_variable_genes_dispersion(...)`| Seurat / Cell Ranger dispersion HVG (chunked)   |
| `chunked_highly_variable_features_pegasus(...)`| Pegasus HVF (mean/var pass + LOESS)             |
| `chunked_normalize_pearson_residuals(...)`  | Lazy analytic Pearson residuals (Lause 2021)       |
| `chunked_regress(adata, keys=...)`          | Covariate regress-out — one chunked OLS pass       |
| `chunked_scrublet_prepare(adata)`           | Bounded HVG-subset prep for out-of-core Scrublet   |

### `AnnDataOOM` methods

All `anndata.AnnData` methods and properties are supported. Key ones:

| Property / method                    | Behaviour                                        |
|--------------------------------------|--------------------------------------------------|
| `.shape`, `.n_obs`, `.n_vars`        | Dimensions                                       |
| `.obs`, `.var`                       | Pandas DataFrames (eagerly loaded; small)        |
| `.X`                                 | Lazy `BackedArray` (never loaded)                |
| `.obsm`, `.varm`, `.obsp`, `.varp`   | Dict-of-ndarray (loaded; typically small)        |
| `.layers`                            | `BackedLayers` dict (sidecar HDF5)               |
| `.raw`                               | `_FrozenRaw` snapshot (shares backing file)      |
| `.obs_vector(key)`                   | One column from disk (no full load)              |
| `.chunked_X(chunk_size=1000)`        | Row-chunked iterator                             |
| `adata[idx]`                         | Subsetting (returns new `AnnDataOOM`)            |
| `adata.copy()`                       | Shallow copy (shares backing file, no RAM cost)  |
| `adata.to_adata()`                   | Materialize to standard `anndata.AnnData`        |
| `adata.write(path)`                  | Chunked write — doesn't materialize              |
| `adata.close()`                      | Release file handle                              |
| `repr(adata)` / `_repr_html_()`      | Pretty text / Jupyter display                    |

---

## Benchmark: PBMC 8k (7,750 cells × 20,939 genes)

Full preprocessing pipeline (QC → normalize → HVG → scale → PCA → neighbors → UMAP → Leiden):

| Step             | Python (MB) | anndataoom (MB) |
|------------------|------------:|----------------:|
| read             | 148         | **37**          |
| qc               | 280         | **54**          |
| preprocess       | 328         | **24**          |
| hvg_subset       | 450         | **24**          |
| scale            | 382         | **54**          |
| pca              | 846         | **33**          |
| neighbors        | 1195        | **33**          |
| umap             | 1500        | **34**          |
| leiden           | 1502        | **33**          |
| **Peak**         | **1502**    | **54**          |

→ **27.8× memory savings** on this small dataset; ratio grows with scale.

---

## Supported h5ad formats

| X format     | Reading | Lazy ops     | Notes                                |
|--------------|:-------:|:------------:|--------------------------------------|
| Dense ndarray| ✅      | ✅           | float32 / float64                    |
| CSR sparse   | ✅      | ✅           | Most common scRNA-seq format         |
| CSC sparse   | ✅      | ✅           | Column-oriented                      |

`anndataoom` automatically preserves sparsity through `normalize` and `log1p`
(sparse → sparse), and materializes to dense only where algorithmically
necessary (z-score, PCA).

---

## Architecture

`anndataoom` is a thin Python wrapper over [scverse/anndata-rs](https://github.com/scverse/anndata-rs):

```
┌──────────────────────────────────────────────┐
│  anndataoom (Python package)                 │
│  ┌────────────────────────────────────────┐  │
│  │  AnnDataOOM                            │  │
│  │  ├─ obs, var (pandas.DataFrame)        │  │
│  │  ├─ obsm, varm (dict of ndarray)       │  │
│  │  ├─ layers (BackedLayers — sidecar H5) │  │
│  │  └─ X (BackedArray — wraps ↓)          │  │
│  └────────────────────────────────────────┘  │
│            │                                  │
│            ▼                                  │
│  ┌────────────────────────────────────────┐  │
│  │  anndataoom._backend  (Rust extension)│  │
│  │  ├─ AnnData (pyanndata)                │  │
│  │  ├─ PyArrayElem (chunked() iterator)   │  │
│  │  └─ Statically linked:                 │  │
│  │     ├─ anndata (Rust crate)            │  │
│  │     ├─ anndata-hdf5                    │  │
│  │     └─ HDF5 C library                  │  │
│  └────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘
```

The Rust extension (`anndataoom._backend`) is pinned to a specific commit
of `scverse/anndata-rs` for reproducible builds (the same commit used by
SnapATAC2).

---

## Limitations and caveats

- **Writing back to X is lazy** — modifications via `adata[mask] = value` materialize
  X in memory. Use `adata.obs`, `adata.obsm`, or `adata.write(path)` to persist changes.
- **PCA**: `chunked_pca` auto-selects among three paths. By default it subsets to
  the HVGs first (effective `n_vars` ≈ 2,000), materialises just that block, and
  runs sklearn `randomized_svd` as a single in-memory SVD — bit-identical to
  standard PCA on the leading components used downstream (`|cos| = 1.0`). When the
  HVG block does not fit, it runs an implicit-centering randomized SVD over the
  sparse normalize+log1p view without densifying; a per-chunk Halko path is the
  fallback. `n_power_iters` (default 4) is tunable.
- **Some ops require materialization**: `score_genes_cell_cycle`, `find_markers`,
  non-Harmony batch correction, etc. These auto-materialize with a warning.
- **File mode**: Default `backed='r'` (read-only) protects the source file.
  Use `backed='r+'` if you need to write back (advanced).
- **Concurrent access**: HDF5 files default to exclusive locking. Set
  `HDF5_USE_FILE_LOCKING=FALSE` in the environment if multiple processes
  need to read the same file.

---

## Comparison with alternatives

| Feature                           | `anndata`    | `anndata` (backed='r') | `anndataoom`       |
|-----------------------------------|:------------:|:----------------------:|:------------------:|
| Read without loading matrix       | ❌            | ✅                      | ✅                  |
| Subset (lazy view)                | ✅ (view)     | ✅ (view)               | ✅ (new AnnDataOOM) |
| Chunked iteration                 | ❌            | ❌ (manual)             | ✅                  |
| normalize / log1p                 | In-memory    | ❌ (read-only)          | ✅ (lazy transform) |
| scale                             | In-memory    | ❌                      | ✅ (lazy z-score)   |
| PCA                               | Full SVD     | ❌                      | ✅ (chunked rSVD)   |
| Plotting (scanpy/omicverse)       | ✅            | Limited                 | ✅ (via omicverse)  |
| Modify obs/var                    | ✅            | ❌                      | ✅                  |
| Peak RAM (1M × 30k)               | ~120 GB      | — (can't process)       | ~700 MB            |

---

## Development

```bash
git clone https://github.com/Starlitnightly/anndata-oom
cd anndata-oom

# Install Rust (first time)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Build in editable mode
pip install maturin
maturin develop --release

# Run tests
pip install pytest
pytest tests/
```

### Contributing

Contributions welcome! Areas of interest:

- **More lazy transforms**: Harmony, scVI integration (regress-out, Pearson
  residuals, and dispersion/pegasus HVG are now implemented)
- **Zarr backend**: currently only HDF5 supported
- **Dask interop**: expose `BackedArray` as a `dask.array`
- **Query engine**: SQL-like filtering over chunks

### Release process

1. Bump version in `pyproject.toml` and `Cargo.toml`
2. Update `CHANGELOG.md`
3. Commit, tag, push:
   ```bash
   git commit -am "Release v0.x.0"
   git tag v0.x.0
   git push && git push --tags
   ```
4. GitHub Actions builds wheels for all platforms and publishes to PyPI
   (via [trusted publishing](https://docs.pypi.org/trusted-publishers/))

---

## License

MIT License — see [LICENSE](LICENSE).

Built on [scverse/anndata-rs](https://github.com/scverse/anndata-rs) (MIT,
© Kai Zhang).

---

## Citation

If you use `anndataoom` in published research, please cite:

```
@software{omicverse,
  title  = {OmicVerse: A framework for multi-omic data analysis},
  author = {Zeng, Z. et al.},
  url    = {https://github.com/Starlitnightly/omicverse},
  year   = {2024},
}

@software{anndata_rs,
  title  = {anndata-rs: Rust implementation of AnnData},
  author = {Zhang, Kai},
  url    = {https://github.com/scverse/anndata-rs},
  year   = {2022},
}
```
