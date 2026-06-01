# Changelog

## 0.1.8

### Performance
- **HVG-aware PCA materialise.** `chunked_pca` previously densified the full
  gene panel (~36 k genes) per chunk and then kept only the ~2 000 HVG columns.
  It now slices the HVG columns off the *sparse* raw parent **before**
  densifying (normalisation is per-row, `log1p` element-wise, so the slice
  commutes), so only a `(chunk × n_HVG)` block is ever dense. On a
  50 k × 36 k panel the PCA materialise drops **~39 s → ~14 s (2.8×)**,
  bit-identical to the previous result (max abs diff ~1e-7; top-PC
  `|cos| = 1.0`). See `benchmark/OOM_PCA_SPEEDUP.md`.

### Build / Platform
- **Vendored `anndata-rs` in-tree** (`vendor/anndata-rs/`, rev `aa15c8d`).
  The Rust I/O layer is no longer pulled as a git dependency — source builds
  are self-contained and no longer fetch from GitHub. (Python runtime
  dependencies — `anndata`, `numpy`, … — are unchanged.)
- **Windows x86_64 support.** The vendored `anndata-hdf5` gates the HDF5
  `threadsafe` feature off on Windows (HDF5's CMake refuses thread-safety with
  a static library there), so the vendored static HDF5 now builds on Windows.
  Windows wheels are built in `release.yml` and the test suite runs on Windows
  in CI.

### CI
- Cross-platform test workflow (`.github/workflows/ci.yml`) building the Rust
  extension from source and running the test suite on Linux / macOS
  (arm64 + x86_64) / Windows across Python 3.10 and 3.12.
