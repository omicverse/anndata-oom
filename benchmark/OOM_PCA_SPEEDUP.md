# OOM-CPU PCA speedup — experiment log

Branch `exp/oom-pca-speedup`. Goal: bring `ov-oom` (anndataoom, CPU) PCA toward
in-memory-CPU PCA speed.

## Setup

50 000-cell × 36 377-gene Tabula Sapiens slice (`/tmp/ts_50k.h5ad`, seed-0
subsample of TS-Stromal), `qc → preprocess(shiftlog|pearson, 2000 HVG) → scale`,
then `pca(n_pcs=50)`. Reference times on the same node (H100/CPU):

- **in-memory CPU `ov.pp.pca` PCA-stage = 4.1 s** (target; data already resident in RAM)
- **OOM CPU `chunked_pca` baseline = 39 s** (densifies the full 36 k-gene scaled chunk per pass)

## Strategy comparison (workflow `oom-pca-strategies`, sequential = clean timing)

| strategy | total | build (I/O) | pca-compute | cos (top-10) | RSS Δ |
|---|---|---|---|---|---|
| baseline (full-width densify) | 39 s | — | — | (randomized) | 72 MB |
| **hvg_from_raw** (HVG-slice before densify) | **13.1 s** | 12.4 s | — | 1.000 | 562 MB |
| hvg_materialize (HVG-slice after full normalize) | 18.2 s | 15.2 s | — | 1.000 | 575 MB |
| streaming_cov (1-pass Gram, no materialise) | 26.0 s | — | — | 1.000 | **2 MB** |
| **cache_then_pca** (build once, time PCA only) | 13.2 s | 12.6 s | **0.65 s** | 1.000 | 563 MB |

`cos` = min over the **top-10** PCs vs an exact `covariance_eigh` reference (the
min over all 50 drops on near-degenerate tail PCs only because the baseline uses
randomized SVD — not an error; the materialised matrix is bit-identical to the
full-width path, max abs diff ~1e-7).

## What was implemented

**HVG-aware materialise** in `chunked_pca` (`_materialize_scaled_hvg`,
`_chunked_ops.py`): slice the ~2 000 HVG columns off the **sparse raw parent**
*before* densifying, instead of densifying the full 36 k-gene panel and throwing
away all but the HVG columns. Normalisation is a per-row scaling and `log1p` is
element-wise, so the HVG column slice commutes through both — the result is
identical.

- **OOM PCA 39 s → 14 s (2.8×)**, end-to-end `ov.pp.pca` 11 s; top-PCs `|cos|=1.000`.
- Zero semantic / RSS change; falls back to the full-width path on any
  unexpected backend. All `test_pca_accuracy` / `test_chunked_correctness` pass.

## Why it doesn't reach the 4.1 s in-memory target (and what would)

Profiling the 13 s build (`cs=5000`):

```
raw sparse disk read (rust backend)   = 10.2 s   ← 80%
+ CSR HVG column slice                = +0.6 s
+ normalize + log1p + scale + densify = +1.8 s
```

**80 % of the OOM PCA time is the raw-matrix disk read** that the in-memory path
already paid at load. With the avoidable Python work now ~2 s, a single OOM PCA
call is fundamentally floored by re-reading from disk; chunk size doesn't help.

To actually match in-memory PCA-*stage* speed, the bounded `n_obs × n_HVG`
scaled block must be made resident so the read isn't re-paid per PCA:

- **cache the HVG scaled block** (e.g. materialise it at `scale`-time, RSS-gated):
  the PCA compute itself is then **0.65 s — beating the 4.1 s target**. Cost: the
  one-time build + `n_obs × n_HVG × 4 B` RAM (≈400 MB @50 k, but ≈16 GB @1 M, so
  it must be threshold-gated to preserve the flat-RSS guarantee at extreme scale).
- or **persist the HVG scaled block to the `.layers.h5` sidecar** at scale-time so
  PCA reads ~400 MB dense instead of re-deriving from the full sparse raw.
- **streaming_cov** is the memory-safe fallback (2 MB RSS, 26 s) for when even the
  HVG block must not be materialised.

## Recommendation

Land the HVG-aware materialise (universal 2.8×, no tradeoff). Add an optional,
RSS-gated HVG-block cache for the "PCA stage == in-memory" regime where the
bounded block fits the memory budget.
