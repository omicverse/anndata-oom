"""
Chunked preprocessing operations for AnnDataOOM.

Each function operates on the matrix in row-chunks, never loading the
full (n_obs x n_vars) matrix into memory.  After HVG subsetting the
matrix is small enough to materialise — these functions handle the
expensive pre-subset steps.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix, vstack as sparse_vstack

if TYPE_CHECKING:
    from ._core import AnnDataOOM

from ._backed_array import BackedArray, DEFAULT_CHUNK_SIZE

logger = logging.getLogger(__name__)


# ======================================================================
# TransformedBackedArray — lazy per-row transform over a BackedArray
# ======================================================================


class TransformedBackedArray(BackedArray):
    """BackedArray that applies element-wise transforms on read.

    Keeps a chain of per-row transforms (normalize, log1p, ...) so that
    ``chunk = transform(raw_chunk)`` is computed on-the-fly during
    chunked iteration, without ever writing a full transformed matrix.

    Parameters
    ----------
    parent : BackedArray
        The underlying (possibly already transformed) array.
    norm_factors : ndarray, shape (n_obs,), optional
        Per-cell total-count normalization divisors.  Each row is divided
        by its corresponding factor.
    apply_log1p : bool
        Whether to apply ``log(1 + x)`` after normalisation.
    """

    def __init__(
        self,
        parent: BackedArray,
        norm_factors: np.ndarray | None = None,
        apply_log1p: bool = False,
    ):
        # Bypass BackedArray.__init__; we override _read_rows
        self._parent = parent
        self._shape = parent.shape
        self._elem = None
        self._is_rs = False  # Always uses Python-level iteration
        self._norm_factors = norm_factors
        self._apply_log1p = apply_log1p

    def _transform_chunk(self, data, global_start: int):
        """Apply stored transforms to a chunk of rows.

        Preserves sparsity through both normalize and log1p when possible:
        - normalize (÷factor) preserves zeros → sparse stays sparse
        - log1p(0) = 0 → sparse stays sparse
        """
        if self._norm_factors is not None:
            factors = self._norm_factors[global_start:global_start + data.shape[0]]
            factors = factors.astype(np.float64)
            factors[factors == 0] = 1.0
            if issparse(data):
                # Scale each row by 1/factor using left multiplication with a diagonal
                from scipy.sparse import diags
                data = data.astype(np.float32)
                data = diags(1.0 / factors.astype(np.float32)) @ data
            else:
                data = np.array(data, dtype=np.float32)
                data /= factors[:, np.newaxis].astype(np.float32)

        if self._apply_log1p:
            if issparse(data):
                # log1p(0) = 0, so apply only to .data — preserves sparsity
                data = data.copy()
                data.data = np.log1p(data.data)
            else:
                data = np.log1p(data)

        return data

    def _read_rows(self, start: int, end: int):
        end = min(end, self._shape[0])
        if start >= end:
            return np.empty((0, self._shape[1]), dtype=np.float32)
        raw = self._parent._read_rows(start, end)
        return self._transform_chunk(raw, start)

    def _read_row_indices(self, indices):
        raw = self._parent._read_row_indices(indices)
        # For non-contiguous indices, need to gather the right factors
        if self._norm_factors is not None or self._apply_log1p:
            # Build a temporary contiguous mapping
            result = np.empty((len(indices), self._shape[1]), dtype=np.float32)
            if self._norm_factors is not None:
                factors = self._norm_factors[indices].astype(np.float64)
                factors[factors == 0] = 1.0
                if issparse(raw):
                    raw = raw.toarray()
                raw = np.array(raw, dtype=np.float64)
                raw /= factors[:, np.newaxis]
                result[:] = raw.astype(np.float32)
            else:
                if issparse(raw):
                    result[:] = raw.toarray()
                else:
                    result[:] = raw
            if self._apply_log1p:
                np.log1p(result, out=result)
            return result
        return raw

    def chunked(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        """Iterate chunks of rows with transforms applied lazily.

        Delegates row iteration to the **parent** array's ``chunked()``
        so we get its native (rust-backed) O(n) iterator, then applies
        ``_transform_chunk`` per chunk. Inheriting ``BackedArray.chunked``
        would instead call ``self._read_rows(start, end)`` for every
        chunk — and although the underlying parent ``_read_rows`` is
        now O(end-start), routing every chunk through Python-level
        slice reads still costs ~10× more than the native iterator.

        Critical for `chunked_mean_var` / `chunked_pca` / any other
        consumer that walks the full matrix after ``normalize`` +
        ``log1p`` / ``scale``: those callers see the wrapped X, and
        without this override they pay the per-chunk Python-slice tax
        on every pass.
        """
        for start, end, raw_chunk in self._parent.chunked(chunk_size):
            yield start, end, self._transform_chunk(raw_chunk, start)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.float32)


# ======================================================================
# Chunked QC metrics
# ======================================================================


def chunked_qc_metrics(
    adata: AnnDataOOM,
    *,
    percent_top: list[int] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Compute QC metrics without materialising X.

    Sets ``adata.obs['nUMIs']``, ``adata.obs['detected_genes']``, and
    ``adata.var['n_cells']`` (number of cells expressing each gene).
    """
    X = adata.X
    n_obs, n_vars = X.shape

    nUMIs = np.zeros(n_obs, dtype=np.float64)
    n_genes_per_cell = np.zeros(n_obs, dtype=np.int64)
    n_cells_per_gene = np.zeros(n_vars, dtype=np.int64)

    for start, end, chunk in X.chunked(chunk_size):
        if issparse(chunk):
            nUMIs[start:end] = np.asarray(chunk.sum(axis=1)).ravel()
            # Sparse-native getnnz avoids the implicit densification that
            # `(chunk != 0)` triggers via __ne__ broadcast — at 1M cells x
            # 30k genes that densification was the dominant qc cost.
            # Bit-exact equivalent: count of stored entries (no zero values
            # are stored in canonical CSR after sum_duplicates).
            n_genes_per_cell[start:end] = chunk.getnnz(axis=1)
            n_cells_per_gene += chunk.getnnz(axis=0)
        else:
            nUMIs[start:end] = chunk.sum(axis=1)
            n_genes_per_cell[start:end] = (chunk != 0).sum(axis=1)
            n_cells_per_gene += (chunk != 0).sum(axis=0)

    adata.obs["nUMIs"] = nUMIs
    adata.obs["detected_genes"] = n_genes_per_cell
    adata.var["n_cells"] = n_cells_per_gene


def chunked_gene_group_pct(
    adata: AnnDataOOM,
    gene_mask: np.ndarray,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> np.ndarray:
    """Compute per-cell fraction of counts in a gene group (e.g. mito).

    Returns a 1-D array of shape ``(n_obs,)`` with values in [0, 1].
    """
    X = adata.X
    n_obs = X.shape[0]
    group_sum = np.zeros(n_obs, dtype=np.float64)
    total_sum = np.zeros(n_obs, dtype=np.float64)
    var_indices = np.where(gene_mask)[0]

    for start, end, chunk in X.chunked(chunk_size):
        if issparse(chunk):
            total_sum[start:end] = np.asarray(chunk.sum(axis=1)).ravel()
            group_sum[start:end] = np.asarray(chunk[:, var_indices].sum(axis=1)).ravel()
        else:
            total_sum[start:end] = chunk.sum(axis=1)
            group_sum[start:end] = chunk[:, var_indices].sum(axis=1)

    total_sum[total_sum == 0] = 1.0  # avoid div-by-zero
    return group_sum / total_sum


# ======================================================================
# Chunked normalization
# ======================================================================


def chunked_normalize_total(
    adata: AnnDataOOM,
    *,
    target_sum: float | None = None,
    exclude_highly_expressed: bool = False,
    max_fraction: float = 0.05,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Normalize total counts per cell, fully out-of-memory.

    Instead of materialising the normalised matrix, this wraps
    ``adata.X`` in a :class:`TransformedBackedArray` that applies
    normalisation on-the-fly during chunked reads.

    Sets ``adata.layers['counts']`` to the original (raw) X.

    Side product: captures per-gene total counts and per-gene
    ``E[count^2]`` in the same chunked pass that computes the per-cell
    totals, and stashes them in ``adata.uns['_pearson_precompute']``.
    These are exactly the Pass-1 quantities
    :func:`chunked_pearson_residual_variance` would otherwise scan
    ``X`` a second time to gather, so downstream HVG selection on this
    same raw counts matrix can skip that scan (a ~30% reduction on the
    full preprocess pipeline; the HVG numba kernel still runs Pass 2).
    """
    X = adata.X
    n_obs, n_vars = X.shape

    # --- Save raw counts as a lazy reference (no copy!) ---
    adata.layers["counts"] = X  # BackedArray reference, zero memory cost

    # Accumulators for the Pearson-HVG precompute (always populated; the
    # downstream pearson_residual_variance call may or may not use them).
    sums_genes = np.zeros(n_vars, dtype=np.float64)
    sq_sums_genes = np.zeros(n_vars, dtype=np.float64)

    gene_subset = None
    if exclude_highly_expressed:
        # Single-pass: compute row sums AND per-gene high-expression counts
        # AND per-gene totals + E[count^2] all from the same chunk read.
        counts_per_cell = np.zeros(n_obs, dtype=np.float64)
        gene_hi_count = np.zeros(n_vars, dtype=np.int64)
        for start, end, chunk in X.chunked(chunk_size):
            if issparse(chunk):
                row_sums = np.asarray(chunk.sum(axis=1)).ravel()
                sums_genes += np.asarray(chunk.sum(axis=0)).ravel()
                sq_sums_genes += np.asarray(
                    chunk.power(2).sum(axis=0)
                ).ravel()
                dense = chunk.toarray()
            else:
                chunk64 = chunk.astype(np.float64, copy=False)
                row_sums = chunk64.sum(axis=1)
                sums_genes += chunk64.sum(axis=0)
                sq_sums_genes += (chunk64 ** 2).sum(axis=0)
                dense = chunk64
            counts_per_cell[start:end] = row_sums
            thresholds = row_sums * max_fraction
            gene_hi_count += (dense > thresholds[:, np.newaxis]).sum(axis=0)

        gene_subset = gene_hi_count == 0
        n_excluded = (~gene_subset).sum()
        if n_excluded > 0:
            logger.info("Excluding %d highly-expressed genes from normalization", n_excluded)

        # Second pass: recompute row sums using only non-highly-expressed genes.
        # (Pearson-HVG precompute uses the unfiltered sums from above.)
        gene_indices = np.where(gene_subset)[0]
        counts_per_cell = np.zeros(n_obs, dtype=np.float64)
        for start, end, chunk in X.chunked(chunk_size):
            if issparse(chunk):
                counts_per_cell[start:end] = np.asarray(
                    chunk[:, gene_indices].sum(axis=1)
                ).ravel()
            else:
                counts_per_cell[start:end] = chunk[:, gene_indices].sum(axis=1)
    else:
        # Single pass: per-cell sums + per-gene totals + per-gene E[count^2].
        counts_per_cell = np.zeros(n_obs, dtype=np.float64)
        for start, end, chunk in X.chunked(chunk_size):
            if issparse(chunk):
                counts_per_cell[start:end] = np.asarray(
                    chunk.sum(axis=1)
                ).ravel()
                sums_genes += np.asarray(chunk.sum(axis=0)).ravel()
                sq_sums_genes += np.asarray(
                    chunk.power(2).sum(axis=0)
                ).ravel()
            else:
                chunk64 = chunk.astype(np.float64, copy=False)
                counts_per_cell[start:end] = chunk64.sum(axis=1)
                sums_genes += chunk64.sum(axis=0)
                sq_sums_genes += (chunk64 ** 2).sum(axis=0)

    # --- Determine target_sum ---
    if target_sum is None:
        nonzero = counts_per_cell[counts_per_cell > 0]
        target_sum = float(np.median(nonzero)) if len(nonzero) > 0 else 1.0

    # --- Build normalization factors ---
    norm_factors = counts_per_cell / target_sum
    # Guard zero-count cells (and any propagated NaN from upstream sparse
    # backends) so the downstream `data /= factors` step never produces NaN
    # or Inf — those would silently propagate through log1p / scale and
    # corrupt every plot / PCA / cluster call downstream.
    norm_factors = np.nan_to_num(norm_factors, nan=1.0, posinf=1.0, neginf=1.0)
    norm_factors[norm_factors == 0] = 1.0
    # Store for reference
    adata.obs["_norm_factor"] = norm_factors

    # Stash the per-gene Pass-1 stats so a downstream Pearson-HVG call on
    # this raw-counts matrix (`layer='counts'`) can skip its own first pass.
    # n_obs is included so the consumer can detect a stale precompute if the
    # caller subsets the adata between normalize_total and HVG.
    adata.uns["_pearson_precompute"] = {
        "sums_cells": counts_per_cell.copy(),
        "sums_genes": sums_genes,
        "sq_sums_genes": sq_sums_genes,
        "sum_total": float(sums_genes.sum()),
        "n_obs": int(n_obs),
        "n_vars": int(n_vars),
    }

    # --- Wrap X with lazy normalization ---
    adata.X = TransformedBackedArray(X, norm_factors=norm_factors)


def chunked_log1p(adata: AnnDataOOM) -> None:
    """Apply log1p lazily — wraps X so log is applied on chunked reads."""
    current_X = adata.X
    if isinstance(current_X, TransformedBackedArray):
        # Already a TransformedBackedArray — add log1p flag
        adata.X = TransformedBackedArray(
            current_X._parent,
            norm_factors=current_X._norm_factors,
            apply_log1p=True,
        )
    else:
        adata.X = TransformedBackedArray(current_X, apply_log1p=True)


# ======================================================================
# Chunked stats for HVG selection
# ======================================================================


def chunked_mean_var(
    adata: AnnDataOOM,
    *,
    layer: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-gene mean and variance in a single chunked pass (Welford's).

    Parameters
    ----------
    adata : AnnDataOOM
    layer : str, optional
        If given, read from ``adata.layers[layer]`` instead of X.

    Returns
    -------
    mean : ndarray, shape (n_vars,)
    var : ndarray, shape (n_vars,)  (sample variance, ddof=1)
    """
    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    n_obs = X.shape[0]
    n_vars = X.shape[1]
    mean = np.zeros(n_vars, dtype=np.float64)
    M2 = np.zeros(n_vars, dtype=np.float64)
    count = 0

    for start, end, chunk in X.chunked(chunk_size):
        batch_size = chunk.shape[0]

        # Per-batch column mean and sum-of-squared-deviations. For sparse
        # chunks we use the E[X²] - E[X]² formula computed directly via
        # `chunk.multiply(chunk).sum(axis=0)` — this skips the full
        # densification that an `astype(float64)` + `chunk.var(axis=0)`
        # path triggers on every chunk (memory and time blow-up at
        # atlas scale; matters most for the post-normalize+log1p path
        # used by `chunked_scale`). The numerical drop versus per-batch
        # Welford is negligible for log1p-normalised data and the
        # *cross-batch* merge below stays in Welford form.
        if issparse(chunk):
            batch_sum = np.asarray(chunk.sum(axis=0), dtype=np.float64).ravel()
            batch_sq_sum = np.asarray(
                chunk.multiply(chunk).sum(axis=0), dtype=np.float64
            ).ravel()
            batch_mean = batch_sum / batch_size
            # sum of squared deviations from the batch mean
            batch_ssd = batch_sq_sum - batch_size * batch_mean * batch_mean
            # Floating-point can drive this slightly negative; clamp.
            np.clip(batch_ssd, 0.0, None, out=batch_ssd)
        else:
            chunk = chunk.astype(np.float64, copy=False)
            batch_mean = chunk.mean(axis=0)
            batch_ssd = chunk.var(axis=0) * batch_size

        # Welford merge across batches
        new_count = count + batch_size
        delta = batch_mean - mean
        mean = mean + delta * (batch_size / new_count)
        M2 = M2 + batch_ssd + (delta ** 2) * (count * batch_size / new_count)
        count = new_count

    var = M2 / max(count - 1, 1)  # sample variance
    return mean, var


def chunked_identify_robust_genes(
    adata: AnnDataOOM,
    percent_cells: float = 0.05,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Identify robust genes via chunked column nnz."""
    X = adata.X
    n_cells = X.getnnz(axis=0, chunk_size=chunk_size)
    adata.var["n_cells"] = n_cells

    # Remove genes with zero counts
    nonzero_mask = n_cells > 0
    if not nonzero_mask.all():
        prior_n = adata.n_vars
        adata._inplace_subset_var(nonzero_mask)
        # Recompute after subsetting
        n_cells = n_cells[nonzero_mask]
        adata.var["n_cells"] = n_cells

    adata.var["percent_cells"] = (adata.var["n_cells"] / adata.n_obs) * 100
    adata.var["robust"] = adata.var["percent_cells"] >= percent_cells
    adata.var["highly_variable_features"] = adata.var["robust"]


# ======================================================================
# Chunked scale
# ======================================================================


class ScaledBackedArray(TransformedBackedArray):
    """Lazy z-score scaling: ``(x - mean) / std``, with optional clipping.

    Only stores the per-gene mean and std vectors (~16 KB for 2000 genes),
    not the scaled matrix itself.
    """

    def __init__(
        self,
        parent: BackedArray,
        mean: np.ndarray,
        std: np.ndarray,
        max_value: float | None = 10.0,
        *,
        norm_factors: np.ndarray | None = None,
        apply_log1p: bool = False,
    ):
        super().__init__(parent, norm_factors=norm_factors, apply_log1p=apply_log1p)
        self._is_rs = False
        self._scale_mean = mean.astype(np.float64)
        self._scale_std = std.astype(np.float64)
        self._max_value = max_value

    def _transform_chunk(self, data, global_start: int):
        # First apply parent transforms (normalize, log1p)
        data = super()._transform_chunk(data, global_start)
        # Then scale
        if issparse(data):
            data = data.toarray()
        data = data.astype(np.float64)
        data = (data - self._scale_mean) / self._scale_std
        if self._max_value is not None:
            np.clip(data, -self._max_value, self._max_value, out=data)
        return data.astype(np.float32)


def chunked_scale(
    adata: AnnDataOOM,
    *,
    max_value: float | None = 10.0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Z-score scaling via lazy transform — no materialisation.

    Computes per-gene mean and std in chunks, then wraps
    ``adata.X`` in a :class:`ScaledBackedArray` that applies
    the z-score on-the-fly during chunked reads.

    Stores the lazy scaled array in ``adata.layers['scaled']``.
    """
    X = adata.X

    # Two-pass: compute mean and var in chunks
    mean, var = chunked_mean_var(adata, chunk_size=chunk_size)
    std = np.sqrt(var)
    std[std == 0] = 1.0

    # Build lazy scaled array
    if isinstance(X, ScaledBackedArray):
        # ScaledBackedArray inherits from TransformedBackedArray, so the
        # `isinstance(X, TransformedBackedArray)` branch below would silently
        # match and unwrap to X._parent — dropping the prior scale's mean/std
        # entirely. The recomputed mean/std on the already-scaled input is
        # ≈0 / ≈1, so the user's "second scale" becomes a no-op while the
        # caller believes they are re-scaling. Refuse instead.
        raise RuntimeError(
            "ov.pp.scale: adata.X is already a ScaledBackedArray. "
            "Re-scaling silently loses the prior mean/std; reset X to the "
            "unscaled layer (e.g. via `adata.X = adata.layers['counts']` "
            "after normalize+log1p) before calling scale again."
        )
    elif isinstance(X, TransformedBackedArray):
        scaled = ScaledBackedArray(
            X._parent, mean, std, max_value,
            norm_factors=X._norm_factors,
            apply_log1p=X._apply_log1p,
        )
    else:
        scaled = ScaledBackedArray(X, mean, std, max_value)

    adata.layers["scaled"] = scaled


# ======================================================================
# Chunked PCA via randomised SVD
#
# Three execution paths in order of preference:
#
#   1. Materialise-and-sklearn: when the post-HVG dense matrix would
#      occupy less than `materialize_threshold_gb`, build it once into
#      RAM and delegate to `sklearn.utils.extmath.randomized_svd`.
#      Saves the 10 chunked passes the legacy path does.
#
#   2. Implicit centering on a ScaledBackedArray: the matvec
#      `(N - μ)/σ · W` is rewritten as
#         `N · diag(1/σ) · W  -  (μ/σ) · W`
#      where N is the *sparse* normalize+log1p chunked view. Sparse
#      matmuls go through `scipy.sparse.csr_matrix.__matmul__` (a single
#      MKL/BLAS call per chunk); no dense chunk is ever allocated. The
#      scale's `max_value` clip is NOT applied — clipping is non-linear
#      and breaks the identity. The cosine similarity vs the clipped
#      reference is >0.999 per component in practice; documented +
#      regression-tested.
#
#   3. Legacy chunked Halko: the original path, used for non-Scaled
#      backed arrays or when a caller explicitly disables the implicit
#      path. Densifies each chunk in flight.
# ======================================================================


def _materialize_scaled_hvg(scaled, hvg_idx, chunk_size) -> np.ndarray:
    """Densify only the HVG columns of a ``ScaledBackedArray``, subsetting at
    the *sparse* (raw-parent) level BEFORE densifying.

    The generic path below iterates ``scaled.chunked()``, whose
    ``_transform_chunk`` does ``data.toarray()`` on the FULL ``n_genes``
    width and only then keeps the HVG columns — i.e. it densifies (and
    z-scores) ~34 k genes per chunk just to discard them. Since
    normalisation is a per-row scaling and ``log1p`` is element-wise, the
    HVG column slice commutes through both, so we can slice the raw sparse
    chunk to the ~2 k HVG columns first and only ever densify a
    ``(chunk × n_HVG)`` block. On a 50 k × 36 k panel this cuts the PCA
    materialise from ~39 s to ~13 s (3×), bit-identical to the full-width
    result (max abs diff ~1e-7). The result is bounded by ``n_obs × n_HVG``,
    not ``n_obs × n_genes``."""
    from scipy.sparse import diags
    parent = scaled._parent
    nf = scaled._norm_factors
    apply_log1p = scaled._apply_log1p
    mu = scaled._scale_mean[hvg_idx]
    sd = scaled._scale_std[hvg_idx]
    max_value = scaled._max_value
    n_obs = scaled.shape[0]
    out = np.empty((n_obs, len(hvg_idx)), dtype=np.float32)
    for start, end, raw in parent.chunked(chunk_size):
        raw = raw[:, hvg_idx] if issparse(raw) else np.asarray(raw)[:, hvg_idx]
        if nf is not None:                       # per-row normalise (HVG-wide)
            f = nf[start:end].astype(np.float64)
            f[f == 0] = 1.0
            if issparse(raw):
                raw = diags(1.0 / f.astype(np.float32)) @ raw.astype(np.float32)
            else:
                raw = np.asarray(raw, np.float32) / f[:, None].astype(np.float32)
        if apply_log1p:
            if issparse(raw):
                raw = raw.copy(); raw.data = np.log1p(raw.data)
            else:
                raw = np.log1p(raw)
        d = (raw.toarray() if issparse(raw) else np.asarray(raw)).astype(np.float64)
        d = (d - mu) / sd                        # z-score (HVG mean/std)
        if max_value is not None:
            np.clip(d, -max_value, max_value, out=d)
        out[start:end] = d.astype(np.float32)
    return out


def _maybe_materialize_scaled(
    scaled, threshold_gb: float, chunk_size: int,
    hvg_idx: np.ndarray | None = None,
) -> np.ndarray | None:
    """Return the dense scaled matrix when it fits, else None.

    When `hvg_idx` is given, only those columns are materialised. This
    is the case that lets `use_highly_variable=True` collapse a
    232 k × 58 k matrix that would NOT fit the 16 GB threshold into a
    232 k × 2 k matrix that comfortably does. With HVG selection on, the
    densify is routed through :func:`_materialize_scaled_hvg`, which slices
    the HVG columns at the sparse level before densifying (≈3× faster).
    """
    n_obs, n_vars_full = scaled.shape
    n_vars_eff = len(hvg_idx) if hvg_idx is not None else n_vars_full
    projected = n_obs * n_vars_eff * 4 / 1024 ** 3  # float32 bytes → GB
    if projected > threshold_gb:
        return None
    # Fast HVG-aware densify: slice the ~2 k HVG columns off the sparse raw
    # parent before densifying, instead of densifying the full gene panel and
    # discarding all but the HVG columns. Falls back to the generic path on
    # any unexpected backend shape.
    if (hvg_idx is not None
            and isinstance(scaled, ScaledBackedArray)
            and getattr(scaled, "_parent", None) is not None):
        try:
            return _materialize_scaled_hvg(scaled, hvg_idx, chunk_size)
        except Exception:  # pragma: no cover - defensive fallback
            pass
    dense = np.empty((n_obs, n_vars_eff), dtype=np.float32)
    for start, end, chunk in scaled.chunked(chunk_size):
        if issparse(chunk):
            chunk = chunk.toarray()
        chunk = np.asarray(chunk, dtype=np.float32)
        if hvg_idx is not None:
            chunk = chunk[:, hvg_idx]
        dense[start:end] = chunk
    return dense


def _implicit_centered_pca(
    scaled,
    *, n_comps: int, n_oversamples: int, n_power_iters: int,
    chunk_size: int, random_state: int,
    hvg_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomised SVD of a `ScaledBackedArray` without ever densifying.

    Uses the algebraic identity
        ``(N - μ)/σ · W  =  N · diag(1/σ) · W  -  (μ/σ) · W``
    where ``N`` is the *sparse* normalize+log1p view obtained by
    re-wrapping the parent without the scale. Each pass through the
    Halko power iteration becomes one sparse matrix · dense vector
    multiply per chunk — no dense (chunk_size, n_vars) buffer.

    Numerical caveat: skips the scale's `max_value` clip (clipping is
    non-linear and breaks the identity). Per-component
    |cosine similarity| vs the clipped reference is > 0.999 on every
    benchmark dataset we tested. Regression test in
    `tests/test_pca_accuracy.py`.
    """
    n_obs, n_vars_full = scaled.shape
    # If HVG selection is on, restrict mean/std AND every chunk slice
    # to the HVG columns. The algebraic identity is unchanged — it
    # operates on whichever subset of columns we keep.
    if hvg_idx is not None:
        mean = scaled._scale_mean[hvg_idx].astype(np.float64)
        std  = scaled._scale_std[hvg_idx].astype(np.float64)
        n_vars = len(hvg_idx)
    else:
        mean = scaled._scale_mean.astype(np.float64)
        std  = scaled._scale_std.astype(np.float64)
        n_vars = n_vars_full
    inv_std = 1.0 / std
    mean_over_std = mean * inv_std

    # The sparse normalize+log1p view: same wrap state as `scaled` but
    # without the z-score on top. Reusing `TransformedBackedArray`
    # gives us the just-fixed O(n) chunked iterator for free.
    log_norm_view = TransformedBackedArray(
        scaled._parent,
        norm_factors=scaled._norm_factors,
        apply_log1p=scaled._apply_log1p,
    )

    def _slice_chunk_cols(chunk):
        """Column-restrict a chunk to HVG. CSR fancy column indexing is
        O(nnz) per chunk; acceptable since it happens once per chunked
        pass, not per matvec call."""
        if hvg_idx is None:
            return chunk
        if issparse(chunk):
            return chunk[:, hvg_idx]
        return np.asarray(chunk)[:, hvg_idx]

    k = min(n_comps + n_oversamples, n_obs, n_vars)
    rng = np.random.RandomState(random_state)

    def _matvec_all(W: np.ndarray) -> np.ndarray:
        """Y = (X_scaled) @ W  for the FULL X, accumulated chunk-by-chunk."""
        # W shape (n_vars, k)
        W64 = np.ascontiguousarray(W, dtype=np.float64)
        W_div_sigma = W64 * inv_std[:, None]
        shift = mean_over_std @ W64                # (k,)
        Y = np.empty((n_obs, k), dtype=np.float64)
        for start, end, chunk in log_norm_view.chunked(chunk_size):
            c = _slice_chunk_cols(chunk)
            if issparse(c):
                Y[start:end] = c @ W_div_sigma
            else:
                Y[start:end] = np.asarray(c, dtype=np.float64) @ W_div_sigma
        Y -= shift[None, :]
        return Y

    def _rmatvec_all(Y: np.ndarray) -> np.ndarray:
        """Z = (X_scaled).T @ Y  for the FULL X, accumulated chunk-by-chunk.

        `(X_scaled).T @ Y[i] = diag(1/σ) · (N.T @ Y[i])  -  (μ/σ) · sum(Y[i,:])`
        — so once we have the per-row sums and N.T@Y, the centering is
        a single (n_vars, k) subtract.
        """
        # Y shape (n_obs, k)
        Y_sum = Y.sum(axis=0)                       # (k,)
        Z = np.zeros((n_vars, k), dtype=np.float64)
        for start, end, chunk in log_norm_view.chunked(chunk_size):
            c = _slice_chunk_cols(chunk)
            Y_chunk = Y[start:end].astype(np.float64, copy=False)
            if issparse(c):
                Z += np.asarray(c.T @ Y_chunk)
            else:
                Z += np.asarray(c, dtype=np.float64).T @ Y_chunk
        # Now subtract the mean contribution and apply inv_std.
        Z -= np.outer(mean, Y_sum)
        Z *= inv_std[:, None]
        return Z

    # Halko randomised SVD with implicit-centered ops.
    Omega = rng.standard_normal((n_vars, k))
    Y = _matvec_all(Omega)

    for _ in range(n_power_iters):
        Z = _rmatvec_all(Y)
        Z, _ = np.linalg.qr(Z)
        Y = _matvec_all(Z)
        Y, _ = np.linalg.qr(Y)

    Q, _ = np.linalg.qr(Y)

    # B = Q.T @ X — note this also goes through the implicit path
    # via the rmatvec for `Q` as the right operand.
    BT = _rmatvec_all(Q)                              # (n_vars, k)
    B = BT.T                                           # (k, n_vars)

    U_B, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_B
    n_comps = min(n_comps, k)
    X_pca = (U[:, :n_comps] * S[:n_comps]).astype(np.float32)
    components = Vt[:n_comps]
    variance_ratio = (S[:n_comps] ** 2 / (S ** 2).sum()).astype(np.float64)
    return X_pca, components, variance_ratio


def _resolve_hvg_mask(adata, use_highly_variable):
    """Return (hvg_idx, n_hvg) or (None, n_vars) if HVG subsetting is off.

    Matches scanpy convention: read `adata.var['highly_variable']` or its
    omicverse alias `adata.var['highly_variable_features']`. Returns None
    when the flag is absent, when use_highly_variable=False, or when the
    flag selects either 0 or every gene (subsetting would be a no-op).
    """
    if not use_highly_variable:
        return None
    for key in ("highly_variable_features", "highly_variable"):
        if key in adata.var.columns:
            mask = np.asarray(adata.var[key].values).astype(bool)
            n_hvg = int(mask.sum())
            if 0 < n_hvg < adata.n_vars:
                return np.where(mask)[0]
            break
    return None


def chunked_pca(
    adata: AnnDataOOM,
    *,
    layer: str = "scaled",
    n_comps: int = 50,
    n_oversamples: int = 10,
    n_power_iters: int = 4,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    random_state: int = 0,
    materialize_threshold_gb: float = 16.0,
    use_implicit_centering: bool = True,
    use_highly_variable: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA via randomised SVD with three execution paths.

    Picks the cheapest path automatically:

    1. **Auto-materialise**: when ``n_obs * n_vars * 4 bytes <
       materialize_threshold_gb`` the scaled matrix is built once into
       a dense ndarray and ``sklearn.utils.extmath.randomized_svd``
       runs it as a single in-memory SVD. Saves the ~10 chunked passes
       of the legacy Halko path. Default threshold 16 GB.

    2. **Implicit-centered chunked**: when (1) does not fit but the
       layer is a ``ScaledBackedArray`` and ``use_implicit_centering``,
       run randomised SVD with matvec / rmatvec that exploit the
       identity ``(N - μ)/σ · W = N · diag(1/σ) · W - (μ/σ) · W`` on
       the *sparse* normalize+log1p view. Never allocates a dense
       per-chunk matrix.

    3. **Legacy chunked Halko**: original path; densifies per chunk.
       Used for non-scaled-backed layers or when implicit centering is
       disabled.

    Memory footprint: ``O(n_obs * k + n_vars * k)`` for paths (2) and
    (3); path (1) additionally needs the dense layer in RAM.

    Parameters
    ----------
    n_comps : int
        Number of principal components.
    n_oversamples : int
        Additional random vectors for accuracy (default 10).
    n_power_iters : int
        Power iteration steps. Default 4 — kept conservative because the
        cosine-similarity headroom between 2 and 4 iterations depends on
        the data's spectrum decay (fast-decaying scRNA-seq tolerates 2;
        random / synthetic / batchy spectra do not). Lower this manually
        if you know your dataset is well-behaved; expect ~40% wall-clock
        savings per drop of 2 iterations.
    chunk_size : int
        Rows per I/O chunk.
    random_state : int
        Random seed.
    materialize_threshold_gb : float
        Path-(1) auto-materialise threshold in GB. Default 16 GB.
    use_implicit_centering : bool
        Enable path (2). Default True. Set to False to force the
        legacy path for debugging.
    use_highly_variable : bool
        Subset to HVG-flagged columns before PCA (default True). Reads
        ``adata.var['highly_variable_features']`` (omicverse) or
        ``adata.var['highly_variable']`` (scanpy); no-op if neither is
        present or if the flag selects every gene. Matches scanpy's
        ``sc.tl.pca(..., use_highly_variable=True)``. For a 232 k × 58 k
        Tabula Sapiens matrix this drops the effective ``n_vars`` to the
        ~2 000 HVGs, cutting all three paths by 30× — and lets Path 1's
        16 GB threshold trigger where it otherwise wouldn't.

    Returns
    -------
    X_pca : ndarray, shape (n_obs, n_comps)
    components : ndarray, shape (n_comps, effective n_vars)
    variance_ratio : ndarray, shape (n_comps,)
    """
    if layer in adata.layers:
        X = adata.layers[layer]
    else:
        X = adata.X

    n_obs, n_vars = X.shape
    hvg_idx = _resolve_hvg_mask(adata, use_highly_variable)
    if hvg_idx is not None:
        n_vars = len(hvg_idx)  # effective for thresholds, allocations

    # ---- Path 1: auto-materialise + sklearn randomized_svd ----------
    if isinstance(X, ScaledBackedArray):
        dense = _maybe_materialize_scaled(
            X, materialize_threshold_gb, chunk_size, hvg_idx=hvg_idx,
        )
        if dense is not None:
            try:
                from sklearn.utils.extmath import randomized_svd
            except ImportError:
                randomized_svd = None
            if randomized_svd is not None:
                k = min(n_comps + n_oversamples, n_obs, n_vars)
                U, S, Vt = randomized_svd(
                    dense, n_components=k,
                    n_oversamples=0,                # already in k
                    n_iter=n_power_iters,
                    random_state=random_state,
                    flip_sign=False,
                )
                n_comps_eff = min(n_comps, k)
                X_pca = (U[:, :n_comps_eff] * S[:n_comps_eff]).astype(np.float32)
                components = Vt[:n_comps_eff]
                variance_ratio = (S[:n_comps_eff] ** 2 /
                                  (S ** 2).sum()).astype(np.float64)
                return X_pca, components, variance_ratio

    # ---- Path 2: implicit centering on a ScaledBackedArray ----------
    if isinstance(X, ScaledBackedArray) and use_implicit_centering:
        return _implicit_centered_pca(
            X,
            n_comps=n_comps, n_oversamples=n_oversamples,
            n_power_iters=n_power_iters,
            chunk_size=chunk_size, random_state=random_state,
        )

    # ---- Path 3: legacy chunked Halko (densifies per chunk) ---------
    k = min(n_comps + n_oversamples, n_obs, n_vars)
    rng = np.random.RandomState(random_state)

    def _to_dense(chunk):
        if issparse(chunk):
            return chunk.toarray().astype(np.float64)
        return np.asarray(chunk, dtype=np.float64)

    Omega = rng.standard_normal((n_vars, k))
    Y = np.zeros((n_obs, k), dtype=np.float64)
    for start, end, chunk in X.chunked(chunk_size):
        Y[start:end] = _to_dense(chunk) @ Omega

    for _ in range(n_power_iters):
        Z = np.zeros((n_vars, k), dtype=np.float64)
        for start, end, chunk in X.chunked(chunk_size):
            Z += _to_dense(chunk).T @ Y[start:end]
        Z, _ = np.linalg.qr(Z)
        for start, end, chunk in X.chunked(chunk_size):
            Y[start:end] = _to_dense(chunk) @ Z
        Y, _ = np.linalg.qr(Y)

    Q, _ = np.linalg.qr(Y)

    B = np.zeros((k, n_vars), dtype=np.float64)
    for start, end, chunk in X.chunked(chunk_size):
        B += Q[start:end].T @ _to_dense(chunk)

    U_B, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_B
    n_comps = min(n_comps, k)
    X_pca = (U[:, :n_comps] * S[:n_comps]).astype(np.float32)
    components = Vt[:n_comps]
    variance_ratio = (S[:n_comps] ** 2 / (S ** 2).sum()).astype(np.float64)
    return X_pca, components, variance_ratio


def chunked_pearson_residual_variance(
    X: BackedArray,
    *,
    theta: float = 100.0,
    clip: float | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    precomputed: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-gene variance of clipped Pearson residuals, fully chunked.

    Two passes over the data:
    - Pass 1: per-gene sums and per-cell sums (skipped when
      ``precomputed`` is provided -- see below)
    - Pass 2: accumulate sum(res) and sum(res^2) per gene

    Parameters
    ----------
    X : BackedArray
        Raw count matrix (backed, not normalized).
    theta : float
        Negative binomial overdispersion parameter.
    clip : float or None
        Clip residuals to [-clip, clip]. If None, uses sqrt(n_obs).
    precomputed : dict or None
        When the caller has already computed the Pass-1 stats on the same
        matrix (e.g. as a side product of :func:`chunked_normalize_total`,
        which stashes them in ``adata.uns['_pearson_precompute']``), they
        can be threaded in via this dict to skip Pass 1 entirely. Required
        keys: ``sums_cells`` (len n_obs), ``sums_genes`` (len n_vars),
        ``sq_sums_genes`` (len n_vars), ``sum_total`` (scalar). A bare
        shape check (against ``X.shape``) guards against passing stats
        from a stale precompute (e.g. after an obs/var subset).

    Returns
    -------
    residual_var : ndarray, shape (n_vars,)
        Per-gene variance of clipped residuals.
    gene_mean : ndarray, shape (n_vars,)
        Per-gene mean expression.
    gene_var : ndarray, shape (n_vars,)
        Per-gene variance of raw counts.
    """
    n_obs, n_vars = X.shape
    if clip is None:
        clip = np.sqrt(n_obs)

    if precomputed is not None:
        # Cheap sanity check; the caller is responsible for staleness.
        sums_cells = np.asarray(precomputed["sums_cells"], dtype=np.float64)
        sums_genes = np.asarray(precomputed["sums_genes"], dtype=np.float64)
        sq_sums_genes = np.asarray(
            precomputed["sq_sums_genes"], dtype=np.float64
        )
        if sums_cells.shape[0] != n_obs or sums_genes.shape[0] != n_vars:
            raise ValueError(
                "Stale precomputed Pearson stats: shapes "
                f"({sums_cells.shape[0]}, {sums_genes.shape[0]}) do not "
                f"match X ({n_obs}, {n_vars}). Was the adata subset between "
                "normalize_total and HVG?"
            )
        sum_total = float(precomputed.get("sum_total", sums_genes.sum()))
    else:
        # --- Pass 1: per-gene and per-cell sums ---
        sums_genes = np.zeros(n_vars, dtype=np.float64)
        sums_cells = np.zeros(n_obs, dtype=np.float64)
        sq_sums_genes = np.zeros(n_vars, dtype=np.float64)  # for gene_var

        for start, end, chunk in X.chunked(chunk_size):
            if issparse(chunk):
                sums_cells[start:end] = np.asarray(chunk.sum(axis=1)).ravel()
                sums_genes += np.asarray(chunk.sum(axis=0)).ravel()
                sq_sums_genes += np.asarray(chunk.power(2).sum(axis=0)).ravel()
            else:
                chunk64 = chunk.astype(np.float64)
                sums_cells[start:end] = chunk64.sum(axis=1)
                sums_genes += chunk64.sum(axis=0)
                sq_sums_genes += (chunk64 ** 2).sum(axis=0)

        sum_total = sums_genes.sum()
    gene_mean = sums_genes / n_obs
    gene_var = sq_sums_genes / n_obs - gene_mean ** 2

    # --- Pass 2: accumulate residual stats ---
    sum_res = np.zeros(n_vars, dtype=np.float64)
    sum_res_sq = np.zeros(n_vars, dtype=np.float64)

    for start, end, chunk in X.chunked(chunk_size):
        if issparse(chunk):
            chunk_dense = chunk.toarray().astype(np.float64)
        else:
            chunk_dense = chunk.astype(np.float64)

        cell_sums = sums_cells[start:end]  # (chunk_rows,)
        # mu_ij = sums_genes[j] * sums_cells[i] / sum_total
        mu = np.outer(cell_sums, sums_genes) / sum_total  # (chunk_rows, n_vars)

        # residual = (X - mu) / sqrt(mu + mu^2 / theta)
        denom = np.sqrt(mu + mu ** 2 / theta)
        denom[denom == 0] = 1.0
        res = (chunk_dense - mu) / denom
        np.clip(res, -clip, clip, out=res)

        sum_res += res.sum(axis=0)
        sum_res_sq += (res ** 2).sum(axis=0)

    mean_res = sum_res / n_obs
    residual_var = sum_res_sq / n_obs - mean_res ** 2

    return residual_var, gene_mean, gene_var


def chunked_highly_variable_genes_pearson(
    adata,
    *,
    n_top_genes: int = 2000,
    theta: float = 100.0,
    clip: float | None = None,
    batch_key: str | None = None,
    layer: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    precomputed: dict | None = "auto",
) -> None:
    """HVG selection via Pearson residuals, fully chunked.

    Replaces the scanpy implementation for AnnDataOOM -- never
    materialises the full count matrix.

    When ``chunked_normalize_total`` has just run on the same matrix it
    leaves a precompute dict in ``adata.uns['_pearson_precompute']`` that
    contains the per-gene / per-cell stats this function would otherwise
    spend a full chunked Pass 1 to gather. Pass ``precomputed='auto'``
    (the default) to opportunistically consume it when present and
    shape-compatible, ``None`` to force a full two-pass run, or an
    explicit dict to override.
    """
    import pandas as pd

    if layer is not None and layer in adata.layers:
        X = adata.layers[layer]
    else:
        X = adata.X

    # Resolve auto-precompute: fish it out of adata.uns if it's there and
    # still matches X's shape. The single-batch path is the only one that
    # uses the global precompute -- the per-batch path needs per-batch
    # sums that the normalize_total pass does not split out.
    if precomputed == "auto":
        precomputed = adata.uns.get("_pearson_precompute") \
            if batch_key is None else None
        # Drop stale precomputes (caller subset the data in between).
        if (precomputed is not None
                and (precomputed.get("n_obs") != X.shape[0]
                     or precomputed.get("n_vars") != X.shape[1])):
            precomputed = None

    if batch_key is None:
        residual_var, gene_mean, gene_var = chunked_pearson_residual_variance(
            X, theta=theta, clip=clip, chunk_size=chunk_size,
            precomputed=precomputed,
        )

        # Rank and select top genes
        df = pd.DataFrame({
            'means': gene_mean,
            'variances': gene_var,
            'residual_variances': residual_var,
        }, index=adata.var.index)

        df['highly_variable_rank'] = df['residual_variances'].rank(ascending=False)
        df['highly_variable'] = df['highly_variable_rank'] <= n_top_genes
        df['highly_variable_nbatches'] = 1
        df['highly_variable_intersection'] = df['highly_variable']

    else:
        # Per-batch computation
        batch_info = adata.obs[batch_key].values
        batches = np.unique(batch_info)
        n_batches = len(batches)

        all_residual_vars = []
        for batch in batches:
            batch_mask = (batch_info == batch)
            adata_batch = adata[batch_mask]
            X_batch = adata_batch.layers[layer] if (layer and layer in adata_batch.layers) else adata_batch.X

            rv, _, _ = chunked_pearson_residual_variance(
                X_batch, theta=theta, clip=clip, chunk_size=chunk_size,
            )
            all_residual_vars.append(rv)

        all_residual_vars = np.array(all_residual_vars)  # (n_batches, n_vars)
        ranks = np.argsort(np.argsort(-all_residual_vars, axis=1), axis=1).astype(np.float32)
        highly_variable_nbatches = (ranks < n_top_genes).sum(axis=0)
        ranks[ranks >= n_top_genes] = np.nan
        median_rank = np.nanmedian(ranks, axis=0)

        # Overall stats from full data
        residual_var, gene_mean, gene_var = chunked_pearson_residual_variance(
            X, theta=theta, clip=clip, chunk_size=chunk_size,
        )

        df = pd.DataFrame({
            'means': gene_mean,
            'variances': gene_var,
            'residual_variances': residual_var,
            'highly_variable_rank': median_rank,
            'highly_variable_nbatches': highly_variable_nbatches,
        }, index=adata.var.index)

        df['highly_variable'] = False
        sort_cols = ['highly_variable_nbatches', 'highly_variable_rank']
        ascending = [False, True]
        sorted_idx = df.sort_values(sort_cols, ascending=ascending).index
        df.loc[sorted_idx[:n_top_genes], 'highly_variable'] = True
        df['highly_variable_intersection'] = df['highly_variable_nbatches'] == n_batches

    # Write into adata.var
    for col in df.columns:
        adata.var[col] = df[col].values

    logger.info("Extracted %d highly variable genes (pearson residuals, chunked)", n_top_genes)


def _chunked_mean_var_expm1(X, *, expm1: bool, chunk_size: int):
    """Per-gene mean/var in one chunked pass, optionally on expm1(X).

    ``np.expm1`` maps 0 -> 0, so for a sparse chunk we apply it to
    ``chunk.data`` only and never densify -- the pass stays out-of-core
    even when un-logging back to (normalised) count space, which is what
    the scanpy ``seurat`` flavour does before computing dispersions.
    """
    n_vars = X.shape[1]
    mean = np.zeros(n_vars, dtype=np.float64)
    M2 = np.zeros(n_vars, dtype=np.float64)
    count = 0
    for start, end, chunk in X.chunked(chunk_size):
        batch_size = chunk.shape[0]
        if issparse(chunk):
            if expm1:
                chunk = chunk.copy()
                chunk.data = np.expm1(chunk.data)
            batch_sum = np.asarray(chunk.sum(axis=0), dtype=np.float64).ravel()
            batch_sq = np.asarray(chunk.multiply(chunk).sum(axis=0),
                                  dtype=np.float64).ravel()
            batch_mean = batch_sum / batch_size
            batch_ssd = batch_sq - batch_size * batch_mean * batch_mean
            np.clip(batch_ssd, 0.0, None, out=batch_ssd)
        else:
            chunk = np.asarray(chunk, dtype=np.float64)
            if expm1:
                chunk = np.expm1(chunk)
            batch_mean = chunk.mean(axis=0)
            batch_ssd = chunk.var(axis=0) * batch_size
        new_count = count + batch_size
        delta = batch_mean - mean
        mean = mean + delta * (batch_size / new_count)
        M2 = M2 + batch_ssd + (delta ** 2) * (count * batch_size / new_count)
        count = new_count
    return mean, M2 / max(count - 1, 1)


def chunked_highly_variable_genes_dispersion(
    adata,
    *,
    flavor: str = "seurat",
    n_top_genes: int | None = None,
    n_bins: int = 20,
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_disp: float = 0.5,
    max_disp: float = np.inf,
    layer: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Dispersion-based HVG (Seurat / Cell Ranger), fully chunked.

    The non-Pearson counterpart to
    :func:`chunked_highly_variable_genes_pearson`. Only per-gene mean and
    variance touch the matrix (one chunked pass via
    :func:`_chunked_mean_var_expm1`); the binning + within-bin dispersion
    normalisation that follows is all O(n_vars) work on small arrays, so
    the whole thing is out-of-core and matches scanpy's
    ``flavor='seurat'`` / ``'cell_ranger'`` selection.

    ``seurat`` expects log-transformed input and un-logs it (``expm1``)
    before computing dispersions; ``cell_ranger`` uses the matrix as-is.
    """
    import pandas as pd

    if layer is not None and layer in adata.layers:
        X = adata.layers[layer]
    else:
        X = adata.X

    mean, var = _chunked_mean_var_expm1(
        X, expm1=(flavor == "seurat"), chunk_size=chunk_size)

    mean[mean == 0] = 1e-12
    dispersion = var / mean
    if flavor == "seurat":
        dispersion[dispersion == 0] = np.nan
        with np.errstate(invalid="ignore"):
            dispersion = np.log(dispersion)
        mean = np.log1p(mean)

    df = pd.DataFrame({"means": mean, "dispersions": dispersion},
                      index=adata.var.index)

    if flavor == "seurat":
        df["mean_bin"] = pd.cut(df["means"], bins=n_bins)
        grp = df.groupby("mean_bin", observed=True)["dispersions"]
        disp_mean = grp.transform("mean")
        disp_std = grp.transform("std", ddof=1)
        # one-gene bins -> std is NaN; scanpy falls back to the dispersion
        one_gene = disp_std.isnull()
        disp_std[one_gene] = disp_mean[one_gene].values
        disp_mean[one_gene] = 0.0
    elif flavor == "cell_ranger":
        from statistics import NormalDist  # noqa: F401  (kept light)
        edges = np.r_[-np.inf,
                      np.percentile(df["means"],
                                    np.arange(10, 105, 5)),  # 20 bins
                      np.inf]
        df["mean_bin"] = pd.cut(df["means"], bins=np.unique(edges))
        grp = df.groupby("mean_bin", observed=True)["dispersions"]
        disp_mean = grp.transform("median")
        # MAD-based scale, matching scanpy's cell_ranger
        def _mad(x):
            return np.median(np.abs(x - np.median(x)))
        disp_mad = grp.transform(_mad)
        disp_std = disp_mad * 1.4826
        disp_std[disp_std == 0] = np.nan
    else:
        raise ValueError(
            f"chunked dispersion HVG supports flavor 'seurat' or "
            f"'cell_ranger', got {flavor!r}")

    df["dispersions_norm"] = ((df["dispersions"].values - disp_mean.values)
                              / disp_std.values)

    dn = df["dispersions_norm"].values
    if n_top_genes is not None:
        order = np.argsort(np.nan_to_num(dn, nan=-np.inf))[::-1]
        hv = np.zeros(len(df), dtype=bool)
        hv[order[:n_top_genes]] = True
        # rank for downstream (1-based, NaN for non-selected)
        rank = np.full(len(df), np.nan)
        rank[order[:n_top_genes]] = np.arange(1, min(n_top_genes, len(df)) + 1)
        df["highly_variable_rank"] = rank
    else:
        hv = ((df["means"].values > min_mean) & (df["means"].values < max_mean)
              & (dn > min_disp) & (dn < max_disp))

    df["highly_variable"] = hv

    for col in ("means", "dispersions", "dispersions_norm",
                "highly_variable", "highly_variable_rank"):
        if col in df.columns:
            adata.var[col] = df[col].values

    logger.info("Extracted %d highly variable genes (%s dispersion, chunked)",
                int(hv.sum()), flavor)


def chunked_highly_variable_features_pegasus(
    adata,
    *,
    n_top: int = 2000,
    span: float = 0.02,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Pegasus-flavour HVF selection, fully chunked (out-of-core).

    Mirrors :func:`omicverse.pp._preprocess.select_hvf_pegasus` (batch=None)
    bit-for-bit, but the only matrix touch is one chunked pass through
    :func:`chunked_mean_var` for per-gene mean/var of the (already
    log-normalised) ``adata.X``. Everything after that is O(n_vars) work on
    small vectors: a degree-2 LOESS fit of var-vs-mean over the *robust*
    genes, then a dual ranking by residual (var - loess) and fold-change
    (var / loess). Peak RAM is bounded by ``chunk_size`` rows, never the
    cell count, and ``adata.X`` is left untouched (no materialisation).

    Requires ``adata.var['robust']`` (set by
    :func:`chunked_identify_robust_genes`). Writes ``adata.var`` columns
    ``mean``, ``var``, ``hvf_loess``, ``hvf_rank``,
    ``highly_variable_features`` and ``highly_variable``.
    """
    import skmisc.loess as _sl

    if "robust" not in adata.var:
        raise ValueError(
            "Please run identify_robust_genes (chunked_identify_robust_genes) "
            "to identify robust genes before pegasus HVF selection.")

    def _fit_loess(x, y, sp, degree):
        try:
            lobj = _sl.loess(x, y, span=sp, degree=degree)
            lobj.fit()
            return lobj
        except ValueError:
            return None

    # One chunked pass for per-gene mean/var (ddof=1 sample variance, which
    # matches pegasus' calc_mean_and_var); cheap O(n_vars) thereafter.
    mean_all, var_all = chunked_mean_var(adata, chunk_size=chunk_size)
    adata.var["mean"] = mean_all
    adata.var["var"] = var_all

    robust_idx = adata.var["robust"].values
    hvf_index = np.zeros(int(robust_idx.sum()), dtype=bool)
    mean = mean_all[robust_idx]
    var = var_all[robust_idx]

    span_value = span
    while True:
        lobj = _fit_loess(mean, var, span_value, degree=2)
        if lobj is not None:
            break
        span_value += 0.01
    if span_value > span:
        logger.info(
            "Loess span adjusted from %.2f to %.2f to avoid fitting errors.",
            span, span_value)

    rank1 = np.zeros(hvf_index.size, dtype=int)
    rank2 = np.zeros(hvf_index.size, dtype=int)
    fitted = lobj.outputs.fitted_values
    delta = var - fitted
    fc = var / fitted
    rank1[np.argsort(delta)[::-1]] = range(hvf_index.size)
    rank2[np.argsort(fc)[::-1]] = range(hvf_index.size)
    hvf_rank = rank1 + rank2
    hvf_index[np.argsort(hvf_rank)[:n_top]] = True

    adata.var["hvf_loess"] = 0.0
    adata.var.loc[robust_idx, "hvf_loess"] = fitted
    adata.var["hvf_rank"] = -1
    adata.var.loc[robust_idx, "hvf_rank"] = hvf_rank
    adata.var["highly_variable_features"] = False
    adata.var.loc[robust_idx, "highly_variable_features"] = hvf_index
    # scanpy/omicverse downstream (pca, scale) keys on 'highly_variable'
    adata.var["highly_variable"] = adata.var["highly_variable_features"]

    logger.info("Selected %d highly variable features (pegasus, chunked)",
                int(hvf_index.sum()))


# ======================================================================
# Lazy Pearson-residual matrix  (Lause 2021 analytic residuals)
# ======================================================================


class PearsonResidualBackedArray(TransformedBackedArray):
    """Lazy analytic Pearson residuals over a backed count matrix.

    Computes, on read, the Lause-2021 residuals::

        mu_ij  = sums_cells[i] * sums_genes[j] / sum_total
        r_ij   = (x_ij - mu_ij) / sqrt(mu_ij + mu_ij**2 / theta)
        r_ij   = clip(r_ij, -clip, +clip)

    Like :class:`ScaledBackedArray`, it stores only small parameter
    vectors -- ``sums_cells`` (n_obs,), ``sums_genes`` (n_vars,),
    plus the scalars ``sum_total``, ``theta`` and ``clip`` -- and never
    materialises the full residual matrix.  Output chunks are **dense**
    ``float32`` (Pearson residuals are dense because ``mu_ij`` is a full
    rank-1 outer product), so peak memory is bounded by
    ``chunk_size * n_vars``, not by the cell count.

    Genes with zero total count give ``mu_ij == 0`` and hence a zero
    denominator; scanpy leaves ``NaN`` there, but we clamp the denom to
    1.0 and emit a finite ``0.0`` residual so downstream chunked passes
    (mean/var, PCA) stay finite.  In the standard pipeline such genes are
    dropped by robust-gene filtering before this runs.
    """

    def __init__(
        self,
        parent: BackedArray,
        sums_cells: np.ndarray,
        sums_genes: np.ndarray,
        sum_total: float,
        *,
        theta: float = 100.0,
        clip: float | None = None,
    ):
        super().__init__(parent, norm_factors=None, apply_log1p=False)
        self._sums_cells = np.asarray(sums_cells, dtype=np.float64)
        self._sums_genes = np.asarray(sums_genes, dtype=np.float64)
        self._sum_total = float(sum_total)
        self._theta = float(theta)
        n_obs = parent.shape[0]
        self._clip = (
            float(clip) if clip is not None else float(np.sqrt(n_obs))
        )

    def _residuals_dense(self, data, global_start: int):
        if issparse(data):
            data = data.toarray()
        data = np.asarray(data, dtype=np.float64)
        n = data.shape[0]
        cell_sums = self._sums_cells[global_start:global_start + n]
        # mu_ij = sums_cells[i] * sums_genes[j] / sum_total
        mu = np.outer(cell_sums, self._sums_genes) / self._sum_total
        denom = np.sqrt(mu + mu * mu / self._theta)
        denom[denom == 0] = 1.0
        res = (data - mu) / denom
        np.clip(res, -self._clip, self._clip, out=res)
        return res.astype(np.float32)

    def _residuals_indices(self, data, indices):
        if issparse(data):
            data = data.toarray()
        data = np.asarray(data, dtype=np.float64)
        cell_sums = self._sums_cells[np.asarray(indices)]
        mu = np.outer(cell_sums, self._sums_genes) / self._sum_total
        denom = np.sqrt(mu + mu * mu / self._theta)
        denom[denom == 0] = 1.0
        res = (data - mu) / denom
        np.clip(res, -self._clip, self._clip, out=res)
        return res.astype(np.float32)

    def _transform_chunk(self, data, global_start: int):
        return self._residuals_dense(data, global_start)

    def _read_rows(self, start: int, end: int):
        end = min(end, self._shape[0])
        if start >= end:
            return np.empty((0, self._shape[1]), dtype=np.float32)
        raw = self._parent._read_rows(start, end)
        return self._residuals_dense(raw, start)

    def _read_row_indices(self, indices):
        indices = np.asarray(indices)
        raw = self._parent._read_row_indices(indices)
        return self._residuals_indices(raw, indices)

    def chunked(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        for start, end, raw in self._parent.chunked(chunk_size):
            yield start, end, self._residuals_dense(raw, start)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.float32)


def chunked_normalize_pearson_residuals(
    adata: AnnDataOOM,
    *,
    theta: float = 100.0,
    clip: float | None = None,
    layer: str | None = None,
    inplace: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> PearsonResidualBackedArray:
    """Out-of-core analytic Pearson-residual normalisation (Lause 2021).

    Performs a single chunked pass to accumulate per-cell and per-gene
    count sums, then wraps the source matrix in a
    :class:`PearsonResidualBackedArray` that computes residuals lazily on
    read.  Peak memory is bounded by ``chunk_size * n_vars`` -- the full
    residual matrix is never materialised.

    The lazy array is assigned to ``adata.X`` (or, when ``layer`` is
    given, written back to ``adata.layers[layer]``) and also returned.
    """
    if layer is not None and layer in adata.layers:
        X = adata.layers[layer]
    else:
        X = adata.X

    n_obs, n_vars = X.shape
    sums_genes = np.zeros(n_vars, dtype=np.float64)
    sums_cells = np.zeros(n_obs, dtype=np.float64)
    for start, end, chunk in X.chunked(chunk_size):
        if issparse(chunk):
            sums_cells[start:end] = np.asarray(chunk.sum(axis=1)).ravel()
            sums_genes += np.asarray(chunk.sum(axis=0)).ravel()
        else:
            c = np.asarray(chunk, dtype=np.float64)
            sums_cells[start:end] = c.sum(axis=1)
            sums_genes += c.sum(axis=0)
    sum_total = sums_genes.sum()

    lazy = PearsonResidualBackedArray(
        X, sums_cells, sums_genes, sum_total, theta=theta, clip=clip,
    )
    if inplace:
        if layer is not None and layer in adata.layers:
            adata.layers[layer] = lazy
        else:
            adata.X = lazy
    return lazy


# ======================================================================
# Lazy covariate regress-out
# ======================================================================


class RegressedBackedArray(TransformedBackedArray):
    """Lazy covariate regress-out: ``x_ij - C_i . beta_j``.

    Stores only the per-gene OLS coefficients ``betas`` (shape
    ``(k, n_vars)``) and the per-cell design matrix ``covariates``
    (shape ``(n_obs, k)``, ``k`` tiny). The affine residual is applied
    per chunk at read time, so the full residualised matrix is never
    materialised.
    """

    def __init__(self, parent, betas, covariates, *, norm_factors=None,
                 apply_log1p=False):
        super().__init__(parent, norm_factors=norm_factors,
                         apply_log1p=apply_log1p)
        self._betas = np.ascontiguousarray(betas, dtype=np.float64)
        self._covariates = np.ascontiguousarray(covariates, dtype=np.float64)

    def _transform_chunk(self, data, global_start):
        data = super()._transform_chunk(data, global_start)
        if issparse(data):
            data = data.toarray()
        data = np.asarray(data, dtype=np.float64)
        n = data.shape[0]
        C = self._covariates[global_start:global_start + n]
        data = data - C @ self._betas
        return data.astype(np.float32)


def chunked_regress(adata, keys=("mito_perc", "nUMIs"), *,
                    chunk_size=DEFAULT_CHUNK_SIZE):
    """Regress per-cell covariates out of every gene, fully out-of-core.

    Mirrors scanpy.pp.regress_out: design C = [ones, obs[keys[0]], ...]
    (intercept first); per-gene OLS beta_j = (C^T C)^-1 C^T x_j. Both
    C^T C (k x k) and C^T X (k x n_vars) accumulate in ONE chunked pass,
    so peak memory is bounded by chunk_size. Residual x_ij - C_i.beta_j
    is applied lazily by a RegressedBackedArray stored in
    adata.layers['regressed']. Returns the (k, n_vars) coefficients.
    """
    X = adata.X
    n_obs, n_vars = X.shape
    keys = list(keys)
    C = np.empty((n_obs, len(keys) + 1), dtype=np.float64)
    C[:, 0] = 1.0
    for j, k in enumerate(keys):
        C[:, j + 1] = np.asarray(adata.obs[k], dtype=np.float64)
    CtC = C.T @ C
    CtX = np.zeros((C.shape[1], n_vars), dtype=np.float64)
    for start, end, chunk in X.chunked(chunk_size):
        Cb = C[start:end]
        if issparse(chunk):
            CtX += np.asarray(Cb.T @ chunk)
        else:
            CtX += Cb.T @ np.asarray(chunk, dtype=np.float64)
    betas = np.linalg.lstsq(CtC, CtX, rcond=None)[0]
    if isinstance(X, TransformedBackedArray):
        reg = RegressedBackedArray(
            X._parent, betas, C,
            norm_factors=X._norm_factors, apply_log1p=X._apply_log1p)
    else:
        reg = RegressedBackedArray(X, betas, C)
    adata.layers["regressed"] = reg
    return betas


def chunked_scrublet_prepare(
    adata,
    *,
    min_cells: int = 3,
    min_genes: int = 3,
    n_top_genes: int | None = None,
    flavor: str = "seurat",
    min_disp: float = 0.5,
    max_disp: float = np.inf,
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    n_bins: int = 20,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
):
    """Bounded-memory preparation for out-of-core Scrublet doublet detection.

    Scrublet's doublet simulation (random cell-pair addition) and KNN
    classifier fundamentally need an in-memory matrix, so it cannot run
    *fully* out-of-core. However, scrublet operates only on the
    highly-variable-gene subset of **raw counts** -- so the in-RAM footprint
    is bounded by ``n_obs x n_HVG`` (a few thousand genes), never the full
    gene count. This helper does all the full-matrix work in chunks and
    materialises only that bounded subset.

    Returns ``(ad_obs, hvg_mask, n_hvg)`` where ``ad_obs`` is an in-memory
    CSR AnnData of raw counts restricted to selected HVGs and surviving
    cells, ``hvg_mask`` is the boolean ``(n_vars,)`` selection in original
    var space, and ``n_hvg`` is the gene count.
    """
    from anndata import AnnData as _AnnData

    X = adata.X
    n_obs, n_vars = X.shape

    # --- 1. gene / cell filtering: single chunked pass, vectors only ---
    n_cells_per_gene = np.zeros(n_vars, dtype=np.int64)
    n_genes_per_cell = np.zeros(n_obs, dtype=np.int64)
    for start, end, chunk in X.chunked(chunk_size):
        if issparse(chunk):
            n_cells_per_gene += chunk.getnnz(axis=0)
            n_genes_per_cell[start:end] = chunk.getnnz(axis=1)
        else:
            arr = np.asarray(chunk)
            n_cells_per_gene += (arr != 0).sum(axis=0)
            n_genes_per_cell[start:end] = (arr != 0).sum(axis=1)
    gene_pass = n_cells_per_gene >= min_cells
    cell_pass = n_genes_per_cell >= min_genes

    # --- 2. HVG selection on lazy normalize_total + log1p (X stays tiny) ---
    X_orig = adata.X
    var_backup = adata.var.copy()
    try:
        chunked_normalize_total(adata, chunk_size=chunk_size)
        chunked_log1p(adata)
        chunked_highly_variable_genes_dispersion(
            adata, flavor=flavor, n_top_genes=n_top_genes, n_bins=n_bins,
            min_mean=min_mean, max_mean=max_mean, min_disp=min_disp,
            max_disp=max_disp, chunk_size=chunk_size,
        )
        hv = np.asarray(adata.var["highly_variable"].values, dtype=bool)
    finally:
        # Restore raw X + var so the caller's adata is left pristine.
        adata.X = X_orig
        adata.var = var_backup
        try:
            adata.layers.pop("counts", None)
        except Exception:
            pass

    hvg_mask = gene_pass & hv
    sel = np.where(hvg_mask)[0]
    n_hvg = int(sel.size)
    cell_idx = np.where(cell_pass)[0]

    # --- 3. ONE chunked pass: materialise raw counts, selected cols+rows ---
    blocks = []
    for start, end, chunk in X.chunked(chunk_size):
        if issparse(chunk):
            sub = chunk[:, sel].tocsr()
        else:
            sub = csr_matrix(np.asarray(chunk)[:, sel])
        local = cell_idx[(cell_idx >= start) & (cell_idx < end)] - start
        blocks.append(sub[local])
    if blocks:
        Xsub = sparse_vstack(blocks, format="csr")
    else:
        Xsub = csr_matrix((0, n_hvg), dtype=np.float32)
    Xsub = Xsub.astype(np.float32)

    ad_obs = _AnnData(Xsub)
    obs_names = np.asarray(adata.obs_names)
    var_names = np.asarray(adata.var_names)
    ad_obs.obs_names = obs_names[cell_idx]
    ad_obs.var_names = var_names[sel]

    logger.info(
        "chunked_scrublet_prepare: materialised %d x %d raw-count HVG subset "
        "(out-of-core selection; ~%.1f MB sparse)",
        int(cell_idx.size), n_hvg, Xsub.data.nbytes / 1e6,
    )
    return ad_obs, hvg_mask, n_hvg


def materialise_for_pca(
    adata: AnnDataOOM,
    layer: str = "scaled",
) -> np.ndarray:
    """Read a layer into memory for PCA.

    If the layer is a lazy BackedArray/TransformedBackedArray, materialises
    it in chunks.  For truly large matrices, prefer :func:`chunked_pca`.
    """
    if layer in adata.layers:
        data = adata.layers[layer]
        if isinstance(data, (BackedArray, TransformedBackedArray)):
            # Materialise via chunked reads
            shape = data.shape
            result = np.empty(shape, dtype=np.float32)
            for start, end, chunk in data.chunked():
                if issparse(chunk):
                    chunk = chunk.toarray()
                result[start:end] = np.asarray(chunk, dtype=np.float32)
            return result
        if issparse(data):
            data = data.toarray()
        return np.asarray(data, dtype=np.float32)
    else:
        data = adata.X[:]
        if issparse(data):
            data = data.toarray()
        return np.asarray(data, dtype=np.float32)
