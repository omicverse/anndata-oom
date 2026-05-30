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
from scipy.sparse import issparse

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
            n_genes_per_cell[start:end] = np.asarray((chunk != 0).sum(axis=1)).ravel()
            n_cells_per_gene += np.asarray((chunk != 0).sum(axis=0)).ravel()
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
    """
    X = adata.X
    n_obs, n_vars = X.shape

    # --- Save raw counts as a lazy reference (no copy!) ---
    adata.layers["counts"] = X  # BackedArray reference, zero memory cost

    gene_subset = None
    if exclude_highly_expressed:
        # Single-pass: compute row sums AND per-gene high-expression counts together
        counts_per_cell = np.zeros(n_obs, dtype=np.float64)
        gene_hi_count = np.zeros(n_vars, dtype=np.int64)
        for start, end, chunk in X.chunked(chunk_size):
            if issparse(chunk):
                row_sums = np.asarray(chunk.sum(axis=1)).ravel()
                dense = chunk.toarray()
            else:
                row_sums = chunk.sum(axis=1)
                dense = chunk
            counts_per_cell[start:end] = row_sums
            thresholds = row_sums * max_fraction
            gene_hi_count += (dense > thresholds[:, np.newaxis]).sum(axis=0)

        gene_subset = gene_hi_count == 0
        n_excluded = (~gene_subset).sum()
        if n_excluded > 0:
            logger.info("Excluding %d highly-expressed genes from normalization", n_excluded)

        # Second pass: recompute row sums using only non-highly-expressed genes
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
        # --- Compute per-cell total counts (single pass) ---
        counts_per_cell = X.sum(axis=1, chunk_size=chunk_size)

    # --- Determine target_sum ---
    if target_sum is None:
        nonzero = counts_per_cell[counts_per_cell > 0]
        target_sum = float(np.median(nonzero)) if len(nonzero) > 0 else 1.0

    # --- Build normalization factors ---
    norm_factors = counts_per_cell / target_sum
    # Store for reference
    adata.obs["_norm_factor"] = norm_factors

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
        if issparse(chunk):
            chunk = chunk.toarray()
        chunk = chunk.astype(np.float64)
        batch_size = chunk.shape[0]

        # Batch Welford update
        batch_mean = chunk.mean(axis=0)
        batch_var = chunk.var(axis=0) * batch_size  # sum of squared deviations

        new_count = count + batch_size
        delta = batch_mean - mean
        mean = mean + delta * (batch_size / new_count)
        M2 = M2 + batch_var + (delta ** 2) * (count * batch_size / new_count)
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
    if isinstance(X, TransformedBackedArray):
        scaled = ScaledBackedArray(
            X._parent, mean, std, max_value,
            norm_factors=X._norm_factors,
            apply_log1p=X._apply_log1p,
        )
    else:
        scaled = ScaledBackedArray(X, mean, std, max_value)

    adata.layers["scaled"] = scaled


# ======================================================================
# Chunked PCA via IncrementalPCA
# ======================================================================


def chunked_pca(
    adata: AnnDataOOM,
    *,
    layer: str = "scaled",
    n_comps: int = 50,
    n_oversamples: int = 10,
    n_power_iters: int = 4,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA via randomized SVD with chunked matrix products.

    Uses the Halko–Martinsson–Tropp (2011) algorithm.  The full
    ``(n_obs, n_vars)`` matrix is never materialised; all matrix
    products are accumulated in row-chunks.

    Memory footprint: ``O(n_obs * k + n_vars * k)`` where
    ``k = n_comps + n_oversamples``.

    Parameters
    ----------
    n_comps : int
        Number of principal components.
    n_oversamples : int
        Additional random vectors for accuracy (default 10).
    n_power_iters : int
        Power iteration steps — 2 is usually enough for scRNA-seq.
    chunk_size : int
        Rows per I/O chunk.
    random_state : int
        Random seed.

    Returns
    -------
    X_pca : ndarray, shape (n_obs, n_comps)
    components : ndarray, shape (n_comps, n_vars)
    variance_ratio : ndarray, shape (n_comps,)
    """
    if layer in adata.layers:
        X = adata.layers[layer]
    else:
        X = adata.X

    n_obs, n_vars = X.shape
    k = min(n_comps + n_oversamples, n_obs, n_vars)
    rng = np.random.RandomState(random_state)

    def _to_dense(chunk):
        if issparse(chunk):
            return chunk.toarray().astype(np.float64)
        return np.asarray(chunk, dtype=np.float64)

    # ------------------------------------------------------------------
    # Step 1: Y = X @ Omega  (one pass over X)
    # ------------------------------------------------------------------
    Omega = rng.standard_normal((n_vars, k))  # (n_vars, k)
    Y = np.zeros((n_obs, k), dtype=np.float64)
    for start, end, chunk in X.chunked(chunk_size):
        Y[start:end] = _to_dense(chunk) @ Omega

    # ------------------------------------------------------------------
    # Step 2: Power iteration for accuracy  (2 passes per iteration)
    # ------------------------------------------------------------------
    for _ in range(n_power_iters):
        # Z = X.T @ Y   (accumulate across row-chunks)
        Z = np.zeros((n_vars, k), dtype=np.float64)
        for start, end, chunk in X.chunked(chunk_size):
            Z += _to_dense(chunk).T @ Y[start:end]
        Z, _ = np.linalg.qr(Z)

        # Y = X @ Z
        for start, end, chunk in X.chunked(chunk_size):
            Y[start:end] = _to_dense(chunk) @ Z
        Y, _ = np.linalg.qr(Y)

    # Final QR
    Q, _ = np.linalg.qr(Y)  # (n_obs, k)

    # ------------------------------------------------------------------
    # Step 3: B = Q.T @ X   (one pass over X)
    # ------------------------------------------------------------------
    B = np.zeros((k, n_vars), dtype=np.float64)
    for start, end, chunk in X.chunked(chunk_size):
        B += Q[start:end].T @ _to_dense(chunk)

    # ------------------------------------------------------------------
    # Step 4: SVD on small matrix B  (k × n_vars, fits in memory)
    # ------------------------------------------------------------------
    U_B, S, Vt = np.linalg.svd(B, full_matrices=False)

    # Recover full left singular vectors
    U = Q @ U_B  # (n_obs, k)

    # Truncate to n_comps
    n_comps = min(n_comps, k)
    X_pca = (U[:, :n_comps] * S[:n_comps]).astype(np.float32)
    components = Vt[:n_comps]  # (n_comps, n_vars)

    # Variance explained
    total_var = S ** 2
    variance_ratio = (total_var[:n_comps] / total_var.sum()).astype(np.float64)

    return X_pca, components, variance_ratio


def chunked_pearson_residual_variance(
    X: BackedArray,
    *,
    theta: float = 100.0,
    clip: float | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-gene variance of clipped Pearson residuals, fully chunked.

    Only 2 passes over the data:
    - Pass 1: per-gene sums and per-cell sums
    - Pass 2: accumulate sum(res) and sum(res²) per gene

    Parameters
    ----------
    X : BackedArray
        Raw count matrix (backed, not normalized).
    theta : float
        Negative binomial overdispersion parameter.
    clip : float or None
        Clip residuals to [-clip, clip]. If None, uses sqrt(n_obs).

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
) -> None:
    """HVG selection via Pearson residuals, fully chunked.

    Replaces the scanpy implementation for AnnDataOOM — never
    materialises the full count matrix.
    """
    import pandas as pd

    if layer is not None and layer in adata.layers:
        X = adata.layers[layer]
    else:
        X = adata.X

    if batch_key is None:
        residual_var, gene_mean, gene_var = chunked_pearson_residual_variance(
            X, theta=theta, clip=clip, chunk_size=chunk_size,
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
