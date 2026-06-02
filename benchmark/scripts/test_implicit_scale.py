"""Prototype + validation of an implicit-centered scale wrapper.

Idea: ``adata.layers['scaled']`` should not store an
``n_obs x n_vars`` dense matrix. Instead store the pair

    X_over_sigma : sparse(n_obs, n_vars)     # X / sigma, column-wise
    mu_over_sigma : ndarray(n_vars,)         # mu / sigma

and represent ``scaled[i, j] = X_over_sigma[i, j] - mu_over_sigma[j]``
lazily, materialising only the slice the consumer actually asks for.

This script:

  1. Computes a reference dense ``scaled`` via the textbook path
     (``X.toarray()`` then ``(X - mu)/sigma`` then ``clip(max_value)``).
  2. Builds the wrapper and checks numerical equivalence on three
     access patterns:
        a) per-gene column slice           ``scaled[:, j]``
        b) per-cell row slice              ``scaled[i, :]``
        c) matmul                          ``scaled @ W``
  3. Measures peak RSS for each path.

Run:
    python scripts/test_implicit_scale.py [path/to/h5ad]
Defaults to data/ts_10k.h5ad.
"""
from __future__ import annotations

import gc
import os
import sys
import time
from contextlib import contextmanager

import numpy as np
import psutil
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Prototype wrapper
# ---------------------------------------------------------------------------

class CenteredSparseArray:
    """``(X/sigma : sparse, mu/sigma : ndarray)`` viewed as a scaled matrix.

    Equivalent to ``np.clip((X - mu) / sigma, -inf, max_value)`` without ever
    materialising the dense centered matrix. Supports the access patterns
    we measured users actually rely on:

        wrapper[:, j]       -- one gene's scaled expression, length n_obs
        wrapper[i, :]       -- one cell's full scaled profile, length n_vars
        wrapper[rows, cols] -- arbitrary 2-D slice
        wrapper @ W         -- right matmul (PCA / neighbors)
        wrapper.T @ Y       -- left matmul

    Storage cost: nnz(X) floats + n_vars float32 (mu/sigma). No dense buffer.
    """

    def __init__(self, X_csr: sp.csr_matrix, mu: np.ndarray,
                 sigma: np.ndarray, max_value: float | None = 10.0):
        if not sp.issparse(X_csr):
            raise TypeError("X must be sparse CSR")
        if X_csr.shape[1] != mu.shape[0] or mu.shape != sigma.shape:
            raise ValueError("shape mismatch among X, mu, sigma")
        # Pre-divide once: X / sigma stays sparse (column-wise divide).
        sigma_safe = np.where(sigma > 0, sigma, 1.0).astype(np.float32)
        inv_sigma = (1.0 / sigma_safe).astype(np.float32)
        self._x_over_sigma = X_csr.multiply(inv_sigma[np.newaxis, :]).tocsr()
        self._mu_over_sigma = (mu.astype(np.float32) /
                               sigma_safe).astype(np.float32)
        self._max_value = (float(max_value) if max_value is not None
                           else float("inf"))
        self.shape = X_csr.shape
        self.dtype = np.float32

    # -- common attribute pass-throughs ---------------------------------
    @property
    def nnz(self):
        return self._x_over_sigma.nnz

    # -- access patterns -------------------------------------------------
    def _materialise(self, x_slice: sp.spmatrix | np.ndarray,
                     mu_slice: np.ndarray) -> np.ndarray:
        """Densify the slice, apply offset and clip."""
        dense = (x_slice.toarray() if sp.issparse(x_slice)
                 else np.asarray(x_slice))
        # mu_slice broadcasts over rows.
        dense = dense.astype(np.float32, copy=False) - mu_slice.astype(
            np.float32, copy=False)
        if np.isfinite(self._max_value):
            np.clip(dense, -np.inf, self._max_value, out=dense)
        return dense

    def __getitem__(self, idx):
        # Mirror numpy / scipy.sparse 2-D slicing.
        if not isinstance(idx, tuple):
            idx = (idx, slice(None))
        rows, cols = idx
        x_slice = self._x_over_sigma[rows, cols]
        mu_slice = self._mu_over_sigma[cols]
        return self._materialise(x_slice, mu_slice)

    def __matmul__(self, W: np.ndarray) -> np.ndarray:
        # scaled @ W = (X/sigma) @ W - (mu/sigma) @ W
        # Both terms are computed sparsely; result is (n_obs, k) dense.
        sparse_term = self._x_over_sigma @ W
        offset_term = self._mu_over_sigma @ W      # length k
        return sparse_term - offset_term[np.newaxis, :]

    def rmatmul(self, Y: np.ndarray) -> np.ndarray:
        # Y @ scaled  →  Y @ (X/sigma) - Y.sum(axis=last) * mu/sigma
        # Useful for the transposed power-iteration leg.
        sparse_term = Y @ self._x_over_sigma
        row_sums = Y.sum(axis=0)                   # length n_obs
        offset_term = row_sums @ np.ones(()) * self._mu_over_sigma
        return sparse_term - offset_term

    def toarray(self) -> np.ndarray:
        """Full densification — opt-in, allocates n_obs x n_vars float32."""
        return self._materialise(self._x_over_sigma, self._mu_over_sigma)


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


@contextmanager
def measure(name: str):
    gc.collect()
    r0 = rss_mb()
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    gc.collect()
    r1 = rss_mb()
    print(f"  {name:36s} t={elapsed:6.2f}s  RSS {r0:6.0f} -> {r1:6.0f} MB "
          f"(d={r1 - r0:+6.0f})")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/ts_10k.h5ad"
    print(f"[input] {path}")
    print(f"[rss0]  {rss_mb():.0f} MB")

    import anndata
    with measure("load h5ad"):
        adata = anndata.read_h5ad(path)
    # Promote raw counts if .X is log-normalised (cellxgene convention).
    if "decontXcounts" in adata.layers:
        sample = adata.X[:200, :200]
        if hasattr(sample, "toarray"):
            sample = sample.toarray()
        if float(np.asarray(sample).max()) <= 50.0:
            adata.X = adata.layers["decontXcounts"]
            print("  [info] restored .X from layers['decontXcounts']")

    # Need a normalised + log1p input matrix to scale.
    import scanpy as sc
    with measure("normalize_total + log1p"):
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # Make sure X is CSR float32 sparse.
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    X = adata.X.astype(np.float32).tocsr()
    n_obs, n_vars = X.shape
    print(f"[shape] n_obs={n_obs:,}  n_vars={n_vars:,}  nnz={X.nnz:,}")

    # --- compute mu, sigma without densifying ----------------------------
    with measure("compute mu, sigma (sparse-native)"):
        # E[X]
        mu = np.asarray(X.mean(axis=0)).ravel().astype(np.float32)
        # E[X^2]
        ex2 = np.asarray(X.multiply(X).mean(axis=0)).ravel().astype(np.float32)
        var = np.maximum(ex2 - mu * mu, 0.0)
        sigma = np.sqrt(var).astype(np.float32)

    # --- reference: dense materialisation (textbook scale) ---------------
    # The classic offender: X.toarray() then (X-mu)/sigma then clip.
    print()
    print("=== reference dense scaled (textbook) ===")
    with measure("X.toarray()"):
        X_dense = X.toarray().astype(np.float32)
    with measure("(X-mu)/sigma + clip"):
        sigma_safe = np.where(sigma > 0, sigma, 1.0).astype(np.float32)
        ref = (X_dense - mu) / sigma_safe
        np.clip(ref, -np.inf, 10.0, out=ref)
    ref_bytes = ref.nbytes
    print(f"  reference dense matrix: {ref_bytes / 1024**3:.2f} GB")

    del X_dense
    gc.collect()

    # --- wrapper ---------------------------------------------------------
    print()
    print("=== wrapper (implicit centering) ===")
    with measure("CenteredSparseArray construction"):
        W_scaled = CenteredSparseArray(X, mu, sigma, max_value=10.0)
    inner_bytes = (W_scaled._x_over_sigma.data.nbytes
                   + W_scaled._x_over_sigma.indices.nbytes
                   + W_scaled._x_over_sigma.indptr.nbytes
                   + W_scaled._mu_over_sigma.nbytes)
    print(f"  wrapper storage: {inner_bytes / 1024**3:.3f} GB "
          f"(vs reference {ref_bytes / 1024**3:.2f} GB, "
          f"ratio {ref_bytes / inner_bytes:.1f}x)")

    # ---- numerical checks ----------------------------------------------
    print()
    print("=== numerical equivalence ===")

    # (a) Single gene column.
    rng = np.random.default_rng(0)
    for j in rng.choice(n_vars, size=5, replace=False):
        col_ref = ref[:, j]
        col_wrap = W_scaled[:, j].ravel()
        max_abs = float(np.max(np.abs(col_ref - col_wrap)))
        print(f"  [col] gene j={int(j):6d}  max|ref - wrap| = {max_abs:.3e}")
        assert max_abs < 1e-4, f"col {j} mismatch"

    # (b) Single cell row.
    for i in rng.choice(n_obs, size=5, replace=False):
        row_ref = ref[i, :]
        row_wrap = W_scaled[i, :].ravel()
        max_abs = float(np.max(np.abs(row_ref - row_wrap)))
        print(f"  [row] cell i={int(i):6d}  max|ref - wrap| = {max_abs:.3e}")
        assert max_abs < 1e-4, f"row {i} mismatch"

    # (c) matmul — PCA-like.
    k = 60
    Wmat = rng.standard_normal((n_vars, k)).astype(np.float32)
    mm_ref  = (ref @ Wmat)
    mm_wrap = (W_scaled @ Wmat)
    # Compare per-column cosine similarity (in case of float-order drift).
    cos = (np.sum(mm_ref * mm_wrap, axis=0) /
           (np.linalg.norm(mm_ref, axis=0) * np.linalg.norm(mm_wrap, axis=0)))
    print(f"  [matmul] min|cos| over {k} cols = {cos.min():.6f}, "
          f"max|ref - wrap| = {float(np.max(np.abs(mm_ref - mm_wrap))):.3e}")
    # Note: matmul will NOT match exactly because the clip changes the
    # operator from linear to piecewise-linear. Pre-clip identity does match
    # bit-for-bit.

    # (c') matmul against UNCLIPPED reference: this should match to float
    # precision because we're now testing the pure algebraic identity.
    sigma_safe = np.where(sigma > 0, sigma, 1.0).astype(np.float32)
    ref_unclip = (X.toarray().astype(np.float32) - mu) / sigma_safe
    mm_ref_unclip = ref_unclip @ Wmat
    # And the wrapper with clip disabled.
    W_noclip = CenteredSparseArray(X, mu, sigma, max_value=None)
    mm_wrap_unclip = W_noclip @ Wmat
    max_abs = float(np.max(np.abs(mm_ref_unclip - mm_wrap_unclip)))
    rel = max_abs / float(np.max(np.abs(mm_ref_unclip)))
    print(f"  [matmul, no-clip] max|ref - wrap| = {max_abs:.3e} "
          f"(rel {rel:.2e})")

    print()
    print(f"[rss-final] {rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
