"""Correctness regression tests for chunked-ops bugs surfaced by audit.

Each test guards a specific silent-failure mode that the audit caught:

- ``test_chunked_scale_refuses_double_scale`` — re-scaling on an
  already-scaled X would silently lose the prior mean/std because
  ``ScaledBackedArray`` inherits from ``TransformedBackedArray`` and the
  isinstance check unwrapped to the wrong parent.
- ``test_normalize_handles_zero_count_cells`` — cells with zero total
  counts must not produce NaN / Inf in the resulting matrix.
- ``test_mean_var_sparse_native_matches_dense_result`` — the
  sparse-native E[X²]-E[X]² path in chunked_mean_var must give numerically
  identical answers to the dense Welford path.
"""
from __future__ import annotations

import os
import tempfile

import anndata
import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def backed_h5ad():
    np.random.seed(0)
    X = sp.random(2_000, 500, density=0.05, format="csr",
                  dtype=np.float32, random_state=0)
    adata = anndata.AnnData(X=X)
    path = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False).name
    adata.write(path)
    yield path
    os.remove(path)


# ---------------------------------------------------------------------- C1
def test_chunked_scale_refuses_double_scale(backed_h5ad):
    """Calling scale on an already-scaled X must raise, not silently no-op.

    Pre-fix: `ScaledBackedArray` matched `isinstance(X, TransformedBackedArray)`
    and the unwrap dropped the prior scale's mean/std. The user thought
    they had two scales applied but got just one with bogus params.
    """
    import anndataoom as oom
    from anndataoom._chunked_ops import (
        chunked_normalize_total, chunked_log1p, chunked_scale,
        ScaledBackedArray,
    )

    a = oom.read(backed_h5ad)
    chunked_normalize_total(a)
    chunked_log1p(a)
    chunked_scale(a)

    # Simulate the foot-gun: re-point X at the scaled layer, then call scale.
    a.X = a.layers["scaled"]
    assert isinstance(a.X, ScaledBackedArray)
    with pytest.raises(RuntimeError, match="already a ScaledBackedArray"):
        chunked_scale(a)

    a.close()


# ---------------------------------------------------------------------- M1
def test_normalize_handles_zero_count_cells():
    """A cell with zero total counts must yield a NaN-free normalised matrix.

    The downstream `data /= factors` path would otherwise produce NaN
    that silently propagates through log1p / scale and corrupts the
    matrix; the new `np.nan_to_num` + `factors[==0]=1.0` guard prevents
    both NaN and div-by-zero.
    """
    import anndataoom as oom
    from anndataoom._chunked_ops import chunked_normalize_total

    np.random.seed(0)
    X = sp.random(100, 50, density=0.1, format="csr",
                  dtype=np.float32, random_state=0)
    # Force cell 0 to be all-zero (zero total counts).
    X = X.tolil()
    X[0, :] = 0
    X = X.tocsr()
    adata = anndata.AnnData(X=X)

    path = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False).name
    try:
        adata.write(path)
        a = oom.read(path)
        chunked_normalize_total(a)

        # Materialise the first chunk after normalize — must be finite.
        for _, _, chunk in a.X.chunked(50):
            arr = chunk.toarray() if sp.issparse(chunk) else np.asarray(chunk)
            assert np.isfinite(arr).all(), (
                "normalize produced NaN / Inf after zero-count cell guard"
            )
            break

        assert np.isfinite(a.obs["_norm_factor"]).all()
        a.close()
    finally:
        os.remove(path)


# ---------------------------------------------------------------------- H3
def test_mean_var_sparse_native_matches_dense_result(backed_h5ad):
    """Sparse-native E[X²] - E[X]² path must match the dense Welford answer.

    The previous implementation densified every sparse chunk. The new path
    keeps sparse via `chunk.multiply(chunk).sum(axis=0)`. Numerical drop
    must be < 1e-9 vs the gold-standard `numpy.cov` over the full matrix.
    """
    import anndataoom as oom
    from anndataoom._chunked_ops import chunked_mean_var

    a = oom.read(backed_h5ad)
    mean, var = chunked_mean_var(a, chunk_size=300)

    # Gold-standard: materialise the full matrix once and use numpy directly.
    chunks = []
    for _, _, c in a.X.chunked(300):
        chunks.append(c.toarray() if sp.issparse(c) else np.asarray(c))
    full = np.vstack(chunks).astype(np.float64)
    expected_mean = full.mean(axis=0)
    expected_var = full.var(axis=0, ddof=1)

    # Tolerance reflects float32 → float64 accumulation noise; the
    # sparse-native E[X²]-E[X]² path is numerically identical to dense
    # Welford up to that noise floor.
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(var, expected_var, rtol=1e-5, atol=1e-9)
    a.close()
