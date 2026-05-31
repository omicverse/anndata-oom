"""Numerical equivalence + storage-cost tests for CenteredSparseArray.

The wrapper is the in-memory counterpart of ScaledBackedArray. These
tests pin down the contract that the wrapper-vs-dense difference is at
float32 ulp for slice access, and that the matmul identity holds bit-by-
bit when ``max_value=None``.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from anndataoom import CenteredSparseArray


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic(n_obs=512, n_vars=128, density=0.08, seed=0):
    """Sparse log1p-of-counts-like CSR + reference dense scaled matrix."""
    rng = np.random.default_rng(seed)
    X = sp.random(
        n_obs, n_vars, density=density, format="csr",
        dtype=np.float32, random_state=rng,
    )
    # Inject a few near-zero-variance columns so the std==0 branch is hit.
    X = X.tolil()
    X[:, 3] = 0.0
    X[:, 7] = 0.0
    X = X.tocsr()

    mean = np.asarray(X.mean(axis=0)).ravel().astype(np.float32)
    ex2 = np.asarray(X.multiply(X).mean(axis=0)).ravel().astype(np.float32)
    var = np.maximum(ex2 - mean * mean, 0.0)
    std = np.sqrt(var).astype(np.float32)

    # Reference: textbook dense scale with clip.
    std_safe = np.where(std > 0, std, 1.0).astype(np.float32)
    ref = (X.toarray().astype(np.float32) - mean) / std_safe
    ref_unclip = ref.copy()
    np.clip(ref, -np.inf, 10.0, out=ref)
    return X, mean, std, ref, ref_unclip


@pytest.fixture
def fixtures():
    return _make_synthetic()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_shape_dtype_nnz(fixtures):
    X, mean, std, ref, _ = fixtures
    w = CenteredSparseArray(X, mean, std, max_value=10.0)
    assert w.shape == X.shape == ref.shape
    assert w.dtype == np.float32
    # The wrapper stores ``X / σ``, which has *the same* sparsity pattern
    # as X — zero-variance columns are 0 in X, then 0 after the divide.
    assert w.nnz == X.nnz


def test_column_slice_bit_equal(fixtures):
    """Single-gene access — the canonical 取基因表达 path — must match
    the dense reference to float32 ulp."""
    X, mean, std, ref, _ = fixtures
    w = CenteredSparseArray(X, mean, std, max_value=10.0)
    rng = np.random.default_rng(1)
    for j in rng.choice(X.shape[1], size=20, replace=False):
        col = w[:, j].ravel()
        np.testing.assert_allclose(col, ref[:, j], atol=1e-5, rtol=0)


def test_row_slice_bit_equal(fixtures):
    """Single-cell access must match the dense reference."""
    X, mean, std, ref, _ = fixtures
    w = CenteredSparseArray(X, mean, std, max_value=10.0)
    rng = np.random.default_rng(2)
    for i in rng.choice(X.shape[0], size=20, replace=False):
        row = w[i, :].ravel()
        np.testing.assert_allclose(row, ref[i, :], atol=1e-5, rtol=0)


def test_rectangle_slice(fixtures):
    """Sub-rectangle access (a la ``adata[cells, genes]``)."""
    X, mean, std, ref, _ = fixtures
    w = CenteredSparseArray(X, mean, std, max_value=10.0)
    rows = np.array([5, 17, 42, 100])
    cols = np.array([0, 1, 9, 50, 99])
    block = w[np.ix_(rows, cols)]
    expected = ref[np.ix_(rows, cols)]
    np.testing.assert_allclose(block, expected, atol=1e-5, rtol=0)


def test_matmul_identity_noclip(fixtures):
    """``wrapper @ W`` matches ``ref @ W`` to float precision when
    ``max_value=None`` (the algebraic identity is exact, modulo
    float-order drift)."""
    X, mean, std, _, ref_unclip = fixtures
    w = CenteredSparseArray(X, mean, std, max_value=None)
    rng = np.random.default_rng(3)
    k = 32
    W = rng.standard_normal((X.shape[1], k)).astype(np.float32)
    out_w = w @ W
    out_r = ref_unclip @ W
    np.testing.assert_allclose(out_w, out_r, atol=1e-3, rtol=1e-5)


def test_rmatmul_identity_noclip(fixtures):
    """``Y @ wrapper`` matches ``Y @ ref`` to float precision."""
    X, mean, std, _, ref_unclip = fixtures
    w = CenteredSparseArray(X, mean, std, max_value=None)
    rng = np.random.default_rng(4)
    k = 16
    Y = rng.standard_normal((k, X.shape[0])).astype(np.float32)
    out_w = Y @ w
    out_r = Y @ ref_unclip
    np.testing.assert_allclose(out_w, out_r, atol=1e-3, rtol=1e-5)


def test_transpose_matmul(fixtures):
    """``wrapper.T @ Y`` is the rmatmul leg of randomised SVD."""
    X, mean, std, _, ref_unclip = fixtures
    w = CenteredSparseArray(X, mean, std, max_value=None)
    rng = np.random.default_rng(5)
    k = 16
    Y = rng.standard_normal((X.shape[0], k)).astype(np.float32)
    out_w = w.T @ Y
    out_r = ref_unclip.T @ Y
    np.testing.assert_allclose(out_w, out_r, atol=1e-3, rtol=1e-5)


def test_toarray_matches_reference(fixtures):
    """Full densification reproduces the textbook scaled matrix."""
    X, mean, std, ref, _ = fixtures
    w = CenteredSparseArray(X, mean, std, max_value=10.0)
    dense = w.toarray()
    np.testing.assert_allclose(dense, ref, atol=1e-5, rtol=0)
    assert dense.dtype == np.float32


def test_storage_is_sparse_sized(fixtures):
    """Wrapper storage should track nnz(X), not n_obs × n_vars."""
    X, mean, std, _, _ = fixtures
    w = CenteredSparseArray(X, mean, std, max_value=10.0)
    inner_bytes = (
        w._x_over_sigma.data.nbytes
        + w._x_over_sigma.indices.nbytes
        + w._x_over_sigma.indptr.nbytes
        + w._mu_over_sigma.nbytes
    )
    dense_bytes = X.shape[0] * X.shape[1] * 4
    # On the synthetic 512×128 @ 8% density fixture the wrapper should
    # come in well under half the dense matrix's footprint.
    assert inner_bytes < dense_bytes / 2


def test_zero_variance_gene_safe(fixtures):
    """Genes with std==0 must not blow up (div-by-zero guard)."""
    X, mean, std, ref, _ = fixtures
    assert std[3] == 0
    assert std[7] == 0
    w = CenteredSparseArray(X, mean, std, max_value=10.0)
    col3 = w[:, 3].ravel()
    col7 = w[:, 7].ravel()
    # On a zero-variance column X is all-zero and mean is 0, so the
    # scaled result is the all-zero vector.
    np.testing.assert_allclose(col3, ref[:, 3], atol=1e-5, rtol=0)
    np.testing.assert_allclose(col7, ref[:, 7], atol=1e-5, rtol=0)


def test_clip_lowers_top_tail():
    """``max_value=10`` actually clips: a single huge spike in an
    otherwise zero column has a scaled value of ``sqrt(n-1) ≈ 14`` for a
    population of ``n``, comfortably above the clip."""
    n_obs, n_vars = 200, 4
    data = np.zeros((n_obs, n_vars), dtype=np.float32)
    data[0, 1] = 1000.0  # one spike → scaled ≈ sqrt(199) ≈ 14.1 > 10
    X = sp.csr_matrix(data)
    mean = np.asarray(X.mean(axis=0)).ravel().astype(np.float32)
    ex2 = np.asarray(X.multiply(X).mean(axis=0)).ravel().astype(np.float32)
    std = np.sqrt(np.maximum(ex2 - mean * mean, 0.0)).astype(np.float32)
    w = CenteredSparseArray(X, mean, std, max_value=10.0)
    col1 = w[:, 1].ravel()
    # Spike clipped; zero rows ride at ``-μ/σ`` (small negative).
    assert col1[0] == pytest.approx(10.0, abs=1e-5)
    assert col1[1] < 0  # zero row pre-clip = -μ/σ


def test_no_clip_passes_through():
    """``max_value=None`` disables the clip — top-tail spike survives."""
    n_obs, n_vars = 200, 4
    data = np.zeros((n_obs, n_vars), dtype=np.float32)
    data[0, 1] = 1000.0  # scaled ≈ 14.1 unclipped
    X = sp.csr_matrix(data)
    mean = np.asarray(X.mean(axis=0)).ravel().astype(np.float32)
    ex2 = np.asarray(X.multiply(X).mean(axis=0)).ravel().astype(np.float32)
    std = np.sqrt(np.maximum(ex2 - mean * mean, 0.0)).astype(np.float32)
    w_clip = CenteredSparseArray(X, mean, std, max_value=10.0)
    w_noclip = CenteredSparseArray(X, mean, std, max_value=None)
    # With clip the spike caps at 10; without clip it stays > 10.
    assert w_clip[0, 1].ravel()[0] == pytest.approx(10.0, abs=1e-5)
    spike = w_noclip[0, 1].ravel()[0]
    assert spike > 10.0
    assert spike < 20.0  # sqrt(199) ≈ 14.1


def test_input_validation():
    X = sp.random(10, 20, density=0.3, format="csr",
                  dtype=np.float32, random_state=0)
    mean = np.zeros(20, dtype=np.float32)
    std = np.ones(20, dtype=np.float32)
    # Dense input rejected.
    with pytest.raises(TypeError):
        CenteredSparseArray(X.toarray(), mean, std)
    # Shape mismatch rejected.
    with pytest.raises(ValueError):
        CenteredSparseArray(X, mean[:-1], std)
    with pytest.raises(ValueError):
        CenteredSparseArray(X, mean, std[:-1])
