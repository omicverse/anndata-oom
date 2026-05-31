"""Fused-pass Pearson HVG must equal the from-scratch two-pass version.

`chunked_normalize_total` now stashes the per-cell / per-gene Pass-1
stats in `adata.uns['_pearson_precompute']`, and
`chunked_pearson_residual_variance` (and `_highly_variable_genes_pearson`)
can consume that dict to skip its own Pass 1. This regression test pins
down that the speedup path is numerically equivalent to the legacy
two-pass path on the same input.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def synthetic_h5ad(tmp_path):
    """Small CSR-counts h5ad on disk that AnnDataOOM can open."""
    import anndata as ad

    rng = np.random.default_rng(0)
    n_obs, n_vars = 600, 400
    # Per-cell library sizes drawn from a log-uniform to mimic real depths.
    depths = rng.integers(low=500, high=5000, size=n_obs)
    rows, cols, vals = [], [], []
    for i in range(n_obs):
        n_nz = max(5, int(depths[i] * 0.05))
        gene_idx = rng.choice(n_vars, size=n_nz, replace=False)
        rows.extend([i] * n_nz)
        cols.extend(gene_idx.tolist())
        vals.extend(rng.integers(1, 20, size=n_nz).tolist())
    X = sp.csr_matrix(
        (np.asarray(vals, dtype=np.float32),
         (np.asarray(rows), np.asarray(cols))),
        shape=(n_obs, n_vars),
    )
    adata = ad.AnnData(X=X)
    out = tmp_path / "synth.h5ad"
    adata.write_h5ad(out)
    return out


def _open(path):
    import anndataoom as oom
    return oom.read(str(path))


def test_normalize_total_stashes_precompute(synthetic_h5ad):
    """chunked_normalize_total writes a well-formed precompute dict."""
    import anndataoom as oom

    adata = _open(synthetic_h5ad)
    try:
        oom.chunked_normalize_total(adata, target_sum=1e4)
        pre = adata.uns.get("_pearson_precompute")
        assert pre is not None, "normalize_total did not stash precompute"
        n_obs, n_vars = adata.shape
        assert pre["sums_cells"].shape == (n_obs,)
        assert pre["sums_genes"].shape == (n_vars,)
        assert pre["sq_sums_genes"].shape == (n_vars,)
        assert pre["n_obs"] == n_obs and pre["n_vars"] == n_vars
        assert np.isfinite(pre["sum_total"]) and pre["sum_total"] > 0
        # Mean depth sanity (recoverable from the precompute).
        mean_depth = pre["sums_cells"].mean()
        assert 100 < mean_depth < 10_000
    finally:
        adata.close()


def test_pearson_residual_variance_matches_legacy(synthetic_h5ad):
    """The precompute-fast-path output equals the from-scratch path."""
    import anndataoom as oom
    from anndataoom._chunked_ops import chunked_pearson_residual_variance

    # We need the raw counts BackedArray for the residual variance call,
    # which is what `layer='counts'` ends up pointing at after normalize.
    adata = _open(synthetic_h5ad)
    try:
        oom.chunked_normalize_total(adata, target_sum=1e4)
        pre = adata.uns["_pearson_precompute"]
        X_counts = adata.layers["counts"]

        # Path A: from scratch (legacy two-pass).
        rv_a, gm_a, gv_a = chunked_pearson_residual_variance(
            X_counts, theta=100.0, clip=None,
        )
        # Path B: with precompute (skips Pass 1).
        rv_b, gm_b, gv_b = chunked_pearson_residual_variance(
            X_counts, theta=100.0, clip=None, precomputed=pre,
        )
        np.testing.assert_allclose(rv_a, rv_b, atol=1e-10, rtol=1e-12)
        np.testing.assert_allclose(gm_a, gm_b, atol=1e-10, rtol=1e-12)
        np.testing.assert_allclose(gv_a, gv_b, atol=1e-10, rtol=1e-12)
    finally:
        adata.close()


def test_hvg_pearson_auto_consumes_precompute(synthetic_h5ad):
    """`chunked_highly_variable_genes_pearson` defaults to auto-consume
    the precompute that `chunked_normalize_total` just wrote."""
    import anndataoom as oom
    from anndataoom._chunked_ops import chunked_highly_variable_genes_pearson

    # Path A: legacy (no precompute).
    a = _open(synthetic_h5ad)
    try:
        oom.chunked_normalize_total(a, target_sum=1e4)
        a.uns.pop("_pearson_precompute", None)  # disable auto
        chunked_highly_variable_genes_pearson(
            a, n_top_genes=50, layer="counts", precomputed=None,
        )
        mask_a = a.var["highly_variable"].values.astype(bool)
        rv_a = a.var["residual_variances"].values
    finally:
        a.close()

    # Path B: auto-precompute (default).
    b = _open(synthetic_h5ad)
    try:
        oom.chunked_normalize_total(b, target_sum=1e4)
        chunked_highly_variable_genes_pearson(
            b, n_top_genes=50, layer="counts",  # precomputed='auto' default
        )
        mask_b = b.var["highly_variable"].values.astype(bool)
        rv_b = b.var["residual_variances"].values
    finally:
        b.close()

    # Same per-gene ranking source --> identical mask + values.
    np.testing.assert_allclose(rv_a, rv_b, atol=1e-10, rtol=1e-12)
    np.testing.assert_array_equal(mask_a, mask_b)


def test_stale_precompute_rejected(synthetic_h5ad):
    """If the caller subsets X between normalize and HVG, the staleness
    guard either drops the precompute (auto mode) or raises (explicit)."""
    import anndataoom as oom
    from anndataoom._chunked_ops import chunked_pearson_residual_variance

    adata = _open(synthetic_h5ad)
    try:
        oom.chunked_normalize_total(adata, target_sum=1e4)
        pre = adata.uns["_pearson_precompute"]
        X_counts = adata.layers["counts"]
        # Forge a stale shape.
        bad_pre = dict(pre)
        bad_pre["sums_cells"] = pre["sums_cells"][:-10]
        with pytest.raises(ValueError, match="Stale precomputed"):
            chunked_pearson_residual_variance(
                X_counts, precomputed=bad_pre,
            )
    finally:
        adata.close()
