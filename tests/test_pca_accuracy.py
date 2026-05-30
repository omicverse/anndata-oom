"""Accuracy regression tests for the three `chunked_pca` execution paths.

The v0.2.0 PCA acceleration adds two new paths on top of the legacy
chunked Halko randomised SVD:

  1. auto-materialise + sklearn `randomized_svd` (mathematically
     identical to running randomised SVD on the full dense scaled
     matrix — same algorithm class as the legacy path)
  2. implicit-centered chunked: rewrites the scaled matrix's matvec
     via `(N - μ)/σ · W = N · diag(1/σ) · W - (μ/σ) · W` on the sparse
     normalize+log1p view; SKIPS the scale's max_value clip because
     clipping is non-linear and breaks the identity.

These tests assert per-component |cosine similarity| > 0.999 for paths
(1) and (2) versus the legacy path (3) on a 5 k-row synthetic CSR
input. The threshold is calibrated to absorb the float32→float64 SVD
noise that the randomised algorithm exhibits at small singular values
while still catching real algorithmic regressions.

Run via ``pytest tests/test_pca_accuracy.py -v``.
"""
from __future__ import annotations

import os
import tempfile

import anndata
import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture(scope="module")
def backed_h5ad():
    """Small CSR-h5ad with realistic count distribution."""
    np.random.seed(0)
    n_obs, n_vars, density = 5_000, 500, 0.10
    X = sp.random(n_obs, n_vars, density=density, format="csr",
                  dtype=np.float32, random_state=0)
    X.data = (X.data * 30).astype(np.float32)  # counts-ish magnitude
    adata = anndata.AnnData(X=X)
    path = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False).name
    adata.write(path)
    yield path
    os.remove(path)


def _build_scaled(backed_path):
    """Run normalize → log1p → scale through the public chunked ops,
    returning the ScaledBackedArray reachable via `adata.layers['scaled']`.
    """
    import anndataoom as oom
    from anndataoom._chunked_ops import (
        chunked_normalize_total, chunked_log1p, chunked_scale,
    )

    a = oom.read(backed_path)
    chunked_normalize_total(a, chunk_size=500)
    chunked_log1p(a)
    chunked_scale(a, chunk_size=500)
    return a


def _cosine_per_component(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """|cos| per column, sign-flip invariant (random SVD is sign-ambig)."""
    k = min(A.shape[1], B.shape[1])
    A, B = A[:, :k], B[:, :k]
    num = (A * B).sum(axis=0)
    den = np.linalg.norm(A, axis=0) * np.linalg.norm(B, axis=0)
    den = np.where(den == 0, 1.0, den)
    return np.abs(num / den)


# ---- path-2 vs path-3 (implicit-centered vs legacy chunked) -----------


def test_implicit_matches_legacy(backed_h5ad):
    """Path 2 (implicit centering) ≈ Path 3 (legacy densifying chunked).

    The implicit path skips clipping, so the comparison is to a legacy
    path with `ScaledBackedArray.max_value` set to None at the scale
    step. We can't easily rerun scale here, so we compare both paths
    on the SAME ScaledBackedArray which DOES carry max_value=10 — the
    clip affects a handful of extreme cells; per-component |cos| stays
    > 0.999.
    """
    from anndataoom._chunked_ops import chunked_pca

    a = _build_scaled(backed_h5ad)
    Xp_impl, _, _ = chunked_pca(
        a, n_comps=30, chunk_size=500, random_state=42,
        use_implicit_centering=True, materialize_threshold_gb=0.0,
    )
    Xp_lega, _, _ = chunked_pca(
        a, n_comps=30, chunk_size=500, random_state=42,
        use_implicit_centering=False, materialize_threshold_gb=0.0,
    )
    a.close()

    cos = _cosine_per_component(Xp_impl, Xp_lega)
    assert cos.min() > 0.999, (
        f"min |cos| = {cos.min():.6f}; expected > 0.999. Implicit path "
        f"diverged from legacy — algebraic identity broken or clipping "
        f"effect understated.\nPer-component cos: {cos}"
    )


# ---- path-1 vs path-3 (materialise+sklearn vs legacy chunked) ---------


def test_materialise_matches_legacy(backed_h5ad):
    """Path 1 (auto-materialise + sklearn randomized_svd) ≈ Path 3.

    Both use Halko randomised SVD on the same scaled matrix. They
    differ only in (a) where the dense matrix lives and (b) sklearn's
    internal `randomized_svd` implementation vs our hand-rolled
    chunked one. Numerical drift is float-precision noise.
    """
    from anndataoom._chunked_ops import chunked_pca

    a = _build_scaled(backed_h5ad)
    Xp_mat, _, _ = chunked_pca(
        a, n_comps=30, chunk_size=500, random_state=42,
        materialize_threshold_gb=10.0,           # 5k × 500 dense fits easily
        use_implicit_centering=False,
    )
    Xp_lega, _, _ = chunked_pca(
        a, n_comps=30, chunk_size=500, random_state=42,
        materialize_threshold_gb=0.0,            # force path 3
        use_implicit_centering=False,
    )
    a.close()

    cos = _cosine_per_component(Xp_mat, Xp_lega)
    assert cos.min() > 0.999, (
        f"min |cos| = {cos.min():.6f}; expected > 0.999. Materialised "
        f"path diverged from legacy. \nPer-component cos: {cos}"
    )


# ---- power-iter default is unchanged in v0.2.0 -------------------------


def test_default_n_power_iters_is_four(backed_h5ad):
    """v0.2.0 keeps the v0.1.x default of `n_power_iters=4`.

    Initially we considered lowering the default to 2 (cf. sklearn's
    "auto" heuristic and rapids-singlecell), but that change breaks
    randomised-SVD accuracy on datasets whose singular-value spectrum
    is not the fast-decaying scRNA-seq shape — e.g. random matrices,
    heavily batchy data, or extreme tall-skinny shapes. Conservatism
    here protects users from silent drift. Lower it manually when you
    know your dataset is well-behaved.
    """
    import inspect
    from anndataoom._chunked_ops import chunked_pca

    sig = inspect.signature(chunked_pca)
    assert sig.parameters["n_power_iters"].default == 4, (
        "n_power_iters default changed; if intentional, update the "
        "test + changelog. v0.1.x is 4; lowering breaks downstream "
        "reproducibility for randomised-SVD users."
    )
