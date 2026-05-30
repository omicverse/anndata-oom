"""Regression tests for the chunked-iteration performance contract.

Two performance regressions in the same chunked-read path turned a
million-cell ``scale`` call into a multi-day job in production. We
guard them here so the same shape can't reappear silently.

1. ``BackedArray._read_rows(start, end)`` on a Rust-backed PyArrayElem
   used to scan from row 0 via the ``chunked()`` iterator on every call,
   making each subsequent slice read O(start) instead of O(end - start).
   Fixed by trying ``elem[start:end]`` first (PyArrayElem does support
   slice reads on current anndata-rs builds) and only falling back to
   the chunked scan when that raises.

2. ``TransformedBackedArray`` (the wrapper used after
   ``normalize_total`` + ``log1p`` and again under ``scale``) inherited
   ``BackedArray.chunked``, which falls back to per-chunk
   ``_read_rows(start, end)``. Even with fix (1) above, that loop pays
   Python-level slice overhead per chunk. Fixed by overriding
   ``chunked()`` to delegate to the parent's native ``chunked()``
   iterator and apply ``_transform_chunk`` on each chunk.

Together the two fixes drop the wrapped-vs-raw ratio from ~25-30× to
~1.2× on a 50 k-row CSR-h5ad. For the production 1 M-cell case the
quadratic factor dominated, so the wall-clock drop is roughly 20-50×.
"""
from __future__ import annotations

import os
import tempfile
import time

import anndata
import numpy as np
import pytest
import scipy.sparse as sp


# Synthetic dataset shaped to make any quadratic regression visible in a
# few seconds while staying small enough for CI.
N_OBS = 20_000
N_VARS = 1_500
DENSITY = 0.05
CHUNK_SIZE = 500


@pytest.fixture(scope="module")
def backed_h5ad():
    """Write a sparse h5ad once, reuse across tests in this module."""
    np.random.seed(0)
    X = sp.random(
        N_OBS, N_VARS, density=DENSITY, format="csr",
        dtype=np.float32, random_state=0,
    )
    adata = anndata.AnnData(X=X)
    path = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False).name
    adata.write(path)
    yield path
    os.remove(path)


def _time(fn, *args, **kwargs) -> float:
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - t0


def test_read_rows_slice_is_position_independent(backed_h5ad):
    """``_read_rows(start, end)`` should be O(end - start), not O(end).

    Reading the **first** vs the **last** equal-width slice of a Rust-backed
    matrix used to differ by O(n) because the last read had to scan
    from row 0. With the slice-first path enabled, both reads take
    comparable time.
    """
    import anndataoom as oom

    a = oom.read(backed_h5ad)
    X = a.X  # BackedArray, rust-backed

    width = 1_000
    t_first = _time(X._read_rows, 0, width)
    t_last = _time(X._read_rows, N_OBS - width, N_OBS)
    a.close()

    # The last slice should NOT be dramatically slower than the first.
    # Allow a 5× margin to absorb noise on busy CI runners; the
    # historical quadratic path was ~50-500× slower depending on N.
    assert t_last < t_first * 5 + 0.05, (
        f"last-slice read ({t_last:.3f}s) is too slow vs first-slice "
        f"read ({t_first:.3f}s) — the position-independent slice path "
        f"is regressed."
    )


def test_transformed_chunked_matches_raw_within_constant_factor(backed_h5ad):
    """Iterating ``TransformedBackedArray.chunked()`` should cost O(n).

    Before the override, iterating the wrapped array took O(n²) because
    each chunk paid a parent ``_read_rows(start, end)`` call that itself
    did a full scan. Even after fixing ``_read_rows``, the inherited
    iterator pays per-chunk Python overhead; the override walks the
    parent's native ``chunked()`` directly.
    """
    import anndataoom as oom
    from anndataoom._chunked_ops import TransformedBackedArray

    a = oom.read(backed_h5ad)
    X_raw = a.X
    X_wrap = TransformedBackedArray(X_raw, norm_factors=None, apply_log1p=True)

    # Warm up the rust iterator (first call sometimes pays cache-fill cost)
    list(X_raw.chunked(CHUNK_SIZE))

    t_raw = _time(lambda: [c for _, _, c in X_raw.chunked(CHUNK_SIZE)])
    t_wrap = _time(lambda: [c for _, _, c in X_wrap.chunked(CHUNK_SIZE)])
    a.close()

    # The wrapped iterator does an extra log1p per chunk but otherwise
    # uses the same I/O path. Pre-fix this ratio was ~25-50×.
    ratio = t_wrap / max(t_raw, 1e-6)
    assert ratio < 5.0, (
        f"TransformedBackedArray.chunked is {ratio:.1f}× slower than raw "
        f"(raw={t_raw:.3f}s, wrap={t_wrap:.3f}s) — the chunked() override "
        f"is regressed (or _read_rows fell back to full-scan)."
    )


def test_transformed_chunked_results_are_correct(backed_h5ad):
    """The performance fix must not change the chunk contents."""
    import anndataoom as oom
    from anndataoom._chunked_ops import TransformedBackedArray

    a = oom.read(backed_h5ad)
    X_raw = a.X
    X_wrap = TransformedBackedArray(X_raw, norm_factors=None, apply_log1p=True)

    # Two passes over the same chunk_size should yield identical chunks
    # AND match the expected log1p of the underlying raw chunks.
    expected = []
    for start, end, raw in X_raw.chunked(CHUNK_SIZE):
        if sp.issparse(raw):
            r = raw.copy()
            r.data = np.log1p(r.data)
            expected.append((start, end, r.toarray()))
        else:
            expected.append((start, end, np.log1p(raw)))

    actual = []
    for start, end, c in X_wrap.chunked(CHUNK_SIZE):
        if sp.issparse(c):
            c = c.toarray()
        actual.append((start, end, c))

    assert len(actual) == len(expected)
    for (s_e, e_e, x_e), (s_a, e_a, x_a) in zip(expected, actual):
        assert (s_a, e_a) == (s_e, e_e)
        np.testing.assert_allclose(x_a, x_e, rtol=1e-5, atol=1e-7)

    a.close()
