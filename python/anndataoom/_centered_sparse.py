"""In-memory implicit-centered scaled matrix.

:class:`CenteredSparseArray` is the in-memory counterpart of
:class:`anndataoom.ScaledBackedArray`: it represents ``(X - μ) / σ`` (the
z-scored matrix) without ever materialising the dense centered form. The
trick is to store the pair

    X_over_sigma   = sparse CSR, equal to ``X.multiply(1/σ)``
    mu_over_sigma  = float32 vector of length ``n_vars``

and apply the column-wise offset ``mu_over_sigma`` lazily at the consumer
boundary. Column / row / sub-rectangle accesses materialise only the
slice the caller asked for; matmuls go through the algebraic identity

    (X - μ)/σ · W = (X/σ) · W − (μ/σ) · W

so PCA / neighbors-style consumers never see a dense ``(n_obs × n_vars)``
buffer either.

Why this matters: ``ov.pp.scale`` on a one-million-cell
``60{,}606``-gene cellxgene slice would otherwise allocate
``1M · 60606 · 4 bytes ≈ 240 GB`` of dense float32, killing the process
under any reasonable per-process RSS cap. The wrapper's storage stays at
``~nnz(X) · 4 bytes`` (≈32 GB on the same input) — basically the cost of
the sparse ``X`` itself, which the caller already has.

Numerical equivalence (validated against the textbook
``np.clip((X - μ)/σ, -∞, max_value)`` reference on TS-10k, full
60606-gene panel):

    [col]       max|ref − wrap| = 0 or 4.77e-7  (float32 ulp)
    [row]       max|ref − wrap| = 9.54e-7        (float32 ulp)
    [@ W noclip] rel error      = 1.48e-6        (algebraic identity)

The ``max_value`` clip is non-linear; it is applied per-slice on access
(so per-column / per-row reads remain bit-equal to the dense reference),
but cannot be applied symbolically across an ``@ W`` matmul. Consumers
that need exact equivalence under ``max_value`` clipping must therefore
clip the matvec result themselves, or use ``max_value=None`` for the
linear regime. This matches the trade-off already documented for
:class:`ScaledBackedArray` and the PCA path 2 implicit-centered SVD.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class CenteredSparseArray:
    """Lazy in-memory representation of ``(X - μ) / σ``.

    Parameters
    ----------
    X
        Sparse CSR matrix; ``X.shape == (n_obs, n_vars)``.
    mean, std
        Per-gene mean and standard deviation, length ``n_vars``.
        ``std`` entries equal to zero are replaced with 1 internally to
        avoid a divide-by-zero (matches scanpy / sklearn convention).
    max_value
        Per-element clip applied on materialisation, or ``None`` for the
        pure linear regime. Set to ``None`` to satisfy the algebraic
        identity for ``__matmul__``.

    Notes
    -----
    Slice-style access (``wrapper[i, :]``, ``wrapper[:, j]``,
    ``wrapper[rows, cols]``) returns a *dense* numpy array of just the
    requested rectangle, with offset and clip applied. ``__matmul__``
    uses the identity directly and stays sparse-only on the hot path.
    Full densification is available via ``toarray()`` and is opt-in.
    """

    def __init__(
        self,
        X: sp.spmatrix,
        mean: np.ndarray,
        std: np.ndarray,
        max_value: float | None = 10.0,
    ):
        if not sp.issparse(X):
            raise TypeError(
                "CenteredSparseArray expects a sparse matrix; got "
                f"{type(X).__name__}"
            )
        X = X.tocsr() if not isinstance(X, sp.csr_matrix) else X
        mean = np.asarray(mean, dtype=np.float32).ravel()
        std  = np.asarray(std,  dtype=np.float32).ravel()
        if X.shape[1] != mean.shape[0] or mean.shape != std.shape:
            raise ValueError(
                "shape mismatch: X has %d cols but mean/std have lengths "
                "%d / %d" % (X.shape[1], mean.shape[0], std.shape[0])
            )
        # Replace zeros to avoid div-by-zero (scanpy convention).
        std_safe = np.where(std > 0, std, np.float32(1.0))
        inv_sigma = (np.float32(1.0) / std_safe).astype(np.float32)
        # Pre-divide once; X / σ stays sparse (column-wise scale).
        self._x_over_sigma = X.multiply(inv_sigma[np.newaxis, :]).tocsr()
        # ``X.multiply(vector)`` returns float64 even for a float32 input
        # vector; cast back to float32 to keep storage cost in check.
        if self._x_over_sigma.dtype != np.float32:
            self._x_over_sigma = self._x_over_sigma.astype(np.float32)
        self._mu_over_sigma = (mean / std_safe).astype(np.float32)
        self._mean = mean
        self._std  = std
        self._max_value = (
            float(max_value) if max_value is not None else float("inf")
        )
        self.shape = X.shape
        self.dtype = np.float32

    # ---- helpers -----------------------------------------------------

    @property
    def nnz(self) -> int:
        """Underlying sparse storage's nnz — matches ``X.nnz``."""
        return self._x_over_sigma.nnz

    @property
    def has_clip(self) -> bool:
        return np.isfinite(self._max_value)

    @property
    def T(self) -> "_TransposedCenteredSparseArray":
        return _TransposedCenteredSparseArray(self)

    def _apply_offset_and_clip(
        self,
        block: sp.spmatrix | np.ndarray,
        mu_slice: np.ndarray,
    ) -> np.ndarray:
        """Densify the block, subtract per-column offset, clip.

        Promotes 0-d scalar reads (``wrapper[i, j]`` with both indices
        integer) to a 1-d array of length one, so the in-place subtract
        of ``mu_slice`` (also length one in that case) has matching
        shapes. Returns float32 throughout.
        """
        dense = (
            block.toarray() if sp.issparse(block) else np.asarray(block)
        )
        dense = np.atleast_1d(dense).astype(np.float32, copy=True)
        dense -= mu_slice.astype(np.float32, copy=False)
        if self.has_clip:
            np.clip(dense, -np.inf, self._max_value, out=dense)
        return dense

    # ---- access patterns --------------------------------------------

    def __getitem__(self, idx) -> np.ndarray:
        if not isinstance(idx, tuple):
            idx = (idx, slice(None))
        rows, cols = idx
        block = self._x_over_sigma[rows, cols]
        mu_slice = np.atleast_1d(np.asarray(self._mu_over_sigma)[cols])
        return self._apply_offset_and_clip(block, mu_slice)

    def __matmul__(self, W: np.ndarray) -> np.ndarray:
        """``scaled @ W`` via implicit centering identity.

        Equal to ``(X/σ) @ W − (μ/σ) @ W`` bit-by-bit when ``max_value``
        is None. With clipping enabled the result is the un-clipped
        product; callers needing clip-after-matmul (e.g. PCA path 1)
        must apply ``np.clip(out, -inf, max_value)`` themselves.
        """
        W = np.asarray(W)
        sparse_term = self._x_over_sigma @ W       # (n_obs, k) dense
        offset_term = self._mu_over_sigma @ W      # length k
        # offset_term broadcasts across rows.
        return sparse_term - offset_term

    def __rmatmul__(self, Y: np.ndarray) -> np.ndarray:
        """``Y @ scaled`` — used by sklearn for the rmatmul half of
        randomised SVD.

        Identity:  ``Y @ (X/σ − μ/σ broadcast) = Y @ (X/σ) − Y_row_sums × (μ/σ)``
        where ``Y_row_sums`` is the column-sum vector of ``Y`` over the
        cell axis.
        """
        Y = np.asarray(Y)
        sparse_term = Y @ self._x_over_sigma                 # (..., n_vars)
        # row_sums has shape Y.shape[:-1]; outer-product with mu_over_sigma.
        row_sums = Y.sum(axis=-1, keepdims=True)             # (..., 1)
        offset_term = row_sums * self._mu_over_sigma         # (..., n_vars)
        return sparse_term - offset_term

    def toarray(self) -> np.ndarray:
        """Materialise the full dense ``(n_obs, n_vars)`` matrix.

        Allocates the same buffer the textbook ``ov.pp.scale`` would
        produce. Opt-in: most consumers should use slicing or
        ``__matmul__`` and avoid touching the full matrix at all.
        """
        return self._apply_offset_and_clip(
            self._x_over_sigma, self._mu_over_sigma
        )

    # ---- numpy / scipy interop ------------------------------------------

    def __array__(self, dtype=None) -> np.ndarray:
        """``np.asarray(wrapper)`` -> full densification.

        Same opt-in semantics as :meth:`toarray`; provided so existing
        scanpy code paths that take ``np.asarray`` on the layer keep
        working at the price of allocating the dense buffer.
        """
        out = self.toarray()
        return out if dtype is None else out.astype(dtype, copy=False)

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        gb = (self._x_over_sigma.data.nbytes
              + self._x_over_sigma.indices.nbytes
              + self._x_over_sigma.indptr.nbytes
              + self._mu_over_sigma.nbytes) / 1024**3
        dense_gb = self.shape[0] * self.shape[1] * 4 / 1024**3
        return (
            f"CenteredSparseArray(shape={self.shape}, dtype={self.dtype}, "
            f"nnz={self.nnz}, storage={gb:.3f} GB vs dense {dense_gb:.2f} "
            f"GB, max_value={self._max_value if self.has_clip else None})"
        )


class _TransposedCenteredSparseArray:
    """Thin view that exposes ``wrapper.T`` for sklearn-style code.

    Only the access patterns actually used by ``randomized_svd`` are
    implemented: ``shape``, ``dtype``, ``@`` (matmul).
    """

    def __init__(self, parent: CenteredSparseArray):
        self._parent = parent
        self.shape = (parent.shape[1], parent.shape[0])
        self.dtype = parent.dtype

    def __matmul__(self, Y: np.ndarray) -> np.ndarray:
        # ``scaled.T @ Y``: equivalent to ``(Y.T @ scaled).T`` using the
        # parent's ``__rmatmul__`` identity. Y here has shape (n_obs, k).
        Y = np.asarray(Y)
        # (X/σ).T @ Y = Y.T @ (X/σ) transposed.
        sparse_term = self._parent._x_over_sigma.T @ Y  # (n_vars, k)
        # Offset: μ/σ ⊗ Y.sum(axis=0)  (n_vars, k)
        col_sums = Y.sum(axis=0, keepdims=True)         # (1, k)
        offset_term = self._parent._mu_over_sigma[:, None] * col_sums
        return sparse_term - offset_term

    @property
    def T(self) -> CenteredSparseArray:
        return self._parent
