"""
Lazy matrix proxy for out-of-memory AnnData access.

Wraps an anndata_rs PyArrayElem (or any array-like) and provides
chunked iteration without materializing the full matrix into memory.

anndata_rs PyArrayElem API:
- ``elem.chunked(chunk_size)`` → iterator of ``(ndarray, start, end)``
- ``elem.shape`` → list ``[n_obs, n_vars]``
- No ``__getitem__`` (slice not implemented in Rust main branch)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Tuple

import numpy as np
from scipy.sparse import issparse, csr_matrix, vstack as sp_vstack

if TYPE_CHECKING:
    from scipy.sparse import spmatrix

DEFAULT_CHUNK_SIZE = 1000


def _is_pyarrayelem(obj) -> bool:
    """Check if obj is an anndata_rs PyArrayElem."""
    return type(obj).__name__ == "PyArrayElem"


class BackedArray:
    """Lazy proxy over an anndata_rs-backed matrix element.

    Supports:
    - Chunked row iteration via :meth:`chunked` (uses Rust-native I/O)
    - Slicing: ``arr[0:100]``, ``arr[:, [0,1,2]]``
    - Full materialization via ``arr[:]`` (use sparingly)
    - Chunked aggregations: ``sum``, ``getnnz``, ``mean``, ``var``
    """

    def __init__(self, backed_elem, shape: tuple[int, int] | None = None):
        self._elem = backed_elem
        self._is_rs = _is_pyarrayelem(backed_elem)
        if shape is not None:
            self._shape = tuple(shape)
        else:
            try:
                s = backed_elem.shape
                # anndata_rs returns list, numpy/scipy return tuple
                self._shape = tuple(s)
            except Exception:
                raise ValueError("Cannot determine shape of backed element; pass shape explicitly")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def ndim(self) -> int:
        return 2

    @property
    def dtype(self) -> np.dtype:
        try:
            return self._elem.dtype
        except Exception:
            # Read a tiny slice to infer dtype
            chunk = self._read_rows(0, min(1, self._shape[0]))
            return chunk.dtype

    @property
    def T(self):
        """Transpose (materializes — avoid on large matrices)."""
        return self[:].T

    # ------------------------------------------------------------------
    # Reading helpers
    # ------------------------------------------------------------------

    def _read_rows(self, start: int, end: int) -> np.ndarray | spmatrix:
        """Read rows [start, end) from the backing store."""
        end = min(end, self._shape[0])
        if start >= end:
            return np.empty((0, self._shape[1]))

        if self._is_rs:
            # anndata_rs PyArrayElem: no __getitem__, only chunked() iterator.
            # chunked() always starts from row 0, so we scan and collect
            # the rows that fall into [start, end).
            rows = []
            needed = end - start
            cs_size = max(min(needed, 5000), DEFAULT_CHUNK_SIZE)
            for data, cs, ce in self._elem.chunked(cs_size):
                if ce <= start:
                    continue
                if cs >= end:
                    break
                local_start = max(start - cs, 0)
                local_end = min(end - cs, data.shape[0])
                rows.append(data[local_start:local_end])
                if sum(r.shape[0] for r in rows) >= needed:
                    break
            if not rows:
                return np.empty((0, self._shape[1]))
            if len(rows) == 1:
                return rows[0]
            if issparse(rows[0]):
                return sp_vstack(rows).tocsr()
            return np.vstack(rows)

        # Standard array-like: use __getitem__
        try:
            return self._elem[start:end]
        except Exception:
            return self._elem[slice(start, end)]

    def _read_row_indices(self, indices) -> np.ndarray | spmatrix:
        """Read specific row indices from the backing store.

        Strategy for the anndata-rs path: the rust backend supports
        arbitrary-offset slice reads (``elem[a:b]``) very efficiently —
        ~2-250 ms depending on the width. For scattered index lists we
        group them into contiguous runs (allowing small inter-row gaps)
        and issue one slice read per run, then fancy-index the combined
        in-memory matrix to restore the requested order. That beats the
        full-file ``chunked()`` scan by a big margin when the subset
        covers only part of the dataset.
        """
        if self._is_rs:
            indices = np.asarray(indices)
            n = len(indices)
            if n == 0:
                return np.empty((0, self._shape[1]))

            # Sort + dedupe; we'll restore the caller's order at the end.
            unique = np.unique(indices)

            # Group into runs, merging gaps ≤ GAP_THRESHOLD. Small gaps
            # cost very little extra data per slice read, but each slice
            # call has fixed per-call overhead, so merging nearby runs
            # wins. Empirically, 16 is a decent default on h5ad CSR.
            GAP_THRESHOLD = 16
            if len(unique) == 1:
                breaks = np.array([], dtype=np.int64)
            else:
                diffs = np.diff(unique)
                breaks = np.where(diffs > GAP_THRESHOLD)[0] + 1

            run_starts = np.empty(len(breaks) + 1, dtype=np.int64)
            run_ends = np.empty(len(breaks) + 1, dtype=np.int64)
            prev = 0
            splits = list(breaks) + [len(unique)]
            for i, stop in enumerate(splits):
                run_starts[i] = int(unique[prev])
                run_ends[i] = int(unique[stop - 1]) + 1
                prev = stop

            # Read each run as a range slice
            pieces = [self._elem[int(s):int(e)] for s, e in zip(run_starts, run_ends)]

            # Combine once
            if issparse(pieces[0]):
                combined = pieces[0] if len(pieces) == 1 else sp_vstack(pieces).tocsr()
            else:
                combined = pieces[0] if len(pieces) == 1 else np.vstack(pieces)

            # Build lookup: for each global index, its local position in `combined`.
            # run k contributes (run_ends[k]-run_starts[k]) rows starting at
            # cum_lens[k] in the combined matrix.
            run_lens = run_ends - run_starts
            run_offsets = np.concatenate(([0], np.cumsum(run_lens)[:-1]))
            # Vectorised: find each index's run via searchsorted on run_starts.
            run_of = np.searchsorted(run_starts, indices, side='right') - 1
            local_positions = run_offsets[run_of] + (indices - run_starts[run_of])

            if issparse(combined):
                return combined[local_positions]
            return combined[local_positions]

        try:
            return self._elem[indices]
        except Exception:
            # Fallback: read one-by-one and stack
            chunks = []
            for idx in indices:
                chunks.append(self._elem[idx:idx + 1])
            if len(chunks) == 0:
                return np.empty((0, self._shape[1]))
            if issparse(chunks[0]):
                return sp_vstack(chunks)
            return np.vstack(chunks)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key = key
            col_key = slice(None)

        # Materialize rows
        if isinstance(row_key, slice):
            start, stop, step = row_key.indices(self._shape[0])
            if step == 1:
                data = self._read_rows(start, stop)
            else:
                indices = np.arange(start, stop, step)
                data = self._read_row_indices(indices)
        elif isinstance(row_key, (np.ndarray, list)):
            arr = np.asarray(row_key)
            if arr.dtype == bool:
                arr = np.where(arr)[0]
            data = self._read_row_indices(arr)
        elif isinstance(row_key, (int, np.integer)):
            data = self._read_rows(int(row_key), int(row_key) + 1)
        else:
            data = self._read_rows(0, self._shape[0])

        # Apply column selection
        if not _is_full_slice(col_key):
            if isinstance(col_key, (np.ndarray, list)):
                col_idx = np.asarray(col_key)
                if col_idx.dtype == bool:
                    col_idx = np.where(col_idx)[0]
            elif isinstance(col_key, slice):
                col_idx = np.arange(*col_key.indices(self._shape[1]))
            elif isinstance(col_key, (int, np.integer)):
                col_idx = np.array([int(col_key)])
            else:
                col_idx = col_key

            if issparse(data):
                data = data[:, col_idx]
            else:
                data = data[:, col_idx]

        return data

    # ------------------------------------------------------------------
    # Chunked iteration
    # ------------------------------------------------------------------

    def chunked(
        self, chunk_size: int = DEFAULT_CHUNK_SIZE
    ) -> Iterator[Tuple[int, int, np.ndarray | spmatrix]]:
        """Iterate over row chunks without loading the full matrix.

        Uses the Rust-native ``chunked()`` iterator when available,
        falling back to manual slicing for numpy/scipy arrays.

        Yields
        ------
        start : int
            Start row index (inclusive).
        end : int
            End row index (exclusive).
        chunk : ndarray or sparse matrix
            The chunk of rows ``[start, end)``.
        """
        if self._is_rs:
            # anndata_rs native chunked: yields (data, start, end)
            for data, start, end in self._elem.chunked(chunk_size):
                yield start, end, data
        else:
            n_obs = self._shape[0]
            for start in range(0, n_obs, chunk_size):
                end = min(start + chunk_size, n_obs)
                chunk = self._read_rows(start, end)
                yield start, end, chunk

    def chunked_columns(
        self, chunk_size: int = DEFAULT_CHUNK_SIZE
    ) -> Iterator[Tuple[int, int, np.ndarray]]:
        """Accumulate column-wise statistics via row chunks.

        This is more memory-efficient than transposing.
        Each chunk is a dense (chunk_rows, n_vars) array.

        Yields same as :meth:`chunked`.
        """
        for start, end, chunk in self.chunked(chunk_size):
            if issparse(chunk):
                yield start, end, chunk
            else:
                yield start, end, chunk

    # ------------------------------------------------------------------
    # Sparse-specific helpers
    # ------------------------------------------------------------------

    def sum(
        self,
        axis: int = 0,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        keepdims: bool = False,
    ) -> np.ndarray:
        """Chunked sum along axis without full materialization.

        ``keepdims`` mirrors the numpy API — when True, axis 0 returns
        shape ``(1, n_vars)`` and axis 1 returns ``(n_obs, 1)`` so the
        result broadcasts against the original matrix. Code such as
        ``X / X.sum(axis=1, keepdims=True)`` (standard TF-IDF / per-row
        normalisation) works on a ``BackedArray`` the same way it works
        on a numpy ndarray or a ``scipy.sparse`` matrix.
        """
        if axis == 0:
            # Column sums: accumulate across row chunks
            result = np.zeros(self._shape[1], dtype=np.float64)
            for _, _, chunk in self.chunked(chunk_size):
                if issparse(chunk):
                    result += np.asarray(chunk.sum(axis=0)).ravel()
                else:
                    result += chunk.sum(axis=0)
            return result.reshape(1, -1) if keepdims else result
        elif axis == 1:
            # Row sums: compute per chunk, concatenate
            result = np.empty(self._shape[0], dtype=np.float64)
            for start, end, chunk in self.chunked(chunk_size):
                if issparse(chunk):
                    result[start:end] = np.asarray(chunk.sum(axis=1)).ravel()
                else:
                    result[start:end] = chunk.sum(axis=1)
            return result.reshape(-1, 1) if keepdims else result
        else:
            raise ValueError(f"axis must be 0 or 1, got {axis}")

    def getnnz(
        self,
        axis: int = 0,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        keepdims: bool = False,
    ) -> np.ndarray:
        """Chunked non-zero count along axis."""
        if axis == 0:
            result = np.zeros(self._shape[1], dtype=np.int64)
            for _, _, chunk in self.chunked(chunk_size):
                if issparse(chunk):
                    result += np.asarray((chunk != 0).sum(axis=0)).ravel()
                else:
                    result += (chunk != 0).sum(axis=0)
            return result.reshape(1, -1) if keepdims else result
        elif axis == 1:
            result = np.empty(self._shape[0], dtype=np.int64)
            for start, end, chunk in self.chunked(chunk_size):
                if issparse(chunk):
                    result[start:end] = np.asarray((chunk != 0).sum(axis=1)).ravel()
                else:
                    result[start:end] = (chunk != 0).sum(axis=1)
            return result.reshape(-1, 1) if keepdims else result
        else:
            raise ValueError(f"axis must be 0 or 1, got {axis}")

    def mean(
        self,
        axis: int = 0,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        keepdims: bool = False,
    ) -> np.ndarray:
        """Chunked mean along axis."""
        s = self.sum(axis=axis, chunk_size=chunk_size, keepdims=keepdims)
        n = self._shape[1 - axis]  # divide by the other dimension
        return s / max(n, 1)

    def var(self, axis: int = 0, chunk_size: int = DEFAULT_CHUNK_SIZE, ddof: int = 0) -> np.ndarray:
        """Chunked variance along axis (two-pass for numerical stability)."""
        mu = self.mean(axis=axis, chunk_size=chunk_size)
        n = self._shape[1 - axis]

        if axis == 0:
            # Column variance
            sq_diff = np.zeros(self._shape[1], dtype=np.float64)
            for _, _, chunk in self.chunked(chunk_size):
                if issparse(chunk):
                    dense = chunk.toarray()
                else:
                    dense = chunk
                sq_diff += ((dense - mu[np.newaxis, :]) ** 2).sum(axis=0)
            return sq_diff / max(n - ddof, 1)
        elif axis == 1:
            sq_diff = np.empty(self._shape[0], dtype=np.float64)
            for start, end, chunk in self.chunked(chunk_size):
                if issparse(chunk):
                    dense = chunk.toarray()
                else:
                    dense = chunk
                sq_diff[start:end] = ((dense - mu[start:end, np.newaxis]) ** 2).sum(axis=1)
            return sq_diff / max(n - ddof, 1)
        else:
            raise ValueError(f"axis must be 0 or 1, got {axis}")

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"BackedArray(shape={self._shape})"

    def __len__(self) -> int:
        return self._shape[0]


def _is_full_slice(key) -> bool:
    """Check if a key selects all columns."""
    if isinstance(key, slice):
        return key == slice(None) or key == slice(0, None)
    return False
