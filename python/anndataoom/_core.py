"""
AnnDataOOM — Out-of-Memory AnnData wrapper.

Wraps a snapatac2-backed AnnData to provide the full Python anndata API
while keeping the expression matrix on disk.  Data is only read via
chunked iteration; the full matrix is never materialised before HVG
subsetting.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from ._backed_array import BackedArray, DEFAULT_CHUNK_SIZE
from ._backed_layers import BackedLayers

import logging

logger = logging.getLogger(__name__)


def _drop_unused_categories(df: pd.DataFrame) -> None:
    """Remove unused categories from every categorical column of ``df`` in
    place — mirrors anndata's subset behaviour so plot legends/palettes
    reflect only categories actually present in the subset."""
    for col in df.columns:
        series = df[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            df[col] = series.cat.remove_unused_categories()

# Register AnnDataOOM as a virtual subclass of anndata.AnnData so that
# isinstance(oom_obj, AnnData) returns True. This lets plotting functions
# and other code that type-checks against AnnData work without modification.
_ANNDATA_ABC_REGISTERED = False


def _register_anndata_abc(cls):
    global _ANNDATA_ABC_REGISTERED
    if _ANNDATA_ABC_REGISTERED:
        return
    try:
        from anndata import AnnData as _AnnData
        _AnnData.register(_AnnData)  # no-op, just test if it has register
    except (ImportError, AttributeError, TypeError):
        pass
    # Use the ABCMeta register trick
    try:
        from anndata import AnnData as _AnnData
        _AnnData.__class__.register(_AnnData.__class__, cls)
    except Exception:
        pass
    # Fallback: patch isinstance check directly
    try:
        import anndata
        _orig_AnnData = anndata.AnnData
        _orig_instancecheck = type(_orig_AnnData).__instancecheck__

        def _patched_instancecheck(klass, instance):
            if type(instance).__name__ == 'AnnDataOOM':
                return True
            return _orig_instancecheck(klass, instance)

        type(_orig_AnnData).__instancecheck__ = _patched_instancecheck
        _ANNDATA_ABC_REGISTERED = True
    except Exception:
        pass


class AnnDataOOM:
    """Out-of-memory AnnData that keeps X on disk.

    Parameters
    ----------
    snap_adata
        A ``snapatac2.AnnData`` object (backed by HDF5).
    chunk_size
        Default number of rows per I/O chunk.
    """

    # Class-level flag so external code can quickly type-check
    _is_oom = True

    def __init__(self, rs_adata, *, chunk_size: int = DEFAULT_CHUNK_SIZE):
        _register_anndata_abc(type(self))
        self._snap = rs_adata  # anndata_rs.AnnData (Rust-backed)
        # Preserve the source filename independently of _snap
        # (subset/copy operations set _snap = None but keep _origin_file)
        self._origin_file = getattr(rs_adata, "filename", None)
        self._repr_cache = {}
        self._chunk_size = chunk_size

        # Shape
        self._n_obs = int(rs_adata.n_obs)
        self._n_vars = int(rs_adata.n_vars)

        # X — lazy, wraps PyArrayElem
        self._X = BackedArray(rs_adata.X, shape=(self._n_obs, self._n_vars))

        # obs / var — eagerly convert to pandas (metadata is small)
        self._obs = self._convert_df(rs_adata.obs, rs_adata.obs_names, "obs_names")
        self._var = self._convert_df(rs_adata.var, rs_adata.var_names, "var_names")

        # obsm / varm / obsp / varp
        self._obsm = _copy_axis_arrays(getattr(rs_adata, "obsm", None))
        self._varm = _copy_axis_arrays(getattr(rs_adata, "varm", None))
        self._obsp = _copy_axis_arrays(getattr(rs_adata, "obsp", None))
        self._varp = _copy_axis_arrays(getattr(rs_adata, "varp", None))

        # uns
        self._uns: dict = {}
        try:
            src_uns = getattr(rs_adata, "uns", None)
            if src_uns is not None:
                if hasattr(src_uns, "keys"):
                    for k in src_uns.keys():
                        try:
                            self._uns[k] = src_uns[k]
                        except Exception:
                            pass
        except Exception:
            pass

        # layers — read from anndata_rs if present, plus sidecar for new layers
        backing = getattr(rs_adata, "filename", None)
        self._layers = BackedLayers(
            backing_path=backing,
            shape=(self._n_obs, self._n_vars),
        )
        # Import existing layers from the Rust backend
        rs_layers = getattr(rs_adata, "layers", None)
        if rs_layers is not None and hasattr(rs_layers, "keys"):
            for k in rs_layers.keys():
                # Store a BackedArray wrapping the Rust layer element
                try:
                    elem = rs_layers.el(k)
                    self._layers._in_memory[k] = BackedArray(
                        elem, shape=(self._n_obs, self._n_vars)
                    )
                except Exception:
                    pass

        # raw
        self._raw: AnnDataOOM | _FrozenRaw | None = None

    # ------------------------------------------------------------------
    # DataFrame conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_df(df_obj, names, index_name: str) -> pd.DataFrame:
        """Convert an anndata_rs PyDataFrameElem to a pandas DataFrame.

        String columns are automatically converted to ``category`` dtype,
        which typically reduces memory by 5-20x for large datasets with
        repeated string values (cell types, donor IDs, etc.).
        """
        pdf = None

        # anndata_rs PyDataFrameElem: use df_obj[:] → Polars DataFrame → pandas.
        # Cast string columns to Categorical in Polars BEFORE to_pandas() so
        # we never allocate full object-string columns (saves ~10× RAM on
        # million-row datasets).
        try:
            import polars as pl
            sl = df_obj[:]
            if isinstance(sl, pl.DataFrame):
                string_cols = [c for c in sl.columns if sl[c].dtype == pl.Utf8]
                if string_cols:
                    sl = sl.with_columns([pl.col(c).cast(pl.Categorical)
                                          for c in string_cols])
                pdf = sl.to_pandas()
            elif hasattr(sl, "to_pandas"):
                pdf = sl.to_pandas()
        except Exception:
            pass

        if pdf is None:
            try:
                if hasattr(df_obj, "to_pandas"):
                    pdf = df_obj.to_pandas()
            except Exception:
                pass

        if pdf is None:
            pdf = pd.DataFrame()

        # Fallback — any remaining object columns become categorical
        for col in pdf.columns:
            if pdf[col].dtype == object:
                try:
                    pdf[col] = pdf[col].astype("category")
                except Exception:
                    pass

        # Set index from names (anndata_rs obs_names/var_names are plain lists)
        try:
            idx = names
            if hasattr(idx, "to_list"):
                idx = idx.to_list()
            elif not isinstance(idx, list):
                idx = list(idx)
            str_idx = [str(x) for x in idx]
            if len(str_idx) > 0:
                if len(pdf) == 0:
                    # Empty DataFrame but names exist — create DataFrame with index only
                    pdf = pd.DataFrame(index=pd.Index(str_idx, name=index_name))
                elif len(str_idx) == len(pdf):
                    pdf.index = pd.Index(str_idx, name=index_name)
        except Exception:
            pass

        return pdf

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def X(self) -> BackedArray:
        return self._X

    @X.setter
    def X(self, value):
        if isinstance(value, BackedArray):
            self._X = value
        elif isinstance(value, np.ndarray):
            # Wrap in a BackedArray over an in-memory array
            self._X = BackedArray(value, shape=value.shape)
            self._n_obs, self._n_vars = value.shape
        elif issparse(value):
            self._X = BackedArray(value, shape=value.shape)
            self._n_obs, self._n_vars = value.shape
        else:
            # Try wrapping whatever it is
            self._X = BackedArray(value)
            self._n_obs, self._n_vars = self._X.shape

    @property
    def obs(self) -> pd.DataFrame:
        return self._obs

    @obs.setter
    def obs(self, value):
        if isinstance(value, pd.DataFrame):
            self._obs = value
        else:
            self._obs = pd.DataFrame(value)

    @property
    def var(self) -> pd.DataFrame:
        return self._var

    @var.setter
    def var(self, value):
        if isinstance(value, pd.DataFrame):
            self._var = value
        else:
            self._var = pd.DataFrame(value)

    @property
    def obs_names(self) -> pd.Index:
        return self._obs.index

    @obs_names.setter
    def obs_names(self, value):
        self._obs.index = pd.Index(value)

    @property
    def var_names(self) -> pd.Index:
        return self._var.index

    @var_names.setter
    def var_names(self, value):
        self._var.index = pd.Index(value)

    @property
    def n_obs(self) -> int:
        return self._n_obs

    @property
    def n_vars(self) -> int:
        return self._n_vars

    @property
    def shape(self) -> tuple[int, int]:
        return (self._n_obs, self._n_vars)

    @property
    def layers(self) -> BackedLayers:
        return self._layers

    @layers.setter
    def layers(self, value):
        if isinstance(value, BackedLayers):
            self._layers = value
        elif isinstance(value, dict):
            for k, v in value.items():
                self._layers[k] = v

    @property
    def obsm(self) -> dict:
        return self._obsm

    @obsm.setter
    def obsm(self, value):
        self._obsm = dict(value) if not isinstance(value, dict) else value

    @property
    def varm(self) -> dict:
        return self._varm

    @varm.setter
    def varm(self, value):
        self._varm = dict(value) if not isinstance(value, dict) else value

    @property
    def obsp(self) -> dict:
        return self._obsp

    @obsp.setter
    def obsp(self, value):
        self._obsp = dict(value) if not isinstance(value, dict) else value

    @property
    def varp(self) -> dict:
        return self._varp

    @varp.setter
    def varp(self, value):
        self._varp = dict(value) if not isinstance(value, dict) else value

    @property
    def uns(self) -> dict:
        return self._uns

    @uns.setter
    def uns(self, value):
        self._uns = dict(value) if not isinstance(value, dict) else value

    @property
    def raw(self):
        return self._raw

    @raw.setter
    def raw(self, value):
        if value is None:
            self._raw = None
        elif isinstance(value, AnnDataOOM):
            self._raw = _FrozenRaw.from_oom(value)
        elif hasattr(value, "X"):
            # Assume it's some AnnData-like object
            self._raw = _FrozenRaw.from_adata(value)
        else:
            self._raw = value

    @property
    def is_view(self) -> bool:
        return False

    @property
    def isview(self) -> bool:
        """Alias for is_view."""
        return False

    @property
    def isbacked(self) -> bool:
        return True

    @property
    def filename(self):
        if self._snap is not None:
            fn = getattr(self._snap, "filename", None)
            if fn is not None:
                return fn
        return getattr(self, "_origin_file", None)

    @property
    def T(self):
        """Transpose (materialises to in-memory AnnData)."""
        print(
            "[AnnDataOOM] Warning: .T materialises the full matrix into memory.\n"
            "  This defeats the purpose of out-of-memory processing.\n"
            "  If you need column-wise access, use adata.obs_vector(gene) instead."
        )
        return self.to_adata().T

    # ------------------------------------------------------------------
    # Name utilities
    # ------------------------------------------------------------------

    def var_names_make_unique(self, join: str = "-"):
        self._var.index = _make_index_unique(self._var.index, join)

    def obs_names_make_unique(self, join: str = "-"):
        self._obs.index = _make_index_unique(self._obs.index, join)

    def strings_to_categoricals(self):
        for df in (self._obs, self._var):
            for col in df.columns:
                if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                    cats = pd.Categorical(df[col])
                    if len(cats.categories) < len(cats):
                        df[col] = cats

    def rename_categories(self, key: str, categories):
        """Rename categories of annotation key."""
        for df in (self._obs, self._var):
            if key in df.columns and hasattr(df[key], "cat"):
                df[key] = df[key].cat.rename_categories(categories)

    # ------------------------------------------------------------------
    # Key listing methods
    # ------------------------------------------------------------------

    def obs_keys(self) -> list:
        return list(self._obs.columns)

    def var_keys(self) -> list:
        return list(self._var.columns)

    def obsm_keys(self) -> list:
        return list(self._obsm.keys())

    def varm_keys(self) -> list:
        return list(self._varm.keys())

    def uns_keys(self) -> list:
        return list(self._uns.keys())

    # ------------------------------------------------------------------
    # Chunked access (anndata compat)
    # ------------------------------------------------------------------

    def chunk_X(self, select=None, replace=False):
        """Return a chunk of X (random or selected rows)."""
        if select is None:
            select = min(1000, self._n_obs)
        if isinstance(select, int):
            rng = np.random.default_rng()
            idx = rng.choice(self._n_obs, size=select, replace=replace)
        else:
            idx = np.asarray(select)
        return self._X[idx]

    def chunked_X(self, chunk_size=1000):
        """Iterate over rows of X in chunks."""
        return self._X.chunked(chunk_size)

    # ------------------------------------------------------------------
    # to_memory
    # ------------------------------------------------------------------

    def to_memory(self):
        """Convert to a standard in-memory anndata.AnnData (alias for to_adata)."""
        n_bytes = self._n_obs * self._n_vars * 4
        print(
            f"[AnnDataOOM] Warning: to_memory() loads the full {self._n_obs}x{self._n_vars} "
            f"matrix into RAM (~{n_bytes / 1024**3:.1f} GB).\n"
            f"  For most downstream tasks this is unnecessary:\n"
            f"  - Plotting: ov.pl.umap/embedding/dotplot/violin work directly on AnnDataOOM\n"
            f"  - Gene access: adata.obs_vector(gene) reads a single column without full load\n"
            f"  - Save: adata.write(path) streams chunks to disk without full load"
        )
        return self.to_adata()

    # ------------------------------------------------------------------
    # __setitem__
    # ------------------------------------------------------------------

    def __setitem__(self, index, value):
        """Set a subset of the data matrix."""
        print(
            "[AnnDataOOM] Warning: __setitem__ (adata[...] = value) materialises the full matrix.\n"
            "  For OOM workflows, modify obs/var/obsm directly instead:\n"
            "  - adata.obs['new_col'] = values\n"
            "  - adata.obsm['X_new'] = embedding"
        )
        # Materialise X if needed, then set
        if isinstance(self._X, BackedArray):
            X = self._X[:]
            if issparse(X):
                X = X.toarray()
            X = X.astype(np.float32)
        else:
            X = self._X

        if isinstance(index, tuple) and len(index) == 2:
            obs_idx, var_idx = index
            obs_int = self._resolve_index(obs_idx, self._obs, axis=0)
            var_int = self._resolve_index(var_idx, self._var, axis=1)
            if obs_int is None and var_int is None:
                X[:] = value
            elif obs_int is None:
                X[:, var_int] = value
            elif var_int is None:
                X[obs_int] = value
            else:
                X[np.ix_(obs_int, var_int)] = value
        else:
            obs_int = self._resolve_index(index, self._obs, axis=0)
            if obs_int is not None:
                X[obs_int] = value
            else:
                X[:] = value

        self._X = BackedArray(X, shape=X.shape)

    # ------------------------------------------------------------------
    # Subsetting — the critical part
    # ------------------------------------------------------------------

    def __getitem__(self, index):
        """Subset the AnnDataOOM object.

        Supports:
        - adata[obs_mask] — boolean/int obs selection
        - adata[:, var_mask] — boolean/int var selection
        - adata[obs_mask, var_mask] — both
        - adata[:, 'gene_name'] — single gene by name
        """
        if isinstance(index, str):
            # Single gene by name
            var_idx = np.where(self._var.index == index)[0]
            return self._subset(slice(None), var_idx)

        if isinstance(index, tuple):
            if len(index) == 2:
                obs_idx, var_idx = index
            else:
                raise IndexError(f"AnnDataOOM supports 1D or 2D indexing, got {len(index)}D")
        else:
            obs_idx = index
            var_idx = slice(None)

        return self._subset(obs_idx, var_idx)

    def _subset(self, obs_idx, var_idx) -> AnnDataOOM:
        """Create a new AnnDataOOM with subset of obs and/or var."""
        obs_int = self._resolve_index(obs_idx, self._obs, axis=0)
        var_int = self._resolve_index(var_idx, self._var, axis=1)

        is_obs_all = obs_int is None
        is_var_all = var_int is None

        new = object.__new__(AnnDataOOM)
        new._snap = None
        new._chunk_size = self._chunk_size
        new._origin_file = getattr(self, "_origin_file", None)
        new._repr_cache = {}

        # obs / var — after row slicing, prune unused categories so a subset
        # doesn't carry the full category list of the parent (matches anndata
        # behaviour and keeps legends/palettes clean for downstream plots).
        #
        # When the axis is not sliced we use a shallow (Copy-on-Write) copy —
        # a deep copy of a 1M-row obs DataFrame can take seconds and the
        # subset never mutates these columns in-place on this axis.
        if is_obs_all:
            new._obs = self._obs.copy(deep=False)
        else:
            new._obs = self._obs.iloc[obs_int].reset_index(drop=True)
            new._obs.index = self._obs.index[obs_int]
            _drop_unused_categories(new._obs)

        if is_var_all:
            new._var = self._var.copy(deep=False)
        else:
            new._var = self._var.iloc[var_int].reset_index(drop=True)
            new._var.index = self._var.index[var_int]
            _drop_unused_categories(new._var)

        new._n_obs = len(new._obs)
        new._n_vars = len(new._var)

        # X — create a _SubsetBackedArray that reads only needed rows/cols
        new._X = _SubsetBackedArray(self._X, obs_int, var_int, (new._n_obs, new._n_vars))

        # obsm — preserve sparsity (scipy.sparse supports row-indexed subsetting).
        new._obsm = {}
        for k, v in self._obsm.items():
            try:
                if obs_int is not None:
                    if issparse(v):
                        new._obsm[k] = v[obs_int]
                    else:
                        arr = np.asarray(v)
                        new._obsm[k] = arr[obs_int]
                else:
                    new._obsm[k] = v
            except Exception:
                new._obsm[k] = v

        # varm — preserve sparsity.
        new._varm = {}
        for k, v in self._varm.items():
            try:
                if var_int is not None:
                    if issparse(v):
                        new._varm[k] = v[var_int]
                    else:
                        arr = np.asarray(v)
                        new._varm[k] = arr[var_int]
                else:
                    new._varm[k] = v
            except Exception:
                new._varm[k] = v

        # obsp / varp — drop on subset (size changes)
        new._obsp = {} if not is_obs_all else dict(self._obsp)
        new._varp = {} if not is_var_all else dict(self._varp)

        # uns — shallow copy
        new._uns = dict(self._uns)

        # layers — subset
        if not is_obs_all or not is_var_all:
            new._layers = self._layers.subset(obs_int, var_int)
        else:
            new._layers = self._layers

        # raw
        new._raw = self._raw

        return new

    def _resolve_index(self, idx, df: pd.DataFrame, axis: int):
        """Convert arbitrary index to integer array, or None for 'all'."""
        if isinstance(idx, slice) and idx == slice(None):
            return None

        if isinstance(idx, slice):
            n = self._n_obs if axis == 0 else self._n_vars
            return np.arange(*idx.indices(n))

        if isinstance(idx, str):
            # Single name lookup
            loc = df.index.get_loc(idx)
            if isinstance(loc, (int, np.integer)):
                return np.array([loc])
            return np.where(df.index == idx)[0]

        if isinstance(idx, pd.Series):
            idx = idx.values

        # Convert pandas Index / tuple / any iterable to ndarray
        if isinstance(idx, pd.Index):
            idx = idx.values
        elif isinstance(idx, tuple):
            idx = list(idx)

        arr = np.asarray(idx)
        if arr.dtype == bool:
            return np.where(arr)[0]
        if arr.dtype.kind in ("U", "S", "O"):
            # String array — resolve names to positions
            return np.array([df.index.get_loc(str(name)) for name in arr])
        return arr.astype(int)

    # ------------------------------------------------------------------
    # In-place subsetting (scanpy compat)
    # ------------------------------------------------------------------

    def _inplace_subset_obs(self, mask):
        """In-place observation subsetting (like anndata)."""
        if isinstance(mask, (pd.Series, np.ndarray)):
            arr = np.asarray(mask)
            if arr.dtype == bool:
                indices = np.where(arr)[0]
            else:
                indices = arr.astype(int)
        else:
            indices = np.asarray(mask)

        sub = self._subset(indices, slice(None))
        self._adopt(sub)

    def _inplace_subset_var(self, mask):
        """In-place variable subsetting (like anndata)."""
        if isinstance(mask, (pd.Series, np.ndarray)):
            arr = np.asarray(mask)
            if arr.dtype == bool:
                indices = np.where(arr)[0]
            else:
                indices = arr.astype(int)
        else:
            indices = np.asarray(mask)

        sub = self._subset(slice(None), indices)
        self._adopt(sub)

    def subset(self, obs_indices=None, var_indices=None, *, inplace=True, **kwargs):
        """anndata_rs-compatible subset. Default is in-place."""
        if obs_indices is None:
            obs_idx = slice(None)
        else:
            obs_idx = np.asarray(obs_indices)
        if var_indices is None:
            var_idx = slice(None)
        else:
            var_idx = np.asarray(var_indices)
        sub = self._subset(obs_idx, var_idx)
        if inplace:
            self._adopt(sub)
        else:
            return sub

    def _adopt(self, other: AnnDataOOM):
        """Replace own state with another AnnDataOOM's state."""
        # Preserve _origin_file even across subsets
        origin = getattr(self, "_origin_file", None) or getattr(other, "_origin_file", None)
        self._origin_file = origin
        self._snap = other._snap
        self._X = other._X
        self._obs = other._obs
        self._var = other._var
        self._n_obs = other._n_obs
        self._n_vars = other._n_vars
        self._obsm = other._obsm
        self._varm = other._varm
        self._obsp = other._obsp
        self._varp = other._varp
        self._uns = other._uns
        self._layers = other._layers
        # Keep raw unchanged
        # Invalidate repr cache (storage info may have changed)
        self._repr_cache = {}

    # ------------------------------------------------------------------
    # copy / to_df / to_adata
    # ------------------------------------------------------------------

    def copy(self, to_memory: bool = False) -> "AnnDataOOM | anndata.AnnData":
        """Copy the AnnDataOOM object.

        Parameters
        ----------
        to_memory : bool
            If True, materialise X into an in-memory ``anndata.AnnData``
            (equivalent to ``to_adata()``).  If False (default), return a
            shallow copy that shares the same backing file — obs/var/obsm
            are deep-copied, but X still reads from disk.

        Returns
        -------
        AnnDataOOM (to_memory=False) or anndata.AnnData (to_memory=True)
        """
        if to_memory:
            return self.to_adata()  # Warning printed inside to_adata()

        # Shallow copy: metadata is independent, X shares the backing file
        print(
            "[AnnDataOOM] copy(): shallow copy — obs/var/obsm are independent copies, "
            "X still reads from the same backing file (no memory cost).\n"
            "  Use adata.copy(to_memory=True) or adata.to_adata() to materialise into RAM."
        )
        new = object.__new__(AnnDataOOM)
        new._snap = None  # Don't share _snap — GC of copy must not close it
        new._chunk_size = self._chunk_size
        new._origin_file = getattr(self, "_origin_file", None)
        new._repr_cache = {}
        new._X = self._X  # Shares the BackedArray (same Rust file handle)
        new._obs = self._obs.copy()
        new._var = self._var.copy()
        new._n_obs = self._n_obs
        new._n_vars = self._n_vars
        new._obsm = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v
                      for k, v in self._obsm.items()}
        new._varm = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v
                      for k, v in self._varm.items()}
        new._obsp = dict(self._obsp)
        new._varp = dict(self._varp)
        new._uns = deepcopy(self._uns)
        new._layers = self._layers
        new._raw = self._raw
        return new

    def to_df(self, layer: str | None = None) -> pd.DataFrame:
        """Materialize X (or a layer) as a pandas DataFrame."""
        if self._n_vars > 5000:
            print(
                f"[AnnDataOOM] Warning: to_df() loads {self._n_obs}x{self._n_vars} into a DataFrame.\n"
                f"  For single-gene access, use adata.obs_vector(gene) instead.\n"
                f"  For a small subset, use adata[0:100, 0:10].to_df()."
            )
        if layer is not None:
            data = self._layers[layer][:]
        else:
            data = self._X[:]
        if issparse(data):
            data = data.toarray()
        return pd.DataFrame(
            data,
            index=self._obs.index,
            columns=self._var.index,
        )

    def to_adata(self):
        """Convert to a standard in-memory anndata.AnnData."""
        import anndata

        n_bytes = self._n_obs * self._n_vars * 4
        print(
            f"[AnnDataOOM] Warning: to_adata() loads the full {self._n_obs}x{self._n_vars} "
            f"matrix into RAM (~{n_bytes / 1024**3:.1f} GB).\n"
            f"  For most downstream tasks this is unnecessary:\n"
            f"  - Plotting: ov.pl.umap/embedding/dotplot/violin work directly on AnnDataOOM\n"
            f"  - Preprocessing: ov.pp.qc/preprocess/scale/pca all use chunked operations\n"
            f"  - Save: adata.write(path) streams chunks to disk without full load\n"
            f"  - Gene access: adata.obs_vector(gene) reads one column only"
        )

        X = self._X[:]
        if issparse(X):
            X = X.copy()

        adata = anndata.AnnData(
            X=X,
            obs=self._obs.copy(),
            var=self._var.copy(),
        )

        # Preserve sparsity in obsm / varm — ``np.asarray`` would degrade
        # scipy.sparse matrices to object-dtype ndarrays.
        for k, v in self._obsm.items():
            adata.obsm[k] = v if issparse(v) else np.asarray(v)
        for k, v in self._varm.items():
            adata.varm[k] = v if issparse(v) else np.asarray(v)
        for k, v in self._obsp.items():
            adata.obsp[k] = v
        for k, v in self._varp.items():
            adata.varp[k] = v
        adata.uns = deepcopy(self._uns)

        for k in self._layers.keys():
            adata.layers[k] = self._layers[k][:]

        if self._raw is not None:
            adata.raw = self._raw.to_adata() if hasattr(self._raw, "to_adata") else self._raw

        return adata

    # ------------------------------------------------------------------
    # obs_vector / var_vector (scanpy compat)
    # ------------------------------------------------------------------

    def obs_vector(self, key: str, *, layer: str | None = None) -> np.ndarray:
        """Return a 1D array for a given gene (var name) or obs column.

        When no in-memory cache exists, a gene lookup streams the parent
        matrix chunk-by-chunk — that's a full ~O(nnz) disk scan for a single
        column. For repeated plotting calls, call :meth:`cache_X` once to
        materialise X into an in-memory CSC matrix; subsequent column reads
        then run in a few milliseconds.
        """
        if key in self._obs.columns:
            return self._obs[key].values
        # Must be a var name
        j = self._var.index.get_loc(key)

        # Fast path: opt-in CSC cache
        cache = self._get_cache(layer)
        if cache is not None:
            col = cache[:, j]
            if issparse(col):
                col = col.toarray()
            return np.asarray(col, dtype=np.float32).ravel()

        source = self._layers[layer] if layer is not None else self._X
        return _extract_column(source, j, n_obs=self._n_obs)

    def _get_cache(self, layer):
        attr = '_X_cache_csc' if layer is None else f'_layer_cache_csc_{layer}'
        return getattr(self, attr, None)

    def cache_X(self, layer: str | None = None, force: bool = False) -> None:
        """Materialise ``adata.X`` (or a layer) into an in-memory CSC sparse
        matrix for fast repeated column access.

        Use this before interactive plotting over multiple genes. The first
        call pays a full sequential read of the backed matrix
        (dominant cost for million-cell datasets); every subsequent
        :meth:`obs_vector` / gene-coloured plot then returns in ~ms.

        Memory cost: approximately ``nnz * 12`` bytes (4 idx + 8 value per
        non-zero). A 1M × 45K matrix at 5% density ≈ ~27 GB.

        Parameters
        ----------
        layer
            Layer name, or ``None`` to cache ``.X``.
        force
            Re-materialise even if a cache already exists.
        """
        attr = '_X_cache_csc' if layer is None else f'_layer_cache_csc_{layer}'
        if hasattr(self, attr) and getattr(self, attr) is not None and not force:
            return
        source = self._X if layer is None else self._layers[layer]
        X = source[:]
        if issparse(X):
            X = X.tocsc()
        setattr(self, attr, X)

    def clear_cache(self, layer: str | None = None) -> None:
        """Drop the in-memory CSC cache populated by :meth:`cache_X`."""
        attr = '_X_cache_csc' if layer is None else f'_layer_cache_csc_{layer}'
        if hasattr(self, attr):
            setattr(self, attr, None)

    def var_vector(self, key: str, *, layer: str | None = None) -> np.ndarray:
        """Return a 1D array for a given obs name or var column."""
        if key in self._var.columns:
            return self._var[key].values
        i = self._obs.index.get_loc(key)
        if layer is not None:
            data = self._layers[layer][i, :]
        else:
            data = self._X[i, :]
        if issparse(data):
            data = data.toarray()
        return np.asarray(data).ravel()

    # ------------------------------------------------------------------
    # AnnData-compat helpers
    # ------------------------------------------------------------------

    def _sanitize(self) -> None:
        """Convert string obs/var columns to ``pd.Categorical`` in-place.

        Scanpy calls ``adata._sanitize()`` before plotting (``sc.pl.umap``,
        ``sc.pl.embedding``, …) to normalise string annotations. This
        is a minimal pandas-only implementation — no Rust round-trip.
        """
        for frame in (self._obs, self._var):
            if frame is None:
                continue
            for col in frame.columns:
                s = frame[col]
                if s.dtype == "object":
                    try:
                        frame[col] = pd.Categorical(s)
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def write(self, path: str | Path, **kwargs):
        """Write to h5ad file.

        Uses chunked writing to avoid materialising the full matrix.
        Metadata (obs, var, obsm, etc.) is written first, then X is
        streamed in row-chunks via h5py.
        """
        import h5py

        path = str(path)

        # Write metadata via a minimal anndata (X=None)
        import anndata as _ad
        adata_meta = _ad.AnnData(
            obs=self._obs.copy(),
            var=self._var.copy(),
        )
        # Preserve sparsity — ``np.asarray`` would degrade scipy.sparse
        # matrices to object-dtype ndarrays (breaking sparse obsm like
        # ATAC fragment_paired matrices on persistence + reload).
        for k, v in self._obsm.items():
            adata_meta.obsm[k] = v if issparse(v) else np.asarray(v)
        for k, v in self._varm.items():
            adata_meta.varm[k] = v if issparse(v) else np.asarray(v)
        for k, v in self._obsp.items():
            adata_meta.obsp[k] = v
        adata_meta.uns = deepcopy(self._uns)
        adata_meta.write(path, **kwargs)

        # Now stream X into the file in chunks
        with h5py.File(path, 'a') as f:
            n_obs, n_vars = self._n_obs, self._n_vars

            # Write X. The Rust backend (and anndata ≥ 0.8) expects
            # ``encoding-type`` / ``encoding-version`` attributes on the
            # X dataset — without them, ``oom.read`` panics with
            # "Cannot read shape information from type 'Scalar(f32)'".
            if 'X' in f:
                del f['X']
            # HDF5 chunks must have positive dimensions; fall back to contiguous
            # layout if either axis is zero (e.g. after an empty subset).
            use_chunks = n_obs > 0 and n_vars > 0
            ds = f.create_dataset(
                'X', shape=(n_obs, n_vars), dtype=np.float32,
                chunks=(min(1000, n_obs), n_vars) if use_chunks else None,
                compression='gzip' if use_chunks else None,
                compression_opts=1 if use_chunks else None,
            )
            ds.attrs['encoding-type'] = 'array'
            ds.attrs['encoding-version'] = '0.2.0'
            if n_obs > 0 and n_vars > 0:
                for start, end, chunk in self._X.chunked(1000):
                    if issparse(chunk):
                        chunk = chunk.toarray()
                    ds[start:end] = np.asarray(chunk, dtype=np.float32)

            # Write layers
            if 'layers' not in f:
                f.create_group('layers')
            for key in self._layers.keys():
                layer = self._layers[key]
                lpath = f'layers/{key}'
                if lpath in f:
                    del f[lpath]
                use_chunks_l = n_obs > 0 and n_vars > 0
                ds_l = f.create_dataset(
                    lpath, shape=(n_obs, n_vars), dtype=np.float32,
                    chunks=(min(1000, n_obs), n_vars) if use_chunks_l else None,
                    compression='gzip' if use_chunks_l else None,
                    compression_opts=1 if use_chunks_l else None,
                )
                ds_l.attrs['encoding-type'] = 'array'
                ds_l.attrs['encoding-version'] = '0.2.0'
                if not use_chunks_l:
                    continue
                if isinstance(layer, BackedArray):
                    for start, end, chunk in layer.chunked(1000):
                        if issparse(chunk):
                            chunk = chunk.toarray()
                        ds_l[start:end] = np.asarray(chunk, dtype=np.float32)
                else:
                    arr = np.asarray(layer)
                    if issparse(arr):
                        arr = arr.toarray()
                    ds_l[:] = arr.astype(np.float32)

    def write_h5ad(self, path: str | Path, **kwargs):
        """Alias for write."""
        self.write(path, **kwargs)

    def close(self):
        """Close backing stores."""
        self._layers.close()
        if self._snap is not None:
            try:
                self._snap.close()
            except Exception:
                pass
        # Clear repr cache (density etc. invalid after close)
        self._repr_cache = {}

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        try:
            from . import _repr
            return _repr._format_text(self)
        except Exception:
            # Fallback if the fancy formatter breaks
            return (
                f"AnnDataOOM [{self._n_obs} × {self._n_vars}]"
                f"\n    obs: {list(self._obs.columns)[:5]}"
                f"\n    var: {list(self._var.columns)[:5]}"
            )

    def _repr_html_(self) -> str:
        try:
            from . import _repr
            return _repr._format_html(self)
        except Exception:
            return f"<pre>{self.__repr__()}</pre>"

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ======================================================================
# _SubsetBackedArray — lazy column/row subset over a BackedArray
# ======================================================================


class _SubsetBackedArray(BackedArray):
    """A view over a parent BackedArray with row and/or column selection."""

    def __init__(
        self,
        parent: BackedArray,
        obs_idx: np.ndarray | None,
        var_idx: np.ndarray | None,
        shape: tuple[int, int],
    ):
        # Flatten nested _SubsetBackedArray chains to avoid O(depth) overhead.
        # If parent is also a _SubsetBackedArray, compose the indices and
        # point directly to the grandparent.
        while isinstance(parent, _SubsetBackedArray):
            # Compose obs indices
            if obs_idx is not None and parent._obs_idx is not None:
                obs_idx = parent._obs_idx[obs_idx]
            elif obs_idx is None:
                obs_idx = parent._obs_idx
            # Compose var indices
            if var_idx is not None and parent._var_idx is not None:
                var_idx = parent._var_idx[var_idx]
            elif var_idx is None:
                var_idx = parent._var_idx
            parent = parent._parent

        self._parent = parent
        self._obs_idx = obs_idx  # None means "all rows"
        self._var_idx = var_idx  # None means "all columns"
        self._shape = shape
        self._elem = None
        self._is_rs = False

    def _read_rows(self, start: int, end: int):
        end = min(end, self._shape[0])
        if start >= end:
            return np.empty((0, self._shape[1]))

        if self._obs_idx is not None:
            # Map local [start, end) to parent row indices
            parent_rows = self._obs_idx[start:end]
            data = self._parent._read_row_indices(parent_rows)
        else:
            data = self._parent._read_rows(start, end)

        if self._var_idx is not None:
            if issparse(data):
                data = data[:, self._var_idx]
            else:
                data = data[:, self._var_idx]

        return data

    def _read_row_indices(self, indices):
        if self._obs_idx is not None:
            parent_rows = self._obs_idx[indices]
        else:
            parent_rows = indices

        data = self._parent._read_row_indices(parent_rows)

        if self._var_idx is not None:
            if issparse(data):
                data = data[:, self._var_idx]
            else:
                data = data[:, self._var_idx]

        return data

    def chunked(self, chunk_size=1000):
        """Optimised chunked iteration for subset arrays.

        Instead of calling _read_rows per chunk (which triggers
        _read_row_indices on non-contiguous obs), iterate the parent's
        chunked() once and filter rows/columns on the fly.
        """
        from ._backed_array import DEFAULT_CHUNK_SIZE
        if self._obs_idx is None and self._var_idx is None:
            yield from self._parent.chunked(chunk_size)
            return

        if self._obs_idx is not None:
            obs_set = set(self._obs_idx.tolist())
            # Map parent row → local position
            obs_order = {int(v): i for i, v in enumerate(self._obs_idx)}
        else:
            obs_set = None

        # Accumulate rows from parent chunks
        buf = []
        buf_start = 0

        for p_start, p_end, p_chunk in self._parent.chunked(chunk_size):
            if self._obs_idx is not None:
                # Select rows in this parent chunk that we need
                keep = []
                for local_i in range(p_chunk.shape[0]):
                    global_i = p_start + local_i
                    if global_i in obs_set:
                        keep.append(local_i)
                if not keep:
                    continue
                if issparse(p_chunk):
                    selected = p_chunk[keep]
                else:
                    selected = p_chunk[keep]
            else:
                selected = p_chunk

            # Column filter
            if self._var_idx is not None:
                if issparse(selected):
                    selected = selected[:, self._var_idx]
                else:
                    selected = selected[:, self._var_idx]

            buf.append(selected)
            total_rows = sum(b.shape[0] for b in buf)

            if total_rows >= chunk_size:
                # Flush buffer as a chunk
                if issparse(buf[0]):
                    from scipy.sparse import vstack as sp_vstack
                    combined = sp_vstack(buf)
                else:
                    combined = np.vstack(buf)
                buf_end = buf_start + combined.shape[0]
                yield buf_start, buf_end, combined
                buf_start = buf_end
                buf = []

        # Flush remaining
        if buf:
            if issparse(buf[0]):
                from scipy.sparse import vstack as sp_vstack
                combined = sp_vstack(buf)
            else:
                combined = np.vstack(buf)
            buf_end = buf_start + combined.shape[0]
            yield buf_start, buf_end, combined


# ======================================================================
# _FrozenRaw — lightweight frozen snapshot for .raw
# ======================================================================


class _FrozenRaw:
    """Frozen snapshot of AnnData for the .raw attribute.

    Stores var metadata and a reference to the X matrix.
    After HVG subsetting, .raw preserves the pre-subset state so that
    differential expression can use all genes.
    """

    def __init__(self, X, var: pd.DataFrame, varm: dict | None = None):
        self._X = X
        self._var = var.copy()
        self._varm = varm or {}

    @classmethod
    def from_oom(cls, oom: AnnDataOOM) -> _FrozenRaw:
        return cls(oom.X, oom.var, dict(oom.varm))

    @classmethod
    def from_adata(cls, adata) -> _FrozenRaw:
        X = adata.X
        var = adata.var.copy() if hasattr(adata.var, "copy") else pd.DataFrame(adata.var)
        varm = dict(getattr(adata, "varm", {}))
        return cls(X, var, varm)

    @property
    def X(self):
        return self._X

    @property
    def var(self):
        return self._var

    @property
    def var_names(self):
        return self._var.index

    @property
    def varm(self):
        return self._varm

    @property
    def shape(self):
        return self._X.shape

    @property
    def n_vars(self):
        return self._X.shape[1]

    def obs_vector(self, key: str) -> np.ndarray:
        """Return expression vector for a gene (by var_name)."""
        j = self._var.index.get_loc(key)
        data = self._X[:, j]
        if issparse(data):
            data = data.toarray()
        return np.asarray(data).ravel()

    def var_vector(self, key: str) -> np.ndarray:
        """Return a var column or expression across all genes for one obs."""
        if key in self._var.columns:
            return self._var[key].values
        raise KeyError(f"'{key}' not found in raw.var")

    def __getitem__(self, index):
        """Support adata.raw[:, gene] slicing for compatibility."""
        if isinstance(index, tuple) and len(index) == 2:
            _, var_idx = index
            if isinstance(var_idx, str):
                j = self._var.index.get_loc(var_idx)
                var_idx = np.array([j])
            elif isinstance(var_idx, (list, np.ndarray)):
                var_idx = np.asarray(var_idx)
                if var_idx.dtype == bool:
                    var_idx = np.where(var_idx)[0]
            # Return a mini _FrozenRaw with subset columns
            if isinstance(self._X, BackedArray):
                X_sub = self._X[:, var_idx]
            else:
                X_sub = self._X[:, var_idx]
            var_sub = self._var.iloc[var_idx] if isinstance(var_idx, np.ndarray) else self._var
            return _FrozenRaw(X_sub, var_sub)
        return self

    def to_adata(self):
        """Convert to a standard anndata.AnnData for downstream DE."""
        import anndata

        X = self._X[:]
        if issparse(X):
            X = X.copy()
        adata = anndata.AnnData(X=X, var=self._var.copy())
        for k, v in self._varm.items():
            adata.varm[k] = np.asarray(v)
        return adata


# ======================================================================
# Helpers
# ======================================================================


def _make_index_unique(index: pd.Index, join: str = "-") -> pd.Index:
    """Make a pandas Index unique by appending suffixes."""
    if index.is_unique:
        return index
    from collections import Counter

    values = np.array(index)
    seen = Counter()
    result = list(values)
    dups = index.duplicated(keep="first")
    for i in np.where(dups)[0]:
        v = values[i]
        seen[v] += 1
        result[i] = f"{v}{join}{seen[v]}"
    return pd.Index(result, name=index.name)


def _copy_mapping(m) -> dict:
    """Copy a mapping-like object to a plain dict."""
    result = {}
    try:
        keys = m.keys() if hasattr(m, "keys") else list(m)
        for k in keys:
            try:
                v = m[k]
                if hasattr(v, "copy"):
                    result[k] = v.copy()
                else:
                    result[k] = np.asarray(v)
            except Exception:
                pass
    except Exception:
        pass
    return result


def _copy_axis_arrays(axis_arrays) -> dict:
    """Copy anndata_rs PyAxisArrays to a plain dict.

    PyAxisArrays has ``.keys()`` and ``.__getitem__(key)`` that returns
    either a numpy array (dense) or a scipy.sparse matrix. We must
    preserve sparsity — ``np.asarray`` would silently degrade a
    scipy.sparse obsm entry (e.g. a chromosome-wide fragment matrix) to
    an object-dtype 0-d ndarray.
    """
    if axis_arrays is None:
        return {}
    result = {}
    try:
        for k in axis_arrays.keys():
            try:
                v = axis_arrays[k]
                if isinstance(v, np.ndarray):
                    result[k] = v.copy()
                elif issparse(v):
                    result[k] = v.copy()
                else:
                    result[k] = np.asarray(v)
            except Exception:
                pass
    except Exception:
        pass
    return result


def _extract_column(source, j: int, n_obs: int, chunk_size: int = 5000) -> np.ndarray:
    """Extract column ``j`` from a BackedArray/ndarray/sparse matrix.

    For subset views (``_SubsetBackedArray``) we use direct row-indexed
    reads so an ``obs_vector('gene')`` on a 1M → 4k cell subset touches
    only those 4k parent rows instead of scanning the full matrix.

    For a plain (non-subsetted) ``BackedArray`` we stream chunks so the
    working set stays bounded to ``chunk_size × n_vars``.
    """
    if isinstance(source, _SubsetBackedArray):
        col = source[:, j]
        if issparse(col):
            col = col.toarray()
        return np.asarray(col, dtype=np.float32).ravel()

    if isinstance(source, BackedArray):
        result = np.empty(n_obs, dtype=np.float32)
        for start, end, chunk in source.chunked(chunk_size):
            if issparse(chunk):
                col = chunk[:, j].toarray().ravel()
            else:
                col = np.asarray(chunk[:, j]).ravel()
            result[start:end] = col
        return result
    # Fallback for plain arrays
    col = source[:, j]
    if issparse(col):
        col = col.toarray()
    return np.asarray(col).ravel()
