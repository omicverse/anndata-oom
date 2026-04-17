"""
anndataoom — Out-of-Memory AnnData powered by Rust (anndata-rs).

Drop-in replacement for ``anndata.AnnData`` that keeps the expression matrix
on disk (HDF5, via anndata-rs Rust bindings). Entire preprocessing pipelines
(normalize, log1p, scale, PCA, …) run as **lazy transforms** or **chunked
operations** — the full matrix is never loaded into memory.

Quick start
-----------
    >>> import anndataoom as oom
    >>> adata = oom.read("data.h5ad")
    >>> print(adata)
    AnnDataOOM [Rust · out-of-memory · backed]
    ...

Integration with omicverse
--------------------------
    >>> import omicverse as ov
    >>> adata = ov.read("data.h5ad", backend="rust")   # auto-detects anndataoom
    >>> ov.pp.qc(adata, ...)           # chunked QC
    >>> ov.pp.preprocess(adata, ...)   # lazy normalize + log1p + chunked HVG
"""

from ._core import AnnDataOOM, _FrozenRaw
from ._backed_array import BackedArray, DEFAULT_CHUNK_SIZE
from ._backed_layers import BackedLayers
from ._compat import oom_guard
from ._chunked_ops import (
    TransformedBackedArray,
    ScaledBackedArray,
    chunked_qc_metrics,
    chunked_gene_group_pct,
    chunked_normalize_total,
    chunked_log1p,
    chunked_mean_var,
    chunked_identify_robust_genes,
    chunked_scale,
    chunked_pca,
    chunked_pearson_residual_variance,
    chunked_highly_variable_genes_pearson,
    materialise_for_pca,
)

# The compiled Rust extension (bundled with the wheel)
from . import _backend
from ._backend import AnnData as _RsAnnData
from ._backend import AnnDataSet, Compression, concat


def is_oom(adata) -> bool:
    """Check if an object is an AnnDataOOM instance."""
    return isinstance(adata, AnnDataOOM) or getattr(adata, "_is_oom", False)


def read(path, backed: str = "r", **kwargs) -> AnnDataOOM:
    """Read an .h5ad file and wrap it in :class:`AnnDataOOM`.

    Parameters
    ----------
    path : str or Path
        Path to .h5ad file.
    backed : {'r', 'r+'}
        File open mode. Default 'r' (read-only — safe, doesn't modify source).
    **kwargs
        Forwarded to the Rust backend's read function.

    Returns
    -------
    AnnDataOOM
        Lazy out-of-memory AnnData object. Call ``adata.close()`` when done.
    """
    rs_adata = _backend.read(str(path), backed=backed, **kwargs)
    return AnnDataOOM(rs_adata)


__version__ = "0.1.2"

__all__ = [
    "__version__",
    "read",
    "AnnDataOOM",
    "AnnDataSet",
    "BackedArray",
    "BackedLayers",
    "Compression",
    "TransformedBackedArray",
    "ScaledBackedArray",
    "concat",
    "is_oom",
    "oom_guard",
    "chunked_qc_metrics",
    "chunked_gene_group_pct",
    "chunked_normalize_total",
    "chunked_log1p",
    "chunked_mean_var",
    "chunked_identify_robust_genes",
    "chunked_scale",
    "chunked_pca",
    "chunked_pearson_residual_variance",
    "chunked_highly_variable_genes_pearson",
    "materialise_for_pca",
]
