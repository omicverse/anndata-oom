"""
Decorator for automatic AnnDataOOM → AnnData conversion.

Usage::

    from omicverse.io.anndata_oom._compat import oom_guard

    @oom_guard(
        materialize=True,          # auto-convert OOM to in-memory
        result_keys_uns=['rank_genes_groups'],   # copy these uns keys back
        result_keys_obs=['S_score', 'G2M_score', 'phase'],  # copy obs cols back
        result_keys_obsm=['X_harmony'],  # copy obsm keys back
        suggest='harmony',         # suggest an OOM-native alternative
    )
    def find_markers(adata, groupby, ...):
        ...

When the decorated function receives an AnnDataOOM:
1. Prints a warning explaining why materialisation is needed
2. Calls ``adata.to_adata()`` to get an in-memory copy
3. Runs the original function on the in-memory copy
4. Copies specified result keys back to the OOM object
5. Frees the in-memory copy
"""

from __future__ import annotations

import functools
from typing import Sequence


def _is_oom(obj) -> bool:
    return getattr(obj, "_is_oom", False)


def oom_guard(
    *,
    materialize: bool = True,
    result_keys_uns: Sequence[str] | None = None,
    result_keys_obs: Sequence[str] | None = None,
    result_keys_var: Sequence[str] | None = None,
    result_keys_obsm: Sequence[str] | None = None,
    result_keys_obsp: Sequence[str] | None = None,
    suggest: str | None = None,
):
    """Decorator that auto-converts AnnDataOOM for functions that need in-memory data.

    Parameters
    ----------
    materialize : bool
        If True, automatically convert OOM to in-memory and run.
        If False, just raise a helpful error.
    result_keys_uns : list of str
        Keys in ``adata.uns`` to copy back to the OOM object after the function runs.
        Use ``'*'`` as a single element to copy all new keys.
    result_keys_obs : list of str
        Columns in ``adata.obs`` to copy back. ``'*'`` copies all new columns.
    result_keys_obsm : list of str
        Keys in ``adata.obsm`` to copy back. ``'*'`` copies all new keys.
    result_keys_obsp : list of str
        Keys in ``adata.obsp`` to copy back. ``'*'`` copies all new keys.
    suggest : str
        Name of an OOM-native alternative to suggest in the warning message.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Find adata — first positional arg or 'adata' kwarg
            adata = kwargs.get("adata", args[0] if args else None)
            if adata is None or not _is_oom(adata):
                return func(*args, **kwargs)

            if not materialize:
                msg = (
                    f"[AnnDataOOM] {func.__name__}() requires the expression matrix in memory "
                    f"but auto-materialisation is disabled for this function."
                )
                if suggest:
                    msg += f"\n  Consider using '{suggest}' which works natively with AnnDataOOM."
                raise TypeError(msg)

            # Print warning
            fname = func.__name__
            print(
                f"[AnnDataOOM] {fname}() requires the expression matrix in memory.\n"
                f"  Auto-converting to in-memory AnnData, running {fname}(), "
                f"then copying results back."
            )
            if suggest:
                print(
                    f"  Tip: '{suggest}' works directly on AnnDataOOM without materialisation."
                )

            # Snapshot existing keys to detect new ones
            old_uns = set(adata.uns.keys())
            old_obs_cols = set(adata.obs.columns)
            old_var_cols = set(adata.var.columns)
            old_obsm = set(adata.obsm.keys())
            old_obsp = set(adata.obsp.keys())

            # Convert
            adata_mem = adata.to_adata()

            # Replace adata in args/kwargs
            if "adata" in kwargs:
                kwargs["adata"] = adata_mem
                result = func(*args, **kwargs)
            else:
                result = func(adata_mem, *args[1:], **kwargs)

            # Copy results back
            _copy_back_uns(adata, adata_mem, result_keys_uns, old_uns)
            _copy_back_obs(adata, adata_mem, result_keys_obs, old_obs_cols)
            _copy_back_var(adata, adata_mem, result_keys_var, old_var_cols)
            _copy_back_mapping(adata.obsm, adata_mem.obsm, result_keys_obsm, old_obsm)
            _copy_back_mapping(adata.obsp, adata_mem.obsp, result_keys_obsp, old_obsp)

            del adata_mem
            return result

        # Mark the wrapper so we can inspect it
        wrapper._oom_guarded = True
        return wrapper

    return decorator


def _copy_back_uns(oom, mem, keys, old_keys):
    if keys is None:
        return
    if keys == ["*"]:
        for k in mem.uns:
            if k not in old_keys:
                oom.uns[k] = mem.uns[k]
    else:
        for k in keys:
            if k in mem.uns:
                oom.uns[k] = mem.uns[k]


def _copy_back_obs(oom, mem, keys, old_cols):
    if keys is None:
        return
    if keys == ["*"]:
        for col in mem.obs.columns:
            if col not in old_cols:
                oom.obs[col] = mem.obs[col].values
    else:
        for col in keys:
            if col in mem.obs.columns:
                oom.obs[col] = mem.obs[col].values


def _copy_back_var(oom, mem, keys, old_cols):
    if keys is None:
        return
    # Materialised var may be a gene subset (e.g. after HVG/filtering); align
    # back onto the OOM object's var index, leaving non-shared genes as NA.
    import pandas as pd
    cols = ([c for c in mem.var.columns if c not in old_cols]
            if keys == ["*"] else [c for c in keys if c in mem.var.columns])
    for col in cols:
        s = mem.var[col]
        if mem.var.index.equals(oom.var.index):
            oom.var[col] = s.values
        else:
            oom.var[col] = s.reindex(oom.var.index).values


def _copy_back_mapping(oom_dict, mem_dict, keys, old_keys):
    import numpy as np
    if keys is None:
        return
    if keys == ["*"]:
        for k in mem_dict:
            if k not in old_keys:
                oom_dict[k] = np.asarray(mem_dict[k])
    else:
        for k in keys:
            if k in mem_dict:
                oom_dict[k] = np.asarray(mem_dict[k])
