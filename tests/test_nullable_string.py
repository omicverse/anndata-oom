"""h5ad files written by anndata >= 0.13 with pandas >= 3 store string
indices and string columns as ``nullable-string-array`` groups (``values`` +
``mask``) instead of plain string datasets.  ``anndataoom.read`` must open
them (it used to panic with "not a dataset")."""
from __future__ import annotations

import h5py
import numpy as np
import pytest
import scipy.sparse as sp


def _as_nullable_string_array(group, name, values=None, mask=None):
    """Store ``group[name]`` the way anndata 0.13 + pandas 3 writes strings
    (``values`` default to the existing string dataset)."""
    if values is None:
        if isinstance(group[name], h5py.Group):  # anndata >= 0.13 + pandas >= 3 already
            return
        values = group[name][...]
    del group[name]
    g = group.create_group(name)
    g.attrs["encoding-type"] = "nullable-string-array"
    g.attrs["encoding-version"] = "0.1.0"
    g.attrs["na-value"] = "NaN"
    v = g.create_dataset("values", data=values, dtype=h5py.string_dtype())
    v.attrs["encoding-type"] = "string-array"
    v.attrs["encoding-version"] = "0.2.0"
    m = g.create_dataset("mask", data=np.zeros(len(values), bool) if mask is None else mask)
    m.attrs["encoding-type"] = "array"
    m.attrs["encoding-version"] = "0.2.0"


@pytest.fixture
def pandas3_h5ad(tmp_path):
    import anndata as ad
    rng = np.random.default_rng(0)
    a = ad.AnnData(sp.random(60, 25, density=0.2, format="csr", dtype=np.float32, random_state=rng))
    a.obs_names = [f"cell{i}" for i in range(60)]
    a.var_names = [f"gene{j}" for j in range(25)]
    a.obs["donor"] = np.array([f"d{i % 3}" for i in range(60)], dtype=object)
    a.obs["score"] = rng.normal(size=60)
    a.obs["grp"] = np.where(np.arange(60) % 2, "a", "b")
    a.obs["grp"] = a.obs["grp"].astype("category")
    path = tmp_path / "pandas3.h5ad"
    a.write_h5ad(path)
    with h5py.File(path, "r+") as f:
        for key in ("obs", "var"):
            _as_nullable_string_array(f[key], f[key].attrs["_index"])
        mask = np.zeros(60, bool)
        mask[[3, 7]] = True  # missing donors
        # (anndata writes string columns as categoricals; store this one as a
        # plain nullable string column)
        _as_nullable_string_array(f["obs"], "donor", values=list(a.obs["donor"]), mask=mask)
    return path, a, mask


def test_read_nullable_string_index_and_columns(pandas3_h5ad):
    import anndataoom as oom
    path, ref, mask = pandas3_h5ad
    a = oom.read(str(path))
    try:
        assert a.shape == ref.shape
        assert list(a.obs_names) == list(ref.obs_names)
        assert list(a.var_names) == list(ref.var_names)
        donor = a.obs["donor"]
        assert donor.isna().to_numpy().tolist() == mask.tolist()
        assert [d for d, m in zip(donor, mask) if not m] == \
            [d for d, m in zip(ref.obs["donor"], mask) if not m]
        np.testing.assert_allclose(a.obs["score"].to_numpy(), ref.obs["score"].to_numpy())
        assert list(a.obs["grp"].astype(str)) == list(ref.obs["grp"].astype(str))
        sub = a[5:20]
        assert list(sub.obs_names) == list(ref.obs_names[5:20])
        X = np.vstack([c.toarray() if sp.issparse(c) else np.asarray(c)
                       for _, _, c in a.X.chunked(25)])
        np.testing.assert_array_equal(X, ref.X.toarray())
    finally:
        a.close()
