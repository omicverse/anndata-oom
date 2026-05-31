"""Compose a 1M-cell real h5ad from three Tabula Sapiens tissues.

Stromal (232,684) + Epithelial (228,032) + Immune (592,317) = 1,053,033 real
Tabula Sapiens cells across the same 60,606-gene panel. We use
``decontXcounts`` (DecontX-decontaminated counts) as the canonical raw
matrix per the cellxgene-TS convention, and write the concatenated
result with ``.X = raw counts`` so the bench script's
``_restore_raw_counts_if_normalised`` heuristic stays a no-op.
"""
from __future__ import annotations
import anndata, gc, os
from pathlib import Path

DATA = Path("data")
SOURCES = ["ts_stromal.h5ad", "ts_epithelial.h5ad", "ts_immune.h5ad"]
OUT = DATA / "ts_1M.h5ad"

if OUT.exists():
    print(f"[skip] {OUT} exists")
    raise SystemExit

shards = []
for f in SOURCES:
    a = anndata.read_h5ad(DATA / f)
    # Swap to raw counts BEFORE concat — anndata.concat is much faster
    # when the matrices to merge are the same backend type.
    if "decontXcounts" in a.layers:
        a.X = a.layers["decontXcounts"]
        # Drop the redundant layer to save RAM
        del a.layers["decontXcounts"]
    if "scale_data" in a.layers:
        del a.layers["scale_data"]
    print(f"  {f}: {a.shape}")
    shards.append(a)

# Concat on obs axis. Layers are dropped (we just want X).
result = anndata.concat(shards, axis=0, join="inner", merge="first")
print(f"concat: {result.shape}")

# Free RAM
del shards
gc.collect()

result.write_h5ad(OUT, compression="gzip")
print(f"wrote {OUT}  ({OUT.stat().st_size / 1024**3:.1f} GB)")
