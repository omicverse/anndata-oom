"""Deterministic cell subsamples of TS-Vasculature → ts_5k / ts_10k.

Reproduces the two small benchmark anchors with a fixed seed so the
numbers in the paper are regenerable. Reads ``data/ts_vasculature.h5ad``
and writes ``data/ts_5k.h5ad`` / ``data/ts_10k.h5ad`` (raw-counts ``.X``,
matching concat_1M.py's convention).

    python scripts/subsample.py            # both 5k and 10k, seed 0
    python scripts/subsample.py --n 5000   # just one
"""
from __future__ import annotations
import argparse
from pathlib import Path
import anndata
import numpy as np

DATA = Path("data")
SRC = DATA / "ts_vasculature.h5ad"


def make(n: int, seed: int = 0):
    out = DATA / f"ts_{n // 1000}k.h5ad"
    if out.exists():
        print(f"[skip] {out} exists")
        return
    a = anndata.read_h5ad(SRC)
    if "decontXcounts" in a.layers:          # canonical raw counts
        a.X = a.layers["decontXcounts"]
        del a.layers["decontXcounts"]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(a.n_obs, size=min(n, a.n_obs), replace=False))
    sub = a[idx].copy()
    sub.write_h5ad(out, compression="gzip")
    print(f"wrote {out}  {sub.shape}  (seed={seed})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="one size; default both")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}; download TS-Vasculature first "
                         "(see scripts/fetch_data.sh)")
    for n in ([args.n] if args.n else [5000, 10000]):
        make(n, args.seed)


if __name__ == "__main__":
    main()
