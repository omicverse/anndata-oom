"""End-to-end pipeline benchmark — backend-isolation comparison.

Compares two storage backends through the SAME omicverse pipeline:

    --config ov-anndata   omicverse on a regular in-memory AnnData
    --config ov-oom       omicverse on an AnnDataOOM (chunked, on-disk)

Both configurations call the identical `ov.pp.qc / preprocess /
scale / pca` functions — `omicverse.pp` auto-detects AnnDataOOM via
`is_oom(adata)` and dispatches to the chunked path. The only varying
axis is the storage backend, so the wall-clock and peak-RSS deltas
isolate the backend cost.

Output JSON at results/<config>__<size>.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psutil


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def _gpu_mem_mb() -> float:
    """Peak GPU memory (MB) allocated by torch so far, or 0 if no CUDA.

    In cpu-gpu-mixed mode omicverse offloads PCA (and, where applicable,
    neighbors/UMAP) to the GPU via torch. Host RSS no longer reflects the
    true peak working set, so we also track ``torch.cuda.max_memory_allocated``
    to tell the GPU-resident vs host-resident story apart."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return 0.0


def _reset_gpu_peak() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _init_mode(mode: str) -> str:
    """Set the omicverse execution mode and report the active device.

    ``cpu``            — pure-CPU path (the v0.1.7 baseline).
    ``cpu-gpu-mixed``  — ``ov.settings.cpu_gpu_mixed_init()``; PCA / neighbors /
                          UMAP / clustering route to torch-GPU, preprocessing
                          stays on CPU. Returns a short device tag for the JSON."""
    import omicverse as ov
    if mode == "cpu-gpu-mixed":
        ov.settings.cpu_gpu_mixed_init()
    else:
        ov.settings.cpu_init()
    dev = "cpu"
    try:
        import torch
        if mode == "cpu-gpu-mixed" and torch.cuda.is_available():
            dev = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return dev


@contextmanager
def stage(name: str, log: dict):
    """Capture wall time + RSS delta around a block."""
    gc.collect()
    _reset_gpu_peak()
    rss0 = _rss_mb()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        gc.collect()
        rss1 = _rss_mb()
        gpu_peak = _gpu_mem_mb()
        log[name] = {
            "seconds": round(elapsed, 4),
            "rss_mb_before": round(rss0, 1),
            "rss_mb_after":  round(rss1, 1),
            "rss_mb_delta":  round(rss1 - rss0, 1),
            "gpu_peak_mb":   round(gpu_peak, 1),
        }
        gpu_tag = f"  gpu_peak={gpu_peak:6.0f}MB" if gpu_peak else ""
        print(f"  {name:14s} {elapsed:7.2f}s  rss={rss0:6.0f}→{rss1:6.0f} MB "
              f"(Δ {rss1 - rss0:+6.0f}){gpu_tag}")


def _save_pca(adata, npy_path: Path):
    """Save adata.obsm['X_pca'] (or omicverse's canonical key) for
    cross-backend accuracy comparison. Writes nothing if no key matches."""
    import numpy as np
    candidates = [
        "scaled|original|X_pca",  # omicverse convention
        "X_pca",                  # plain anndata / scanpy
        "X_pca_harmony",
    ]
    for k in candidates:
        if k in adata.obsm:
            arr = np.asarray(adata.obsm[k])
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, arr)
            return k
    return None


_COUNTS_LAYER_CANDIDATES = ("counts", "decontXcounts", "raw_counts", "X_raw")


def _restore_raw_counts_if_normalised(adata) -> str | None:
    """Guard against running preprocess on already-log-normalised input.

    Many public h5ads (cellxgene Tabula Sapiens et al.) store the
    log-normalised matrix in `.X` and the original counts in a layer —
    cellxgene-TS uses `layers['decontXcounts']` (DecontX-decontaminated
    counts). Running `ov.pp.normalize_total` + `log1p` again on that
    `.X` would double-normalise and silently corrupt every downstream
    statistic.

    Heuristic: if `.X` looks log-normalised (max <= 50, typical log1p
    range for CP10k or CP1M data) AND a counts-like layer is present
    with integer-ish values, swap `.X` back to the raw counts. No-op
    otherwise.

    Returns the layer name that was promoted, or None when no swap was
    needed.
    """
    # Estimate the max from a random small sample to keep this O(1) for
    # AnnDataOOM. The threshold tolerates Pearson-residual-style inputs
    # too (which can go higher than log1p but stay <50 in practice).
    sample = adata.X[:200, :200]
    if hasattr(sample, "toarray"):
        sample = sample.toarray()
    x_max = float(np.asarray(sample).max() if sample.size else 0.0)
    if x_max > 50.0:
        # Looks like raw counts already (Smart-seq3 / total-seq protein
        # can push this much higher); nothing to do.
        return None

    for cand in _COUNTS_LAYER_CANDIDATES:
        if cand in adata.layers:
            layer_sample = adata.layers[cand][:200, :200]
            if hasattr(layer_sample, "toarray"):
                layer_sample = layer_sample.toarray()
            arr = np.asarray(layer_sample)
            if arr.size and arr.max() > 50.0 and float(arr.min()) >= 0.0:
                adata.X = adata.layers[cand]
                return cand
    return None


def _run_omicverse(adata, pca_npy: Path | None = None,
                    use_implicit_centering: bool = False,
                    mode: str = "cpu") -> dict:
    """Common omicverse pipeline body. Backend type drives the path.

    ``mode`` selects the omicverse execution backend (``cpu`` or
    ``cpu-gpu-mixed``); the same logical stages run either way, so the
    per-stage RSS / wall-clock / GPU-mem deltas isolate the accelerator cost."""
    import omicverse as ov

    _init_mode(mode)
    log = {}
    restored = _restore_raw_counts_if_normalised(adata)
    if restored is not None:
        print(f"  [info] restored adata.X from layers[{restored!r}] — "
              f"avoiding double-normalisation")
    with stage("qc", log):
        # Skip doublet detection: a fixed-cost preprocessing step that is
        # orthogonal to the storage-backend axis we want to isolate, and
        # scdblfinder fails on tiny / synthetic batches.
        ov.pp.qc(adata, mode="seurat", doublets=False)
    with stage("preprocess", log):
        # preprocess does normalize + log1p + HVG in one call; for the OOM
        # path the wrap composes lazily (no read happens until next stage).
        ov.pp.preprocess(adata, mode="shiftlog|pearson",
                         n_HVGs=2000, batch_key=None)
    with stage("scale", log):
        # When `use_implicit_centering=True`, `ov.pp.scale` writes a
        # lazy CenteredSparseArray to `adata.uns['_scaled_implicit']`
        # instead of densifying `adata.layers['scaled']`. Lets the
        # in-memory backend fit million-cell × 60k-gene inputs under a
        # typical 256 GB per-process RSS cap (anndataoom v0.1.7, omicverse#804).
        ov.pp.scale(adata, max_value=10,
                    use_implicit_centering=use_implicit_centering)
    with stage("pca", log):
        ov.pp.pca(adata, n_pcs=50, layer="scaled")
    if pca_npy is not None:
        which = _save_pca(adata, pca_npy)
        log["_pca_key"] = which
    return log


def run_ov_anndata(input_h5ad: Path, pca_npy: Path | None,
                   mode: str = "cpu") -> dict:
    """omicverse pipeline on a regular in-memory AnnData (dense scale)."""
    import anndata

    log = {}
    with stage("load", log):
        adata = anndata.read_h5ad(str(input_h5ad))
    log.update(_run_omicverse(adata, pca_npy=pca_npy, mode=mode))
    return log


def run_ov_anndata_implicit(input_h5ad: Path, pca_npy: Path | None) -> dict:
    """omicverse pipeline on in-memory AnnData with implicit-centered scale.

    Same as `run_ov_anndata` but ``ov.pp.scale(use_implicit_centering=True)``
    so the dense materialisation is skipped. New in v0.1.7; unblocks
    million-cell in-memory pipelines under typical RSS caps.
    """
    import anndata

    log = {}
    with stage("load", log):
        adata = anndata.read_h5ad(str(input_h5ad))
    log.update(_run_omicverse(adata, pca_npy=pca_npy,
                              use_implicit_centering=True))
    return log


def run_ov_oom(input_h5ad: Path, pca_npy: Path | None,
               mode: str = "cpu") -> dict:
    """omicverse pipeline on an AnnDataOOM (chunked, on-disk)."""
    import anndataoom as oom

    log = {}
    with stage("load", log):
        adata = oom.read(str(input_h5ad))
    try:
        log.update(_run_omicverse(adata, pca_npy=pca_npy, mode=mode))
    finally:
        adata.close()
    return log


def _restore_raw_counts_backed(adata) -> str | None:
    """Swap `.X` to raw counts on a `backed='r'` AnnData.

    Scanpy's `backed='r'` exposes `.X` as a `_CSRDataset` proxy that
    doesn't support assignment in-place. Materialising the swap requires
    `adata.to_memory()` first, but doing so here would defeat the entire
    point of the backed path. Instead, we delay the materialise until
    after QC and ALWAYS write `.X = layers[counts]` on the in-memory
    copy. Returns the layer that should be promoted later, or None if
    no swap is needed.
    """
    if "decontXcounts" in adata.layers:
        # Peek at the first chunk of `.X`; if it looks log-normalised
        # we'll need to swap. Backed `.X[:200, :200]` works (returns a
        # sparse subset) even though full operations don't.
        sample = adata.X[:200, :200]
        if hasattr(sample, "toarray"):
            sample = sample.toarray()
        if float(np.asarray(sample).max() if sample.size else 0.0) <= 50.0:
            return "decontXcounts"
    return None


def run_scanpy_backed(input_h5ad: Path, pca_npy: Path | None) -> dict:
    """Pure scanpy pipeline opening the h5ad via `backed='r'`.

    Workflow (the closest "scanpy + backed" comparison most users
    actually run):
        1. Open with backed='r' — zero-cost load
        2. Compute QC chunked on the backed AnnData
        3. `.to_memory()` -- backed='r' is read-only, so normalize /
           log1p / HVG / scale require an in-memory copy
        4. Run scanpy's pipeline functions
        5. PCA can stay chunked via IncrementalPCA on the dense scaled
           matrix, but for a fair PCA stage we use the same in-memory
           PCA as the other configs (sklearn randomized SVD).
    """
    import anndata
    import scanpy as sc
    import scanpy.experimental.pp as scep

    log = {}
    with stage("load", log):
        adata = anndata.read_h5ad(str(input_h5ad), backed='r')

    counts_layer = _restore_raw_counts_backed(adata)
    if counts_layer is not None:
        print(f"  [info] will restore .X from layers[{counts_layer!r}] "
              f"after to_memory() (backed='r' is read-only)")

    with stage("qc", log):
        # NOTE on backed='r' + scanpy: `sc.pp.calculate_qc_metrics`
        # internally calls `axis_nnz(X, axis=1)`, which is a single-
        # dispatch generic that has no handler registered for backed's
        # `_CSRDataset`. The fallback hits `np.count_nonzero(X)` on the
        # proxy object and raises `AxisError: axis 1 is out of bounds`.
        # Workaround used in practice: materialise to memory before QC.
        # This is what most "backed='r'" tutorials end up doing too --
        # it's the first op that requires it. The bench therefore
        # reports the real wall-clock and RSS of the to_memory() step
        # at the QC stage, not at load.
        adata = adata.to_memory()
        if counts_layer is not None:
            adata.X = adata.layers[counts_layer]
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False,
                                    inplace=True)

    with stage("preprocess", log):
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        scep.highly_variable_genes(
            adata, flavor="pearson_residuals",
            n_top_genes=2000, layer="counts", batch_key=None,
            subset=False, inplace=True,
        )

    with stage("scale", log):
        # Standard sc.pp.scale -- densifies the full panel (the same
        # OOM hazard as the in-memory ov-anndata config). No
        # use_implicit_centering equivalent in scanpy as of writing.
        sc.pp.scale(adata, max_value=10)

    with stage("pca", log):
        sc.pp.pca(adata, n_comps=50, mask_var="highly_variable")

    if pca_npy is not None:
        # scanpy writes obsm['X_pca'] in-place.
        if "X_pca" in adata.obsm:
            arr = np.asarray(adata.obsm["X_pca"])
            pca_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(pca_npy, arr)
            log["_pca_key"] = "X_pca"
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--config", required=True,
                    choices=["ov-anndata", "ov-anndata-implicit",
                             "scanpy-backed", "ov-oom",
                             "ov-oom-mixed", "ov-anndata-mixed"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # The `-mixed` configs run the identical pipeline under
    # ov.settings.cpu_gpu_mixed_init(); everything else stays on CPU. The
    # config name is the only axis that varies, so the mixed-vs-cpu JSON
    # deltas isolate the GPU-offload cost (PCA / neighbors / UMAP).
    mode = "cpu-gpu-mixed" if args.config.endswith("-mixed") else "cpu"

    print(f"[{args.config}] {args.input.name}  (mode={mode})")
    t0 = time.perf_counter()
    pca_npy = args.out.with_suffix(".pca.npy")
    device = "cpu"
    if args.config in ("ov-anndata", "ov-anndata-mixed"):
        log = run_ov_anndata(args.input, pca_npy=pca_npy, mode=mode)
    elif args.config == "ov-anndata-implicit":
        log = run_ov_anndata_implicit(args.input, pca_npy=pca_npy)
    elif args.config == "scanpy-backed":
        log = run_scanpy_backed(args.input, pca_npy=pca_npy)
    else:  # ov-oom / ov-oom-mixed
        log = run_ov_oom(args.input, pca_npy=pca_npy, mode=mode)
    if mode == "cpu-gpu-mixed":
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.get_device_name(0)
        except Exception:
            pass
    total = time.perf_counter() - t0

    out = {
        "config": args.config,
        "mode": mode,
        "device": device,
        "input": str(args.input),
        "input_mb": round(args.input.stat().st_size / 1024 / 1024, 1),
        "total_seconds": round(total, 4),
        "stages": log,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n[done] total={total:.2f}s  →  {args.out}")


if __name__ == "__main__":
    main()
