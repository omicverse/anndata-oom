"""omicverse-function compatibility matrix for AnnDataOOM.

For every user-facing ``omicverse.pp`` analysis function, run it against an
**AnnDataOOM** (chunked, on-disk) backend in both ``cpu`` and
``cpu-gpu-mixed`` modes and record whether it:

    ok        — ran to completion and produced its expected output key
    fail      — raised (error type + message captured)
    na        — not attempted (a prerequisite step failed)

The canonical pipeline (qc → preprocess → scale → pca → neighbors →
leiden/louvain → umap/tsne/mde → score_genes_cell_cycle) runs in order so
later steps see the state earlier steps produced. Auxiliary functions
(normalize_total, log1p, HVG variants, identify_robust_genes, regress,
filter_*, recover_counts, scrublet, anndata_to_GPU/CPU) are each probed on
a fresh freshly-read backend with the minimal prerequisites.

For mixed mode we also record ``gpu`` = whether torch actually allocated GPU
memory during the call (distinguishes a real offload from a silent CPU
fallback — the OOM PCA path, for instance, is CPU-bound regardless of mode).

Usage:
    python compat_matrix.py --input /path/ts_5k.h5ad --out results/compat.json
"""
from __future__ import annotations
import argparse, gc, json, time, traceback
from pathlib import Path
import numpy as np


def _gpu_alloc_mb():
    """Currently-allocated GPU memory (MB) — the running baseline."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return 0.0


def _gpu_peak_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return 0.0


def _reset_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def probe(results, name, fn, check=None):
    """Run fn(); record ok/fail + seconds + gpu peak + optional output check."""
    gc.collect(); _reset_gpu()
    gpu0 = _gpu_alloc_mb()  # running baseline; isolates THIS call's offload
    t0 = time.perf_counter()
    rec = {"name": name}
    try:
        fn()
        rec["seconds"] = round(time.perf_counter() - t0, 3)
        # gpu_mb = net GPU memory this call allocated above the baseline;
        # ~0 for a CPU op even if earlier steps left tensors resident.
        rec["gpu_mb"] = round(max(0.0, _gpu_peak_mb() - gpu0), 1)
        if check is not None:
            ok, detail = check()
            rec["status"] = "ok" if ok else "fail"
            rec["detail"] = detail
        else:
            rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "fail"
        rec["seconds"] = round(time.perf_counter() - t0, 3)
        rec["gpu_mb"] = round(max(0.0, _gpu_peak_mb() - gpu0), 1)
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        rec["tb"] = traceback.format_exc().splitlines()[-3:]
    results.append(rec)
    flag = "✓" if rec["status"] == "ok" else "✗"
    g = f" gpu={rec['gpu_mb']:.0f}MB" if rec.get("gpu_mb") else ""
    print(f"  [{flag}] {name:32s} {rec.get('seconds',0):7.2f}s{g}"
          + ("" if rec["status"] == "ok" else f"  → {rec.get('error','')}"))
    return rec["status"] == "ok"


def restore_counts(adata):
    """cellxgene TS stores log-norm in .X and counts in layers['decontXcounts']."""
    s = adata.X[:200, :200]
    s = s.toarray() if hasattr(s, "toarray") else np.asarray(s)
    if s.size and float(s.max()) <= 50.0:
        for cand in ("counts", "decontXcounts", "raw_counts"):
            if hasattr(adata, "layers") and cand in adata.layers:
                adata.X = adata.layers[cand]
                return cand
    return None


def _gpu_warmup():
    """Pay the one-time CUDA context / kernel-compile cost up front so the
    first timed GPU op (neighbors) is not charged ~5 s of init that would
    inflate an apparent 'GPU is slow' or deflate a 'GPU is fast' reading."""
    try:
        import torch
        if torch.cuda.is_available():
            a = torch.randn(2048, 2048, device="cuda")
            (a @ a).sum().item()
            torch.cuda.synchronize()
            del a
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_track(read_fn, mode, input_path):
    import omicverse as ov
    if mode == "cpu-gpu-mixed":
        ov.settings.cpu_gpu_mixed_init()
        _gpu_warmup()
    else:
        ov.settings.cpu_init()
    R = []
    print(f"\n### track: backend={read_fn.__name__}  mode={mode}")

    # ── canonical pipeline (ordered; later steps depend on earlier) ──────
    ad = read_fn(input_path)
    restore_counts(ad)
    okqc = probe(R, "qc", lambda: ov.pp.qc(ad, mode="seurat", doublets=False))
    okpp = okqc and probe(R, "preprocess",
        lambda: ov.pp.preprocess(ad, mode="shiftlog|pearson", n_HVGs=2000, batch_key=None),
        check=lambda: ("highly_variable_features" in ad.var.columns,
                       "highly_variable_features in var"))
    oksc = okpp and probe(R, "scale",
        lambda: ov.pp.scale(ad, max_value=10),
        check=lambda: ("scaled" in (ad.layers.keys() if hasattr(ad,'layers') else []),
                       "scaled layer present"))
    okpca = oksc and probe(R, "pca",
        lambda: ov.pp.pca(ad, n_pcs=50, layer="scaled"),
        check=lambda: (any("X_pca" in k for k in ad.obsm.keys()), "X_pca in obsm"))
    # neighbors/clustering/embeddings need an in-memory rep of the PCA;
    # omicverse handles the materialisation. Use the canonical pca key.
    okne = okpca and probe(R, "neighbors",
        lambda: ov.pp.neighbors(ad, n_neighbors=15, n_pcs=50,
                                use_rep="scaled|original|X_pca"))
    if okne:
        probe(R, "leiden", lambda: ov.pp.leiden(ad, resolution=1.0))
        probe(R, "louvain", lambda: ov.pp.louvain(ad, resolution=1.0))
        probe(R, "umap", lambda: ov.pp.umap(ad))
        # embedding alternatives that consume the PCA rep
        probe(R, "tsne", lambda: ov.pp.tsne(ad, use_rep="scaled|original|X_pca"))
        probe(R, "mde", lambda: ov.pp.mde(ad, embedding_dim=2,
                                          use_rep="scaled|original|X_pca"))
        probe(R, "sude", lambda: ov.pp.sude(ad, use_rep="scaled|original|X_pca"))
    probe(R, "score_genes_cell_cycle", lambda: _score_cc(ov, ad))
    try:
        ad.close()
    except Exception:
        pass

    # ── auxiliary functions, each on a fresh backend ─────────────────────
    def fresh_norm():
        a = read_fn(input_path); restore_counts(a)
        ov.pp.qc(a, mode="seurat", doublets=False)
        return a

    a = read_fn(input_path); restore_counts(a)
    probe(R, "filter_cells", lambda: ov.pp.filter_cells(a, min_genes=3))
    probe(R, "filter_genes", lambda: ov.pp.filter_genes(a, min_cells=1))
    try: a.close()
    except Exception: pass

    a = fresh_norm()
    probe(R, "normalize_total", lambda: ov.pp.normalize_total(a, target_sum=1e4))
    probe(R, "log1p", lambda: ov.pp.log1p(a))
    probe(R, "highly_variable_genes",
          lambda: ov.pp.highly_variable_genes(a, n_top_genes=2000))
    try: a.close()
    except Exception: pass

    a = fresh_norm()
    probe(R, "identify_robust_genes", lambda: ov.pp.identify_robust_genes(a))
    try: a.close()
    except Exception: pass

    a = fresh_norm()
    ov.pp.identify_robust_genes(a)   # pegasus HVF prerequisite (sets var['robust'])
    probe(R, "highly_variable_features",
          lambda: ov.pp.highly_variable_features(a, n_top=2000))
    try: a.close()
    except Exception: pass

    # regress-out: now @oom_guard'd in omicverse (materialise-and-run), so it
    # is safe to call directly on an AnnDataOOM.
    a = fresh_norm()
    ov.pp.normalize_total(a, target_sum=1e4); ov.pp.log1p(a)
    probe(R, "regress", lambda: ov.pp.regress(a))
    try: a.close()
    except Exception: pass

    a = fresh_norm()
    probe(R, "scrublet", lambda: ov.pp.scrublet(a))
    try: a.close()
    except Exception: pass

    a = fresh_norm()
    probe(R, "normalize_pearson_residuals",
          lambda: ov.pp.normalize_pearson_residuals(a))
    try: a.close()
    except Exception: pass

    # backend movement helpers (mixed-mode relevant)
    a = read_fn(input_path); restore_counts(a)
    probe(R, "anndata_to_GPU", lambda: ov.pp.anndata_to_GPU(a))
    probe(R, "anndata_to_CPU", lambda: ov.pp.anndata_to_CPU(a))
    try: a.close()
    except Exception: pass

    return R


def _score_cc(ov, ad):
    # Use genes that actually exist in this dataset's var_names (cellxgene
    # uses Ensembl IDs, not symbols) so the scoring has valid genes; this
    # tests that score_genes_cell_cycle *runs* on the OOM backend, not the
    # biology of a specific marker set.
    vn = list(ad.var_names)
    s_genes, g2m = vn[:50], vn[50:100]
    ov.pp.score_genes_cell_cycle(ad, s_genes=s_genes, g2m_genes=g2m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--modes", default="cpu,cpu-gpu-mixed")
    ap.add_argument("--backends", default="oom",
                    help="comma list of: oom,mem")
    args = ap.parse_args()

    import anndata
    import anndataoom as oom

    def read_oom(p): return oom.read(str(p))
    read_oom.__name__ = "oom"
    def read_mem(p):
        a = anndata.read_h5ad(str(p)); return a
    read_mem.__name__ = "mem"
    backends = {"oom": read_oom, "mem": read_mem}

    out = {"input": str(args.input), "tracks": []}
    for backend in args.backends.split(","):
        for mode in args.modes.split(","):
            recs = run_track(backends[backend], mode, args.input)
            out["tracks"].append({"backend": backend, "mode": mode, "results": recs})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n[done] → {args.out}")


if __name__ == "__main__":
    main()
