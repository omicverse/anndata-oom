"""OOM chunked-PCA acceleration strategies — single-strategy benchmark.

Goal: get ov-oom-cpu PCA toward in-memory-cpu PCA speed. The current OOM
PCA densifies the FULL-width (36k-gene) scaled chunk per pass; the lever is
to subset the ~2000 HVG columns *before* densifying, and to avoid re-reading
from disk more than necessary.

Run ONE strategy and emit a JSON result line `RESULT {...}`:

    python oom_pca_strategies.py --input /tmp/ts_50k.h5ad --strategy hvg_materialize

Strategies
----------
baseline        : anndataoom.chunked_pca(device='cpu') as-is (full-width densify)
hvg_materialize : read scaled chunks, densify only HVG cols, build n_obs x nHVG
                  dense once, then sklearn covariance_eigh
hvg_from_raw    : subset HVG from the RAW sparse chunk FIRST, then normalize+
                  log1p+scale on 2000-wide, materialize, covariance_eigh
streaming_cov   : one streaming pass accumulating the 2000x2000 Gram matrix
                  (covariance_eigh) + one pass to project — never materialises
                  the n_obs x nHVG matrix
cache_then_pca  : materialise the HVG scaled matrix ONCE (the bounded
                  n_obs x nHVG cache), then time ONLY the PCA on the cache
                  (models caching the scaled HVG block at scale-time so the
                  PCA stage matches in-memory)

Correctness reference: exact `covariance_eigh` PCA computed on the HVG scaled
matrix built by the verified reconstruction (so all exact-method strategies
match it; the randomized `baseline` is reported but compared loosely).
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
from scipy.sparse import diags, issparse


def _rss_mb():
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def prep(path):
    import omicverse as ov, anndataoom as oom
    ov.settings.cpu_init()
    a = oom.read(path)
    s = a.X[:200, :200]
    s = s.toarray() if hasattr(s, "toarray") else np.asarray(s)
    if float(s.max()) <= 50 and "decontXcounts" in a.layers:
        a.X = a.layers["decontXcounts"]
    ov.pp.qc(a, mode="seurat", doublets=False)
    ov.pp.preprocess(a, mode="shiftlog|pearson", n_HVGs=2000, batch_key=None)
    ov.pp.scale(a, max_value=10)
    return a


def make_hvg_chunker(X, hvg, *, subset_first):
    """Yield (s, e, dense float64 2000-wide scaled+clipped chunk).

    subset_first=False : reproduce ScaledBackedArray (normalize+log1p full
                         width sparse, then HVG slice, then scale).
    subset_first=True  : HVG-slice the RAW sparse chunk first, then do the
                         (mathematically identical) per-row normalize+log1p+
                         scale on the 2000-wide block — cheaper normalize.
    """
    parent, nf, lg = X._parent, X._norm_factors, X._apply_log1p
    mu, sd, mv = X._scale_mean[hvg], X._scale_std[hvg], X._max_value

    def gen(cs):
        for s, e, raw in parent.chunked(cs):
            if subset_first:
                raw = raw[:, hvg] if issparse(raw) else np.asarray(raw)[:, hvg]
            if nf is not None:
                f = nf[s:e].astype(np.float64); f[f == 0] = 1.0
                if issparse(raw):
                    raw = diags(1.0 / f.astype(np.float32)) @ raw.astype(np.float32)
                else:
                    raw = np.asarray(raw, np.float32) / f[:, None].astype(np.float32)
            if lg:
                if issparse(raw):
                    raw = raw.copy(); raw.data = np.log1p(raw.data)
                else:
                    raw = np.log1p(raw)
            if not subset_first:
                raw = raw[:, hvg]
            d = (raw.toarray() if issparse(raw) else np.asarray(raw)).astype(np.float64)
            d = (d - mu) / sd
            if mv is not None:
                np.clip(d, -mv, mv, out=d)
            yield s, e, d
    return gen


def exact_reference(chunker, n_obs, d, k, cs):
    """Exact covariance_eigh PCA from the verified chunker (ground truth)."""
    C = np.zeros((d, d)); colsum = np.zeros(d)
    for s, e, dc in chunker(cs):
        C += dc.T @ dc; colsum += dc.sum(0)
    mean = colsum / n_obs
    C -= n_obs * np.outer(mean, mean)
    w, V = np.linalg.eigh(C)
    o = np.argsort(w)[::-1][:k]; Vk = V[:, o]
    Xpca = np.empty((n_obs, k), np.float32)
    for s, e, dc in chunker(cs):
        Xpca[s:e] = ((dc - mean) @ Vk).astype(np.float32)
    return Xpca


def cosmin(A, B):
    n = min(A.shape[1], B.shape[1])
    return float(np.min([
        abs(np.dot(A[:, i], B[:, i]) /
            (np.linalg.norm(A[:, i]) * np.linalg.norm(B[:, i]) + 1e-12))
        for i in range(n)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--chunk-size", type=int, default=5000)
    ap.add_argument("--n-comps", type=int, default=50)
    args = ap.parse_args()
    cs, k = args.chunk_size, args.n_comps

    import anndataoom as oom
    from anndataoom import chunked_pca

    a = prep(args.input)
    X = a.layers["scaled"]
    hvg = np.where(a.var["highly_variable_features"].values)[0]
    n_obs, d = X.shape[0], len(hvg)

    # ground-truth (exact covariance_eigh on verified HVG reconstruction)
    ref = exact_reference(make_hvg_chunker(X, hvg, subset_first=False),
                          n_obs, d, k, cs)

    from sklearn.decomposition import PCA as skPCA
    rss0 = _rss_mb()
    t0 = time.perf_counter()
    build_s = 0.0

    if args.strategy == "baseline":
        Xp, _, _ = chunked_pca(a, layer="scaled", n_comps=k,
                               chunk_size=cs, device="cpu")
    elif args.strategy in ("hvg_materialize", "hvg_from_raw"):
        sf = args.strategy == "hvg_from_raw"
        chunker = make_hvg_chunker(X, hvg, subset_first=sf)
        M = np.empty((n_obs, d), np.float32)
        for s, e, dc in chunker(cs):
            M[s:e] = dc.astype(np.float32)
        build_s = time.perf_counter() - t0
        Xp = skPCA(n_components=k, svd_solver="covariance_eigh").fit_transform(M)
    elif args.strategy == "streaming_cov":
        chunker = make_hvg_chunker(X, hvg, subset_first=True)
        Xp = exact_reference(chunker, n_obs, d, k, cs)
    elif args.strategy == "cache_then_pca":
        # build the bounded HVG cache (paid once, e.g. at scale-time), then
        # time ONLY the PCA on the cached matrix — models in-memory PCA stage
        chunker = make_hvg_chunker(X, hvg, subset_first=True)
        M = np.empty((n_obs, d), np.float32)
        for s, e, dc in chunker(cs):
            M[s:e] = dc.astype(np.float32)
        build_s = time.perf_counter() - t0
        t_pca = time.perf_counter()
        Xp = skPCA(n_components=k, svd_solver="covariance_eigh").fit_transform(M)
        pca_only = time.perf_counter() - t_pca
    else:
        raise SystemExit(f"unknown strategy {args.strategy}")

    total_s = time.perf_counter() - t0
    rss_mb = _rss_mb() - rss0
    cos = cosmin(ref, Xp[:n_obs]) if Xp.shape[0] == n_obs else cosmin(ref, Xp)
    out = {
        "strategy": args.strategy,
        "n_obs": int(n_obs), "n_hvg": int(d),
        "total_s": round(total_s, 2),
        "build_s": round(build_s, 2),
        "pca_only_s": round(locals().get("pca_only", float("nan")), 2),
        "cos_min_vs_exact": round(cos, 4),
        "rss_delta_mb": round(rss_mb, 0),
    }
    try:
        a.close()
    except Exception:
        pass
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
