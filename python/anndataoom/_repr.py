"""
AnnDataOOM representation — clean card layout for terminal and Jupyter.

- Terminal: properly-aligned Unicode box drawing (uses wcswidth for
  East-Asian-ambiguous chars so borders never break).
- Jupyter: card with zebra-striped sections, collapsible details,
  and an SVG mini-matrix visualisation next to the storage summary.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from scipy.sparse import issparse


# ──────────────────────────────────────────────────────────────────────
# Visual-width aware padding (East-Asian-ambiguous safe)
# ──────────────────────────────────────────────────────────────────────

try:
    from wcwidth import wcswidth as _wcswidth  # type: ignore
except ImportError:
    def _wcswidth(s: str) -> int:
        # Fallback: treat East-Asian wide as 2, everything else as 1
        import unicodedata
        total = 0
        for ch in s:
            if unicodedata.category(ch).startswith("M"):
                continue
            w = unicodedata.east_asian_width(ch)
            total += 2 if w in ("F", "W") else 1
        return total


def _pad(text: str, width: int) -> str:
    """Pad ``text`` on the right to ``width`` visual columns."""
    diff = width - _wcswidth(text)
    return text + " " * max(diff, 0)


# ──────────────────────────────────────────────────────────────────────
# Cached summaries
# ──────────────────────────────────────────────────────────────────────

def _get_repr_cache(adata) -> dict:
    if not hasattr(adata, "_repr_cache") or adata._repr_cache is None:
        adata._repr_cache = {}
    return adata._repr_cache


def _describe_storage(adata, sample_chunk: bool = True) -> dict:
    cache = _get_repr_cache(adata)
    if "storage" in cache:
        return cache["storage"]

    info: dict[str, Any] = {}

    fname = getattr(adata, "filename", None)
    if fname is not None:
        info["filename"] = str(fname)
        try:
            info["file_size_mb"] = os.path.getsize(str(fname)) / 1024 ** 2
        except Exception:
            info["file_size_mb"] = None
    else:
        info["filename"] = None
        info["file_size_mb"] = None

    X = adata._X
    info["x_class"] = type(X).__name__
    info["x_dtype"] = None
    info["x_format"] = None
    info["density"] = None
    info["chunk_mb"] = None
    info["sample_chunk_rows"] = None

    if sample_chunk:
        try:
            # Drill to bottom BackedArray (skip subset/transform wrappers)
            bottom = X
            while hasattr(bottom, "_parent") and bottom._parent is not None:
                bottom = bottom._parent
            target = min(1000, bottom.shape[0])
            for s, e, chunk in bottom.chunked(target):
                info["x_dtype"] = str(chunk.dtype)
                info["x_format"] = (
                    type(chunk).__name__ if issparse(chunk) else "ndarray"
                )
                if issparse(chunk):
                    total = chunk.shape[0] * chunk.shape[1]
                    info["density"] = chunk.nnz / max(total, 1)
                    info["chunk_mb"] = (
                        chunk.data.nbytes
                        + chunk.indices.nbytes
                        + chunk.indptr.nbytes
                    ) / 1024 ** 2
                else:
                    info["density"] = 1.0
                    info["chunk_mb"] = chunk.nbytes / 1024 ** 2
                info["sample_chunk_rows"] = chunk.shape[0]
                break
        except Exception:
            pass

    cache["storage"] = info
    return info


def _describe_chain(X) -> list[dict]:
    chain = []
    node = X
    while node is not None:
        cls = type(node).__name__
        desc: dict[str, Any] = {"class": cls, "shape": tuple(node.shape)}

        if cls == "_SubsetBackedArray":
            obs_idx = getattr(node, "_obs_idx", None)
            var_idx = getattr(node, "_var_idx", None)
            parts = []
            if obs_idx is not None:
                parts.append(f"obs: {len(obs_idx)}")
            if var_idx is not None:
                parts.append(f"var: {len(var_idx)}")
            desc["tag"] = "subset"
            desc["detail"] = ", ".join(parts) if parts else "–"
        elif cls == "TransformedBackedArray":
            parts = []
            if getattr(node, "_norm_factors", None) is not None:
                parts.append("normalize")
            if getattr(node, "_apply_log1p", False):
                parts.append("log1p")
            desc["tag"] = "transform"
            desc["detail"] = " · ".join(parts) if parts else "identity"
        elif cls == "ScaledBackedArray":
            mx = getattr(node, "_max_value", None)
            desc["tag"] = "scale"
            clip = f", clip=±{mx}" if mx is not None else ""
            desc["detail"] = f"z-score (μ,σ stored{clip})"
        elif cls == "BackedArray":
            desc["tag"] = "backed"
            desc["detail"] = ("Rust (anndata-rs)"
                              if getattr(node, "_is_rs", False)
                              else "in-memory")
        else:
            desc["tag"] = "other"
            desc["detail"] = ""

        chain.append(desc)
        node = getattr(node, "_parent", None)

    return list(reversed(chain))


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _fmt_shape(n_obs: int, n_vars: int) -> str:
    return f"{n_obs:,} × {n_vars:,}"


def _preview_keys(keys: list, max_n: int = 5, sep: str = " · ") -> str:
    if not keys:
        return "–"
    preview = sep.join(str(k) for k in keys[:max_n])
    if len(keys) > max_n:
        preview += f"  +{len(keys) - max_n}"
    return preview


def _fit_preview(keys: list, max_width: int, sep: str = " · ") -> str:
    """Greedy fit of column-name preview into max_width visual columns.

    Reserves room for a trailing "  +N" counter when overflow happens,
    so the number of hidden entries is always visible.
    """
    if not keys:
        return "–"
    total = len(keys)
    # Estimate worst-case suffix width: "  +N" where N ≤ total
    suffix_reserve = len(f"  +{total}")
    budget = max(max_width - suffix_reserve, 0)

    fitted = []
    used = 0
    for k in keys:
        name = str(k)
        piece = (sep if fitted else "") + name
        w = _wcswidth(piece)
        if used + w > budget:
            break
        fitted.append(name)
        used += w

    if len(fitted) == total:
        # Everything fits — no +N needed, reclaim the reserved space
        return sep.join(fitted)
    hidden = total - len(fitted)
    text = sep.join(fitted)
    # Append compact counter
    suffix = f"  +{hidden}"
    # If nothing fit at all, show at least one name with ellipsis
    if not fitted:
        name = str(keys[0])
        if _wcswidth(name) + suffix_reserve > max_width:
            # Truncate first name to fit
            while _wcswidth(name) + suffix_reserve > max_width and len(name) > 1:
                name = name[:-1]
            name += "…"
        return name + f"  +{total - 1}"
    return text + suffix


def _summary_line(storage: dict) -> str:
    parts = []
    if storage.get("x_format"):
        parts.append(storage["x_format"])
    if storage.get("x_dtype"):
        parts.append(storage["x_dtype"])
    d = storage.get("density")
    if d is not None and d < 1.0:
        parts.append(f"{d*100:.1f}% density")
    chunk_mb = storage.get("chunk_mb")
    sample_rows = storage.get("sample_chunk_rows")
    if chunk_mb is not None and sample_rows:
        parts.append(f"~{chunk_mb:.1f} MB/chunk ({sample_rows:,} rows)")
    sz = storage.get("file_size_mb")
    if sz is not None:
        parts.append(f"{sz:.1f} MB disk")
    return " · ".join(parts)


# ──────────────────────────────────────────────────────────────────────
# Plain text formatter — strict alignment via _wcswidth
# ──────────────────────────────────────────────────────────────────────

_WIDTH = 74
_INNER = _WIDTH - 4


def _hline(left: str, fill: str, right: str, w: int = _WIDTH) -> str:
    return left + fill * (w - 2) + right


def _row(content: str, w: int = _INNER) -> str:
    # Truncate if too long (visual width aware)
    while _wcswidth(content) > w:
        content = content[:-1]
    return f"│ {_pad(content, w)} │"


def _format_text(adata) -> str:
    storage = _describe_storage(adata)
    chain = _describe_chain(adata._X)

    tag = ("Rust · lazy · backed" if len(chain) > 1
           else "Rust · out-of-memory · backed")

    left = "AnnDataOOM"
    right = tag
    gap = _INNER - _wcswidth(left) - _wcswidth(right)
    if gap < 1:
        gap = 1
    header = left + " " * gap + right

    n_obs_str = f"{adata.n_obs:,}"
    n_vars_str = f"{adata.n_vars:,}"
    dims_line = f"   {n_obs_str}   ×   {n_vars_str}"
    # Labels aligned under numbers
    obs_pad = _wcswidth(n_obs_str) + 1  # +1 for space
    dims_labels = "   " + "obs".ljust(obs_pad + 3) + "    " + "vars"

    summary = _summary_line(storage) or "(unable to inspect X)"

    lines = [
        _hline("╭", "─", "╮"),
        _row(header),
        _hline("├", "─", "┤"),
        _row(""),
        _row(dims_line),
        _row(dims_labels),
        _row(""),
        _row("   " + summary),
    ]
    if storage.get("filename"):
        lines.append(_row("   " + os.path.basename(storage["filename"])))
    lines.append(_row(""))
    lines.append(_hline("├", "─", "┤"))

    sections = [
        ("obs",     list(adata.obs.columns)),
        ("var",     list(adata.var.columns)),
        ("obsm",    list(adata.obsm.keys())),
        ("varm",    list(adata.varm.keys())),
        ("obsp",    list(adata.obsp.keys())),
        ("varp",    list(adata.varp.keys())),
        ("layers",  list(adata.layers.keys()) if hasattr(adata.layers, "keys") else []),
    ]
    for name, keys in sections:
        n = len(keys)
        count = f"({n})" if n else "(–)"
        prefix = f"▸ {name:<7s}{count:<5s}  "
        avail = _INNER - _wcswidth(prefix)
        preview = _fit_preview(keys, avail) if keys else ""
        lines.append(_row(prefix + preview))

    if adata.raw is not None:
        raw_shape = adata.raw.shape
        raw_line = f"▸ {'raw':<7s}       {raw_shape[0]:,} × {raw_shape[1]:,} (pre-subset)"
    else:
        raw_line = f"▸ {'raw':<7s}(–)"
    lines.append(_row(raw_line))

    if len(chain) > 1:
        lines.append(_hline("├", "─", "┤"))
        lines.append(_row(f"Transform chain ({len(chain)} nodes):"))
        cls_w = max(len(n["class"]) for n in chain)
        shp_w = max(len(_fmt_shape(*n["shape"])) for n in chain)
        for i, node in enumerate(chain):
            shape_str = _fmt_shape(*node["shape"])
            detail = node.get("detail", "")
            line = f"  [{i}] {node['class']:<{cls_w}}  {shape_str:>{shp_w}}  {node['tag']}"
            if detail:
                line += f" · {detail}"
            lines.append(_row(line))

    lines.append(_hline("╰", "─", "╯"))
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# HTML formatter — card + zebra + SVG matrix viz
# ──────────────────────────────────────────────────────────────────────

_HTML_STYLE = """
<style>
.adoom {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 12px;
    max-width: 640px;
    color: var(--jp-ui-font-color0, #1f2328);
    background: var(--jp-layout-color0, #ffffff);
    border: 1px solid var(--jp-border-color2, #d0d7de);
    border-radius: 6px;
    overflow: hidden;
    margin: 4px 0;
    font-variant-numeric: tabular-nums;
    line-height: 1.4;
}
.adoom-head {
    display: flex; align-items: baseline; gap: 10px;
    padding: 6px 10px;
    background: var(--jp-layout-color1, #f6f8fa);
    border-bottom: 1px solid var(--jp-border-color3, #eaeef2);
    flex-wrap: wrap;
}
.adoom-head .title { font-weight: 600; font-size: 12px; }
.adoom-head .dims {
    font-size: 13px; font-weight: 600;
    color: var(--jp-ui-font-color0, #1f2328);
    letter-spacing: -0.01em;
}
.adoom-head .dims-label {
    font-size: 10px; font-weight: 400;
    color: var(--jp-ui-font-color2, #656d76);
    margin-left: 2px;
}
.adoom-head .dims-sep {
    color: var(--jp-ui-font-color3, #8c959f);
    font-weight: 300; margin: 0 2px;
}
.adoom-head .tag {
    margin-left: auto;
    font-size: 10px; color: var(--jp-ui-font-color2, #656d76);
    padding: 1px 7px;
    background: var(--jp-layout-color2, #eaeef2);
    border-radius: 10px;
}
.adoom-body {
    display: flex; gap: 10px; padding: 6px 10px;
    align-items: center;
    border-bottom: 1px solid var(--jp-border-color3, #eaeef2);
    font-size: 11px;
}
.adoom-body .summary {
    flex: 1; color: var(--jp-ui-font-color1, #424a53);
    overflow: hidden; text-overflow: ellipsis;
}
.adoom-body code {
    font-family: "SFMono-Regular", Menlo, Consolas, monospace;
    font-size: 10px; padding: 0 3px;
    background: var(--jp-layout-color2, #eaeef2);
    border-radius: 2px;
}
.adoom-viz { flex-shrink: 0; }
.adoom-sections { display: flex; flex-direction: column; }
.adoom-sec {
    border-bottom: 1px solid var(--jp-border-color3, #eaeef2);
}
.adoom-sec:last-child { border-bottom: none; }
.adoom-sec:nth-child(odd) { background: var(--jp-layout-color1, #f6f8fa); }
.adoom-sec > summary {
    cursor: pointer; list-style: none;
    padding: 3px 10px;
    display: flex; align-items: center; gap: 8px;
    transition: background 0.1s;
    font-size: 11px;
}
.adoom-sec > summary::-webkit-details-marker { display: none; }
.adoom-sec > summary::marker { content: ""; }
.adoom-sec > summary:hover {
    background: var(--jp-layout-color2, rgba(208,215,222,0.5));
}
.adoom-sec.muted > summary {
    cursor: default; color: var(--jp-ui-font-color3, #8c959f);
    font-size: 10px;
}
.adoom-sec.muted > summary:hover { background: inherit; }
.adoom-chev {
    color: var(--jp-ui-font-color3, #8c959f);
    transition: transform 0.15s ease;
    width: 8px; display: inline-block;
    font-size: 10px;
}
.adoom-sec[open] > summary > .adoom-chev { transform: rotate(90deg); }
.adoom-sec.muted > summary > .adoom-chev { visibility: hidden; }
.adoom-sec-name { font-weight: 500; min-width: 48px; }
.adoom-sec-count {
    color: var(--jp-ui-font-color2, #656d76);
    font-size: 10px; min-width: 26px; text-align: right;
    font-variant-numeric: tabular-nums;
}
.adoom-sec-preview {
    color: var(--jp-ui-font-color2, #656d76);
    font-size: 11px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    flex: 1;
}
.adoom-sec-body { padding: 0 10px 6px 32px; }
.adoom-sec-body table {
    border-collapse: collapse; font-size: 10px; width: 100%;
}
.adoom-sec-body td, .adoom-sec-body th {
    padding: 2px 8px; text-align: left;
}
.adoom-sec-body tr:nth-child(even) {
    background: var(--jp-layout-color2, rgba(208,215,222,0.25));
}
.adoom-sec-body th {
    font-weight: 500; color: var(--jp-ui-font-color2, #656d76);
    font-size: 9px; text-transform: uppercase; letter-spacing: 0.04em;
    border-bottom: 1px solid var(--jp-border-color3, #eaeef2);
}
.adoom-sec-body code {
    font-family: "SFMono-Regular", Menlo, Consolas, monospace;
    font-size: 10px;
}
.adoom-chain {
    padding: 4px 10px 6px 32px;
    font-family: "SFMono-Regular", Menlo, Consolas, monospace;
    font-size: 10px;
}
.adoom-chain-row {
    display: grid;
    grid-template-columns: 22px 70px 130px auto;
    gap: 8px; padding: 1px 0;
    align-items: center;
}
.adoom-chain-tag {
    padding: 0 5px; border-radius: 2px; font-weight: 500;
    font-size: 9px; text-align: center; letter-spacing: 0.02em;
}
.tag-backed    { background: rgba(100, 116, 139, 0.15); color: #475569; }
.tag-subset    { background: rgba(217, 119,   6, 0.15); color: #92400e; }
.tag-transform { background: rgba( 37,  99, 235, 0.15); color: #1d4ed8; }
.tag-scale     { background: rgba(220,  38,  38, 0.15); color: #991b1b; }
</style>
"""


def _escape(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _svg_matrix_viz(n_obs: int, n_vars: int, density: float | None) -> str:
    """Compact inline matrix visualisation (rectangle + chunk bands +
    density dots). No axis labels — the shape numbers already appear
    in the header. Total SVG size ≤ 56 × 34 px."""
    import math

    w_logical = max(n_vars, 1)
    h_logical = max(n_obs, 1)
    ratio_h = math.sqrt(h_logical / max(w_logical, 1))
    aspect = max(0.4, min(1.8, ratio_h))

    MAX = 44
    if aspect >= 1:
        W = int(MAX / aspect)
        H = MAX
    else:
        W = MAX
        H = int(MAX * aspect)
    W = max(W, 20)
    H = max(H, 20)

    border = "#8c959f"
    band = "#d0d7de"
    dot = "#1d4ed8"

    PAD = 2
    svg_w = W + PAD * 2
    svg_h = H + PAD * 2

    parts = [
        f'<svg class="adoom-viz" viewBox="0 0 {svg_w} {svg_h}" '
        f'width="{svg_w}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg">'
    ]
    parts.append(
        f'<rect x="{PAD}" y="{PAD}" width="{W}" height="{H}" fill="white" '
        f'stroke="{border}" stroke-width="0.7" rx="1"/>'
    )

    # Chunk bands — 6 horizontal lines
    n_bands = 6
    band_h = H / n_bands
    for i in range(1, n_bands):
        y = PAD + i * band_h
        parts.append(
            f'<line x1="{PAD}" y1="{y:.1f}" x2="{PAD+W}" y2="{y:.1f}" '
            f'stroke="{band}" stroke-width="0.3"/>'
        )

    if density is not None and density > 0:
        import random
        rng = random.Random(hash((n_obs, n_vars, round(density, 4))))
        n_dots = min(int(density * 60), 60)
        for _ in range(n_dots):
            x = PAD + rng.random() * W
            y = PAD + rng.random() * H
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="0.5" '
                f'fill="{dot}" opacity="0.6"/>'
            )

    parts.append('</svg>')
    return "".join(parts)


def _format_html(adata) -> str:
    storage = _describe_storage(adata)
    chain = _describe_chain(adata._X)

    tag = ("Rust · lazy · backed" if len(chain) > 1
           else "Rust · out-of-memory · backed")

    parts = [_HTML_STYLE, '<div class="adoom">']

    # Compact single-row header: title · dims · tag
    parts.append('<div class="adoom-head">')
    parts.append('<span class="title">AnnDataOOM</span>')
    parts.append(
        f'<span class="dims">{adata.n_obs:,}<span class="dims-label">obs</span>'
        f'<span class="dims-sep">×</span>{adata.n_vars:,}<span class="dims-label">vars</span></span>'
    )
    parts.append(f'<span class="tag">{_escape(tag)}</span>')
    parts.append('</div>')

    # Single-row body: summary line + tiny SVG
    summary_parts = []
    if storage.get("x_format"):
        s = f'<code>{_escape(storage["x_format"])}</code>'
        if storage.get("x_dtype"):
            s += f' {_escape(storage["x_dtype"])}'
        d = storage.get("density")
        if d is not None and d < 1.0:
            s += f' · {d*100:.1f}%'
        summary_parts.append(s)
    chunk_mb = storage.get("chunk_mb")
    if chunk_mb is not None:
        summary_parts.append(f'~{chunk_mb:.1f} MB/chunk')
    if storage.get("file_size_mb"):
        summary_parts.append(f'{storage["file_size_mb"]:.0f} MB disk')
    if storage.get("filename"):
        summary_parts.append(f'<code>{_escape(os.path.basename(storage["filename"]))}</code>')

    parts.append('<div class="adoom-body">')
    parts.append(
        f'<span class="summary">{" · ".join(summary_parts) if summary_parts else "(no X info)"}</span>'
    )
    parts.append(_svg_matrix_viz(adata.n_obs, adata.n_vars, storage.get("density")))
    parts.append('</div>')

    # Sections — non-empty first, empty ones merged into one muted row
    parts.append('<div class="adoom-sections">')

    all_sections = [
        ("obs", list(adata.obs.columns), lambda: _render_df_table(adata.obs)),
        ("var", list(adata.var.columns), lambda: _render_df_table(adata.var)),
        ("obsm", list(adata.obsm.keys()), lambda: _render_axis_arrays(adata.obsm)),
        ("varm", list(adata.varm.keys()), lambda: _render_axis_arrays(adata.varm)),
        ("obsp", list(adata.obsp.keys()), lambda: _render_axis_arrays(adata.obsp)),
        ("varp", list(adata.varp.keys()), lambda: _render_axis_arrays(adata.varp)),
        ("layers", list(adata.layers.keys()) if hasattr(adata.layers, "keys") else [],
         lambda: _render_axis_arrays(adata.layers)),
    ]

    # raw is special — treat it as a pseudo-section
    has_raw = adata.raw is not None

    chev = '<span class="adoom-chev">›</span>'

    empty_names = []
    for name, keys, renderer in all_sections:
        n = len(keys)
        if n == 0:
            empty_names.append(name)
            continue
        preview_txt = _preview_keys(keys, 6)
        parts.append('<details class="adoom-sec">')
        parts.append(
            f'<summary>{chev}'
            f'<span class="adoom-sec-name">{_escape(name)}</span>'
            f'<span class="adoom-sec-count">{n}</span>'
            f'<span class="adoom-sec-preview">{_escape(preview_txt)}</span>'
            f'</summary>'
        )
        parts.append('<div class="adoom-sec-body">')
        parts.append(renderer())
        parts.append('</div></details>')

    # raw — show when non-empty, add to empty list when absent
    if has_raw:
        raw_shape = adata.raw.shape
        parts.append('<details class="adoom-sec">')
        parts.append(
            f'<summary>{chev}'
            f'<span class="adoom-sec-name">raw</span>'
            f'<span class="adoom-sec-count">1</span>'
            f'<span class="adoom-sec-preview">{raw_shape[0]:,} × {raw_shape[1]:,} (pre-subset)</span>'
            f'</summary></details>'
        )
    else:
        empty_names.append("raw")

    # Merged empty row — a non-expandable details (no body) so CSS selectors work
    if empty_names:
        parts.append('<details class="adoom-sec muted">')
        parts.append(
            f'<summary>{chev}'
            f'<span class="adoom-sec-name">empty</span>'
            f'<span class="adoom-sec-count">–</span>'
            f'<span class="adoom-sec-preview">{_escape(" · ".join(empty_names))}</span>'
            f'</summary>'
        )
        parts.append('</details>')

    # Transform chain — only show when non-trivial
    if len(chain) > 1:
        parts.append('<details class="adoom-sec" open>')
        parts.append(
            f'<summary>{chev}'
            f'<span class="adoom-sec-name">chain</span>'
            f'<span class="adoom-sec-count">{len(chain)}</span>'
            f'<span class="adoom-sec-preview">transform pipeline</span>'
            f'</summary>'
        )
        parts.append('<div class="adoom-chain">')
        for i, node in enumerate(chain):
            shape_str = _fmt_shape(*node["shape"])
            detail = node.get("detail", "")
            parts.append(
                f'<div class="adoom-chain-row">'
                f'<span>[{i}]</span>'
                f'<span class="adoom-chain-tag tag-{node["tag"]}">{_escape(node["tag"])}</span>'
                f'<span><code>{_escape(node["class"])}</code></span>'
                f'<span>{_escape(shape_str)} · {_escape(detail)}</span>'
                f'</div>'
            )
        parts.append('</div></details>')

    parts.append('</div>')  # /sections
    parts.append('</div>')  # /card
    return "".join(parts)


def _render_df_table(df) -> str:
    parts = ['<table><thead><tr><th>name</th><th>dtype</th><th>preview</th></tr></thead><tbody>']
    for col in list(df.columns)[:20]:
        try:
            series = df[col]
            dtype = str(series.dtype)
            try:
                uniq = series.unique()
                if len(uniq) > 3:
                    prev = ", ".join(str(u) for u in uniq[:3]) + f", … ({len(uniq)})"
                else:
                    prev = ", ".join(str(u) for u in uniq)
            except Exception:
                prev = ""
            parts.append(
                f'<tr><td><code>{_escape(col)}</code></td>'
                f'<td>{_escape(dtype)}</td>'
                f'<td>{_escape(prev)}</td></tr>'
            )
        except Exception:
            pass
    if len(df.columns) > 20:
        parts.append(f'<tr><td colspan="3"><em>+{len(df.columns) - 20} more</em></td></tr>')
    parts.append('</tbody></table>')
    return "".join(parts)


def _render_axis_arrays(mapping) -> str:
    parts = ['<table><thead><tr><th>key</th><th>shape</th><th>dtype</th></tr></thead><tbody>']
    for k in mapping.keys():
        try:
            v = mapping[k]
            shape = getattr(v, "shape", ("?",))
            dtype = str(getattr(v, "dtype", "?"))
            parts.append(
                f'<tr><td><code>{_escape(k)}</code></td>'
                f'<td>{_escape(str(shape))}</td>'
                f'<td>{_escape(dtype)}</td></tr>'
            )
        except Exception:
            parts.append(f'<tr><td><code>{_escape(k)}</code></td><td colspan="2">–</td></tr>')
    parts.append('</tbody></table>')
    return "".join(parts)


def _format_read_message(filename: str, size_mb: float | None, elapsed: float,
                         ram_delta_mb: float | None = None) -> str:
    lines = [
        "📂 Reading with anndata-rs (Rust · out-of-memory)",
        f"   {filename}" + (f"  ({size_mb:.1f} MB)" if size_mb else ""),
    ]
    extra = f" · +{ram_delta_mb:.0f} MB RAM" if ram_delta_mb is not None else ""
    lines.append(f"   ✓ Loaded in {elapsed:.2f}s{extra}")
    lines.append("")
    lines.append("💡 Data stays on disk. Use ov.pp.* for chunked processing.")
    lines.append("   adata.close() when done · adata.to_adata() to materialise")
    lines.append("")
    return "\n".join(lines)
