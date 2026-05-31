"""Generate paper figures and table fragments from results/*.json.

Outputs:
    figures/fig_totaltime.pdf   — 2-panel total wall-clock + peak RSS
    figures/fig_stages.pdf      — per-stage stacked bars
    paper/table_datasets.tex    — dataset metadata table body
    paper/table_headline.tex    — wall-clock + RSS headline table body
    paper/numbers.tex           — inline-citable \newcommand definitions
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
PAPER = ROOT / "paper"
FIG.mkdir(exist_ok=True)


# Dataset metadata — kept in code so we don't re-parse h5ad files at plot
# time. Order is the x-axis order in figures.
DATASETS = [
    ("ts_5k",            5_000, 60_606, "TS-5k",        "Tabula Sapiens vasculature (sub)"),
    ("ts_10k",          10_000, 60_606, "TS-10k",       "Tabula Sapiens vasculature (sub)"),
    ("ts_vasculature",  42_650, 60_606, "TS-Vasc",      "Tabula Sapiens vasculature"),
    ("ts_stromal",     232_684, 60_606, "TS-Stromal",   "Tabula Sapiens stromal compartment"),
    ("ts_epithelial",  228_032, 60_606, "TS-Epi",       "Tabula Sapiens epithelial compartment"),
    ("ts_immune",      592_317, 60_606, "TS-Immune",    "Tabula Sapiens immune compartment"),
    ("ts_1M",        1_053_033, 60_606, "TS-1M",        "Tabula Sapiens stromal+epi+immune (concat)"),
]

CONFIGS = [
    ("ov-anndata",          "ov + anndata (in-mem, dense scale)"),
    ("ov-anndata-implicit", "ov + anndata (in-mem, implicit scale v0.1.7)"),
    ("scanpy-backed",       "scanpy + backed='r'"),
    ("ov-oom",              "ov + anndataoom (OOM)"),
]

CONFIG_COLOR = {
    "ov-anndata":          "#444a73",   # deep blue-grey
    "ov-anndata-implicit": "#7a89d4",   # lighter blue (variant of in-mem)
    "scanpy-backed":       "#9a6ba8",   # muted violet
    "ov-oom":              "#e89148",   # the omicverse orange
}


def load_results():
    """Returns dict keyed by (config, dataset_key) → result dict.

    Missing entries (e.g. OOMs) come back as None.
    """
    out = {}
    for cfg, _ in CONFIGS:
        for key, *_ in DATASETS:
            p = RES / f"{cfg}__{key}.json"
            out[(cfg, key)] = json.loads(p.read_text()) if p.exists() else None
    return out


def total_time(r):
    return r["total_seconds"]


def peak_rss(r):
    """Approximate peak RSS as the max of rss_mb_after across stages."""
    return max(
        s.get("rss_mb_after", 0)
        for s in r["stages"].values()
        if isinstance(s, dict)
    )


def fig_totaltime(res):
    n_cells = np.array([d[1] for d in DATASETS])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    # Left: total wall-clock
    ax = axes[0]
    for cfg, label in CONFIGS:
        ys = []
        for key, *_ in DATASETS:
            r = res[(cfg, key)]
            ys.append(total_time(r) if r else np.nan)
        ys = np.array(ys, dtype=float)
        # Plot solid for completed, X marker for missing
        ax.plot(n_cells, ys, marker="o", lw=2.2,
                color=CONFIG_COLOR[cfg], label=label)
        # OOM marker
        for x, y in zip(n_cells, ys):
            if np.isnan(y):
                ax.annotate("OOM", (x, ax.get_ylim()[1] * 0.85),
                            ha="center", color="firebrick", fontsize=8,
                            fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cells")
    ax.set_ylabel("wall-clock (s)")
    ax.set_title("(a) end-to-end pipeline time")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    # Right: peak RSS
    ax = axes[1]
    for cfg, label in CONFIGS:
        ys = []
        for key, *_ in DATASETS:
            r = res[(cfg, key)]
            ys.append(peak_rss(r) if r else np.nan)
        ys = np.array(ys, dtype=float)
        ax.plot(n_cells, ys, marker="s", lw=2.2,
                color=CONFIG_COLOR[cfg], label=label)
        for x, y in zip(n_cells, ys):
            if np.isnan(y):
                ax.annotate("OOM", (x, ax.get_ylim()[1] * 0.85),
                            ha="center", color="firebrick", fontsize=8,
                            fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cells")
    ax.set_ylabel("peak RSS (MB)")
    ax.set_title("(b) peak resident memory")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    fig.tight_layout()
    out_pdf = FIG / "fig_totaltime.pdf"
    out_png = FIG / "fig_totaltime.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out_pdf}")


def fig_stages(res):
    """Per-stage stacked-bar comparison across datasets."""
    stages = ["load", "qc", "preprocess", "scale", "pca"]
    stage_colors = ["#5e80b4", "#8c6bb1", "#41a386", "#d68b48", "#c75e3a"]

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(13, 3.8),
                              sharey=False)
    for i, (key, *_, label, _) in enumerate(DATASETS):
        ax = axes[i]
        positions = np.arange(len(CONFIGS))
        for j, (cfg, _) in enumerate(CONFIGS):
            r = res[(cfg, key)]
            bottom = 0.0
            if r is None:
                ax.text(positions[j], 0.5, "OOM", ha="center", va="center",
                        color="firebrick", fontweight="bold")
                continue
            for s, color in zip(stages, stage_colors):
                st = r["stages"].get(s, {})
                t = st.get("seconds", 0) if isinstance(st, dict) else 0
                ax.bar(positions[j], t, bottom=bottom, color=color,
                       edgecolor="white", lw=0.5,
                       label=s if (i == 0 and j == 0) else None)
                bottom += t
            ax.text(positions[j], bottom * 1.04, f"{bottom:.0f}s",
                    ha="center", fontsize=8)
        ax.set_xticks(positions)
        ax.set_xticklabels([c[0] for c in CONFIGS],
                           rotation=20, fontsize=8, ha="right")
        ax.set_title(label, fontsize=10)
        if i == 0:
            ax.set_ylabel("time (s)")
        ax.grid(True, axis="y", lw=0.3, alpha=0.4)

    axes[0].legend(loc="upper left", fontsize=8, frameon=False, ncol=1)
    fig.tight_layout()
    out_pdf = FIG / "fig_stages.pdf"
    out_png = FIG / "fig_stages.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out_pdf}")


def _f(v):  # safe float format
    return f"{v:.1f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "--"


def table_datasets():
    """Datasets table body — no preamble, just rows."""
    lines = [
        "\\begin{tabular}{lrrrl}",
        "\\toprule",
        "Dataset & Cells & Genes & On-disk (MB) & Source \\\\",
        "\\midrule",
    ]
    for key, n_cells, n_genes, label, src in DATASETS:
        p = ROOT / "data" / f"{key}.h5ad"
        sz = p.stat().st_size / 1024 / 1024 if p.exists() else 0.0
        lines.append(
            f"{label} & {n_cells:,} & {n_genes:,} & {sz:.1f} & {src} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    out = PAPER / "table_datasets.tex"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def table_headline(res):
    """Wall-clock and peak RSS for each (config × dataset) cell.

    Four-column variant: ov-anndata (dense scale), ov-anndata (implicit
    v0.1.7), scanpy backed='r', ov-oom. ``--`` denotes runs that hit
    the per-process RSS cap; ``OOM`` is the wall-clock column for those.
    """
    cfg_labels = ["ov-anndata", "ov-anndata-implicit",
                  "scanpy-backed", "ov-oom"]
    cfg_short = {
        "ov-anndata": "ov-anndata",
        "ov-anndata-implicit": r"ov-anndata\textsuperscript{*}",
        "scanpy-backed": "scanpy-backed",
        "ov-oom": "ov-oom",
    }
    # Column groups: each config gets (time, RSS).
    col_spec = "l" + "rr" * len(cfg_labels)
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]
    # Header row: config labels spanning 2 cols each.
    header_cells = [""]
    for c in cfg_labels:
        header_cells.append(
            f"\\multicolumn{{2}}{{c}}{{\\texttt{{{cfg_short[c]}}}}}"
        )
    lines.append(" & ".join(header_cells) + " \\\\")
    # Sub-header.
    sub_cells = [""]
    for _ in cfg_labels:
        sub_cells += ["t (s)", "RSS (MB)"]
    lines.append(" & ".join(sub_cells) + " \\\\")
    lines.append("\\midrule")
    for key, *_, label, _ in DATASETS:
        row = [label]
        for c in cfg_labels:
            r = res.get((c, key))
            t = _f(total_time(r)) if r else "OOM"
            m = _f(peak_rss(r))   if r else "--"
            row += [t, m]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out = PAPER / "table_headline.tex"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def write_numbers(res):
    """Inline-citable \\newcommand definitions used by main.tex.

    Writes to ``numbers_auto.tex`` so the hand-curated ``numbers.tex`` (which
    holds the v0.1.6 PCA-acceleration micro-benchmark numbers, hand-set after
    a separate side-experiment) is not clobbered. ``main.tex`` ``\\input``s
    both files.
    """
    lines = ["% Auto-generated by scripts/plot.py — do NOT edit by hand."]

    def cmd(name, val):
        lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    # Memory ratio at smallest completed pair.
    r5k_a = res.get(("ov-anndata", "ts_5k"))
    r5k_o = res.get(("ov-oom",     "ts_5k"))
    if r5k_a and r5k_o:
        cmd("bnetSmall", f"{peak_rss(r5k_a) / peak_rss(r5k_o):.1f}")

    # Largest size where ov-anndata completed and largest where OOM-killed.
    last_inmem_key = None
    first_oom_key  = None
    for key, n, *_, label, _ in DATASETS:
        ra = res.get(("ov-anndata", key))
        if ra:
            last_inmem_key = (key, n, label)
        elif first_oom_key is None and res.get(("ov-oom", key)) is not None:
            first_oom_key = (key, n, label)
    if last_inmem_key:
        cmd("largestInMem", f"{last_inmem_key[1]:,}")
        cmd("largestInMemLabel", last_inmem_key[2])
    if first_oom_key:
        cmd("firstOomSize", f"{first_oom_key[1]:,}")
        cmd("firstOomLabel", first_oom_key[2])

    # Headline 1M ov-oom numbers (the marquee result).
    r1m_o = res.get(("ov-oom", "ts_1M"))
    if r1m_o:
        cmd("oneMCells", "1{,}053{,}033")
        cmd("oneMTotalSec",  f"{total_time(r1m_o):.0f}")
        cmd("oneMTotalMin",  f"{total_time(r1m_o)/60:.1f}")
        cmd("oneMPeakMB",    f"{peak_rss(r1m_o):.0f}")
        cmd("oneMPeakGB",    f"{peak_rss(r1m_o)/1024:.1f}")
        # PCA stage at 1M.
        pca_s = r1m_o["stages"].get("pca", {}).get("seconds")
        if pca_s is not None:
            cmd("oneMPcaSec", f"{pca_s:.0f}")
            cmd("oneMPcaMin", f"{pca_s/60:.1f}")

    # Speedup + memory ratio on the largest comparable pair.
    if last_inmem_key:
        ra = res[("ov-anndata", last_inmem_key[0])]
        ro = res[("ov-oom",     last_inmem_key[0])]
        if ra and ro:
            cmd("largeSpeedup",   f"{total_time(ra) / total_time(ro):.2f}")
            cmd("largeMemRatio",  f"{peak_rss(ra) / peak_rss(ro):.1f}")
            cmd("largeCompareLabel", last_inmem_key[2])

    # Per-process RSS cap that produced the OOMs.
    cmd("rssCapGB", "256")

    # ── Implicit-centering unblock: largest size where ov-anndata-implicit
    # completes when the plain ov-anndata path is OOM-killed. This is the
    # cleanest narrative for "what does v0.1.7 buy you" on the in-memory
    # backend.
    largest_implicit_unblock = None
    for key, n, *_, label, _ in DATASETS:
        ra = res.get(("ov-anndata", key))
        ri = res.get(("ov-anndata-implicit", key))
        if ra is None and ri is not None:
            largest_implicit_unblock = (key, n, label, ri)
    if largest_implicit_unblock:
        key, n, label, ri = largest_implicit_unblock
        cmd("implicitUnblockSize", f"{n:,}")
        cmd("implicitUnblockLabel", label)
        cmd("implicitUnblockTotalSec", f"{total_time(ri):.0f}")
        cmd("implicitUnblockTotalMin", f"{total_time(ri)/60:.1f}")
        cmd("implicitUnblockPeakGB",   f"{peak_rss(ri)/1024:.0f}")

    # ── Where does scanpy-backed die? First dataset where it OOMs.
    first_backed_oom = None
    for key, n, *_, label, _ in DATASETS:
        if res.get(("scanpy-backed", key)) is None \
                and res.get(("ov-oom", key)) is not None:
            first_backed_oom = (key, n, label)
            break
    if first_backed_oom:
        cmd("scanpyBackedFirstOomSize", f"{first_backed_oom[1]:,}")
        cmd("scanpyBackedFirstOomLabel", first_backed_oom[2])

    # ── How much memory does ov-oom save on the largest pair where the
    # in-memory dense path *and* scanpy-backed both still complete?
    # (TS-Epi is typically the answer; gives the cleanest "all four
    # configs fit, here is the spread" comparison.)
    for key, n, *_, label, _ in reversed(DATASETS):
        ra = res.get(("ov-anndata", key))
        rb = res.get(("scanpy-backed", key))
        ri = res.get(("ov-anndata-implicit", key))
        ro = res.get(("ov-oom", key))
        if all(x is not None for x in (ra, rb, ri, ro)):
            cmd("fourWayLabel", label)
            cmd("fourWayCells",   f"{n:,}")
            cmd("fourWayRssAnn",  f"{peak_rss(ra)/1024:.0f}")
            cmd("fourWayRssImp",  f"{peak_rss(ri)/1024:.0f}")
            cmd("fourWayRssBack", f"{peak_rss(rb)/1024:.0f}")
            cmd("fourWayRssOom",  f"{peak_rss(ro)/1024:.1f}")
            cmd("fourWayTimeOom", f"{total_time(ro):.0f}")
            break

    out = PAPER / "numbers_auto.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def main():
    res = load_results()
    print(f"loaded {sum(v is not None for v in res.values())}/{len(res)} results")
    fig_totaltime(res)
    fig_stages(res)
    table_datasets()
    table_headline(res)
    write_numbers(res)


if __name__ == "__main__":
    main()
