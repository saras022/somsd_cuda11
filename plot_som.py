#!/usr/bin/env python3
"""
plot_som.py -- Vector plot generator for somsd_cuda_11 training results.

Reads the "-res <file>" output produced by `somsd` (one line per training
sample: "<graph_id> <node_id> <winner_x> <winner_y> <label>") and renders a
plot of where samples landed on the SOM grid, colored/labelled by class.

Designed to scale from small maps up to high-resolution maps (e.g. a
3840x2160 codebook grid): below --scatter-limit points it draws a true
per-point scatter with a legend; above that it switches automatically to a
2D density heatmap (one bin per grid cell) so the plot stays fast and
legible instead of drawing millions of overlapping markers.

Usage:
    python3 plot_som.py results.txt --out som_plot.png
    python3 plot_som.py results.txt --xdim 3840 --ydim 2160 --out som_plot.png
    python3 plot_som.py results.txt --mode heatmap --out som_density.png

Requires: numpy, matplotlib (pip install numpy matplotlib)
"""
import argparse
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_results(path):
    """Parse a somsd result file: gnum nodeid x y label"""
    xs, ys, labels = [], [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                x = int(parts[2])
                y = int(parts[3])
            except ValueError:
                continue
            label = parts[4] if len(parts) > 4 else "unlabeled"
            xs.append(x)
            ys.append(y)
            labels.append(label)
    if not xs:
        sys.exit(f"No data rows parsed from '{path}'. "
                  f"Expected lines like: 'graphid nodeid x y label'.")
    return np.array(xs), np.array(ys), labels


def scatter_plot(xs, ys, labels, xdim, ydim, out, dpi, point_size, alpha, title):
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20" if len(unique_labels) <= 20 else "gist_ncar")
    color_of = {lab: cmap(i / max(1, len(unique_labels) - 1) if len(unique_labels) > 1 else 0.0)
                for i, lab in enumerate(unique_labels)}
    colors = [color_of[l] for l in labels]

    fig_w = max(8, min(24, xdim / 200))
    fig_h = max(6, min(24, ydim / 200))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("#0b0f14")
    fig.patch.set_facecolor("#0b0f14")

    ax.scatter(xs, ys, s=point_size, c=colors, alpha=alpha, linewidths=0)

    ax.set_xlim(-0.5, xdim - 0.5)
    ax.set_ylim(-0.5, ydim - 0.5)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("map x", color="#cfd8dc")
    ax.set_ylabel("map y", color="#cfd8dc")
    ax.tick_params(colors="#78909c")
    for spine in ax.spines.values():
        spine.set_color("#37474f")
    ax.set_title(title, color="#eceff1", fontsize=14)

    if 1 < len(unique_labels) <= 40:
        handles = [Line2D([0], [0], marker='o', color='none',
                           markerfacecolor=color_of[l], markersize=8, label=l)
                   for l in unique_labels]
        legend = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                            facecolor="#0b0f14", edgecolor="#37474f", fontsize=8)
        for text in legend.get_texts():
            text.set_color("#cfd8dc")

    fig.tight_layout()
    fig.savefig(out, dpi=dpi, facecolor=fig.get_facecolor())
    print(f"Wrote {out} ({len(xs)} points, {len(unique_labels)} distinct labels)")


def heatmap_plot(xs, ys, labels, xdim, ydim, out, dpi, title, bins_cap=1200):
    # Downsample the grid to a manageable number of bins per axis for very
    # high-resolution maps, while keeping bins square in map-space.
    bx = min(xdim, bins_cap)
    by = min(ydim, bins_cap)

    density, xedges, yedges = np.histogram2d(
        xs, ys, bins=[bx, by], range=[[0, xdim], [0, ydim]])

    fig_w = max(8, min(24, xdim / 200))
    fig_h = max(6, min(24, ydim / 200))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        np.log1p(density.T), origin="upper",
        extent=[0, xdim, ydim, 0],
        cmap="magma", aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("log(1 + sample count)", color="#333")

    ax.set_xlabel("map x")
    ax.set_ylabel("map y")
    ax.set_title(title, fontsize=14)

    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    print(f"Wrote {out} (density heatmap, {bx}x{by} bins over a {xdim}x{ydim} map, "
          f"{len(xs)} samples)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_file", help="Path to the -res output file from somsd")
    ap.add_argument("--out", default="som_plot.png", help="Output image path")
    ap.add_argument("--xdim", type=int, default=None,
                     help="Map horizontal size (default: inferred from data + 1)")
    ap.add_argument("--ydim", type=int, default=None,
                     help="Map vertical size (default: inferred from data + 1)")
    ap.add_argument("--mode", choices=["auto", "scatter", "heatmap"], default="auto",
                     help="Plot style. 'auto' picks scatter for small maps and "
                          "a density heatmap for high-resolution maps.")
    ap.add_argument("--scatter-limit", type=int, default=200_000,
                     help="Above this many samples, 'auto' mode switches to heatmap.")
    ap.add_argument("--point-size", type=float, default=6.0)
    ap.add_argument("--alpha", type=float, default=0.65)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    xs, ys, labels = load_results(args.result_file)
    xdim = args.xdim or int(xs.max()) + 1
    ydim = args.ydim or int(ys.max()) + 1
    title = args.title or f"SOM winner coordinates ({xdim}x{ydim} map, {len(xs)} samples)"

    mode = args.mode
    if mode == "auto":
        mode = "heatmap" if len(xs) > args.scatter_limit else "scatter"

    if mode == "scatter":
        scatter_plot(xs, ys, labels, xdim, ydim, args.out, args.dpi,
                     args.point_size, args.alpha, title)
    else:
        heatmap_plot(xs, ys, labels, xdim, ydim, args.out, args.dpi, title)


if __name__ == "__main__":
    main()
