"""Draw the one-column synthetic comparison from the archived trial data."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np


REFERENCE_SIZES = {"True": 95, "False": 97}
ETAS = np.array([1.0, 0.1, 0.01, 0.001, 0.0001])
SCALES = [
    ("vertex", "gq", "Vertex scale", "#245B86", "o", "-"),
    ("cheap", "gq_cheap", "Inexpensive scale", "#BC5A2D", "s", (0, (3, 2))),
]


def collect(archive):
    data = archive / "synthetic" / "data"
    with (data / "certificate_results.csv").open(newline="") as stream:
        trials = list(csv.DictReader(stream))
    if len(trials) != 960:
        raise ValueError("Expected 192 problems at five candidate weights each")
    ratios = defaultdict(list)
    keys = set()
    for row in trials:
        group, eta = row["interior"], float(row["eta"])
        key = (row["index"], row["mode"], eta)
        if key in keys:
            raise ValueError(f"Repeated candidate design point: {key}")
        keys.add(key)
        first_order = float(row["gfw"])
        for rule, column, *_ in SCALES:
            quadratic = float(row[column])
            value = 1.0 if first_order == quadratic == 0 else first_order / quadratic
            if not np.isfinite(value) or value < 1 - 1e-10:
                raise ValueError(f"Unexpected certificate ratio at {key}")
            ratios[group, eta, rule].append(value)
    quantiles = {}
    for group, expected in REFERENCE_SIZES.items():
        for eta in ETAS:
            for rule, *_ in SCALES:
                values = ratios[group, eta, rule]
                if len(values) != expected:
                    raise ValueError(f"Expected {expected} problems for {group}, {eta}")
                quantiles[group, eta, rule] = np.quantile(values, [.25, .5, .75])
    with (data / "certificate_summary.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            for rule, *_ in SCALES:
                actual = quantiles[row["interior"], float(row["eta"]), rule]
                np.testing.assert_allclose(actual[1], float(row["fw_over_" + rule]), rtol=2e-12)
                if rule == "vertex":
                    np.testing.assert_allclose(actual[[0, 2]], [
                        float(row["fw_over_vertex_q25"]),
                        float(row["fw_over_vertex_q75"])
                    ], rtol=2e-12)
    return quantiles


def draw(quantiles, output):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.edgecolor": "#52606A", "axes.linewidth": .55,
        "xtick.color": "#35424B", "ytick.color": "#35424B", "text.color": "#24313A",
        "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "dejavusans",
    })
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(3.47, 3.06),
                            gridspec_kw={"height_ratios": [1.16, 1]})
    fig.subplots_adjust(left=.205, right=.97, bottom=.158, top=.827, hspace=.61)
    for ax, (group, count), title, letter in zip(
            axes, REFERENCE_SIZES.items(), ("Interior references", "Boundary references"), "AB"):
        ax.set_xscale("log")
        ax.set_xlim(1.40, .0000715)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linewidth=.5, color="#DCE1E5")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.7, width=.55, pad=2.5)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_title(f"{letter}   {title}", loc="left", fontsize=8.2, weight="bold", pad=4.5)
        ax.text(1, 1.055, f"$n={count}$", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5, color="#4B5A64")
        for rule, _, label, color, marker, linestyle in SCALES:
            q1, med, q3 = np.array([quantiles[group, eta, rule] for eta in ETAS]).T
            ax.fill_between(ETAS, q1, q3, color=color, alpha=.16, linewidth=0, zorder=1)
            ax.plot(ETAS, med, label=label, color=color, linestyle=linestyle,
                    linewidth=1.15, marker=marker, markersize=3.3, markeredgewidth=.85,
                    markerfacecolor=("white" if rule == "vertex" else color), zorder=3)
    axes[0].set_yscale("log")
    axes[0].set_yticks([1, 100, 10000], labels=["1", r"$10^2$", r"$10^4$"])
    axes[0].set_ylim(.95, 69000)
    axes[0].yaxis.set_minor_locator(NullLocator())
    axes[1].set_ylim(.93, 3.05)
    axes[1].set_yticks([1, 2, 3])
    axes[1].set_xticks(ETAS, labels=["1", r"$10^{-1}$", r"$10^{-2}$",
                                     r"$10^{-3}$", r"$10^{-4}$"])
    axes[1].set_xlabel(r"Approach parameter $\eta$", labelpad=4)
    fig.text(.018, .50, r"Certificate gain $R$", rotation=90, ha="left", va="center")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, 1),
               ncol=2, frameon=False, fontsize=7.8, handlelength=2.2,
               columnspacing=1.7, borderaxespad=.2, handletextpad=.55)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"Title": "Synthetic certificate gains by reference group"})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Extracted SCI archive root")
    parser.add_argument("--output", type=Path, required=True, help="Output PDF path")
    args = parser.parse_args()
    draw(collect(args.archive), args.output)


if __name__ == "__main__":
    main()
