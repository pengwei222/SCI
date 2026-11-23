"""Rebuild the two-column MRCLAM certificate gain figure from saved evaluations."""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullLocator
import numpy as np


ETAS = np.array([.1, .03, .01, .003, .001, .0003, .0001])
COUNTS = {(2, False): 94, (3, False): 165, (2, True): 270, (3, True): 717}
STYLES = {
    "vertex": ("#245B86", "o", "-", "Vertex scale"),
    "cheap": ("#BC5A2D", "s", (0, (3.1, 1.7)), "Inexpensive scale"),
}


def group_evaluations(results):
    groups, counts = defaultdict(list), Counter()
    seen, failures = set(), 0
    with (results / "same_point_tightness.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            dataset, boundary = int(row["dataset"]), row["reference_boundary"] == "True"
            eta, rule = float(row["eta"]), row["rule"]
            key = (dataset, int(row["seed"]), int(row["epoch"]), eta, rule)
            if key in seen:
                raise ValueError(f"Repeated epoch/perturbation/method: {key}")
            seen.add(key)
            raw_gain = float(row["tightening_ratio"])
            fw, effective = float(row["fw_gap"]), float(row["effective_gap"])
            expected_gain = 1. if fw == effective == 0. else fw / effective
            np.testing.assert_allclose(raw_gain, expected_gain, rtol=1e-12)
            if row["certificate_failed"] == "True":
                assert rule == "vertex" and raw_gain == 1.
                failures += 1
            groups[dataset, boundary, rule, eta].append(raw_gain)
            counts[dataset, boundary, rule, eta] += 1
    assert len(seen) == 17444 and failures == 14
    summary = {}
    for case, n in COUNTS.items():
        for rule in STYLES:
            for eta in ETAS:
                key = (*case, rule, eta)
                assert counts[key] == n
                summary[key] = np.quantile(groups[key], [.25, .5, .75])
    with (results / "tightness_summary.csv").open(newline="") as stream:
        archived = list(csv.DictReader(stream))
    assert len(archived) == len(summary)
    for row in archived:
        key = (int(row["dataset"]), row["reference_boundary"] == "True",
               row["rule"], float(row["eta"]))
        np.testing.assert_allclose(summary[key][1], float(row["median_tightening"]),
                                   rtol=1e-12)
    return summary


def draw(summary, target):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.0, "axes.labelsize": 8.0,
        "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.edgecolor": "#64717A", "axes.linewidth": .58,
        "xtick.color": "#37454F", "ytick.color": "#37454F",
        "text.color": "#25333D", "axes.labelcolor": "#25333D",
        "mathtext.fontset": "dejavusans", "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.10, 2.26), facecolor="white")
    lefts = [.085, .315, .572, .800]
    width = .18
    for i, ((dataset, boundary), n) in enumerate(COUNTS.items()):
        ax = fig.add_axes((lefts[i], .249, width, .52))
        ax.set_xscale("log")
        ax.set_xlim(.135, .000071)
        ax.xaxis.set_major_locator(FixedLocator([.1, .01, .001, .0001]))
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.xaxis.set_minor_locator(NullLocator())
        if boundary:
            ax.set_ylim(.94, 2.83)
            ax.set_yticks([1, 1.5, 2, 2.5])
            ax.set_yticklabels(["1", "1.5", "2", "2.5"])
            ax.axhline(1., color="#798791", lw=.62, zorder=1, ls=(0, (2, 2)))
        else:
            ax.set_yscale("log")
            ax.set_ylim(1.0, 1350.)
            ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
            ax.yaxis.set_major_formatter(LogFormatterMathtext())
            ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(axis="y", color="#DCE2E6", lw=.55, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="major", width=.55, length=2.2, pad=2.)
        if i % 2 == 1:
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.spines["left"].set_visible(False)
        ax.text(.5, 1.045, f"D{dataset}  ·  {n} inputs", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7.7)
        for rule, (color, marker, linestyle, _) in STYLES.items():
            lo, med, hi = np.array([summary[dataset, boundary, rule, eta] for eta in ETAS]).T
            ax.fill_between(ETAS, lo, hi, color=color, alpha=.15, edgecolor="none", zorder=1)
            ax.plot(ETAS, med, color=color, ls=linestyle, lw=1.22, marker=marker,
                    ms=3.1, mfc="white" if rule == "vertex" else color,
                    mec=color, mew=.75, zorder=3)
    for center, label in [(.290, "Interior reference weights"), (.776, "Boundary reference weights")]:
        fig.text(center, .952, label, ha="center", va="center", weight="bold", size=8.8)
    fig.text(.013, .51, "Certificate gain", rotation=90, ha="center", va="center", size=8.2)
    fig.text(.527, .097, r"Perturbation parameter $\eta$", ha="center", va="center", size=8)
    handles = [Line2D([], [], label=label, linewidth=1.25, color=c, marker=m, ls=ls,
                      ms=3.6, mfc="white" if rule == "vertex" else c, mew=.75)
               for rule, (c, m, ls, label) in STYLES.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, -.009),
               frameon=False, ncol=2, columnspacing=1.9, handlelength=2.3,
               handletextpad=.5, borderpad=.05, fontsize=7.7)
    fig.savefig(target, metadata={"Title": "Certificate gain on MRCLAM fusion inputs"})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path,
                        help="Experiment archive with expanded/certificate_extension/results/")
    parser.add_argument("--output", required=True, type=Path, help="Output vector PDF")
    args = parser.parse_args()
    data = group_evaluations(args.archive / "expanded/certificate_extension/results")
    draw(data, args.output)
