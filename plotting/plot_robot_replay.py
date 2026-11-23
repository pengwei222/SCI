"""Plot the first two minutes of the retained UTIAS Dataset 4 SCI inputs.

Run: python plot_robot_replay.py --archive /path/to/SCI_Experiments_Revised --out .
Dependencies: NumPy and Matplotlib. Every quantity is read from the archive.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np


def read_inputs(root):
    with np.load(root / "diagnostics/data/d4_reference.npz") as archive:
        epochs = archive["epoch"].copy()
        paths = archive["truth"].reshape(-1, 5, 3)[:, :, :2].copy()
    assert np.array_equal(epochs, np.arange(30, 1381, 5))

    paired_path = root / "expanded/results/recorded_additional/paired.csv"
    with paired_path.open(newline="") as stream:
        paired = [r for r in csv.DictReader(stream)
                  if r["dataset"] == "4" and float(r["tolerance"]) == .001
                  and r["contrast"] == "complete" and r["method"] == "newton-cheap"]
    paired_by_epoch = {int(row["epoch"]): row for row in paired}
    assert len(paired_by_epoch) == len(paired) == len(epochs)
    assert set(paired_by_epoch) == set(epochs)

    times, weights, original_ms, complete_ms = [], [], [], []
    for epoch in epochs[:25]:
        row = paired_by_epoch[epoch]
        assert row["seed"] == "42" and row["paired_success"] == "True"
        assert int(float(row["repeats"])) == 3
        file = root / "expanded/inputs" / (row["case_id"] + ".npz")
        with np.load(file, allow_pickle=False) as archive:
            meta = json.loads(archive["metadata"].item())
            assert meta["dataset"] == 4 and meta["seed"] == 42
            assert meta["case_id"] == row["case_id"] and meta["epoch"] == epoch
            weights.append(archive["ref_w"].copy())
            times.append(meta["time"])
        complete = 1000.0 * float(row["end_to_end_seconds"])
        complete_ms.append(complete)
        original_ms.append(complete * float(row["paired_speedup"]))
    weights = np.asarray(weights)
    original_ms, complete_ms = np.asarray(original_ms), np.asarray(complete_ms)
    assert np.isfinite(paths).all() and np.isfinite(weights).all()
    assert np.all(weights >= 0) and np.allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-14)
    assert np.array_equal(times, np.arange(31., 152., 5.))
    assert np.all(np.isfinite(original_ms)) and np.all(np.isfinite(complete_ms))
    assert np.all(original_ms > 0) and np.all(complete_ms > 0)
    # The compact plot data include the paired table; in the full archive we
    # additionally check both traces against the independent per-method table.
    methods_path = paired_path.with_name("per_case.csv")
    if methods_path.exists():
        observed = {"original-fw": {}, "newton-cheap": {}}
        with methods_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                epoch = int(row["epoch"])
                if (row["method"] in observed and row["dataset"] == "4" and row["seed"] == "42"
                        and float(row["tolerance"]) == .001 and epoch in paired_by_epoch):
                    observed[row["method"]][epoch] = 1000.0 * float(row["end_to_end_seconds"])
        for method, times_ms in (("original-fw", original_ms), ("newton-cheap", complete_ms)):
            np.testing.assert_allclose([observed[method][e] for e in epochs[:25]], times_ms,
                                       rtol=1e-12, atol=0)
    return np.asarray(times), paths[:25], weights, original_ms, complete_ms


INK = "#202833"
GRID = "#dae2e8"
ROBOT_COLORS = ["#1865a0", "#c45400", "#18826d", "#9666a4", "#98780c"]


def label(fig, x, y, letter, title):
    fig.text(x, y, letter, fontsize=10.5, fontweight="bold", va="baseline")
    fig.text(x+.027, y, title, fontsize=9, fontweight="bold", va="baseline")


def plot(root, out):
    times, paths, weights, original_ms, complete_ms = read_inputs(root)
    dt = times - times[0]
    fig = plt.figure(figsize=(7.12, 3.0))
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.6,
        "mathtext.fontset": "dejavusans", "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": "#71808b",
        "axes.linewidth": .55, "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": .5, "ytick.major.width": .5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    layout = fig.add_axes([.064, .17, .241, .72])
    for robot, color in enumerate(ROBOT_COLORS):
        route = paths[:, robot]
        layout.plot(route[:, 0], route[:, 1], color=color, linewidth=1.15,
                    marker=".", markersize=2.2, solid_capstyle="round")
        layout.plot(*route[-1], marker="o", color=color, ms=3, zorder=3)
        layout.annotate(str(robot+1), route[0], xytext=(0, 0),
                        textcoords="offset points", va="center", ha="center",
                        fontsize=6.5, fontweight="bold", zorder=5, color=color,
                        bbox=dict(boxstyle="circle,pad=.18", facecolor="white",
                                  edgecolor=color, linewidth=.8))
    layout.set(xlabel="$x$ (m)", ylabel="$y$ (m)", aspect="equal",
               xlim=(-.15, 5.0), ylim=(-5., 4.05),
               xticks=[0, 2, 4], yticks=[-4, -2, 0, 2, 4])
    for coordinate, bounds in enumerate((layout.get_xlim(), layout.get_ylim())):
        assert paths[...,coordinate].min() > bounds[0] and paths[...,coordinate].max() < bounds[1]
    layout.spines[["top", "right"]].set_visible(False)
    layout.grid(color=GRID, linewidth=.4, zorder=-5)
    layout.set_axisbelow(True)
    label(fig, .027, .935, "a", "Ground-truth paths")

    x0, width = .464, .510
    heat = fig.add_axes([x0, .495, width, .338])
    cmap = LinearSegmentedColormap.from_list(
        "branch_weight", ["#f9f7ee", "#c5e0d1", "#6cb3ae", "#287b8b", "#174566"])
    colors = Normalize(vmin=0, vmax=.55)
    xedges = np.arange(-2.5, 123, 5)
    mesh = heat.pcolormesh(xedges, np.arange(6), weights.T, cmap=cmap,
                           norm=colors, linewidth=.3, edgecolors="white")
    heat.set_xlim(xedges[[0,-1]])
    heat.set_ylim(5,0)
    heat.set_yticks(np.arange(5)+.5, [f"Robot {i}" for i in range(1,6)])
    for tick, color in zip(heat.get_yticklabels(), ROBOT_COLORS):
        tick.set_color(color)
        tick.set_fontweight("bold")
    heat.tick_params(length=0, bottom=False, labelbottom=False, pad=4)
    heat.spines[:].set_visible(False)
    label(fig, .350, .935, "b", "Reference fusion weights")
    key = fig.add_axes([.824,.916,.15,.022])
    cb = fig.colorbar(mesh, cax=key, orientation="horizontal", ticks=[0,.25,.5])
    cb.set_ticklabels(["0","0.25","0.50"])
    cb.ax.tick_params(length=2, pad=2, labelsize=6.8)
    cb.outline.set_visible(False)

    panel = fig.add_axes([x0, .157, width, .194], sharex=heat)
    panel.plot(dt, original_ms, color="#586574", linewidth=1.0, marker="s",
               markersize=2.8, markerfacecolor="white", markeredgewidth=.65,
               linestyle=(0,(3,1.6)), label="Original")
    panel.plot(dt, complete_ms, color="#16756F", linewidth=1.35, marker="o",
               markersize=3, markeredgewidth=0, label="Complete")
    panel.set(xlim=xedges[[0,-1]], ylim=(0,8.2), xticks=[0,30,60,90,120],
              yticks=[0,2,4,6,8], ylabel="Time (ms)",
              xlabel="Time since first retained epoch (s)")
    panel.yaxis.grid(True, color=GRID, lw=.45)
    panel.set_axisbelow(True)
    panel.spines[["top","right"]].set_visible(False)
    panel.tick_params(axis="y", pad=2, labelsize=7.2)
    label(fig, .350, .400, "c", "End-to-end runtime")
    fig.legend(*panel.get_legend_handles_labels(), loc="center right",
               bbox_to_anchor=(.976, .412), ncol=2, frameon=False,
               borderaxespad=0, columnspacing=.85, handletextpad=.35,
               handlelength=1.75, fontsize=7.2)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "robot_replay_window.pdf", metadata={"Title":"Five-robot SCI replay, Dataset 4"})
    fig.savefig(out / "robot_replay_window.png", dpi=220)
    plt.close(fig)
    print("D4 seed 42 | epochs 30:5:150 | 25/271 paired inputs")
    for label_, values in (("Original (ms)", original_ms), ("Complete (ms)", complete_ms),
                           ("Paired speedup", original_ms / complete_ms)):
        print(f"{label_}: min {values.min():.4f}, median {np.median(values):.4f}, max {values.max():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot(args.archive, args.out)
