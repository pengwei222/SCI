"""Draw the exact scalar SCI example at its final typeset size."""

from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("redrawn"))
output = parser.parse_args().output
output.mkdir(parents=True, exist_ok=True)
width, height = 125.5, 88.2
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
})
fig = plt.figure(figsize=(width / 72, height / 72), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
ax.axis("off")

ink, slate, blue = "#232B31", "#63717A", "#205987"
cx, base_y, side = width / 2, 12.1, 72.8
left = (cx - side / 2, base_y)
right = (cx + side / 2, base_y)
top = (cx, base_y + sqrt(3) * side / 2)
ax.add_patch(Polygon([left, right, top], closed=True, facecolor="#F1F5F7",
                     edgecolor="#84939D", linewidth=0.7, joinstyle="round"))

def point(t):
    weights = (t, 0.5 - t, 0.5)
    vertices = (left, right, top)
    return tuple(sum(w * v[i] for w, v in zip(weights, vertices)) for i in (0, 1))

a, b = point(0.5), point(0)
level = a[1]
ax.plot([a[0], b[0]], [level, level], color=blue, lw=1.9, solid_capstyle="round")
for (x, y), direction in ((a, -1), (b, 1)):
    ax.plot([x + direction * 1.7, x + direction * 6.0], [y, y],
            color=slate, lw=0.55)
    ax.plot(x, y, "o", markersize=3.4, markeredgewidth=0.9,
            markeredgecolor=blue, markerfacecolor="white")

def label(x, y, s, size=8.3, color=ink, align="center"):
    ax.text(x, y, s, fontsize=size, color=color, ha=align, va="center")

label(cx, top[1] + 6.5, r"$w_3=1$")
label(left[0], base_y - 7.1, r"$w_1=1$")
label(right[0], base_y - 7.1, r"$w_2=1$")
label(cx, level - 8.4, r"$P=8/3$", 8.7, blue)
for (x, y), direction, t, estimate in (
        (a, -1, "1/2", "2/3"), (b, 1, "0", "5/3")):
    center = x + direction * 23.6
    label(center, y + 4.9, rf"$t={t}$", 8.0, slate)
    label(center, y - 4.4, rf"$\hat{{x}}={estimate}$", 9.0)

fig.savefig(output / "flat_geometry.pdf", metadata={"Title": "Optimal SCI weights and fused estimates"})
