"""ring figures: the ring topology + measured busbw/algbw curves."""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIG = os.path.join(ROOT, "assets")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 140, "font.size": 9})

# Fig 1: the ring, one reduce-scatter step annotated
fig, ax = plt.subplots(figsize=(5.2, 5.2))
N = 4
pos = [(math.cos(2 * math.pi * i / N - math.pi / 2), math.sin(2 * math.pi * i / N - math.pi / 2))
       for i in range(N)]
for i, (x, y) in enumerate(pos):
    ax.add_patch(Rectangle((x - .32, y - .22), .64, .44, facecolor="#eef2fa", edgecolor="#2255a8", lw=1.4))
    ax.text(x, y + .06, f"rank {i}", ha="center", fontsize=9)
    ax.text(x, y - .12, f"buf[{i}] +=", ha="center", fontsize=6.5, color="#555")
for i in range(N):
    a, b = pos[i], pos[(i + 1) % N]
    ax.add_patch(FancyArrowPatch((a[0] * .78, a[1] * .78), (b[0] * .78, b[1] * .78),
                                 arrowstyle="-|>", mutation_scale=14, color="#c0504d", lw=1.6))
ax.text(0, 1.35, "step s: rank i sends buf[(i−1−s) mod N] → i+1", ha="center", fontsize=8.5)
ax.text(0, -1.42, "after N−1 steps: rank i holds FULL sum of chunk i", ha="center", fontsize=8.5, color="#2e6b2f")
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.6, 1.55)
ax.axis("off")
ax.set_aspect("equal")
ax.set_title("ring reduce-scatter, one step", fontsize=10)
fig.tight_layout()
fig.savefig(f"{FIG}/ring_step.png")
plt.close(fig)

# Fig 2: measured curves from the receipt
rec = json.load(open(f"{ROOT}/receipts/receipt.json"))
m = rec["measurements"]
Ns = [x["N"] for x in m]
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(Ns, [x["algbw_gbps"] for x in m], "o-", color="#c0504d", label="algbw (S/t)")
ax.plot(Ns, [x["busbw_gbps"] for x in m], "s-", color="#4472c4", label="busbw = algbw·2(N−1)/N")
ax.set_xlabel("ring size N")
ax.set_ylabel("Gbit/s (2 MiB allreduce, loopback TCP)")
ax.set_title("Local TCP throughput and formula normalization")
ax.legend()
ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(f"{FIG}/busbw.png")
plt.close(fig)
print("ring figures ->", sorted(os.listdir(FIG)))
