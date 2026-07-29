"""
Plot: APC beta vs Mean Jaccard Similarity (APC vs Greedy) for Qwen2.5-VL-7B.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "jaccard_apc_vs_greedy.csv")
OUT_PATH   = os.path.join(SCRIPT_DIR, "jaccard_apc_vs_greedy.png")

df = pd.read_csv(CSV_PATH)

fig, ax = plt.subplots(figsize=(8, 5))

# Main line + markers
ax.plot(df["beta"], df["mean_jaccard"],
        color="#2563EB", linewidth=2.2, zorder=3,
        marker="o", markersize=7, markerfacecolor="white",
        markeredgecolor="#2563EB", markeredgewidth=2)

# Shaded area under curve
ax.fill_between(df["beta"], df["mean_jaccard"],
                alpha=0.10, color="#2563EB")

# Annotate start and end points
ax.annotate(f'β=0\n({df["mean_jaccard"].iloc[0]:.3f})',
            xy=(df["beta"].iloc[0], df["mean_jaccard"].iloc[0]),
            xytext=(0.04, df["mean_jaccard"].iloc[0] - 0.05),
            fontsize=9, color="#1e40af",
            arrowprops=dict(arrowstyle="->", color="#1e40af", lw=1.2))

ax.annotate(f'β=1\n({df["mean_jaccard"].iloc[-1]:.3f})',
            xy=(df["beta"].iloc[-1], df["mean_jaccard"].iloc[-1]),
            xytext=(0.82, df["mean_jaccard"].iloc[-1] + 0.03),
            fontsize=9, color="#1e40af",
            arrowprops=dict(arrowstyle="->", color="#1e40af", lw=1.2))

# Reference: sample-level baseline (β=0 value as "random sampling" baseline)
ax.axhline(y=df["mean_jaccard"].iloc[0], color="gray",
           linestyle="--", linewidth=1.2, alpha=0.6, label="Sampling baseline (β=0)")

# Axes
ax.set_xlabel("APC Threshold β", fontsize=13, labelpad=8)
ax.set_ylabel("Mean Jaccard Similarity\n(APC vs Greedy)", fontsize=13, labelpad=8)
ax.set_title("APC Converges to Greedy Search as β Increases\n"
             "Qwen2.5-VL-7B · LLaVA-Bench (60 questions)",
             fontsize=13, fontweight="bold", pad=12)

ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0.30, 0.85)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)

ax.legend(fontsize=10, loc="upper left")
ax.tick_params(labelsize=10)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
