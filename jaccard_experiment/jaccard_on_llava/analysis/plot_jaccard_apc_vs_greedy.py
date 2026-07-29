"""
Plot BPE-token Jaccard similarity (APC vs Greedy) as a function of APC beta.
"""

import os
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "jaccard_apc_vs_greedy.csv")
OUT_PATH   = os.path.join(SCRIPT_DIR, "jaccard_apc_vs_greedy.png")

# ── load data ──────────────────────────────────────────────────────────────────
betas, jaccards = [], []
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        betas.append(float(row["beta"]))
        jaccards.append(float(row["mean_jaccard"]))

# ── plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(betas, jaccards, marker="o", linewidth=2, markersize=6,
        color="#2563EB", markerfacecolor="white", markeredgewidth=2)

# annotate β=0.1 (VCD default) and β=1.0 (greedy endpoint)
for beta, jac, label, ha in [
    (0.1,  jaccards[betas.index(0.1)],  "VCD default\n(β=0.1)",  "left"),
    (1.0,  jaccards[betas.index(1.0)],  "β=1.0\n(≈ Greedy)",     "right"),
]:
    ax.annotate(
        label,
        xy=(beta, jac),
        xytext=(beta + (0.05 if ha == "left" else -0.05), jac - 0.07),
        fontsize=8.5,
        ha=ha,
        color="#374151",
        arrowprops=dict(arrowstyle="->", color="#9CA3AF", lw=1.2),
    )

# reference lines
ax.axhline(jaccards[betas.index(0.0)], color="#9CA3AF", linestyle="--",
           linewidth=1, label=f"β=0 baseline (direct sampling ≈ {jaccards[betas.index(0.0)]:.3f})")
ax.axhline(1.0, color="#6EE7B7", linestyle="--",
           linewidth=1, label="Perfect greedy match (1.0)")

ax.set_xlabel("APC Beta (β)", fontsize=12)
ax.set_ylabel("Mean BPE-Token Jaccard Similarity", fontsize=12)
ax.set_title(
    "APC β vs. Jaccard Similarity with Greedy Search\n"
    "LLaVA-v1.5-7B · LLaVA-Bench (In-the-Wild) · BPE Tokenizer · 60 Questions",
    fontsize=12,
    fontweight="bold",
    pad=12,
)

ax.set_xlim(-0.03, 1.05)
ax.set_ylim(0.28, 1.02)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=9, loc="upper left")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
