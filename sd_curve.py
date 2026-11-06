import csv
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
IMG_DIR  = os.path.join(os.path.dirname(__file__), "images")

os.makedirs(IMG_DIR, exist_ok=True)

# Load strength-duration data from exp3.csv
duration_ms = []
current_uA  = []

with open(os.path.join(DATA_DIR, "exp3.csv"), newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        duration_ms.append(float(row["Duration (ms)"]))
        current_uA.append(float(row["Current(µA)"]))

duration_ms = np.array(duration_ms)
current_uA  = np.array(current_uA)

# Sort by ascending duration for a clean curve
order      = np.argsort(duration_ms)
duration_ms = duration_ms[order]
current_uA  = current_uA[order]

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(duration_ms, current_uA, marker="o", linewidth=1.5,
        markersize=5, color="#eb6834")

ax.set_xlabel("Pulse Duration (ms)")
ax.set_ylabel("Threshold Current (µA)")
ax.set_title("Strength-Duration Curve — Frog Sciatic Nerve")

ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
fig.tight_layout()

out_path = os.path.join(IMG_DIR, "sd_curve.png")
fig.savefig(out_path, dpi=300)
plt.close(fig)

print(f"Saved: {out_path}")
