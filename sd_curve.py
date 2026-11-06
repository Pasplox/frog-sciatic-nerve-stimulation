import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
IMG_DIR  = os.path.join(os.path.dirname(__file__), "images")

os.makedirs(IMG_DIR, exist_ok=True)

# Load strength-duration data from exp3.csv
duration_ms = []
current_uA  = []

with open(os.path.join(DATA_DIR, "exp3.csv"), newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # CSV labels duration as "ms" but values are in µs (confirmed by source xlsx)
        duration_ms.append(float(row["Duration (ms)"]) / 1000)
        current_uA.append(float(row["Current(µA)"]))

duration_ms = np.array(duration_ms)
current_uA  = np.array(current_uA)

# Sort by ascending duration for a clean curve
order      = np.argsort(duration_ms)
duration_ms = duration_ms[order]
current_uA  = current_uA[order]

# Fit Weiss/Lapicque model: I = I_rh * (1 + t_c / t)
def sd_model(d, b, c):
    return b * (1 + c / d)

popt, _ = curve_fit(sd_model, duration_ms, current_uA, p0=[30, 100])
b, c = popt

t_fit   = np.linspace(duration_ms.min(), duration_ms.max(), 300)
I_fit   = sd_model(t_fit, b, c)

fig, ax = plt.subplots(figsize=(6, 4))

ax.scatter(duration_ms, current_uA, color="#eb6834", zorder=3, label="Data")
ax.plot(t_fit, I_fit, color="#333333", linewidth=1.5,
        label=r"$I = b\left(1 + \frac{c}{d}\right)$")

ax.set_xlabel("Pulse Duration (ms)")
ax.set_ylabel("Threshold Current (µA)")
ax.set_title("Strength-Duration Curve — Frog Sciatic Nerve")
ax.axhline(b, linestyle=":", color="#888888", linewidth=1.2,
           label=f"Rheobase (b) = {b:.1f} µA")
ax.axvline(c, linestyle=":", color="#4a90d9", linewidth=1.2,
           label=f"Chronaxie (c) = {c:.3f} ms")
ax.legend(fontsize=9)

ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
fig.tight_layout()

out_path = os.path.join(IMG_DIR, "sd_curve.png")
fig.savefig(out_path, dpi=300)
plt.close(fig)

print(f"Saved: {out_path}")
