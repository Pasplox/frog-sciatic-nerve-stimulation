import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

COLOR_NERVE = "#eb6834"


def load_labview_tsv(path):
    """Return (time, stimulus, nerve) arrays from a LabVIEW tab-separated file."""
    rows = []
    end_of_header_count = 0
    with open(path, "r") as f:
        for line in f:
            if "***End_of_Header***" in line:
                end_of_header_count += 1
                continue
            if end_of_header_count < 2:
                continue
            # First content line after second header is the column-name row
            if not rows and line.strip().startswith("X_Value"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                t  = float(parts[0])
                s  = float(parts[1])
                nv = float(parts[2])
                rows.append((t, s, nv))
            except ValueError:
                continue
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1], arr[:, 2]


trial_files = sorted(glob.glob(os.path.join(DATA_DIR, "exp1_trial*.csv")))

n = len(trial_files)
fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), constrained_layout=True)
fig.patch.set_facecolor("#fcfcfb")

if n == 1:
    axes = [axes]

for ax, path in zip(axes, trial_files):
    trial_name = os.path.splitext(os.path.basename(path))[0]  # e.g. exp1_trial1
    trial_num  = trial_name.split("trial")[-1]

    time, stimulus, nerve = load_labview_tsv(path)

    ax.plot(time * 1e3, nerve, color=COLOR_NERVE, linewidth=1.5,
            label="Recorded Nerve Voltage", zorder=2)

    ax.set_facecolor("#fcfcfb")
    ax.set_title(f"Trial {trial_num}", fontsize=11, fontweight="semibold",
                 color="#0b0b0b", loc="left", pad=6)
    ax.set_xlabel("Time (ms)", fontsize=9, color="#52514e")
    ax.set_ylabel("Voltage (V)", fontsize=9, color="#52514e")
    ax.tick_params(colors="#52514e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#d4d3cc")
        spine.set_linewidth(0.8)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which="major", color="#e8e7e1", linewidth=0.6, zorder=0)
    ax.grid(which="minor", color="#f0efeb", linewidth=0.3, zorder=0)


fig.suptitle("Experiment 1 — Frog Sciatic Nerve CAP Trials",
             fontsize=13, fontweight="bold", color="#0b0b0b", y=1.01)

img_dir = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(img_dir, exist_ok=True)
out_path = os.path.join(img_dir, "exp1_trials.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out_path}")
