# EXPERIMENT 1 — Compound Action Potential (CAP) Analysis
# Loads five graded-stimulus trials of frog sciatic nerve recordings,
# computes the CAP amplitude for each, and saves a figure of all trials.
#
# Each trial was recorded with LabVIEW at 50 kHz (Δt = 20 µs).
# Column layout in every CSV: time (s) | stimulus pulse (V) | nerve voltage (V)
#
# CAP amplitude is defined as:
#   baseline (mean of 50 ms pre-stimulus) − trough (minimum within 5 ms post-stimulus)
# A larger amplitude means more axons were recruited by that stimulus intensity.

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

COLOR_NERVE = "#eb6834"


def load_labview_tsv(path):
    """Return (time, stimulus, nerve) arrays from a LabVIEW tab-separated file.

    LabVIEW TSV files contain two header blocks separated by ***End_of_Header***
    markers before the numeric data begins. The column order is:
        X_Value (time, s) | channel 1 (stimulus, V) | channel 2 (nerve, V)
    """
    rows = []
    end_of_header_count = 0
    with open(path, "r") as f:
        for line in f:
            if "***End_of_Header***" in line:
                end_of_header_count += 1
                continue
            if end_of_header_count < 2:          # skip both header blocks
                continue
            if not rows and line.strip().startswith("X_Value"):
                continue                          # skip column-name row
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


# ── Load files ────────────────────────────────────────────────────────────────
trial_files = sorted(glob.glob(os.path.join(DATA_DIR, "exp1_trial*.csv")))

# ── Figure setup: one subplot per trial ───────────────────────────────────────
n = len(trial_files)
fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), constrained_layout=True)
fig.patch.set_facecolor("#fcfcfb")

if n == 1:
    axes = [axes]

for ax, path in zip(axes, trial_files):
    trial_name = os.path.splitext(os.path.basename(path))[0]
    trial_num  = trial_name.split("trial")[-1]

    time, stimulus, nerve = load_labview_tsv(path)

    # ── Locate the stimulus artifact ──────────────────────────────────────────
    # The stimulus artifact appears as a large positive transient on the nerve
    # channel due to capacitive coupling; it marks the stimulus onset time.
    max_idx = np.argmax(nerve)
    max_t   = time[max_idx] * 1e3              # convert to ms for plotting

    # ── Baseline ──────────────────────────────────────────────────────────────
    # Average the 50 ms of quiescent signal immediately before the stimulus.
    # This accounts for any DC offset on the recording electrode.
    dt_s       = time[1] - time[0]             # sample period (s)
    win_pre    = int(0.050 / dt_s)             # 50 ms → number of samples
    base_start = max(0, max_idx - win_pre)
    baseline   = np.mean(nerve[base_start:max_idx])

    # ── CAP trough ────────────────────────────────────────────────────────────
    # The CAP arrives as a negative deflection 1–5 ms after the stimulus.
    # Search the 5 ms window immediately following the artifact peak.
    win_post = int(0.005 / dt_s)               # 5 ms → number of samples
    post_end = min(len(nerve), max_idx + win_post)
    cap_idx  = max_idx + np.argmin(nerve[max_idx:post_end])
    cap_val  = nerve[cap_idx]
    cap_t    = time[cap_idx] * 1e3             # ms
    cap_amp  = baseline - cap_val              # positive value; trough < baseline

    print(f"Trial {trial_num}: baseline = {baseline:.6f} V | "
          f"CAP trough = {cap_val:.6f} V at {cap_t:.3f} ms | "
          f"CAP amplitude = {cap_amp*1e3:.4f} mV")

    # ── Plot ──────────────────────────────────────────────────────────────────
    ax.plot(time * 1e3, nerve, color=COLOR_NERVE, linewidth=1.5, zorder=2)
    ax.axhline(baseline, color="#52514e", linewidth=0.8, linestyle="--",
               zorder=1, label=f"Baseline {baseline*1e3:.2f} mV")
    ax.plot(cap_t, cap_val, "o", color="#2a78d6", markersize=6, zorder=4,
            label=f"CAP amp {cap_amp*1e3:.4f} mV")
    ax.set_xlim(max_t - 5, max_t + 5)         # ±5 ms window around stimulus

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
    ax.legend(fontsize=8, framealpha=0.85, edgecolor="#d4d3cc",
              facecolor="#fcfcfb", labelcolor="#0b0b0b")

fig.suptitle("Experiment 1 — Frog Sciatic Nerve CAP Trials",
             fontsize=13, fontweight="bold", color="#0b0b0b", y=1.01)

# ── Save ──────────────────────────────────────────────────────────────────────
img_dir = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(img_dir, exist_ok=True)
out_path = os.path.join(img_dir, "exp1_trials.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out_path}")
