import csv
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "ncv_results.csv")

TRIAL_METADATA = {
    1: {"current_uA": 100, "distance_mm": 21},
    2: {"current_uA": 100, "distance_mm": 35},
    3: {"current_uA": 150, "distance_mm": 35},
    4: {"current_uA": 200, "distance_mm": 35},
    5: {"current_uA": 200, "distance_mm": 21},
    6: {"current_uA": 150, "distance_mm": 21},
    7: {"current_uA": 175, "distance_mm": 21},
}


def get_trial_file(trial_num):
    path = os.path.join(DATA_DIR, f"exp2_trial{trial_num}.csv")
    if os.path.exists(path):
        return path
    # trial 1 is stored as exp2.csv
    if trial_num == 1:
        fallback = os.path.join(DATA_DIR, "exp2.csv")
        if os.path.exists(fallback):
            return fallback
    return None


def parse_labview_csv(filepath):
    """Return lists of (time, stim, loc1, loc2) from a LabVIEW tab-separated file."""
    time_col, stim_col, loc1_col, loc2_col = [], [], [], []

    with open(filepath, "r") as f:
        end_of_header_count = 0
        past_headers = False
        skip_col_header = False

        for line in f:
            line = line.rstrip("\n")

            if "***End_of_Header***" in line:
                end_of_header_count += 1
                if end_of_header_count == 2:
                    skip_col_header = True  # next non-blank line is column names
                continue

            if end_of_header_count < 2:
                continue

            if skip_col_header:
                skip_col_header = False
                past_headers = True
                continue  # skip the "X_Value  Untitled  Untitled 1  Untitled 2" line

            if not past_headers or not line.strip():
                continue

            parts = line.split("\t")
            if len(parts) < 4:
                continue
            try:
                t = float(parts[0])
                stim = float(parts[1])
                v1 = float(parts[2])
                v2 = float(parts[3])
            except ValueError:
                continue

            time_col.append(t)
            stim_col.append(stim)
            loc1_col.append(v1)
            loc2_col.append(v2)

    return time_col, stim_col, loc1_col, loc2_col


ARTIFACT_SKIP_SAMPLES = 20   # ~0.4 ms at 20 µs/sample — clears the stim artifact tail
CAP_SEARCH_WINDOW = 750      # ~15 ms search window; well beyond any physiological travel time


def find_cap_time(time_col, voltage_col, stim_peak_idx):
    """Return the time of the post-artifact CAP peak in a nerve voltage channel.

    Searches for the maximum in a fixed window starting just after the stim
    artifact (anchored to the stim channel peak), so distant noise is excluded.
    """
    search_start = stim_peak_idx + ARTIFACT_SKIP_SAMPLES
    search_end = min(stim_peak_idx + CAP_SEARCH_WINDOW, len(voltage_col))
    window = voltage_col[search_start:search_end]
    cap_idx = search_start + window.index(max(window))
    return time_col[cap_idx]


def main():
    results = []

    for trial_num in sorted(TRIAL_METADATA.keys()):
        filepath = get_trial_file(trial_num)
        if filepath is None:
            print(f"Trial {trial_num}: file not found, skipping.")
            continue

        time_col, stim_col, loc1_col, loc2_col = parse_labview_csv(filepath)
        if not time_col:
            print(f"Trial {trial_num}: no data parsed from {filepath}, skipping.")
            continue

        stim_peak_idx = stim_col.index(max(stim_col))
        t1 = find_cap_time(time_col, loc1_col, stim_peak_idx)
        t2 = find_cap_time(time_col, loc2_col, stim_peak_idx)
        travel_time_s = t2 - t1

        meta = TRIAL_METADATA[trial_num]
        results.append({
            "Trial Number": trial_num,
            "Current (uA)": meta["current_uA"],
            "Distance (mm)": meta["distance_mm"],
            "Travel Time (ms)": round(travel_time_s * 1000, 4),
        })
        print(
            f"Trial {trial_num}: current={meta['current_uA']} µA, "
            f"dist={meta['distance_mm']} mm, "
            f"t_loc1={t1*1e3:.4f} ms, t_loc2={t2*1e3:.4f} ms, "
            f"travel time={travel_time_s*1e3:.4f} ms"
        )

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Trial Number", "Current (uA)", "Distance (mm)", "Travel Time (ms)"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {OUTPUT_FILE}")
    plot_travel_time_vs_distance()


BACKUP_FILE = os.path.join(os.path.dirname(__file__), "data", "exp2_backup.csv")


def plot_travel_time_vs_distance():
    distances, travel_times = [], []
    with open(BACKUP_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            distances.append(float(row["Distance (mm)"]))
            travel_times.append(float(row["Time(ms)"]))
    distances = np.array(distances)
    travel_times = np.array(travel_times)

    slope, intercept = np.polyfit(distances, travel_times, 1)
    fit_line = slope * distances + intercept
    ss_res = np.sum((travel_times - fit_line) ** 2)
    ss_tot = np.sum((travel_times - travel_times.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    x_range = np.linspace(distances.min() - 2, distances.max() + 2, 200)

    _, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(distances, travel_times, color="#2563EB", s=60, zorder=3,
               label="Trials")
    ax.plot(x_range, slope * x_range + intercept, color="#DC2626", linewidth=1.5,
            label=f"y = {slope:.4f}x + {intercept:.4f}\n$R^2$ = {r2:.4f}")

    ax.set_xlabel("Distance (mm)")
    ax.set_ylabel("Travel Time (ms)")
    ax.set_title("AP Travel Time vs. Electrode Separation")
    ax.legend(framealpha=0.9)
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "ncv_travel_time.png")
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
