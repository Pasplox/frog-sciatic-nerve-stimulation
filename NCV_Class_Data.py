import csv
import numpy as np
import pandas as pd

XLSX = "frog-sciatic-nerve-stimulation/data/Conduction Velocity Class Data Prior.xlsx"


def load_group(sheet_name):
    df = pd.read_excel(XLSX, sheet_name=sheet_name, header=None)
    # rows 2+ are data; columns 1=Distance(mm), 2=Time(ms)
    data = df.iloc[2:, 1:3].astype(float).reset_index(drop=True)
    data.columns = ["Distance (mm)", "Time (ms)"]
    return data


def regression_slope(data):
    """Fit Distance = slope * Time + intercept; slope is NCV in mm/ms (= m/s)."""
    x = data["Time (ms)"].values
    y = data["Distance (mm)"].values
    slope, _ = np.polyfit(x, y, 1)
    return slope


def main():
    xl = pd.ExcelFile(XLSX)
    groups = xl.sheet_names

    slopes = []
    print(f"{'Group':<10} {'Slope (m/s)':<15}")
    print("-" * 25)
    for group in groups:
        data = load_group(group)
        slope = regression_slope(data)
        slopes.append(slope)
        print(f"{group:<10} {slope:.4f}")

    slopes = np.array(slopes)
    mean_slope = slopes.mean()
    sem_slope = slopes.std(ddof=1) / np.sqrt(len(slopes))

    print("-" * 25)
    print(f"{'Mean':<10} {mean_slope:.4f} m/s")
    print(f"{'SEM':<10} {sem_slope:.4f} m/s")

    rows = [{"Group": g, "Slope (m/s)": round(s, 4)} for g, s in zip(groups, slopes)]
    rows.append({"Group": "Mean", "Slope (m/s)": round(mean_slope, 4)})
    rows.append({"Group": "SEM",  "Slope (m/s)": round(sem_slope, 4)})

    out = "ncv_class_results.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Group", "Slope (m/s)"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
