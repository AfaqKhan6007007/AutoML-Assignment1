import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

DATASETS = ["credit-g", "breast-w", "phoneme", "Phishing_Legitimate_full", "gas-drift"]
SEEDS = [11, 22, 33, 44, 55]
METHODS = ["default", "random", "smbo", "hyperband", "foundation"]
MAX_TREES = 81
N_TRIALS = 14

# ensure correct path for saving 
script_dir = Path(__file__).resolve().parent
DIR = script_dir / "../data/"
DIR.mkdir(parents=True, exist_ok=True)

def load(seed, ds):
    with open(f"{DIR}/expseed_{seed}_{ds}.json") as f:
        return json.load(f)


def curve(run: dict[Any]):
    """ 
    run: only new trees are considered for resource on x-axis. Y-axis 
    is the best score so far. 
    
    """
    resource = 0
    previous = {}
    x = []
    y = []
    best = float("-inf")

    for h in run.get("history", []):
        config = h["config_id"]
        trees = h["n_trees"]
        previous_trees = previous.get(config, 0)

        resource += max(0, trees - previous_trees)
        previous[config] = max(previous_trees, trees)

        best = max(best, h["objective"])
        x.append(resource)
        y.append(best)

    return np.array(x), np.array(y)
   
def on_grid(x, y, grid):
    i = np.searchsorted(x, grid, side="right") - 1
    return np.where(i >= 0, y[np.clip(i, 0, None)], np.nan)

for ds in DATASETS:
    runs = {m: [] for m in METHODS}
    finals = {m: [] for m in METHODS}

    for s in SEEDS:
        for r in load(s, ds):
            if r.get("history"):
                runs[r["method"]].append(curve(r))

            fr = r.get("final_result", r.get("result"))
            finals[r["method"]].append(
                fr["metrics"]["balanced_accuracy"]
            )

    # One plot per dataset
    fig, ax = plt.subplots(figsize=(5, 4))

    grid = np.linspace(
        0,
        max(x[-1] for m in METHODS for x, _ in runs[m]),
        300
    )

    for m in METHODS:
        if not runs[m]:
            continue

        if m == "default":
            Y = np.array([
                [y[-1]] * len(grid)
                for _, y in runs[m]
            ])
        else:
            Y = np.array([
                on_grid(x, y, grid)
                for x, y in runs[m]
            ])

        mu, sd = np.nanmean(Y, 0), np.nanstd(Y, 0)

        ax.plot(
            grid,
            mu,
            label=m,
            linestyle="--" if m == "default" else "-"
        )

        ax.fill_between(
            grid,
            mu - sd,
            mu + sd,
            alpha=0.2
        )

    ax.set_title(ds)
    ax.set_xlabel("n_trees (cumulative)")
    ax.set_ylabel("incumbent balanced accuracy")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save one plot per dataset
    plot_path = DIR / f"balanced_accuracy_curve_{ds}.png"
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)

    print(f"Saved plot to: {plot_path}")


csv_path = DIR / "final_balanced_accuracy.csv"

import csv

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "dataset",
        "method",
        "mean_balanced_accuracy",
        "std_balanced_accuracy",
    ])

    for ds in DATASETS:
        finals = {m: [] for m in METHODS}

        for s in SEEDS:
            for r in load(s, ds):
                if "final_result" in r:
                    score = r["final_result"]["metrics"]["balanced_accuracy"]

                elif "result" in r:
                    score = r["result"]["metrics"]["balanced_accuracy"]

                else:
                    continue

                finals[r["method"]].append(score)

        for m in METHODS:
            if finals[m]:
                writer.writerow([
                    ds,
                    m,
                    np.mean(finals[m]),
                    np.std(finals[m]),
                ])

print(f"Saved anytime curve to: {DIR / 'balanced_accuracy_curves_1.png'}")
print(f"Saved final results to: {csv_path}")