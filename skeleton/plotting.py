import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

DATASETS = ["credit-g", "breast-w", "phoneme"]
SEEDS = [11, 22, 33, 44, 55]
METHODS = ["default", "random", "smbo", "hyperband"]
MAX_TREES = 243
N_TRIALS = 20

# ensure correct path for saving 
script_dir = Path(__file__).resolve().parent
DIR = script_dir / "../data"
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

    for h in run["history"]:
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

fig, axes = plt.subplots(1, len(DATASETS), figsize=(5 * len(DATASETS), 4), squeeze=False)

for ax, ds in zip(axes[0], DATASETS):
    runs = {m: [] for m in METHODS}
    finals = {m: [] for m in METHODS}
    for s in SEEDS:
        for r in load(s, ds):
            runs[r["method"]].append(curve(r))
            finals[r["method"]].append(r["final_result"]["metrics"]["balanced_accuracy"])

    grid = np.linspace(0, max(x[-1] for m in METHODS for x, _ in runs[m]), 300)

    for m in METHODS:
        if not runs[m]:
            continue
        if m == "default":
            Y = np.array([[y[-1]] * len(grid) for _, y in runs[m]])
        else:
            Y = np.array([on_grid(x, y, grid) for x, y in runs[m]])
        mu, sd = np.nanmean(Y, 0), np.nanstd(Y, 0)
        ax.plot(grid, mu, label=m, linestyle="--" if m == "default" else "-")
        ax.fill_between(grid, mu - sd, mu + sd, alpha=0.2)

    print(ds)
    for m in METHODS:
        if finals[m]:
            print(f"  {m:10s} test bal. acc: {np.mean(finals[m]):.4f} ± {np.std(finals[m]):.4f}")

    ax.set_title(ds)
    ax.set_xlabel("n_trees (cumulative)")
    ax.set_ylabel("incumbent balanced accuracy")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("balanced_accuracy_curves_1.png", dpi=200)
