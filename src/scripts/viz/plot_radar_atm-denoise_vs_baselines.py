def parse_eval_res(eval_res_raw: dict) -> dict:

    eval_res = {}

    for d, r in eval_res_raw.items():
        if d != "average_of_tasks":
            eval_res[d] = r[0]["acc/test"]
        else:
            eval_res[d] = r

    return eval_res

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import List

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import List

def radar_plot(
    data: dict,
    radar_points: List[str],
    colors: dict,
    export_file_path: str,
    legend_labels: dict = None,
    legend_label_colors: dict = None,
):
    # Latex settings for publication-quality plots
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'

    first_method = next(iter(data.values()))
    tasks = list(first_method.keys())
    num_tasks = len(tasks)
    
    angles = np.linspace(0, 2 * np.pi, num_tasks, endpoint=False).tolist()
    angles += angles[:1]

    # Create the polar plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for method, results in data.items():
        values = [results[task] for task in tasks]
        values += values[:1]  # close the loop
        line_color = colors.get(method, 'b') 
        ax.plot(angles, values, linewidth=2, color=line_color, label=method)
        ax.fill(angles, values, alpha=0.1, color=line_color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_points, fontsize=18)
    ax.set_yticklabels([])

    all_values = [results[task] for results in data.values() for task in tasks]
    max_val = max(all_values)
    ax.set_ylim(0, max_val * 1.1)

    # Prepare custom legend handles
    handles = []
    for method in data.keys():
        label = legend_labels.get(method, method) if legend_labels else method
        color_ = (legend_label_colors.get(method, colors.get(method, 'b'))
                  if legend_label_colors else colors.get(method, 'b'))
        handle = Line2D([0], [0], color=color_, lw=2, label=label)
        handles.append(handle)

    # Provide extra space at the bottom of the figure
    plt.subplots_adjust(bottom=0.25)

    # Place the legend below the plot, centered
    #  - 'upper center' positions the legend’s top center at the anchor
    #  - (0.5, -0.1) anchors it below the bottom of the axes
    ax.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.025),
        # ncol=len(data),  # number of legend columns (optional)
        ncol=2,  # number of legend columns (optional)
        fontsize=18
    )

    plt.tight_layout()
    plt.savefig(export_file_path, dpi=400)
    
    return fig, ax



from rich import print
from rich.pretty import pprint
from tvp.utils.io_utils import import_json_from_disk
import os

from tvp.data.datasets.constants import DATASET_TO_STYLED

def datasets_to_styled(datasets: List[str]) -> List[str]:
    return ["Average" if d == "average_of_tasks" else d for d in datasets]

def main():
    BUDGETS = [2, 4, 10]
    METHODS = ["ta", "bc", "ties", "dare", "atm"]
    legend_labels = [
        "Task Arithmetic", "Model Breadcrumbs", "TIES-merging", "DARE", "PH-ATM (10 iterations)"
    ]
    legend_colors = [
        "#ffbe0b", "#fb5607", "#3a86ff", "#8338ec", "#02c39a"
    ]
    COLORS_DICT = {
        "ta": "#ffbe0b",
        "bc": "#fb5607",
        "ties": "#3a86ff",
        "dare": "#8338ec",
        "atm": "#02c39a"
    }
    LEGEND_DICT = {
        "ta": "Task Arithmetic",
        "bc": "Model Breadcrumbs",
        "ties": "TIES-merging",
        "dare": "DARE",
        "atm": "PH-ATM (10 iterations)"
    }

    eval_res = {}

    for m in METHODS:

        conf_res_name = "none" if m in ["ta", "atm"] else m
        train_batches = 1.0
        ord = 1 if m != "atm" else 10
        eps_per_ord = "CONVERGENCE" if m != "atm" else 1
        atm_version = "atm-true" if m != "atm" else "atm-denoise"

        eval_file_path = (
            f"./evaluations/{atm_version}/"
            f"ViT-B-16_0_{atm_version}_"
            f"confl_res_{conf_res_name}_"
            f"train_batches_{train_batches}_"
            f"ord_{ord}_"
            f"eps_per_ord_{eps_per_ord}_"
            f"merged.json"
        )

        eval_res_raw = import_json_from_disk(
            file_path=eval_file_path
        )["results"]

        eval_res[m] = parse_eval_res(eval_res_raw)


    pprint(eval_res, expand_all=True)

    export_file_path = f"./plots/atm-denoise_vs_baselines/atm-denoise_vs_baselines.png"
    os.makedirs(os.path.dirname(export_file_path), exist_ok=True)
    radar_plot(
        data=eval_res,
        radar_points=datasets_to_styled(list(eval_res["atm"].keys())),
        colors=COLORS_DICT,
        export_file_path=export_file_path,
        legend_labels=LEGEND_DICT,
        legend_label_colors=COLORS_DICT
    )













if __name__ == "__main__":
    main()