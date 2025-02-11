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
import matplotlib.patches as mpatches

def radar_plot(
    data: dict,
    radar_points: List[str],
    legend_labels: List[str],
    legend_colors: List[str],
    export_file_path: str
):
    """
    Create a radar plot from a dictionary with structure:
    
        {
            'method1': {
                'task1': value1,
                'task2': value2,
                ...
                'average_of_tasks': average_value
            },
            'method2': { ... },
            ...
        }
    """

    # latex stuff for the paper
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'

    first_method = next(iter(data.values()))
    tasks = [key for key in first_method.keys()]
    num_tasks = len(tasks)
    
    # Compute the angles at which each axis is drawn.
    # We add one extra angle so that the plot closes (first point == last point)
    angles = np.linspace(0, 2 * np.pi, num_tasks, endpoint=False).tolist()
    angles += angles[:1]  # repeat the first angle to close the loop

    # Create the polar plot
    fig, ax = plt.subplots(figsize=(7.5,7.5), subplot_kw=dict(polar=True))
    
    # Draw one line per method
    for method, results in data.items():
        values = [results[task] for task in tasks]
        values += values[:1]
        color = legend_colors[method]
        ax.plot(angles, values, label=method, linewidth=2, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    # Set the labels for each task on the outer circle
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_points, fontsize=18)
    
    ax.set_yticklabels([])

    # Create custom legend handles using the provided labels and colors.
    handles = [
        mpatches.Patch(color=color, label=label) for label, color in zip(legend_labels, list(legend_colors.values()))
    ]
    # Add the legend to the combined figure.
    fig.legend(handles=handles, loc='lower center', ncol=len(legend_labels), fontsize=18)

    
    # Optionally, set the range of the radial axis.
    # Here, we determine the maximum value among all tasks and methods
    all_values = [results[task] for results in data.values() for task in tasks]
    max_val = max(all_values)
    ax.set_ylim(0, max_val * 1.1)

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

    EPOCHS = ["1", "CONVERGENCE"]
    LABELS = ["TA 1 epoch", "TA convergence epochs"]
    COLORS_DICT = {
        "1": "#ffbe0b",
        "CONVERGENCE": "#ff006e"
    }

    eval_res = {}

    for e in EPOCHS:
        print(f"Epochs: {e}")
        
        # evaluations/atm-true/ViT-B-16_0_atm-true_confl_res_bc_train_batches_1.0_ord_1_eps_per_ord_CONVERGENCE_merged.json
        eval_res_raw = import_json_from_disk(
            file_path=f"./evaluations/atm-true/ViT-B-16_0_atm-true_confl_res_none_train_batches_1.0_ord_1_eps_per_ord_{e}_merged.json"
        )["results"]

        eval_res[e] = parse_eval_res(eval_res_raw)

    pprint(eval_res, expand_all=True)


    export_file_path = f"./plots/ta_1_ep_vs_ta_conv/ta_1_ep_vs_ta_conv.png"
    os.makedirs(os.path.dirname(export_file_path), exist_ok=True)
    radar_plot(
        data=eval_res,
        radar_points=datasets_to_styled(list(eval_res["1"].keys())),
        legend_colors=COLORS_DICT,
        legend_labels=LABELS,
        export_file_path=export_file_path
    )













if __name__ == "__main__":
    main()