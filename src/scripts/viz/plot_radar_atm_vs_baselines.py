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

def radar_plot(
    data: dict,
    radar_points: List[str],
    title: str,
    colors: dict,
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
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    
    # Draw one line per method
    for method, results in data.items():
        values = [results[task] for task in tasks]
        values += values[:1]
        color = colors[method]
        ax.plot(angles, values, label=method, linewidth=2, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    # Set the labels for each task on the outer circle
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_points, fontsize=18)
    
    ax.set_yticklabels([])

    
    # Optionally, set the range of the radial axis.
    # Here, we determine the maximum value among all tasks and methods
    all_values = [results[task] for results in data.values() for task in tasks]
    max_val = max(all_values)
    ax.set_ylim(0, max_val * 1.1)
    
    # Add a title and a legend
    plt.title(title, size=18, y=1.08)

    plt.savefig(export_file_path, dpi=400)
    
    return fig, ax

import matplotlib.patches as mpatches

def combine_radar_plots(
    radar_axes_dict: dict, 
    legend_labels: List[str], 
    legend_colors: List[str],
    export_file_path: str
):
    """
    Combine individual radar plots (returned as (fig, ax) tuples) stored in radar_axes_dict
    into a single figure with a 1xn grid, and add a custom legend with the provided labels and colors.

    Parameters:
      - radar_axes_dict: dict, keys are identifiers (e.g., budgets) and values are tuples (fig, ax) from radar_plot.
      - legend_labels: list of strings for the legend.
      - legend_colors: list of colors corresponding to the legend_labels.
    """
    n = len(radar_axes_dict)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5.75), subplot_kw=dict(polar=True))
    if n == 1:
        axes = [axes]  # ensure axes is iterable
    
    # Re-plot the content from each individual radar plot into the new axes.
    for ax_new, (key, (old_fig, old_ax)) in zip(axes, radar_axes_dict.items()):
        # Transfer each line from the old axis
        for line in old_ax.get_lines():
            ax_new.plot(line.get_xdata(), line.get_ydata(),
                        color=line.get_color(),
                        linewidth=line.get_linewidth())
        # Transfer the filled areas (patches)
        for patch in old_ax.patches:
            # Get the coordinates from the patch (assumes a Polygon)
            xy = patch.get_xy()
            ax_new.fill(xy[:, 0], xy[:, 1],
                        color=patch.get_facecolor(),
                        alpha=patch.get_alpha())
            
        # Transfer tick settings and limits
        ax_new.set_xticks(old_ax.get_xticks())
        ax_new.set_xticklabels(old_ax.get_xticklabels(), fontsize=18)
        ax_new.set_yticklabels([])  # remove radial labels
        ax_new.set_ylim(old_ax.get_ylim())
        
        # Optionally, add a subplot title (using the key as the title)
        ax_new.set_title(f"K = {key}", size=18, y=1.08)
    
    # Create custom legend handles using the provided labels and colors.
    handles = [mpatches.Patch(color=color, label=label)
               for label, color in zip(legend_labels, legend_colors)]
    # Add the legend to the combined figure.
    fig.legend(handles=handles, loc='lower center', ncol=len(legend_labels), fontsize=18)

    plt.tight_layout()

    plt.savefig(export_file_path, dpi=400)
    
    return fig, axes

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
        "Task Arithmetic", "Model Breadcrumbs", "TIES-merging", "DARE", "PA-ATM"
    ]
    legend_colors = [
        "#ffbe0b", "#fb5607", "#3a86ff", "#8338ec", "#ff006e"
    ]
    COLORS_DICT = {
        "ta": "#ffbe0b",
        "bc": "#fb5607",
        "ties": "#3a86ff",
        "dare": "#8338ec",
        "atm": "#ff006e"
    }

    radar_plots: dict = {}

    for b in BUDGETS:

        eval_res = {}

        for m in METHODS:
            print(f"Budget: {b}, Method: {m}")

            conf_res_name = "none" if m == "ta" or m == "atm" else m
            # train_batches = 0.1 if m == "atm" else 1.0
            train_batches = 1.0
            ord = 1 if m != "atm" else b
            eps_per_ord = b if m != "atm" else 1

            print(f"conf_res_name: {conf_res_name}")
            print(f"train_batches: {train_batches}")
            print(f"ord: {ord}")
            print(f"eps_per_ord: {eps_per_ord}")

            eval_file_path = (
                f"./evaluations/atm-true/"
                f"ViT-B-16_0_atm-true_"
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

        export_file_path = f"./plots/atm_vs_baselines/atm_vs_baselines_budget_{b}.png"
        os.makedirs(os.path.dirname(export_file_path), exist_ok=True)
        radar_plots[b] = radar_plot(
            data=eval_res,
            radar_points=datasets_to_styled(list(eval_res["atm"].keys())),
            title=f"K = {b}",
            colors=COLORS_DICT,
            export_file_path=export_file_path
        )

        print(f"{'='*50}\n\n")

    export_file_path = f"./plots/atm_vs_baselines/atm_vs_baselines_combined.png"
    os.makedirs(os.path.dirname(export_file_path), exist_ok=True)
    combine_radar_plots(
        radar_axes_dict=radar_plots,
        legend_labels=legend_labels,
        legend_colors=legend_colors,
        export_file_path=export_file_path
    )













if __name__ == "__main__":
    main()