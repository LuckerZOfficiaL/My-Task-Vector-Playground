from rich import print
from rich.pretty import pprint

import os
import json
from src.tvp.data.datasets.constants import DATASETS_PAPER_TSV_14, DATASET_TO_STYLED

import matplotlib.pyplot as plt
import numpy as np


def import_json_from_disk(file_path: str):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    return data


def plot_radar_plot(
    categories: list,
    radar_plot_data: dict,
    title: str,
    export_file_path: str
):

    methods = radar_plot_data.values()
    labels = radar_plot_data.keys()

    COLORS = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Yellow-Green
        "#17becf",  # Cyan
        "#000000",  # black
    ]

    num_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    methods = [method + [method[0]] for method in methods]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for method, label, color in zip(methods, labels, COLORS):
        ax.fill(angles, method, color=color, alpha=0.2)
        ax.plot(angles, method, color=color, label=label)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=20)

    plt.legend(loc=(0.95, 0.95), fontsize=12)

    plt.title(title, size=12, color='black', y=1.1)

    os.makedirs(os.path.dirname(export_file_path), exist_ok=True)
    plt.savefig(export_file_path, dpi=300, bbox_inches='tight')


def get_baseline() -> dict:
    
        # evaluations/merged_wita_grid_search/ViT-B-16_0_ta_adamw_wd_0.1_lr_scheduler_cosine_annealing_warmup_steps_200_wita_num_iters_5_top_k_weakest_5_top_k_strongest_0_merged_paper-tsv-14.json
    
        file_path = (
            f"./evaluations/merged_wita_grid_search/"
            f"ViT-B-16"
            f"_0"
            f"_ta"
            f"_adamw_wd_0.1"
            f"_lr_scheduler_cosine_annealing_warmup_steps_200"
            f"_wita_num_iters_1"
            f"_top_k_weakest_14"
            f"_top_k_strongest_0"
            f"_merged_paper-tsv-14"
            f".json"
        )
    
        try:
            if os.path.exists(file_path):
                data = import_json_from_disk(file_path)
                config_accs_raw = data["results"]
    
                config_accs = {}
                for k, v in config_accs_raw.items():
                    config_accs[k] = v[0]["acc/test"] if k != "average_of_tasks" else v
    
                return config_accs
            else:
                print(f"Baseline not found...")
                return None
        except Exception as e:
            print(f"An error occurred: {e}")



def get_config_accs(h, w, s) -> dict:

    # evaluations/merged_wita_grid_search/ViT-B-16_0_ta_adamw_wd_0.1_lr_scheduler_cosine_annealing_warmup_steps_200_wita_num_iters_5_top_k_weakest_5_top_k_strongest_0_merged_paper-tsv-14.json

    file_path = (
        f"./evaluations/merged_wita_grid_search/"
        f"ViT-B-16"
        f"_0"
        f"_ta"
        f"_adamw_wd_0.1"
        f"_lr_scheduler_cosine_annealing_warmup_steps_200"
        f"_wita_num_iters_{h}"
        f"_top_k_weakest_{w}"
        f"_top_k_strongest_{s}"
        f"_merged_paper-tsv-14"
        f".json"
    )

    try:
        if os.path.exists(file_path):
            data = import_json_from_disk(file_path)
            config_accs_raw = data["results"]

            config_accs = {}
            for k, v in config_accs_raw.items():
                config_accs[k] = v[0]["acc/test"] if k != "average_of_tasks" else v

            return config_accs
        else:
            print(f"Config h={h}, w={w}, s={s} NOT tested...")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")




def main():

    H = [5, 10, 15]
    TOP_K_WEAKEST = [5, 7, 10, 12]
    TOP_K_STRONGEST = [0, 1, 3]

    for h in H:

        radar_plot_data = {}

        for w in TOP_K_WEAKEST:
            for s in TOP_K_STRONGEST:

                config_accs = get_config_accs(h, w, s)

                if config_accs is None:
                    continue

                if list(config_accs.keys()) != [DATASET_TO_STYLED[t] for t in DATASETS_PAPER_TSV_14] + ["average_of_tasks"]:
                    raise ValueError(
                        "Dataset keys do not match!\n"
                        f"Expected: {[DATASET_TO_STYLED[t] for t in DATASETS_PAPER_TSV_14] + ['average_of_tasks']}\n"
                        f"Got: {config_accs.keys()}"
                    )

                config_name = f"w={w}_s={s}"
                radar_plot_data[config_name] = list(config_accs.values())
        
        radar_plot_data["baseline"] = list(get_baseline().values())

        export_file_path = f"./plots/wita_radar_plot/h_{h}_paper-tsv-14.png"
        plot_radar_plot(
            categories=[DATASET_TO_STYLED[t] for t in DATASETS_PAPER_TSV_14] + ["average_of_tasks"],
            radar_plot_data=radar_plot_data,
            export_file_path=export_file_path,
            title=f"h={h}"
        )
        
if __name__ == '__main__':
    main()