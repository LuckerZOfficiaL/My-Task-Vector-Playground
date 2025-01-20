from rich import print
from rich.pretty import pprint

import numpy as np

import os
from src.tvp.utils.io_utils import import_json_from_disk

import matplotlib.pyplot as plt
import seaborn as sns


def get_avg_task_acc(h, w, s) -> float:

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
            avg_task_acc = data["results"]["average_of_tasks"]
            return avg_task_acc
        else:
            print(f"Config h={h}, w={w}, s={s} NOT tested...")
            return 0
    except Exception as e:
        print(f"An error occurred: {e}")


def plot_heatmap(
    data: np.ndarray, 
    h: int,
    xticklabels: list, yticklabels: list,
    xlabel: str, ylabel: str,
):
    sns.heatmap(
        data, 
        cmap='RdYlGn', 
        annot=True, 
        fmt='.6f',
        xticklabels=xticklabels, 
        yticklabels=yticklabels,
        vmin=0
    )

    plt.title(f"h={h}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    export_file_path = f"./plots/wita_grid_search/h_{h}_paper-tsv-14.png"
    os.makedirs(os.path.dirname(export_file_path), exist_ok=True)
    plt.savefig(
        export_file_path, dpi=400, bbox_inches='tight'
    )

    plt.close()



def main():

    H = [5, 10, 15]
    TOP_K_WEAKEST = [5, 7, 10, 12]
    TOP_K_STRONGEST = [0, 1, 3]
    TOP_K_STRONGEST.reverse() # NOTE: for plotting reasons

    AVG_ACC_TO_BEAT = 0.6922547093459538

    heatmap_data = np.empty((len(H), len(TOP_K_STRONGEST), len(TOP_K_WEAKEST))) 

    for h_idx, h in enumerate(H):
        for w_idx, w in enumerate(TOP_K_WEAKEST):
            for s_idx, s in enumerate(TOP_K_STRONGEST):

                avg_task_acc = get_avg_task_acc(h, w, s)

                # heatmap_data[h_idx, w_idx, s_idx] = avg_task_acc
                heatmap_data[h_idx, s_idx, w_idx] = avg_task_acc - AVG_ACC_TO_BEAT

        plot_heatmap(
            data=heatmap_data[h_idx, :, :], 
            h=h,
            xticklabels=TOP_K_WEAKEST, yticklabels=TOP_K_STRONGEST,
            xlabel="Top K Weakest", ylabel="Top K Strongest"
        )





if __name__ == '__main__':
    main()