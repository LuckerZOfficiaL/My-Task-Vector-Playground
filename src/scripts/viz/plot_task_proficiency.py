from rich import print
from rich.pretty import pprint

from tvp.data.datasets.constants import DATASETS_PAPER_TSV_20
from tvp.data.datasets.constants import DATASET_TO_STYLED

from tvp.utils.io_utils import import_json_from_disk, export_json_to_disk

from typing import Dict, List

import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def main():
    TASK_PROFICIECY_FILE = f"evaluations/ft_progress_merging/ViT-B-16_0_ta_adamw_wd_0.1_lr_scheduler_cosine_annealing_warmup_steps_200_merged_{'-'.join([DATASET_TO_STYLED[t] for t in DATASETS_PAPER_TSV_20])}.json"
    TASK_PROFICIECY_ZS_FILE = f"evaluations/zs/zs.json"

    task_proficiency = import_json_from_disk(TASK_PROFICIECY_FILE)["results_all_ratios"]
    task_proficiency_zs = import_json_from_disk(TASK_PROFICIECY_ZS_FILE)

    RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    plot_data: Dict[str, List[float]] = {}

    for dataset_idx, dataset in enumerate(DATASETS_PAPER_TSV_20):

        plot_data[dataset] = []
        plot_data[dataset].append(
            task_proficiency_zs[DATASET_TO_STYLED[dataset]][0]["acc/test"]
        )

        # zs added to the list, so gotta start from the ratio 0.1
        for ratio_idx, ratio in enumerate(RATIOS[1:]):

            plot_data[dataset].append(
                task_proficiency[
                    str(ratio)
                ][DATASET_TO_STYLED[dataset]][0]["acc/test"]
            )

    plot_data["average_of_tasks"] = np.array(list(plot_data.values())).mean(axis=0).tolist()

    # pprint(plot_data, expand_all=True)

    export_file_path = f"./evaluations/task_proficiency/task_proficiency.json"
    os.makedirs(os.path.dirname(export_file_path), exist_ok=True)
    export_json_to_disk(
        data=plot_data, 
        export_dir=os.path.dirname(export_file_path), 
        file_name=os.path.basename(export_file_path).replace(".json", "")
    )

    export_file_path = f"./plots/task_proficiency/heatmap/task_proficiency.png"
    os.makedirs(os.path.dirname(export_file_path), exist_ok=True)
    
    plt.figure(figsize=(12, 12))
    sns.heatmap(
        data=np.array(list(plot_data.values())), 
        annot=True, 
        fmt=".2f", 
        cmap="RdYlGn", 
        cbar=True,
        vmin=0, 
        vmax=1,
        xticklabels=[f"{int(ratio*100)}%" for ratio in RATIOS],
        yticklabels=[DATASET_TO_STYLED[t] for t in list(plot_data.keys())],
    )
    plt.xlabel("TA Training Steps %")
    plt.ylabel("Task")

    plt.yticks(rotation=0, fontsize=10)  # Rotate and reduce font size
    plt.tight_layout()  # Automatically adjust layout to prevent overlapping

    plt.title("Task Proficiency")

    plt.savefig(export_file_path, dpi=400)

    plt.close()

    ############################################################################

    # plot the heatmap as single line plots

    for dataset_idx, dataset in enumerate(plot_data.keys()):
            
            export_file_path = f"./plots/task_proficiency/line/task_proficiency_{DATASET_TO_STYLED[dataset]}.png"
            os.makedirs(os.path.dirname(export_file_path), exist_ok=True)
            
            plt.figure(figsize=(10, 6))
            plt.plot(
                [int(ratio*100) for ratio in RATIOS], 
                plot_data[dataset], 
                label="Task Proficiency"
            )
            plt.xlabel("TA Training Steps %")
            plt.xticks([int(ratio*100) for ratio in RATIOS])
            plt.ylabel("Task Proficiency")
            plt.title(f"Task Proficiency {DATASET_TO_STYLED[dataset]}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(export_file_path, dpi=400)
            plt.close()

    ############################################################################

    plot_data_error: Dict[str, List[float]] = {}
    for dataset_idx, dataset in enumerate(DATASETS_PAPER_TSV_20):

        plot_data_error[dataset] = []
        plot_data_error[dataset].append(
            1 - task_proficiency_zs[DATASET_TO_STYLED[dataset]][0]["acc/test"]
        )

        # zs added to the list, so gotta start from the ratio 0.1
        for ratio_idx, ratio in enumerate(RATIOS[1:]):

            plot_data_error[dataset].append(
                1 - task_proficiency[
                    str(ratio)
                ][DATASET_TO_STYLED[dataset]][0]["acc/test"]
            )

    plot_data_error["average_of_tasks"] = np.array(list(plot_data_error.values())).mean(axis=0).tolist()
    
    # pprint(plot_data_error, expand_all=True)
    export_file_path = f"./evaluations/task_proficiency/task_proficiency_error.json"
    os.makedirs(os.path.dirname(export_file_path), exist_ok=True)
    export_json_to_disk(
        data=plot_data_error, 
        export_dir=os.path.dirname(export_file_path),
        file_name=os.path.basename(export_file_path).replace(".json", "")
    )
    
    export_file_path = f"./plots/task_proficiency/heatmap/task_proficiency_error.png"
    os.makedirs(os.path.dirname(export_file_path), exist_ok=True)
    
    plt.figure(figsize=(12, 12))
    sns.heatmap(
        data=np.array(list(plot_data_error.values())), 
        annot=True, 
        fmt=".2f", 
        cmap="RdYlGn", 
        cbar=True,
        vmin=0, 
        vmax=1,
        xticklabels=[f"{int(ratio*100)}%" for ratio in RATIOS],
        yticklabels=[DATASET_TO_STYLED[t] for t in list(plot_data.keys())],
    )
    plt.xlabel("TA Training Steps %")
    plt.ylabel("Task")

    plt.yticks(rotation=0, fontsize=10)  # Rotate and reduce font size
    plt.tight_layout()  # Automatically adjust layout to prevent overlapping

    plt.title("Task Proficiency Error")

    plt.savefig(export_file_path, dpi=400)

    plt.close()

    ############################################################################

    # plot the heatmap as single line plots

    for dataset_idx, dataset in enumerate(plot_data_error.keys()):        
        export_file_path = f"./plots/task_proficiency/line/task_proficiency_error_{DATASET_TO_STYLED[dataset]}.png"
        os.makedirs(os.path.dirname(export_file_path), exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(
            [int(ratio*100) for ratio in RATIOS], 
            plot_data_error[dataset], 
            label="Task Proficiency Error"
        )
        plt.xlabel("TA Training Steps %")
        plt.xticks([int(ratio*100) for ratio in RATIOS])
        plt.ylabel("Task Proficiency Error")
        plt.title(f"Task Proficiency Error {DATASET_TO_STYLED[dataset]}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(export_file_path, dpi=400)
        plt.close()






if __name__ == "__main__":
    main()