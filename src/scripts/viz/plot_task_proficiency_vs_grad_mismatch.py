from rich import print
from rich.pretty import pprint

from tvp.data.datasets.constants import DATASETS_PAPER_TSV_20
from tvp.data.datasets.constants import DATASET_TO_STYLED

from tvp.utils.io_utils import import_json_from_disk

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import copy


def plot_dataset_metrics(
    task_proficiency: Dict[str, List[float]], 
    grad_mismatch: Dict[str, List[float]],
    export_path: str
):
    PROGRESS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    if task_proficiency.keys() != grad_mismatch.keys():
        raise ValueError("Keys of task_proficiency and grad_mismatch do not match")
    else:
        datasets = task_proficiency.keys()

    for dataset in datasets:
        proficiency_metrics = task_proficiency[dataset]
        mismatch_metrics = copy.deepcopy(grad_mismatch[dataset])
        mismatch_metrics = [np.nan] + mismatch_metrics

        # Create a figure and axis
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot task proficiency on the first y-axis
        ax1.set_xlabel('Training Progress')
        ax1.set_xticks(PROGRESS)
        ax1.set_ylabel('Task Proficiency\n(Fine-tuned accuracy)', color='#228B22')
        ax1.plot(PROGRESS, proficiency_metrics, label='Task Proficiency', color='#228B22')
        ax1.tick_params(axis='y', labelcolor='#228B22')

        # Create a second y-axis for grad mismatch
        ax2 = ax1.twinx()
        ax2.set_ylabel('Grad Similarity\n(Cosine similarity)', color='#4169E1')
        ax2.plot(PROGRESS, mismatch_metrics, label='Grad Similarity', color='#4169E1')
        ax2.tick_params(axis='y', labelcolor='#4169E1')

        # Add title and legend
        plt.title(f'Metrics for Dataset: {dataset}')
        fig.tight_layout()  # Adjust layout so labels don't overlap
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

        plt.title(f"Task Proficiency vs. Grad Similarity ({DATASET_TO_STYLED[dataset]})")

        # Save the plot
        plt.savefig(
            export_path.replace("___DATASET_NAME_PLACEHOLDER___", dataset),
            dpi=400
        )

        plt.close()


def main():
    TASK_PROFICIECY_FILE = f"evaluations/task_proficiency/task_proficiency.json"
    GRAD_MISMATCH_FILE = f"evaluations/grad_mismatch/grad_mismatch_layerwise_params.json"

    task_proficiency: dict = import_json_from_disk(TASK_PROFICIECY_FILE)
    grad_mismatch: dict = import_json_from_disk(GRAD_MISMATCH_FILE)
    
    plot_export_path = "plots/task_proficiency_vs_grad_mismatch/task_pro_vs_grad_mis____DATASET_NAME_PLACEHOLDER___.png"
    plot_dataset_metrics(task_proficiency, grad_mismatch, plot_export_path)

    ############################################################################

    task_proficiency = {
        "average_of_tasks": np.array(list(task_proficiency.values())).mean(axis=0).tolist()
    }
    grad_mismatch = {
        "average_of_tasks": np.array(list(grad_mismatch.values())).mean(axis=0).tolist()
    }

    plot_export_path = "plots/task_proficiency_vs_grad_mismatch/task_pro_vs_grad_mis____DATASET_NAME_PLACEHOLDER___.png"
    plot_dataset_metrics(task_proficiency, grad_mismatch, plot_export_path)






if __name__ == "__main__":
    main()