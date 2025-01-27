from rich import print
from rich.pretty import pprint

from tvp.data.datasets.constants import DATASETS_PAPER_TSV_20

import numpy as np
from typing import Dict

import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(
    data: np.ndarray, 
    x_labels: str, 
    y_labels: str, 
    cmap: str,
    title: str,
    export_path: str
):
    """
    Plots a heatmap with the given data, labels, and title.

    Parameters:
    - data: 2D numpy array
    - x_labels: list of labels for the x-axis
    - y_labels: list of labels for the y-axis
    - title: title of the heatmap
    """
    plt.figure(figsize=(16, 16))  # Adjust figure size as needed
    sns.heatmap(
        data,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap=cmap,  # Color palette
        annot=True,     # Add annotations for the values
        fmt=".4f",      # Format of the annotations
        cbar=True       # Display the color bar
    )
    
    # Customizing ticks
    plt.xticks(rotation=45)  # Horizontal x-axis labels
    plt.yticks(rotation=0)  # Horizontal y-axis labels

    # Add title
    plt.title(title, fontsize=12, pad=20)

    plt.tight_layout()

    plt.savefig(export_path, dpi=500)

    
    plt.close()


def main():

    datasets = DATASETS_PAPER_TSV_20

    cos_sims_path = f"evaluations/tvs_sims_dists/ViT-B-16_0_ta_adamw_wd_0.1_lr_scheduler_cosine_annealing_warmup_steps_200_merged_paper-tsv-20_cos_sims.npy"

    cos_sims: Dict[str, np.ndarray] = np.load(
        cos_sims_path, allow_pickle=True
    ).item()

    for ratio in cos_sims.keys():
        plot_heatmap(
            data=cos_sims[ratio],
            x_labels=datasets,
            y_labels=datasets,
            cmap="RdYlGn",
            title=f"TVs cosine similarities, {int(ratio*100)}% training progress",
            export_path=f"plots/tvs_sims_dists/cos_sims/ViT-B-16_0_ta_adamw_wd_0.1_lr_scheduler_cosine_annealing_warmup_steps_200_merged_paper-tsv-20_cos_sims_{ratio}.png"
        )

    euclidean_dists_path = f"evaluations/tvs_sims_dists/ViT-B-16_0_ta_adamw_wd_0.1_lr_scheduler_cosine_annealing_warmup_steps_200_merged_paper-tsv-20_l2_dists.npy"

    l2_dists: Dict[str, np.ndarray] = np.load(
        euclidean_dists_path, allow_pickle=True
    ).item()

    for ratio in l2_dists.keys():
        plot_heatmap(
            data=l2_dists[ratio],
            x_labels=datasets,
            y_labels=datasets,
            cmap="RdYlGn_r",
            title=f"TVs L2 distances, {int(ratio*100)}% training progress",
            export_path=f"plots/tvs_sims_dists/l2_dists/ViT-B-16_0_ta_adamw_wd_0.1_lr_scheduler_cosine_annealing_warmup_steps_200_merged_paper-tsv-20_l2_dists_{ratio}.png"
        )


        


if __name__ == "__main__":
    main()