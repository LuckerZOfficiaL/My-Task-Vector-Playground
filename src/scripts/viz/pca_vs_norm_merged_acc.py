from rich import print
from rich.pretty import pprint

from src.tvp.utils.io_utils import list_all_files_in_dir, import_json_from_disk
from src.tvp.utils.vectors import pairwise_cos_sim, pairwise_euclidean_dist
from typing import List

import numpy as np
from numpy import ndarray
from typing import Dict

import os

from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import torch

def task_to_color(task: str) -> str:
    COLOR_MAP = {
        "cars": "red",
        "dtd": "blue",
        "eurosat": "green",
        "gtsrb": "purple",
        "mnist": "orange",
        "resisc45": "cyan",
        "svhn": "magenta",
        "sun397": "brown",
        "stl10": "pink",
        "oxfordiiitpet": "lime",
        "flowers102": "teal",
        "cifar100": "gold",
        "pcam": "darkred",
        "fer2013": "navy",
        "cifar10": "olive",
        "food101": "darkgreen",
        "fashionmnist": "salmon",
        "renderedsst2": "turquoise",
        "emnist": "orchid",
        "kmnist": "darkblue",
        "zs": "black"
    }

    return COLOR_MAP[task]


def plot(
    tasks_pca_embeddings: ndarray,
    task_colors: List[str],
    task_names: List[str],
    title: str,
    export_path: str
):
    plt.figure(figsize=(8, 8))
    
    plt.scatter(
        x=tasks_pca_embeddings[:, 0], 
        y=tasks_pca_embeddings[:, 1], 
        c=task_colors, 
        s=50, 
        edgecolors=task_colors
    )

    # Add legend
    for task, color in zip(task_names, task_colors):
        plt.scatter([], [], c=color, label=task, s=50, edgecolors=color)  # Invisible points for the legend
    plt.legend(title="Tasks", loc="best")
    
    plt.title(title)

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    
    plt.grid(True)
    
    plt.tight_layout()
    
    plt.savefig(export_path, dpi=400)
    
    plt.close()


def main():
    SUBSETS_DIR = "evaluations/merged_subsets/paper-tsv-20/ta/optim_adamw_wd_0.1/cosine_annealing_warmup_steps_200/subset_size_05"

    subset_config_file_path_list: List[str] = list_all_files_in_dir(SUBSETS_DIR)


    # pprint(subset_configs)
    print(f"Number of subsets: {len(subset_config_file_path_list)}")

    PCA_FILE_PATH = "plots/pca_embedding/pca_embedding_2D_checkpoints_Cars_DTD_EuroSAT_GTSRB_MNIST_RESISC45_SVHN_SUN397_STL10_OxfordIIITPet_Flowers102_CIFAR100_PCAM_FER2013_CIFAR10_Food101_FashionMNIST_RenderedSST2_EMNIST_KMNIST_dict.npy"
    pca_embeddings: Dict[str, ndarray] = np.load(PCA_FILE_PATH, allow_pickle=True).item()
    # print(f"PCA embedding dict: ")
    # pprint(pca_embeddings, expand_all=True)

    for config_idx, subset_config_file_path in tqdm(
        iterable=enumerate(subset_config_file_path_list), 
        desc="Plotting PCA of subsets",
        total=len(subset_config_file_path_list),
        colour="green"
    ):

        subset_eval_result: dict = import_json_from_disk(
            file_path=os.path.join(SUBSETS_DIR, subset_config_file_path)
        )["results"]
        # pprint(subset_eval_result, expand_all=True)

        task_names: List[str] = list(subset_eval_result.keys())
        task_names.pop(task_names.index("average_of_tasks"))
        avg_norm_merged_acc = subset_eval_result["average_of_tasks"]

        task_colors = [task_to_color(task.lower()) for task in task_names]
        task_colors.append(task_to_color("zs"))

        task_pca_embeddings: List[ndarray] = [
            pca_embeddings[f"{task.lower()}, 100%"] for task in task_names
        ]
        task_pca_embeddings.append(pca_embeddings["zs"])
        task_pca_embeddings = np.vstack(task_pca_embeddings)

        cos_sim = float(pairwise_cos_sim(torch.tensor(task_pca_embeddings)).mean())
        l2_dist = float(pairwise_euclidean_dist(torch.tensor(task_pca_embeddings)).mean())
        task_names_for_plot = copy.deepcopy(task_names)
        task_names_for_plot.append("zero-shot")
        plot(
            tasks_pca_embeddings=task_pca_embeddings,
            task_colors=task_colors,
            task_names=task_names_for_plot,
            title=f"{' '.join(task_names).replace('average_of_tasks', '')}\nAvg. Norm. Merged Acc.: {avg_norm_merged_acc:.8f}\nAvg cos sim: {cos_sim:.8f}\nAvg l2 dist: {l2_dist:.8f}",
            export_path=f"plots/pca_vs_norm_merged_acc/pca_vs_norm_merged_acc_{avg_norm_merged_acc:.8f}_{'-'.join(task_names).replace('average_of_tasks', '')}.png"
        )







if __name__ == '__main__':
    main()