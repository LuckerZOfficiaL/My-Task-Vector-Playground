import hydra
from nn_core.common import PROJECT_ROOT
from omegaconf import DictConfig
from nn_core.callbacks import NNTemplateCore
from nn_core.model_logging import NNLogger

def init_logger(cfg: DictConfig) -> NNLogger:
    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None)
    )
    
    logger: NNLogger = NNLogger(
        logging_cfg=cfg.train.logging, 
        cfg=cfg, 
        resume_id=template_core.resume_id
    )

    return logger


from typing import Union, List, Dict
from tvp.modules.encoder import ClassificationHead, ImageEncoder
from tvp.utils.io_utils import load_model_from_artifact
from torch import Tensor
from torch.nn.utils import parameters_to_vector
from numpy import ndarray
from src.tvp.data.datasets.constants import DATASET_TO_STYLED

def get_zeroshot_model(logger: NNLogger) -> Union[ClassificationHead, ImageEncoder]:

    zeroshot_identifier = f"ViT-B-16_pt"
    zeroshot_model = load_model_from_artifact(
        artifact_path=f"{zeroshot_identifier}:latest", run=logger.experiment
    )

    return zeroshot_model

def compose_artifact_name(dataset: str, ratio: float) -> str:
    ft_ta_identifier = (
        f"ViT-B-16"
        f"____DATASET_NAME_PLACEHOLDER___"
        f"_0"
        f"_ta"
        f"_adamw"
        f"_wd_0.1"
        f"_lr_scheduler_cosine_annealing"
        f"_warmup_steps_200"
        f"____STEP_RATIO_PLACEHOLDER___"
        f":latest"
    )

    return ft_ta_identifier.replace(
        "___DATASET_NAME_PLACEHOLDER___", dataset
    ).replace(
        "___STEP_RATIO_PLACEHOLDER___", f"step_{ratio}"
    )

def get_task_vectors_dict(
    datasets: List[str],
    logger: NNLogger,
    zeroshot_vec: Tensor
) -> Dict[str, ImageEncoder]:

    task_vectors_dict: Dict[str, Tensor] = {}

    for task in datasets:
        ft_model_identifier = compose_artifact_name(
            dataset=DATASET_TO_STYLED[task], ratio=1.0
        )
        
        ft_model: ImageEncoder = load_model_from_artifact(
            artifact_path=ft_model_identifier, run=logger.experiment
        )

        ft_vec: Tensor = parameters_to_vector(
            parameters=ft_model.parameters()
        ).detach().cpu()

        task_vectors_dict[task] = ft_vec - zeroshot_vec

    return task_vectors_dict


import matplotlib.pyplot as plt
import numpy as np

def plot(
    x: List[float],
    y: List[float],
    x_label: str,
    y_label: str,
    title: str,
    save_path: str
):
    plt.figure()

    # Scatter plot
    plt.plot(x, y, "o", label="Merged subsets", color="blue")
    
    # Compute trendline
    coefficients = np.polyfit(x, y, deg=1)  # Linear fit (degree=1)
    trendline = np.polyval(coefficients, x)  # Evaluate the polynomial at x points

    # Plot trendline
    plt.plot(x, trendline, label="Trendline", color="red")
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.legend()  # Add legend to distinguish data points and trendline
    
    plt.savefig(save_path, dpi=400)
    plt.close()


from rich import print 
from rich.pretty import pprint

from src.tvp.data.datasets.constants import DATASETS_PAPER_TSV_20

from src.tvp.utils.io_utils import list_all_files_in_dir, import_json_from_disk
from tqdm import tqdm
import os
from src.tvp.utils.vectors import pairwise_cos_sim, pairwise_euclidean_dist
import torch

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="task_vectors.yaml")
def main(cfg: DictConfig):
    datasets = DATASETS_PAPER_TSV_20

    logger: NNLogger = init_logger(cfg)

    zs_vec = parameters_to_vector(
        parameters=get_zeroshot_model(logger).parameters()
    ).detach().cpu()

    task_vectors_dict: Dict[str, ndarray] = get_task_vectors_dict(
        datasets=datasets,
        logger=logger,
        zeroshot_vec=zs_vec
    )

    SUBSETS_DIR = "evaluations/merged_subsets/paper-tsv-20/ta/optim_adamw_wd_0.1/cosine_annealing_warmup_steps_200/subset_size_05"

    subset_config_file_path_list: List[str] = list_all_files_in_dir(SUBSETS_DIR)

    cos_sims: List[float] = []
    l2_dists: List[float] = []
    avg_norm_merged_accs: List[float] = []

    print(f"\n\n\n")

    for config_idx, subset_config_file_path in tqdm(
        iterable=enumerate(subset_config_file_path_list), 
        desc="Gathering cos sim and l2 dist vs. avg norm merged acc data",
        total=len(subset_config_file_path_list),
        colour="green"
    ):

        subset_eval_result: dict = import_json_from_disk(
            file_path=os.path.join(SUBSETS_DIR, subset_config_file_path)
        )["results"]

        task_names: List[str] = list(subset_eval_result.keys())
        task_names.pop(task_names.index("average_of_tasks"))
        avg_norm_merged_acc = subset_eval_result["average_of_tasks"]

        task_vectors_subset_dict: Dict[str, Tensor] = {
            task: task_vectors_dict[task.lower()] for task in task_names
        }

        cos_sims.append(
            float(
                pairwise_cos_sim(
                    task_vectors=torch.stack(list(task_vectors_subset_dict.values()))
                ).mean()
            )
        )

        l2_dists.append(
            float(
                pairwise_euclidean_dist(
                    task_vectors=torch.stack(list(task_vectors_subset_dict.values()))
                ).mean()
            )
        )

        avg_norm_merged_accs.append(float(avg_norm_merged_acc))

    plot(
        x=cos_sims,
        y=avg_norm_merged_accs,
        x_label="Avg all-against-all cosine similarity",
        y_label="Avg. Norm Merged Acc.",
        title="Cosine Similarity vs. Avg. Norm Merged Acc.",
        save_path="./plots/sims_and_dists_vs_avg_norm_merged_acc/cos_sims/cos_sims_vs_avg_norm_merged_acc.png"
    )

    plot(
        x=l2_dists,
        y=avg_norm_merged_accs,
        x_label="Avg all-against-all L2 distance",
        y_label="Avg. Norm Merged Acc.",
        title="L2 Distance vs. Avg. Norm Merged Acc.",
        save_path="./plots/sims_and_dists_vs_avg_norm_merged_acc/l2_dists/l2_dists_vs_avg_norm_merged_acc.png"
    )

    





if __name__ == "__main__":
    main()