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

from typing import Union
from tvp.modules.encoder import ClassificationHead, ImageEncoder
from tvp.utils.io_utils import load_model_from_artifact

def get_zeroshot_model(logger: NNLogger) -> Union[ClassificationHead, ImageEncoder]:

    zeroshot_identifier = f"ViT-B-16_pt"
    zeroshot_model = load_model_from_artifact(
        artifact_path=f"{zeroshot_identifier}:latest", run=logger.experiment
    )

    return zeroshot_model

from torch import Tensor
from torch.nn.utils import parameters_to_vector

def get_task_vector(zeroshot_model: ImageEncoder, ft_model: ImageEncoder) -> Tensor:

    zeroshot_vec = parameters_to_vector(zeroshot_model.parameters())
    ft_vec = parameters_to_vector(ft_model.parameters())
    
    task_vector = ft_vec - zeroshot_vec

    return task_vector

from tvp.data.datasets.constants import DATASET_TO_STYLED

def compose_ft_artifact_name(
    dataset: str, 
    confl_res: str, 
    train_batches: float, 
    order: int, 
    eps_per_ord: int
) -> str:

    return (
        f"ViT-B-16"
        f"_{DATASET_TO_STYLED[dataset]}_0"
        f"_atm-true"
        f"_confl_res_{confl_res}"
        f"_train_batches_{train_batches}"
        f"_ord_{order}"
        f"_eps_per_ord_{eps_per_ord}"
        f":latest"
    )

def compose_merged_artifact_name(
    confl_res: str, 
    train_batches: float, 
    order: int, 
    eps_per_ord: int
) -> str:

    return (
        f"ViT-B-16_0_atm-true"
        f"_confl_res_{confl_res}"
        f"_train_batches_{train_batches}"
        f"_ord_{order}"
        f"_eps_per_ord_{eps_per_ord}"
    )

import torch
from torch.nn import functional as F
import numpy as np

def get_cos_sim_or_l2_dist(
    tvs_tensor: Tensor, 
    metric: str
) -> np.ndarray:

    if metric == "cos_sim":
        norm_tensor = F.normalize(tvs_tensor, p=2, dim=1)
        all_vs_all_metric = torch.mm(norm_tensor, norm_tensor.T)
    elif metric == "l2_dist":
        all_vs_all_metric = torch.cdist(tvs_tensor, tvs_tensor, p=2)

    return all_vs_all_metric.cpu().detach().numpy()

def get_cos_sim_or_l2_dist_export_path(
    metric: str, 
    file_name: str
) -> str:

    return (
        f"./evaluations/ord_vs_avg_tv_{metric}/{file_name}"
    )

from rich import print
from rich.pretty import pprint
import hydra
from nn_core.common import PROJECT_ROOT
from tvp.data.datasets.constants import DATASETS_PAPER_TA
import os
from tvp.utils.io_utils import export_json_to_disk

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="task_vectors.yaml")
def main(cfg: DictConfig) -> None:

    datasets = DATASETS_PAPER_TA

    MAX_ORDERS = 30
    CONFL_RES = "none"
    TRAIN_BATCHES = 0.1
    EPS_PER_ORD = 1

    logger = init_logger(cfg)

    zeroshot_model: ImageEncoder = get_zeroshot_model(logger)

    sim_dists_export_dir = "./evaluations/sims_dists_vs_zs"
    os.makedirs(sim_dists_export_dir, exist_ok=True)

    for order in range(1, MAX_ORDERS + 1):

        task_vectors = {}

        for dataset in datasets:
            artifact_name = compose_ft_artifact_name(
                dataset=dataset,
                confl_res=CONFL_RES, 
                train_batches=TRAIN_BATCHES, 
                order=order, 
                eps_per_ord=EPS_PER_ORD
            )

            ft_model = load_model_from_artifact(
                artifact_path=artifact_name, run=logger.experiment
            )

            task_vector = get_task_vector(zeroshot_model, ft_model)

            task_vectors[dataset] = task_vector

        tvs_tensor = torch.stack(list(task_vectors.values()))

        cos_sims = get_cos_sim_or_l2_dist(tvs_tensor, "cos_sim")
        print(f"cos sims for order {order}")
        pprint(cos_sims, expand_all=True)

        l2_dists = get_cos_sim_or_l2_dist(tvs_tensor, "l2_dist")
        print(f"l2 dists for order {order}")
        pprint(l2_dists, expand_all=True)

        print(f"\n\n")

        np.save(
            file=os.path.join(
                sim_dists_export_dir, 
                compose_merged_artifact_name(
                    confl_res=CONFL_RES, 
                    train_batches=TRAIN_BATCHES, 
                    order=order, 
                    eps_per_ord=EPS_PER_ORD
                ) + "_cos_sims"
            ),
            arr=cos_sims
        )

        np.save(
            file=os.path.join(
                sim_dists_export_dir, 
                compose_merged_artifact_name(
                    confl_res=CONFL_RES, 
                    train_batches=TRAIN_BATCHES, 
                    order=order, 
                    eps_per_ord=EPS_PER_ORD
                ) + "_l2_dists"
            ),
            arr=l2_dists
        )

    

    










if __name__ == "__main__":
    main()