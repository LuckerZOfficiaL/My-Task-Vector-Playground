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

from tvp.data.datasets.constants import DATASET_TO_STYLED

def get_artifact_name_merged(
    atm_version: str,
    confl_res: str, 
    train_batches: float, 
    order: int, 
    eps_per_ord: int
) -> str:

    return (
        f"ViT-B-16_0"
        f"_{atm_version}"
        f"_confl_res_{confl_res}"
        f"_train_batches_{train_batches}"
        f"_ord_{order}"
        f"_eps_per_ord_{eps_per_ord}"
        f"_merged"
        f":latest"
    )

from typing import Tuple, Dict
import numpy as np
from sklearn.decomposition import PCA

def perform_pca(
    data_dict: dict, 
    num_components: int,
    pca_export_path: Union[str, None]
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    checkpoints = np.array(list(data_dict.values()))
    print(f"checkpoints.shape: {checkpoints.shape}")

    pca = PCA(n_components=num_components)
    checkpoints_reduced = pca.fit_transform(checkpoints)

    # zs_reduced = checkpoints_reduced[-1, :]
    # print(f"zs_reduced.shape: {zs_reduced.shape}")

    checkpoints_reduced_dict = {}
    for idx, key in enumerate(data_dict.keys()):
        checkpoints_reduced_dict[key] = checkpoints_reduced[idx, :]
    
    pca_stats = {
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "explained_variance": pca.explained_variance_,
        "singular_values": pca.singular_values_,
        "mean": pca.mean_,
        "components": pca.components_,
    }

    if pca_export_path is not None:
        np.save(pca_export_path.replace(".npy", "_ndarray.npy"), checkpoints_reduced)
        np.save(pca_export_path.replace(".npy", "_dict.npy"), checkpoints_reduced_dict)
        np.save(pca_export_path.replace(".npy", "_stats.npy"), pca_stats)

    return checkpoints_reduced, checkpoints_reduced_dict, pca_stats

from rich import print
from rich.pretty import pprint
import hydra
from nn_core.common import PROJECT_ROOT
from tvp.data.datasets.constants import DATASETS_PAPER_TA
import os
from tvp.utils.io_utils import export_json_to_disk
import torch
from torch.nn.utils import parameters_to_vector

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="task_vectors.yaml")

def main(cfg: DictConfig) -> None:

    logger = init_logger(cfg)

    MAX_ORDERS = 10

    checkpoints = {}

    checkpoints["zeroshot"] = parameters_to_vector(
        get_zeroshot_model(logger).parameters()
    ).detach().cpu().numpy()

    checkpoints["breadcrumbs"] = parameters_to_vector(
        load_model_from_artifact(
            artifact_path=get_artifact_name_merged(
                atm_version="atm-true",
                confl_res="bc",
                train_batches=1.0,
                order=1,
                eps_per_ord=10
            ),
            run=logger.experiment
        ).parameters()
    ).detach().cpu().numpy()

    checkpoints["ties"] = parameters_to_vector(
        load_model_from_artifact(
            artifact_path=get_artifact_name_merged(
                atm_version="atm-true",
                confl_res="ties",
                train_batches=1.0,
                order=1,
                eps_per_ord=10
            ),
            run=logger.experiment
        ).parameters()
    ).detach().cpu().numpy()

    checkpoints["dare"] = parameters_to_vector(
        load_model_from_artifact(
            artifact_path=get_artifact_name_merged(
                atm_version="atm-true",
                confl_res="dare",
                train_batches=1.0,
                order=1,
                eps_per_ord=10
            ),
            run=logger.experiment
        ).parameters()
    ).detach().cpu().numpy()

    checkpoints["ta"] = parameters_to_vector(
        load_model_from_artifact(
            artifact_path=get_artifact_name_merged(
                atm_version="atm-true",
                confl_res="none",
                train_batches=1.0,
                order=1,
                eps_per_ord=10
            ),
            run=logger.experiment
        ).parameters()
    ).detach().cpu().numpy()

    for order in range(1, MAX_ORDERS + 1):

        checkpoints[f"atm-true order {order}"] = parameters_to_vector(
            load_model_from_artifact(
                artifact_path=get_artifact_name_merged(
                    atm_version="atm-true",
                    confl_res="none", 
                    train_batches=1.0, 
                    order=order, 
                    eps_per_ord=1
                ), 
                run=logger.experiment
            ).parameters()
        ).detach().cpu().numpy()
        
        checkpoints[f"atm-denoise order {order}"] = parameters_to_vector(
            load_model_from_artifact(
                artifact_path=get_artifact_name_merged(
                    atm_version="atm-denoise",
                    confl_res="none", 
                    train_batches=1.0, 
                    order=order, 
                    eps_per_ord=1
                ), 
                run=logger.experiment
            ).parameters()
        ).detach().cpu().numpy()

    pca_export_path = f"evaluations/pca_trajectory/pca_trajectory.npy"
    os.makedirs(os.path.dirname(pca_export_path), exist_ok=True)
    checkpoints_reduced, checkpoints_reduced_dict, pca_stats = perform_pca(
        data_dict=checkpoints, 
        num_components=2,
        pca_export_path=pca_export_path
    )

    print("pca stats:")
    pprint(pca_stats, expand_all=True)
    



if __name__ == "__main__":
    main()