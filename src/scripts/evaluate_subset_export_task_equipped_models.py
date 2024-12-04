## Imports
import logging
import os
from typing import Dict, List, Union, Optional
import wandb
import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningModule
from tqdm import tqdm
import torch
import torch.nn as nn
from lightning.pytorch import Callback
from omegaconf import DictConfig, ListConfig
import copy
import torch.nn.functional as F
import numpy as np


from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

# Force the execution of __init__.py if this file is executed directly.
import tvp  # noqa
from tvp.data.datamodule import MetaData
from tvp.data.datasets.registry import get_dataset
from tvp.task_vectors.task_vectors import TaskVector
from tvp.utils.io_utils import load_model_from_artifact, export_model_to_disk
from tvp.utils.utils import build_callbacks
from torch.nn.utils import vector_to_parameters
from torch.nn.utils import parameters_to_vector
from hydra.utils import instantiate
import hydra
from hydra import initialize, compose
from typing import Dict, List

from tvp.competitors.my_ties import ties_merging
from tvp.competitors.my_breadcrumbs import model_breadcrumbs
from tvp.competitors.their_ties import *
from tvp.competitors.my_dare import *

import json


pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")

DATASET_NAME_TO_TA_FT_EPOCHS = {
    "Cars": 35,
    "DTD": 76,
    "EuroSAT": 12,
    "GTSRB": 11,
    "MNIST": 5,
    "RESISC45": 15,
    "SUN397": 14,
    "SVHN": 4,
    "CIFAR10": 6,
    "CIFAR100": 6,
    "STL10": 60,
    "Food101": 4,
    "Flowers102": 147,
    "FER2013": 10,
    "PCAM": 1,
    "OxfordIIITPet": 82,
    "RenderedSST2": 39,
    "EMNIST": 2,
    "FashionMNIST": 5,
    "KMNIST": 5,
}

# TODO fill these in with the correct values
TASK_SPECIFIC_ACCS = {
    "Cars": 0.8661857843399048,
    "DTD": 0.7597517967224121,
    "EuroSAT": 0.9927314519882202,
    "GTSRB": 0.9870942234992981,
    "MNIST": 0.9934999942779541,
    "RESISC45": 0.9371428489685059,
    "SUN397": 0.7319899201393127,
    "SVHN": 0.9676936268806458,
    "CIFAR10": 0.9642999768257141,
    "CIFAR100": 0.8623999953269958,
    "STL10": 0.9641249775886536,
    "Food101": 0.8809900879859924,
    "Flowers102": 0.9461700916290283,
    "FER2013": 0.7180272936820984,
    "PCAM": 0.87713623046875,
    "OxfordIIITPet": 0.8648133277893066,
    "RenderedSST2": 0.7649642825126648,
    "EMNIST": 0.9957000017166138,
    "FashionMNIST": 0.9336000084877014,
    "KMNIST": 0.9757999777793884,
}

import psutil

def get_memory_usage():
    """Returns the memory usage of the current process in MB and GB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_in_mb = memory_info.rss / (1024 ** 2)  # Convert bytes to MB
    memory_in_gb = memory_info.rss / (1024 ** 3)  # Convert bytes to GB
    return memory_in_mb, memory_in_gb


def apply_task_vector(model, task_vector, scaling_coef=1):
    #model.load_state_dict({k: v + task_vector[k] for k, v in model.state_dict().items()})
    model.load_state_dict({k: v + 1/(scaling_coef)*task_vector[k] for k, v in model.state_dict().items()})


# Make task vectors orthogonal among them
def tv_orthogonalization(vectors, method="gs"): # gs: gram schmidt
    if method == "gs":
        orthogonal_vectors = []
        for v in vectors:
            for u in orthogonal_vectors:
                v = v - (torch.dot(v, u) / torch.dot(u, u)) * u
            orthogonal_vectors.append(v)
        return torch.stack(orthogonal_vectors)
    else:
        raise ValueError("Unsupported method.")


def print_pairwise_cos_sim(task_vectors): # input shape: [num_vectors, vector_size]:
    norm_tensor = F.normalize(task_vectors, p=2, dim=1)
    cosine_similarity_matrix = torch.mm(norm_tensor, norm_tensor.T)
    cosine_similarity_matrix_np = cosine_similarity_matrix.detach().numpy()
    print("\nPairwise Cosine Similarity Matrix:")
    print(cosine_similarity_matrix_np)
    print("\n")


def generate_orthogonal_directions_for_tv(state_dict, num_directions): # returns a dictionary where keys are the parameter names and the values are many orthogonal directions
    orthogonal_directions = {}
    for key, tensor in state_dict.items():
        shape = tensor.shape
        flat_dim = tensor.numel()
        random_matrix = np.random.randn(flat_dim, num_directions)
        q, _ = np.linalg.qr(random_matrix)
        orthogonal_directions[key] = torch.tensor(q, dtype=torch.float32).view(*shape, num_directions)
    return orthogonal_directions


def project_onto_direction(tensor, direction):
    flat_tensor = tensor.view(-1)
    flat_direction = direction.view(-1)
    projection = torch.matmul(flat_tensor, flat_direction) / torch.norm(flat_direction, dim=0)**2
    projected_tensor = (flat_direction*projection).view(tensor.shape)
    return projected_tensor


def project_tv(tv, orthogonal_directions, task_id):
    projected_state_dict = {}
    for key, tensor in tv.items():
        direction = orthogonal_directions[key][..., task_id].to("cuda")
        projected_tensor = project_onto_direction(tensor, direction)
        projected_state_dict[key] = projected_tensor
    return projected_state_dict


def upload_model_to_wandb(
    model: Union[LightningModule, nn.Module], artifact_name, run, cfg: DictConfig, metadata: Dict
):
    model = model.cpu()

    pylogger.info(f"Uploading artifact {artifact_name}")

    model_artifact = wandb.Artifact(name=artifact_name, type="checkpoint", metadata=metadata)

    temp_path = "temp_checkpoint.ckpt"

    if isinstance(model, LightningModule):
        trainer = pl.Trainer(
            plugins=[NNCheckpointIO(jailing_dir="./tmp")],
        )

        trainer.strategy.connect(model)
        trainer.save_checkpoint(temp_path)

        model_artifact.add_file(temp_path + ".zip", name="trained.ckpt.zip")
        path_to_remove = temp_path + ".zip"

    else:
        torch.save(model.state_dict(), temp_path)

        model_artifact.add_file(temp_path, name="trained.ckpt")
        path_to_remove = temp_path

    run.log_artifact(model_artifact)

    os.remove(path_to_remove)


def run(cfg: DictConfig) -> str:
    
    seed_index_everything(cfg)

    
    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    
    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )
    
    
    logger: NNLogger = NNLogger(
        logging_cfg=cfg.train.logging, 
        cfg=cfg, 
        resume_id=template_core.resume_id
    )


    if cfg.order == 1:
        zeroshot_identifier = f"{cfg.nn.module.model.model_name}_pt"
    else:
        raise NotImplementedError("Only order 1 is supported for now")
    
    zeroshot_model_atm = load_model_from_artifact(
        artifact_path=f"{zeroshot_identifier}:latest", 
        run=logger.experiment
    )
    
    zeroshot_model_ta = load_model_from_artifact(
        artifact_path=f"{zeroshot_identifier}:latest", run=logger.experiment
    )

    finetuned_id_fn_atm = lambda dataset: (
        f"{cfg.nn.module.model.model_name}_"
        f"{dataset}_"
        f"{cfg.seed_index}_"
        f"epochs_1_"
        f"order_1"
        f":latest"
    )
    
    finetuned_id_fn_ta = lambda dataset: (
        f"{cfg.nn.module.model.model_name}_"
        f"{dataset}_"
        f"{cfg.seed_index}_"
        f"epochs_{DATASET_NAME_TO_TA_FT_EPOCHS[dataset]}_"
        f"order_{cfg.order}"
        f":latest"
    )


    finetuned_models_atm = {
        dataset: load_model_from_artifact(
            artifact_path=finetuned_id_fn_atm(dataset), 
            run=logger.experiment
        ) for dataset in cfg.task_vectors.to_apply
    }
    
    finetuned_models_ta = {
        dataset: load_model_from_artifact(
            artifact_path=finetuned_id_fn_ta(dataset), 
            run=logger.experiment
        ) for dataset in cfg.task_vectors.to_apply
    }


    # Task vectors
    flatten = lambda model: parameters_to_vector(model.parameters())


    zeroshot_vec_atm = flatten(zeroshot_model_atm)

    zeroshot_vec_ta = flatten(zeroshot_model_ta)


    # task_vectors_atm = [
    #     TaskVector.from_models(
    #         pretrained_model=zeroshot_model_atm, 
    #         finetuned_model=finetuned_models_atm[dataset]
    #     ) for dataset in cfg.task_vectors.to_apply
    # ]
    
    # task_vectors_ta = [
    #     TaskVector.from_models(
    #         pretrained_model=zeroshot_model_ta, 
    #         finetuned_model=finetuned_models_ta[dataset]
    #     ) for dataset in cfg.task_vectors.to_apply
    # ]

    
    with torch.no_grad():
        task_vectors_atm = torch.stack(
            [
                flatten(finetuned_models_atm[dataset]) - zeroshot_vec_atm for dataset in cfg.task_vectors.to_apply
            ]
        )
        
        task_vectors_ta = torch.stack(
            [
                flatten(finetuned_models_ta[dataset]) - zeroshot_vec_ta for dataset in cfg.task_vectors.to_apply
            ]
        )
    

    print("\nRunning vanilla merging...\n")

    print_pairwise_cos_sim(task_vectors_atm)
    
    print_pairwise_cos_sim(task_vectors_ta)

    if cfg.task_vectors.merging_method != "ties":
        task_vector_aggregator_atm = instantiate(cfg.task_vectors.aggregator)
        multi_task_vector_atm = task_vector_aggregator_atm(task_vectors_atm)
        
        task_vector_aggregator_ta = instantiate(cfg.task_vectors.aggregator)
        multi_task_vector_ta = task_vector_aggregator_ta(task_vectors_ta)

        cos_sim_atm_ta = F.cosine_similarity(multi_task_vector_atm, multi_task_vector_ta, dim=0).item()
        pylogger.info(f"\nCosine similarity between ATM and TA multi-task vectors: {cos_sim_atm_ta}\n")

        model_name_cos_sim = (
            f"{cfg.nn.module.model.model_name}_"
            f"applied_TVs_{'_'.join(cfg.task_vectors.to_apply)}_"
            f"{cfg.seed_index}_"
            f"atm_vs_ta_cos_sim"
        )
        model_path_cos_sim = f"{cfg.core.storage_dir}/{model_name_cos_sim}.json"
        pylogger.info(f"\nExporting cosine similarity to {model_path_cos_sim}\n")
        with open(model_path_cos_sim, "w") as f:
            json.dump({"cos_sim_atm_ta": cos_sim_atm_ta}, f)


    delta_model_atm = copy.deepcopy(zeroshot_model_atm)
    vector_to_parameters(multi_task_vector_atm, delta_model_atm.parameters())
    
    delta_model_ta = copy.deepcopy(zeroshot_model_ta)
    vector_to_parameters(multi_task_vector_ta, delta_model_ta.parameters())
    
    
    task_equipped_model_atm = copy.deepcopy(zeroshot_model_atm)
    apply_task_vector(
        model=task_equipped_model_atm, 
        task_vector=delta_model_atm.state_dict(), 
        scaling_coef=cfg.task_vectors.scaling_coefficient
    )
    
    task_equipped_model_ta = copy.deepcopy(zeroshot_model_ta)
    apply_task_vector(
        model=task_equipped_model_ta, 
        task_vector=delta_model_ta.state_dict(), 
        scaling_coef=cfg.task_vectors.scaling_coefficient
    )


    model_name_atm = (
        f"{cfg.nn.module.model.model_name}_"
        f"applied_TVs_{'_'.join(cfg.task_vectors.to_apply)}_"
        f"{cfg.seed_index}_"
        f"epochs_1_"
        f"order_1"
    )
    model_path_atm = f"{cfg.core.storage_dir}/{model_name_atm}.ckpt"
    pylogger.info(f"\nExporting ATM model to {model_path_atm}\n")
    export_model_to_disk(
        model=task_equipped_model_atm, 
        model_name=model_name_atm, 
        model_path=model_path_atm
    )

    model_name_ta = (
        f"{cfg.nn.module.model.model_name}_"
        f"applied_TVs_{'_'.join(cfg.task_vectors.to_apply)}_"
        f"{cfg.seed_index}_"
        f"epochs_max_"
        f"order_{cfg.order}"
    )
    model_path_ta = f"{cfg.core.storage_dir}/{model_name_ta}.ckpt"
    pylogger.info(f"\nExporting TA model to {model_path_ta}\n")
    export_model_to_disk(
        model=task_equipped_model_ta,
        model_name=model_name_ta,
        model_path=model_path_ta
    )


    


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="task_vectors.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
