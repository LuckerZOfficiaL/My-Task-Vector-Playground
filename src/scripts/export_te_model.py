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
from omegaconf import DictConfig, ListConfig, OmegaConf
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
from tvp.data.constants import DATASET_NAME_TO_TA_FT_EPOCHS_LOWERCASE
from tvp.data.constants import DATASET_NAME_TO_TA_FT_EPOCHS_UPPERCASE
from tvp.data.constants import DATASET_NAME_TO_NUM_TRAIN_BATCHES_LOWERCASE
from tvp.data.constants import DATASET_NAME_TO_NUM_TRAIN_BATCHES_UPPERCASE
from tvp.task_vectors.task_vectors import TaskVector
from tvp.utils.io_utils import load_model_from_artifact, export_dict_to_json, export_model_to_disk
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

from rich.pretty import pprint


pylogger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


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

    pprint(OmegaConf.to_container(cfg, resolve=True))

    optim_name = cfg.nn.module.optimizer._target_.split(".")[-1]

    artifact_name = (
        f"{cfg.nn.module.model.model_name}_"
        f"applied_task_vectors_{'-'.join(cfg.task_vectors.to_apply)}_"
        f"{cfg.seed_index}_"
        f"acc_grad_batches_{cfg.accumulate_grad_batches}_"
        f"epochs_{cfg.max_epochs if cfg.max_epochs is not None else 'TA'}_"
        f"optim_{optim_name}_"
        f"order_{cfg.order}"
    )

    task_equipped_model_export_path = os.path.join(
        cfg.task_equipped_model_export_dir, f"{artifact_name}.ckpt"
    )

    if os.path.exists(task_equipped_model_export_path):
        pylogger.info(f"Model already exists: {task_equipped_model_export_path}")
        pylogger.info("Skipping...")
        
        return task_equipped_model_export_path
    
    seed_index_everything(cfg)

    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )
    logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)


    if cfg.order == 1:
        zeroshot_identifier = f"{cfg.nn.module.model.model_name}_pt"
    else:
        raise NotImplementedError("Only order 1 is supported for now")
    
    zeroshot_model = load_model_from_artifact(artifact_path=f"{zeroshot_identifier}:latest", run=logger.experiment)
    
    finetuned_id_fn = lambda dataset: (
        f"{cfg.nn.module.model.model_name}_"
        f"{dataset}_"
        f"{cfg.seed_index}_"
        f"batch_size_{cfg.nn.data.batch_size.train}_"
        f"lim_train_batches_{cfg.limit_train_batches if cfg.limit_train_batches is not None else DATASET_NAME_TO_NUM_TRAIN_BATCHES_UPPERCASE[dataset]}_"
        f"acc_grad_batches_{cfg.accumulate_grad_batches}_"
        f"epochs_{cfg.max_epochs if cfg.max_epochs is not None else DATASET_NAME_TO_TA_FT_EPOCHS_UPPERCASE[dataset]}_"
        f"optim_{optim_name}_"
        f"order_{cfg.order}"
        f":latest"
    )

    finetuned_models = {}

    for dataset in cfg.task_vectors.to_apply:
        pylogger.info(f"Loading finetuned model for {dataset}: {finetuned_id_fn(dataset)}")
        
        finetuned_models[dataset] = load_model_from_artifact(
            artifact_path=finetuned_id_fn(dataset), 
            run=logger.experiment
        )

    # Task vectors
    flatten = lambda model: parameters_to_vector(model.parameters())

    zeroshot_vec = flatten(zeroshot_model)
    
    task_vectors = [
        TaskVector.from_models(
            pretrained_model=zeroshot_model, 
            finetuned_model=finetuned_models[dataset]
        ) for dataset in cfg.task_vectors.to_apply
    ]

    with torch.no_grad():
        task_vectors = torch.stack(
            [
                flatten(finetuned_models[dataset]) - zeroshot_vec for dataset in cfg.task_vectors.to_apply
            ]
        )
    
    if cfg.task_vectors.merging_method == "ties":
        print("\nRunning TIES...\n")
        #task_vectors = ties_merging(task_vectors, cfg.task_vectors.ties_topk)
        multi_task_vector = their_ties_merging(
            reset_type="topk",
            flat_task_checks=task_vectors, 
            reset_thresh=cfg.task_vectors.ties_topk,
            resolve_method="none",
            merge_func="mean"
        )
    
    elif cfg.task_vectors.merging_method == "breadcrumbs":
        print("\nRunning Model Breadcrumbs...\n")
        task_vectors = model_breadcrumbs(
            task_vectors,
            beta=cfg.task_vectors.breadcrumbs_beta,
            gamma=cfg.task_vectors.breadcrumbs_gamma
        )
    
    elif cfg.task_vectors.merging_method == "dare":
        print("\nRunning DARE Merging...\n")
        task_vectors = my_dare(
            task_vectors, 
            ref_model=zeroshot_model, 
            p=cfg.task_vectors.dare_rate
        )
    
    else: 
        print("\nRunning vanilla merging...\n")
    
    if cfg.task_vectors.orthogonalize:
        print("\nOrthogonalizing task vectors...\n")
        task_vectors = tv_orthogonalization(task_vectors, method='gs')

    print_pairwise_cos_sim(task_vectors)

    if cfg.task_vectors.merging_method != "ties":
        task_vector_aggregator = instantiate(cfg.task_vectors.aggregator)
        multi_task_vector = task_vector_aggregator(task_vectors)

    delta_model = copy.deepcopy(zeroshot_model)
    vector_to_parameters(multi_task_vector, delta_model.parameters())
    
    task_equipped_model = copy.deepcopy(zeroshot_model)
    apply_task_vector(
        model=task_equipped_model, 
        task_vector=delta_model.state_dict(), 
        scaling_coef=cfg.task_vectors.scaling_coefficient
    )

    export_model_to_disk(
        model=task_equipped_model, 
        model_path=task_equipped_model_export_path
    )

    return task_equipped_model_export_path




    

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="task_vectors.yaml")
def main(cfg: omegaconf.DictConfig):
    
    task_equipped_model_export_path = run(cfg)

    print(f"[export_te_model.main] task equipped model exported to: {task_equipped_model_export_path}")


if __name__ == "__main__":
    main()
