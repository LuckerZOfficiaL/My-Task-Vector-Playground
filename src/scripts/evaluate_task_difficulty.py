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
from tvp.utils.io_utils import load_model_from_artifact
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
    logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)


    if cfg.order == 1:
        zeroshot_identifier = f"{cfg.nn.module.model.model_name}_pt"
    else:
        raise NotImplementedError("Only order 1 is supported for now")
    
    zeroshot_models = {
        dataset_name: load_model_from_artifact(artifact_path=f"{zeroshot_identifier}:latest", run=logger.experiment)
        for dataset_name in cfg.eval_datasets
    }

    seed_index_everything(cfg)

    results = {}

    for dataset_name in cfg.eval_datasets:

        classification_head_identifier = f"{cfg.nn.module.model.model_name}_{dataset_name}_head"
        classification_head = load_model_from_artifact(
            artifact_path=f"{classification_head_identifier}:latest", run=logger.experiment
        )

        model = hydra.utils.instantiate(
            cfg.nn.module, 
            encoder=zeroshot_models[dataset_name], 
            classifier=classification_head, 
            _recursive_=False
        )

        dataset = get_dataset(
            dataset_name,
            preprocess_fn=model.encoder.train_preprocess,
            location=cfg.nn.data.data_path,
            batch_size=cfg.nn.data.batch_size.train,
        )

        callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

        storage_dir: str = cfg.core.storage_dir

        pylogger.info("Instantiating the <Trainer>")
        trainer = pl.Trainer(
            default_root_dir=storage_dir,
            plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
            logger=False,
            callbacks=callbacks,
            **cfg.train.trainer,
        )

        # Evaluation
        if cfg.eval_on_train:
            pylogger.info("Evaluating on the training set")
            trainer.test(model=model, dataloaders=dataset.train_loader)

        pylogger.info("Evaluating on the test set!")
        test_results = trainer.test(model=model, dataloaders=dataset.test_loader)

        results[dataset_name] = test_results

        test_results[0]["acc/test/normalized"] = test_results[0]["acc/test"] / TASK_SPECIFIC_ACCS[dataset_name]


    print(results)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="task_vectors.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
