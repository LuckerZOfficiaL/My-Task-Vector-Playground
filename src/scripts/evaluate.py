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
from tvp.task_vectors.task_vectors import TaskVector
from tvp.utils.io_utils import load_model_from_artifact, export_json_to_disk, export_merged_model_to_disk
from tvp.utils.utils import build_callbacks
from torch.nn.utils import vector_to_parameters
from torch.nn.utils import parameters_to_vector
from hydra.utils import instantiate
import hydra
from hydra import initialize, compose
from typing import Dict, List

from competitors.my_ties import ties_merging
from competitors.my_breadcrumbs import model_breadcrumbs
from competitors.their_ties import *
from competitors.my_dare import *

from rich.pretty import pprint


pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def run(cfg: DictConfig) -> str:

    print("evaluate cfg")
    pprint(OmegaConf.to_container(cfg, resolve=True))

    artifact_name = (
        f"{cfg.nn.module.model.model_name}_"
        f"{cfg.seed_index}_"
        f"{cfg.ft_regime}_"
        f"{cfg.optimizer_name}_"
        f"merged_{'-'.join(cfg.task_vectors.to_apply)}"
    )

    if cfg.eval_skip_if_exists and os.path.exists(f"{cfg.evaluation_export_dir}/{artifact_name}.json"):
        print(f"\n\n\n")
        pylogger.info(f"Skipping evaluation.")
        pylogger.info(f"Artifact already exists: {cfg.evaluation_export_dir}/{artifact_name}.json")
        pylogger.info(f"cfg.eval_skip_if_exists: {cfg.eval_skip_if_exists}")
        print(f"\n\n\n")

        return 

    seed_index_everything(cfg)

    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )
    logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)

    zeroshot_identifier = f"{cfg.nn.module.model.model_name}_pt"
    
    zeroshot_model = load_model_from_artifact(artifact_path=f"{zeroshot_identifier}:latest", run=logger.experiment)

    finetuned_id_fn = lambda dataset: f"{cfg.nn.module.model.model_name}_{dataset}_{cfg.seed_index}_{cfg.ft_regime}_{cfg.optimizer_name}:latest"

    finetuned_models = {
        dataset: load_model_from_artifact(artifact_path=finetuned_id_fn(dataset), run=logger.experiment)
        for dataset in cfg.task_vectors.to_apply
    }

    # Task vectors
    flatten = lambda model: parameters_to_vector(model.parameters())

    zeroshot_vec = flatten(zeroshot_model)
    task_vectors = [
        TaskVector.from_models(zeroshot_model, finetuned_models[dataset]) for dataset in cfg.task_vectors.to_apply
    ]

    def apply_task_vector(model, task_vector, scaling_coef=1):
        #model.load_state_dict({k: v + task_vector[k] for k, v in model.state_dict().items()})
        model.load_state_dict({k: v + 1/(scaling_coef)*task_vector[k] for k, v in model.state_dict().items()})

    with torch.no_grad():
        task_vectors = torch.stack(
            [flatten(finetuned_models[dataset]) - zeroshot_vec for dataset in cfg.task_vectors.to_apply]
        )
    
    print_pairwise_cos_sim(task_vectors)

    
    task_vector_aggregator = instantiate(cfg.task_vectors.aggregator)
    multi_task_vector = task_vector_aggregator(task_vectors)

    delta_model = copy.deepcopy(zeroshot_model)
    vector_to_parameters(multi_task_vector, delta_model.parameters())
    
    task_equipped_model = copy.deepcopy(zeroshot_model)
    apply_task_vector(task_equipped_model, delta_model.state_dict(), scaling_coef=cfg.task_vectors.scaling_coefficient)

    if cfg.upload_merged_to_wandb:
        metadata = {"model_name": f"{cfg.nn.module.model.model_name}", "model_class": "tvp.modules.encoder.ImageEncoder"}
        upload_model_to_wandb(task_equipped_model, artifact_name, logger.experiment, cfg, metadata)

    seed_index_everything(cfg)

    results = {}

    for dataset_name in cfg.eval_datasets:

        pylogger.info(f"Evaluating on dataset: {dataset_name}\n\n")

        classification_head_identifier = f"{cfg.nn.module.model.model_name}_{dataset_name}_head"
        classification_head = load_model_from_artifact(
            artifact_path=f"{classification_head_identifier}:latest", run=logger.experiment
        )

        model = hydra.utils.instantiate(
            cfg.nn.module, encoder=task_equipped_model, classifier=classification_head, _recursive_=False
        )

        dataset = get_dataset(
            dataset_name,
            preprocess_fn=model.encoder.train_preprocess,
            location=cfg.nn.data.data_path,
            batch_size=cfg.nn.data.batch_size.train,
        )
        pylogger.info(f"num train samples: {len(dataset.train_dataset)}, num train batches: {len(dataset.train_loader)}")
        pylogger.info(f"num test samples: {len(dataset.test_dataset)}, num test batches: {len(dataset.test_loader)}")
        print("\n\n")

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

    results["average_of_tasks"] = sum(
        item[0]['acc/test'] for item in results.values()
    ) / len(results)


    print(f"\n\n")
    pprint(results)

    export_json_to_disk(
        {
            "results": results,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
        },
        cfg.evaluation_export_dir,
        artifact_name
    )







def print_pairwise_cos_sim(task_vectors): # input shape: [num_vectors, vector_size]:
    norm_tensor = F.normalize(task_vectors, p=2, dim=1)
    cosine_similarity_matrix = torch.mm(norm_tensor, norm_tensor.T)
    cosine_similarity_matrix_np = cosine_similarity_matrix.detach().numpy()
    print(f"\n")
    pylogger.info("Pairwise Cosine Similarity Matrix:")
    pylogger.info(cosine_similarity_matrix_np)
    print(f"\n")


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






@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="task_vectors.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
