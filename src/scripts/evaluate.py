## Imports
import logging
import os
from typing import Dict, List, Union, Optional, Tuple
import wandb
import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningModule
from tvp.modules.encoder import ClassificationHead, ImageEncoder
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import Tensor
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
from tvp.utils.io_utils import load_model_from_artifact, export_json_to_disk, export_merged_model_to_disk, upload_model_to_wandb
from tvp.utils.utils import build_callbacks
from torch.nn.utils import vector_to_parameters
from torch.nn.utils import parameters_to_vector
from hydra.utils import instantiate
import hydra
from hydra import initialize, compose
from typing import Dict, List, Callable

from tvp.utils.vectors import print_pairwise_cos_sim, print_pairwise_euclidean_dist, orthogonalize_task_vectors, apply_task_vector

from rich.pretty import pprint


pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def run(cfg: DictConfig) -> str:

    print("evaluate cfg")
    pprint(OmegaConf.to_container(cfg, resolve=True))

    lr_scheduler_warmup_steps = "" if cfg.nn.module.lr_scheduler.warmup_steps_or_ratio is None else f"_warmup_steps_{cfg.nn.module.lr_scheduler.warmup_steps_or_ratio}"
    
    if cfg.eval_orthogonalization_method == "none":
        orthogonalization_method = ""
    elif cfg.eval_orthogonalization_method == "pc_grad":
        orthogonalization_method = "_pc_grad"
    elif cfg.eval_orthogonalization_method == "sorted_pc_grad":
        orthogonalization_method = "_sorted_pc_grad"
    else:
        raise ValueError(f"Unknown orthogonalization method: {cfg.eval_orthogonalization_method}")
    
    artifact_name = (
        f"{cfg.nn.module.model.model_name}"
        f"_{cfg.seed_index}"
        f"_{cfg.ft_regime}"
        f"_{cfg.optimizer_name}"
        f"_wd_{cfg.nn.module.optimizer.weight_decay}"
        f"_lr_scheduler_{cfg.lr_scheduler_name}"
        f"{lr_scheduler_warmup_steps}"
        f"{orthogonalization_method}"
        f"_merged_{'-'.join(cfg.task_vectors.to_apply)}"
    )

    print(f"\n\n\n")
    pylogger.info(f"Merrged artifact name: {artifact_name}")
    print(f"\n\n\n")

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
    
    zeroshot_model: ImageEncoder = load_model_from_artifact(
        artifact_path=f"{zeroshot_identifier}:latest", run=logger.experiment
    )

    finetuned_id_fn = lambda dataset: (
        f"{cfg.nn.module.model.model_name}"
        f"_{dataset}"
        f"_{cfg.seed_index}"
        f"_{cfg.ft_regime}"
        f"_{cfg.optimizer_name}"
        f"_wd_{cfg.nn.module.optimizer.weight_decay}"
        f"_lr_scheduler_{cfg.lr_scheduler_name}"
        f"{lr_scheduler_warmup_steps}"
        f":latest"
    )

    eval_results, cos_sims, euclidean_dists = eval(
        finetuned_id_fn=finetuned_id_fn,
        logger=logger,
        cfg=cfg,
        zeroshot_model=zeroshot_model,
        artifact_name=artifact_name,
        template_core=template_core,
    )        

    export_json_to_disk(
        {
            "results": eval_results,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
        },
        cfg.evaluation_export_dir,
        artifact_name
    )


def eval_merged_model(
    cfg: DictConfig,
    task_equipped_model,
    template_core,
    logger,

):  
    results = {}

    for dataset_idx, dataset_name in enumerate(cfg.eval_datasets):

        pylogger.info(f"Evaluating on dataset: {dataset_name} ({dataset_idx + 1}/{len(cfg.eval_datasets)})\n\n")

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

    return results


def eval(
    finetuned_id_fn: Callable[[str], str],
    logger: NNLogger,
    cfg: DictConfig,
    zeroshot_model: ImageEncoder,
    artifact_name: str,
    template_core: NNTemplateCore,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    finetuned_models: Dict[str, ImageEncoder] = {
        dataset: load_model_from_artifact(artifact_path=finetuned_id_fn(dataset), run=logger.experiment)
        for dataset in cfg.task_vectors.to_apply
    }

    # Task vectors
    flatten: Callable[[ImageEncoder], Tensor] = lambda model: parameters_to_vector(model.parameters())

    with torch.no_grad():
        task_vectors: Dict[str, Tensor] = {
            dataset: flatten(finetuned_models[dataset]) - flatten(zeroshot_model) for dataset in cfg.task_vectors.to_apply
        }
    
    pylogger.info(f"pairwise cosine similarity between task vectors before orthogonalization:")
    cos_sims: np.ndarray = print_pairwise_cos_sim(torch.stack(list(task_vectors.values())))

    pylogger.info(f"pairwise euclidean distance between task vectors before orthogonalization:")
    euclidean_dists: np.ndarray = print_pairwise_euclidean_dist(torch.stack(list(task_vectors.values())))

    task_vectors = orthogonalize_task_vectors(task_vectors, cfg, artifact_name)

    pylogger.info(f"pairwise cosine similarity between task vectors after orthogonalization:")
    cos_sims: np.ndarray = print_pairwise_cos_sim(torch.stack(list(task_vectors.values())))

    pylogger.info(f"pairwise euclidean distance between task vectors after orthogonalization:")
    euclidean_dists: np.ndarray = print_pairwise_euclidean_dist(torch.stack(list(task_vectors.values())))
 
    task_vector_aggregator = instantiate(cfg.task_vectors.aggregator)
    multi_task_vector = task_vector_aggregator(torch.stack(list(task_vectors.values())))

    delta_model = copy.deepcopy(zeroshot_model)
    vector_to_parameters(multi_task_vector, delta_model.parameters())
    
    task_equipped_model = copy.deepcopy(zeroshot_model)
    apply_task_vector(task_equipped_model, delta_model.state_dict(), scaling_coef=cfg.task_vectors.scaling_coefficient)

    if cfg.upload_merged_to_wandb:
        metadata = {"model_name": f"{cfg.nn.module.model.model_name}", "model_class": "tvp.modules.encoder.ImageEncoder"}
        upload_model_to_wandb(task_equipped_model, artifact_name, logger.experiment, cfg, metadata)

    seed_index_everything(cfg)

    # eval_results = eval_merged_model(
    #     cfg=cfg, 
    #     # task_equipped_model=task_equipped_model, 
    #     task_equipped_model=copy.deepcopy(zeroshot_model), 
    #     template_core=template_core, 
    #     logger=logger
    # )
    eval_results = {"key": -421337}

    print(f"\n\n")
    pylogger.info(f"[evaluate.eval()] eval_results:")
    pprint(eval_results, expand_all=True)

    return eval_results, cos_sims, euclidean_dists


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="task_vectors.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
