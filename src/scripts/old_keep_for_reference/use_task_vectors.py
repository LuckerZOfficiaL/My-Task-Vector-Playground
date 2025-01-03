import logging
import os
from typing import List, Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from lightning.pytorch import Callback
from omegaconf import DictConfig, ListConfig

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

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def run(cfg: DictConfig) -> str:
    """Generic train loop.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Returns:
        the run directory inside the storage_dir used by the current experiment
    """
    seed_index_everything(cfg)

    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )
    logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)

    zeroshot_identifier = f"{cfg.nn.module.model.model_name}_pt"

    zeroshot_model = load_model_from_artifact(artifact_path=f"{zeroshot_identifier}:latest", run=logger.experiment)

    finetuned_id_fn = lambda dataset: f"{cfg.nn.module.model.model_name}_{dataset}_{cfg.seed_index}:latest"

    finetuned_models = {
        dataset: load_model_from_artifact(artifact_path=finetuned_id_fn(dataset), run=logger.experiment)
        for dataset in cfg.task_vectors.to_apply
    }

    task_vectors = [TaskVector(zeroshot_model, finetuned_models[dataset]) for dataset in cfg.task_vectors.to_apply]

    task_vector_sum = sum(task_vectors)

    task_vector_mean = task_vector_sum / len(task_vectors)

    # Apply the resulting task vector
    image_encoder = task_vector_mean.apply_to(zeroshot_model, scaling_coef=cfg.task_vectors.scaling_coefficient)

    classification_head_identifier = f"{cfg.nn.module.model.model_name}_{cfg.nn.data.dataset.dataset_name}_head"
    classification_head = load_model_from_artifact(
        artifact_path=f"{classification_head_identifier}:latest", run=logger.experiment
    )

    model = hydra.utils.instantiate(
        cfg.nn.module, encoder=image_encoder, classifier=classification_head, _recursive_=False
    )

    dataset = get_dataset(
        cfg.nn.data.train_dataset,
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
        logger=logger,
        callbacks=callbacks,
        **cfg.train.trainer,
    )

    # pylogger.info("Evaluating on the training set")
    # trainer.test(model=model, dataloaders=dataset.train_loader)

    pylogger.info("Evaluating on the test set!")
    trainer.test(model=model, dataloaders=dataset.test_loader)

    if logger is not None:
        logger.experiment.finish()

    return None


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="task_vectors.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
