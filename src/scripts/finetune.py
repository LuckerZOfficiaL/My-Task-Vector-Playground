import logging
import os
import time
from typing import Dict, List, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, ListConfig, OmegaConf
import json
from pytorch_lightning import Callback, LightningModule
from tqdm import tqdm

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

from tvp.data.datasets.registry import get_dataset
from tvp.modules.encoder import ImageEncoder
from tvp.modules.heads import get_classification_head
from tvp.pl_module.image_classifier import ImageClassifier
from tvp.utils.io_utils import get_class, load_model_from_artifact, export_run_data_to_disk, upload_model_to_wandb
from tvp.utils.utils import LabelSmoothing, build_callbacks

from rich.pretty import pprint

pylogger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


def _replace_train_dataset(cfg: DictConfig):
    
    dataset_conf = OmegaConf.load(f"conf/nn/data/dataset/{cfg.task_to_finetune}.yaml")

    cfg.nn.data.dataset = dataset_conf

    return cfg


def _set_max_epochs(cfg: DictConfig):
    if cfg.ft_regime == "atm":
        cfg.nn.data.dataset.ft_epochs = 1
    elif cfg.ft_regime == "ta":
        cfg.nn.data.dataset.ft_epochs = cfg.nn.data.dataset.ft_epochs
    else:
        raise ValueError(f"Invalid finetuning regime: {cfg.ft_regime}")

    return cfg


def _edit_cfg(cfg: DictConfig):
    print("finetune cfg before edits")
    pprint(OmegaConf.to_container(cfg, resolve=True))
    print("\n\n")

    cfg = _replace_train_dataset(cfg)
    cfg = _set_max_epochs(cfg)

    print("finetune cfg after edits")
    pprint(OmegaConf.to_container(cfg, resolve=True))
    print("\n\n")

    return cfg

# NOTE this assume the case where only just one single lr scheduler is used!
def _get_warmpup_steps(model: ImageClassifier, cfg: DictConfig):
    if "lr_scheduler" in cfg.nn.module:
        
        if "cosine_annealing" in cfg.lr_scheduler_name:

            if isinstance(cfg.nn.module.lr_scheduler.warmup_steps_or_ratio, int):
                return cfg.nn.module.lr_scheduler.warmup_steps_or_ratio
            elif isinstance(cfg.nn.module.lr_scheduler.warmup_steps_or_ratio, float):
                return int(model.max_train_steps * cfg.nn.module.lr_scheduler.warmup_steps_or_ratio)
            else:
                raise ValueError(f"Invalid warmup step configuration for cosine annealing scheduler. Expected either an int or a float, got {type(cfg.nn.module.lr_scheduler.warmup_steps_or_ratio)}.")
    
    else:
        return None


def run(cfg: DictConfig):

    cfg = _edit_cfg(cfg=cfg)

    seed_index_everything(cfg)

    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )

    logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)

    zeroshot_identifier = f"{cfg.nn.module.model.model_name}_pt" 
    pylogger.info(f"zeroshot_identifier: {zeroshot_identifier}\n\n")

    classification_head_identifier = f"{cfg.nn.module.model.model_name}_{cfg.nn.data.dataset.dataset_name}_head"
    pylogger.info(f"classification_head_identifier: {classification_head_identifier}\n\n")

    if cfg.reset_pretrained_model:
        pylogger.info("Instantiating the <ImageEncoder> (triggered by cfg.reset_pretrained_model)\n\n")
        image_encoder: ImageEncoder = hydra.utils.instantiate(cfg.nn.module.model, keep_lang=False)
        model_class = get_class(image_encoder)

        metadata = {"model_name": cfg.nn.module.model.model_name, "model_class": model_class}
        upload_model_to_wandb(image_encoder, zeroshot_identifier, logger.experiment, cfg, metadata)

    else:
        image_encoder = load_model_from_artifact(artifact_path=f"{zeroshot_identifier}:latest", run=logger.experiment)

    if cfg.reset_classification_head:
        pylogger.info("Instantiating the <ClassificationHead> (triggered by cfg.reset_classification_head)\n\n")
        classification_head = get_classification_head(
            cfg.nn.module.model.model_name,
            cfg.nn.data.train_dataset,
            cfg.nn.data.data_path,
            cfg.misc.ckpt_path,
            cache_dir=cfg.misc.cache_dir,
            openclip_cachedir=cfg.misc.openclip_cachedir,
        )

        model_class = get_class(classification_head)
        metadata = {
            "model_name": cfg.nn.module.model.model_name,
            "model_class": model_class,
            "num_classes": cfg.nn.data.dataset.num_classes,
            "input_size": classification_head.in_features,
        }

        upload_model_to_wandb(
            classification_head, classification_head_identifier, logger.experiment, cfg, metadata=metadata
        )

    else:
        classification_head = load_model_from_artifact(
            artifact_path=f"{classification_head_identifier}:latest", 
            run=logger.experiment
        )

    model: ImageClassifier = hydra.utils.instantiate(
        cfg.nn.module, encoder=image_encoder, classifier=classification_head, _recursive_=False
    )

    dataset = get_dataset(
        cfg.nn.data.train_dataset,
        preprocess_fn=model.encoder.train_preprocess,
        location=cfg.nn.data.data_path,
        batch_size=cfg.nn.data.batch_size.train,
    )
    model.max_train_steps = len(dataset.train_loader) * cfg.nn.data.dataset.ft_epochs
    model.cosine_annealing_warmup_steps = _get_warmpup_steps(model, cfg)
    
    print("\n\n")
    pylogger.info(f"num train samples: {len(dataset.train_dataset)}, num train batches: {len(dataset.train_loader)}")
    pylogger.info(f"num test samples: {len(dataset.test_dataset)}, num test batches: {len(dataset.test_loader)}")
    pylogger.info(f"max train steps: {model.max_train_steps}")
    pylogger.info(f"cosine annealing warmup steps: {model.cosine_annealing_warmup_steps}")
    print("\n\n")

    model.freeze_head()

    callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

    storage_dir: str = cfg.core.storage_dir

    pylogger.info("Instantiating the <Trainer>")
    trainer = pl.Trainer(
        default_root_dir=storage_dir,
        # plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
        enable_checkpointing=False,
        max_epochs=cfg.nn.data.dataset.ft_epochs, 
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        **cfg.train.trainer,
    )
    print("\n\n")
    pylogger.info(f"max_epochs: {trainer.max_epochs}, max_steps: {trainer.max_steps}")
    pylogger.info(f"accumulate_grad_batches: {trainer.accumulate_grad_batches}")
    print("\n\n")

    pylogger.info("Starting training!")
    trainer.fit(model=model, train_dataloaders=dataset.train_loader, val_dataloaders=dataset.test_loader, ckpt_path=template_core.trainer_ckpt_path)

    pylogger.info("Starting testing!")
    trainer.test(model=model, dataloaders=dataset.test_loader)

    # NOTE only works if one lr_scheduler is used
    lr_scheduler_warmup_steps = f"_warmup_steps_{cfg.nn.module.lr_scheduler.warmup_steps_or_ratio}" if model.lr_schedulers() is not None else ""
    artifact_name = (
        f"{cfg.nn.module.model.model_name}"
        f"_{cfg.nn.data.dataset.dataset_name}"
        f"_{cfg.seed_index}"
        f"_{cfg.ft_regime}"
        f"_{cfg.optimizer_name}"
        f"_wd_{cfg.nn.module.optimizer.weight_decay}"
        f"_lr_scheduler_{cfg.lr_scheduler_name}"
        f"{lr_scheduler_warmup_steps}"
    )

    print("\n\n")
    pylogger.info(f"artifact_name: {artifact_name}")

    print("\n\n")
    pylogger.info(f"optimizer(s): {model.optimizers()}")
    pylogger.info(f"lr_scheduler(s): {model.lr_schedulers()}")
    print("\n\n")

    model_class = get_class(image_encoder)
    metadata = {"model_name": cfg.nn.module.model.model_name, "model_class": model_class}
    upload_model_to_wandb(model.encoder, artifact_name, logger.experiment, cfg, metadata)

    if logger is not None:
        logger.experiment.finish()

    if cfg.timestamp is not None:
        export_run_data_to_disk(
            cfg=cfg, 
            logger=logger, 
            export_dir=f"./evaluations/ft/{cfg.ft_regime}/{cfg.optimizer_name}/lr_scheduler_{cfg.lr_scheduler_name}{lr_scheduler_warmup_steps}", 
            file_base_name=artifact_name
        )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="finetune.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)

if __name__ == "__main__":
    main()