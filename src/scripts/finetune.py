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
from tvp.utils.io_utils import get_class, load_model_from_artifact
from tvp.utils.utils import LabelSmoothing, build_callbacks

from rich.pretty import pprint
from rich import print
import json

from tvp.utils.io_utils import load_yaml

pylogger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


def run(cfg: DictConfig):

    print(f"cfg before edits")
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    # to handle the dataset, probably the best solution is to get the dataset param from the cli
    # then load its yaml config and change it in the cfg object.
    # all other approaches fail because of how yaml merges file and stuff, PD:
    if cfg.dataset_name:
        cfg.nn.data.dataset = load_yaml(f"conf/nn/data/dataset/{cfg.dataset_name}.yaml")

    print(f"cfg after edits")
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    seed_index_everything(cfg)

    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )

    logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)

    if cfg.order == 1:
        zeroshot_identifier = f"{cfg.nn.module.model.model_name}_pt" 
    else:
        raise NotImplementedError("Only order 1 is supported for now")

    classification_head_identifier = f"{cfg.nn.module.model.model_name}_{cfg.nn.data.dataset.dataset_name}_head"

    if cfg.reset_pretrained_model:
        image_encoder: ImageEncoder = hydra.utils.instantiate(cfg.nn.module.model, keep_lang=False)
        model_class = get_class(image_encoder)

        metadata = {"model_name": cfg.nn.module.model.model_name, "model_class": model_class}
        upload_model_to_wandb(image_encoder, zeroshot_identifier, logger.experiment, cfg, metadata)

    else:
        image_encoder = load_model_from_artifact(artifact_path=f"{zeroshot_identifier}:latest", run=logger.experiment)

    if cfg.reset_classification_head:
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

    # NOTE
    # Ilharco does this in the original code...
    # 
    # When "Val" is appended to the dataset name:
    # dataset.train_loader --> 90% of train split 
    # dataset.test_loader  --> 10% of train split
    # 
    # the original test split is lost, probably intended to be used as validation...
    # 
    # For this reason, inside the get_dataset method, before assigning the 10% train split to the test split,
    # the original test split is copied inside the val split, in order to be used to do val.
    dataset = get_dataset(
        cfg.nn.data.train_dataset,
        preprocess_fn=model.encoder.train_preprocess,
        location=cfg.nn.data.data_path,
        batch_size=cfg.nn.data.batch_size.train,
    )
    pylogger.info(f"number of data sample in training set: {len(dataset.train_loader.dataset)} ({len(dataset.train_loader)} batches)")
    pylogger.info(f"number of data sample in validation set: {len(dataset.val_loader.dataset)} ({len(dataset.val_loader)} batches)")
    pylogger.info(f"number of data sample in testing set: {len(dataset.test_loader.dataset)} ({len(dataset.test_loader)} batches)")

    model.freeze_head()

    callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

    storage_dir: str = cfg.core.storage_dir

    optim_name = cfg.nn.module.optimizer._target_.split(".")[-1]

    pylogger.info("Instantiating the <Trainer>")
    trainer = pl.Trainer(
        default_root_dir=storage_dir,
        plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
        max_epochs=cfg.max_epochs,
        logger=logger,
        callbacks=callbacks,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
        limit_test_batches=cfg.limit_test_batches,
        log_every_n_steps=min(50, cfg.limit_train_batches),
        **cfg.train.trainer,
    )

    artifact_name = (
        f"{cfg.nn.module.model.model_name}_"
        f"{cfg.nn.data.dataset.dataset_name}_"
        f"{cfg.seed_index}_"
        f"batch_size_{cfg.nn.data.batch_size.train}_"
        f"lim_train_batches_{cfg.limit_train_batches}_"
        f"acc_grad_batches_{cfg.accumulate_grad_batches}_"
        f"epochs_{cfg.max_epochs}_"
        f"optim_{optim_name}_"
        f"order_{cfg.order}"
    )
    pylogger.info(f"artifact name: {artifact_name}")

    pylogger.info(f"Starting training for {trainer.max_epochs} epochs/{trainer.max_steps} steps!")
    trainer.fit(
        model=model, 
        train_dataloaders=dataset.train_loader, 
        val_dataloaders=dataset.val_loader,
        ckpt_path=template_core.trainer_ckpt_path
    )

    pylogger.info("Starting testing!")
    trainer.test(model=model, dataloaders=dataset.test_loader)

    model_class = get_class(image_encoder)
    
    metadata = {"model_name": cfg.nn.module.model.model_name, "model_class": model_class}
    upload_model_to_wandb(model.encoder, artifact_name, logger.experiment, cfg, metadata)

    if logger is not None:
        logger.experiment.finish()


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


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="finetune.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)

if __name__ == "__main__":
    main()