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
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Callback, LightningModule
from tqdm import tqdm

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

from tvp.data.datasets.registry import get_text_dataset
from tvp.modules.text_encoder import TextEncoder
from tvp.modules.text_heads import get_classification_head
from tvp.pl_module.text_classifier import TextClassifier
from tvp.utils.io_utils import get_class, load_model_from_artifact
from tvp.utils.utils import LabelSmoothing, build_callbacks

pylogger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")

num_to_th = {
    1: "st",
    2: "nd",
    3: "rd",
    4: "th",
    5: "th",
    6: "th",
    7: "th",
    8: "th",
    9: "th",
    10:"th",
    11: "th",
    12: "th",
    13: "th",
    14: "th",
    15: "th",
    16: "th",
    17: "th",
    18: "th",
    19: "th",
    20:"th"
}

def run(cfg: DictConfig):
    
    
    seed_index_everything(cfg)

    
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
        #zeroshot_identifier = f"{cfg.nn.module.model.model_name}_{cfg.epochs}Eps{cfg.order - 1}{num_to_th[cfg.order - 1]}OrderUnifiedModel_0" 
        zeroshot_identifier = f"{cfg.nn.module.model.model_name}_{cfg.merging_method}_{cfg.epochs}Eps{cfg.order - 1}{num_to_th[cfg.order - 1]}OrderUnifiedModel_0" 


    classification_head_identifier = f"{cfg.nn.module.model.model_name}_{cfg.nn.data.dataset.dataset_name}_head"

    
    if cfg.reset_pretrained_model:
        text_encoder: TextEncoder = hydra.utils.instantiate(cfg.nn.module.model, keep_lang=False)
        
        model_class = get_class(text_encoder)
        metadata = {"model_name": cfg.nn.module.model.model_name, "model_class": model_class}
        
        upload_model_to_wandb(
            text_encoder, zeroshot_identifier, logger.experiment, cfg, metadata
        )

    else:
        text_encoder = load_model_from_artifact(
            artifact_path=f"{zeroshot_identifier}:latest", 
            run=logger.experiment
        )

    if cfg.reset_classification_head:
        classification_head = get_classification_head(
            input_size=cfg.nn.module.model.hidden_size,
            num_classes=cfg.nn.data.dataset.num_classes
        )

        model_class = get_class(classification_head)
        
        metadata = {
            "model_name": cfg.nn.module.model.model_name,
            "model_class": model_class,
            "num_classes": cfg.nn.data.dataset.num_classes,
            "input_size": cfg.nn.module.model.hidden_size,
        }

        upload_model_to_wandb(
            classification_head, 
            classification_head_identifier, 
            logger.experiment, 
            cfg, 
            metadata=metadata
        )

    else:
        classification_head = load_model_from_artifact(
            artifact_path=f"{classification_head_identifier}:latest", 
            run=logger.experiment
        )

    model: TextClassifier = hydra.utils.instantiate(
        cfg.nn.module, 
        encoder=text_encoder, classifier=classification_head, 
        _recursive_=False
    )

    dataset = get_text_dataset(
        dataset_name=cfg.nn.data.train_dataset,
        tokenizer_name=cfg.nn.module.model.model_name,
        train_split_ratio_for_val=cfg.nn.data.splits_pct.val,
        max_seq_length=cfg.nn.data.max_seq_length,
        batch_size=cfg.nn.data.batch_size.train,
        num_workers=cfg.nn.data.num_workers.train
    )

    model.freeze_head()

    callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

    storage_dir: str = cfg.core.storage_dir

    pylogger.info("Instantiating the <Trainer>")
    trainer = pl.Trainer(
        default_root_dir=storage_dir,
        plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
        max_epochs=cfg.epochs, 
        logger=logger,
        callbacks=callbacks,
        **cfg.train.trainer,
    )

    pylogger.info("Starting training!")
    trainer.fit(
        model=model, 
        train_dataloaders=dataset.train_loader, 
        ckpt_path=template_core.trainer_ckpt_path
    )

    pylogger.info("Starting testing!")
    trainer.test(model=model, dataloaders=dataset.test_loader)


    #artifact_name = f"{cfg.nn.module.model.model_name}_{cfg.nn.data.dataset.dataset_name}_{cfg.seed_index}_{cfg.epochs}Eps{cfg.order}{num_to_th[cfg.order]}Order"
    artifact_name = f"{cfg.nn.module.model.model_name}_{cfg.nn.data.dataset.dataset_name}_{cfg.seed_index}_pcgrad_{cfg.epochs}Eps{cfg.order}{num_to_th[cfg.order]}Order"

    model_class = get_class(text_encoder)
    
    metadata = {
        "model_name": cfg.nn.module.model.model_name, 
        "model_class": model_class
    }

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