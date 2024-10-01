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

from tvp.data.datasets.registry import get_dataset
from tvp.modules.encoder import ImageEncoder
from tvp.modules.heads import get_classification_head
from tvp.pl_module.image_classifier import ImageClassifier
from tvp.utils.io_utils import get_class, load_model_from_artifact
from tvp.utils.utils import LabelSmoothing, build_callbacks

from math import ceil

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
    11:"th",
    12:"th",
    13:"th",
    14:"th",
    15:"th",
    16:"th",
    17:"th",
    18:"th",
    19:"th",
    20:"th",
    21:"th",
    22:"th",
    23:"th",
    24:"th",
    25:"th",
    26:"th",
    27:"th",
    28:"th",
    29:"th",
    30:"th",
    31:"th",
    32:"th",
    33:"th",
    34:"th",
    35:"th",
}


def run(cfg: DictConfig):
    seed_index_everything(cfg)

    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )

    logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)

    #zeroshot_identifier = f"{cfg.nn.module.model.model_name}_pt" # pretrained checkpoint
    #zeroshot_identifier = f"{cfg.nn.module.model.model_name}_{cfg.nn.data.dataset.dataset_name}_0__PosthocClipping0.1" # for additional fine-tuning
    if cfg.order == 1:
        zeroshot_identifier = f"{cfg.nn.module.model.model_name}_pt" 
    else:
        #zeroshot_identifier = f"{cfg.nn.module.model.model_name}_{cfg.epochs}Eps{cfg.order - 1}{num_to_th[cfg.order - 1]}OrderUnifiedModel_0" 
        # zeroshot_identifier = f"{cfg.nn.module.model.model_name}_{cfg.merging_method}_{cfg.finetuning_method}_{cfg.epochs}Eps{cfg.order - 1}{num_to_th[cfg.order - 1]}OrderUnifiedModel_0"  
        # zeroshot_identifier = f"{cfg.nn.module.model.model_name}_{cfg.merging_method}_{cfg.finetuning_method}_avg_clipping_{cfg.epochs}Eps{cfg.order - 1}{num_to_th[cfg.order - 1]}OrderUnifiedModel_0"  
        # zeroshot_identifier = f"{cfg.nn.module.model.model_name}_{cfg.merging_method}_{cfg.finetuning_method}_unified_momentum_{cfg.epochs}Eps{cfg.order - 1}{num_to_th[cfg.order - 1]}OrderUnifiedModel_0"  
        zeroshot_identifier = f"{cfg.nn.module.model.model_name}_{cfg.merging_method}_{cfg.finetuning_method}_acc_grad_batches_{cfg.epochs}Eps{cfg.order - 1}{num_to_th[cfg.order - 1]}OrderUnifiedModel_0"  


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

    if cfg.train.save_grads:
        save_grads_dir = os.path.join(
            cfg.train.save_grads_dir, 
            f"{cfg.nn.module.model.model_name}", 
            f"{cfg.nn.data.dataset.dataset_name}",
            f"{logger._experiment.id}",
        )

        os.makedirs(save_grads_dir, exist_ok=True)

        pylogger.info(f"Saving grad norms to {save_grads_dir}")
    else:
        save_grads_dir = None

    model: ImageClassifier = hydra.utils.instantiate(
        cfg.nn.module, encoder=image_encoder, classifier=classification_head, _recursive_=False, 
        save_grad_norms=cfg.train.save_grad_norms, save_grads_dir=save_grads_dir
    )

    dataset = get_dataset(
        cfg.nn.data.train_dataset,
        preprocess_fn=model.encoder.train_preprocess,
        location=cfg.nn.data.data_path,
        batch_size=cfg.nn.data.batch_size.train,
    )

    model.freeze_head()

    callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

    storage_dir: str = cfg.core.storage_dir

    pylogger.info("Instantiating the <Trainer>")
    
    accumulate_grad_batches = 1 if cfg.accumulate_grad_batches == False else ceil(len(dataset.train_loader.dataset) / cfg.nn.data.batch_size.train)
    if accumulate_grad_batches > 1:
        pylogger.info(f"Accumulating gradients over {accumulate_grad_batches} batches")

    trainer = pl.Trainer(
        default_root_dir=storage_dir,
        plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
        #max_epochs=int(cfg.nn.data.dataset.ft_epochs/cfg.epoch_divisor),
        # max_epochs=cfg.epochs,
        max_epochs=cfg.nn.data.dataset.ft_epochs,
        logger=logger,
        callbacks=callbacks,
        accumulate_grad_batches=accumulate_grad_batches,
        **cfg.train.trainer,
    )

    pylogger.info(f"Starting fine-tuning on {cfg.ft_on_data_split} data split!")
    if cfg.ft_on_data_split == "train":
        ft_dataloader = dataset.train_loader
    elif cfg.ft_on_data_split == "val":
        ft_dataloader = dataset.test_loader
    else:
        raise ValueError(f"Unknown data split to fine-tune on: {cfg.ft_on_data_split}. Possible values: \"train\" or \"val\"")
    
    print("\n\n")
    pylogger.info("Finetuning on {} data split!".format(cfg.ft_on_data_split))
    pylogger.info("len(dataset.train_loader.dataset): {}".format(len(dataset.train_loader.dataset)))
    pylogger.info("len(dataset.test_loader.dataset): {}".format(len(dataset.test_loader.dataset)))
    trainer.fit(model=model, train_dataloaders=ft_dataloader, ckpt_path=template_core.trainer_ckpt_path)
    print("\n\n")

    pylogger.info("Starting testing!")
    trainer.test(model=model, dataloaders=dataset.test_loader)

    #artifact_name = f"{cfg.nn.module.model.model_name}_{cfg.nn.data.dataset.dataset_name}_{cfg.seed_index}_10Eps1Order"
    #artifact_name = f"{cfg.nn.module.model.model_name}_{cfg.nn.data.dataset.dataset_name}_{cfg.seed_index}_One{cfg.epoch_divisor}Eps{cfg.order}{num_to_th[cfg.order]}Order"
    #artifact_name = f"{cfg.nn.module.model.model_name}_{cfg.nn.data.dataset.dataset_name}_{cfg.seed_index}_sparseClipping{str(model.sparsity_percentile)}"
    #artifact_name = f"{cfg.nn.module.model.model_name}_{cfg.nn.data.dataset.dataset_name}_{cfg.seed_index}_2ndOrder" #2nd order means that the model is trained on the 1st order unified model
    #artifact_name = f"{cfg.nn.module.model.model_name}_{cfg.nn.data.dataset.dataset_name}_{cfg.seed_index}_7Eps1stOrder"
    #artifact_name = f"{cfg.nn.module.model.model_name}_{cfg.nn.data.dataset.dataset_name}_{cfg.seed_index}_10Eps{cfg.order}{num_to_th[cfg.order]}Order"
    # artifact_name = f"{cfg.nn.module.model.model_name}_{cfg.nn.data.dataset.dataset_name}_{cfg.seed_index}_{cfg.merging_method}_{cfg.finetuning_method}_{cfg.epochs}Eps{cfg.order}{num_to_th[cfg.order]}Order"
    # artifact_name = f"{cfg.nn.module.model.model_name}_{cfg.nn.data.dataset.dataset_name}_{cfg.seed_index}_{cfg.merging_method}_{cfg.finetuning_method}_avg_clipping_{cfg.epochs}Eps{cfg.order}{num_to_th[cfg.order]}Order"
    # artifact_name = f"{cfg.nn.module.model.model_name}_{cfg.nn.data.dataset.dataset_name}_{cfg.seed_index}_{cfg.merging_method}_{cfg.finetuning_method}_unified_momentum_{cfg.epochs}Eps{cfg.order}{num_to_th[cfg.order]}Order"
    artifact_name = f"{cfg.nn.module.model.model_name}_{cfg.nn.data.dataset.dataset_name}_{cfg.seed_index}_{cfg.merging_method}_{cfg.finetuning_method}_acc_grad_batches_{cfg.epochs}Eps{cfg.order}{num_to_th[cfg.order]}Order"

    model_class = get_class(image_encoder)
    
    #metadata = {"model_name": cfg.nn.module.model.model_name, "model_class": model_class, "strategy: ": "sparseClipping"}
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