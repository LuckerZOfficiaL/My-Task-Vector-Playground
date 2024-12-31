import logging
import os
import time
from typing import Dict, List, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.nn.functional as F
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
from tvp.utils.io_utils import get_class, load_model_from_artifact, load_yaml
from tvp.utils.utils import LabelSmoothing, build_callbacks

from rich.pretty import pprint

import json

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

    if cfg.accumulate_grad_batches:
        # cfg.nn.module.optimizer.lr /= DATASET_NAME_TO_NUM_BATCHES_UPPERCASE[cfg.nn.data.dataset.dataset_name]
        cfg.nn.module.optimizer.lr /= 1

    print(f"cfg after edits")
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    seed_index_everything(cfg)

    # Instantiate template_core and logger (no changes here)
    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )
    logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)

    if cfg.order == 1:
        zeroshot_identifier = f"{cfg.nn.module.model.model_name}_pt"
    else:
        raise NotImplementedError("Only order 1 is supported for now")

    classification_head_identifier = f"{cfg.nn.module.model.model_name}_{cfg.nn.data.dataset.dataset_name}_head"

    # Handle optional reset of pretrained encoder
    if cfg.reset_pretrained_model:
        image_encoder: ImageEncoder = hydra.utils.instantiate(cfg.nn.module.model, keep_lang=False)
        model_class = get_class(image_encoder)
        metadata = {"model_name": cfg.nn.module.model.model_name, "model_class": model_class}
        upload_model_to_wandb(image_encoder, zeroshot_identifier, logger.experiment, cfg, metadata)
    else:
        image_encoder = load_model_from_artifact(
            artifact_path=f"{zeroshot_identifier}:latest", 
            run=logger.experiment
        )

    # Handle optional reset of classification head
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

    # Instantiate the final model (encoder + head)
    model: ImageClassifier = hydra.utils.instantiate(
        cfg.nn.module, 
        encoder=image_encoder, 
        classifier=classification_head, 
        _recursive_=False
    )
    model = model.half()

    # Prepare dataset
    dataset = get_dataset(
        cfg.nn.data.train_dataset,
        preprocess_fn=model.encoder.train_preprocess,
        location=cfg.nn.data.data_path,
        batch_size=cfg.nn.data.batch_size.train,
    )

    # Freeze the classification head as the original example does
    model.freeze_head()

    # Build callbacks, if needed
    callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

    storage_dir: str = cfg.core.storage_dir
    optim_name = cfg.nn.module.optimizer._target_.split(".")[-1]

    # ----------------
    # MANUAL TRAIN LOOP
    # ----------------

    # 1) Instantiate your optimizer from the cfg
    optimizer: Optimizer = hydra.utils.instantiate(cfg.nn.module.optimizer, params=model.parameters())

    # 2) (Optional) Learning rate scheduler
    #    If you have a scheduler in your config, instantiate it here:
    # scheduler = hydra.utils.instantiate(cfg.nn.module.lr_scheduler, optimizer=optimizer)
    # or set scheduler = None if you do not use one.
    scheduler = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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
    print(f"artifact name: {artifact_name}")
    pylogger.info(f"Starting training for {cfg.max_epochs} epochs (manual loop, manual optim) for {cfg.nn.data.dataset.dataset_name}")

    lr = cfg.nn.module.optimizer.lr
    accumulate_steps = cfg.accumulate_grad_batches
    clip_val = cfg.train.trainer.gradient_clip_val

    model.train()
    for epoch in range(cfg.max_epochs):
        accumulate_count = 0
        
        tqdm_dataset = tqdm(dataset.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (inputs, targets) in enumerate(tqdm_dataset):

            if batch_idx >= cfg.limit_train_batches:
                break
        
            inputs: torch.Tensor = inputs.half().to(device)
            targets: torch.Tensor = targets.to(device)
            
            outputs = model.forward(inputs)
            loss = F.cross_entropy(outputs, targets)  # classification example
            
            if accumulate_count == 0:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.zero_()

            loss.backward()
            
            accumulate_count += 1
            
            if accumulate_count == accumulate_steps:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            param.data -= lr * (param.grad / accumulate_steps)
                            # param.data -= lr * (param.grad * accumulate_steps)
                            # param.data -= lr * param.grad

                accumulate_count = 0

            tqdm_dataset.set_postfix({'loss': loss.item()})
        
        # incomplete batch
        if accumulate_count > 0:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.data -= lr * (param.grad / accumulate_count)
                        # param.data -= lr * (param.grad * accumulate_count)
                        # param.data -= lr * param.grad
            accumulate_count = 0

    # save gradients
    grads = {}
    with torch.no_grad():
        for param_name, param in model.named_parameters():
            if param.grad is not None:
                grads[param_name] = param.grad.detach().cpu()

    torch.save(grads, os.path.join(cfg.save_grads_dir, f"{artifact_name}.pt"))
                



    # ---------------------
    # TEST LOOP W/ TQDM
    # ---------------------
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    test_dataloader_tqdm = tqdm(dataset.test_loader, desc="Testing", leave=True)
    with torch.no_grad():
        for images, labels in test_dataloader_tqdm:
            images: torch.Tensor = images.half().to(device)
            labels: torch.Tensor = labels.to(device)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            test_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # You can also set a postfix for the test loop
            # test_dataloader_tqdm.set_postfix({'test_loss': loss.item()})

    test_loss /= len(dataset.test_loader)
    accuracy = correct / total
    pylogger.info(f"{cfg.nn.data.dataset.dataset_name} test Loss: {test_loss:.8f} | test acc: {accuracy:.8f}")

    # Upload final model artifact
    model_class = get_class(image_encoder)
    metadata = {"model_name": cfg.nn.module.model.model_name, "model_class": model_class}
    # upload_model_to_wandb(model.encoder, artifact_name, logger.experiment, cfg, metadata)

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