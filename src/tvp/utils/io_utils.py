import logging
from pathlib import Path
from pydoc import locate

import torch

from nn_core.serialization import load_model

from omegaconf import DictConfig, OmegaConf
from nn_core.model_logging import NNLogger
import os
import json

from tvp.modules.encoder import ClassificationHead, ImageEncoder

from typing import Union, Dict
from pytorch_lightning import LightningModule
import wandb
from torch import nn
import lightning as pl
from nn_core.serialization import NNCheckpointIO
import os

pylogger = logging.getLogger(__name__)


def load_model_from_artifact(run, artifact_path):
    pylogger.info(f"Loading model from artifact {artifact_path}")

    artifact = run.use_artifact(artifact_path)
    artifact.download()

    ckpt_path = Path(artifact.file())

    model_class = locate(artifact.metadata["model_class"])

    if model_class == ImageEncoder:
        model = model_class(**artifact.metadata)
    elif model_class == ClassificationHead:
        model = model_class(normalize=True, **artifact.metadata)

    model.load_state_dict(torch.load(ckpt_path))

    return model


def export_run_data_to_disk(
    cfg: DictConfig, 
    logger: NNLogger,
    export_dir: str,
    file_base_name: str
):
    import wandb
    
    # Authenticate with Weights & Biases
    wandb.login()
    api = wandb.Api()

    # Fetch the run details from W&B
    run = api.run(f"{cfg.core.entity}/{cfg.core.project_name}/{logger.experiment.id}")

    # Retrieve the run's history
    history = run.history()

    # Convert the history to a list of dictionaries
    epoch_data = history.to_dict('records')

    # Ensure the export directory exists
    os.makedirs(export_dir, exist_ok=True)

    # Define file paths
    json_file_path = os.path.join(export_dir, f"{file_base_name}_epoch_data.json")
    csv_file_path = os.path.join(export_dir, f"{file_base_name}_history.csv")

    # Export epoch data as a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(epoch_data, json_file, indent=4)

    # Export the history DataFrame as a CSV file
    history.to_csv(csv_file_path, index=False)

    pylogger.info(f"Epoch data exported to: {json_file_path}")
    pylogger.info(f"History DataFrame exported to: {csv_file_path}")

    export_json_to_disk(
        OmegaConf.to_container(cfg, resolve=True), 
        export_dir, 
        f"{file_base_name}_cfg"
    )


def export_json_to_disk(data: dict, export_dir: str, file_name: str):
    # Ensure the export directory exists
    os.makedirs(export_dir, exist_ok=True)

    # Define file path
    file_path = os.path.join(export_dir, f"{file_name}.json")

    # Export data as a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    pylogger.info(f"Data exported to: {file_path}")


def import_json_from_disk(file_path: str):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    return data


def export_merged_model_to_disk(
    merged_model,
    cfg: dict,
    export_dir: str,
    model_name: str
):
    
    os.makedirs(export_dir, exist_ok=True)

    model_path = os.path.join(export_dir, f"{model_name}.pt")

    torch.save(merged_model.state_dict(), model_path)

    pylogger.info(f"Model exported to: {model_path}")


def list_all_files_in_dir(dir_path: str):
    return [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]


def upload_model_to_wandb(
    model: Union[LightningModule, nn.Module], artifact_name, run, cfg: DictConfig, metadata: Dict
):
    # model = model.cpu()

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


def get_class(model):
    return model.__class__.__module__ + "." + model.__class__.__qualname__
