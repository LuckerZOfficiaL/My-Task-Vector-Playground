import logging
from pathlib import Path
from pydoc import locate

import torch

from nn_core.serialization import load_model

from tvp.modules.encoder import ClassificationHead, ImageEncoder

import yaml
from typing import Dict, List

from nn_core.model_logging import NNLogger
import wandb
from omegaconf import DictConfig

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

def export_model_to_disk(model, model_path: str):
    
    pylogger.info(f"Saving model to {model_path}")

    torch.save(model.state_dict(), model_path)

def load_yaml(path: str) -> Dict:
    try:
        with open(path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        pylogger.error(f"Error loading yaml file: {e}")
        return None


import os
import json
import pandas as pd

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


def export_dict_to_json(
    data: dict, 
    filename: str,
    export_description: str="dictionary"
):
    """
    Export a dictionary to a JSON file.

    :param data: The dictionary to export
    :param filename: The name of the JSON file to create
    """
    try:
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        pylogger.info(f"{export_description} successfully exported to {filename}")
    except Exception as e:
        pylogger.error(f"An error occurred while exporting the {export_description}: {e}")


def load_dict_from_json(
    filename: str,
    import_description: str="dictionary"
) -> Dict:
    """
    Load a dictionary from a JSON file.

    :param filename: The name of the JSON file to load
    :return: The dictionary
    """
    try:
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
        return data
    except Exception as e:
        pylogger.error(f"An error occurred while importing the {import_description}: {e}")
        return None


def export_list_of_strings_to_file(
    data: List[str], 
    filename: str,
    export_description: str="list of strings"
):
    """
    Export a list of strings to a text file.

    :param data: The list of strings to export
    :param filename: The name of the text file to create
    """
    try:
        with open(filename, 'w') as file:
            for item in data:
                file.write(f"{item}\n")
        pylogger.info(f"{export_description} successfully exported to {filename}")
    except Exception as e:
        pylogger.error(f"An error occurred while exporting the {export_description}: {e}")


def load_list_of_strings_from_file(
    filename: str,
    import_description: str="list of strings"
) -> List[str]:
    """
    Load a list of strings from a text file.

    :param filename: The name of the text file to load
    :return: The list of strings
    """
    try:
        with open(filename, 'r') as file:
            data = file.readlines()
        return [item.strip() for item in data]
    except Exception as e:
        pylogger.error(f"An error occurred while importing the {import_description}: {e}")
        return None


def get_all_files_in_dir(dir_path: str) -> List[str]:
    return [file for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, file))]


def get_class(model):
    return model.__class__.__module__ + "." + model.__class__.__qualname__
