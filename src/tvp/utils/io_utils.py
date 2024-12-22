import logging
from pathlib import Path
from pydoc import locate

import torch

from nn_core.serialization import load_model

from tvp.modules.encoder import ClassificationHead, ImageEncoder

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

def export_model_to_disk(model, model_name: str, model_path: str):

    # make directory if it doesn't exist
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    pylogger.info(f"Exporting {model_name} to {model_path}")

    torch.save(model.state_dict(), model_path)


def get_class(model):
    return model.__class__.__module__ + "." + model.__class__.__qualname__
