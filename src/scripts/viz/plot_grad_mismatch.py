import hydra
from nn_core.common import PROJECT_ROOT
from omegaconf import DictConfig

from nn_core.callbacks import NNTemplateCore
from nn_core.model_logging import NNLogger

from typing import Union, List, Dict
from tvp.modules.encoder import ClassificationHead, ImageEncoder

from src.tvp.data.datasets.constants import DATASETS_PAPER_TSV_20
from src.tvp.data.datasets.constants import DATASET_TO_STYLED
from src.tvp.data.datasets.constants import DATASET_TO_NUM_BATCHES

import torch
from torch import cosine_similarity
from tvp.utils.io_utils import load_model_from_artifact, import_json_from_disk, export_json_to_disk
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from copy import deepcopy

import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from rich import print
from rich.pretty import pprint


def init_logger(cfg: DictConfig) -> NNLogger:
    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None)
    )
    
    logger: NNLogger = NNLogger(
        logging_cfg=cfg.train.logging, 
        cfg=cfg, 
        resume_id=template_core.resume_id
    )

    return logger


def get_zeroshot_model(logger: NNLogger) -> Union[ClassificationHead, ImageEncoder]:

    zeroshot_identifier = f"ViT-B-16_pt"
    zeroshot_model = load_model_from_artifact(
        artifact_path=f"{zeroshot_identifier}:latest", run=logger.experiment
    )

    return zeroshot_model


def get_task_vector(zeroshot_model: ImageEncoder, ft_model: ImageEncoder) -> Tensor:

    zeroshot_vec = parameters_to_vector(zeroshot_model.parameters())
    ft_vec = parameters_to_vector(ft_model.parameters())
    
    task_vector = ft_vec - zeroshot_vec

    return task_vector


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="task_vectors.yaml")
def main(cfg: DictConfig):

    COMPUTE_GRAD_MISMATCH: bool = True
    GRAD_MISMATCH: str = "./evaluations/grad_mismatch/grad_mismatch.json"

    PLOT_GRAD_MISMATCH: bool = True

    TA_PROGRESS_RATIO_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    if COMPUTE_GRAD_MISMATCH:

        logger = init_logger(cfg)

        zeroshot_model: ImageEncoder = get_zeroshot_model(logger)
    
        grad_atm_identifier = (
            f"./grads/"
            f"ViT-B-16"
            f"___DATASET_NAME_PLACEHOLDER___"
            f"_0"
            f"_atm"
            f"_adamw"
            f"_wd_0.1"
            f"_lr_scheduler_cosine_annealing"
            f"_warmup_steps_0.1"
            f"_step_1.0"
            f"_grads"
            f"_acc_grad_batches____DATASET_NUM_BATCHES_PLACEHOLDER___"
            f".pt"
        )

        ft_ta_identifier = (
            f"ViT-B-16"
            f"___DATASET_NAME_PLACEHOLDER___"
            f"_0"
            f"_ta"
            f"_adamw"
            f"_wd_0.1"
            f"_lr_scheduler_cosine_annealing"
            f"_warmup_steps_200"
            f"___STEP_RATIO_PLACEHOLDER___"
            f":latest"
        )

        grad_mismatches: Dict[str, List[float]] = {}

        for dataset_idx, dataset in enumerate(DATASETS_PAPER_TSV_20):

            grad_mismatches[dataset] = []

            grad_atm_named_parameters: Dict[str, Tensor] = torch.load(
                grad_atm_identifier.replace(
                    "___DATASET_NAME_PLACEHOLDER___", f"_{DATASET_TO_STYLED[dataset]}"
                ).replace(
                    "___DATASET_NUM_BATCHES_PLACEHOLDER___", str(DATASET_TO_NUM_BATCHES[dataset][cfg.nn.data.batch_size.train])
                )
            )

            for ta_progress_idx, ta_progress_ratio in enumerate(TA_PROGRESS_RATIO_LIST):

                print(f"Dataset: {dataset} ({dataset_idx + 1}/{len(DATASETS_PAPER_TSV_20)}), TA Progress Ratio: {ta_progress_ratio} ({ta_progress_idx + 1}/{len(TA_PROGRESS_RATIO_LIST)})")
                
                ft_ta_model: ImageEncoder = load_model_from_artifact(
                    artifact_path=ft_ta_identifier.replace(
                        "___DATASET_NAME_PLACEHOLDER___", f"_{DATASET_TO_STYLED[dataset]}"
                    ).replace(
                        "___STEP_RATIO_PLACEHOLDER___", f"_step_{ta_progress_ratio}"
                    ), 
                    run=logger.experiment
                )

                tv_ta_vec = get_task_vector(zeroshot_model, ft_ta_model)
                vector_to_parameters(tv_ta_vec, ft_ta_model.parameters())

                grad_atm_model: ImageEncoder = deepcopy(zeroshot_model)

                # TODO some grads are None. Replace them with zero tensors.
                #      take the shape from the corresponding parameter in the zeroshot model
                #      and replace the None with a zero tensor of that shape
                for name, grad in grad_atm_named_parameters.items():
                    if grad_atm_named_parameters[name] is None:
                        grad_atm_named_parameters[name] = torch.zeros_like(dict(ft_ta_model.named_parameters())[name])
                        
                        if dict(ft_ta_model.named_parameters())[name].allclose(torch.zeros_like(dict(ft_ta_model.named_parameters())[name])):
                            print(f"Replaced None with zeros for {name} successfully!")
                        else:
                            print(f"Replaced None with zeros for {name} NOT successfully!")

                grad_atm_model.load_state_dict(grad_atm_named_parameters)
                grad_atm_vec = parameters_to_vector(grad_atm_model.parameters())

                grad_mismatches[dataset].append(
                    cosine_similarity(tv_ta_vec, -1 * grad_atm_vec, dim=0).item()
                )

                

            print("\n\n\n\n\n")

        print("\n\n\n\n\n")
        print("Computed cosine similarities:")

        os.makedirs(os.path.dirname(GRAD_MISMATCH), exist_ok=True)
        export_json_to_disk(
            data=grad_mismatches, 
            export_dir=os.path.dirname(GRAD_MISMATCH), 
            file_name=os.path.basename(GRAD_MISMATCH.replace(".json", ""))
        )

    else:
        grad_mismatches = import_json_from_disk(file_path=GRAD_MISMATCH)
        print("\n\n\n\n\n")
        print("Loaded cosine similarities:")
    
    pprint(grad_mismatches, expand_all=True)

    if PLOT_GRAD_MISMATCH:

        export_file_path = "./plots/grad_mismatch/grad_mismatch.png"
        os.makedirs(os.path.dirname(export_file_path), exist_ok=True)
        
        plt.figure(figsize=(12, 12))
        sns.heatmap(
            data=np.array(list(grad_mismatches.values())), 
            annot=True, 
            fmt=".2f", 
            cmap="RdYlGn", 
            cbar=True,
            vmin=0, 
            vmax=1,
            xticklabels=[f"{int(ratio*100)}%" for ratio in TA_PROGRESS_RATIO_LIST],
            yticklabels=[DATASET_TO_STYLED[t] for t in list(grad_mismatches.keys())],
        )
        plt.xlabel("TA Training Steps %")
        plt.ylabel("Task")

        plt.yticks(rotation=0, fontsize=10)  # Rotate and reduce font size
        plt.tight_layout()  # Automatically adjust layout to prevent overlapping

        plt.savefig(export_file_path, dpi=400)





if __name__ == "__main__":
    main()