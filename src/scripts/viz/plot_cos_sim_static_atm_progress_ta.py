import hydra
from nn_core.common import PROJECT_ROOT
from omegaconf import DictConfig

from nn_core.callbacks import NNTemplateCore
from nn_core.model_logging import NNLogger

from typing import Union, List, Dict
from tvp.modules.encoder import ClassificationHead, ImageEncoder

from src.tvp.data.datasets.constants import DATASETS_PAPER_TSV_20
from src.tvp.data.datasets.constants import DATASET_TO_STYLED

from tvp.utils.io_utils import load_model_from_artifact, import_json_from_disk
from torch import Tensor
from torch.nn.utils import parameters_to_vector

from torch import cosine_similarity

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

    COMPUTE_TV_SIMS: bool = False
    TV_SIMS_PATH: str = "./evaluations/atm_ta_tv_similarity/atm_ta_tv_similarity.json"

    PLOT_TV_SIMS: bool = True

    TA_PROGRESS_RATIO_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    if COMPUTE_TV_SIMS:

        logger = init_logger(cfg)

        zeroshot_model: ImageEncoder = get_zeroshot_model(logger)
    

        ft_atm_identifier = (
            f"ViT-B-16"
            f"___DATASET_NAME_PLACEHOLDER___"
            f"_0"
            f"_atm"
            f"_adamw"
            f"_wd_0.1"
            f"_lr_scheduler_cosine_annealing"
            f"_warmup_steps_0.1"
            f":latest"
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

        cos_sims: Dict[str, List[float]] = {}

        for dataset_idx, dataset in enumerate(DATASETS_PAPER_TSV_20):

            cos_sims[dataset] = []

            ft_atm_model: ImageEncoder = load_model_from_artifact(
                artifact_path=ft_atm_identifier.replace(
                    "___DATASET_NAME_PLACEHOLDER___", f"_{DATASET_TO_STYLED[dataset]}"
                ), 
                run=logger.experiment
            )

            tv_atm = get_task_vector(zeroshot_model, ft_atm_model)

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

                tv_ta = get_task_vector(zeroshot_model, ft_ta_model)

                cos_sims[dataset].append(cosine_similarity(tv_atm, tv_ta, dim=0).item())

            print("\n\n\n\n\n")

        print("\n\n\n\n\n")
        print("Computed cosine similarities:")
    else:
        cos_sims = import_json_from_disk(file_path=TV_SIMS_PATH)
        print("\n\n\n\n\n")
        print("Loaded cosine similarities:")
    
    cos_sims["average_of_tasks"] = np.array(list(cos_sims.values())).mean(axis=0).tolist()

    pprint(cos_sims, expand_all=True)

    if PLOT_TV_SIMS:

        export_file_path = "./plots/atm_ta_tv_similarity/heatmap/atm_ta_tv_similarity.png"
        os.makedirs(os.path.dirname(export_file_path), exist_ok=True)
        
        plt.figure(figsize=(12, 12))
        sns.heatmap(
            data=np.array(list(cos_sims.values())), 
            annot=True, 
            fmt=".2f", 
            cmap="RdYlGn", 
            cbar=True,
            vmin=0, 
            vmax=1,
            xticklabels=[f"{int(ratio*100)}%" for ratio in TA_PROGRESS_RATIO_LIST],
            yticklabels=[DATASET_TO_STYLED[t] for t in list(cos_sims.keys())],
        )
        plt.xlabel("TA Training Steps %")
        plt.ylabel("Task")

        plt.yticks(rotation=0, fontsize=10)  # Rotate and reduce font size
        plt.tight_layout()  # Automatically adjust layout to prevent overlapping

        plt.savefig(export_file_path, dpi=400)

        plt.close()

        ########################################################################

        # plot the heatmap as single line plots

        for dataset_idx, dataset in enumerate(cos_sims.keys()):

            export_file_path = f"./plots/atm_ta_tv_similarity/line/atm_ta_tv_similarity_{dataset}.png"
            os.makedirs(os.path.dirname(export_file_path), exist_ok=True)

            plt.figure(figsize=(10, 6))

            plt.plot(
                TA_PROGRESS_RATIO_LIST, 
                cos_sims[dataset], 
            )

            plt.xlabel("TA Training Steps %")
            plt.xticks(TA_PROGRESS_RATIO_LIST, [f"{int(ratio*100)}%" for ratio in TA_PROGRESS_RATIO_LIST])
            plt.ylabel("Cosine Similarity")

            plt.title(f"{DATASET_TO_STYLED[dataset]}: ATM vs TA Cosine Similarity")

            plt.tight_layout()

            plt.savefig(export_file_path, dpi=400)

            plt.close()





if __name__ == "__main__":
    main()