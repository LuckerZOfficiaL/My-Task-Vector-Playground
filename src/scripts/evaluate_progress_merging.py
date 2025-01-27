## Imports
import logging
import os
from typing import Dict, List, Union, Optional
import wandb
import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningModule
from tvp.modules.encoder import ClassificationHead, ImageEncoder
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import Tensor
from lightning.pytorch import Callback
from omegaconf import DictConfig, ListConfig, OmegaConf
import copy
import torch.nn.functional as F
import numpy as np


from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

# Force the execution of __init__.py if this file is executed directly.
import tvp  # noqa
from tvp.data.datamodule import MetaData
from tvp.data.datasets.registry import get_dataset
from tvp.task_vectors.task_vectors import TaskVector
from tvp.utils.io_utils import load_model_from_artifact, export_json_to_disk, export_merged_model_to_disk, upload_model_to_wandb
from tvp.utils.utils import build_callbacks
from torch.nn.utils import vector_to_parameters
from torch.nn.utils import parameters_to_vector
from hydra.utils import instantiate
import hydra
from hydra import initialize, compose
from typing import Dict, List, Callable

from tvp.utils.vectors import print_pairwise_cos_sim, orthogonalize_task_vectors, apply_task_vector
from scripts.evaluate import eval

from rich.pretty import pprint


pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def run(cfg: DictConfig) -> str:

    print(f"\n\n\n\n\n\n\n\n")
    print(f"evaluate_progress_merging")

    print("evaluate cfg")
    pprint(OmegaConf.to_container(cfg, resolve=True))

    lr_scheduler_warmup_steps = "" if cfg.nn.module.lr_scheduler.warmup_steps_or_ratio is None else f"_warmup_steps_{cfg.nn.module.lr_scheduler.warmup_steps_or_ratio}"
    
    if cfg.eval_orthogonalization_method == "none":
        orthogonalization_method = ""
    elif cfg.eval_orthogonalization_method == "pc_grad":
        orthogonalization_method = "_pc_grad"
    elif cfg.eval_orthogonalization_method == "sorted_pc_grad":
        orthogonalization_method = "_sorted_pc_grad"
    else:
        raise ValueError(f"Unknown orthogonalization method: {cfg.eval_orthogonalization_method}")
    
    artifact_name = (
        f"{cfg.nn.module.model.model_name}"
        f"_{cfg.seed_index}"
        f"_{cfg.ft_regime}"
        f"_{cfg.optimizer_name}"
        f"_wd_{cfg.nn.module.optimizer.weight_decay}"
        f"_lr_scheduler_{cfg.lr_scheduler_name}"
        f"{lr_scheduler_warmup_steps}"
        f"{orthogonalization_method}"
        f"_merged_{'-'.join(cfg.task_vectors.to_apply)}"
    )

    print(f"\n\n\n")
    pylogger.info(f"Merrged artifact name: {artifact_name}")
    print(f"\n\n\n")

    

    seed_index_everything(cfg)

    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )
    logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)

    zeroshot_identifier = f"{cfg.nn.module.model.model_name}_pt"
    
    zeroshot_model: ImageEncoder = load_model_from_artifact(
        artifact_path=f"{zeroshot_identifier}:latest", run=logger.experiment
    )

    eval_results: Dict[str, dict] = {}
    cos_sims: Dict[str, np.ndarray] = {}
    euclidean_dists: Dict[str, np.ndarray] = {}

    for ratio in cfg.eval_merged_ratios:

        print(f"\n\n\n")
        pylogger.info(f"Running evaluation for ratio {ratio}")

        eval_results[ratio] = {}
        pylogger.info(f"{eval_results}")

        finetuned_id_fn = lambda dataset: (
            f"{cfg.nn.module.model.model_name}"
            f"_{dataset}"
            f"_{cfg.seed_index}"
            f"_{cfg.ft_regime}"
            f"_{cfg.optimizer_name}"
            f"_wd_{cfg.nn.module.optimizer.weight_decay}"
            f"_lr_scheduler_{cfg.lr_scheduler_name}"
            f"{lr_scheduler_warmup_steps}"
            f"_step_{ratio}"
            f":latest"
        )

        artifact_name_tmp = artifact_name.replace("_merged_", f"_merged_step_{ratio}")

        if cfg.eval_skip_if_exists and os.path.exists(f"{cfg.evaluation_export_dir}/{artifact_name_tmp}.json"):
            print(f"\n\n\n")
            pylogger.info(f"Skipping evaluation.")
            pylogger.info(f"Artifact already exists: {cfg.evaluation_export_dir}/{artifact_name_tmp}.json")
            pylogger.info(f"cfg.eval_skip_if_exists: {cfg.eval_skip_if_exists}")
            print(f"\n\n\n")

            continue

        if cfg.eval_ft_progress_merging:

            tmp_eval_results = {}
            tmp_eval_results[ratio] = {}
            tvs_to_apply_original = copy.deepcopy(cfg.task_vectors.to_apply)
            
            zeroshot_model_all_zeros = copy.deepcopy(zeroshot_model)
            for parameter in zeroshot_model_all_zeros.parameters():
                parameter.data = torch.zeros_like(parameter.data)
                if parameter.grad is not None:
                    parameter.grad.zero_()
            for buffer in zeroshot_model_all_zeros.buffers():
                buffer.data = torch.zeros_like(buffer)

            for dataset in tvs_to_apply_original:

                cfg_copy = copy.deepcopy(cfg)

                cfg_copy.task_vectors.to_apply = [dataset]
                cfg_copy.eval_datasets = [dataset]

                # for single datasets, we do not care about cos sims and euclidean dists
                tmp_eval_results[ratio][dataset], _, _ = eval(
                    finetuned_id_fn=finetuned_id_fn,
                    logger=logger,
                    cfg=cfg_copy,
                    zeroshot_model=zeroshot_model_all_zeros,
                    artifact_name=artifact_name_tmp,
                    template_core=template_core,
                )[dataset]
                
                print(f"\n\n\n")
                pylogger.info(f"Results for ratio {ratio} and dataset {dataset}")
                pprint(tmp_eval_results, expand_all=True)
                print(f"\n\n\n")

            eval_results[ratio] = tmp_eval_results[ratio]


        else:
            eval_results[ratio], cos_sims[ratio], euclidean_dists[ratio] = eval(
                finetuned_id_fn=finetuned_id_fn,
                logger=logger,
                cfg=cfg,
                zeroshot_model=zeroshot_model,
                artifact_name=artifact_name_tmp,
                template_core=template_core,
            )

        print(f"\n\n\n")
        pylogger.info(f"Results for ratio {ratio}")
        pprint(eval_results[ratio], expand_all=True)
        print(f"\n\n\n")

        print(f"\n\n\n")
        pylogger.info(f"Results for ALL ratios")
        pprint(eval_results, expand_all=True)
        print(f"\n\n\n")

    
    print(f"\n\n\n\n\n")
    pylogger.info(f"Results for ALL ratios before export")
    pprint(eval_results, expand_all=True)
    print(f"\n\n\n\n\n")

    print(f"\n\n\n\n\n")
    pylogger.info(f"Cosine similarities for ALL ratios before export")
    pprint(cos_sims, expand_all=True)
    print(f"\n\n\n\n\n")

    print(f"\n\n\n\n\n")
    pylogger.info(f"Euclidean distances for ALL ratios before export")
    pprint(euclidean_dists, expand_all=True)
    print(f"\n\n\n\n\n")

    export_json_to_disk(
        {
            "results_all_ratios": eval_results,
            "results": eval_results[1.0],
            "cfg": OmegaConf.to_container(cfg, resolve=True),
        },
        cfg.evaluation_export_dir,
        artifact_name
    )

    sims_dists_artifact_name = artifact_name.replace(f"_merged_{'-'.join(cfg.task_vectors.to_apply)}", f"_merged_{cfg.tvs_to_apply_group_name}")
    np.save(f"./evaluations/tvs_sims_dists/{sims_dists_artifact_name}_cos_sims.npy", cos_sims)
    np.save(f"./evaluations/tvs_sims_dists/{sims_dists_artifact_name}_l2_dists.npy", euclidean_dists)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="task_vectors.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
