## Imports
import logging
import os
from typing import Dict, List, Union, Optional
import wandb
import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningModule
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
from typing import Dict, List

from tvp.utils.vectors import print_pairwise_cos_sim, orthogonalize_task_vectors

from src.scripts.evaluate import eval_merged_model
from src.tvp.utils.vectors import sort_tvs_by_norm_merged_accuracy
from src.tvp.utils.vectors import apply_conflict_res_method

from rich.pretty import pprint


pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def run(cfg: DictConfig) -> str:

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

    if cfg.conflict_res_method == "none":
        conflict_res_method = ""
    elif cfg.conflict_res_method == "bc":
        conflict_res_method = f"_bc_beta_{cfg.task_vectors.breadcrumbs.beta}_gamma_{cfg.task_vectors.breadcrumbs.gamma}"
    elif cfg.conflict_res_method == "dare":
        conflict_res_method = f"_dare_rate_{cfg.task_vectors.dare.rate}"
    elif cfg.conflict_res_method == "ties":
        conflict_res_method = f"_ties_lambda_{cfg.task_vectors.ties.ties_lambda}_top_k_{cfg.task_vectors.ties.top_k}_merge_func_{cfg.task_vectors.ties.merge_func}"
    else:
        raise ValueError(f"Unknown conflict resolution method: {cfg.conflict_res_method}")

    wita = f"_wita_num_iters_{cfg.wita.num_iters}_top_k_weakest_{cfg.wita.top_k_weakest}_top_k_strongest_{cfg.wita.top_k_strongest}"
    
    artifact_name = (
        f"{cfg.nn.module.model.model_name}"
        f"_{cfg.seed_index}"
        f"_{cfg.ft_regime}"
        f"_{cfg.optimizer_name}"
        f"_wd_{cfg.nn.module.optimizer.weight_decay}"
        f"_lr_scheduler_{cfg.lr_scheduler_name}"
        f"{lr_scheduler_warmup_steps}"
        f"{orthogonalization_method}"
        f"{conflict_res_method}"
        f"{wita}"
        # f"_merged_{'-'.join(cfg.task_vectors.to_apply)}"
        f"_merged_{cfg.tvs_to_apply_group_name}"
    )

    print(f"\n\n\n")
    pylogger.info(f"Merrged artifact name: {artifact_name}")
    print(f"\n\n\n")

    if cfg.eval_skip_if_exists and os.path.exists(f"{cfg.evaluation_export_dir}/{artifact_name}.json"):
        print(f"\n\n\n")
        pylogger.info(f"Skipping evaluation.")
        pylogger.info(f"Artifact already exists: {cfg.evaluation_export_dir}/{artifact_name}.json")
        pylogger.info(f"cfg.eval_skip_if_exists: {cfg.eval_skip_if_exists}")
        print(f"\n\n\n")

        return 

    seed_index_everything(cfg)

    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )
    logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)

    zeroshot_identifier = f"{cfg.nn.module.model.model_name}_pt"
    
    zeroshot_model = load_model_from_artifact(artifact_path=f"{zeroshot_identifier}:latest", run=logger.experiment)

    finetuned_id_fn = lambda dataset: (
        f"{cfg.nn.module.model.model_name}"
        f"_{dataset}"
        f"_{cfg.seed_index}"
        f"_{cfg.ft_regime}"
        f"_{cfg.optimizer_name}"
        f"_wd_{cfg.nn.module.optimizer.weight_decay}"
        f"_lr_scheduler_{cfg.lr_scheduler_name}"
        f"{lr_scheduler_warmup_steps}"
        f":latest"
    )

    finetuned_models = {
        dataset: load_model_from_artifact(artifact_path=finetuned_id_fn(dataset), run=logger.experiment)
        for dataset in cfg.task_vectors.to_apply
    }

    # Task vectors
    flatten = lambda model: parameters_to_vector(model.parameters())

    zeroshot_vec = flatten(zeroshot_model)
    # task_vectors = [
        # TaskVector.from_models(zeroshot_model, finetuned_models[dataset]) for dataset in cfg.task_vectors.to_apply
    # ]

    def apply_task_vector(model, task_vector, scaling_coef=1):
        #model.load_state_dict({k: v + task_vector[k] for k, v in model.state_dict().items()})
        model.load_state_dict({k: v + scaling_coef*task_vector[k] for k, v in model.state_dict().items()})

    # before adding support for tv sort by difficulty
    # with torch.no_grad():
    #     task_vectors = torch.stack(
    #         [flatten(finetuned_models[dataset]) - zeroshot_vec for dataset in cfg.task_vectors.to_apply]
    #     )

    print(f"\n\n\n")

    M_h = copy.deepcopy(zeroshot_model)

    eval_results = {}

    for h in range(cfg.wita.num_iters):

        print(f"\n\n\n\n\n")
        pylogger.info(f"WITA iteration {h}/{cfg.wita.num_iters}")

        seed_index_everything(cfg)

        eval_results[h] = eval_merged_model(
            cfg=cfg,
            task_equipped_model=M_h,
            template_core=template_core,
            logger=logger
        )

        print()
        pprint(eval_results[h])

        with torch.no_grad():
            task_vectors = {
                dataset: flatten(finetuned_models[dataset]) - flatten(M_h) for dataset in cfg.task_vectors.to_apply
            }

        task_vectors_sorted = sort_tvs_by_norm_merged_accuracy(
            task_vectors=task_vectors,
            merged_accs_file_path_or_dict=eval_results[h],
            ft_summary_file_path="./evaluations/ft_summary/ft_summary_ta_adamw_wd_0.1_lr_scheduler_cosine_annealing_warmup_steps_200.csv"
        )

        # get the top weakest k task vectors
        task_vectors_top_weakest = {k: task_vectors_sorted[k] for k in list(task_vectors_sorted.keys())[:cfg.wita.top_k_weakest]}
        pylogger.info(f"Top weakest task vectors:")
        pylogger.info(task_vectors_top_weakest.keys())
        print(f"\n\n")

        # get the top strongest k task vectors
        if cfg.wita.top_k_strongest > 0:
            task_vectors_top_strongest = {k: task_vectors_sorted[k] for k in list(task_vectors_sorted.keys())[-cfg.wita.top_k_strongest:]}
        else:
            task_vectors_top_strongest = {}
        pylogger.info(f"Top strongest task vectors:")
        pylogger.info(task_vectors_top_strongest.keys())
        print(f"\n\n")

        # merge the two dicts
        task_vectors: Dict[str, Tensor] = {**task_vectors_top_weakest, **task_vectors_top_strongest}
        print("task_vectors")
        pylogger.info(task_vectors.keys())
        print(f"\n\n")

        pylogger.info(f"pairwise cosine similarity between task vectors before orthogonalization:")
        print_pairwise_cos_sim(torch.stack(list(task_vectors.values())))

        task_vectors: Dict[str, Tensor] = orthogonalize_task_vectors(task_vectors, cfg, artifact_name)

        task_vectors: Union[Dict[str, Tensor], Tensor] = apply_conflict_res_method(
            task_vectors=task_vectors, cfg=cfg, ref_model=M_h
        )

        if cfg.conflict_res_method != "ties":
            pylogger.info(f"pairwise cosine similarity between task vectors after orthogonalization:")
            print_pairwise_cos_sim(torch.stack(list(task_vectors.values())) if isinstance(task_vectors, dict) else task_vectors)

        # NOTE: this is needed because ties method already comprises an aggregation step
        if cfg.conflict_res_method == "ties":
            multi_task_vector = task_vectors * cfg.task_vectors.ties.ties_lambda
        else:
            task_vector_aggregator = instantiate(cfg.task_vectors.aggregator)
            multi_task_vector = task_vector_aggregator(torch.stack(list(task_vectors.values())))

        delta_model = copy.deepcopy(M_h)
        vector_to_parameters(multi_task_vector, delta_model.parameters())
    
        task_equipped_model = copy.deepcopy(M_h)
        apply_task_vector(
            model=task_equipped_model, 
            task_vector=delta_model.state_dict(), 
            scaling_coef=cfg.task_vectors.scaling_coefficient*(h + 1)/cfg.wita.num_iters
        )

        M_h = copy.deepcopy(task_equipped_model)

        

    seed_index_everything(cfg)

    print(f"\n\n\n")
    pylogger.info(f"Final evaluation")

    eval_results[cfg.wita.num_iters] = eval_merged_model(
        cfg=cfg, 
        task_equipped_model=M_h, 
        template_core=template_core, 
        logger=logger
    )

    print(f"\n\n")
    pprint(eval_results)

    export_json_to_disk(
        {
            "results": eval_results[cfg.wita.num_iters],
            "results_all_iters": eval_results,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
        },
        cfg.evaluation_export_dir,
        artifact_name
    )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="task_vectors.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
