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
from lightning.pytorch import Callback
from omegaconf import DictConfig, ListConfig
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
from tvp.utils.io_utils import load_model_from_artifact
from tvp.utils.utils import build_callbacks
from torch.nn.utils import vector_to_parameters
from torch.nn.utils import parameters_to_vector
from hydra.utils import instantiate
import hydra
from hydra import initialize, compose
from typing import Dict, List

# from competitors.my_ties import ties_merging
# from competitors.my_breadcrumbs import model_breadcrumbs
# from competitors.their_ties import *
# from competitors.my_dare import *


pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def run(cfg: DictConfig) -> str:
    epoch_divisor = cfg.epoch_divisor
    order = cfg.order

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

    dataset_name_to_ta_eps = {
        "CIFAR100": 6,
        "DTD": 76,
        "EuroSAT": 12,
        "GTSRB": 11,
        "MNIST": 5,
        "RESISC45": 15,
        "SVHN": 4,
    }

    task_specific_accs = {
        "CIFAR100": 0.8592000007629395,
        "DTD": 0.7716186046600342,
        "EuroSAT": 0.9937962889671326,
        "GTSRB": 0.9958708882331848,
        "MNIST": 0.9929999709129333,
        "RESISC45": 0.9534391760826111,
        "SVHN": 0.9607999920845032,
    }

    """Generic train loop.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Returns:
        the run directory inside the storage_dir used by the current experiment
    """

    # print(f"\n\n\n{cfg}\n\n\n")

    seed_index_everything(cfg)

    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )
    logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)


    if order == 1:
        zeroshot_identifier = f"{cfg.nn.module.model.model_name}_pt"
    # else:
    #     zeroshot_identifier = f"{cfg.nn.module.model.model_name}_One{epoch_divisor}Eps{order-1}{num_to_th[order-1]}OrderUnifiedModel_0" 
    #     zeroshot_identifier = f"{cfg.nn.module.model.model_name}_10Eps{order-1}{num_to_th[order-1]}OrderUnifiedModel_{cfg.seed_index}"
    
    zeroshot_model_atm = load_model_from_artifact(artifact_path=f"{zeroshot_identifier}:latest", run=logger.experiment)
    zeroshot_model_ta  = load_model_from_artifact(artifact_path=f"{zeroshot_identifier}:latest", run=logger.experiment)

    #finetuned_id_fn = lambda dataset: f"{cfg.nn.module.model.model_name}_{dataset}_{cfg.seed_index}_PosthocClipAndTrain0.1:v0" 
    #finetuned_id_fn = lambda dataset: f"{cfg.nn.module.model.model_name}_{dataset}_{cfg.seed_index}__PosthocClipping0.1:v0" 
    #finetuned_id_fn = lambda dataset: f"{cfg.nn.module.model.model_name}_{dataset}_{cfg.seed_index}_sparseClipping0.01:v0" 
    #finetuned_id_fn = lambda dataset: f"{cfg.nn.module.model.model_name}_{dataset}_{cfg.seed_index}_2ndOrder:v0"
    #finetuned_id_fn = lambda dataset: f"{cfg.nn.module.model.model_name}_{dataset}_{cfg.seed_index}_One{epoch_divisor}Eps{order}{num_to_th[order]}Order:v0"
    #finetuned_id_fn = lambda dataset: f"{cfg.nn.module.model.model_name}_{dataset}_{cfg.seed_index}_One{epoch_divisor}Eps{order}{num_to_th[order]}Order:latest"
    #finetuned_id_fn = lambda dataset: f"{cfg.nn.module.model.model_name}_{dataset}_{cfg.seed_index}_One4Eps1stOrder:v0"
    #finetuned_id_fn = lambda dataset: f"{cfg.nn.module.model.model_name}_{dataset}_{cfg.seed_index}:v0"
    #finetuned_id_fn = lambda dataset: f"{cfg.nn.module.model.model_name}_{dataset}_{cfg.seed_index}_10Eps1stOrder:latest"
    #finetuned_id_fn = lambda dataset: f"{cfg.nn.module.model.model_name}_{dataset}_{cfg.seed_index}_2Eps{cfg.order}{num_to_th[cfg.order]}Order:latest"

    finetuned_id_fn_atm = lambda dataset: f"{cfg.nn.module.model.model_name}_{dataset}_{cfg.seed_index}_1Eps1stOrder:latest"
    finetuned_id_fn_ta  = lambda dataset: f"{cfg.nn.module.model.model_name}_{dataset}_{cfg.seed_index}_{dataset_name_to_ta_eps[dataset]}Eps1stOrder:latest"

    finetuned_models_atm = {
        dataset: load_model_from_artifact(artifact_path=finetuned_id_fn_atm(dataset), run=logger.experiment)
        for dataset in cfg.task_vectors.to_apply
    }
    finetuned_models_ta = {
        dataset: load_model_from_artifact(artifact_path=finetuned_id_fn_ta(dataset), run=logger.experiment)
        for dataset in cfg.task_vectors.to_apply
    }

    zeroshot_orig_weights_atm = copy.deepcopy(zeroshot_model_atm.state_dict())
    zeroshot_orig_weights_ta  = copy.deepcopy(zeroshot_model_ta.state_dict())

    # Task vectors
    flatten = lambda model: parameters_to_vector(model.parameters())

    zeroshot_vec_atm = flatten(zeroshot_model_atm)
    zeroshot_vec_ta  = flatten(zeroshot_model_ta)
    
    task_vectors_atm = [
        TaskVector.from_models(zeroshot_model_atm, finetuned_models_atm[dataset]) for dataset in cfg.task_vectors.to_apply
    ]
    task_vectors_ta = [
        TaskVector.from_models(zeroshot_model_ta, finetuned_models_ta[dataset]) for dataset in cfg.task_vectors.to_apply
    ]

    def apply_task_vector(model, task_vector, scaling_coef=1):
        #model.load_state_dict({k: v + task_vector[k] for k, v in model.state_dict().items()})
        model.load_state_dict({k: v + 1/(scaling_coef)*task_vector[k] for k, v in model.state_dict().items()})

    # Make task vectors orthogonal among them
    # def tv_orthogonalization(vectors, method="gs"): # gs: gram schmidt
    #     if method == "gs":
    #         orthogonal_vectors = []
    #         for v in vectors:
    #             for u in orthogonal_vectors:
    #                 v = v - (torch.dot(v, u) / torch.dot(u, u)) * u
    #             orthogonal_vectors.append(v)
    #         return torch.stack(orthogonal_vectors)
    #     else:
    #         raise ValueError("Unsupported method.")

    with torch.no_grad():
        task_vectors_atm = torch.stack(
            [flatten(finetuned_models_atm[dataset]) - zeroshot_vec_atm for dataset in cfg.task_vectors.to_apply]
        )
        task_vectors_ta = torch.stack(
            [flatten(finetuned_models_ta[dataset]) - zeroshot_vec_ta for dataset in cfg.task_vectors.to_apply]
        )
    
    # if cfg.task_vectors.merging_method == "ties":
    #     print("\nRunning TIES...\n")
    #     #task_vectors = ties_merging(task_vectors, cfg.task_vectors.ties_topk)
    #     multi_task_vector = their_ties_merging(reset_type="topk",
    #                                       flat_task_checks=task_vectors, 
    #                                       reset_thresh=cfg.task_vectors.ties_topk,
    #                                       resolve_method="none",
    #                                       merge_func="mean")
    # elif cfg.task_vectors.merging_method == "breadcrumbs":
    #     print("\nRunning Model Breadcrumbs...\n")
    #     task_vectors = model_breadcrumbs(task_vectors,beta=cfg.task_vectors.breadcrumbs_beta, gamma=cfg.task_vectors.breadcrumbs_gamma)
    # elif cfg.task_vectors.merging_method == "dare":
    #     print("\nRunning DARE Merging...\n")
    #     task_vectors = my_dare(task_vectors, ref_model=zeroshot_model, p=cfg.task_vectors.dare_rate)
    # else: print("\nRunning vanilla merging...\n")
    print("\nRunning vanilla merging...\n")
    # if cfg.task_vectors.orthogonalize:
    #     task_vectors = tv_orthogonalization(task_vectors, method='gs')

    print("\nATM pairwise cosine similarity:")
    print_pairwise_cos_sim(task_vectors_atm)
    print("\nTA pairwise cosine similarity:")
    print_pairwise_cos_sim(task_vectors_ta)

    if cfg.task_vectors.merging_method != "ties":
        task_vector_aggregator_atm = instantiate(cfg.task_vectors.aggregator)
        task_vector_aggregator_ta = instantiate(cfg.task_vectors.aggregator)
        
        multi_task_vector_atm = task_vector_aggregator_atm(task_vectors_atm)
        multi_task_vector_ta = task_vector_aggregator_ta(task_vectors_ta)

    delta_model_atm = copy.deepcopy(zeroshot_model_atm)
    vector_to_parameters(multi_task_vector_atm, delta_model_atm.parameters())
    task_equipped_model_atm = copy.deepcopy(zeroshot_model_atm)
    apply_task_vector(task_equipped_model_atm, delta_model_atm.state_dict(), scaling_coef=cfg.task_vectors.scaling_coefficient)
    
    delta_model_ta = copy.deepcopy(zeroshot_model_ta)
    vector_to_parameters(multi_task_vector_ta, delta_model_ta.parameters())
    task_equipped_model_ta = copy.deepcopy(zeroshot_model_ta)
    apply_task_vector(task_equipped_model_ta, delta_model_ta.state_dict(), scaling_coef=cfg.task_vectors.scaling_coefficient)


    # Save the unified model as artifact
    #artifact_name = f"{cfg.nn.module.model.model_name}_2stOrderUnifiedModel_{cfg.seed_index}"
    #artifact_name = f"{cfg.nn.module.model.model_name}_One{epoch_divisor}Eps{order}{num_to_th[order]}OrderUnifiedModel_{cfg.seed_index}"
    #artifact_name = f"{cfg.nn.module.model.model_name}_HalfEpsSomeDatasets2ndOrderUnifiedModel_{cfg.seed_index}" #################
    #artifact_name = f"{cfg.nn.module.model.model_name}_10Eps_UnifiedModel_{cfg.seed_index}"
    #artifact_name = f"{cfg.nn.module.model.model_name}_TIEScrumbs10EpsUnifiedModel_{cfg.seed_index}"
    #Eps{cfg.order}{num_to_th[cfg.order]}
    #artifact_name = f"{cfg.nn.module.model.model_name}_Breadcrumbs10Eps{order}{num_to_th[order]}OrderUnifiedModel_{cfg.seed_index}"
    #metadata = {"model_name": f"{cfg.nn.module.model.model_name}", "model_class": "tvp.modules.encoder.ImageEncoder"}
    #upload_model_to_wandb(task_equipped_model, artifact_name, logger.experiment, cfg, metadata)


    seed_index_everything(cfg)

    results_atm = {}

    for dataset_name in cfg.eval_datasets:

        classification_head_identifier = f"{cfg.nn.module.model.model_name}_{dataset_name}_head"
        classification_head = load_model_from_artifact(
            #artifact_path=f"{classification_head_identifier}:v0", run=logger.experiment
            artifact_path=f"{classification_head_identifier}:latest", run=logger.experiment
        )

        model = hydra.utils.instantiate(
            cfg.nn.module, encoder=task_equipped_model_atm, classifier=classification_head, _recursive_=False
        )

        dataset = get_dataset(
            dataset_name,
            preprocess_fn=model.encoder.train_preprocess,
            location=cfg.nn.data.data_path,
            batch_size=cfg.nn.data.batch_size.train,
        )

        callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

        storage_dir: str = cfg.core.storage_dir

        pylogger.info("Instantiating the <Trainer>")
        trainer = pl.Trainer(
            default_root_dir=storage_dir,
            plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
            logger=False,
            callbacks=callbacks,
            **cfg.train.trainer,
        )

        # Evaluation
        if cfg.eval_on_train:
            pylogger.info("Evaluating on the training set")
            trainer.test(model=model, dataloaders=dataset.train_loader)

        pylogger.info("Evaluating on the test set!")
        test_results = trainer.test(model=model, dataloaders=dataset.test_loader)

        test_results[0]["acc/test/normalized"] = test_results[0]["acc/test"] / task_specific_accs[dataset_name]

        results_atm[dataset_name] = test_results

    # print(f"\nResults ATM: {results_atm}\n")

    avg_acc_normalized_atm = sum([results_atm[dataset_name][0]["acc/test/normalized"] for dataset_name in results_atm.keys()]) / len(results_atm.keys())
    avg_acc_atm            = sum([results_atm[dataset_name][0]["acc/test"]            for dataset_name in results_atm.keys()]) / len(results_atm.keys())

    seed_index_everything(cfg)
    
    results_ta = {}

    for dataset_name in cfg.eval_datasets:

        classification_head_identifier = f"{cfg.nn.module.model.model_name}_{dataset_name}_head"
        classification_head = load_model_from_artifact(
            #artifact_path=f"{classification_head_identifier}:v0", run=logger.experiment
            artifact_path=f"{classification_head_identifier}:latest", run=logger.experiment
        )

        model = hydra.utils.instantiate(
            cfg.nn.module, encoder=task_equipped_model_ta, classifier=classification_head, _recursive_=False
        )

        dataset = get_dataset(
            dataset_name,
            preprocess_fn=model.encoder.train_preprocess,
            location=cfg.nn.data.data_path,
            batch_size=cfg.nn.data.batch_size.train,
        )

        callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

        storage_dir: str = cfg.core.storage_dir

        pylogger.info("Instantiating the <Trainer>")
        trainer = pl.Trainer(
            default_root_dir=storage_dir,
            plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
            logger=False,
            callbacks=callbacks,
            **cfg.train.trainer,
        )

        # Evaluation
        if cfg.eval_on_train:
            pylogger.info("Evaluating on the training set")
            trainer.test(model=model, dataloaders=dataset.train_loader)

        pylogger.info("Evaluating on the test set!")
        test_results = trainer.test(model=model, dataloaders=dataset.test_loader)

        test_results[0]["acc/test/normalized"] = test_results[0]["acc/test"] / task_specific_accs[dataset_name]

        results_ta[dataset_name] = test_results

    # print(f"\nResults TA: {results_ta}\n")

    avg_acc_normalized_ta = sum([results_ta[dataset_name][0]["acc/test/normalized"] for dataset_name in results_ta.keys()]) / len(results_ta.keys())
    avg_acc_ta            = sum([results_ta[dataset_name][0]["acc/test"]            for dataset_name in results_ta.keys()]) / len(results_ta.keys())

    cos_sim_atm_ta = F.cosine_similarity(multi_task_vector_atm, multi_task_vector_ta, dim=0).item()

    results = {
        "applied_task_vectors": cfg.task_vectors.to_apply,
        "avg_acc_normalized_atm": avg_acc_normalized_atm,
        "avg_acc_atm": avg_acc_atm,
        "results_atm": results_atm,
        "avg_acc_normalized_ta": avg_acc_normalized_ta,
        "avg_acc_ta": avg_acc_ta,
        "results_ta": results_ta,
        "cos_sim_atm_ta": cos_sim_atm_ta,
    }

    print(f"\n\n\n\nResults ATM and TA: {results}\n\n\n\n")





def print_pairwise_cos_sim(task_vectors): # input shape: [num_vectors, vector_size]:
    norm_tensor = F.normalize(task_vectors, p=2, dim=1)
    cosine_similarity_matrix = torch.mm(norm_tensor, norm_tensor.T)
    cosine_similarity_matrix_np = cosine_similarity_matrix.detach().numpy()
    print("\nPairwise Cosine Similarity Matrix:")
    print(cosine_similarity_matrix_np)
    print("\n")


def generate_orthogonal_directions_for_tv(state_dict, num_directions): # returns a dictionary where keys are the parameter names and the values are many orthogonal directions
    orthogonal_directions = {}
    for key, tensor in state_dict.items():
        shape = tensor.shape
        flat_dim = tensor.numel()
        random_matrix = np.random.randn(flat_dim, num_directions)
        q, _ = np.linalg.qr(random_matrix)
        orthogonal_directions[key] = torch.tensor(q, dtype=torch.float32).view(*shape, num_directions)
    return orthogonal_directions

def project_onto_direction(tensor, direction):
    flat_tensor = tensor.view(-1)
    flat_direction = direction.view(-1)
    projection = torch.matmul(flat_tensor, flat_direction) / torch.norm(flat_direction, dim=0)**2
    projected_tensor = (flat_direction*projection).view(tensor.shape)
    return projected_tensor

def project_tv(tv, orthogonal_directions, task_id):
    projected_state_dict = {}
    for key, tensor in tv.items():
        direction = orthogonal_directions[key][..., task_id].to("cuda")
        projected_tensor = project_onto_direction(tensor, direction)
        projected_state_dict[key] = projected_tensor
    return projected_state_dict


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






@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="task_vectors.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()