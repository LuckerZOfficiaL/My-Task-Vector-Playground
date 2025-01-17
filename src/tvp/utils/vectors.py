from rich import print
from rich.pretty import pprint

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from src.tvp.utils.io_utils import import_json_from_disk

import pandas as pd
from pandas import DataFrame

import logging


pylogger = logging.getLogger(__name__)


def print_pairwise_cos_sim(task_vectors): # input shape: [num_vectors, vector_size]:
    norm_tensor = F.normalize(task_vectors, p=2, dim=1)
    cosine_similarity_matrix = torch.mm(norm_tensor, norm_tensor.T)
    cosine_similarity_matrix_np = cosine_similarity_matrix.detach().numpy()
    print(f"\n")
    pylogger.info("Pairwise Cosine Similarity Matrix:")
    pylogger.info(cosine_similarity_matrix_np)
    print(f"\n")


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


def gram_schmidt(vectors: Tensor) -> Tensor: 
    orthogonal_vectors = []
    
    for v in vectors:
        
        for u in orthogonal_vectors:
            v = v - (torch.dot(v, u) / torch.dot(u, u)) * u
        
        orthogonal_vectors.append(v)
    
    return torch.stack(orthogonal_vectors) 


def _get_norm_merged_acc(accs: dict, ft_summary: DataFrame):

    accs_norm = {}

    for t in accs.keys():
        
        if "average_of_tasks" in t: continue

        accs_norm[t] = accs[t][0]["acc/test"] / float(ft_summary[ft_summary["dataset"] == t]["acc_test"])

    return accs_norm


# NOTE this can potentially support any possible metric for sorting the task vectors
def sort_tvs_by_norm_merged_accuracy(
    task_vectors: dict,
    merged_accs_file_path: str,
    ft_summary_file_path: str,
) -> dict:  
    
    pylogger.info(f"Sorting task vectors by norm merged accuracy")
    pylogger.info(f"Loading merged accuracies from {merged_accs_file_path}")

    merged_accs = import_json_from_disk(file_path=merged_accs_file_path)["results"]

    pylogger.info(f"Loading ft summary from {ft_summary_file_path}")
    
    ft_summary = pd.read_csv(ft_summary_file_path)

    accs_norm = _get_norm_merged_acc(accs=merged_accs, ft_summary=ft_summary)
    pylogger.info(f"Norm merged accuracies before sorting: {accs_norm}")

    accs_norm_sorted = dict(sorted(accs_norm.items(), key=lambda item: item[1]))
    pylogger.info(f"Norm merged accuracies after sorting: {accs_norm_sorted}")

    task_vectors_sorted = {k: task_vectors[k] for k in accs_norm_sorted.keys()}

    return task_vectors_sorted


def orthogonalize_task_vectors(
    task_vectors: dict,
    cfg,
    artifact_name: str,
) -> Tensor:

    if cfg.eval_orthogonalization_method == "pc_grad":
        pylogger.info(f"Applying PCGrad")

        task_vectors = gram_schmidt(vectors=torch.stack(list(task_vectors.values())))
    
    elif cfg.eval_orthogonalization_method == "sorted_pc_grad":

        pylogger.info(f"Applying Sorted PCGrad")

        task_vectors_sorted = sort_tvs_by_norm_merged_accuracy(
            task_vectors=task_vectors, 
            merged_accs_file_path=f"./evaluations/merged/{artifact_name.replace('_sorted_pc_grad', '')}.json",
            ft_summary_file_path="./evaluations/ft_summary/ft_summary_ta_adamw_wd_0.1_lr_scheduler_cosine_annealing_warmup_steps_200.csv",
        )

        task_vectors = gram_schmidt(vectors=torch.stack(list(task_vectors_sorted.values())))
    
    elif cfg.eval_orthogonalization_method == "none":

        pylogger.info(f"Orthogonalization method: None")

        task_vectors = torch.stack(list(task_vectors.values()))
    
    else:
        raise ValueError(f"Unknown orthogonalization method: {cfg.eval_orthogonalization_method}")

    return task_vectors

    