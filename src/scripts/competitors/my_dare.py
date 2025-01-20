import torch
import random
import copy
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from torch import Tensor

from typing import Dict


def my_dare(task_vectors: Dict[str, Tensor], ref_model, p=0.9):
    pruned_task_vectors = {}
    
    ref_model_copy = copy.deepcopy(ref_model)

    for task_name, task_vector in task_vectors.items():
        vector_to_parameters(task_vector, ref_model_copy.parameters())

        with torch.no_grad():
            for param in ref_model_copy.parameters():
                num_elements = param.numel()
                num_drop = int(p * num_elements)
                param_flat = param.view(-1)
                drop_indices = random.sample(range(num_elements), num_drop)
                param_flat[drop_indices] = 0
                param_flat *= 1 / (1 - p)
        
        pruned_task_vectors[task_name] = parameters_to_vector(ref_model_copy.parameters())
    
    return pruned_task_vectors
