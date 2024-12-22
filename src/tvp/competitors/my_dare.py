import torch
import random
import copy
from torch.nn.utils import vector_to_parameters, parameters_to_vector

def my_dare(flattened_vectors, ref_model, p=0.9):
    pruned_vectors = []
    ref_model_copy = copy.deepcopy(ref_model)

    for i in range(flattened_vectors.size(0)):
        flattened_vector = flattened_vectors[i]
        vector_to_parameters(flattened_vector, ref_model_copy.parameters())

        with torch.no_grad():
            for param in ref_model_copy.parameters():
                num_elements = param.numel()
                num_drop = int(p * num_elements)
                param_flat = param.view(-1)
                drop_indices = random.sample(range(num_elements), num_drop)
                param_flat[drop_indices] = 0
                param_flat *= 1 / (1 - p)
        
        pruned_flattened_vector = parameters_to_vector(ref_model_copy.parameters())
        pruned_vectors.append(pruned_flattened_vector)
        return torch.stack(pruned_vectors)
