import torch

from typing import Dict

def model_breadcrumbs(task_vectors: Dict[str, torch.Tensor], beta=0.85, gamma=0.993):
    pruned_task_vectors = {}

    for task_name, task_vector in task_vectors.items():
        num_params = task_vector.numel()
        
        # Step 1: Calculate thresholds directly
        abs_tensor = task_vector.abs()
        beta_value = int(beta * num_params)
        gamma_value = int((1 - gamma) * num_params)
        
        # Efficient computation of thresholds using partial sorting
        if beta_value > 0:
            beta_threshold = torch.topk(abs_tensor, beta_value, sorted=False).values.min()
        else:
            beta_threshold = abs_tensor.min()
        
        if gamma_value > 0:
            gamma_threshold = torch.topk(abs_tensor, gamma_value, sorted=False).values.min()
        else:
            gamma_threshold = abs_tensor.max()
        
        # Step 2: Prune using thresholds
        pruned = torch.where((abs_tensor >= beta_threshold) & (abs_tensor <= gamma_threshold), task_vector, torch.tensor(0.0))
        
        pruned_task_vectors[task_name] = pruned
    
    return pruned_task_vectors

"""# 
task_vectors = torch.tensor([
    [-0.6157, -0.4206, 0.2174, -0.5956, -0.3906, -2.7919, 0.7029, -1.2054, -0.0169, -0.4693, -2.0810, -1.2877, -0.4426, -0.4871, 0.5076, 0.5087],
    [-1.3138, 1.1421, -0.1657, 0.8621, -1.7934, 0.5377, 0.4645, 0.0699, 0.4182, -1.0612, 1.0529, 1.2909, -0.3178, -1.3791, 0.0143, -0.0656],
    [0.7429, 1.2865, 0.5754, 1.1790, 1.2531, 0.1931, 0.1933, -1.5641, 1.5267, -0.5896, 0.7202, 1.5949, -0.9238, -0.5806, -0.3140, 0.3183]
])

print("Task Vectors:\n", task_vectors)
pruned_vectors = model_breadcrumbs(task_vectors)
print("Pruned Task Vectors:\n", pruned_vectors)
"""

