import torch
import torch.nn.functional as F
import numpy as np


# This script is copied from the original TIES implementation, with slight modifications

def aggregate(checks, merge_func):
    if merge_func == "mean":
        return torch.mean(checks, dim=0)
    elif merge_func == "sum":
        return torch.sum(checks, dim=0)
    elif merge_func == "median":
        return torch.median(checks, dim=0)[0]
    elif merge_func == "magnitude":
        return torch.sum(torch.abs(checks), dim=0)
    else:
        raise ValueError(f"Unknown merge function: {merge_func}")


def resolve_signs(checks, method):
    if method == "mass":
        # Choose the sign that maximizes the absolute sum
        signs = torch.sign(checks)
        sum_mass = torch.sum(torch.abs(checks), dim=0)
        signs[sum_mass == 0] = 1
        return checks * signs
    elif method == "normfrac":
        # Normalize by the fraction of the norm
        norm = torch.norm(checks, dim=0)
        norm[norm == 0] = 1
        return checks / norm
    elif method == "normmass":
        # Normalize by the norm times the absolute sum
        norm = torch.norm(checks, dim=0)
        sum_mass = torch.sum(torch.abs(checks), dim=0)
        norm_mass = norm * sum_mass
        norm_mass[norm_mass == 0] = 1
        return checks / norm_mass
    else:
        return checks


def their_ties_merging(
        reset_type,
        flat_task_checks,
        reset_thresh,
        resolve_method,
        merge_func):
    if reset_type == "topk":
        flat_task_checks = prune_topk(flat_task_checks, reset_thresh)
    if reset_type == "nf" or reset_type == "std":
        # If pruning is specified, apply it
        raise NotImplementedError("Pruning methods are not implemented in this snippet.")
    
    # Resolve signs if needed
    checks = resolve_signs(flat_task_checks, resolve_method)

    # Aggregate the resolved checks
    merged_check = aggregate(checks, merge_func)
    
    return merged_check


def prune_topk(tensors, k):
    """Keeps the top-k percentage of entries based on magnitude."""
    k_value = int(tensors.numel() * k)
    flattened = tensors.view(-1)
    # Get the threshold by selecting the top-k value
    threshold = flattened.abs().topk(k_value, sorted=False).values.min()
    # Zero out elements below the threshold
    pruned = torch.where(flattened.abs() >= threshold, flattened, torch.tensor(0.0, device=tensors.device))
    return pruned.view_as(tensors)

"""def main():
    # Assuming `task_vectors` is your tensor with shape [n, d]
    task_vectors = torch.tensor([[-0.6157, -0.4206,  0.2174, -0.5956, -0.3906, -2.7919,  0.7029, -1.2054,
         -0.0169, -0.4693, -2.0810, -1.2877, -0.4426, -0.4871,  0.5076,  0.5087],
        [-1.3138,  1.1421, -0.1657,  0.8621, -1.7934,  0.5377,  0.4645,  0.0699,
          0.4182, -1.0612,  1.0529,  1.2909, -0.3178, -1.3791,  0.0143, -0.0656],
        [ 0.7429,  1.2865,  0.5754,  1.1790,  1.2531,  0.1931,  0.1933, -1.5641,
          1.5267, -0.5896,  0.7202,  1.5949, -0.9238, -0.5806, -0.3140,  0.3183]])
    # Define parameters for merging
    reset_type = "topk"  # You can choose "nf", "topk", or "std" for pruning
    reset_thresh = 0.1  # Define threshold if using pruning
    resolve_method = "none"  # Choose "mass", "normfrac", "normmass", or "none"
    merge_func = "mean"  # Choose "mean", "sum", "median", or "magnitude"

    #print("Pruned:", prune_topk(task_vectors, 0.4))
    # Perform the merging
    merged_tv = their_ties_merging(
        reset_type=reset_type,
        flat_task_checks=task_vectors,
        reset_thresh=reset_thresh,
        resolve_method=resolve_method,
        merge_func=merge_func
    )

    print("Merged Task Vector:", merged_tv)

if __name__ == "__main__":
    main()"""