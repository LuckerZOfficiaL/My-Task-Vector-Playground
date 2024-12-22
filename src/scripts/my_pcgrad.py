import torch

def my_pcgrad(task_vectors):
    """
    Apply PCGrad to the task vectors using PyTorch.

    Arguments:
    - task_vectors: tensor of shape [n, d], where n is the number of tasks and d is the dimensionality of each task vector.

    Returns:
    - tensor of shape [n, d], where conflicting task vectors have been projected to resolve conflicts.
    """
    num_tasks, dim = task_vectors.shape
    projected_vectors = task_vectors.clone()  # Clone to avoid in-place modifications

    for i in range(num_tasks):
        for j in range(num_tasks):
            if i != j:
                # Compute the dot product between task vectors[i] and task vectors[j]
                dot_product = torch.dot(projected_vectors[i], task_vectors[j])

                # If the task vectors conflict (i.e., dot product is negative), project task_vectors[i] onto the normal plane of task_vectors[j]
                if dot_product < 0:
                    projection = (dot_product / torch.dot(task_vectors[j], task_vectors[j])) * task_vectors[j]
                    projected_vectors[i] = projected_vectors[i] - projection
    return projected_vectors


def cosine_similarity_matrix(vectors):
    """
    Compute the cosine similarity matrix for task vectors using PyTorch.

    Arguments:
    - vectors: tensor of shape [n, d], where n is the number of tasks and d is the dimensionality of each task vector.

    Returns:
    - Tensor of shape [n, n], representing pairwise cosine similarities.
    """
    num_tasks = vectors.shape[0]
    similarity_matrix = torch.zeros((num_tasks, num_tasks))  # Initialize similarity matrix

    for i in range(num_tasks):
        for j in range(num_tasks):
            # Compute cosine similarity between vectors[i] and vectors[j]
            dot_product = torch.dot(vectors[i], vectors[j])
            norm_i = torch.norm(vectors[i])
            norm_j = torch.norm(vectors[j])
            similarity_matrix[i, j] = dot_product / (norm_i * norm_j)

    return similarity_matrix