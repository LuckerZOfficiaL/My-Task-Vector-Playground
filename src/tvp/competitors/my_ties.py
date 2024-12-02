import numpy as np

def ties_merging(task_vectors, k_percent):
    # Step 1: Trim
    num_vectors, vector_size = task_vectors.shape
    k = int(k_percent * vector_size)
    
    trimmed_vectors = np.zeros_like(task_vectors)
    for i in range(num_vectors):
        # Get the indices of the top k% values by magnitude
        top_k_indices = np.argsort(np.abs(task_vectors[i]))[-k:]
        # Keep only the top k% values, zero the rest out
        trimmed_vectors[i, top_k_indices] = task_vectors[i, top_k_indices]
    
    # Step 2: Elect sign
    # Compute the aggregate sign vector
    positive_magnitudes = np.sum(trimmed_vectors * (trimmed_vectors > 0), axis=0)
    negative_magnitudes = np.sum(trimmed_vectors * (trimmed_vectors < 0), axis=0)
    sign_vector = np.where(positive_magnitudes > np.abs(negative_magnitudes), 1, -1)
    
    # Step 3: Disjoint merge
    merged_vector = np.zeros(vector_size)
    for i in range(vector_size):
        # Collect the values that agree with the elected sign
        agreeing_values = trimmed_vectors[:, i] * (np.sign(trimmed_vectors[:, i]) == sign_vector[i])
        agreeing_values = agreeing_values[agreeing_values != 0]  # Filter out zeros
        if len(agreeing_values) > 0:
            merged_vector[i] = np.mean(agreeing_values)
    
    return merged_vector#, positive_magnitudes, negative_magnitudes, sign_vector, trimmed_vectors



"""task_vectors = np.array([
    [0.2, 0.1, -0.4, 0.3],
    [-0.1, 0.4, -0.2, 0.1],
    [0.3, -0.3, 0.2, -0.4]
])
k_percent = 0.5  # Keep the top 50% of values by magnitude

merged_vector, positive_magnitudes, negative_magnitudes, sign_vector, trimmed_vectors = ties_merging(task_vectors, k_percent)
print("trimmed\n", trimmed_vectors)
print("positive ", positive_magnitudes)
print("negative", negative_magnitudes)
print("sign", sign_vector)
print("Merged Task Vector:", merged_vector)"""
