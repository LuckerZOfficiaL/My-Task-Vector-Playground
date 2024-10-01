import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn.functional as F

from rich import print

def load_gradients_from_folder(folder_path, n_grads):
    """
    Load gradient tensors from a specified folder.
    
    Args:
        folder_path (str): Path to the folder containing gradient tensors.
        
    Returns:
        gradients (list): List of gradient tensors.
        epoch_files (list): List of file names for reference.
    """
    # List all files in the directory
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".pt")])
    files = files[:n_grads]
    print(f"Loading gradients from {len(files)} files in {folder_path}:\n {files}")
    
    gradients = []
    for file in files:
        file_path = os.path.join(folder_path, file)
        grad_tensor = torch.load(file_path)  # Load each tensor
        print(f"Loaded gradient tensor of shape {grad_tensor.shape} from {file_path}")
        gradients.append(grad_tensor)
    
    return gradients, files

# def compute_cosine_similarity_all_vs_all(gradients):
#     """
#     Compute all-vs-all cosine similarity between a list of gradient tensors.
    
#     Args:
#         gradients (list): List of gradient tensors.
        
#     Returns:
#         cosine_sim_matrix (np.array): Cosine similarity matrix of shape (n_epochs, n_epochs).
#     """
#     n_epochs = len(gradients)
#     # Initialize an empty matrix to store cosine similarities
#     cosine_sim_matrix = np.zeros((n_epochs, n_epochs))

#     # Normalize gradients to unit vectors
#     gradients_normalized = [grad / grad.norm(p=2) for grad in gradients]

#     # Compute all-vs-all cosine similarity
#     for i in range(n_epochs):
#         for j in range(n_epochs):
#             # Compute cosine similarity between two gradient vectors
#             cosine_sim = F.cosine_similarity(
#                 gradients_normalized[i], gradients_normalized[j], dim=0
#             )
#             cosine_sim_matrix[i, j] = cosine_sim.item()
    
#     return cosine_sim_matrix

def compute_cosine_similarity_lower_triangle(gradients):
    """
    Compute lower-triangular all-vs-all cosine similarity between a list of gradient tensors,
    including the principal diagonal.

    Args:
        gradients (list): List of gradient tensors.

    Returns:
        cosine_sim_matrix (np.array): Lower-triangular cosine similarity matrix of shape (n_epochs, n_epochs).
    """
    n_epochs = len(gradients)
    # Initialize an empty matrix to store cosine similarities
    cosine_sim_matrix = np.zeros((n_epochs, n_epochs))

    # Normalize gradients to unit vectors
    gradients_normalized = [grad / grad.norm(p=2) for grad in gradients]

    # Compute lower-triangular all-vs-all cosine similarity, including the diagonal
    for i in range(n_epochs):
        for j in range(i + 1):  # Include diagonal (j <= i)
            # Compute cosine similarity between two gradient vectors
            cosine_sim = F.cosine_similarity(gradients_normalized[i], gradients_normalized[j], dim=0)
            cosine_sim_matrix[i, j] = cosine_sim.item()
    
    return cosine_sim_matrix



# def plot_cosine_similarity_matrix(cosine_sim_matrix, epoch_files, save_dir):
#     """
#     Plot the cosine similarity matrix using a heatmap and save it to disk.

#     Args:
#         cosine_sim_matrix (np.array): Cosine similarity matrix.
#         epoch_files (list): List of file names corresponding to epochs.
#         save_dir (str): Directory to save the heatmap image.
#     """
#     plt.figure(figsize=(12, 10))  # Adjust the size for better clarity
#     sns.heatmap(
#         cosine_sim_matrix, 
#         xticklabels=epoch_files, 
#         yticklabels=epoch_files, 
#         annot=True, 
#         fmt=".2f", 
#         cmap="coolwarm"
#     )
#     plt.title("Cosine Similarity Matrix of Gradients Across Epochs")
#     plt.xlabel("Epoch")
#     plt.ylabel("Epoch")

#     # Define the save path for the heatmap image
#     save_path = os.path.join(save_dir, "cosine_similarity_heatmap.png")
    
#     # Save the heatmap with higher DPI (e.g., 300 DPI)
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"Cosine similarity heatmap saved to: {save_path}")

#     # Close the plot to free memory
#     plt.close()

def plot_cosine_similarity_matrix(cosine_sim_matrix, epoch_files, save_dir):
    """
    Plot the cosine similarity matrix using a heatmap and save it to disk, showing only the
    lower triangular part along with the principal diagonal.

    Args:
        cosine_sim_matrix (np.array): Cosine similarity matrix.
        epoch_files (list): List of file names corresponding to epochs.
        save_dir (str): Directory to save the heatmap image.
    """
    plt.figure(figsize=(12, 10))  # Adjust the size for better clarity

    # Create a mask for the upper triangle only, excluding the diagonal
    mask = np.triu(np.ones_like(cosine_sim_matrix, dtype=bool), k=1)

    sns.heatmap(
        cosine_sim_matrix,
        mask=mask,  # Apply the mask to hide the upper triangle but keep the diagonal
        xticklabels=epoch_files,
        yticklabels=epoch_files,
        annot=True,
        fmt=".2f",
        cmap="coolwarm"
    )
    plt.title("Cosine Similarity Matrix of Gradients Across Epochs (Lower Triangular + Diagonal)")
    plt.xlabel("Epoch")
    plt.ylabel("Epoch")

    # Define the save path for the heatmap image
    save_path = os.path.join(save_dir, "cosine_similarity_heatmap.png")

    # Save the heatmap with higher DPI (e.g., 300 DPI)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Cosine similarity heatmap saved to: {save_path}")

    # Close the plot to free memory
    plt.close()




# Define your folder path containing the gradient tensors
folder_path = "./grads/ViT-B-16/DTD/z8h1ptk9"  # Replace this with the path to your folder
n_grads = 10

# Load gradient tensors
gradients, epoch_files = load_gradients_from_folder(folder_path, n_grads)

# Compute cosine similarity matrix
cosine_sim_matrix = compute_cosine_similarity_lower_triangle(gradients)

# Plot the similarity matrix
plot_cosine_similarity_matrix(cosine_sim_matrix, epoch_files, folder_path)
