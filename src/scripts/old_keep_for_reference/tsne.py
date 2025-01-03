import wandb
import torch
import os
import numpy as np
from sklearn.manifold import TSNE
#from openTSNE import TSNE
import matplotlib.pyplot as plt

plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16) 


def run():
    wandb.login()


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
        10: "th"
    }

    datasets = ["CIFAR100", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45",  "SVHN"]

    pt_name = ['ViT-B-16_pt']
    baseline_names = ['ViT-B-16_'+dataset+'_0_10Eps1stOrder' for dataset in datasets]
    tva_unified_names = ["ViT-B-16_10Eps_UnifiedModel_0"]
    ties_unified_names = ["ViT-B-16_TIES10Eps1stOrderUnifiedModel_0"]
    bc_unified_names = ["ViT-B-16_Breadcrumbs10Eps1stOrderUnifiedModel_0"]
    hota_unified_names = ["ViT-B-16_OneNoneEps"+str(order)+num_to_th[order]+"OrderUnifiedModel_0" for order in range(1, 11)]

    run = wandb.init(project="task-vectors-playground", job_type="artifact")
    pt_ckpt = {}
    tva_ckpts = {}
    ties_ckpts = {}
    bc_ckpts = {}
    hota_ckpts = {}

    for name in pt_name:
        artifact = run.use_artifact(name+":latest", type='checkpoint')  # Change type if needed
        artifact_dir = artifact.download()
        ckpt_path = os.path.join(artifact_dir, 'trained.ckpt')  # Update with the correct filename
        pt_ckpt[name] = torch.load(ckpt_path)

    for name in tva_unified_names:
        artifact = run.use_artifact(name+":latest", type='checkpoint')  # Change type if needed
        artifact_dir = artifact.download()
        ckpt_path = os.path.join(artifact_dir, 'trained.ckpt')
        tva_ckpts[name] = torch.load(ckpt_path)

    for name in ties_unified_names:
        artifact = run.use_artifact(name+":latest", type='checkpoint')  # Change type if needed
        artifact_dir = artifact.download()
        ckpt_path = os.path.join(artifact_dir, 'trained.ckpt')
        ties_ckpts[name] = torch.load(ckpt_path)

    for name in bc_unified_names:
        artifact = run.use_artifact(name+":latest", type='checkpoint')  # Change type if needed
        artifact_dir = artifact.download()
        ckpt_path = os.path.join(artifact_dir, 'trained.ckpt')
        bc_ckpts[name] = torch.load(ckpt_path)

    for name in hota_unified_names:
        artifact = run.use_artifact(name+":latest", type='checkpoint')  # Change type if needed
        artifact_dir = artifact.download()
        ckpt_path = os.path.join(artifact_dir, 'trained.ckpt')
        hota_ckpts[name] = torch.load(ckpt_path)

    def flatten_model_params(ckpt):
        tensors = [param.flatten() for param in ckpt.values()]
        flat_tensor = torch.cat(tensors)
        return flat_tensor


    """pt_arr = np.random.rand(1, 1000)    
    tva_arr = np.random.rand(1, 1000)  
    ties_arr = np.random.rand(1, 1000)  
    bc_arr = np.random.rand(1, 1000)  
    hota_arr = np.random.rand(10, 1000)"""


    
    pt_arr = np.array([flatten_model_params(ckpt).numpy() for ckpt in pt_ckpt.values()])
    tva_arr = np.array([flatten_model_params(ckpt).numpy() for ckpt in tva_ckpts.values()])
    ties_arr = np.array([flatten_model_params(ckpt).numpy() for ckpt in ties_ckpts.values()])
    bc_arr = np.array([flatten_model_params(ckpt).numpy() for ckpt in bc_ckpts.values()])
    hota_arr = np.array([flatten_model_params(ckpt).numpy() for ckpt in hota_ckpts.values()])
    

    # Combine all arrays into one collective array
    collective_arr = np.vstack([pt_arr, tva_arr, ties_arr, bc_arr, hota_arr])

    # Labels for each point
    labels = ['Pretrained', 'Task Arithmetic', 'TIES', 'Breadcrumbs'] + [f'HOTA {i+1}th' for i in range(hota_arr.shape[0])]

    # Grouping information (assign a group ID to each array)
    group_ids = ['Pretrained'] * pt_arr.shape[0] + ['Task Arithmetic'] * tva_arr.shape[0] + ['TIES'] * ties_arr.shape[0] + ['Breadcrumbs'] * bc_arr.shape[0] + ['HOTA'] * hota_arr.shape[0]

    # Assign colors to each group
    colors = {'Pretrained': 'black', 'Task Arithmetic': 'green', 'TIES': 'blue', 'Breadcrumbs': 'orange', 'HOTA': 'red'}

    for perplexity in [1, 2, 4, 7, 10]:
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=0, perplexity=2)
        data_2d = tsne.fit_transform(collective_arr)

        # Create a plot
        plt.figure(figsize=(10, 6))

        # Plot the 2D result, assigning colors based on group ID
        for group_id, color in colors.items():
            indices = [i for i, g in enumerate(group_ids) if g == group_id]
            plt.scatter(data_2d[indices, 0], data_2d[indices, 1], alpha=0.8, color=color, label=group_id, s=50)

        # Annotate each point with its label
        for i in range(data_2d.shape[0]):
            plt.annotate(labels[i], (data_2d[i, 0], data_2d[i, 1]), fontsize=8, alpha=0.8)

        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True)
        plt.legend(prop={'size': 14})
        plt.show()



        plot_filename = 'tsne_' + "perplexity" + str(perplexity) + ".png"
        plt.savefig(plot_filename)
        plt.close()

        # Initialize a wandb run
        wandb.init(project='task-vectors-playground', entity='gladia')
        artifact = wandb.Artifact(plot_filename, type='figure')
        artifact.add_file(plot_filename)
        wandb.log_artifact(artifact)

        # Finish the wandb run
        wandb.finish()

def main():
    run()


if __name__ == "__main__":
    main()
