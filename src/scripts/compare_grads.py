import os

import torch

from rich import print

from tvp.data.constants import DATASET_NAME_TO_NUM_BATCHES_UPPERCASE

from torch.linalg import vector_norm

def main():
    GRADS_FOLDER = "./grads_manual_loop"

    DATASETS = [
        "CIFAR100", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"
    ]

    for dataset in DATASETS[:1]:

        grads_sgd = torch.load(
            f=os.path.join(
                GRADS_FOLDER, 
                f"ViT-B-16_{dataset}_0_batch_size_32_lim_train_batches_1_acc_grad_batches_1_epochs_1_optim_SGD_order_1.pt"
            ),
            map_location=torch.device("cpu")
        )

        grads_gd = torch.load(
            f=os.path.join(
                GRADS_FOLDER, 
                f"ViT-B-16_{dataset}_0_batch_size_32_lim_train_batches_{DATASET_NAME_TO_NUM_BATCHES_UPPERCASE[dataset]}_acc_grad_batches_{DATASET_NAME_TO_NUM_BATCHES_UPPERCASE[dataset]}_epochs_1_optim_SGD_order_1.pt"
            ),
            map_location=torch.device("cpu")
        )

        print(f"\n\n\n")
        print(f"Dataset: {dataset}")

        for layer_sgd, layer_gd in zip(grads_sgd.keys(), grads_gd.keys()):
            print(f"grads_sgd[layer].shape: {grads_sgd[layer_sgd].shape}")
            print(f"grads_gd[layer].shape : {grads_gd[layer_gd].shape}")

            print(f"cosine similarity: {torch.nn.functional.cosine_similarity(grads_sgd[layer_sgd].flatten(), grads_sgd[layer_gd].flatten(), dim=0)}")




if __name__ == "__main__":
    main()