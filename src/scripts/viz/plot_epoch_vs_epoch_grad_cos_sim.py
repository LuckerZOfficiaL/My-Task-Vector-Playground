from torch import cosine_similarity
from typing import Dict
from torch import Tensor

# NOTE make sure grads_first and grads_second do NOT have any None grads
def layerwise_cosine_similarity(
    grads_first: Dict[str, Tensor],
    grads_second: Dict[str, Tensor],
) -> float:

    if grads_first.keys() != grads_second.keys():
        raise ValueError("Parameter keys do not match!")

    count_params = 0
    running_cos_sin = 0

    for layer_name in grads_first.keys():

        if grads_first[layer_name].shape != grads_second[layer_name].shape:
            raise ValueError(f"Grad shape mismatch for {layer_name}!")

        cos_sim = cosine_similarity(
            grads_first[layer_name].flatten().cpu(),
            grads_second[layer_name].flatten().cpu(),
            dim=0
        ).item()
        running_cos_sin += cos_sim * grads_first[layer_name].numel()
        count_params += grads_first[layer_name].numel()

    return running_cos_sin / count_params

def remove_none_grads(grads: Dict[str, Tensor]) -> Dict[str, Tensor]:

    return {k: v for k, v in grads.items() if v is not None}

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def heatmap(
    data: Tensor,
    row_labels: list[str],
    col_labels: list[str],
    export_path: str,

):

    # latex stuff for the paper
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'

    sns.set_theme()

    fig, ax = plt.subplots()

    sns.heatmap(data, annot=True, fmt=".2f", ax=ax, cmap="RdYlGn")

    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Epoch")

    plt.savefig(export_path, dpi=400)

    plt.show()

from rich import print
from rich.pretty import pprint
from tvp.data.datasets.constants import DATASETS_PAPER_TA, DATASET_TO_STYLED, DATASET_TO_NUM_BATCHES
import torch
import os
import tqdm

def main():

    EPOCHS = list(range(0, 10))
    
    for d in DATASETS_PAPER_TA:

        if d == "dtd":
            break

        grads: Dict[int, Tensor] = {}

        for e in EPOCHS:

            print(f"Dataset: {d}, Epoch: {e}")

            grad_file_path = (
                f"./grads/atm-true/ViT-B-16"
                f"_{DATASET_TO_STYLED[d]}_0_atm-true_confl_res_none"
                f"_train_batches_1.0_ord_1_eps_per_ord_10"
                f"_ep_{e}_batch_{DATASET_TO_NUM_BATCHES[d][128] - 1}_grads.pt"
            )

            grad: Dict[str, Tensor] = torch.load(
                f=grad_file_path, map_location="cpu"
            )
            grad: Dict[str, Tensor] = remove_none_grads(grad)

            # pprint(grad, expand_all=True)

            grads[e] = grad

        cos_sims_layerwise = torch.empty(len(EPOCHS), len(EPOCHS))
        cos_sims_flattened = torch.empty(len(EPOCHS), len(EPOCHS))

        for e in EPOCHS:
            for e_inv in EPOCHS:

                print(f"Dataset: {d}, Epochs: {e} vs. {e_inv}")

                cos_sims_layerwise[e][e_inv] = layerwise_cosine_similarity(grads[e], grads[e_inv])
                cos_sims_flattened[e][e_inv] = cosine_similarity(
                    torch.cat([v.flatten() for v in grads[e].values()]),
                    torch.cat([v.flatten() for v in grads[e_inv].values()]),
                    dim=0
                ).item()

        heatmap_export_path = (
            f"./plots/heatmap_epoch_vs_epoch_grad_cos_sim/"
            f"heatmap_epoch_vs_epoch_grad_cos_sim"
            f"_{DATASET_TO_STYLED[d]}_layerwise.png"
        )
        os.makedirs(os.path.dirname(heatmap_export_path), exist_ok=True)
        heatmap(
            data=cos_sims_layerwise,
            row_labels=[e + 1 for e in EPOCHS],
            col_labels=[e + 1 for e in EPOCHS],
            export_path=heatmap_export_path
        )

        heatmap_export_path = (
            f"./plots/heatmap_epoch_vs_epoch_grad_cos_sim/"
            f"heatmap_epoch_vs_epoch_grad_cos_sim"
            f"_{DATASET_TO_STYLED[d]}_flattened.png"
        )
        os.makedirs(os.path.dirname(heatmap_export_path), exist_ok=True)
        heatmap(
            data=cos_sims_flattened,
            row_labels=[e + 1 for e in EPOCHS],
            col_labels=[e + 1 for e in EPOCHS],
            export_path=heatmap_export_path
        )

        print(f"\n\n\n")



        





if __name__ == '__main__':
    main()
