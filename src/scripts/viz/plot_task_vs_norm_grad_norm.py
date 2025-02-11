from typing import Dict
from torch import Tensor

def remove_none_grads(grad: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {k: v for k, v in grad.items() if v is not None}

from rich import print
from rich.pretty import pprint
from tvp.data.datasets.constants import DATASETS_PAPER_TA, DATASET_TO_STYLED, DATASET_TO_NUM_BATCHES
import torch

def main():

    EPOCHS = list(range(0, 5))

    data: Dict[str, Dict[int, float]] = {}

    epoch_wise_grad_norm_sum: Dict[int, float] = {e: 0.0 for e in EPOCHS}

    for d in DATASETS_PAPER_TA:

        if d == "resisc45":
            break

        data[d] = {}

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

            grad_tensor = torch.cat([v.flatten() for v in grad.values()])
            
            norm_grad = grad_tensor.norm(p=2)

            data[d][e] = norm_grad.item()

            epoch_wise_grad_norm_sum[e] += norm_grad.item()
    
    print(f"data:")
    pprint(data, expand_all=True)

    print(f"epoch_wise_grad_norm_sum:")
    pprint(epoch_wise_grad_norm_sum, expand_all=True)


    

if __name__ == '__main__':
    main()