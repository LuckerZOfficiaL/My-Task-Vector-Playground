from omegaconf import DictConfig

from nn_core.callbacks import NNTemplateCore
from nn_core.model_logging import NNLogger

def init_logger(cfg: DictConfig) -> NNLogger:
    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None)
    )
    
    logger: NNLogger = NNLogger(
        logging_cfg=cfg.train.logging, 
        cfg=cfg, 
        resume_id=template_core.resume_id
    )

    return logger

def get_grad_name(
    dataset: str,
    batch: int
) -> str:

    return (
        f"./grads/atm-true/"
        f"ViT-B-16_{dataset}_0"
        f"_atm-true_confl_res_none_train_batches_1.0"
        f"_ord_1_eps_per_ord_10_ep_0_batch_{batch}_grads.pt"
    )

def get_ckpt_name(
    dataset: str,
    epoch: int,
    batch: int
) -> str:
    # tries to load --> ckpts/atm-true/ViT-B-16_Flowers102_0_atm-true_confl_res_none_train_batches_1.0_ord_1_eps_per_ord_10_ep_1_batch_7_ckpt.pt
    # actual file ----> ckpts/atm-true/ViT-B-16_Flowers102_0_atm-true_confl_res_none_train_batches_1.0_ord_1_eps_per_ord_10_ep_0_batch_7.pt
    return (
        f"./ckpts/atm-true/"
        f"ViT-B-16_{dataset}_0"
        f"_atm-true_confl_res_none_train_batches_1.0"
        f"_ord_1_eps_per_ord_10_ep_{epoch}_batch_{batch}_ckpt.pt"
    )

from typing import Dict, Union, List
from torch import Tensor

def remove_none_grads(
    grads: Dict[str, Tensor]
) -> Union[Dict[str, Tensor], List[str]]:

    grads_cleaned: Dict[str, Tensor] = {}
    removed_layers: List[str] = []
    
    for k, v in grads.items():
        if v is not None:
            grads_cleaned[k] = v
        else:
            removed_layers.append(k)

    return grads_cleaned, removed_layers

def remove_layers_with_none_grads(
    ckpt: Dict[str, Tensor],
    layers_with_none_grad: List[str]
) -> Dict[str, Tensor]:

    ckpt_cleaned: Dict[str, Tensor] = {}

    for k, v in ckpt.items():
        if k not in layers_with_none_grad:
            ckpt_cleaned[k] = v

    return ckpt_cleaned

def get_task_vector(
    pt: Dict[str, Tensor],
    ft: Dict[str, Tensor],
) -> Dict[str, Tensor]:

    if pt.keys() != ft.keys():
        raise ValueError("The keys of the two pt and ft ckpts are not the same")
    
    tv: Dict[str, Tensor] = {}

    for k, v in pt.items():
        tv[k] = ft[k] - pt[k]

    return tv

from torch.nn.functional import cosine_similarity as torch_cosine_similarity

def cosine_simimilarity(
    tv: Dict[str, Tensor],
    grads_first_ep: Dict[str, Tensor]
) -> float:

    tv_flat: Tensor = torch.cat([v.flatten() for v in tv.values()])
    grads_flat: Tensor = torch.cat([v.flatten() for v in grads_first_ep.values()])

    cos_sim: Tensor = torch_cosine_similarity(grads_flat, -1 * tv_flat, dim=0)

    return cos_sim.item()

from rich import print
from rich.pretty import pprint
from tvp.data.datasets.constants import DATASETS_PAPER_TA

import hydra
from nn_core.common import PROJECT_ROOT
import torch 
from tvp.data.datasets.constants import DATASET_TO_STYLED, DATASET_TO_NUM_BATCHES
import warnings

# Filter out the specific FutureWarning from torch.load regarding weights_only
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`*",
    category=FutureWarning
)

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="task_vectors.yaml")
def main(cfg: DictConfig) -> None:

    logger: NNLogger = init_logger(cfg)

    cos_sims: Dict[str, List[float]] = {}
    tv_mags: Dict[str, List[float]] = {}
    tv_mags_sum_per_ep: Dict[int, float] = {}
    
    for d in DATASETS_PAPER_TA:

        cos_sims[d] = []
        tv_mags[d] = []

        print(f"Dataset: {d}")

        grads_first_ep: Dict[str, Tensor] = torch.load(
            f=get_grad_name(
                dataset=DATASET_TO_STYLED[d], 
                batch=DATASET_TO_NUM_BATCHES[d][128] - 1
            ),
            map_location="cpu"
        )
        grads_first_ep, layers_with_none_grad = remove_none_grads(grads_first_ep)

        for e in range(1, 10):

            tv_mags_sum_per_ep[e] = 0.0

            ckpt_current: Dict[str, Tensor] = torch.load(
                f=get_ckpt_name(
                    dataset=DATASET_TO_STYLED[d], 
                    epoch=e, 
                    batch=DATASET_TO_NUM_BATCHES[d][128] - 1
                ),
                map_location="cpu"
            )
            ckpt_current = remove_layers_with_none_grads(ckpt_current, layers_with_none_grad)

            ckpt_previous: Dict[str, Tensor] = torch.load(
                f=get_ckpt_name(
                    dataset=DATASET_TO_STYLED[d], 
                    epoch=e-1,
                    batch=DATASET_TO_NUM_BATCHES[d][128] - 1
                ),
                map_location="cpu"
            )
            ckpt_previous = remove_layers_with_none_grads(ckpt_previous, layers_with_none_grad)

            if ckpt_current.keys() != ckpt_previous.keys():
                raise ValueError("The keys of the two checkpoints are not the same")

            if grads_first_ep.keys() != ckpt_current.keys():
                raise ValueError("The keys of the first epoch grads are not the same as the checkpoint keys")

            tv: Dict[str, Tensor] = get_task_vector(
                pt=ckpt_previous, ft=ckpt_current
            )

            cos_sims[d].append(cosine_simimilarity(tv, grads_first_ep))
            tv_mags[d].append(
                torch.norm(
                    torch.cat([v.flatten() for v in tv.values()]), p=2
                ).item()
            )
            tv_mags_sum_per_ep[e] += tv_mags[d][-1]

    pprint(cos_sims, expand_all=True)
    print("\n")
    pprint(tv_mags, expand_all=True)
    print("\n")

    for d in tv_mags.keys():
        for i in range(len(tv_mags[d])):
            tv_mags[d][i] /= sum(tv_mags[d])

    pprint(tv_mags, expand_all=True)



if __name__ == "__main__":
    main()

