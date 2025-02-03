from rich import print
from rich.pretty import pprint

from tvp.data.datasets.constants import DATASETS_PAPER_TSV_20
from tvp.data.datasets.constants import DATASETS_PAPER_TA

import numpy as np
from typing import Dict

import seaborn as sns
import matplotlib.pyplot as plt


def main():

    datasets = DATASETS_PAPER_TA
    CONFL_RES_METHODS = ["none", "ties", "bc", "dare"]

    TRAIN_BATCHES = 1.0
    ORD = 1
    EPS_PER_ORD = 10
    CONFL_RES_PLACEHOLDER = "__CONFL_RES_METHOD_NAME__"
    
    cos_sims_path = (
        f"evaluations/sims_dists/"
        f"ViT-B-16_0_atm-true"
        f"_confl_res_{CONFL_RES_PLACEHOLDER}"
        f"_train_batches_{TRAIN_BATCHES}"
        f"_ord_{ORD}"
        f"_eps_per_ord_{EPS_PER_ORD}"
        f"_merged_cos_sims.npy"
    )

    avg_cos_sims = {}
    for confl_res_method in CONFL_RES_METHODS:
        print(f"Loading cos sims for {confl_res_method}")
        cos_sims: np.ndarray = np.load(
            cos_sims_path.replace(CONFL_RES_PLACEHOLDER, confl_res_method), 
            allow_pickle=True
        )
        mask = ~np.eye(cos_sims.shape[0], dtype=bool)
        cos_sim = cos_sims[mask].mean()

        avg_cos_sims[confl_res_method] = cos_sim

    cos_sims_path = (
        f"evaluations/sims_dists_vs_zs/"
        f"ViT-B-16_0_atm-true"
        f"_confl_res_none"
        f"_train_batches_0.1"
        f"_ord_10"
        f"_eps_per_ord_1"
        f"_merged_cos_sims.npy"
    )
    cos_sims: np.ndarray = np.load(cos_sims_path, allow_pickle=True)
    mask = ~np.eye(cos_sims.shape[0], dtype=bool)
    cos_sims = cos_sims[mask].mean()
    avg_cos_sims["atm"] = cos_sims

    pprint(avg_cos_sims, expand_all=True)

    euclidean_dists_path = (
        f"evaluations/sims_dists/"
        f"ViT-B-16_0_atm-true"
        f"_confl_res_{CONFL_RES_PLACEHOLDER}"
        f"_train_batches_{TRAIN_BATCHES}"
        f"_ord_{ORD}"
        f"_eps_per_ord_{EPS_PER_ORD}"
        f"_merged_euclidean_dists.npy"
    )

    avg_euclidean_dists = {}

    for confl_res_method in CONFL_RES_METHODS:
        print(f"Loading euclidean dists for {confl_res_method}")
        l2_dists: np.ndarray = np.load(
            euclidean_dists_path.replace(CONFL_RES_PLACEHOLDER, confl_res_method), 
            allow_pickle=True
        )
        mask = ~np.eye(l2_dists.shape[0], dtype=bool)
        l2_dist = l2_dists[mask].mean()
        avg_euclidean_dists[confl_res_method] = l2_dist

    euclidean_dists_path = (
        f"evaluations/sims_dists_vs_zs/"
        f"ViT-B-16_0_atm-true"
        f"_confl_res_none"
        f"_train_batches_0.1"
        f"_ord_10"
        f"_eps_per_ord_1"
        f"_merged_euclidean_dists.npy"
    )
    l2_dists: np.ndarray = np.load(euclidean_dists_path, allow_pickle=True)
    mask = ~np.eye(l2_dists.shape[0], dtype=bool)
    l2_dists = l2_dists[mask].mean()
    avg_euclidean_dists["atm"] = l2_dists

    pprint(avg_euclidean_dists, expand_all=True)


if __name__ == "__main__":
    main()