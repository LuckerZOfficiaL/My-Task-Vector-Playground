from rich import print
from rich.pretty import pprint

import numpy as np

def main():
    

    MAX_ORDERS = 30

    cos_sims_per_order = {}

    for order in range(1, MAX_ORDERS + 1):
        cos_sim_path = (
            f"evaluations/sims_dists_vs_zs/"
            f"ViT-B-16_0_atm-true"
            f"_confl_res_none"
            f"_train_batches_0.1"
            f"_ord_{order}"
            f"_eps_per_ord_1"
            f"_cos_sims"
            f".npy"
        )

        cos_sims: np.ndarray = np.load(cos_sim_path, allow_pickle=True)
        mask = ~np.eye(cos_sims.shape[0], dtype=bool)
        cos_sims = cos_sims[mask].mean()

        cos_sims_per_order[order] = cos_sims

    pprint(cos_sims_per_order, expand_all=True)


if __name__ == "__main__":
    main()