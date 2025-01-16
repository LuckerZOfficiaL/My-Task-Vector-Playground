from rich import print
from rich.pretty import pprint

import pandas as pd
from pandas import DataFrame
from src.tvp.utils.io_utils import import_json_from_disk, export_json_to_disk
import os


def main():

    # NOTE these have to be kept fixed to these values
    # as this is the only setup that makes sense to consider, when selecting task difficulties
    FT_REGIME = "ta"
    OPTIM = "adamw_wd_0.1"
    LR_SCHEDULER = "lr_scheduler_cosine_annealing_warmup_steps_200"
    
    FT_SUMMARY_FILE = f"./evaluations/ft_summary/ft_summary_{FT_REGIME}_{OPTIM}_{LR_SCHEDULER}.csv"

    ft_summary: DataFrame = pd.read_csv(FT_SUMMARY_FILE)
    pprint(ft_summary, expand_all=True)

    metrics = []

    for _, dataset_summary in ft_summary.iterrows():
        print(f"\n\n\nDataset: {dataset_summary}")

        acc_first_epoch = dataset_summary["acc_val_first_epoch"]
        acc_last_epoch = dataset_summary["acc_test"]
        
        acc_gap = acc_last_epoch - acc_first_epoch
        acc_ratio =  acc_first_epoch / acc_last_epoch

        loss_first_epoch = dataset_summary["loss_val_first_epoch"]
        loss_last_epoch = dataset_summary["loss_test"]

        loss_gap = loss_first_epoch - loss_last_epoch
        normalized_loss_gap = loss_gap / loss_first_epoch

        metrics.append(
            {
                "dataset": dataset_summary["dataset"],
                "acc_gap": acc_gap,
                "acc_ratio": acc_ratio,
                "loss_gap": loss_gap,
                "normalized_loss_gap": normalized_loss_gap
            }
        )

    metrics = pd.DataFrame(metrics)
    pprint(metrics, expand_all=True)

    export_dir = f"./evaluations/task_difficulty"
    os.makedirs(export_dir, exist_ok=True)
    file_name = f"task_difficulty_metrics_{FT_REGIME}_{OPTIM}_{LR_SCHEDULER}.csv"

    metrics.to_csv(os.path.join(export_dir, file_name), index=True)

if __name__ == "__main__":
    main()