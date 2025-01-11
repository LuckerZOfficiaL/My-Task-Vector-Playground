from rich import print
from rich.pretty import pprint

from src.tvp.utils.io_utils import import_json_from_disk, export_json_to_disk
import os


def main():
    FT_SUMMARY_FILE = "./evaluations/ft_summary/ft_summary.json"

    ft_summary: dict = import_json_from_disk(FT_SUMMARY_FILE)
    pprint(ft_summary, expand_all=True)

    metrics = {}

    for dataset, dataset_summary in ft_summary.items():
        print(f"\n\n\nDataset: {dataset}")

        for optimizer, optimizer_summary in dataset_summary["ta"].items():
            print(f"\n\n\nOptimizer: {optimizer}")

            if optimizer not in metrics:
                metrics[optimizer] = {}

            acc_first_epoch = ft_summary[dataset]["ta"][optimizer]["acc/val"]["first_epoch"]
            acc_last_epoch = ft_summary[dataset]["ta"][optimizer]["acc/test"]
            
            acc_gap = acc_last_epoch - acc_first_epoch
            acc_ratio =  acc_first_epoch / acc_last_epoch

            loss_first_epoch = ft_summary[dataset]["ta"][optimizer]["loss/val"]["first_epoch"]
            loss_last_epoch = ft_summary[dataset]["ta"][optimizer]["loss/test"]

            loss_gap = loss_first_epoch - loss_last_epoch
            normalized_loss_gap = (loss_first_epoch - loss_last_epoch) / loss_first_epoch

            metrics[optimizer][dataset] = {
                "acc_gap": acc_gap,
                "acc_ratio": acc_ratio,
                "loss_gap": loss_gap,
                "normalized_loss_gap": normalized_loss_gap
            }

    pprint(metrics, expand_all=True)

    export_dir = "./evaluations/task_difficulty"
    os.makedirs(export_dir, exist_ok=True)
    export_json_to_disk(
        data=metrics, 
        export_dir=export_dir, 
        file_name="task_difficulty_metrics"
    )

if __name__ == "__main__":
    main()