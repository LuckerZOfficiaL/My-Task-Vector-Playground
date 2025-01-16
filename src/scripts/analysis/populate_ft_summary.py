from rich import print

from src.tvp.data.datasets.constants import DATASETS_PAPER_TSV_20
from src.tvp.data.datasets.constants import DATASET_TO_STYLED

import pandas as pd

import os
from src.tvp.utils.io_utils import export_json_to_disk


def _load_df(dataset: str, ft_regime: str, optim: str, lr_scheduler: str):

    df_path = os.path.join(
        "./evaluations/ft",
        ft_regime,
        optim.split("_")[0],
        f"lr_scheduler_{lr_scheduler}",
        f"ViT-B-16_{dataset}_0_{ft_regime}_{optim}_lr_scheduler_{lr_scheduler}_history.csv"
    )

    # print(f"loading df: {df_path}")

    df = pd.read_csv(df_path)
    
    return df


def _get_max_epoch(df: pd.DataFrame):
    
    return int(df["epoch"].max())


def _get_metric_at_epoch(
    df: pd.DataFrame, 
    epoch: int,
    metric: str
):

    df = df.dropna(subset=[metric])

    df = df[df["epoch"] == epoch]

    acc = df[metric].values[0]

    acc = float(acc)

    return acc



def main():

    run_data = {}

    configs = [
        {
            "ft_regime": "ta",
            "optim": "adamw_wd_0.1",
            "lr_scheduler": "cosine_annealing_warmup_steps_200"
        },
        
        {
            "ft_regime": "ta",
            "optim": "adam_wd_0.0",
            "lr_scheduler": "none"
        },

        {
            "ft_regime": "atm",
            "optim": "adam_wd_0.0",
            "lr_scheduler": "none"
        },

        {
            "ft_regime": "atm",
            "optim": "adamw_wd_0.1",
            "lr_scheduler": "cosine_annealing_warmup_steps_200"
        },

        {
            "ft_regime": "atm",
            "optim": "adamw_wd_0.1",
            "lr_scheduler": "cosine_annealing_warmup_steps_0.1"
        }
    ]

    for config in configs:

        rows = []

        for dataset in DATASETS_PAPER_TSV_20:

            dataset = DATASET_TO_STYLED[dataset]

            print(f"Dataset: {dataset}, FT Regime: {config['ft_regime']}, Optim: {config['optim']}, LR Scheduler: {config['lr_scheduler']}")
            
            try: 
                df = _load_df(
                    dataset=dataset,
                    ft_regime=config["ft_regime"],
                    optim=config["optim"],
                    lr_scheduler=config["lr_scheduler"]
                )

                rows.append(
                    {
                        "dataset": dataset,
                        "ft_regime": config["ft_regime"],
                        "optim": config["optim"],
                        "lr_scheduler": config["lr_scheduler"],
                        "acc_val_first_epoch": _get_metric_at_epoch(df, 0, "acc/val"),
                        "acc_val_last_epoch": _get_metric_at_epoch(df, _get_max_epoch(df) - 1, "acc/val"),
                        "loss_val_first_epoch": _get_metric_at_epoch(df, 0, "loss/val"),
                        "loss_val_last_epoch": _get_metric_at_epoch(df, _get_max_epoch(df) - 1, "loss/val"),
                        "acc_test": _get_metric_at_epoch(df, _get_max_epoch(df), "acc/test"),
                        "loss_test": _get_metric_at_epoch(df, _get_max_epoch(df), "loss/test")
                
                    }
                )
            except Exception as e:
                print(f"Error: {e}")
                print(f"\n\n")

        df = pd.DataFrame(rows)

        export_dir = "./evaluations/ft_summary/"
        os.makedirs(export_dir, exist_ok=True)
        export_path = os.path.join(
            export_dir,
            f"ft_summary_{config['ft_regime']}_{config['optim']}_lr_scheduler_{config['lr_scheduler']}.csv"
        )
        df.to_csv(export_path, index=True)


    
    
    
    
    
    
    
    
    exit()
    
    for dataset in DATASETS_PAPER_TSV_20:

        dataset = DATASET_TO_STYLED[dataset]

        run_data[dataset] = {}

        for ft_regime in ["atm", "ta"]:

            run_data[dataset][ft_regime] = {}
            
            for optim in ["adam_wd_0.0", "adamw_wd_0.1"]:

                run_data[dataset][ft_regime][optim] = {}

                for lr_scheduler in optim_to_lr_scheduler[optim]:

                    print(f"Dataset: {dataset}, FT Regime: {ft_regime}, Optim: {optim}, LR Scheduler: {lr_scheduler}")


                    run_data[dataset][ft_regime][optim][lr_scheduler] = {}

                    df = _load_df(dataset, ft_regime, optim, lr_scheduler)

                    #     continue

                    #     run_data[dataset][ft_regime][optim] = {
                    #         "acc/val": {
                    #             "first_epoch": _get_metric_at_epoch(
                    #                 df=df, epoch=0, metric="acc/val"
                    #             ),
                    #             "last_epoch": _get_metric_at_epoch(
                    #                 df=df, epoch=_get_max_epoch(df) - 1, metric="acc/val"
                    #             )
                    #         },
                            
                    #         "loss/val": {
                    #             "first_epoch": _get_metric_at_epoch(
                    #                 df=df, epoch=0, metric="loss/val"
                    #             ),
                    #             "last_epoch": _get_metric_at_epoch(
                    #                 df=df, epoch=_get_max_epoch(df) - 1, metric="loss/val"
                    #             )
                    #         },

                    #         "acc/test": _get_metric_at_epoch(df, _get_max_epoch(df), "acc/test"),

                    #         "loss/test": _get_metric_at_epoch(df, _get_max_epoch(df), "loss/test")
                    #     }

                    # except Exception as e:
                            
                    #     print(f"Dataset: {dataset}, FT Regime: {ft_regime}, Optim: {optim}")
                    #     print(f"Error: {e}")
                    #     print(f"\n\n")

    export_dir = "./evaluations/ft_summary/"
    os.makedirs(export_dir, exist_ok=True)
    export_json_to_disk(
        data=run_data, export_dir=export_dir, file_name="ft_summary"
    )





if __name__ == '__main__':
    main()