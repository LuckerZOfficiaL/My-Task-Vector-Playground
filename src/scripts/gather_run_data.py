from rich import print

from src.tvp.data.datasets.constants import DATASETS_PAPER_TSV_20
from src.tvp.data.datasets.constants import DATASET_TO_STYLED

import pandas as pd

import os
from src.tvp.utils.io_utils import export_json_to_disk


def _load_df(dataset: str, ft_regime: str, optim: str):

    df_path = f"./evaluations/ft/ViT-B-16_{dataset}_0_{ft_regime}_{optim}_history.csv"

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
    
    for dataset in DATASETS_PAPER_TSV_20:

        dataset = DATASET_TO_STYLED[dataset]

        run_data[dataset] = {}

        for ft_regime in ["atm", "ta"]:

            run_data[dataset][ft_regime] = {}
            
            for optim in ["adam", "sgd"]:

                try: 

                    run_data[dataset][ft_regime][optim] = {}

                    df = _load_df(dataset, ft_regime, optim)

                    run_data[dataset][ft_regime][optim] = {
                        "acc/val": {
                            "first_epoch": _get_metric_at_epoch(
                                df=df, epoch=0, metric="acc/val"
                            ),
                            "last_epoch": _get_metric_at_epoch(
                                df=df, epoch=_get_max_epoch(df) - 1, metric="acc/val"
                            )
                        },
                        
                        "loss/val": {
                            "first_epoch": _get_metric_at_epoch(
                                df=df, epoch=0, metric="loss/val"
                            ),
                            "last_epoch": _get_metric_at_epoch(
                                df=df, epoch=_get_max_epoch(df) - 1, metric="loss/val"
                            )
                        },

                        "acc/test": _get_metric_at_epoch(df, _get_max_epoch(df), "acc/test"),

                        "loss/test": _get_metric_at_epoch(df, _get_max_epoch(df), "loss/test")
                    }

                except Exception as e:
                        
                    print(f"Dataset: {dataset}, FT Regime: {ft_regime}, Optim: {optim}")
                    print(f"Error: {e}")
                    print(f"\n\n")

    export_dir = "./evaluations/ft_summary/"
    os.makedirs(export_dir, exist_ok=True)
    export_json_to_disk(
        data=run_data, export_dir=export_dir, file_name="ft_summary"
    )





if __name__ == '__main__':
    main()