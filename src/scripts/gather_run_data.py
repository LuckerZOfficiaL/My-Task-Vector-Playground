from rich import print

from src.tvp.data.datasets.constants import DATASETS_PAPER_ATM
from src.tvp.data.datasets.constants import DATASET_TO_STYLED

import pandas as pd


def _load_df(dataset: str, ft_regime: str, optim: str):

    df_path = f"./evaluations/ft/ViT-B-16_{dataset}_0_{ft_regime}_{optim}_history.csv"

    print(f"loading df: {df_path}")

    df = pd.read_csv(df_path)
    
    # convert epoch column to int
    df["epoch"] = df["epoch"].astype(int)

    return df


def _get_acc_at_epoch(
    df: pd.DataFrame, 
    epoch: int
):

    df_epoch = df[df["epoch"] == epoch]



def main():

    run_data = {}
    

    for dataset in DATASETS_PAPER_ATM:

        dataset = DATASET_TO_STYLED[dataset]

        run_data[dataset] = {}

        for ft_regime in ["atm", "ta"]:

            run_data[dataset][ft_regime] = {}
            
            for optim in ["adam", "sgd"]:

                run_data[dataset][ft_regime][optim] = {}

                print(f"Dataset: {dataset}, FT Regime: {ft_regime}, Optim: {optim}")

                df = _load_df(dataset, ft_regime, optim)


                run_data[dataset][ft_regime][optim] = {
                    "acc/val": {
                        "first_epoch": -1,
                        "last_epoch": -2
                    },

                    "acc/test": -3
                }


                print("\n\n")





if __name__ == '__main__':
    main()