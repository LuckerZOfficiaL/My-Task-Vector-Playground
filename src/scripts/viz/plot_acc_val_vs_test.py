from rich import print
from rich.pretty import pprint

from src.tvp.utils.io_utils import import_json_from_disk

def main():
    FT_SUMMARY_FILE = "./evaluations/ft_summary/ft_summary.json"

    ft_summary: dict = import_json_from_disk(FT_SUMMARY_FILE)

    comparison_acc_test = {}
    comparison_acc_val  = {}

    acc_gap_adam = []
    acc_gap_sgd = []

    for dataset in ft_summary.keys():

        dataset_summary: dict = ft_summary[dataset]

        for ft_regime in ["atm", "ta"]:

            for optim in ["adam", "sgd"]:
                key = f"{dataset}_{ft_regime}_{optim}"
                comparison_acc_test[key] = dataset_summary[ft_regime][optim]["acc/test"]

                comparison_acc_val[key] = dataset_summary[ft_regime][optim]["acc/val"]["last_epoch"]

                acc_gap = comparison_acc_test[key] - comparison_acc_val[key]
                print(f"{key}: {acc_gap}")

                acc_gap_adam.append(acc_gap) if optim == "adam" else acc_gap_sgd.append(acc_gap)


    print(f"Adam: {sum(acc_gap_adam) / len(acc_gap_adam)}")
    print(f"SGD: {sum(acc_gap_sgd) / len(acc_gap_sgd)}")

    
    


    
    
    














if __name__ == "__main__":
    main()