from rich import print
from rich.pretty import pprint

from src.tvp.utils.io_utils import import_json_from_disk

from src.tvp.data.datasets.constants import DATASETS_PAPER_TSV_20
from src.tvp.data.datasets.constants import DATASET_TO_STYLED

import matplotlib.pyplot as plt
import numpy as np
import os


def radar_plot(atm: dict, ta: dict, title: str, plot_path: str):
    labels = list(atm.keys())
    atm_values = list(atm.values())
    ta_values = list(ta.values())

    # Repeat the first value to close the radar chart
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Closing the loop

    # # Add the first value to the end of each data list to close the loop
    atm_values += atm_values[:1]
    ta_values += ta_values[:1]

    # Create a polar subplot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot the ATM data
    ax.fill(angles, atm_values, color="red", alpha=0.25, label="ATM")
    ax.plot(angles, atm_values, color="red", linewidth=2)

    # Plot the TA data
    ax.fill(angles, ta_values, color="blue", alpha=0.25, label="TA")
    ax.plot(angles, ta_values, color="blue", linewidth=2)

    # Format the radar chart
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])  # Exclude the repeated angle
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1)

    # Add a title
    plt.title(title, size=14, color="black", loc="center")

    # Add a legend
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    # Save the plot
    plt.savefig(plot_path, bbox_inches="tight", dpi=400)
    plt.close()


def main():
    FT_SUMMARY_FILE = "./evaluations/ft_summary/ft_summary.json"
    ft_summary: dict = import_json_from_disk(FT_SUMMARY_FILE)   

    print(f"ft_summary = ")
    pprint(ft_summary, expand_all=True)
    print()
    
    MERGED_SUMMARY_DIR = "./evaluations/merged"

    PLOT_DIR = "./plots/merged"
    os.makedirs(PLOT_DIR, exist_ok=True)

    tasks = "-".join([DATASET_TO_STYLED[t] for t in DATASETS_PAPER_TSV_20])

    for optim in ["adam", "sgd"]:

        merged_summary_atm: dict = import_json_from_disk(
            f"{MERGED_SUMMARY_DIR}/ViT-B-16_0_atm_{optim}_merged_{tasks}.json"
        )["results"]

        # pprint(merged_summary_atm, expand_all=True)

        merged_summary_ta: dict = import_json_from_disk(
            f"{MERGED_SUMMARY_DIR}/ViT-B-16_0_ta_{optim}_merged_{tasks}.json"
        )["results"]

        # pprint(merged_summary_ta, expand_all=True)

        assert merged_summary_atm.keys() == merged_summary_ta.keys()
      
        acc_test_atm = {
            dataset: merged_summary_atm[dataset][0]["acc/test"] for dataset in merged_summary_atm.keys() if "average_of_tasks" not in dataset
        }
        acc_test_atm["average_of_tasks"] = sum(acc_test_atm.values()) / len(acc_test_atm)

        print(f"acc_test_atm_{optim} = ")
        pprint(acc_test_atm, expand_all=True)
        print()
        
        acc_test_ta = {
            dataset: merged_summary_ta[dataset][0]["acc/test"] for dataset in merged_summary_ta.keys() if "average_of_tasks" not in dataset
        }
        acc_test_ta["average_of_tasks"] = sum(acc_test_ta.values()) / len(acc_test_ta)

        print(f"acc_test_ta_{optim} = ")
        pprint(acc_test_ta, expand_all=True)  
        print()
        
        acc_test_atm_norm = {
            dataset: acc_test_atm[dataset] / ft_summary[dataset]["ta"][optim]["acc/test"] for dataset in merged_summary_atm.keys() if "average_of_tasks" not in dataset
        }
        acc_test_atm_norm["average_of_tasks"] = sum(acc_test_atm_norm.values()) / len(acc_test_atm_norm)

        print(f"acc_test_atm_norm_{optim} = ")
        pprint(acc_test_atm_norm, expand_all=True)
        print()

        acc_test_ta_norm = {
            dataset: acc_test_ta[dataset] / ft_summary[dataset]["ta"][optim]["acc/test"] for dataset in merged_summary_ta.keys() if "average_of_tasks" not in dataset
        }
        acc_test_ta_norm["average_of_tasks"] = sum(acc_test_ta_norm.values()) / len(acc_test_ta_norm)

        print(f"acc_test_ta_norm_{optim} = ")
        pprint(acc_test_ta_norm, expand_all=True)
        print()

        radar_plot(
            atm=acc_test_atm,
            ta=acc_test_ta,
            title=f"ATM vs. TA, optim={optim}, norm=False",
            plot_path=f"{PLOT_DIR}/atm_vs_ta_{optim}_norm_false_{tasks}.png"
        )
        
        radar_plot(
            atm=acc_test_atm_norm,
            ta=acc_test_ta_norm,
            title=f"ATM vs. TA, optim={optim}, norm=True",
            plot_path=f"{PLOT_DIR}/atm_vs_ta_{optim}_norm_true_{tasks}.png"
        )
        




    


    
    
    














if __name__ == "__main__":
    main()