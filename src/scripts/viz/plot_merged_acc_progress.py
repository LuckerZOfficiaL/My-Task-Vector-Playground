from rich import print
from rich.pretty import pprint

from tvp.data.datasets.constants import DATASETS_PAPER_ATM
from tvp.data.datasets.constants import DATASETS_PAPER_TSV_14
from tvp.data.datasets.constants import DATASETS_PAPER_TSV_20
from tvp.data.datasets.constants import DATASETS_PAPER_TA
from tvp.data.datasets.constants import DATASET_TO_STYLED
from tvp.utils.io_utils import import_json_from_disk

import os
import matplotlib.pyplot as plt


def task_dict_key_to_str(dict_key: str):
    if dict_key == "20":
        return "paper-tsv-20"
    elif dict_key == "14":
        return "paper-tsv-14"
    elif dict_key == "atm":
        return "paper-atm"
    elif dict_key == "ta":
        return "paper-ta"
    else:
        raise ValueError(f"Unknown task dict key: {dict_key}")

def plot_ratio_vs_acc(
    plot_data: dict,
    plot_exort_path: str
):
    """
    Plots the ratio (x-axis) vs accuracy (y-axis) for the given plot_data.
    
    Args:
        plot_data (dict): A dictionary where each key represents a line plot, 
                          and its associated value is another dictionary where:
                          - Keys are x-axis values (ratios) as strings.
                          - Values are y-axis values (e.g., accuracies).
    """
    # Create a figure for the plot
    plt.figure(figsize=(8, 6))
    
    # Loop through the keys in plot_data to create individual plots
    for key, data in plot_data.items():
        # Convert x-axis keys to float and extract y-axis values
        x = [float(k) for k in data.keys()]
        y = list(data.values())
        
        # Plot each line
        plt.plot(x, y, label=f"{task_dict_key_to_str(key)} tasks")
    
    # Add labels, title, and legend
    plt.xlabel("Training progress (% of total steps)")
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.ylabel("Normalized merged accuracy")
    plt.title("Normalized merged accuracies at different training progresses")
    plt.legend(title="Task sets")
    
    plt.savefig(plot_exort_path, dpi=400)

def main():

    ZS_ACCS = {
        "atm": 0.540858622108187,
        "ta": 0.5513642728328705,
        "14" : 0.6095666268042156,
        "20" : 0.5924221374094486
    }

    datasets = {}
    datasets['atm'] = "-".join([DATASET_TO_STYLED[t] for t in DATASETS_PAPER_ATM])
    datasets['ta'] = "-".join([DATASET_TO_STYLED[t] for t in DATASETS_PAPER_TA])
    datasets['14'] = "-".join([DATASET_TO_STYLED[t] for t in DATASETS_PAPER_TSV_14])
    datasets['20'] = "-".join([DATASET_TO_STYLED[t] for t in DATASETS_PAPER_TSV_20])

    acc_paths = {}
    acc_paths['atm'] = f"evaluations/merged_progress_merging/ViT-B-16_0_ta_adamw_wd_0.1_lr_scheduler_cosine_annealing_warmup_steps_200_merged_{datasets['atm']}.json"
    acc_paths['ta'] = f"evaluations/merged_progress_merging/ViT-B-16_0_ta_adamw_wd_0.1_lr_scheduler_cosine_annealing_warmup_steps_200_merged_{datasets['ta']}.json"
    acc_paths['14'] = f"evaluations/merged_progress_merging/ViT-B-16_0_ta_adamw_wd_0.1_lr_scheduler_cosine_annealing_warmup_steps_200_merged_{datasets['14']}.json"
    acc_paths['20'] = f"evaluations/merged_progress_merging/ViT-B-16_0_ta_adamw_wd_0.1_lr_scheduler_cosine_annealing_warmup_steps_200_merged_{datasets['20']}.json"


    accs = {}
    accs['atm'] = import_json_from_disk(acc_paths['atm'])["results_all_ratios"]
    accs['ta'] = import_json_from_disk(acc_paths['ta'])["results_all_ratios"]
    accs['14'] = import_json_from_disk(acc_paths['14'])["results_all_ratios"]
    accs['20'] = import_json_from_disk(acc_paths['20'])["results_all_ratios"]

    print("\n\n")
    pprint(accs, expand_all=True)
    print("\n\n")

    EXPECTED_RATIOS = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]

    for dataset_size in accs.keys():
        if set(accs[dataset_size].keys()) != set(EXPECTED_RATIOS):
            raise ValueError(f"Expected ratios: {set(EXPECTED_RATIOS)}. Found ratios: {set(accs[dataset_size].keys())}")

    plot_data = {}

    for dataset_cardinality in datasets.keys():
        print(f"Dataset cardinality: {dataset_cardinality}")

        plot_data[dataset_cardinality] = {}

        plot_data[dataset_cardinality]["0.0"] = ZS_ACCS[dataset_cardinality]
        for ratio in EXPECTED_RATIOS:
            
            plot_data[dataset_cardinality][ratio] = accs[dataset_cardinality][ratio]['average_of_tasks']
    

    pprint(plot_data, expand_all=True)

    plot_export_path = "plots/merged_progress_merging/merged_progress_merging_plot.png"
    os.makedirs(os.path.dirname(plot_export_path), exist_ok=True)
    plot_ratio_vs_acc(plot_data, plot_export_path)




    


if __name__ == "__main__":
    main()