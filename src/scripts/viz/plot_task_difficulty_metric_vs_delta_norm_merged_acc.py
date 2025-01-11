from rich import print
from rich.pretty import pprint

from src.tvp.utils.io_utils import import_json_from_disk, list_all_files_in_dir
import os

from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def _check_list_of_merged_accs(merged_accs_file_ta, merged_accs_file_atm):
    if len(merged_accs_file_ta) != len(merged_accs_file_atm):
        raise ValueError(
            "The number of TA and ATM merged accuracies files are not equal."
        )

    for ta, atm in zip(merged_accs_file_ta, merged_accs_file_atm):
        ta = ta.replace("_ta_", "")
        atm = atm.replace("_atm_", "")

        if ta != atm:
            raise ValueError(
                f"TA and ATM merged accuracies files are not equal: {ta} != {atm}"
            )


def _get_list_of_merged_accs(merged_accs_dir: str):
    merged_accs_files = list_all_files_in_dir(merged_accs_dir)
    merged_accs_files = [os.path.join(merged_accs_dir, f) for f in merged_accs_files]
    
    merged_accs_files = [f for f in merged_accs_files if f.endswith(".json")]

    merged_accs_file_ta = [
        f for f in merged_accs_files if "_ta_" in f
    ]
    merged_accs_file_ta = sorted(merged_accs_file_ta)

    merged_accs_file_atm = [
        f for f in merged_accs_files if "_atm_" in f
    ]
    merged_accs_file_atm = sorted(merged_accs_file_atm)

    _check_list_of_merged_accs(merged_accs_file_ta, merged_accs_file_atm)

    return merged_accs_files, merged_accs_file_ta, merged_accs_file_atm


def _get_norm_merged_acc(accs: dict, ft_summary: dict):

    accs_norm = {}

    for t in accs.keys():
        
        if "average_of_tasks" in t:
            continue

        accs_norm[t] = accs[t][0]["acc/test"] / ft_summary[t]["ta"]["adam"]["acc/test"]

    accs_norm["average_of_tasks"] = sum(accs_norm.values()) / len(accs_norm.keys())

    return accs_norm["average_of_tasks"]


def _prepare_data_for_plot(
    merged_accs_file_ta: List[str], 
    merged_accs_file_atm: List[str],
    ft_summary: dict,
    task_difficulties: dict
):

    df_row_list = []

    for ta, atm in zip(merged_accs_file_ta, merged_accs_file_atm):

        if ta.replace("_ta_", "") != atm.replace("_atm_", ""):
            raise ValueError(
                f"TA and ATM merged accuracies files are not equal: {ta} != {atm}"
            )
    
        accs_ta: dict = import_json_from_disk(ta)["results"]
        accs_atm: dict = import_json_from_disk(atm)["results"]

        if accs_ta.keys() != accs_atm.keys():
            raise ValueError(
                f"TA and ATM merged accuracies keys are not equal: {accs_ta.keys()} != {accs_atm.keys()}"
            )

        # pprint(accs_ta, expand_all=True)
        # pprint(accs_atm, expand_all=True)

        norm_merged_acc_ta = _get_norm_merged_acc(accs_ta, ft_summary)
        norm_merged_acc_atm = _get_norm_merged_acc(accs_atm, ft_summary)

        avg_acc_gap = sum(
            [task_difficulties["adam"][t]["acc_gap"] for t in accs_ta.keys() if t != "average_of_tasks"]
        ) / len(accs_ta.keys())

        avg_acc_ratio = sum(
            [task_difficulties["adam"][t]["acc_ratio"] for t in accs_ta.keys() if t != "average_of_tasks"]
        ) / len(accs_ta.keys())

        avg_loss_gap = sum(
            [task_difficulties["adam"][t]["loss_gap"] for t in accs_ta.keys() if t != "average_of_tasks"]
        ) / len(accs_ta.keys())

        avg_norm_loss_gap = sum(
            [task_difficulties["adam"][t]["normalized_loss_gap"] for t in accs_ta.keys() if t != "average_of_tasks"]
        ) / len(accs_ta.keys())

        df_row_list.append({
            "tasks": ta.split("/")[-1].split("_")[5].split(".")[0],
            "norm_merged_acc_delta": norm_merged_acc_atm - norm_merged_acc_ta,
            "avg_acc_gap": avg_acc_gap,
            "avg_acc_ratio": avg_acc_ratio,
            "avg_loss_gap": avg_loss_gap,
            "avg_normalized_loss_gap": avg_norm_loss_gap
        })

    # TODO check this with Luca
    df = pd.DataFrame(df_row_list)

    # pprint(df)

    return df


def _plot_or_save(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    save_path: str = None
):
    """
    Plots a scatter plot using specified columns from the DataFrame and custom labels.
    Adds a regression line to the scatter plot.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data.
    - x_col (str): The column to use for the x-axis.
    - y_col (str): The column to use for the y-axis.
    - title (str): The title of the plot.
    - x_label (str): The label for the x-axis.
    - y_label (str): The label for the y-axis.
    - save_path (str, optional): If provided, saves the plot to the specified path.
                                  Otherwise, the plot will be displayed.

    Returns:
    None
    """
    # Check if specified columns exist in the DataFrame
    if x_col not in df.columns:
        raise ValueError(f"Columns '{x_col}' not found in the DataFrame.")
    
    if y_col not in df.columns:
        raise ValueError(f"Columns '{y_col}' not found in the DataFrame.")

    # Extract data
    x = df[x_col]
    y = df[y_col]

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.7, edgecolor='k', label="Merged accuracies")

    # Fit a regression line
    coeffs = np.polyfit(x, y, deg=1)  # Linear regression (degree 1)
    regression_line = np.polyval(coeffs, x)
    plt.plot(x, regression_line, color='red', linestyle='-', linewidth=2, label="Trend line")

    # Add labels, title, and grid
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save or display the plot
    if save_path:
        plt.savefig(save_path, format='png', dpi=400, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    # Close the figure to free up memory
    plt.close()


def _plot(
    df: pd.DataFrame,
    add_to_title: str,
    plot_save_path: str
):

    METRICS = ["acc_gap", "acc_ratio", "loss_gap", "normalized_loss_gap"]
    METRIC_NAMES = ["Accuracy Gap", "Accuracy Ratio", "Loss Gap", "Normalized Loss Gap"]
    # TODO check this with Luca
    METRIC_LABELS = [
        "Accuracy Gap\n(higher means more difficult task)", 
        "Accuracy Ratio\n(higher means simpler task)", 
        "Loss Gap\n(higher means more difficult task)", 
        "Normalized Loss Gap\n(higher means more difficult task)"
    ]


    for metric, metric_name, metric_label in zip(METRICS, METRIC_NAMES, METRIC_LABELS):

        _plot_or_save(
            df=df,
            x_col=f"avg_{metric}",
            y_col="norm_merged_acc_delta",
            title=f"{metric_name} vs. Delta Normalized Merged Accuracy\n{add_to_title}",
            x_label=f"Average {metric_label}",
            y_label="Delta Normalized Merged Accuracy\n(norm_merged_acc_atm - norm_merged_acc_ta)",
            save_path=plot_save_path.replace(".png", f"_{metric}.png")
        )




def main():
    TASK_DIFFICULTY_FILE = "./evaluations/task_difficulty/task_difficulty_metrics.json"
    # TODO check this with Luca
    task_difficulties = import_json_from_disk(TASK_DIFFICULTY_FILE)

    FT_SUMMARY_FILE = "./evaluations/ft_summary/ft_summary.json"
    ft_summary = import_json_from_disk(FT_SUMMARY_FILE)

    # pprint(task_difficulties, expand_all=True)

    merged_accs_dir = "./evaluations/merged_subsets/paper-tsv-20/subset_size_05"
    _, merged_accs_file_ta, merged_accs_file_atm = _get_list_of_merged_accs(merged_accs_dir)

    df: pd.DataFrame = _prepare_data_for_plot(
        merged_accs_file_ta=merged_accs_file_ta, 
        merged_accs_file_atm=merged_accs_file_atm,
        ft_summary=ft_summary,
        task_difficulties=task_difficulties
    )

    plot_dir = "./plots/task_difficulty_metric_vs_delta_norm_merged_acc"
    os.makedirs(plot_dir, exist_ok=True)
    plot_name = "tasks_paper-tsv-20_num_subsets_200_subset_size_05"
    _plot(
        df=df,
        add_to_title="Tasks = paper-tsv-20. Num subsets = 200. Subset size = 05",
        plot_save_path=f"{plot_dir}/{plot_name}.png"
    )



if __name__ == '__main__':
    main()