from rich import print
from rich.pretty import pprint

import pandas as pd
from src.tvp.utils.io_utils import import_json_from_disk, list_all_files_in_dir
import os

from typing import List

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np


def _check_list_of_merged_accs(merged_accs_files_ta, merged_accs_files_atm):
    if len(merged_accs_files_ta) != len(merged_accs_files_atm):
        raise ValueError(
            "The number of TA and ATM merged accuracies files are not equal."
        )

    for ta, atm in zip(merged_accs_files_ta, merged_accs_files_atm):
        ta = ta.split("/")[-1].split("_")[-1].split(".")[0]
        atm = atm.split("/")[-1].split("_")[-1].split(".")[0]

        if ta != atm:
            raise ValueError(
                f"TA and ATM merged accuracies files are not equal: {ta} != {atm}"
            )


def _get_list_of_merged_accs(
    merged_accs_dir_atm: str,
    merged_accs_dir_ta: str
):  
    merged_accs_files_atm = list_all_files_in_dir(merged_accs_dir_atm)
    merged_accs_files_atm = [os.path.join(merged_accs_dir_atm, f) for f in merged_accs_files_atm]
    merged_accs_files_atm = sorted(merged_accs_files_atm)

    merged_accs_files_ta = list_all_files_in_dir(merged_accs_dir_ta)
    merged_accs_files_ta = [os.path.join(merged_accs_dir_ta, f) for f in merged_accs_files_ta]
    merged_accs_files_ta = sorted(merged_accs_files_ta)

    _check_list_of_merged_accs(merged_accs_files_ta, merged_accs_files_atm)

    return merged_accs_files_atm, merged_accs_files_ta


def _get_norm_merged_acc(accs: dict, ft_summary: DataFrame):

    accs_norm = {}

    for t in accs.keys():
        
        if "average_of_tasks" in t:
            continue

        # accs_norm[t] = accs[t][0]["acc/test"] / ft_summary[t]["ta"]["adam"]["acc/test"]
        accs_norm[t] = accs[t][0]["acc/test"] / float(ft_summary[ft_summary["dataset"] == t]["acc_test"])

    accs_norm["average_of_tasks"] = sum(accs_norm.values()) / len(accs_norm.keys())

    return accs_norm["average_of_tasks"]


def _prepare_data_for_plot(
    merged_accs_files_ta: List[str], 
    merged_accs_files_atm: List[str],
    ft_summary: DataFrame,
    task_difficulties: DataFrame
):

    df_row_list = []

    for ta, atm in zip(merged_accs_files_ta, merged_accs_files_atm):

        if ta.split("/")[-1].split("_")[-1] != atm.split("/")[-1].split("_")[-1]:
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

        # pprint(norm_merged_acc_ta, expand_all=True)
        # pprint(norm_merged_acc_atm, expand_all=True)

        avg_acc_gap = sum(
            [float(task_difficulties[task_difficulties["dataset"] == t]["acc_gap"].iloc[0]) for t in accs_ta.keys() if t != "average_of_tasks"]
        ) / len(accs_ta.keys())

        avg_acc_ratio = sum(
            [float(task_difficulties[task_difficulties["dataset"] == t]["acc_ratio"].iloc[0]) for t in accs_ta.keys() if t != "average_of_tasks"]
        ) / len(accs_ta.keys())

        avg_loss_gap = sum(
            [float(task_difficulties[task_difficulties["dataset"] == t]["loss_gap"].iloc[0]) for t in accs_ta.keys() if t != "average_of_tasks"]
        ) / len(accs_ta.keys())

        avg_norm_loss_gap = sum(
            [float(task_difficulties[task_difficulties["dataset"] == t]["normalized_loss_gap"].iloc[0]) for t in accs_ta.keys() if t != "average_of_tasks"]
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
    TASK_DIFFICULTY_FILE = "./evaluations/task_difficulty/task_difficulty_metrics_ta_adamw_wd_0.1_lr_scheduler_cosine_annealing_warmup_steps_200.csv"
    task_difficulties = pd.read_csv(TASK_DIFFICULTY_FILE)

    FT_SUMMARY_FILE = "./evaluations/ft_summary/ft_summary_ta_adamw_wd_0.1_lr_scheduler_cosine_annealing_warmup_steps_200.csv"
    ft_summary = pd.read_csv(FT_SUMMARY_FILE)

    # pprint(task_difficulties, expand_all=True)

    TASKS = "paper-tsv-20"
    SUBSET_SIZE = "05"
    OPTIM = "adamw_wd_0.1"
    LR_SCHEDULER_TA = "cosine_annealing_warmup_steps_200"
    LR_SCHEDULER_ATM = "none"
    ATM_SUBDIR = f"atm/optim_{OPTIM}/{LR_SCHEDULER_ATM}"
    TA_SUBDIR  = f"ta/optim_{OPTIM}/{LR_SCHEDULER_TA}"
    EVALS_DIR = f"./evaluations/merged_subsets/{TASKS}"

    merged_accs_dir_atm = f"{EVALS_DIR}/{ATM_SUBDIR}/subset_size_{SUBSET_SIZE}"
    merged_accs_dir_ta = f"{EVALS_DIR}/{TA_SUBDIR}/subset_size_{SUBSET_SIZE}"
    merged_accs_files_atm, merged_accs_files_ta = _get_list_of_merged_accs(
        merged_accs_dir_atm=merged_accs_dir_atm,
        merged_accs_dir_ta=merged_accs_dir_ta
    )

    df: pd.DataFrame = _prepare_data_for_plot(
        merged_accs_files_ta=merged_accs_files_ta, 
        merged_accs_files_atm=merged_accs_files_atm,
        ft_summary=ft_summary,
        task_difficulties=task_difficulties
    )

    num_subsets = len(df)

    plot_dir = "./plots/task_difficulty_metric_vs_delta_norm_merged_acc/"
    os.makedirs(plot_dir, exist_ok=True)
    plot_name = f"{TASKS}_num_subsets_{num_subsets}_subset_size_{SUBSET_SIZE}_atm_{OPTIM}_{LR_SCHEDULER_ATM}_ta_{OPTIM}_{LR_SCHEDULER_TA}"
    _plot(
        df=df,
        add_to_title=f"Tasks = {TASKS}. Num subsets = {num_subsets}. Subset size = {SUBSET_SIZE}",
        plot_save_path=f"{plot_dir}/{plot_name}.png"
    )



if __name__ == '__main__':
    main()