import copy

from rich.pretty import pprint
from rich import print

from tvp.data.constants import DATASETS_20
from tvp.data.constants import DATASET_NAME_TO_STYLED_NAME
from tvp.data.constants import DATASET_NAME_TO_NUM_TRAIN_BATCHES_UPPERCASE
from tvp.data.constants import DATASET_NAME_TO_TA_FT_EPOCHS_UPPERCASE

import pandas as pd

import os

from typing import List

from tvp.utils.io_utils import get_all_files_in_dir, load_dict_from_json

import matplotlib.pyplot as plt
import numpy as np


RUN_DATA_DIR = "./run_data"

MODEL_NAME = "ViT-B-16"
BATCH_SIZE = 32
ACC_GRAD_BATCHES = 1
OPTIM = "SGD"
ORDER = 1

TASK_DIFFICULTY_DF_PATH = (
    f"./task_difficulty/"
    f"{MODEL_NAME}_"
    f"DATASET-20_"
    f"0_"
    f"batch_size_{BATCH_SIZE}_"
    f"lim_train_batches_ALL-BATCHES_"
    f"acc_grad_batches_{ACC_GRAD_BATCHES}_"
    f"epochs_TA_"
    f"optim_{OPTIM}_"
    f"order_{ORDER}_"
    f"difficulties.csv"
)

DATASETS_TO_EXCLUDE = ["pcam"]

DATASETS = copy.deepcopy(DATASETS_20)

# TA
MAX_EPOCHS = None
# ATM
# MAX_EPOCHS = 1

if MAX_EPOCHS is None:
    TA_OR_ATM = "TA"
elif MAX_EPOCHS == 1:
    TA_OR_ATM = "ATM"
else:
    raise ValueError(f"Invalid value for MAX_EPOCHS, expected None or 1, got {MAX_EPOCHS}")

TASK_EQUIPPED_ACCS_DIR = "./evaluations/evaulate_import_te_models"
NUM_TASKS = 5

# TODO add support for differentiation between ATM and TA
CORRELATION_PLOTS_EXPORT_DIR = f"./plots/correlations/{TA_OR_ATM}"
os.makedirs(CORRELATION_PLOTS_EXPORT_DIR, exist_ok=True)

for dataset in DATASETS_TO_EXCLUDE:
    DATASETS.remove(dataset)


def _populate_datasets_metrics():
    datasets_metrics = {}

    task_difficulty_df = pd.read_csv(TASK_DIFFICULTY_DF_PATH)

    for dataset_name in DATASETS:
        dataset_name = DATASET_NAME_TO_STYLED_NAME[dataset_name]

        df_name = (
            f"{MODEL_NAME}_"
            f"{dataset_name}_"
            f"0_"
            f"batch_size_{BATCH_SIZE}_"
            f"lim_train_batches_{DATASET_NAME_TO_NUM_TRAIN_BATCHES_UPPERCASE[dataset_name]}_"
            f"acc_grad_batches_{ACC_GRAD_BATCHES}_"
            f"epochs_{DATASET_NAME_TO_TA_FT_EPOCHS_UPPERCASE[dataset_name]}_"
            f"optim_{OPTIM}_"
            f"order_{ORDER}_"
            f"history.csv"
        )

        task_ft_df = pd.read_csv(os.path.join(RUN_DATA_DIR, df_name))

        acc_test = float(
            task_ft_df.dropna(subset=["acc/test"])["acc/test"].iloc[0]
        )
        acc_gap = float(
            task_difficulty_df[
                task_difficulty_df["dataset_name"] == dataset_name.lower()
            ]["acc_gap"].iloc[0]
        )
        acc_ratio = float(
            task_difficulty_df[
                task_difficulty_df["dataset_name"] == dataset_name.lower()
            ]["acc_ratio"].iloc[0]
        )
        loss_gap = float(
            task_difficulty_df[
                task_difficulty_df["dataset_name"] == dataset_name.lower()
            ]["loss_gap"].iloc[0]
        )
        normalized_loss_gap = float(
            task_difficulty_df[
                task_difficulty_df["dataset_name"] == dataset_name.lower()
            ]["normalized_loss_gap"].iloc[0]
        )

        datasets_metrics[dataset_name.lower()] = {
            "acc_test": acc_test,
            "acc_gap": acc_gap,
            "acc_ratio": acc_ratio,
            "loss_gap": loss_gap,
            "normalized_loss_gap": normalized_loss_gap
        }


    return datasets_metrics


def _populate_list_of_task_equipped_models():

    list_of_task_equipped_models = get_all_files_in_dir(TASK_EQUIPPED_ACCS_DIR)

    num_dashes = 2 + NUM_TASKS - 1

    list_of_task_equipped_models = [
        model
        for model in list_of_task_equipped_models
        if model.count("-") == num_dashes and not any(
            DATASET_NAME_TO_STYLED_NAME[dataset] in model for dataset in DATASETS_TO_EXCLUDE
        )
    ]

    list_of_task_equipped_models = [
        model
        for model in list_of_task_equipped_models
        if f"_epochs_{TA_OR_ATM}_" in model
    ]

    return list_of_task_equipped_models


def _compute_normalized_merged_accs(
    datasets_metrics: dict, 
    merged_accs: dict
):

    datasets_metrics = {
        DATASET_NAME_TO_STYLED_NAME[dataset_name]: datasets_metrics[dataset_name]
        for dataset_name in datasets_metrics.keys()
    }

    return {
        dataset_name: merged_accs[dataset_name][0]["acc/test"] / datasets_metrics[dataset_name]["acc_test"]
        for dataset_name in merged_accs
    }



def _populate_metrics_for_task_equipped_models(
    datasets_metrics: dict,
    list_of_task_equipped_models: List[str]
):

    task_equipped_models_metrics = {}

    for task_equipped_model in list_of_task_equipped_models:

        merged_accs_full_dict = load_dict_from_json(
            filename=os.path.join(TASK_EQUIPPED_ACCS_DIR, task_equipped_model),
            import_description="task-equipped accuracies"
        )

        # print("datasets metrics")
        # pprint(datasets_metrics)

        merged_accs = merged_accs_full_dict["dataset_evals"]
        # print(f"merged accuracies")
        # pprint(merged_accs)

        normalized_merged_accs = _compute_normalized_merged_accs(
            datasets_metrics=datasets_metrics,
            merged_accs=merged_accs
        )
        # print(f"normalized merged accuracies")
        # pprint(normalized_merged_accs)

        acc_gap = {
            dataset_name: datasets_metrics[dataset_name.lower()]["acc_gap"]
            for dataset_name in merged_accs.keys()
        }
        
        acc_ratio = {
            dataset_name: datasets_metrics[dataset_name.lower()]["acc_ratio"]
            for dataset_name in merged_accs.keys()
        }
        
        loss_gap = {
            dataset_name: datasets_metrics[dataset_name.lower()]["loss_gap"]
            for dataset_name in merged_accs.keys()
        }
        
        normalized_loss_gap = {
            dataset_name: datasets_metrics[dataset_name.lower()]["normalized_loss_gap"]
            for dataset_name in merged_accs.keys()
        }

        task_equipped_models_metrics[task_equipped_model] = {
            "normalized_merged_accs": normalized_merged_accs,
            "acc_gap": acc_gap,
            "acc_ratio": acc_ratio,
            "loss_gap": loss_gap,
            "normalized_loss_gap": normalized_loss_gap
        }

    return task_equipped_models_metrics


# def plot_or_export(
#     avg_merged_accs: list, 
#     avg_metric: list, 
#     metric_name: str,
#     metric_y_axis_label: str,
#     plot_title: str,
#     export_path: str
# ):
#     # Define colors for easy modification
#     acc_color = 'darkviolet'
#     metric_color = 'dodgerblue'

#     x_values = range(len(avg_merged_accs))  # Common x values (index of the lists)

#     fig, ax1 = plt.subplots()  # Create the figure and the first y-axis

#     # Plot avg_merged_accs on the left y-axis
#     ax1.scatter(x_values, avg_merged_accs, color=acc_color, label='avg merged accs', alpha=0.7)
#     ax1.set_xlabel('Merged task subsets')
#     ax1.set_ylabel('avg accs merged accs (higher is better)', color=acc_color)
#     ax1.tick_params(axis='y', labelcolor=acc_color)

#     # Add regression line for avg_merged_accs
#     acc_fit = np.polyfit(x_values, avg_merged_accs, 1)
#     acc_fit_line = np.polyval(acc_fit, x_values)
#     ax1.plot(x_values, acc_fit_line, color=acc_color, label='avg accs trend')

#     # Create a second y-axis for the metric
#     ax2 = ax1.twinx()
#     ax2.scatter(x_values, avg_metric, color=metric_color, label=metric_name, alpha=0.7)
#     ax2.set_ylabel(metric_y_axis_label, color=metric_color)
#     ax2.tick_params(axis='y', labelcolor=metric_color)

#     # Add regression line for avg_metric
#     metric_fit = np.polyfit(x_values, avg_metric, 1)
#     metric_fit_line = np.polyval(metric_fit, x_values)
#     ax2.plot(x_values, metric_fit_line, color=metric_color, label=f'{metric_name} trend')

#     # Set custom x-axis labels for the first, middle, and last values
#     ax1.set_xticks([0, len(x_values)//2, len(x_values)-1])
#     ax1.set_xticklabels([0, len(x_values)//2, len(x_values)-1])

#     # Remove grid
#     ax1.grid(False)

#     # Add title
#     plt.title(plot_title)

#     # Check if an export path is given
#     if export_path:
#         plt.savefig(export_path, dpi=300)  # Save the plot to the specified file
#         plt.close()  # Close the plot to avoid displaying it
#         print(f"Plot exported to: {export_path}")
#     else:
#         plt.show()  # Show the plot if no export path is given


def plot_or_export(
    avg_merged_accs: list,
    avg_metric: list,
    metric_name: str,
    metric_x_axis_label: str,
    plot_title: str,
    export_path: str
):
    """
    Creates a scatter plot with metric on the x-axis and accuracy on the y-axis.
    Adds a correlation line between x and y.
    Allows exporting the plot to a file.

    Parameters:
        avg_merged_accs (list): Average accuracies for each subset.
        avg_metric (list): Average metric values for each subset.
        metric_name (str): Name of the metric being plotted.
        metric_x_axis_label (str): Label for the x-axis.
        plot_title (str): Title of the plot.
        export_path (str): File path to save the plot image.
    """
    # Check if lengths match
    if len(avg_merged_accs) != len(avg_metric):
        raise ValueError("The lengths of avg_merged_accs and avg_metric must match.")

    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x=avg_metric, y=avg_merged_accs, color='blue', alpha=0.7)

    # Calculate the trendline (correlation line)
    coefficients = np.polyfit(x=avg_metric, y=avg_merged_accs, deg=1)  # Linear fit (degree=1)
    trendline = np.poly1d(coefficients)  # Create the trendline function
    trendline_values = trendline(avg_metric)  # Compute y values for the trendline

    # Plot the trendline
    plt.plot(avg_metric, trendline_values, color='black', linestyle='-')

    # Labels and title
    plt.xlabel(metric_x_axis_label, fontsize=12)
    plt.ylabel('Normalized Merged Accuracy (higher is better)', fontsize=12)
    plt.title(plot_title, fontsize=14)
    plt.legend()

    # Save the plot to the export path
    if export_path:
        plt.savefig(export_path, format='png', dpi=400)
        print(f"Plot exported to {export_path}")
    else:
        print("No export path provided; displaying the plot instead.")
        plt.show()
    
    # Clear the plot to free up memory
    plt.clf()



def _plot_correlation(
    normalized_merged_accs: dict,
    metric: dict,
    metric_name: str,
    metric_y_axis_label: str,
    plot_title: str,
    export_path: str
):
    # print(f"normalized merged accs")
    # pprint(normalized_merged_accs)

    # print(f"metric")
    # pprint(metric)

    if len(normalized_merged_accs) != len(metric):
        raise ValueError(
            f"The number of datasets in the normalized merged accuracies {len(normalized_merged_accs)} and the metric {len(metric)} must be the same."
        )

    avg_merged_accs  = {}

    for subset_name, merged_acc_dict in normalized_merged_accs.items():
        avg_merged_accs[subset_name] = sum(merged_acc_dict.values()) / len(merged_acc_dict)

    # print(f"avg merged accs")
    # pprint(avg_merged_accs)
    
    avg_metric = {}

    for subset_name, metric in metric.items():
        avg_metric[subset_name] = sum(metric.values()) / len(metric)

    # print(f"avg metric")
    # pprint(avg_metric)

    plot_or_export(
        avg_merged_accs=list(avg_merged_accs.values()),
        avg_metric=list(avg_metric.values()),
        metric_name=metric_name,
        metric_x_axis_label=metric_y_axis_label,
        plot_title=plot_title,
        export_path=export_path
    )


def _plot_correlations(
    task_equipped_metrics: dict,
    export_dir: str
):

    normalized_merged_accs = {
        model: task_equipped_metrics[model]["normalized_merged_accs"]
        for model in list(task_equipped_metrics.keys())
    }
    acc_gaps = {
        model: task_equipped_metrics[model]["acc_gap"]
        for model in list(task_equipped_metrics.keys())
    }

    export_path = os.path.join(export_dir, "acc_gap.png")
    _plot_correlation(
        normalized_merged_accs=normalized_merged_accs,
        metric=acc_gaps,
        metric_name="accuracy gap",
        metric_y_axis_label="accuracy gap (higher means harder task)",
        plot_title="acc gap, normalized merged accs correlation",
        export_path=export_path
    )
    
    acc_ratios = {
        model: task_equipped_metrics[model]["acc_ratio"]
        for model in list(task_equipped_metrics.keys())
    }

    export_path = os.path.join(export_dir, "acc_ratio.png")
    _plot_correlation(
        normalized_merged_accs=normalized_merged_accs,
        metric=acc_ratios,
        metric_name="accuracy ratio",
        metric_y_axis_label="accuracy ratio (higher means easier task)",
        plot_title="acc ratio, normalized merged accs correlation",
        export_path=export_path
    )
    
    loss_gaps = {
        model: task_equipped_metrics[model]["loss_gap"]
        for model in list(task_equipped_metrics.keys())
    }

    export_path = os.path.join(export_dir, "loss_gap.png")
    _plot_correlation(
        normalized_merged_accs=normalized_merged_accs,
        metric=loss_gaps,
        metric_name="loss gap",
        metric_y_axis_label="loss gap (higher means harder task)",
        plot_title="loss gap, normalized merged accs correlation",
        export_path=export_path
    )
    
    normalized_loss_gaps = {
        model: task_equipped_metrics[model]["normalized_loss_gap"]
        for model in list(task_equipped_metrics.keys())
    }

    export_path = os.path.join(export_dir, "normalized_loss_gap.png")
    _plot_correlation(
        normalized_merged_accs=normalized_merged_accs,
        metric=normalized_loss_gaps,
        metric_name="normalized loss gap",
        metric_y_axis_label="normalized loss gap (higher means harder task)",
        plot_title="normalized loss gap, normalized merged accs correlation",
        export_path=export_path
    )
        
        

        


def main():
    datasets_metrics = _populate_datasets_metrics()
    # print(f"datasets metrics")
    # pprint(datasets_metrics)

    list_of_task_equipped_models = _populate_list_of_task_equipped_models()
    # print(f"list of task equipped models")
    # pprint(list_of_task_equipped_models)
    print(f"Number of task-equipped models: {len(list_of_task_equipped_models)}")

    task_equipped_metrics = _populate_metrics_for_task_equipped_models(
        datasets_metrics=datasets_metrics,
        list_of_task_equipped_models=list_of_task_equipped_models
    )
    # print(f"task equipped metrics")
    # pprint(task_equipped_metrics, expand_all=True)

    _plot_correlations(
        task_equipped_metrics=task_equipped_metrics,
        export_dir=CORRELATION_PLOTS_EXPORT_DIR
    )




if __name__ == '__main__':
    main()