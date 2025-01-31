from rich import print
from rich.pretty import pprint

from tvp.utils.io_utils import import_json_from_disk

import os
import matplotlib.pyplot as plt

def plot_and_save_line(x_points, y_points, x_label, y_label, title, x_ticks, y_ticks, output_path):
    """
    Plots a line given x and y points and saves the figure.

    Parameters:
        x_points (list): List of x values.
        y_points (list): List of y values.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the plot.
        x_ticks (list): Tick positions for the x-axis.
        y_ticks (list): Tick positions for the y-axis.
        output_path (str): Path to save the output plot (e.g., 'plot.png').
    """
    plt.figure(figsize=(8, 6))  # Set figure size
    plt.plot(x_points, y_points, marker='o', linestyle='-', color='b', label="Line")  # Plot the line
    
    plt.xlabel(x_label)  # Set x-axis label
    plt.ylabel(y_label)  # Set y-axis label
    plt.title(title)      # Set plot title
    
    plt.xticks(x_ticks)   # Set x-axis ticks
    plt.yticks(y_ticks)   # Set y-axis ticks
    
    plt.legend()          # Show legend
    
    print(f"Saving plot to '{output_path}'...")
    plt.savefig(output_path, dpi=400)  # Save the figure with 400 DPI
    plt.close()  # Close the plot to free memory


def main():
    orders = list(range(1, 25))

    TRAIN_MODE: str = "valFT"
    # TRAIN_MODE: str = "trainFT"
    train_batches_ratio: float = 0.1 if TRAIN_MODE == "valFT" else 1.0

    ord_eval_res_file_path = (
        f"evaluations/atm-true/ViT-B-16_0_atm-true_confl_res_none_train_batches_{train_batches_ratio}_ord_"
        f"___ORD_PLACEHOLDER___"
        f"_eps_per_ord_1_merged.json"
    )

    avg_norm_merged_accs = []
    
    for ord in orders:
        ord_eval_res: dict = import_json_from_disk(
            file_path=ord_eval_res_file_path.replace("___ORD_PLACEHOLDER___", str(ord))
        )["results"]

        pprint(ord_eval_res, expand_all=True)

        avg_norm_merged_accs.append(ord_eval_res["average_of_tasks"])

    pprint(avg_norm_merged_accs, expand_all=True)

    export_file_path = (
        f"plots/ord_vs_avg_norm_merged_acc/ord_vs_avg_norm_merged_acc_{TRAIN_MODE}.png"
    )
    os.makedirs(os.path.dirname(export_file_path), exist_ok=True)
    plot_and_save_line(
        x_points=orders,
        y_points=avg_norm_merged_accs,
        x_label="Order",
        y_label="Average Normalized Merged Accuracy",
        title=f"Average Normalized Merged Accuracy vs. Order\n{TRAIN_MODE}",
        x_ticks=orders,
        y_ticks=None,
        output_path=export_file_path
    )

    




if __name__ == '__main__':
    main()