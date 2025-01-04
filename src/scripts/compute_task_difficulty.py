from tvp.data.constants import DATASETS_20
from tvp.data.constants import DATASET_NAME_TO_STYLED_NAME
from tvp.data.constants import DATASET_NAME_TO_NUM_TRAIN_BATCHES_LOWERCASE
from tvp.data.constants import DATASET_NAME_TO_TA_FT_EPOCHS_LOWERCASE

import pandas as pd

from rich.pretty import pprint

import os
import matplotlib.pyplot as plt


def plot_or_save_metric(metric_name, metric_values, dataset_names, title, ylabel, save_to_disk=False, output_dir="./plots"):
    """
    Plots or saves a bar chart for a given metric.

    Args:
        metric_name (str): The name of the metric.
        metric_values (list): The values of the metric.
        dataset_names (list): The dataset names corresponding to the metric values.
        title (str): The title of the plot.
        ylabel (str): The label for the y-axis.
        save_to_disk (bool): If True, saves the plot to disk; otherwise, displays it.
        output_dir (str): The directory where plots will be saved if save_to_disk is True.
    """
    plt.figure(figsize=(12, 6))
    plt.bar(dataset_names, metric_values, color='skyblue')
    plt.title(title)
    plt.ylabel(rf'${ylabel}$')
    plt.xlabel('Dataset Name')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_to_disk:
        import os
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{metric_name}.png"
        plt.savefig(output_path, dpi=400)
        print(f"Plot saved to: {output_path}")
        plt.close()
    else:
        plt.show()

def main():
    RUN_DATA_DIR = "./run_data"

    BATCH_SIZE = 32

    ACC_GRAD_BATCHES = 1

    OPTIM = "SGD"

    EPOCH_NAN_PLACEHOLDER_VALUE = -6969

    DATASETS_TO_SKIP = []

    rows = []

    TASK_DIFFICULTY_DIR = "./task_difficulty"

    for dataset in DATASETS_20:

        if dataset in DATASETS_TO_SKIP:
            continue

        print(f"dataset: {dataset}")
        print("\n\n", f"*"*50, "\n\n")

        df_file_path = (
            f"{RUN_DATA_DIR}/"
            f"ViT-B-16_"
            f"{DATASET_NAME_TO_STYLED_NAME[dataset]}_"
            f"0_"
            f"batch_size_{BATCH_SIZE}_"
            f"lim_train_batches_{DATASET_NAME_TO_NUM_TRAIN_BATCHES_LOWERCASE[dataset]}_"
            f"acc_grad_batches_{ACC_GRAD_BATCHES}_"
            f"epochs_{DATASET_NAME_TO_TA_FT_EPOCHS_LOWERCASE[dataset]}_"
            f"optim_{OPTIM}_"
            f"order_1_"
            f"history.csv"
        )

        df = pd.read_csv(df_file_path)

        # Fill NaN epochs with placeholder and convert to integer
        df['epoch'] = df['epoch'].fillna(EPOCH_NAN_PLACEHOLDER_VALUE).astype(int)

        print(df)
        print("\n\n", f"*"*50, "\n\n")

        # Filter rows where both validation loss and accuracy are available, and sort by epoch
        filtered_df = df[
            df['acc/val'].notna() & df['loss/val'].notna()
        ].sort_values(by='epoch')

        print(filtered_df)
        print("\n\n", f"*"*50, "\n\n")

        # Extract first and last epochs' loss and accuracy
        loss_first_epoch = float(filtered_df.iloc[0]['loss/val'])
        loss_last_epoch = float(filtered_df.iloc[-1]['loss/val'])
        acc_first_epoch = float(filtered_df.iloc[0]['acc/val'])
        acc_last_epoch = float(filtered_df.iloc[-1]['acc/val'])
        
        row = {
            'dataset_name': dataset,

            # Accuracy Gap: Accuracy improvement from the first to the last epoch
            # - Higher values --> fine-tuning significantly improves the model
            # - Lower values --> little to no improvement from fine-tuning
            'acc_gap': acc_last_epoch - acc_first_epoch,
            
            # Accuracy Ratio: Ratio of initial accuracy to final accuracy
            # - Closer to 1 --> the pretrained model already performs well on the task
            # - Closer to 0 --> pt model does not perform well on the task.
            'acc_ratio': acc_first_epoch / acc_last_epoch,

            # Loss Gap: Absolute decrease in loss during fine-tuning
            # - Higher values --> fine-tuning significantly improves the model
            # - Lower values --> little to no improvement from fine-tuning
            'loss_gap': loss_first_epoch - loss_last_epoch,

            # Loss Gap: Proportionate decrease in loss during fine-tuning
            # - Higher values --> fine-tuning significantly i   mproves the model
            # - Lower values --> little to no improvement from fine-tuning
            'normalized_loss_gap': (loss_first_epoch - loss_last_epoch) / loss_first_epoch,
        }

        rows.append(row)

    # Create a DataFrame to store metrics for all datasets
    difficulty_df = pd.DataFrame(rows)

    # Print the difficulty DataFrame for inspection
    pprint(difficulty_df)

    # Save the DataFrame to a CSV file
    difficulty_df_file_path = (
        f"{TASK_DIFFICULTY_DIR}/"
        f"ViT-B-16_"
        f"DATASET-20_"
        f"0_"
        f"batch_size_{BATCH_SIZE}_"
        f"lim_train_batches_ALL-BATCHES_"
        f"acc_grad_batches_{ACC_GRAD_BATCHES}_"
        f"epochs_TA_"
        f"optim_{OPTIM}_"
        f"order_1_"
        f"difficulties.csv"
    )

    difficulty_df.to_csv(difficulty_df_file_path, index=True)

    # Extract dataset names and metrics
    dataset_names = difficulty_df['dataset_name']
    metrics = {
        'acc_gap': ('Accuracy Gap', r'\text{Ep}_{\text{last}} \, \text{Accuracy} - \text{Ep}_1 \, \text{Accuracy} \, (\text{higher means more challenging task})'),
        'acc_ratio': ('Accuracy Ratio', r'\frac{\text{Ep}_1 \, \text{Accuracy}}{\text{Ep}_{\text{last}} \, \text{Accuracy}} \, (\text{closer to 1 means easier task})'),
        'loss_gap': ('Loss Gap', r'\text{Ep}_1 \, \text{Loss} - \text{Ep}_{\text{last}} \, \text{Loss} \, (\text{higher means more challenging task})'),
        'normalized_loss_gap': ('Normalized Loss Gap', r'\frac{\text{Ep}_1 \, \text{Loss} - \text{Ep}_{\text{last}} \, \text{Loss}}{\text{Ep}_1 \, \text{Loss}} \, (\text{higher means more challenging task})'),
    }

    # Flag to decide whether to plot or save
    SAVE_PLOTS_TO_DISK = True  # Set to False to display plots instead
    OUTPUT_DIR = f"./plots/task_difficulty"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate a bar plot for each metric, sorting by the appropriate order
    for metric, (title, ylabel) in metrics.items():
        # Decide sorting order based on the metric
        if metric in ['acc_gap', 'loss_gap', 'normalized_loss_gap']:
            sort_ascending = False  # Higher values are more challenging
        elif metric == 'acc_ratio':
            sort_ascending = True  # Closer to 1 means easier task

        # Sort the DataFrame by the current metric
        sorted_difficulty_df = difficulty_df.sort_values(by=metric, ascending=sort_ascending)

        # Extract the sorted dataset names and corresponding metric values
        sorted_dataset_names = sorted_difficulty_df['dataset_name']
        sorted_metric_values = sorted_difficulty_df[metric]

        plot_or_save_metric(
            metric_name=metric,
            metric_values=sorted_metric_values,
            dataset_names=sorted_dataset_names,
            title=title,
            ylabel=ylabel,
            save_to_disk=SAVE_PLOTS_TO_DISK,
            output_dir=OUTPUT_DIR,
        )


if __name__ == "__main__":
    main()
