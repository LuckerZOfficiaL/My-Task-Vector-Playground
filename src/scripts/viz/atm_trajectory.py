import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl

def plot_2d_embeddings(embeddings, labels, colors, file_path, legend_handles, label_offset=(5, 5)):
    """
    Plot 2D embeddings with labels and custom colors, with adjustable label positions
    and a legend that maps each color to its corresponding name.

    Parameters:
        embeddings (list or array-like): List of 2D points. Each point should be iterable with 2 elements.
        labels (list of str): List of labels corresponding to each point.
        colors (list): List of colors corresponding to each point.
        file_path (str): The path (including filename) to save the plot image.
        label_offset (tuple): A tuple (x_offset, y_offset) for label positioning in points.
                              Adjust this to fine-tune label placement.
    """

    # latex stuff for the paper
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'

    # Ensure the input lists have the same length.
    if not (len(embeddings) == len(labels) == len(colors)):
        raise ValueError("The lengths of embeddings, labels, and colors must be the same.")

    plt.figure(figsize=(6, 6))
    
    # Plot each point and annotate it with a label using an offset.
    for point, label, color in zip(embeddings, labels, colors):
        plt.scatter(point[0], point[1], color=color, s=75)
        plt.annotate(
            label, 
            xy=(point[0], point[1]), 
            xytext=label_offset, 
            textcoords='offset points',
            fontsize=8,
            ha='center'
        )
    
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    
    # Create custom legend handles that map each color to its label.
    plt.legend(handles=legend_handles, fontsize=12, loc='center')

    
    plt.tight_layout()
    
    # Save the plot to the specified file path.
    plt.savefig(file_path, dpi=400)
    print(f"Plot saved to {file_path}")
    plt.close()

def num_to_cardinality(num):
    if num == 1:
        return "1st"
    elif num == 2:
        return "2nd"
    elif num == 3:
        return "3rd"
    else:
        return f"{num}th"

def get_labels(labels):
    new_labels = []

    for l in labels:
        if "atm" in l:
            order = int(l.split(' ')[-1])
            new_labels.append(
                f"{num_to_cardinality(order)} order" if order in [1, 5, 10] else " "
            )
        else:
            new_labels.append(" ")

    return new_labels

from matplotlib.lines import Line2D

def get_legend_handles():

    colors = ["#000000", "#ffbe0b", "#3a86ff", "#fb5607", "#8338ec", "#ff006e"]
    labels = ["Pre-trained", "Task Arithmetic", "TIES-merging", "Model Breadcrumbs", "DARE", "ATM"]

    legend_handles = [
        Line2D(
            [0], [0], marker='o', color='w', label=label, markerfacecolor=color,
            markersize=12
        ) for color, label in zip(colors, labels)
    ]

    return legend_handles
    

def labels_to_colors(labels):
    """
    Convert a list of labels to a list of colors.

    Parameters:
        labels (list of str): List of labels.

    Returns:
        list: List of colors corresponding to the input labels.
    """
    # Define a color for each label.
    label_to_color = {
        "zeroshot": "#000000",
        "ta": "#ffbe0b",
        "ties": "#3a86ff",
        "breadcrumbs": "#fb5607",
        "dare": "#8338ec",
        "atm order 1": "#ff006e",
        "atm order 2": "#ff006e",
        "atm order 3": "#ff006e",
        "atm order 4": "#ff006e",
        "atm order 5": "#ff006e",
        "atm order 6": "#ff006e",
        "atm order 7": "#ff006e",
        "atm order 8": "#ff006e",
        "atm order 9": "#ff006e",
        "atm order 10": "#ff006e"
    }
    
    # Map each label to its corresponding color.
    colors = [label_to_color[label] for label in labels]
    return colors

from rich import print
from rich.pretty import pprint
import numpy as np

def main():
    atm_trajectory_embeddings: dict = np.load(
        file="evaluations/pca_trajectory/pca_trajectory_dict.npy",
        allow_pickle=True
    ).item()

    pprint(atm_trajectory_embeddings, expand_all=True)

    plot_2d_embeddings(
        embeddings=list(atm_trajectory_embeddings.values()),
        labels=get_labels(list(atm_trajectory_embeddings.keys())),
        colors=labels_to_colors(list(atm_trajectory_embeddings.keys())),
        file_path="plots/pca_trajectory/pca_trajectory.png",
        label_offset=(-24, -2.5),
        legend_handles=get_legend_handles()
    )


if __name__ == '__main__':
    main()