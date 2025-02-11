import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_lines(
    data: dict,
    x_axis_label, 
    x_ticks, 
    y_axis_label, 
    export_path,
    legend_names: dict,
    line_colors: dict
):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # latex stuff for the paper
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'

    fig, ax = plt.subplots()
    
    for key, points in data.items():
        x_vals = [pt[0] for pt in points]
        y_vals = [pt[1] for pt in points]
        # Use the provided legend name and color for each key
        label = legend_names.get(key, key)
        color = line_colors.get(key)
        ax.plot(x_vals, y_vals, label=label, color=color)
    
    ax.set_xlabel(x_axis_label, fontsize=16)
    ax.set_ylabel(y_axis_label, fontsize=16)
    
    ax.set_xticks(x_ticks)

    ax.tick_params(axis='both', which='major', labelsize=16)

    ax.grid(True)

    ax.legend(bbox_to_anchor=(0.5, 0.275), loc='center', fontsize=14)

    plt.tight_layout()

    plt.savefig(export_path, dpi=400)
    plt.close(fig)


from rich import pretty
from rich.pretty import pprint
from tvp.utils.io_utils import import_json_from_disk
import os

def main():
    BUDGETS = [2, 4, 6, 8, 10]

    COLORS = ["#ffbe0b", "#3a86ff", "#fb5607", "#8338ec", "#ff006e"]
    METHODS = [
        # "Task Arithmetic", "TIES-merging", "Model Breadcrumbs", "DARE", "ATM"
        "ta", "ties", "bc", "dare", "atm"
    ]

    METHODS_TO_LEGEND_NAMES = {
        "ta": "Task Arithmetic",
        "ties": "TIES-merging",
        "bc": "Model Breadcrumbs",
        "dare": "DARE",
        "atm": "PA-ATM"
    }
    METHODS_TO_COLORS = {
        "ta": "#ffbe0b",
        "ties": "#3a86ff",
        "bc": "#fb5607",
        "dare": "#8338ec",
        "atm": "#ff006e"
    }

    data_dict = {}

    for m in METHODS:

        data_dict[m] = []

        for b in BUDGETS:
            print(f"{m} @ {b}")

            conf_res_name = "none" if m == "ta" or m == "atm" else m
            # train_batches = 0.1 if m == "atm" else 1.0
            train_batches = 1.0
            ord = 1 if m != "atm" else b
            eps_per_ord = b if m != "atm" else 1

            eval_file_path = (
                f"./evaluations/atm-true/"
                f"ViT-B-16_0_atm-true_"
                f"confl_res_{conf_res_name}_"
                f"train_batches_{train_batches}_"
                f"ord_{ord}_"
                f"eps_per_ord_{eps_per_ord}_"
                f"merged.json"
            )

            avg_norm_merged_acc = import_json_from_disk(
                file_path=eval_file_path
            )["results"]["average_of_tasks"]

            data_dict[m].append((b, avg_norm_merged_acc))

    pprint(data_dict, expand_all=True)

    export_path = f"plots/budget_vs_avg_norm_merged_acc/budget_vs_avg_norm_merged_acc.png"
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    plot_lines(
        data=data_dict,
        x_axis_label="Budget",
        x_ticks=BUDGETS,
        y_axis_label="Average Merged Accuracy",
        legend_names=METHODS_TO_LEGEND_NAMES,
        line_colors=METHODS_TO_COLORS,
        export_path=export_path
    )





if __name__ == "__main__":
    main()