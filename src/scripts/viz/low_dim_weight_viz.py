from rich import print
from rich.pretty import pprint

from src.tvp.data.datasets.constants import DATASETS_PAPER_TA
from src.tvp.data.datasets.constants import DATASET_TO_STYLED

import hydra
from nn_core.common import PROJECT_ROOT
from omegaconf import DictConfig
from nn_core.callbacks import NNTemplateCore
from nn_core.model_logging import NNLogger

from typing import Union, List, Dict
from tvp.modules.encoder import ClassificationHead, ImageEncoder

from tvp.utils.io_utils import load_model_from_artifact
from torch.nn.utils import parameters_to_vector
import torch

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA


def init_logger(cfg: DictConfig) -> NNLogger:
    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None)
    )
    
    logger: NNLogger = NNLogger(
        logging_cfg=cfg.train.logging, 
        cfg=cfg, 
        resume_id=template_core.resume_id
    )

    return logger


def get_zeroshot_model(logger: NNLogger) -> Union[ClassificationHead, ImageEncoder]:

    zeroshot_identifier = f"ViT-B-16_pt"
    zeroshot_model = load_model_from_artifact(
        artifact_path=f"{zeroshot_identifier}:latest", run=logger.experiment
    )

    return zeroshot_model


def compose_artifact_name(dataset: str, ratio: float) -> str:
    ft_ta_identifier = (
        f"ViT-B-16"
        f"____DATASET_NAME_PLACEHOLDER___"
        f"_0"
        f"_ta"
        f"_adamw"
        f"_wd_0.1"
        f"_lr_scheduler_cosine_annealing"
        f"_warmup_steps_200"
        f"____STEP_RATIO_PLACEHOLDER___"
        f":latest"
    )

    return ft_ta_identifier.replace(
        "___DATASET_NAME_PLACEHOLDER___", dataset
    ).replace(
        "___STEP_RATIO_PLACEHOLDER___", f"step_{ratio}"
    )


def get_zeroshot_model(logger: NNLogger) -> Union[ClassificationHead, ImageEncoder]:

    zeroshot_identifier = f"ViT-B-16_pt"
    zeroshot_model = load_model_from_artifact(
        artifact_path=f"{zeroshot_identifier}:latest", run=logger.experiment
    )

    return zeroshot_model


def plot_interactive_pca(
    checkpoints_reduced: np.ndarray, 
    datasets: List[str],
    ratios: List[float],
    num_components: int,
    pca_output_path: str, 
):
    """
    Creates an interactive PCA plot (2D or 3D) and saves it to disk.

    Args:
        checkpoints_reduced: PCA-reduced checkpoints.
        pca_output_path: Path to save the output HTML file.
        num_components: Number of PCA components (2 for 2D, 3 for 3D).
    """
    if num_components not in [2, 3]:
        raise ValueError("num_components must be either 2 or 3")


    

    # Initialize the figure
    fig = go.Figure()

    for dataset_idx, dataset in enumerate(datasets):

        labels = [f"{dataset}, {int(ratio*100)}%" for ratio in ratios]

        if num_components == 3:
            # Add 3D scatter plot
            fig.add_trace(go.Scatter3d(
                x=checkpoints_reduced[
                    len(ratios)*dataset_idx : len(ratios)*dataset_idx+len(ratios), 0
                ],
                y=checkpoints_reduced[
                    len(ratios)*dataset_idx : len(ratios)*dataset_idx+len(ratios), 1
                ],
                z=checkpoints_reduced[
                    len(ratios)*dataset_idx : len(ratios)*dataset_idx+len(ratios), 2
                ],
                mode='lines+markers',
                name=dataset,
                line=dict(width=2),
                marker=dict(size=6),
                text=labels,  # Add point labels for hover
                hoverinfo="text",  # Show only the labels on hover
            ))
        else:
            # Add 2D scatter plot
            fig.add_trace(go.Scatter(
                x=checkpoints_reduced[
                    len(ratios)*dataset_idx : len(ratios)*dataset_idx+len(ratios), 0
                ],
                y=checkpoints_reduced[
                    len(ratios)*dataset_idx : len(ratios)*dataset_idx+len(ratios), 1
                ],
                mode='lines+markers',
                name=dataset,
                line=dict(width=2),
                marker=dict(size=6),
                text=labels,  # Add point labels for hover
                hoverinfo="text",  # Show only the labels on hover
            ))

    zs_reduced = checkpoints_reduced[-1, :]
    if num_components == 3:
        fig.add_trace(go.Scatter3d(
            x=[zs_reduced[0]],
            y=[zs_reduced[1]],
            z=[zs_reduced[2]],
            mode='markers',
            name="zeroshot",
            marker=dict(size=8, color='red', symbol='diamond'),
            text=["zeroshot"],
            hoverinfo="text",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=[zs_reduced[0]],
            y=[zs_reduced[1]],
            mode='markers',
            name="zeroshot",
            marker=dict(size=8, color='red', symbol='diamond'),
            text=["zeroshot"],
            hoverinfo="text",
        ))

    # Layout configuration
    if num_components == 3:
        fig.update_layout(
            title="Interactive 3D PCA Plot",
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3",
            ),
            legend_title="Datasets",
        )
    else:
        fig.update_layout(
            title="Interactive 2D PCA Plot",
            xaxis_title="PC1",
            yaxis_title="PC2",
            legend_title="Datasets",
        )

    # Save the plot
    fig.write_html(pca_output_path)


def perform_pca(
    data_dict: dict, 
    num_components: int,
    pca_export_path: Union[str, None]
):
    checkpoints = np.array(list(data_dict.values()))
    print(f"checkpoints.shape: {checkpoints.shape}")

    pca = PCA(n_components=num_components)
    checkpoints_reduced = pca.fit_transform(checkpoints)

    # zs_reduced = checkpoints_reduced[-1, :]
    # print(f"zs_reduced.shape: {zs_reduced.shape}")

    checkpoints_reduced_dict = {}
    for idx, key in enumerate(data_dict.keys()):
        checkpoints_reduced_dict[key] = checkpoints_reduced[idx, :]
    
    pca_stats = {
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "explained_variance": pca.explained_variance_,
        "singular_values": pca.singular_values_,
        "mean": pca.mean_,
        "components": pca.components_,
    }

    if pca_export_path is not None:
        np.save(pca_export_path.replace(".npy", "_ndarray.npy"), checkpoints_reduced_dict)
        np.save(pca_export_path.replace(".npy", "_dict.npy"), checkpoints_reduced)
        np.save(pca_export_path.replace(".npy", "_stats.npy"), pca_stats)

    return checkpoints_reduced, checkpoints_reduced_dict, pca_stats


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="task_vectors.yaml")
def main(cfg: DictConfig):
    datasets = DATASETS_PAPER_TA

    RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    logger: NNLogger = init_logger(cfg)

    plot_data: Dict[str, np.ndarray] = {}

    for dataset_idx, dataset in enumerate(datasets):

        for ratio_idx, ratio in enumerate(RATIOS):

            print(f"Dataset: {dataset} ({dataset_idx + 1}/{len(datasets)}), Ratio: {ratio}")

            artifact_name = compose_artifact_name(
                dataset=DATASET_TO_STYLED[dataset], ratio=ratio
            )
            model = load_model_from_artifact(
                run=logger.experiment, artifact_path=artifact_name
            )
            model_vec = parameters_to_vector(model.parameters())

            plot_data[f"{dataset}, {int(ratio*100)}%"] = model_vec.detach().cpu().numpy()

    plot_data["zs"] = parameters_to_vector(
        get_zeroshot_model(logger).parameters()
    ).detach().cpu().numpy()

    PERFORM_PCA: bool = True
    NUM_COMPONENTS: int = 2
    PCA_PATH = f"./plots/pca_embedding/pca_embedding_{NUM_COMPONENTS}D_{'_'.join([DATASET_TO_STYLED[t] for t in datasets])}.npy"

    if PERFORM_PCA:
        print(f"\n\n")
        print(f"Performing PCA... this may take a while (approx. 35 mins for 8 tasks and 10 ratios, 2 mins for 2 tasks and 4 ratios)")
        print(f"\n\n")

        checkpoints_reduced, checkpoints_reduced_dict, pca_stats = perform_pca(
            data_dict=plot_data, 
            num_components=NUM_COMPONENTS,
            pca_export_path=PCA_PATH
        )
    else:
        print(f"\n\n")
        print(f"Loading PCA data from {PCA_PATH}")
        print(f"\n\n")
        checkpoints_reduced = np.load(PCA_PATH.replace(".npy", "_dict.npy"))
        checkpoints_reduced_dict = np.load(PCA_PATH.replace(".npy", "_ndarray.npy"), allow_pickle=True).item()
        pca_stats = np.load(PCA_PATH.replace(".npy", "_stats.npy"), allow_pickle=True).item()

    print(f"\n\n")
    print(f"PCA stats:")
    pprint(pca_stats, expand_all=True)

    print(f"\n\n")
    print(f"checkpoints_reduced:")
    pprint(checkpoints_reduced, expand_all=True)
    print(f"checkpoints_reduced.shape: {checkpoints_reduced.shape}")

    print(f"\n\n")
    print(f"checkpoints_reduced_dict:")
    pprint(checkpoints_reduced_dict, expand_all=True)

    pca_plot_output_path = f"./plots/low_dim_weight_viz/low_dim_weight_viz_{NUM_COMPONENTS}D_{'-'.join([DATASET_TO_STYLED[t] for t in datasets])}.html"
    plot_interactive_pca(
        checkpoints_reduced=checkpoints_reduced, 
        datasets=datasets,
        ratios=RATIOS,
        num_components=NUM_COMPONENTS,
        pca_output_path=pca_plot_output_path
    )


            
        


if __name__ == '__main__':
    main()