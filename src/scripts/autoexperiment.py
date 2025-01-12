# This scripts runs an entire experiment, it goes from order 1 all the way to order "desired_orders".
from rich import print
from rich.pretty import pprint

import yaml
import subprocess

import argparse

from src.tvp.data.datasets.constants import DATASETS_PAPER_ATM
from src.tvp.data.datasets.constants import DATASETS_PAPER_TSV_8
from src.tvp.data.datasets.constants import DATASETS_PAPER_TSV_14
from src.tvp.data.datasets.constants import DATASETS_PAPER_TSV_20
from src.tvp.data.datasets.constants import DATASETS_PAPER_TSV_20_MINUS_PAPER_ATM
from src.tvp.data.datasets.constants import DATASET_TO_STYLED


DATA_YAML_FILE = "conf/nn/data/default.yaml"


# NOTE used a method not a dict in order to also include some input validation!
def _get_optim_class(optim_name: str):
    if optim_name.lower() == "adam":
        return "torch.optim.Adam"
    elif optim_name.lower() == "adamw":
        return "torch.optim.AdamW"
    elif optim_name.lower() == "sgd":
        return "torch.optim.SGD"
    else:
        raise ValueError(f"Invalid optimizer name: {optim_name}")


# NOTE used a method not a dict in order to also include some input validation!
def _handle_task_group_name(task_group_name: str):
    if task_group_name.lower() == "paper-atm":
        return DATASETS_PAPER_ATM
    elif task_group_name.lower() == "paper-tsv-8":
        return DATASETS_PAPER_TSV_8
    elif task_group_name.lower() == "paper-tsv-14":
        return DATASETS_PAPER_TSV_14
    elif task_group_name.lower() == "paper-tsv-20":
        return DATASETS_PAPER_TSV_20
    elif task_group_name.lower() == "paper-tsv-20-minus-paper-atm":
        return DATASETS_PAPER_TSV_20_MINUS_PAPER_ATM
    else:
        raise ValueError(f"Invalid task group name: {task_group_name}")


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "t", "yes", "y", "1"}:
        return True
    elif value.lower() in {"false", "f", "no", "n", "0"}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _validate_args(args: dict):

    if not args["called_from_bash"]:
        print(f"[bold red]This script should be called from bash[/bold red]")
        exit(-1)

    if args["ft_regime"].lower() not in ["atm", "ta"]:
        raise ValueError(f"Invalid finetuning regime: {args['ft_regime']}")

    if args["ft_task_group_name"] is None and args["ft_task_names"] is None:
        raise ValueError("Either --ft-task-group-name or --ft-task-names should be provided")

    if args["tvs_to_apply_group_name"] is None and args["tvs_to_apply_names"] is None:
        raise ValueError("Either --tvs-to-apply-group-name or --tvs-to-apply-names should be provided")

    if args["eval_dataset_group_name"] is None and args["eval_dataset_names"] is None:
        raise ValueError("Either --eval-dataset-group-name or --eval-dataset-names should be provided")
    
    if args["ft_task_group_name"] is not None and args["ft_task_names"] is not None:
        raise ValueError("Either --ft-task-group-name or --ft-task-names should be provided, not both")

    if args["tvs_to_apply_group_name"] is not None and args["tvs_to_apply_names"] is not None:
        raise ValueError("Either --tvs-to-apply-group-name or --tvs-to-apply-names should be provided, not both")

    if args["eval_dataset_group_name"] is not None and args["eval_dataset_names"] is not None:
        raise ValueError("Either --eval-dataset-group-name or --eval-dataset-names should be provided, not both")


    if args["ft_task_group_name"] is not None:
        args["ft_tasks"] = _handle_task_group_name(args["ft_task_group_name"])
    else:
        args["ft_tasks"] = args["ft_task_names"]

    if args["tvs_to_apply_group_name"] is not None:
        args["tvs_to_apply"] = [
            DATASET_TO_STYLED[t] for t in _handle_task_group_name(args["tvs_to_apply_group_name"])
        ]
    else:
        args["tvs_to_apply"] = args["tvs_to_apply_names"]

    if args["eval_dataset_group_name"] is not None:
        args["eval_datasets"] = [
            DATASET_TO_STYLED[t] for t in _handle_task_group_name(args["eval_dataset_group_name"])
        ]
    else :
        args["eval_datasets"] = args["eval_dataset_names"]

    args["optim_class"] = _get_optim_class(args["optim_name"])

    args["lr_scheduler_name"] = "_CosineAnnealingLRScheduler" if args["use_lr_scheduler"] else ""

    return args


def _parse_args():
    parser = argparse.ArgumentParser(description="Run an experiment")
    parser.add_argument("--ft-task-group-name", type=str, help="Which task group to consider. Options: ['paper-atm', 'paper-tsv-8', 'paper-tsv-14', 'paper-tsv-20']")
    parser.add_argument("--ft-task-names", type=str, nargs='+', help="Tasks to consider.")
    parser.add_argument("--ft-regime", type=str, required=True, help="Finetuning regime. Options: ['atm', 'ta']")
    parser.add_argument("--tvs-to-apply-group-name", type=str, help="Task vectors group to apply. Options: ['paper-atm', 'paper-tsv-8', 'paper-tsv-14', 'paper-tsv-20']")
    parser.add_argument("--tvs-to-apply-names", type=str, nargs='+', help="Task vectors group to apply.")
    parser.add_argument("--eval-dataset-group-name", type=str, help="Evaluation datasets group to evaluate on. Options: ['paper-atm', 'paper-tsv-8', 'paper-tsv-14', 'paper-tsv-20']")
    parser.add_argument("--eval-dataset-names", type=str, nargs='+', help="Evaluation datasets to evaluate on.")
    parser.add_argument("--optim-name", type=str, required=True, help="Optimizer to use. Options: ['adam', 'sgd']")
    parser.add_argument("--weight-decay", type=float, required=True, help="Weight decay to use")
    parser.add_argument("--use-lr-scheduler", type=str_to_bool, required=True, help="Flag to indicate if learning rate scheduler should be used (true/false)")
    parser.add_argument("--perform-ft", type=str_to_bool, required=True, help="Flag to indicate if finetuning should be performed (true/false)")
    parser.add_argument("--perform-eval", type=str_to_bool, required=True, help="Flag to indicate if evaluation should be performed (true/false)")
    parser.add_argument("--eval-skip-if-exists", type=str_to_bool, required=True, help="Flag to indicate if evaluation should be skipped if the evaluation results already exist (true/false)")
    parser.add_argument("--called-from-bash", action="store_true", help="Flag to indicate if script was called from bash")
    parser.add_argument("--upload-to-wandb", type=str_to_bool, required=True, help="Flag to indicate if merged model should be uploaded to wandb (true/false)")
    parser.add_argument("--evaluation-export-dir", type=str, required=True, help="Directory to export evaluation results")
    parser.add_argument("--timestamp", type=str, help="Timestamp used to identify the experiment")
    
    args = parser.parse_args()

    args = vars(args)

    return _validate_args(args)


def main():
    args = _parse_args()

    pprint(args, expand_all=True)

    timestamp = "null" if args["timestamp"] is None else args["timestamp"]

    if args["perform_ft"]:

        ft_tasks = args["ft_tasks"]
        print(f"\n\nFinetuning tasks: {ft_tasks}\n\n")

        for dataset_id, task_to_finetune in enumerate(ft_tasks):

            print(
                f"\n\n{task_to_finetune} ({dataset_id + 1}/{len(ft_tasks)})\n\n"
            )
            
            subprocess.run(
                [
                    "python", 
                    "src/scripts/finetune.py",
                    f"+task_to_finetune={task_to_finetune}",
                    f"+ft_regime={args['ft_regime']}",
                    f"+optimizer_name={args['optim_name']}",
                    f"nn.module.optimizer._target_={args['optim_class']}",
                    f"+nn.module.optimizer.weight_decay={args['weight_decay']}",
                    f"+use_lr_scheduler={args['use_lr_scheduler']}",
                    f"+lr_scheduler_name={args['lr_scheduler_name']}",
                    f"+timestamp={timestamp}",
                ], 
                check=True
            )
    
    if args["perform_eval"]:

        print(f"\n\nEvaluation datasets: {args['eval_datasets']}\n\n")

        subprocess.run(
            [
                "python", 
                "src/scripts/evaluate.py",
                f"+ft_regime={args['ft_regime']}",
                f"task_vectors.to_apply={args['tvs_to_apply']}",
                f"eval_datasets={args['eval_datasets']}",
                f"+optimizer_name={args['optim_name']}",
                f"nn.module.optimizer._target_={args['optim_class']}",
                f"+nn.module.optimizer.weight_decay={args['weight_decay']}",
                f"+use_lr_scheduler={args['use_lr_scheduler']}",
                f"+lr_scheduler_name={args['lr_scheduler_name']}",
                f"+upload_merged_to_wandb={args['upload_to_wandb']}",
                f"+evaluation_export_dir={args['evaluation_export_dir']}",
                f"+eval_skip_if_exists={args['eval_skip_if_exists']}",
                f"+timestamp={timestamp}",
            ], 
            check=True
        )
    

if __name__ == "__main__":
    main()