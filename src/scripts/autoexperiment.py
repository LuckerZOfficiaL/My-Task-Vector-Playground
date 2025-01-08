# This scripts runs an entire experiment, it goes from order 1 all the way to order "desired_orders".
from rich import print
from rich.pretty import pprint

import yaml
import subprocess

import argparse

from src.tvp.data.datasets.constants import DATASETS_07
from src.tvp.data.datasets.constants import DATASET_TO_STYLED


DATA_YAML_FILE = "conf/nn/data/default.yaml"


def _validate_args(args: dict):

    if not args["called_from_bash"]:
        print(f"[bold red]This script should be called from bash[/bold red]")
        exit(-1)

    if args["ft_regime"].lower() not in ["atm", "ta"]:
        raise ValueError(f"Invalid finetuning regime: {args['ft_regime']}")

    if args["num_tasks"] == 7:
        args["tasks"] = DATASETS_07
    elif args["num_tasks"] == 20:
        raise NotImplementedError("20 tasks not implemented yet")
    else:
        raise ValueError(f"Invalid number of tasks: {args['num_tasks']}")

    if args["optim_name"].lower() == "adam":
        args["optim_class"] = "torch.optim.Adam"
    elif args["optim_name"].lower() == "sgd":
        args["optim_class"] = "torch.optim.SGD"
    else:
        raise ValueError(f"Invalid optimizer name: {args['optim_name']}")

    return args


def _parse_args():
    parser = argparse.ArgumentParser(description="Run an experiment")
    parser.add_argument("--num-tasks", type=int, required=True, help="Number of tasks to consider. Options: [7, 20]")
    parser.add_argument("--ft-regime", type=str, required=True, help="Finetuning regime. Options: ['atm', 'ta']")
    parser.add_argument("--optim-name", type=str, required=True, help="Optimizer to use. Options: ['adam', 'sgd']")
    # TODO add possibility of turning ft and eval on/off on their own
    
    parser.add_argument("--called-from-bash", action="store_true", help="Flag to indicate if script was called from bash")
    parser.add_argument("--timestamp", type=str, help="Timestamp used to identify the experiment")
    
    args = parser.parse_args()

    args = vars(args)

    return _validate_args(args)


def main():
    args = _parse_args()

    pprint(args, expand_all=True)

    datasets = args["tasks"]
    tvs_to_apply = [DATASET_TO_STYLED[t] for t in args["tasks"]]
    evaluate_on_datasets = [DATASET_TO_STYLED[t] for t in args["tasks"]]

    for dataset_id, task_to_finetune in enumerate(datasets):

        print(
            f"\n\n{task_to_finetune} ({dataset_id + 1}/{len(datasets)})\n\n"
        )
        
        subprocess.run(
            [
                "python", 
                "src/scripts/finetune.py",
                f"+task_to_finetune={task_to_finetune}",
                f"+ft_regime={args['ft_regime']}",
                f"+optimizer_name={args['optim_name']}",
                f"nn.module.optimizer._target_={args['optim_class']}",
                f"+timestamp={args['timestamp']}",
            ], 
            check=True
        )

    subprocess.run(
        [
            "python", 
            "src/scripts/evaluate.py",
            f"+ft_regime={args['ft_regime']}",
            f"task_vectors.to_apply={tvs_to_apply}",
            f"eval_datasets={evaluate_on_datasets}",
            f"+optimizer_name={args['optim_name']}",
            f"nn.module.optimizer._target_={args['optim_class']}",
            f"+timestamp={args['timestamp']}",
        ], 
        check=True
    )
    

if __name__ == "__main__":
    main()