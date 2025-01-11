from rich import print
from rich.pretty import pprint

import argparse
from src.tvp.data.datasets.constants import DATASETS_PAPER_TSV_20
from src.tvp.data.datasets.constants import DATASET_TO_STYLED

from itertools import combinations

from typing import List

import random

import subprocess


SEED = 421337


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "t", "yes", "y", "1"}:
        return True
    elif value.lower() in {"false", "f", "no", "n", "0"}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")
    

def _handle_task_group_name(task_group_name: str):

    if task_group_name.lower() == "paper-tsv-20":
        return DATASETS_PAPER_TSV_20
    else:
        raise ValueError(f"Invalid task group name: {task_group_name}")


def _validate_args(args: dict):

    args["tvs_to_apply"] = _handle_task_group_name(args["tvs_to_apply_group_name"])
    args["tvs_to_apply"] = [DATASET_TO_STYLED[dataset] for dataset in args["tvs_to_apply"]]

    if args["subset_size"] > len(args["tvs_to_apply"]):
        raise ValueError(f"The subset size ({args['subset_size']}) is greater than the number of datasets available ({len(args['tvs_to_apply'])}).")

    if args["optim"].lower() not in ["adam", "sgd"]:
        raise ValueError(f"Invalid optimizer name: {args['optim']}")
    args["optim"] = args["optim"].lower()

    if args["ft_regime"].lower() not in ["atm", "ta"]:
        raise ValueError(f"Invalid finetuning regime: {args['ft_regime']}")
    args["ft_regime"] = args["ft_regime"].lower()

    
    return args


def _gen_all_possible_subsets(tvs_to_apply: list, subset_size: int):
    
    datasets = sorted(tvs_to_apply)
    datasets_all_possible_subsets = list(combinations(datasets, subset_size))
    datasets_all_possible_subsets = [list(subset) for subset in datasets_all_possible_subsets]

    return datasets_all_possible_subsets


def _sample_subset_list(all_possible_subsets: List[List[str]], num_subsets: int):
    random.seed(SEED)
    
    dataset_random_subsets = random.sample(
        population=all_possible_subsets, 
        k=num_subsets
    )

    return dataset_random_subsets


def parse_args():
    parser = argparse.ArgumentParser(description="AutoSubsets script")

    parser.add_argument("--tvs-to-apply-group-name", type=str, required=True, help="Which task group to consider. Options: ['paper-atm', 'paper-tsv-8', 'paper-tsv-14', 'paper-tsv-20']")
    parser.add_argument("--num-subsets", type=int, required=True, help="Number of subsets to create")
    parser.add_argument("--subset-size", type=int, required=True, help="Size of each subset")
    parser.add_argument("--optim", type=str, required=True, help="Optimization algorithm to use")
    parser.add_argument("--ft-regime", type=str, required=True, help="Finetuning regime. Options: ['atm', 'ta']")
    parser.add_argument("--start-idx", type=int, required=True, help="Start index of the subset list")
    parser.add_argument("--end-idx", type=int, required=True, help="End index of the subset list")
    parser.add_argument("--eval-skip-if-exists", type=str_to_bool, required=True, help="Skip evaluation if the evaluation file already exists")

    args = vars(parser.parse_args())

    args = _validate_args(args)

    return args

def main():
    args = parse_args()

    # pprint(args, expand_all=True)

    all_possible_subsets = _gen_all_possible_subsets(
        tvs_to_apply=args["tvs_to_apply"], subset_size=args["subset_size"]
    )

    random_subsets = _sample_subset_list(
        all_possible_subsets=all_possible_subsets, num_subsets=args["num_subsets"]
    )

    random_subsets = random_subsets[args["start_idx"]:args["end_idx"]]

    for subset_idx, subset in enumerate(random_subsets):

        print(f"\n\nSubset {subset_idx+1}/{len(random_subsets)}\n\n")

        subset_str = " ".join(subset)

        subprocess.run(
            [
                f"bash",
                f"src/scripts/autoexperiment.sh", 
                f"--ft-task-group-name paper-tsv-20", 
                f"--tvs-to-apply-names {subset_str}",
                f"--eval-dataset-names {subset_str}",
                f"--optim-name {args['optim']}", 
                f"--ft-regime {args['ft_regime']}", 
                f"--perform-ft false",
                f"--perform-eval true",
                f"--upload-to-wandb false",
                f"--evaluation-export-dir evaluations/merged_subsets/{args['tvs_to_apply_group_name']}/subset_size_{str(args['subset_size']).zfill(2)}",
                f"--eval-skip-if-exists {args['eval_skip_if_exists']}",
            ]
        )




if __name__ == "__main__":
    main()