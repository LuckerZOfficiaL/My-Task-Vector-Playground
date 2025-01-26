from rich import print
from rich.pretty import pprint

import argparse

import subprocess


def str_to_float_or_int(value: str):
    if value is None or value.lower() == "none":  # Handle None or "None"
        return None
    try:
        if '.' in value or 'e' in value.lower():  # Check for float-like representation
            return float(value)
        return int(value)  # If not, it's an integer
    except ValueError:
        raise ValueError(f"Unable to parse '{value}' as int, float, or None.")


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "t", "yes", "y", "1"}:
        return True
    elif value.lower() in {"false", "f", "no", "n", "0"}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")
    

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--optim-name", type=str, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--lr-scheduler-name", type=str, required=True)
    parser.add_argument("--cosine-annealing-warmup-step-number-or-ratio", type=str_to_float_or_int, required=True)
    parser.add_argument("--ft-regime", type=str, required=True)
    parser.add_argument("--tvs-to-apply-group-name", type=str)
    parser.add_argument("--tvs-to-apply-names", type=str, nargs="+")
    parser.add_argument("--eval-dataset-names", type=str, nargs="+")
    parser.add_argument("--eval-dataset-group-name", type=str)
    parser.add_argument("--eval-orthogonalization-method", type=str, required=True)
    parser.add_argument("--eval-ft-progress-merging", type=str_to_bool, required=True)
    parser.add_argument("--eval-skip-if-exists", type=bool, required=True)

    args = parser.parse_args()
    args = vars(args)

    return args


def main():

    args = parse_args()

    pprint(args, expand_all=True)

    if args["eval_ft_progress_merging"]:
        eval_dir = "./evaluations/ft_progress_merging"
    else:
        eval_dir = "./evaluations/merged_progress_merging"

    subprocess.run(
        [
            # bash src/scripts/autoexperiment.sh --perform-ft false --optim-name adamw --weight-decay 0.1 --lr-scheduler-name cosine_annealing --cosine-annealing-warmup-step-number-or-ratio 200 --ft-regime ta --perform-eval true --tvs-to-apply-group-name paper-tsv-14  --eval-dataset-group-name paper-tsv-14  --eval-skip-if-exists false --upload-merged-to-wandb false --evaluation-export-dir ./evaluations/merged --eval-orthogonalization-method none --eval-use-wita true --wita-top-k-weakest 14 --wita-top-k-strongest 0 --wita-num-iters 1  --timestamp
            f"bash",
            f"src/scripts/autoexperiment.sh",
            f"--perform-ft false",
            f"--optim-name {args['optim_name']}",
            f"--weight-decay {args['weight_decay']}",
            f"--lr-scheduler-name {args['lr_scheduler_name']}",
            f"--cosine-annealing-warmup-step-number-or-ratio {args['cosine_annealing_warmup_step_number_or_ratio']}",
            f"--ft-regime {args['ft_regime']}",
            f"--perform-eval true",
            f"--tvs-to-apply-group-name {args['tvs_to_apply_group_name']}",
            f"--eval-dataset-group-name {args['eval_dataset_group_name']}",
            f"--eval-skip-if-exists {args['eval_skip_if_exists']}",
            f"--eval-orthogonalization-method {args['eval_orthogonalization_method']}",
            f"--eval-use-wita false",
            f"--eval-use-merged-ratios true",
            f"--eval-ft-progress-merging {args['eval_ft_progress_merging']}",
            f"--upload-merged-to-wandb false",
            f"--evaluation-export-dir {eval_dir}",
            f"--timestamp"
        ]
    )


if __name__ == "__main__":
    main()