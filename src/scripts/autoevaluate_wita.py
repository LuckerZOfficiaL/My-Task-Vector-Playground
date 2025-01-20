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
    

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--optim-name", type=str, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--lr-scheduler-name", type=str, required=True)
    parser.add_argument("--cosine-annealing-warmup-step-number-or-ratio", type=str_to_float_or_int, required=True)
    parser.add_argument("--ft-regime", type=str, required=True)
    parser.add_argument("--tvs-to-apply-group-name", type=str, required=True)
    parser.add_argument("--eval-dataset-group-name", type=str, required=True)
    parser.add_argument("--eval-orthogonalization-method", required=True, type=str)
    parser.add_argument("--eval-conflict-res-method", required=True, type=str)
    parser.add_argument("--eval-skip-if-exists", type=bool, required=True)

    # parser.add_argument("--combo-start-idx", type=int, required=True)
    # parser.add_argument("--combo-end-idx", type=int, required=True)

    args = parser.parse_args()
    args = vars(args)

    return args


def main():

    args = parse_args()

    pprint(args, expand_all=True)

    H = [5, 10, 15]
    TOP_K_WEAKEST = [5, 7, 10]
    TOP_K_STRONGEST = [0, 1, 3]

    configs = []

    # for h in H:
    #     for w in TOP_K_WEAKEST:
    #         for s in TOP_K_STRONGEST:
    #             configs.append(
    #                 {
    #                     "H": h,
    #                     "top_k_weakest": w,
    #                     "top_k_strongest": s
    #                 }
    #             )

    '''
    
    Config h=10, w=5, s=3 NOT tested...
    Config h=10, w=7, s=1 NOT tested...
    Config h=10, w=7, s=0 NOT tested...
    Config h=10, w=10, s=3 NOT tested...
    Config h=15, w=7, s=3 NOT tested...
    
    '''
    configs = [
        {
            "H": 10,
            "top_k_weakest": 10,
            "top_k_strongest": 0
        }
    ]

    for i, config in enumerate(configs):
        # if i >= args["combo_start_idx"] and i <= args["combo_end_idx"]:
        #     print("\n\n\n\n")
        #     print(f"STARTING Config {i}/{args['combo_end_idx'] - args['combo_start_idx']}:")
        #     print(f"STARTING Config {i + 1}/{args['combo_end_idx'] - args['combo_start_idx'] + 1}:")
        #     pprint(config)
        #     print("\n\n\n\n")

        print("\n\n\n\n")
        print(f"STARTING Config {i + 1}/{len(configs)}:")
        pprint(config, expand_all=True)
        print("\n\n\n\n")

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
                f"--upload-merged-to-wandb false",
                f"--evaluation-export-dir ./evaluations/merged_wita_grid_search",
                f"--eval-orthogonalization-method {args['eval_orthogonalization_method']}",
                f"--eval-conflict-res-method {args['eval_conflict_res_method']}",
                f"--eval-use-wita true",
                f"--wita-top-k-weakest {config['top_k_weakest']}",
                f"--wita-top-k-strongest {config['top_k_strongest']}",
                f"--wita-num-iters {config['H']}",
                f"--timestamp"
            ]
        )

        # print("\n\n\n\n")
        # print(f"DONE Config {i}/{args['combo_end_idx'] - args['combo_start_idx']}:")
        # print(f"DONE Config {i + 1}/{args['combo_end_idx'] - args['combo_start_idx'] + 1}:")
        # pprint(config)
        # print("\n\n\n\n")
        print("\n\n\n\n")
        print(f"DONE Config {i + 1}/{len(configs)}:")
        pprint(config, expand_all=True)
        print("\n\n\n\n")


if __name__ == "__main__":
    main()