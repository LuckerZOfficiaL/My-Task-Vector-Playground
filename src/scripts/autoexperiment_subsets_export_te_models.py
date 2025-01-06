# This scripts runs an entire experiment, it goes from order 1 all the way to order "DESIRED_ORDERS".
from rich import print

import os
import subprocess

import builtins
import tqdm

from tvp.data.constants import DATASET_NAME_TO_TA_FT_EPOCHS_LOWERCASE
from tvp.data.constants import DATASET_NAME_TO_NUM_TRAIN_BATCHES_LOWERCASE
from tvp.data.constants import DATASET_NAME_TO_NUM_VAL_BATCHES_LOWERCASE
from tvp.data.constants import DATASET_NAME_TO_NUM_TEST_BATCHES_LOWERCASE
from tvp.data.constants import DATASETS_07, DATASETS_20
from tvp.data.constants import DATASET_NAME_TO_STYLED_NAME

from itertools import combinations

import argparse
from rich import print

from tvp.utils.io_utils import export_list_of_strings_to_file

import random


# Define a custom print function that redirects to tqdm.write
def tqdm_print(*args, **kwargs):
    # Join all arguments into a single string and pass to tqdm.write
    tqdm.write(" ".join(map(str, args)), **kwargs)

# Override the built-in print globally
builtins.print = tqdm_print


def _verify_args(args):
    # Convert "None" string to Python None
    args.timestamp = None if args.timestamp == "None" else args.timestamp

    return args

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", help="Timestamp for this run")
    parser.add_argument("--called-from-bash-script", help="Indicates if the script was called from bash")
    parser.add_argument("--subset-length", type=int, required=True, help="Length of the subset")
    parser.add_argument("--num-subsets", type=int, required=True, help="Number of subsets")
    parser.add_argument("--subsets-idx-start", type=int, required=True, help="Index of the first subset")
    parser.add_argument("--subsets-idx-end", type=int, required=True, help="Index of the last subset")
    args = parser.parse_args()

    args = _verify_args(args)

    args = vars(args)

    return args

args = parse_args()

if not args["called_from_bash_script"]:
    print(f"[bold red]This script should be called from a bash script.")
    exit()


print(args)

print("\n\n\n", f"*"*80, "\n\n\n")
print(f"Running autoexperiment_subsets_export_te_models.py with timestamp: {args['timestamp']}")
print("\n\n\n", f"*"*80, "\n\n\n")

TASK_EQUIPPED_MODELS_EXPORT_DIR = "./task_equipped_models"
os.makedirs(TASK_EQUIPPED_MODELS_EXPORT_DIR, exist_ok=True)

SGD = "torch.optim.SGD"
ADAM = "torch.optim.Adam"

EPOCH_DIVISOR = "None"
DESIRED_ORDERS = 1

# ATM
MAX_EPOCHS = 1
# TA
# MAX_EPOCHS = None

SEED = 421337

DATASETS_TO_EXCLUDE = "pcam"

datasets = DATASETS_20
datasets = [dataset for dataset in datasets if DATASETS_TO_EXCLUDE not in dataset]
datasets = [DATASET_NAME_TO_STYLED_NAME[dataset] for dataset in datasets]

if args["subset_length"] > len(datasets):
    print(f"[bold red]The subset length is greater than the number of datasets available.")
    exit()

datasets = sorted(datasets)
datasets_all_possible_subsets = list(combinations(datasets, args["subset_length"]))
datasets_all_possible_subsets = [list(subset) for subset in datasets_all_possible_subsets]

print(f"len(datasets_all_possible_subsets) = {len(datasets_all_possible_subsets)}\n\n")
print(f"args['subset_length'] = {args['subset_length']}\n\n")

random.seed(SEED)
dataset_random_subsets = random.sample(
    population=datasets_all_possible_subsets, 
    k=min(args["num_subsets"], len(datasets_all_possible_subsets))
)
print(f"len(dataset_random_subsets) before considering start and end idx = {len(dataset_random_subsets)}\n\n")

dataset_random_subsets = dataset_random_subsets[args["subsets_idx_start"]:args["subsets_idx_end"]]

print(f"len(dataset_random_subsets) after considering start and end idx = {len(dataset_random_subsets)}\n\n")

task_equipped_model_export_path_list = []

for order in range(1, DESIRED_ORDERS+1):

    for dataset_subset_id, dataset_subset in enumerate(dataset_random_subsets):

        print(f"Dataset subset: {dataset_subset}, {dataset_subset_id + 1}/{len(dataset_random_subsets)}")

        accumulate_grad_batches = 1
        accumulate_grad_batches = "null" if accumulate_grad_batches is None else accumulate_grad_batches
        
        limit_train_batches = None
        limit_train_batches = "null" if limit_train_batches is None else limit_train_batches
        
        max_epochs = "null" if MAX_EPOCHS is None else MAX_EPOCHS
        
        # timestamp = args["timestamp"]
        # timestamp = "null" if timestamp is None else timestamp

        try: 
            result = subprocess.run(
                [
                    "python", 
                    "src/scripts/export_te_model.py",
                    f"order={order}",
                    f"epoch_divisor={EPOCH_DIVISOR}",
                    f"+accumulate_grad_batches={accumulate_grad_batches}",
                    f"+max_epochs={max_epochs}",
                    # f"nn.module.optimizer._target_={SGD if accumulate_grad_batches else ADAM}",
                    f"nn.module.optimizer._target_={SGD if accumulate_grad_batches else SGD}",
                    f"accumulate_grad_batches={accumulate_grad_batches}",
                    f"+limit_train_batches={limit_train_batches}",
                    f"+task_equipped_model_export_dir={TASK_EQUIPPED_MODELS_EXPORT_DIR}",
                    f"task_vectors.to_apply={dataset_subset}",
                    f"eval_datasets={dataset_subset}",
                ], 
                check=True,
                # NOTE this has to be turned off to capture the output when debugging
                capture_output=True,
                text=True,
            )

            # Capture all stdout
            stdout_lines = result.stdout.splitlines()

            # Filter for the specific print statement
            export_line = next(
                (line for line in stdout_lines if "[export_te_model.main] task equipped model exported to:" in line),
                None
            )

            # Extract the path from the message
            if export_line:
                export_path = export_line.split("task equipped model exported to:")[1].strip()
                print(f"Captured export path: {export_path}")
                task_equipped_model_export_path_list.append(export_path)
            else:
                print("Export path not found in the output.")

            print("\n\n")
        
        except subprocess.CalledProcessError as e:
            # Subprocess failed
            print(f"Subprocess failed with return code {e.returncode}")
            print("Captured output:")
            print(e.stdout)  # Print captured stdout from the failed process
            print("Captured errors:")
            print(e.stderr)  # Print captured stderr from the failed process

            exit()

TASK_EQUIPPED_MODEL_PATH_LIST_EXPORT_DIR = "./task_equipped_models_path_list"
os.makedirs(TASK_EQUIPPED_MODEL_PATH_LIST_EXPORT_DIR, exist_ok=True)

filename = (
    f"task_equipped_model_export_path_list_"
    f"max_epochs_{'TA' if MAX_EPOCHS is None else 'ATM' if MAX_EPOCHS == 1 else MAX_EPOCHS}_"
    f"datasets_{len(datasets)}_"
    f"subset_length_{args['subset_length']}"
    f"_random_num_subsets_{args['num_subsets']}"
    f"_random_subsets_idx_start_{args['subsets_idx_start']}"
    f"_random_subsets_idx_end_{args['subsets_idx_end']}"
    ".txt"
)

export_list_of_strings_to_file(
    data=task_equipped_model_export_path_list,
    filename=os.path.join(TASK_EQUIPPED_MODEL_PATH_LIST_EXPORT_DIR, filename),
    export_description="task equipped model export path list"
)
    