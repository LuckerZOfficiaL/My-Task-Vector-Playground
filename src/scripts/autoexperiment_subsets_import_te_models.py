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

from tvp.utils.io_utils import load_list_of_strings_from_file


# Define a custom print function that redirects to tqdm.write
def tqdm_print(*args, **kwargs):
    # Join all arguments into a single string and pass to tqdm.write
    tqdm.write(" ".join(map(str, args)), **kwargs)

# Override the built-in print globally
# builtins.print = tqdm_print


def _verify_args(args):
    # Convert "None" string to Python None
    args.timestamp = None if args.timestamp == "None" else args.timestamp

    return args

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", help="Timestamp for this run")
    parser.add_argument("--called-from-bash-script", help="Indicates if the script was called from bash")
    parser.add_argument("--te-model-path-list-file", type=str, required=True, help="File where the list of task-equipped models is stored")
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
print(f"Running autoexperiment_subsets_import_te_models.py with timestamp: {args['timestamp']}")
print("\n\n\n", f"*"*80, "\n\n\n")

task_equipped_models_file_path_list = load_list_of_strings_from_file(
    args["te_model_path_list_file"]
)

print(task_equipped_models_file_path_list, "\n\n")

for task_equipped_model_file_path in task_equipped_models_file_path_list:

    print(f"Working on task-equipped model: {task_equipped_model_file_path}")
    applied_tvs = task_equipped_model_file_path.split(
        "/"
    )[-1].split("_")[4].split("-")

    subprocess.run(
        [
            "python", 
            "src/scripts/import_te_model.py",
            f"+task_equipped_model_file_path={task_equipped_model_file_path}",
            f"task_vectors.to_apply={applied_tvs}",
            f"eval_datasets={applied_tvs}",
        ], 
        check=True
    )