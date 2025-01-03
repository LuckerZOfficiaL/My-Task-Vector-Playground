# This scripts runs an entire experiment, it goes from order 1 all the way to order "DESIRED_ORDERS".
from rich import print

from tvp.utils.utils import list_of_strings_to_string
import subprocess

import builtins
import tqdm

from tvp.data.constants import DATASET_NAME_TO_TA_FT_EPOCHS_LOWERCASE
from tvp.data.constants import DATASET_NAME_TO_NUM_TRAIN_BATCHES_LOWERCASE
from tvp.data.constants import DATASET_NAME_TO_NUM_VAL_BATCHES_LOWERCASE
from tvp.data.constants import DATASET_NAME_TO_NUM_TEST_BATCHES_LOWERCASE
from tvp.data.constants import DATASETS_07, DATASETS_20
from tvp.data.constants import DATASET_NAME_TO_STYLED_NAME

import os

import argparse
from rich import print


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
print(f"Running autoexperiment.py with timestamp: {args['timestamp']}")
print("\n\n\n", f"*"*80, "\n\n\n")

UPLOAD_ARTIFACTS = True

EXPORT_RUN_DATA = True
EXPORT_RUN_DATA_DIR = "./run_data"

SGD = "torch.optim.SGD"
ADAM = "torch.optim.Adam"

EPOCH_DIVISOR = "None"
DESIRED_ORDERS = 1

STRATEGY = "sgd, all samples"

# ATM
# MAX_EPOCHS = 1
# TA
MAX_EPOCHS = None

SAVE_GRADS = True
SAVE_GRADS_DIR = "./grads_manual_loop"
os.makedirs(SAVE_GRADS_DIR, exist_ok=True)

datasets = DATASETS_07

for order in range(1, DESIRED_ORDERS+1):

    # for dataset_id, dataset in enumerate(datasets):

    #     print(f"[bold]\n\n\n{dataset} ({dataset_id + 1}/{len(datasets)}), order ({order}/{DESIRED_ORDERS})\n\n\n")

    #     if STRATEGY == "sgd, all samples":
    #         accumulate_grad_batches = 1
    #         limit_train_batches = DATASET_NAME_TO_NUM_TRAIN_BATCHES_LOWERCASE[dataset]
    #         limit_val_batches = DATASET_NAME_TO_NUM_VAL_BATCHES_LOWERCASE[dataset]
    #         limit_test_batches = DATASET_NAME_TO_NUM_TEST_BATCHES_LOWERCASE[dataset]
    #         batch_size = 32
    #         lr = 1e-5
    #     elif STRATEGY == "gd, all samples":
    #         accumulate_grad_batches = DATASET_NAME_TO_NUM_TRAIN_BATCHES_LOWERCASE[dataset]
    #         limit_train_batches = DATASET_NAME_TO_NUM_TRAIN_BATCHES_LOWERCASE[dataset]
    #         batch_size = 32
    #         lr = 1e-5
    #     elif STRATEGY == "sgd, limited number of samples":
    #         accumulate_grad_batches = 1
    #         batch_size = 32
    #         num_samples = 256
    #         limit_train_batches = int(num_samples/batch_size)
    #         lr = 1e-5
    #     elif STRATEGY == "gd, limited number of samples, one batch":
    #         num_samples = 256
    #         batch_size = 256
    #         accumulate_grad_batches = int(1)
    #         limit_train_batches = int(1)
    #         lr = 1e-5
    #     elif STRATEGY == "gd, limited number of samples, all batches":
    #         num_samples = 256
    #         batch_size = 32
    #         accumulate_grad_batches = int(num_samples/batch_size)
    #         limit_train_batches = int(num_samples/batch_size)
    #         lr = 1e-5
    #     else:
    #         raise NotImplementedError(f"Strategy {STRATEGY} not implemented")

    #     subprocess.run(
    #         [
    #             "python3", 
    #             "src/scripts/finetune.py",
    #             # "src/scripts/finetune_manual_loop.py",
    #             # "src/scripts/finetune_manual_loop_manual_optim.py",
    #             f"order={order}",
    #             f"epoch_divisor={EPOCH_DIVISOR}",
    #             f"accumulate_grad_batches={accumulate_grad_batches}",
    #             # f"nn.module.optimizer._target_={SGD if accumulate_grad_batches else ADAM}",
    #             f"nn.module.optimizer._target_={SGD if accumulate_grad_batches else SGD}",
    #             f"nn.module.optimizer.lr={lr}",
    #             f"+dataset_name={dataset}",
    #             f"max_epochs={DATASET_NAME_TO_TA_FT_EPOCHS_LOWERCASE[dataset] if MAX_EPOCHS is None else MAX_EPOCHS}",
    #             f"+save_grads={SAVE_GRADS}",
    #             f"+save_grads_dir={SAVE_GRADS_DIR}",
    #             f"+limit_train_batches={limit_train_batches}",
    #             f"+limit_val_batches={limit_val_batches}",
    #             f"+limit_test_batches={limit_test_batches}",
    #             f"nn.data.batch_size.train={batch_size}",
    #             f"+strategy={repr(STRATEGY)}",
    #             f"+run_data.export_run_data={EXPORT_RUN_DATA}",
    #             f"+run_data.export_run_data_dir={EXPORT_RUN_DATA_DIR}",
    #             f"+upload_artifacts={UPLOAD_ARTIFACTS}"
    #         ], 
    #         check=True
    #     )

    accumulate_grad_batches = 1
    accumulate_grad_batches = "null" if accumulate_grad_batches is None else accumulate_grad_batches
    
    limit_train_batches = None
    limit_train_batches = "null" if limit_train_batches is None else limit_train_batches
    
    max_epochs = "null" if MAX_EPOCHS is None else MAX_EPOCHS
    
    timestamp = args["timestamp"]
    timestamp = "null" if timestamp is None else timestamp

    EVAL_DATASETS = [
        DATASET_NAME_TO_STYLED_NAME[dataset] for dataset in DATASETS_20
    ]
    TASK_VECTORS_TO_APPLY = [
        DATASET_NAME_TO_STYLED_NAME[dataset] for dataset in DATASETS_20
    ]

    subprocess.run(
        [
            "python", 
            "src/scripts/evaluate.py",
            f"order={order}",
            f"epoch_divisor={EPOCH_DIVISOR}",
            f"+accumulate_grad_batches={accumulate_grad_batches}",
            f"+max_epochs={max_epochs}",
            # f"nn.module.optimizer._target_={SGD if accumulate_grad_batches else ADAM}",
            f"nn.module.optimizer._target_={SGD if accumulate_grad_batches else SGD}",
            f"accumulate_grad_batches={accumulate_grad_batches}",
            f"+limit_train_batches={limit_train_batches}",
            f"+upload_artifacts={UPLOAD_ARTIFACTS}",
            f"+timestamp={timestamp}",
            f"eval_datasets={EVAL_DATASETS}",
            f"task_vectors.to_apply={TASK_VECTORS_TO_APPLY}",
        ], 
        check=True
    )
    