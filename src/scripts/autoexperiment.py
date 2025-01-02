# This scripts runs an entire experiment, it goes from order 1 all the way to order "DESIRED_ORDERS".
from rich import print

from tvp.utils.io_utils import load_yaml
import subprocess

import builtins
import tqdm

from tvp.data.constants import DATASET_NAME_TO_TA_FT_EPOCHS_LOWERCASE
from tvp.data.constants import DATASET_NAME_TO_NUM_TRAIN_BATCHES_LOWERCASE
from tvp.data.constants import DATASET_NAME_TO_NUM_VAL_BATCHES_LOWERCASE
from tvp.data.constants import DATASET_NAME_TO_NUM_TEST_BATCHES_LOWERCASE
from tvp.data.constants import DATASETS_07, DATASETS_20

import os

# Define a custom print function that redirects to tqdm.write
def tqdm_print(*args, **kwargs):
    # Join all arguments into a single string and pass to tqdm.write
    tqdm.write(" ".join(map(str, args)), **kwargs)

# Override the built-in print globally
builtins.print = tqdm_print

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

    for dataset_id, dataset in enumerate(datasets):

        print(f"[bold]\n\n\n{dataset} ({dataset_id + 1}/{len(datasets)}), order ({order}/{DESIRED_ORDERS})\n\n\n")

        if STRATEGY == "sgd, all samples":
            accumulate_grad_batches = 1
            limit_train_batches = DATASET_NAME_TO_NUM_TRAIN_BATCHES_LOWERCASE[dataset]
            limit_val_batches = DATASET_NAME_TO_NUM_VAL_BATCHES_LOWERCASE[dataset]
            limit_test_batches = DATASET_NAME_TO_NUM_TEST_BATCHES_LOWERCASE[dataset]
            batch_size = 32
            lr = 1e-5
        elif STRATEGY == "gd, all samples":
            accumulate_grad_batches = DATASET_NAME_TO_NUM_TRAIN_BATCHES_LOWERCASE[dataset]
            limit_train_batches = DATASET_NAME_TO_NUM_TRAIN_BATCHES_LOWERCASE[dataset]
            batch_size = 32
            lr = 1e-5
        elif STRATEGY == "sgd, limited number of samples":
            accumulate_grad_batches = 1
            batch_size = 32
            num_samples = 256
            limit_train_batches = int(num_samples/batch_size)
            lr = 1e-5
        elif STRATEGY == "gd, limited number of samples, one batch":
            num_samples = 256
            batch_size = 256
            accumulate_grad_batches = int(1)
            limit_train_batches = int(1)
            lr = 1e-5
        elif STRATEGY == "gd, limited number of samples, all batches":
            num_samples = 256
            batch_size = 32
            accumulate_grad_batches = int(num_samples/batch_size)
            limit_train_batches = int(num_samples/batch_size)
            lr = 1e-5
        else:
            raise NotImplementedError(f"Strategy {STRATEGY} not implemented")

        subprocess.run(
            [
                "python3", 
                "src/scripts/finetune.py",
                # "src/scripts/finetune_manual_loop.py",
                # "src/scripts/finetune_manual_loop_manual_optim.py",
                f"order={order}",
                f"epoch_divisor={EPOCH_DIVISOR}",
                f"accumulate_grad_batches={accumulate_grad_batches}",
                # f"nn.module.optimizer._target_={SGD if accumulate_grad_batches else ADAM}",
                f"nn.module.optimizer._target_={SGD if accumulate_grad_batches else SGD}",
                f"nn.module.optimizer.lr={lr}",
                f"+dataset_name={dataset}",
                f"max_epochs={DATASET_NAME_TO_TA_FT_EPOCHS_LOWERCASE[dataset] if MAX_EPOCHS is None else MAX_EPOCHS}",
                f"+save_grads={SAVE_GRADS}",
                f"+save_grads_dir={SAVE_GRADS_DIR}",
                f"+limit_train_batches={limit_train_batches}",
                f"+limit_val_batches={limit_val_batches}",
                f"+limit_test_batches={limit_test_batches}",
                f"nn.data.batch_size.train={batch_size}",
                f"+strategy={repr(STRATEGY)}",
                f"+run_data.export_run_data={EXPORT_RUN_DATA}",
                f"+run_data.export_run_data_dir={EXPORT_RUN_DATA_DIR}",
                f"+upload_artifacts={UPLOAD_ARTIFACTS}"
            ], 
            check=True
        )

    # subprocess.run(
    #     [
    #         "python", 
    #         "src/scripts/evaluate.py",
    #         f"order={order}",
    #         f"epoch_divisor={EPOCH_DIVISOR}",
    #         f"+accumulate_grad_batches={accumulate_grad_batches}",
    #         f"+max_epochs={MAX_EPOCHS}",
    #         # f"nn.module.optimizer._target_={SGD if accumulate_grad_batches else ADAM}",
    #         f"nn.module.optimizer._target_={SGD if accumulate_grad_batches else SGD}",

    #     ], 
    #     check=True
    # )
    