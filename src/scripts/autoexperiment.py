# This scripts runs an entire experiment, it goes from order 1 all the way to order "DESIRED_ORDERS".
from rich import print

from tvp.utils.io_utils import load_yaml
import subprocess

import builtins
import tqdm

from tvp.data.constants import DATASET_NAME_TO_TA_FT_EPOCHS_LOWERCASE

# Define a custom print function that redirects to tqdm.write
def tqdm_print(*args, **kwargs):
    # Join all arguments into a single string and pass to tqdm.write
    tqdm.write(" ".join(map(str, args)), **kwargs)

# Override the built-in print globally
builtins.print = tqdm_print

SGD = "torch.optim.SGD"
ADAM = "torch.optim.Adam"

EPOCH_DIVISOR = "None"
DESIRED_ORDERS = 1

ACCUMULATE_GRAD_BATCHES = True

# ATM
MAX_EPOCHS = 1
# TA
# MAX_EPOCHS = None

DATASETS_07 = ["cifar100", "dtd", "eurosat", "gtsrb", "mnist", "resisc45", "svhn"]
DATASETS_20 = [
    "cars", "dtd", "eurosat", "gtsrb", "mnist", "resisc45", "sun397", 
    "svhn", "cifar10", "cifar100", "stl10", "flowers102", "food101", 
    "fer2013", "pcam", "oxfordiiitpet", "renderedsst2", "emnist", 
    "fashionmnist", "kmnist"
]
datasets = ["eurosat"]

for order in range(1, DESIRED_ORDERS+1):

    for dataset_id, dataset in enumerate(datasets):

        print(f"[bold]\n\n\n{dataset} ({dataset_id + 1}/{len(datasets)}), order ({order}/{DESIRED_ORDERS})\n\n\n")

        subprocess.run(
            [
                "python", 
                "src/scripts/finetune.py",
                f"order={order}",
                f"epoch_divisor={EPOCH_DIVISOR}",
                f"accumulate_grad_batches={ACCUMULATE_GRAD_BATCHES}",
                # f"nn.module.optimizer._target_={SGD if ACCUMULATE_GRAD_BATCHES else ADAM}",
                f"nn.module.optimizer._target_={SGD if ACCUMULATE_GRAD_BATCHES else SGD}",
                f"+dataset_name={dataset}",
                f"max_epochs={DATASET_NAME_TO_TA_FT_EPOCHS_LOWERCASE[dataset] if MAX_EPOCHS is None else MAX_EPOCHS}",
            ], 
            check=True
        )

    subprocess.run(
        [
            "python", 
            "src/scripts/evaluate.py",
            f"order={order}",
            f"epoch_divisor={EPOCH_DIVISOR}",
            f"+accumulate_grad_batches={ACCUMULATE_GRAD_BATCHES}",
            f"+max_epochs={MAX_EPOCHS}",
            # f"nn.module.optimizer._target_={SGD if ACCUMULATE_GRAD_BATCHES else ADAM}",
            f"nn.module.optimizer._target_={SGD if ACCUMULATE_GRAD_BATCHES else SGD}",

        ], 
        check=True
    )
    