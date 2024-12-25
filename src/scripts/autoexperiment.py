# This scripts runs an entire experiment, it goes from order 1 all the way to order "desired_orders".
from rich import print

import yaml
import subprocess

import builtins
import tqdm

# Define a custom print function that redirects to tqdm.write
def tqdm_print(*args, **kwargs):
    # Join all arguments into a single string and pass to tqdm.write
    tqdm.write(" ".join(map(str, args)), **kwargs)

# Override the built-in print globally
builtins.print = tqdm_print

DATASET_NAME_TO_TA_FT_EPOCHS = {
    "Cars": 35,
    "DTD": 76,
    "EuroSAT": 12,
    "GTSRB": 11,
    "MNIST": 5,
    "RESISC45": 15,
    "SUN397": 14,
    "SVHN": 4,
    "CIFAR10": 6,
    "CIFAR100": 6,
    "STL10": 60,
    "Food101": 4,
    "Flowers102": 147,
    "FER2013": 10,
    "PCAM": 1,
    "OxfordIIITPet": 82,
    "RenderedSST2": 39,
    "EMNIST": 2,
    "FashionMNIST": 5,
    "KMNIST": 5,
}
DATASET_NAME_TO_TA_FT_EPOCHS = {
    k.lower(): v for k, v in DATASET_NAME_TO_TA_FT_EPOCHS.items()
}

DATASET_NAME_TO_NUM_BATCHES = {
    "CIFAR100": 1407,
    "EuroSAT": 675,
    "GTSRB": 750,
    "MNIST": 1719,
    "RESISC45": 532,
    "DTD": 127,
    "SVHN": 2134,
}
DATASET_NAME_TO_NUM_BATCHES = {
    k.lower(): v for k, v in DATASET_NAME_TO_NUM_BATCHES.items()
}

epoch_divisor = "None"
desired_orders = 1

yaml_file = "conf/nn/data/default.yaml"
ft_conf_file = "conf/finetune.yaml"
tv_conf_file = "conf/task_vectors.yaml"
module_conf_file = "conf/nn/module/default.yaml"

ACCUMULATE_GRAD_BATCHES = True

# ATM
MAX_EPOCHS = 1
# TA
# MAX_EPOCHS = None

for order in range(1, desired_orders+1):

    # adjust hyperparameters in finetune.yaml
    with open(ft_conf_file, "r") as file:
            config = yaml.safe_load(file)
            config['epoch_divisor'] = epoch_divisor
            config['order'] = order
            config['accumulate_grad_batches'] = ACCUMULATE_GRAD_BATCHES
            # print(config)
    with open(ft_conf_file, "w") as file:
        yaml.dump(config, file)
    
    # adjust hyperparameters in nn/module/default.yaml
    with open(module_conf_file, "r") as file:
            config = yaml.safe_load(file)

            if ACCUMULATE_GRAD_BATCHES:
                config["optimizer"]["_target_"] = "torch.optim.SGD"
            else:
                # config["optimizer"]["_target_"] = "torch.optim.Adam"
                config["optimizer"]["_target_"] = "torch.optim.SGD"
            # print(config)
    with open(module_conf_file, "w") as file:
        yaml.dump(config, file)

    # adjust hyperparameters in task_vectors.yaml
    with open(tv_conf_file, "r") as file:
            config = yaml.safe_load(file)
            config['epoch_divisor'] = epoch_divisor
            config['order'] = order
            # print(config)
    with open(tv_conf_file, "w") as file:
        yaml.dump(config, file)
    

    datasets_7 = ["cifar100", "dtd", "eurosat", "gtsrb", "mnist", "resisc45", "svhn"]
    datasets_20 = [
        "cars", "dtd", "eurosat", "gtsrb", "mnist", "resisc45", "sun397", 
        "svhn", "cifar10", "cifar100", "stl10", "flowers102", "food101", 
        "fer2013", "pcam", "oxfordiiitpet", "renderedsst2", "emnist", 
        "fashionmnist", "kmnist"
    ]
    datasets = datasets_7
    for dataset_id, dataset in enumerate(datasets): # modify the dataset hyperparameter in config

        print(f"[bold]\n\n\n{dataset} ({dataset_id + 1}/{len(datasets)}), order ({order}/{desired_orders})\n\n\n")

        with open(yaml_file, "r") as file:
            config = yaml.safe_load(file)
            config['defaults'][0]['dataset'] = dataset
            # print(config)

        with open(yaml_file, "w") as file:
            yaml.dump(config, file)

        # adjust hyperparameters in finetune.yaml
        with open(ft_conf_file, "r") as file:
                config = yaml.safe_load(file)
                config['max_epochs'] = DATASET_NAME_TO_TA_FT_EPOCHS[dataset] if MAX_EPOCHS is None else MAX_EPOCHS
                # print(config)
        with open(ft_conf_file, "w") as file:
            yaml.dump(config, file)

        # adjust hyperparameters in nn/module/default.yaml
        with open(module_conf_file, "r") as file:
                config = yaml.safe_load(file)

                if ACCUMULATE_GRAD_BATCHES:
                    config["optimizer"]["lr"] = float(config["optimizer"]["lr"]) / DATASET_NAME_TO_NUM_BATCHES[dataset]
                    # config["optimizer"]["lr"] = 1e-5
                else:
                    config["optimizer"]["lr"] = 1e-5
        
        with open(module_conf_file, "w") as file:
            yaml.dump(config, file)

        subprocess.run(["python", "src/scripts/finetune.py"], check=True)

    subprocess.run(["python", "src/scripts/evaluate.py"], check=True)
    