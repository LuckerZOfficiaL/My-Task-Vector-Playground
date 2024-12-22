# This scripts runs an entire experiment, it goes from order 1 all the way to order "desired_orders".
from rich import print

import yaml
import subprocess

epochs = 1
desired_orders = 10
merging_method = "normal"
finetuning_method = "normal"

yaml_file = "conf/nn/data/default.yaml"
ft_conf_file = "conf/finetune.yaml"
tv_conf_file = "conf/task_vectors.yaml"

for order in range(1, desired_orders+1):

    # adjust hyperparameters in finetune.yaml
    with open(ft_conf_file, "r") as file:
            config = yaml.safe_load(file)
            config['epochs'] = epochs
            config['order'] = order
            config['merging_method'] = merging_method
            config['finetuning_method'] = finetuning_method
            config['ft_on_data_split'] = "train"
            print(config)
    with open(ft_conf_file, "w") as file:
        yaml.dump(config, file)

    # adjust hyperparameters in task_vectors.yaml
    with open(tv_conf_file, "r") as file:
            config = yaml.safe_load(file)
            config['epochs'] = epochs
            config['order'] = order
            config['task_vectors']['merging_method'] = merging_method
            config['finetuning_method'] = finetuning_method
            print(config)
    with open(tv_conf_file, "w") as file:
        yaml.dump(config, file)
    
    

    datasets_7 = ["cifar100", "dtd", "eurosat", "gtsrb", "mnist", "resisc45", "svhn"]
    datasets_20 = [
        "cars", "dtd", "eurosat", "gtsrb", "mnist", "resisc45", "sun397", 
        "svhn", "cifar10", "cifar100", "stl10", "flowers102", "food101", 
        "fer2013", "pcam", "oxfordiiitpet", "renderedsst2", "emnist", 
        "fashionmnist", "kmnist"
    ]
    # for dataset_id, dataset in enumerate(datasets_20): # modify the dataset hyperparameter in config

    #     print(f"[bold]\n\n\n{dataset} ({dataset_id + 1}/{len(datasets_20)}), order ({order}/{desired_orders})\n\n\n")

    #     with open(yaml_file, "r") as file:
    #         config = yaml.safe_load(file)
    #         config['defaults'][0]['dataset'] = dataset
    #         print(config)

    #     with open(yaml_file, "w") as file:
    #         yaml.dump(config, file)

    #     subprocess.run(["python", "src/scripts/finetune.py"], check=True)

    subprocess.run(["python", "src/scripts/evaluate.py"], check=True)
    