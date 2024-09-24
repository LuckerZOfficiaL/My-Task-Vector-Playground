# This scripts runs an entire experiment, it goes from order 1 all the way to order "desired_orders".
from rich import print

import yaml
import subprocess

epochs = 1
desired_orders = 10
merging_method = "normal"
finetuning_method = "valft"

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
            config['ft_on_data_split'] = "train" if order == 1 else "val"
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
    
    

    datasets = ["cifar100", "dtd", "eurosat", "gtsrb", "mnist", "resisc45", "svhn"]
    for dataset_id, dataset in enumerate(datasets): # modify the dataset hyperparameter in config

        print(f"[bold]\n\n\n{dataset} ({dataset_id + 1}/{len(datasets)}), order ({order}/{desired_orders})\n\n\n")

        with open(yaml_file, "r") as file:
            config = yaml.safe_load(file)
            config['defaults'][0]['dataset'] = dataset
            print(config)

        with open(yaml_file, "w") as file:
            yaml.dump(config, file)

        subprocess.run(["python", "src/scripts/finetune.py"], check=True)

    subprocess.run(["python", "src/scripts/evaluate.py"], check=True)
    