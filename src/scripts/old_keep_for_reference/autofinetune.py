# This script runs the framework for 1 order, it iteratively does the following in an automated manner:
# 1) update the dataset name in the default.yaml file (previously done manually)
# 2) run finetuning.py

import yaml
import subprocess
    
yaml_file = "conf/nn/data/default.yaml"

datasets = ["cifar100", "dtd", "eurosat", "gtsrb", "mnist", "resisc45", "svhn"]

for dataset in datasets:
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
        config['defaults'][0]['dataset'] = dataset
        print(config)

    with open(yaml_file, "w") as file:
        yaml.dump(config, file)

    subprocess.run(["python", "src/scripts/finetune.py"], check=True)

    print(f"\n\n\nExperiment for dataset {dataset} completed.\n\n\n")

subprocess.run(["python", "src/scripts/evaluate.py"], check=True) # run this to evaluate the unified multitask model after the current oder of TVA
