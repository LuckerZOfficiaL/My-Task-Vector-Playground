from rich import print

from itertools import combinations

import yaml
import subprocess

datasets = ["cifar100", "dtd", "eurosat", "gtsrb", "mnist", "resisc45", "svhn"]
datasets_to_stylized = {
    "cifar100": "CIFAR100",
    "dtd": "DTD",
    "eurosat": "EuroSAT",
    "gtsrb": "GTSRB",
    "mnist": "MNIST",
    "resisc45": "RESISC45",
    "svhn": "SVHN"
}

all_combinations = []

for r in range(2, len(datasets) + 1):
    all_combinations.extend(combinations(datasets, r))

all_combinations = list(all_combinations)

tv_conf_file = "conf/task_vectors.yaml"

for combo_id, combo in enumerate(all_combinations):
    combo = list(combo)
    combo = [datasets_to_stylized[dataset] for dataset in combo]
    print(f"{combo_id}/{len(all_combinations)}, {combo}")

    with open(tv_conf_file, "r") as file:
        config = yaml.safe_load(file)
        config['task_vectors']['to_apply'] = combo
        config['eval_datasets'] = combo
    with open(tv_conf_file, "w") as file:
        yaml.dump(config, file)


    subprocess.run(["python", "src/scripts/evaluate_subset.py"], check=True)