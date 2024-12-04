from rich import print

from itertools import combinations

import yaml
import subprocess

datasets_7 = ["cifar100", "dtd", "eurosat", "gtsrb", "mnist", "resisc45", "svhn"]
datasets_14 = [
    "mnist", "gtsrb", "eurosat", "dtd", "cars", "fer2013", "pcam", "cifar100", 
    "flowers102", "oxfordiiitpet", "stl10", "sun397", "svhn", "resisc45"
]
datasets_20 = [
    "cars", "dtd", "eurosat", "gtsrb", "mnist", "resisc45", "sun397", 
    "svhn", "cifar10", "cifar100", "stl10", "flowers102", "food101", 
    "fer2013", "pcam", "oxfordiiitpet", "renderedsst2", "emnist", 
    "fashionmnist", "kmnist"
]
datasets_to_stylized = {
    "cifar100": "CIFAR100",
    "dtd": "DTD",
    "eurosat": "EuroSAT",
    "gtsrb": "GTSRB",
    "mnist": "MNIST",
    "resisc45": "RESISC45",
    "svhn": "SVHN",
    "cars": "Cars",
    "sun397": "SUN397",
    "cifar10": "CIFAR10",
    "stl10": "STL10",
    "flowers102": "Flowers102",
    "food101": "Food101",
    "fer2013": "FER2013",
    "pcam": "PCAM",
    "oxfordiiitpet": "OxfordIIITPet",
    "renderedsst2": "RenderedSST2",
    "emnist": "EMNIST",
    "fashionmnist": "FashionMNIST",
    "kmnist": "KMNIST"
}

# assign this to pick the dataset_* list you want to use!
datasets = datasets_20

# assign these to pick the range of combinations you want to use!
datasets_combo_min = 19 # inclusive of the first combo length you want!
datasets_combo_max = 19 # inclusive of the last combo length you want!

all_combinations = []

for r in range(datasets_combo_min, datasets_combo_max + 1):
    print(r)
    all_combinations.extend(combinations(datasets, r))

all_combinations = list(all_combinations)

print(f"Total number of combinations: {len(all_combinations)}")

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


    subprocess.run(
        ["python", "src/scripts/evaluate_subset_export_task_equipped_models.py"], 
        check=True
    )
    subprocess.run(
        ["python", "src/scripts/evaluate_subset_import_task_equipped_models.py"], 
        check=True
    )