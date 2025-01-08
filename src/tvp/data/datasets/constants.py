DATASETS_PAPER_ATM = [
    "dtd", "gtsrb", "eurosat", "resisc45", "cifar100", "mnist", "svhn"
]

DATASETS_PAPER_TSV_8 = [
    "cars", "dtd", "eurosat", "gtsrb", "mnist", "resisc45", "svhn", "sun397"
]

DATASETS_PAPER_TSV_14 = [
    "cars", "dtd", "eurosat", "gtsrb", "mnist", "resisc45", "svhn", "sun397", 
    "stl10", "oxfordiiitpet", "flowers102", "cifar100", "pcam", "fer2013"
]

DATASETS_PAPER_TSV_20 = [
    "cars", "dtd", "eurosat", "gtsrb", "mnist", "resisc45", "svhn", "sun397", 
    "stl10", "oxfordiiitpet", "flowers102", "cifar100", "pcam", "fer2013",
    "cifar10", "food101", "fashionmnist", "renderedsst2", "emnist", "kmnist"
]

DATASETS_PAPER_TSV_20_MINUS_PAPER_ATM = [
    dataset
    for dataset in DATASETS_PAPER_TSV_20
    if dataset not in DATASETS_PAPER_ATM
]

DATASET_TO_STYLED = {
    "dtd": "DTD",
    "gtsrb": "GTSRB",
    "eurosat": "EuroSAT",
    "resisc45": "RESISC45",
    "cifar100": "CIFAR100",
    "mnist": "MNIST",
    "svhn": "SVHN",
    "cars": "CARS",
    "sun397": "SUN397",
    "stl10": "STL10",
    "oxfordiiitpet": "OxfordIIITPet",
    "flowers102": "Flowers102",
    "pcam": "PCAM",
    "fer2013": "FER2013",
    "cifar10": "CIFAR10",
    "food101": "Food101",
    "fashionmnist": "FashionMNIST",
    "renderedsst2": "RenderedSST2",
    "emnist": "EMNIST",
    "kmnist": "KMNIST"
}