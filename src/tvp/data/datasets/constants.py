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
    "cars": "Cars",
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

DATASET_TO_NUM_BATCHES = {
    "cars": {
        128: 58
    },
    "dtd": {
        128: 32
    },
    "eurosat": {
        128: 169
    },
    "gtsrb": {
        128: 188
    },
    "mnist": {
        128: 430
    },
    "resisc45": {
        128: 133
    },
    "svhn": {
        128: 534
    },
    "sun397": {
        128: 140
    },
    "stl10": {
        128: 36
    },
    "oxfordiiitpet": {
        128: 26
    },
    "flowers102": {
        128: 8
    },
    "cifar100": {
        128: 352
    },
    "pcam": {
        128: 2009
    },
    "fer2013": {
        128: 202
    },
    "cifar10": {
        128: 352
    },
    "food101": {
        128: 553
    },
    "fashionmnist": {
        128: 430
    },
    "renderedsst2": {
        128: 49
    },
    "emnist": {
        128: 1836
    },
    "kmnist": {
        128: 430
    }
}