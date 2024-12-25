DATASET_NAME_TO_NUM_BATCHES_UPPERCASE = {
    "CIFAR100": 1407,
    "EuroSAT": 675,
    "GTSRB": 750,
    "MNIST": 1719,
    "RESISC45": 532,
    "DTD": 127,
    "SVHN": 2134,
}

DATASET_NAME_TO_NUM_BATCHES_LOWERCASE = {
    k.lower(): v for k, v in DATASET_NAME_TO_NUM_BATCHES_UPPERCASE.items()
}

DATASET_NAME_TO_TA_FT_EPOCHS_UPPERCASE = {
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
DATASET_NAME_TO_TA_FT_EPOCHS_LOWERCASE = {
    k.lower(): v for k, v in DATASET_NAME_TO_TA_FT_EPOCHS_UPPERCASE.items()
}