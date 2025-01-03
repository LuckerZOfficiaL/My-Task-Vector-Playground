DATASETS_07 = ["cifar100", "dtd", "eurosat", "gtsrb", "mnist", "resisc45", "svhn"]
DATASETS_20 = DATASETS_07 + [
    "cars", "cifar10", "emnist", "fashionmnist", "fer2013", "flowers102", 
    "food101", "kmnist", "oxfordiiitpet", 
    "pcam", 
    "renderedsst2", "stl10", "sun397"
]

DATASET_NAME_TO_STYLED_NAME = {
    "cifar100": "CIFAR100",
    "dtd": "DTD",
    "eurosat": "EuroSAT",
    "gtsrb": "GTSRB",
    "mnist": "MNIST",
    "resisc45": "RESISC45",
    "svhn": "SVHN",
    "cars": "Cars",
    "cifar10": "CIFAR10",
    "emnist": "EMNIST",
    "fashionmnist": "FashionMNIST",
    "fer2013": "FER2013",
    "flowers102": "Flowers102",
    "food101": "Food101",
    "kmnist": "KMNIST",
    "oxfordiiitpet": "OxfordIIITPet",
    "pcam": "PCAM",
    "renderedsst2": "RenderedSST2",
    "stl10": "STL10",
    "sun397": "SUN397",
}

###### BEGIN Ilharco et al. arzigogolato method + put the unused test split into the val split ######

DATASET_NAME_TO_NUM_TRAIN_BATCHES_UPPERCASE = {
    "CIFAR100": 1407,
    "DTD": 127,
    "EuroSAT": 675,
    "GTSRB": 750,
    "MNIST": 1719,
    "RESISC45": 532,
    "SVHN": 2134,
    "Cars": 230,
    "CIFAR10": 1407,
    "EMNIST": 7344,
    "FashionMNIST": 1719,
    "FER2013": 808,
    "Flowers102": 29,
    "Food101": 2211,
    "KMNIST": 1719,
    "OxfordIIITPet": 104,
    "PCam": 8036,
    "RenderedSST2": 195,
    "STL10": 141,
    "SUN397": 559,
}

DATASET_NAME_TO_NUM_VAL_BATCHES_UPPERCASE = {
    "CIFAR100": 313,
    "DTD": 36,
    "EuroSAT": 675,
    "GTSRB": 395,
    "MNIST": 313,
    "RESISC45": 197,
    "SVHN": 814,
    "Cars": 252,
    "CIFAR10": 313,
    "EMNIST": 1250,
    "FashionMNIST": 313,
    "FER2013": 225,
    "Flowers102": 193,
    "Food101": 790,
    "KMNIST": 313,
    "OxfordIIITPet": 115,
    "PCam": 1024,
    "RenderedSST2": 57,
    "STL10": 250,
    "SUN397": 621,
}

DATASET_NAME_TO_NUM_TEST_BATCHES_UPPERCASE = {
    "CIFAR100": 157,
    "DTD": 15,
    "EuroSAT": 675,
    "GTSRB": 84,
    "MNIST": 157,
    "RESISC45": 60,
    "SVHN": 157,
    "Cars": 26,
    "CIFAR10": 157,
    "EMNIST": 157,
    "FashionMNIST": 157,
    "FER2013": 90,
    "Flowers102": 4,
    "Food101": 157,
    "KMNIST": 157,
    "OxfordIIITPet": 12,
    "PCam": 157,
    "RenderedSST2": 22,
    "STL10": 16,
    "SUN397": 63,
}

###### END Ilharco et al. arzigogolato method + put the unused test split into the val split ######


####### BEGIN len(test_set) = len(val_set) #######
# DATASET_NAME_TO_NUM_TRAIN_BATCHES_UPPERCASE = {
#     "CIFAR100": 1250,
#     "DTD": 141,
#     "EuroSAT": 675,
#     "GTSRB": 395,
#     "MNIST": 1563,
#     "RESISC45": 591,
#     "SVHN": 1476
# }

# DATASET_NAME_TO_NUM_VAL_BATCHES_UPPERCASE = {
#     "CIFAR100": 313,
#     "DTD": 36,
#     "EuroSAT": 675,
#     "GTSRB": 395,
#     "MNIST": 313,
#     "RESISC45": 197,
#     "SVHN": 814
# }

# DATASET_NAME_TO_NUM_TEST_BATCHES_UPPERCASE = {
#     "CIFAR100": 313,
#     "DTD": 36,
#     "EuroSAT": 675,
#     "GTSRB": 395,
#     "MNIST": 313,
#     "RESISC45": 197,
#     "SVHN": 814
# }
####### END len(test_set) = len(val_set) #######

####### BEGIN len(val_set) = 0.1 * len(train_set) and len(val_set) = 0.1 * len(val_set) for dataset w/ own val split #######

# DATASET_NAME_TO_NUM_TRAIN_BATCHES_UPPERCASE = {
#     "CIFAR100": 1407,
#     "DTD": 141,
#     "EuroSAT": 675,
#     "GTSRB": 750,
#     "MNIST": 1688,
#     "RESISC45": 591,
#     "SVHN": 2061
# }

# DATASET_NAME_TO_NUM_VAL_BATCHES_UPPERCASE = {
#     "CIFAR100": 157,
#     "DTD": 4,
#     "EuroSAT": 68,
#     "GTSRB": 84,
#     "MNIST": 188,
#     "RESISC45": 20,
#     "SVHN": 229
# }

# DATASET_NAME_TO_NUM_TEST_BATCHES_UPPERCASE = {
#     "CIFAR100": 313,
#     "DTD": 36,
#     "EuroSAT": 675,
#     "GTSRB": 395,
#     "MNIST": 313,
#     "RESISC45": 197,
#     "SVHN": 814
# }

####### END len(val_set) = 0.1 * len(train_set) and len(val_set) = 0.1 * len(val_set) for dataset w/ own val split #######


DATASET_NAME_TO_NUM_TRAIN_BATCHES_LOWERCASE = {
    k.lower(): v for k, v in DATASET_NAME_TO_NUM_TRAIN_BATCHES_UPPERCASE.items()
}

DATASET_NAME_TO_NUM_VAL_BATCHES_LOWERCASE = {
    k.lower(): v for k, v in DATASET_NAME_TO_NUM_VAL_BATCHES_UPPERCASE.items()
}

DATASET_NAME_TO_NUM_TEST_BATCHES_LOWERCASE = {
    k.lower(): v for k, v in DATASET_NAME_TO_NUM_TEST_BATCHES_UPPERCASE.items()
}

DATASET_NAME_TO_TA_FT_EPOCHS_UPPERCASE = {
    "Cars": 35,
    "DTD": 76,
    "EuroSAT": 12,
    "GTSRB": 11,
    "MNIST": 5,
    "RESISC45": 15,
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
    "SUN397": 14,
}
DATASET_NAME_TO_TA_FT_EPOCHS_LOWERCASE = {
    k.lower(): v for k, v in DATASET_NAME_TO_TA_FT_EPOCHS_UPPERCASE.items()
}

# batch_size = 32
# lim_train_batches = all batches
# acc_grad_batches = 1
# optim = SGD
DATASET_NAME_TO_FT_TA_TEST_ACC = {
    "SUN397": 0.7017632126808167,
}