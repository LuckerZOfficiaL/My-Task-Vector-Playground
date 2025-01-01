###### BEGIN Ilharco et al. arzigogolato method + put the unused test split into the val split ######

DATASET_NAME_TO_NUM_TRAIN_BATCHES_UPPERCASE = {
    "CIFAR100": 1407,
    "DTD": 127,
    "EuroSAT": 675,
    "GTSRB": 750,
    "MNIST": 1719,
    "RESISC45": 532,
    "SVHN": 2134,
}

DATASET_NAME_TO_NUM_VAL_BATCHES_UPPERCASE = {
    "CIFAR100": 313,
    "DTD": 36,
    "EuroSAT": 675,
    "GTSRB": 395,
    "MNIST": 313,
    "RESISC45": 197,
    "SVHN": 814,
}

DATASET_NAME_TO_NUM_TEST_BATCHES_UPPERCASE = {
    "CIFAR100": 157,
    "DTD": 15,
    "EuroSAT": 675,
    "GTSRB": 84,
    "MNIST": 157,
    "RESISC45": 60,
    "SVHN": 157,
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

DATASET_NAME_TO_NUM_TRAIN_BATCHES_UPPERCASE = {
    "CIFAR100": 1407,
    "DTD": 141,
    "EuroSAT": 675,
    "GTSRB": 750,
    "MNIST": 1688,
    "RESISC45": 591,
    "SVHN": 2061
}

DATASET_NAME_TO_NUM_VAL_BATCHES_UPPERCASE = {
    "CIFAR100": 157,
    "DTD": 4,
    "EuroSAT": 68,
    "GTSRB": 84,
    "MNIST": 188,
    "RESISC45": 20,
    "SVHN": 229
}

DATASET_NAME_TO_NUM_TEST_BATCHES_UPPERCASE = {
    "CIFAR100": 313,
    "DTD": 36,
    "EuroSAT": 675,
    "GTSRB": 395,
    "MNIST": 313,
    "RESISC45": 197,
    "SVHN": 814
}

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