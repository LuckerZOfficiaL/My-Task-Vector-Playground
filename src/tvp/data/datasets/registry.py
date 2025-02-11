import copy
import inspect
import random
import sys

import torch
from torch.utils.data.dataset import random_split

from src.tvp.data.datasets.cars import Cars
from src.tvp.data.datasets.cifar10 import CIFAR10
from src.tvp.data.datasets.cifar100 import CIFAR100
from src.tvp.data.datasets.dtd import DTD
from src.tvp.data.datasets.emnist import EMNIST
from src.tvp.data.datasets.eurosat import EuroSAT, EuroSATVal
from src.tvp.data.datasets.fashionmnist import FashionMNIST
from src.tvp.data.datasets.fer2013 import FER2013
from src.tvp.data.datasets.flowers102 import Flowers102
from src.tvp.data.datasets.food101 import Food101
from src.tvp.data.datasets.gtsrb import GTSRB
from src.tvp.data.datasets.imagenet import ImageNet
from src.tvp.data.datasets.kmnist import KMNIST
from src.tvp.data.datasets.mnist import MNIST
from src.tvp.data.datasets.oxfordpets import OxfordIIITPet
from src.tvp.data.datasets.pcam import PCAM
from src.tvp.data.datasets.resisc45 import RESISC45
from src.tvp.data.datasets.sst2 import RenderedSST2
from src.tvp.data.datasets.stl10 import STL10
from src.tvp.data.datasets.sun397 import SUN397
from src.tvp.data.datasets.svhn import SVHN

registry = {name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)}


class GenericDataset(object):
    def __init__(self):
        self.train_dataset = None
        self.train_loader = None
        self.val_dataset = None
        self.val_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.classnames = None


def split_train_into_train_val(
    dataset, new_dataset_class_name, batch_size, num_workers, val_fraction, max_val_samples=None, seed=0
):
    assert val_fraction > 0.0 and val_fraction < 1.0
    total_size = len(dataset.train_dataset)
    val_size = int(total_size * val_fraction)
    if max_val_samples is not None:
        val_size = min(val_size, max_val_samples)
    train_size = total_size - val_size

    assert val_size > 0
    assert train_size > 0

    lengths = [train_size, val_size]

    trainset, valset = random_split(dataset.train_dataset, lengths, generator=torch.Generator().manual_seed(seed))
    if new_dataset_class_name == "MNISTVal":
        assert trainset.indices[0] == 36044

    new_dataset = None

    new_dataset_class = type(new_dataset_class_name, (GenericDataset,), {})
    new_dataset = new_dataset_class()

    new_dataset.train_dataset = trainset
    new_dataset.train_loader = torch.utils.data.DataLoader(
        new_dataset.train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    new_dataset.val_dataset = valset
    new_dataset.val_loader = torch.utils.data.DataLoader(
        new_dataset.val_dataset, batch_size=batch_size, num_workers=num_workers
    )

    new_dataset.test_dataset = dataset.test_dataset
    new_dataset.test_loader = torch.utils.data.DataLoader(
        new_dataset.test_dataset, batch_size=batch_size, num_workers=num_workers
    )

    new_dataset.classnames = copy.copy(dataset.classnames)

    return new_dataset


def get_dataset(
    dataset_name, preprocess_fn, location, batch_size=128, num_workers=6, val_fraction=0.1, max_val_samples=5000
):
    if dataset_name.endswith("Val"):
        # Handle val splits
        if dataset_name in registry:
            dataset_class = registry[dataset_name]
        else:
            base_dataset_name = dataset_name.split("Val")[0]
            base_dataset = get_dataset(base_dataset_name, preprocess_fn, location, batch_size, num_workers)
            dataset = split_train_into_train_val(
                base_dataset, dataset_name, batch_size, num_workers, val_fraction, max_val_samples
            )
            return dataset
    else:
        assert (
            dataset_name in registry
        ), f"Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}"
        dataset_class = registry[dataset_name]
    dataset = dataset_class(preprocess_fn, location=location, batch_size=batch_size, num_workers=num_workers)
    return dataset
