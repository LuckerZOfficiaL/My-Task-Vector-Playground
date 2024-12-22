import os
import shutil

import torch
import torchvision.datasets as datasets

from rich import print


class SUN397:
    def __init__(self, preprocess, location=os.path.expanduser("~/data"), batch_size=32, num_workers=16):
        # Data loading code
        traindir = os.path.join(location, "sun397", "train")
        valdir = os.path.join(location, "sun397", "val")

        self.train_dataset = datasets.ImageFolder(traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.ImageFolder(valdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, num_workers=num_workers
        )
        idx_to_class = dict((v, k) for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i][2:].replace("_", " ") for i in range(len(idx_to_class))]

        assert len(self.classnames) == 397, f"Expected 397 classes, got {len(self.classnames)}"


def make_sun397():
    import os
    import shutil
    from tqdm import tqdm

    # Load the file Testing_01.txt and read its content
    file_path = "Training_01.txt"
    with open(file_path, 'r') as file:
        strings_list = [line.strip() for line in file]

    # Base paths
    source_base = "/mnt/KS_2TB/PARA/Projects/ATM/Higher-Order-Task-Vector-Playground/data/SUN397"
    destination_base = "/mnt/KS_2TB/PARA/Projects/ATM/Higher-Order-Task-Vector-Playground/data/sun397/train"

    for file_name in tqdm(strings_list):
        source_path = os.path.join(source_base, file_name.lstrip('/'))
        file_name_trimmed = "/".join(file_name.split("/")[2:])
        destination_path = os.path.join(destination_base, file_name_trimmed) 

        # Debugging
        # print(f"Source Path: {repr(source_path)}")
        # print(f"Destination Path: {repr(destination_path)}")
        
        # Check if source exists
        if not os.path.exists(source_path):
            print(f"[bold red]File not found: {source_path}")
            continue

        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        # Copy file
        try:
            shutil.copy(source_path, destination_path)
            # print(f"[bold green]Copied: {source_path} -> {destination_path}")
        except Exception as e:
            print(f"[bold red]Error copying {source_path} to {destination_path}: {e}")



if __name__ == "__main__":
    # run this function to prepare the dataset as per Editing Models with Task 
    # Arithmetic guys requirements
    # make_sun397()

    # run this to test whether the prep has been done correctly
    sun397 = SUN397(preprocess=None, location="./data")