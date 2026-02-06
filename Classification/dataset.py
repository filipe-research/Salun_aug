"""
    function for loading datasets
    contains: 
        CIFAR-10
        CIFAR-100   
"""
import copy
import glob
import os
from shutil import move

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torchvision.transforms import TrivialAugmentWide
from tqdm import tqdm
from medmnist import BloodMNIST, PathMNIST, OrganAMNIST, OCTMNIST, DermaMNIST, PneumoniaMNIST

def _get_label_array(dataset: torch.utils.data.Dataset):
    """Return the label array reference and a string key indicating where labels live."""
    if hasattr(dataset, "targets"):
        return dataset.targets, "targets"
    if hasattr(dataset, "labels"):
        return dataset.labels, "labels"
    if hasattr(dataset, "_labels"):
        return dataset._labels, "_labels"
    raise AttributeError("Dataset has no targets/labels/_labels")


def mark_indexes_only(dataset: torch.utils.data.Dataset, indexes):
    """Mark samples by flipping labels to negative (same convention used by replace_indexes(only_mark=True))."""
    labels, key = _get_label_array(dataset)
    labels = np.array(labels)
    labels[indexes] = -labels[indexes] - 1
    # write back preserving attribute
    if key == "targets":
        dataset.targets = labels
    elif key == "labels":
        dataset.labels = labels
    else:
        dataset._labels = labels


def select_removal_indexes_binary(labels: np.ndarray, total_to_remove: int, mode: str = "random",
                                 skew_malignant_frac: float = 0.7, seed: int = 0):
    """Select indexes to remove/mark for a binary dataset (labels in {0,1}).

    mode:
      - "random": uniform over all samples
      - "balanced": remove the same fraction from each class
      - "skewed": remove `skew_malignant_frac` of removals from malignant (label=1)

    Returns: numpy array of indexes.
    """
    rng = np.random.RandomState(seed)
    labels = labels.astype(np.int64)
    n = len(labels)
    total_to_remove = int(total_to_remove)
    total_to_remove = max(0, min(total_to_remove, n))

    if total_to_remove == 0:
        return np.array([], dtype=np.int64)

    all_idx = np.arange(n, dtype=np.int64)

    if mode == "random":
        return rng.choice(all_idx, size=total_to_remove, replace=False)

    idx_mal = np.flatnonzero(labels == 1) #Assume que a maligna é a classe 1 - correto
    idx_ben = np.flatnonzero(labels == 0)
    n_mal = len(idx_mal)
    n_ben = len(idx_ben)

    if n_mal == 0 or n_ben == 0:
        # fallback
        return rng.choice(all_idx, size=total_to_remove, replace=False)

    if mode == "balanced":
        # same fraction r from each class
        r = total_to_remove / float(n)
        rm_mal = int(round(r * n_mal))
        rm_ben = int(round(r * n_ben))
        # adjust to match total exactly
        delta = total_to_remove - (rm_mal + rm_ben)
        if delta > 0:
            # add to the class with more remaining capacity
            for _ in range(delta):
                if (rm_mal < n_mal) and (rm_ben >= n_ben or (n_mal - rm_mal) >= (n_ben - rm_ben)):
                    rm_mal += 1
                elif rm_ben < n_ben:
                    rm_ben += 1
        elif delta < 0:
            for _ in range(-delta):
                if rm_mal > 0 and (rm_ben == 0 or rm_mal >= rm_ben):
                    rm_mal -= 1
                elif rm_ben > 0:
                    rm_ben -= 1

    elif mode == "skewed":
        skew_malignant_frac = float(skew_malignant_frac)
        skew_malignant_frac = min(max(skew_malignant_frac, 0.0), 1.0)
        rm_mal = int(round(skew_malignant_frac * total_to_remove))
        rm_ben = total_to_remove - rm_mal
        # clip to available counts and re-adjust
        rm_mal = min(rm_mal, n_mal)
        rm_ben = min(rm_ben, n_ben)
        cur = rm_mal + rm_ben
        if cur < total_to_remove:
            # fill remaining from whichever class still has capacity
            remaining = total_to_remove - cur
            cap_mal = n_mal - rm_mal
            cap_ben = n_ben - rm_ben
            add_mal = min(remaining, cap_mal)
            rm_mal += add_mal
            remaining -= add_mal
            if remaining > 0:
                rm_ben += min(remaining, cap_ben)

    else:
        raise ValueError(f"Unknown removal mode: {mode}")

    sel_mal = rng.choice(idx_mal, size=rm_mal, replace=False) if rm_mal > 0 else np.array([], dtype=np.int64)
    sel_ben = rng.choice(idx_ben, size=rm_ben, replace=False) if rm_ben > 0 else np.array([], dtype=np.int64)
    return np.concatenate([sel_mal, sel_ben]).astype(np.int64)


def cifar10_dataloaders_no_val(
    batch_size=128, data_dir="datasets/cifar10", num_workers=2
):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-10\t 45000 images for training \t 5000 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)
    val_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def svhn_dataloaders(
    batch_size=128,
    data_dir="datasets/svhn",
    num_workers=2,
    class_to_replace: int = None,
    num_indexes_to_replace=None,
    indexes_to_replace=None,
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
    no_aug=False,
):
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: SVHN\t 45000 images for training \t 5000 images for validation\t"
    )

    train_set = SVHN(data_dir, split="train", transform=train_transform, download=True)

    test_set = SVHN(data_dir, split="test", transform=test_transform, download=True)

    train_set.labels = np.array(train_set.labels)
    test_set.labels = np.array(test_set.labels)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.labels) + 1):
        class_idx = np.where(train_set.labels == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.labels = train_set_copy.labels[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.labels = train_set_copy.labels[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        if num_indexes_to_replace is None or num_indexes_to_replace == 4454:
            test_set.data = test_set.data[test_set.labels != class_to_replace]
            test_set.labels = test_set.labels[test_set.labels != class_to_replace]

    if indexes_to_replace is not None:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader


def cifar100_dataloaders(
    batch_size=128,
    data_dir="datasets/cifar100",
    num_workers=2,
    class_to_replace: int = None,
    num_indexes_to_replace=None,
    indexes_to_replace=None,
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
    no_aug=False,
    aug_mode=None
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else: #baseline
        
        if aug_mode == "crop-flip" or aug_mode==None:
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        elif aug_mode == "crop-flip-randaug":
            rand_augment = transforms.RandAugment(num_ops=2, magnitude=9)
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    rand_augment,
                    transforms.ToTensor(),
                ]
            )
            print("-------------------- Tudo ok!!")
        
        elif aug_mode == "crop-flip-autoaug":
            auto_augment = AutoAugment(policy=AutoAugmentPolicy.CIFAR10)
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    auto_augment, 
                    transforms.ToTensor(),
                ]
            )
        elif aug_mode == "crop-flip-rerase":
            random_erasing = transforms.RandomErasing(
                p=0.5,             # Probability of applying Random Erasing
                scale=(0.02, 0.33), # Range of proportion of erased area relative to image
                ratio=(0.3, 3.3),  # Aspect ratio of erased area
                value=0,           # Value to fill the erased area (e.g., 0 for black)
                inplace=False      # Whether to modify the image in-place
            )
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    random_erasing
                ])
        elif aug_mode == "crop-flip-trivial":
            trivial_augment = TrivialAugmentWide()
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    trivial_augment,  
                    transforms.ToTensor(),
                    
                ])
        elif aug_mode == "crop-flip-augmix":
            
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.AugMix(),  
                    transforms.ToTensor(),
                    
                ])

        else:
            print("Invalid Augmentation")


    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")
    train_set = CIFAR100(data_dir, train=True, transform=train_transform, download=True)

    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)
    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        if num_indexes_to_replace is None:
            test_set.data = test_set.data[test_set.targets != class_to_replace]
            test_set.targets = test_set.targets[test_set.targets != class_to_replace]
    if indexes_to_replace is not None or indexes_to_replace == 450:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader


def cifar100_dataloaders_no_val(
    batch_size=128, data_dir="datasets/cifar100", num_workers=2
):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = CIFAR100(data_dir, train=True, transform=train_transform, download=True)
    val_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


class TinyImageNetDataset(Dataset):
    def __init__(self, image_folder_set, norm_trans=None, start=0, end=-1):
        self.imgs = []
        self.targets = []
        self.transform = image_folder_set.transform
        for sample in tqdm(image_folder_set.imgs[start:end]):
            self.targets.append(sample[1])
            img = transforms.ToTensor()(Image.open(sample[0]).convert("RGB"))
            if norm_trans is not None:
                img = norm_trans(img)
            self.imgs.append(img)
        self.imgs = torch.stack(self.imgs)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.imgs[idx]), self.targets[idx]
        else:
            return self.imgs[idx], self.targets[idx]


class TinyImageNet:
    """
    TinyImageNet dataset.
    """

    def __init__(self, args, normalize=False):
        self.args = args

        self.norm_layer = (
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if normalize
            else None
        )

        self.tr_train = [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        self.tr_test = []

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

        self.train_path = os.path.join(args.data_dir, "train/")
        self.val_path = os.path.join(args.data_dir, "val/")
        self.test_path = os.path.join(args.data_dir, "test/")

        if os.path.exists(os.path.join(self.val_path, "images")):
            if os.path.exists(self.test_path):
                os.rename(self.test_path, os.path.join(args.data_dir, "test_original"))
                os.mkdir(self.test_path)
            val_dict = {}
            val_anno_path = os.path.join(self.val_path, "val_annotations.txt")
            with open(val_anno_path, "r") as f:
                for line in f.readlines():
                    split_line = line.split("\t")
                    val_dict[split_line[0]] = split_line[1]

            paths = glob.glob(os.path.join(args.data_dir, "val/images/*"))
            for path in paths:
                file = path.split("/")[-1]
                folder = val_dict[file]
                if not os.path.exists(self.val_path + str(folder)):
                    os.mkdir(self.val_path + str(folder))
                    os.mkdir(self.val_path + str(folder) + "/images")
                if not os.path.exists(self.test_path + str(folder)):
                    os.mkdir(self.test_path + str(folder))
                    os.mkdir(self.test_path + str(folder) + "/images")

            for path in paths:
                file = path.split("/")[-1]
                folder = val_dict[file]
                if len(glob.glob(self.val_path + str(folder) + "/images/*")) < 25:
                    dest = self.val_path + str(folder) + "/images/" + str(file)
                else:
                    dest = self.test_path + str(folder) + "/images/" + str(file)
                move(path, dest)

            os.rmdir(os.path.join(self.val_path, "images"))

    def data_loaders(
        self,
        batch_size=128,
        data_dir="datasets/tiny",
        num_workers=2,
        class_to_replace: int = None,
        num_indexes_to_replace=None,
        indexes_to_replace=None,
        seed: int = 1,
        only_mark: bool = False,
        shuffle=True,
        no_aug=False,
    ):
        train_set = ImageFolder(self.train_path, transform=self.tr_train)
        train_set = TinyImageNetDataset(train_set, self.norm_layer)
        test_set = ImageFolder(self.test_path, transform=self.tr_test)
        test_set = TinyImageNetDataset(test_set, self.norm_layer)
        train_set.targets = np.array(train_set.targets)
        train_set.targets = np.array(train_set.targets)
        rng = np.random.RandomState(seed)
        valid_set = copy.deepcopy(train_set)
        valid_idx = []
        for i in range(max(train_set.targets) + 1):
            class_idx = np.where(train_set.targets == i)[0]
            valid_idx.append(
                rng.choice(class_idx, int(0.0 * len(class_idx)), replace=False)
            )
        valid_idx = np.hstack(valid_idx)
        train_set_copy = copy.deepcopy(train_set)

        valid_set.imgs = train_set_copy.imgs[valid_idx]
        valid_set.targets = train_set_copy.targets[valid_idx]

        train_idx = list(set(range(len(train_set))) - set(valid_idx))

        train_set.imgs = train_set_copy.imgs[train_idx]
        train_set.targets = train_set_copy.targets[train_idx]

        if class_to_replace is not None and indexes_to_replace is not None:
            raise ValueError(
                "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
            )
        if class_to_replace is not None:
            replace_class(
                train_set,
                class_to_replace,
                num_indexes_to_replace=num_indexes_to_replace,
                seed=seed - 1,
                only_mark=only_mark,
            )
            if num_indexes_to_replace is None or num_indexes_to_replace == 500:
                test_set.targets = np.array(test_set.targets)
                test_set.imgs = test_set.imgs[test_set.targets != class_to_replace]
                test_set.targets = test_set.targets[
                    test_set.targets != class_to_replace
                ]
                test_set.targets = test_set.targets.tolist()
        if indexes_to_replace is not None:
            replace_indexes(
                dataset=train_set,
                indexes=indexes_to_replace,
                seed=seed - 1,
                only_mark=only_mark,
            )

        loader_args = {"num_workers": 0, "pin_memory": False}

        def _init_fn(worker_id):
            np.random.seed(int(seed))

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        val_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, val_loader, test_loader


def cifar10_dataloaders(
    batch_size=128,
    data_dir="datasets/cifar10",
    num_workers=2,
    random_to_replace: int = None,
    class_to_replace: int = None,
    num_indexes_to_replace=None,
    indexes_to_replace=None,
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
    no_aug=False,
    aug_mode=None
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        if aug_mode == "crop-flip" or aug_mode==None:
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        elif aug_mode == "crop-flip-randaug":
            rand_augment = transforms.RandAugment(num_ops=2, magnitude=9)
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    rand_augment,
                    transforms.ToTensor(),
                ]
            )
            
        elif aug_mode == "crop-flip-autoaug":
            auto_augment = AutoAugment(policy=AutoAugmentPolicy.CIFAR10)
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    auto_augment, 
                    transforms.ToTensor(),
                ]
            )
        elif aug_mode == "crop-flip-rerase":
            random_erasing = transforms.RandomErasing(
                p=0.5,             # Probability of applying Random Erasing
                scale=(0.02, 0.33), # Range of proportion of erased area relative to image
                ratio=(0.3, 3.3),  # Aspect ratio of erased area
                value=0,           # Value to fill the erased area (e.g., 0 for black)
                inplace=False      # Whether to modify the image in-place
            )
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    random_erasing
                ])
        elif aug_mode == "crop-flip-trivial":
            trivial_augment = TrivialAugmentWide()
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    trivial_augment,  
                    transforms.ToTensor(),
                    
                ])
        elif aug_mode == "crop-flip-augmix":
            
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.AugMix(),  
                    transforms.ToTensor(),
                    
                ])
        else:
            print("Invalid Augmentation")
            print(aug_mode)

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-10\t 45000 images for training \t 5000 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)

    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        if num_indexes_to_replace is None or num_indexes_to_replace == 4500:
            test_set.data = test_set.data[test_set.targets != class_to_replace]
            test_set.targets = test_set.targets[test_set.targets != class_to_replace]
    if indexes_to_replace is not None:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader


def medmnist_dataloaders(
    batch_size=128,
    #data_dir="datasets/medmnist",
    data_dir="/home/pesquisador/pesquisa/datasets",
    num_workers=2,
    random_to_replace: int = None,
    class_to_replace: int = None,
    num_indexes_to_replace=None,
    indexes_to_replace=None,
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
    no_aug=False,
    aug_mode=None,
    im_size=64,
    dataset=None
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        if aug_mode == "crop-flip" or aug_mode==None:
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(im_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        elif aug_mode == "crop-flip-randaug":
            rand_augment = transforms.RandAugment(num_ops=2, magnitude=9)
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(im_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    rand_augment,
                    transforms.ToTensor(),
                ]
            )
        elif aug_mode == "crop-flip-autoaug":
            auto_augment = AutoAugment(policy=AutoAugmentPolicy.CIFAR10)
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    auto_augment, 
                    transforms.ToTensor(),
                ]
            )
        elif aug_mode == "crop-flip-rerase":
            random_erasing = transforms.RandomErasing(
                p=0.5,             # Probability of applying Random Erasing
                scale=(0.02, 0.33), # Range of proportion of erased area relative to image
                ratio=(0.3, 3.3),  # Aspect ratio of erased area
                value=0,           # Value to fill the erased area (e.g., 0 for black)
                inplace=False      # Whether to modify the image in-place
            )
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    random_erasing
                ])
        elif aug_mode == "crop-flip-trivial":
            trivial_augment = TrivialAugmentWide()
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    trivial_augment,  
                    transforms.ToTensor(),
                    
                ])
        elif aug_mode == "crop-flip-augmix":
            
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.AugMix(),  
                    transforms.ToTensor(),
                    
                ])
        else:
            print("Invalid Augmentation")
            print(aug_mode)

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )


    if dataset == "bloodmnist":
        train_set = BloodMNIST( split='train',  root=data_dir,download=True, size=im_size, transform=train_transform)

        valid_set = BloodMNIST( split='val',  root=data_dir,download=True, size=im_size, transform=train_transform)

        test_set = BloodMNIST( split='test',  root=data_dir, transform=test_transform, download=True,  size=im_size)
    
    elif dataset == "pathmnist":
        train_set = PathMNIST( split='train',  root=data_dir,download=True, size=im_size, transform=train_transform)
        valid_set = PathMNIST( split='val',  root=data_dir,download=True, size=im_size, transform=train_transform)

        test_set = PathMNIST( split='test',  root=data_dir, transform=test_transform, download=True,  size=im_size)
 
    
    elif dataset == "organamnist":
        train_set = OrganAMNIST( split='train',  root=data_dir,download=True, size=im_size, transform=train_transform)
        valid_set = OrganAMNIST( split='val',  root=data_dir,download=True, size=im_size, transform=train_transform)

        test_set = OrganAMNIST( split='test',  root=data_dir, transform=test_transform, download=True,  size=im_size)
    
    elif dataset == "octmnist":
        train_set = OCTMNIST( split='train',  root=data_dir,download=True, size=im_size, transform=train_transform)
        valid_set = OCTMNIST( split='val',  root=data_dir,download=True, size=im_size, transform=train_transform)

        test_set = OCTMNIST( split='test',  root=data_dir, transform=test_transform, download=True,  size=im_size)

    else:
        
        raise ValueError("Dataset not supprot yet !")

    # train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)
    

    # import pdb; pdb.set_trace()

    #train_set.targets = np.array(train_set.targets)
    train_set.labels = np.array(train_set.labels).squeeze()
    train_set.labels = train_set.labels.astype('int32')
    #test_set.targets = np.array(test_set.targets)
    test_set.labels = np.array(test_set.labels).squeeze()
    test_set.labels = test_set.labels.astype('int32')

    # rng = np.random.RandomState(seed)
    # valid_set = copy.deepcopy(train_set)
    # valid_idx = []
    # for i in range(max(train_set.targets) + 1):
    #     class_idx = np.where(train_set.targets == i)[0]
    #     valid_idx.append(
    #         rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
    #     )
    # valid_idx = np.hstack(valid_idx)
    # train_set_copy = copy.deepcopy(train_set)

    # valid_set.data = train_set_copy.data[valid_idx]
    # valid_set.targets = train_set_copy.targets[valid_idx]
    valid_set.labels = np.array(valid_set.labels).squeeze()
    valid_set.labels = valid_set.labels.astype('int32')

    # train_idx = list(set(range(len(train_set))) - set(valid_idx))

    # train_set.data = train_set_copy.data[train_idx]
    # train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    
    if class_to_replace is not None:
        
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        #if num_indexes_to_replace is None or num_indexes_to_replace == 4500:
        if num_indexes_to_replace is None :
            #test_set.data = test_set.data[test_set.targets != class_to_replace]
            
            test_set.imgs = test_set.imgs[test_set.labels != class_to_replace]
            test_set.labels = test_set.labels[test_set.labels != class_to_replace]
    if indexes_to_replace is not None:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader


def dermamnist_bin_dataloaders(
    batch_size=128,
    #data_dir="datasets/medmnist",
    data_dir="/home/pesquisador/pesquisa/datasets",
    num_workers=2,
    random_to_replace: int = None,
    class_to_replace: int = None,
    num_indexes_to_replace=None,
    indexes_to_replace=None,
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
    no_aug=False,
    aug_mode=None,
    im_size=64,
    dataset=None,
    removal_mode: str = "random",  # random | balanced | skewed
    skew_malignant_frac: float = 0.7,
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        if aug_mode == "crop-flip" or aug_mode==None:
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(im_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        elif aug_mode == "crop-flip-randaug":
            rand_augment = transforms.RandAugment(num_ops=2, magnitude=9)
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(im_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    rand_augment,
                    transforms.ToTensor(),
                ]
            )
        elif aug_mode == "crop-flip-autoaug":
            auto_augment = AutoAugment(policy=AutoAugmentPolicy.CIFAR10)
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    auto_augment, 
                    transforms.ToTensor(),
                ]
            )
        elif aug_mode == "crop-flip-rerase":
            random_erasing = transforms.RandomErasing(
                p=0.5,             # Probability of applying Random Erasing
                scale=(0.02, 0.33), # Range of proportion of erased area relative to image
                ratio=(0.3, 3.3),  # Aspect ratio of erased area
                value=0,           # Value to fill the erased area (e.g., 0 for black)
                inplace=False      # Whether to modify the image in-place
            )
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    random_erasing
                ])
        elif aug_mode == "crop-flip-trivial":
            trivial_augment = TrivialAugmentWide()
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    trivial_augment,  
                    transforms.ToTensor(),
                    
                ])
        elif aug_mode == "crop-flip-augmix":
            
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.AugMix(),  
                    transforms.ToTensor(),
                    
                ])
        else:
            print("Invalid Augmentation")
            print(aug_mode)

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )


    
    train_set = DermaMNIST( split='train',  root=data_dir,download=True, size=im_size, transform=train_transform)

    valid_set = DermaMNIST( split='val',  root=data_dir,download=True, size=im_size, transform=train_transform)

    test_set = DermaMNIST( split='test',  root=data_dir, transform=test_transform, download=True,  size=im_size)
    
   

    # train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)
    

    # import pdb; pdb.set_trace()
    MALIGNANT_CLASSES = [0, 1, 4]
    BENIGN_CLASSES = [2, 3, 5, 6]

    #train_set.targets = np.array(train_set.targets)
    train_set.labels = np.array(train_set.labels).squeeze()
    train_set.labels = train_set.labels.astype('int32')
    #test_set.targets = np.array(test_set.targets)
    test_set.labels = np.array(test_set.labels).squeeze()
    test_set.labels = test_set.labels.astype('int32')

    valid_set.labels = np.array(valid_set.labels).squeeze()
    valid_set.labels = valid_set.labels.astype('int32')

    train_set.original_labels = train_set.labels.copy()
    valid_set.original_labels = valid_set.labels.copy()
    test_set.original_labels = test_set.labels.copy()

    train_set.labels =  np.isin(train_set.labels, MALIGNANT_CLASSES).astype(np.int32)
    valid_set.labels =  np.isin(valid_set.labels, MALIGNANT_CLASSES).astype(np.int32)
    test_set.labels =  np.isin(test_set.labels, MALIGNANT_CLASSES).astype(np.int32)

    # Log class distribution
    print("=" * 50)
    print("DermaMNIST Binary - Class Distribution")
    print("=" * 50)
    for name, ds in [("Train", train_set), ("Val", valid_set), ("Test", test_set)]:
        n_malignant = (ds.labels == 1).sum()
        n_benign = (ds.labels == 0).sum()
        total = len(ds.labels)
        print(f"  {name}: Malignant={n_malignant} ({100*n_malignant/total:.1f}%) | "
              f"Benign={n_benign} ({100*n_benign/total:.1f}%) | Total={total}")
    print("=" * 50)


    # rng = np.random.RandomState(seed)
    # valid_set = copy.deepcopy(train_set)
    # valid_idx = []
    # for i in range(max(train_set.targets) + 1):
    #     class_idx = np.where(train_set.targets == i)[0]
    #     valid_idx.append(
    #         rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
    #     )
    # valid_idx = np.hstack(valid_idx)
    # train_set_copy = copy.deepcopy(train_set)

    # valid_set.data = train_set_copy.data[valid_idx]
    # valid_set.targets = train_set_copy.targets[valid_idx]
    

    # train_idx = list(set(range(len(train_set))) - set(valid_idx))

    # train_set.data = train_set_copy.data[train_idx]
    # train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    
    if class_to_replace is not None:
        
        # replace_class(
        #     train_set,
        #     class_to_replace,
        #     num_indexes_to_replace=num_indexes_to_replace,
        #     seed=seed - 1,
        #     only_mark=only_mark,
        # )
        # For MU experiments we typically mark samples (only_mark=True) instead of replacing.
        # When class_to_replace == -1, the original code marks a random subset of the whole train set.
        # Here we support class-balanced and skewed removal policies for the binary task.
        if class_to_replace == -1 and only_mark and num_indexes_to_replace is not None:
            labels_arr = np.array(train_set.labels).astype(np.int64)
            idx_to_mark = select_removal_indexes_binary(
                labels=labels_arr,
                total_to_remove=int(num_indexes_to_replace),
                mode=str(removal_mode),
                skew_malignant_frac=float(skew_malignant_frac),
                seed=int(seed - 1),
            )
            print(
                f"[Removal] mode={removal_mode} total={len(idx_to_mark)} "
                f"(mal={int((labels_arr[idx_to_mark]==1).sum())}, ben={int((labels_arr[idx_to_mark]==0).sum())})"
            )
            mark_indexes_only(train_set, idx_to_mark)
        else:
            # Fallback to original behavior (replace or mark a specific class)
            replace_class(
                train_set,
                class_to_replace,
                num_indexes_to_replace=num_indexes_to_replace,
                seed=seed - 1,
                only_mark=only_mark,
            )
        #if num_indexes_to_replace is None or num_indexes_to_replace == 4500:
        if num_indexes_to_replace is None :
            #test_set.data = test_set.data[test_set.targets != class_to_replace]
            
            test_set.imgs = test_set.imgs[test_set.labels != class_to_replace]
            test_set.original_labels = test_set.original_labels[test_set.labels != class_to_replace]
            test_set.labels = test_set.labels[test_set.labels != class_to_replace]
    if indexes_to_replace is not None:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader

def pneumonia_dataloaders(
    batch_size=128,
    #data_dir="datasets/medmnist",
    data_dir="/home/pesquisador/pesquisa/datasets",
    num_workers=2,
    random_to_replace: int = None,
    class_to_replace: int = None,
    num_indexes_to_replace=None,
    indexes_to_replace=None,
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
    no_aug=False,
    aug_mode=None,
    im_size=64,
    dataset=None,
    removal_mode: str = "random",  # random | balanced | skewed
    skew_malignant_frac: float = 0.7,
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        if aug_mode == "crop-flip" or aug_mode==None:
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(im_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        elif aug_mode == "crop-flip-randaug":
            rand_augment = transforms.RandAugment(num_ops=2, magnitude=9)
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(im_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    rand_augment,
                    transforms.ToTensor(),
                ]
            )
        elif aug_mode == "crop-flip-autoaug":
            auto_augment = AutoAugment(policy=AutoAugmentPolicy.CIFAR10)
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    auto_augment, 
                    transforms.ToTensor(),
                ]
            )
        elif aug_mode == "crop-flip-rerase":
            random_erasing = transforms.RandomErasing(
                p=0.5,             # Probability of applying Random Erasing
                scale=(0.02, 0.33), # Range of proportion of erased area relative to image
                ratio=(0.3, 3.3),  # Aspect ratio of erased area
                value=0,           # Value to fill the erased area (e.g., 0 for black)
                inplace=False      # Whether to modify the image in-place
            )
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    random_erasing
                ])
        elif aug_mode == "crop-flip-trivial":
            trivial_augment = TrivialAugmentWide()
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    trivial_augment,  
                    transforms.ToTensor(),
                    
                ])
        elif aug_mode == "crop-flip-augmix":
            
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.AugMix(),  
                    transforms.ToTensor(),
                    
                ])
        else:
            print("Invalid Augmentation")
            print(aug_mode)

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )


    
    train_set = PneumoniaMNIST( split='train',  root=data_dir,download=True, size=im_size, transform=train_transform)

    valid_set = PneumoniaMNIST( split='val',  root=data_dir,download=True, size=im_size, transform=train_transform)

    test_set = PneumoniaMNIST( split='test',  root=data_dir, transform=test_transform, download=True,  size=im_size)
    
   

    

    #train_set.targets = np.array(train_set.targets)
    train_set.labels = np.array(train_set.labels).squeeze()
    train_set.labels = train_set.labels.astype('int32')
    #test_set.targets = np.array(test_set.targets)
    test_set.labels = np.array(test_set.labels).squeeze()
    test_set.labels = test_set.labels.astype('int32')

    valid_set.labels = np.array(valid_set.labels).squeeze()
    valid_set.labels = valid_set.labels.astype('int32')

    # train_set.original_labels = train_set.labels.copy()
    # valid_set.original_labels = valid_set.labels.copy()
    # test_set.original_labels = test_set.labels.copy()

    # train_set.labels =  np.isin(train_set.labels, MALIGNANT_CLASSES).astype(np.int32)
    # valid_set.labels =  np.isin(valid_set.labels, MALIGNANT_CLASSES).astype(np.int32)
    # test_set.labels =  np.isin(test_set.labels, MALIGNANT_CLASSES).astype(np.int32)

    # Log class distribution
    print("=" * 50)
    print("PneumoMNIST Binary - Class Distribution")
    print("=" * 50)
    for name, ds in [("Train", train_set), ("Val", valid_set), ("Test", test_set)]:
        n_malignant = (ds.labels == 1).sum()
        n_benign = (ds.labels == 0).sum()
        total = len(ds.labels)
        print(f"  {name}: Malignant={n_malignant} ({100*n_malignant/total:.1f}%) | "
              f"Benign={n_benign} ({100*n_benign/total:.1f}%) | Total={total}")
    print("=" * 50)


    # rng = np.random.RandomState(seed)
    # valid_set = copy.deepcopy(train_set)
    # valid_idx = []
    # for i in range(max(train_set.targets) + 1):
    #     class_idx = np.where(train_set.targets == i)[0]
    #     valid_idx.append(
    #         rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
    #     )
    # valid_idx = np.hstack(valid_idx)
    # train_set_copy = copy.deepcopy(train_set)

    # valid_set.data = train_set_copy.data[valid_idx]
    # valid_set.targets = train_set_copy.targets[valid_idx]
    

    # train_idx = list(set(range(len(train_set))) - set(valid_idx))

    # train_set.data = train_set_copy.data[train_idx]
    # train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    
    if class_to_replace is not None:
        
        # replace_class(
        #     train_set,
        #     class_to_replace,
        #     num_indexes_to_replace=num_indexes_to_replace,
        #     seed=seed - 1,
        #     only_mark=only_mark,
        # )
        # For MU experiments we typically mark samples (only_mark=True) instead of replacing.
        # When class_to_replace == -1, the original code marks a random subset of the whole train set.
        # Here we support class-balanced and skewed removal policies for the binary task.
        if class_to_replace == -1 and only_mark and num_indexes_to_replace is not None:
            labels_arr = np.array(train_set.labels).astype(np.int64)
            idx_to_mark = select_removal_indexes_binary(
                labels=labels_arr,
                total_to_remove=int(num_indexes_to_replace),
                mode=str(removal_mode),
                skew_malignant_frac=float(skew_malignant_frac),
                seed=int(seed - 1),
            )
            print(
                f"[Removal] mode={removal_mode} total={len(idx_to_mark)} "
                f"(mal={int((labels_arr[idx_to_mark]==1).sum())}, ben={int((labels_arr[idx_to_mark]==0).sum())})"
            )
            mark_indexes_only(train_set, idx_to_mark)
        else:
            # Fallback to original behavior (replace or mark a specific class)
            replace_class(
                train_set,
                class_to_replace,
                num_indexes_to_replace=num_indexes_to_replace,
                seed=seed - 1,
                only_mark=only_mark,
            )
        #if num_indexes_to_replace is None or num_indexes_to_replace == 4500:
        if num_indexes_to_replace is None :
            #test_set.data = test_set.data[test_set.targets != class_to_replace]
            
            test_set.imgs = test_set.imgs[test_set.labels != class_to_replace]
            test_set.original_labels = test_set.original_labels[test_set.labels != class_to_replace]
            test_set.labels = test_set.labels[test_set.labels != class_to_replace]
    if indexes_to_replace is not None:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader


def replace_indexes(
    dataset: torch.utils.data.Dataset, indexes, seed=0, only_mark: bool = False
):
    
    if not only_mark:
        rng = np.random.RandomState(seed)
        new_indexes = rng.choice(
            list(set(range(len(dataset))) - set(indexes)), size=len(indexes)
        )
        dataset.data[indexes] = dataset.data[new_indexes]
        try:
            dataset.targets[indexes] = dataset.targets[new_indexes]
        except:
            dataset.labels[indexes] = dataset.labels[new_indexes]
        else:
            dataset._labels[indexes] = dataset._labels[new_indexes]
    else:
        # Notice the -1 to make class 0 work
        try:
            dataset.targets[indexes] = -dataset.targets[indexes] - 1
        except:
            try:
                dataset.labels[indexes] = -dataset.labels[indexes] - 1
            except:
                dataset._labels[indexes] = -dataset._labels[indexes] - 1


def replace_class(
    dataset: torch.utils.data.Dataset,
    class_to_replace: int,
    num_indexes_to_replace: int = None,
    seed: int = 0,
    only_mark: bool = False,
):
    if class_to_replace == -1:
        try:
            indexes = np.flatnonzero(np.ones_like(dataset.targets))
        except:
            try:
                indexes = np.flatnonzero(np.ones_like(dataset.labels))
            except:
                indexes = np.flatnonzero(np.ones_like(dataset._labels))
    else:
        try:
            indexes = np.flatnonzero(np.array(dataset.targets) == class_to_replace)
        except:
            try:
                indexes = np.flatnonzero(np.array(dataset.labels) == class_to_replace)
            except:
                indexes = np.flatnonzero(np.array(dataset._labels) == class_to_replace)

    if num_indexes_to_replace is not None:
        assert num_indexes_to_replace <= len(
            indexes
        ), f"Want to replace {num_indexes_to_replace} indexes but only {len(indexes)} samples in dataset"
        rng = np.random.RandomState(seed)
        indexes = rng.choice(indexes, size=num_indexes_to_replace, replace=False)
        print(f"Replacing indexes {indexes}")
    
    replace_indexes(dataset, indexes, seed, only_mark)




if __name__ == "__main__":
    train_loader, val_loader, test_loader = cifar10_dataloaders()
    for i, (img, label) in enumerate(train_loader):
        print(torch.unique(label).shape)
