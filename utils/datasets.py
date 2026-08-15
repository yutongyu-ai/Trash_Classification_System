# =====================================
# TrashNet dataset loading utilities
# =====================================

import os
import random
import shutil
from collections import Counter
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def _split_dataset(
    raw_dir,
    train_dir,
    val_dir,
    test_dir,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42
):

    if all(os.path.exists(d) and len(os.listdir(d)) > 0
           for d in [train_dir, val_dir, test_dir]):
        return

    # Wipe any partial split before regenerating, so a previous incomplete
    # run can't leave stale files that end up duplicated across train/val/test.
    for d in [train_dir, val_dir, test_dir]:
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    rng = random.Random(seed)
    classes = os.listdir(raw_dir)

    for cls in classes:
        cls_path = os.path.join(raw_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

        images = os.listdir(cls_path)
        rng.shuffle(images)

        n = len(images)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        for img in train_imgs:
            shutil.copy(os.path.join(cls_path, img),
                        os.path.join(train_dir, cls, img))

        for img in val_imgs:
            shutil.copy(os.path.join(cls_path, img),
                        os.path.join(val_dir, cls, img))

        for img in test_imgs:
            shutil.copy(os.path.join(cls_path, img),
                        os.path.join(test_dir, cls, img))


def get_trashnet_train(
    data_root="data",
    batch_size=32,
    num_workers=2,
    image_size=224
):
    raw_dir = os.path.join(data_root, "trashnet", "dataset-resized")
    train_dir = os.path.join(data_root, "trashnet", "train")
    val_dir = os.path.join(data_root, "trashnet", "val")
    test_dir = os.path.join(data_root, "trashnet", "test")

    _split_dataset(raw_dir, train_dir, val_dir, test_dir)

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader


def get_trashnet_test(
    data_root="data",
    batch_size=32,
    num_workers=2,
    image_size=224
):
    test_dir = os.path.join(data_root, "trashnet", "test")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return test_loader


def get_class_weights(dataset):
    """
    Inverse-frequency class weights for an ImageFolder-style dataset, for
    use with nn.CrossEntropyLoss(weight=...) on imbalanced data (TrashNet's
    "trash" class has ~3.6x fewer images than "paper").

    weight_i = N / (K * n_i), so a perfectly balanced dataset yields all
    weights == 1.
    """
    counts = Counter(dataset.targets)
    num_classes = len(dataset.classes)
    total = len(dataset.targets)

    weights = torch.zeros(num_classes)
    for cls_idx in range(num_classes):
        count = counts.get(cls_idx, 0)
        weights[cls_idx] = total / (num_classes * count) if count > 0 else 0.0

    return weights

