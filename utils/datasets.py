# =====================================
# Download STL10 and TrashNet datasets
# =====================================

import os
import random
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def _split_dataset(
    raw_dir,
    train_dir,
    val_dir,
    test_dir,
    train_ratio=0.7,
    val_ratio=0.15
):

    if all(os.path.exists(d) and len(os.listdir(d)) > 0
           for d in [train_dir, val_dir, test_dir]):
        return

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    classes = os.listdir(raw_dir)

    for cls in classes:
        cls_path = os.path.join(raw_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

        images = os.listdir(cls_path)
        random.shuffle(images)

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


def get_stl10_train(batch_size=32, shuffle=True):
    """ return training dataloader
    Args:
        batch_size: dataloader batchsize
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),  # ⭐ 替换 CenterCrop
        transforms.RandomHorizontalFlip(),  # ⭐ 加翻转
        transforms.ColorJitter(  # ⭐ 加颜色扰动
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    full_train = datasets.STL10(
        root='./data', split='train', download=True, transform=None
    )

    # train / val split
    total_len = len(full_train)  # 5000
    val_len = int(total_len * 0.2)
    train_len = total_len - val_len

    train_dataset, val_dataset = random_split(full_train, [train_len, val_len])

    # Set transform to train and val datasets
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def get_stl10_test(batch_size=32, shuffle=True):
    """ return test dataloader
    Args:
        batch_size: dataloader batchsize
        shuffle: whether to shuffle
    Returns: stl10_test_loader:torch dataloader object
    """

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    stl10_test = datasets.STL10(
        root='./data', split='test', download=True, transform=test_transform)
    stl10_test_loader = DataLoader(
        stl10_test,  batch_size=batch_size, shuffle=shuffle)

    return stl10_test_loader

