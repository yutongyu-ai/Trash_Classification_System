import os

import pytest
import torch
from PIL import Image

from utils.datasets import _split_dataset, get_class_weights


class _FakeDataset:
    def __init__(self, targets, classes):
        self.targets = targets
        self.classes = classes


def test_balanced_classes_get_equal_weight():
    ds = _FakeDataset(targets=[0, 0, 1, 1], classes=["a", "b"])
    weights = get_class_weights(ds)
    assert torch.allclose(weights, torch.tensor([1.0, 1.0]))


def test_minority_class_gets_higher_weight():
    ds = _FakeDataset(targets=[0, 0, 0, 1], classes=["a", "b"])
    weights = get_class_weights(ds)
    assert weights[1] > weights[0]


def test_class_weights_length_matches_num_classes():
    ds = _FakeDataset(targets=[0, 1, 2, 2, 2], classes=["a", "b", "c"])
    weights = get_class_weights(ds)
    assert len(weights) == 3


def _make_fake_class_images(raw_dir, class_counts):
    for cls, count in class_counts.items():
        cls_dir = os.path.join(raw_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(count):
            Image.new("RGB", (8, 8), color=(i % 256, i % 256, i % 256)).save(
                os.path.join(cls_dir, f"{cls}_{i}.jpg")
            )


@pytest.fixture
def fake_raw_dataset(tmp_path):
    raw_dir = tmp_path / "raw"
    _make_fake_class_images(str(raw_dir), {"cardboard": 20, "glass": 20})
    return raw_dir


def test_split_dataset_respects_ratios(tmp_path, fake_raw_dataset):
    train_dir, val_dir, test_dir = (
        tmp_path / "train",
        tmp_path / "val",
        tmp_path / "test",
    )
    _split_dataset(
        str(fake_raw_dataset),
        str(train_dir),
        str(val_dir),
        str(test_dir),
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
    )
    n_train = len(os.listdir(train_dir / "cardboard"))
    n_val = len(os.listdir(val_dir / "cardboard"))
    n_test = len(os.listdir(test_dir / "cardboard"))
    assert n_train == 14
    assert n_val == 3
    assert n_train + n_val + n_test == 20


def test_split_dataset_is_reproducible_with_same_seed(tmp_path, fake_raw_dataset):
    dirs_a = [tmp_path / f"a_{s}" for s in ("train", "val", "test")]
    dirs_b = [tmp_path / f"b_{s}" for s in ("train", "val", "test")]
    _split_dataset(str(fake_raw_dataset), *map(str, dirs_a), seed=42)
    _split_dataset(str(fake_raw_dataset), *map(str, dirs_b), seed=42)

    assert sorted(os.listdir(dirs_a[0] / "cardboard")) == sorted(
        os.listdir(dirs_b[0] / "cardboard")
    )


def test_split_dataset_no_overlap_between_splits(tmp_path, fake_raw_dataset):
    train_dir, val_dir, test_dir = (
        tmp_path / "train",
        tmp_path / "val",
        tmp_path / "test",
    )
    _split_dataset(
        str(fake_raw_dataset), str(train_dir), str(val_dir), str(test_dir), seed=42
    )

    train_files = set(os.listdir(train_dir / "cardboard"))
    val_files = set(os.listdir(val_dir / "cardboard"))
    test_files = set(os.listdir(test_dir / "cardboard"))

    assert not (train_files & val_files)
    assert not (train_files & test_files)
    assert not (val_files & test_files)
