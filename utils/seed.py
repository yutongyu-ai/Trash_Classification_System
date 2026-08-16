import random

import numpy as np
import torch


def set_seed(seed=42):
    """Seed python/numpy/torch RNGs and force cuDNN determinism, so a given
    seed reproduces the same model init, data shuffling, and augmentation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
