import random

import numpy as np

from src.utils.seed import seed_everything


def test_seed_everything_deterministic():
    seed_everything(42)
    val_py1 = random.random()
    val_np1 = np.random.rand()

    seed_everything(42)
    val_py2 = random.random()
    val_np2 = np.random.rand()

    assert val_py1 == val_py2
    assert val_np1 == val_np2


def test_seed_everything_different_seeds():
    seed_everything(42)
    val_np1 = np.random.rand()

    seed_everything(999)
    val_np2 = np.random.rand()

    assert val_np1 != val_np2
