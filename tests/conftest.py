"""Shared fixtures: small synthetic label images."""

import numpy as np
import pytest


@pytest.fixture
def sample_image_data():
    """50^3 label image with three boxes."""
    data = np.zeros((50, 50, 50), dtype=np.uint32)
    data[10:20, 10:20, 10:20] = 1
    data[30:40, 30:40, 30:40] = 2
    data[15:25, 35:45, 15:25] = 3
    return data


@pytest.fixture
def sample_resolution():
    return [18.0, 18.0, 18.0]
