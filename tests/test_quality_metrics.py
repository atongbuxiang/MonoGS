import math

import numpy as np

from utils.quality_metrics import compute_path_length


def test_compute_path_length_sums_adjacent_translation_distances():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
            [3.0, 4.0, 12.0],
        ],
        dtype=np.float64,
    )

    assert math.isclose(compute_path_length(positions), 17.0)


def test_compute_path_length_returns_zero_for_single_pose():
    positions = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)

    assert compute_path_length(positions) == 0.0
