"""Tests for imagemesh.smoothing."""

import numpy as np

from imagemesh.smoothing import edge_table, unique_faces


def test_edge_table_matches_numpy():
    rng = np.random.default_rng(0)
    faces = np.array([rng.choice(30, 4, replace=False) for _ in range(200)])
    fval = rng.integers(0, 5, len(faces))
    e = np.sort(np.stack([faces, np.roll(faces, -1, axis=1)], -1).reshape(-1, 2), axis=1)
    key = e[:, 0] * 30 + e[:, 1]
    ukey, inv, valence = np.unique(key, return_inverse=True, return_counts=True)
    lo = np.full(len(ukey), 99)
    hi = np.full(len(ukey), -1)
    np.minimum.at(lo, inv, np.repeat(fval, 4))
    np.maximum.at(hi, inv, np.repeat(fval, 4))
    for got, want in zip(edge_table(faces, 30, fval), (ukey // 30, ukey % 30, valence, lo, hi)):
        np.testing.assert_array_equal(got, want)


def test_unique_faces_keeps_first():
    faces = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [3, 2, 1, 0], [1, 2, 3, 0], [0, 1, 2, 4]])
    np.testing.assert_array_equal(unique_faces(faces), faces[[0, 1, 4]])
