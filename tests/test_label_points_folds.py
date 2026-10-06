"""Tests for imagemesh.winding_number.label_points on folded multi-label
surfaces (regions overlapping or turned inside out), where the per-bit
classification can combine the bits of several regions."""

import numpy as np
import pyvista as pv

from imagemesh.winding_number import label_points


def _box_soup(boxes):
    """Triangle soup of axis-aligned boxes, given as (bounds, label, inverted),
    with boundary_labels (label, 0): the outward normals point from the label
    to the background. An inverted box has its faces flipped, so its winding
    number is -1 inside, as in a region a folded surface turns inside out."""
    V, F, L = [], [], []
    n = 0
    for bounds, label, inverted in boxes:
        box = pv.Box(bounds).triangulate()
        f = box.regular_faces
        if inverted:
            f = f[:, [0, 2, 1]]
        V.append(np.array(box.points))
        F.append(f + n)
        L.append(np.tile([label, 0], (len(f), 1)))
        n += box.n_points
    return np.vstack(V), np.vstack(F), np.vstack(L)


def test_label_points_overlap_does_not_mix_labels():
    """Where two regions overlap (a fold of the surface), the bits of their
    indices must not be combined into the index of an unrelated region: 5 and
    7 have the indices 1 and 2, whose bits make 3, the index of 9."""
    V, F, L = _box_soup(
        [
            ((0, 2, 0, 2, 0, 2), 5, False),
            ((1, 3, 0, 2, 0, 2), 7, False),
            ((5, 6, 5, 6, 5, 6), 9, False),
        ]
    )
    Q = np.array([[1.4, 1.0, 1.0], [0.5, 1.0, 1.0], [2.5, 1.0, 1.0], [5.5, 5.5, 5.5], [4, 4, 4]])
    labels = label_points(V, F, L, Q)
    assert labels[0] in (5, 7)
    np.testing.assert_array_equal(labels[1:], [5, 7, 9, 0])


def test_label_points_inverted_fold_does_not_mix_labels():
    """A point inside the regions 10 and 60 and an inverted 20 (winding
    numbers +1, +1, -1, so it is inside the domain exactly once) has the bit
    sums of index 1 + 6 - 2 = 5: all 0 or 1, the index of 50, which is
    elsewhere. It must get one of the regions it is inside."""
    V, F, L = _box_soup(
        [
            ((0, 2, 0, 2, 0, 2), 10, False),
            ((1, 3, 0, 2, 0, 2), 60, False),
            ((0.5, 2.5, 0.5, 1.5, 0.5, 1.5), 20, True),
            ((10, 11, 0, 1, 0, 1), 30, False),
            ((12, 13, 0, 1, 0, 1), 40, False),
            ((14, 15, 0, 1, 0, 1), 50, False),
        ]
    )
    Q = np.array([[1.4, 1.0, 1.0], [14.5, 0.5, 0.5], [0.2, 0.2, 0.2], [2.8, 0.2, 0.2]])
    labels = label_points(V, F, L, Q)
    assert labels[0] in (10, 60)
    np.testing.assert_array_equal(labels[1:], [50, 10, 60])
