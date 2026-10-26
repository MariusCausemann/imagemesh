"""Tests for imagemesh.handles (handle removal)."""

import nbmorph
import numpy as np

from imagemesh.handles import (
    components_per_label,
    fill_handles,
    handle_counts,
    skeleton_loops,
    tighten_loop,
)


def ring(shape=(40, 40, 12)):
    """A square ring of label 1 (one handle) around a 12 x 12 hole along z."""
    img = np.zeros(shape, dtype=np.uint16)
    img[8:32, 8:32, 3:9] = 1
    img[14:26, 14:26, 3:9] = 0
    return img


def test_fill_handles_fills_the_hole():
    img = ring()
    assert handle_counts(img, 1)[1] == 1
    out = fill_handles(img)
    assert handle_counts(out, 1)[1] == 0
    assert np.all(out[img == 1] == 1)  # only background was filled
    assert (out == 1).sum() > (img == 1).sum()


def test_fill_handles_keeps_long_loops():
    img = ring()
    out = fill_handles(img, max_loop_length=40)
    np.testing.assert_array_equal(out, img)


def test_fill_handles_cuts_threaded_cell():
    """A rod (label 2) through the ring is cut, and the cells stay separated."""
    img = ring((40, 40, 24))
    img[8:32, 8:32, 3:9] = 0
    img[8:32, 8:32, 9:15] = 1
    img[14:26, 14:26, 9:15] = 0
    img[18:22, 18:22, 2:22] = 2
    out = fill_handles(img)
    assert handle_counts(out, 2)[1] == 0
    np.testing.assert_array_equal(nbmorph.separate_labels_box(out), out)
    assert components_per_label(out, 2)[2] == 2


def test_tighten_loop_hugs_the_hole():
    """In a sheet with a small hole, the skeleton loop lies far from the hole; the
    tightened loop runs around it, inside the sheet."""
    m = np.zeros((20, 90, 90), bool)
    m[8:13, 5:85, 5:85] = True
    y, x = np.mgrid[:90, :90]
    m[8:13][:, (y - 40) ** 2 + (x - 50) ** 2 <= 16] = False
    (loop,) = skeleton_loops(m)
    tight = tighten_loop(m, loop)
    assert m[tuple(tight.T)].all()
    assert np.all(np.max(np.abs(np.diff(np.vstack([tight, tight[:1]]), axis=0)), axis=1) == 1)
    d = np.hypot(tight[:, 1] - 40, tight[:, 2] - 50)
    assert d.max() < 7 < np.hypot(loop[:, 1] - 40, loop[:, 2] - 50).min()


def test_fill_handles_curved_sheet():
    """A small hole in a curved sheet is closed by a small membrane, not by a wall
    across the curvature of the sheet."""
    z, y, x = np.mgrid[:100, :100, :100]
    r = np.hypot(y - 70, x - 50)
    ang = np.degrees(np.arctan2(y - 70, x - 50))
    shell = (np.abs(r - 30) <= 2) & (ang < -10) & (ang > -170) & (z > 10) & (z < 90)
    hole = (z - 50) ** 2 + (x - 50) ** 2 <= 16
    img = (shell & ~(hole & (y < 70))).astype(np.uint16)
    out = fill_handles(img, fillet_radius=0)
    assert handle_counts(out, 1)[1] == 0
    added = (out == 1) & (img == 0)
    assert added.sum() < 2 * (hole & shell).sum()
    assert (added & ((z - 50) ** 2 + (x - 50) ** 2 > 36)).sum() < 0.1 * added.sum()
