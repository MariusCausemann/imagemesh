"""Tests for imagemesh.handles (handle removal)."""

import nbmorph
import numpy as np

from imagemesh.handles import components_per_label, fill_handles, handle_counts


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
