"""Tests for imagemesh.morphology."""

from unittest.mock import patch

import numpy as np
import pytest
import scipy.ndimage as ndi

from imagemesh.morphology import (
    binary_fill_holes,
    dilate,
    erode,
    get_lowest_point,
    grow_into_region,
    mergecells,
    ncells,
    removeislands,
    separate_labels,
    smooth,
)


@pytest.fixture
def tiny_seg():
    """20^3 volume with two concentric spherical labels (1 outside 2)."""
    N = 20
    x = (np.indices((N, N, N)) + 0.5) / N
    dist = np.linalg.norm(x - 0.5, axis=0)
    data = np.zeros((N, N, N), dtype=np.uint8)
    data[dist < 0.45] = 1
    data[dist < 0.3] = 2
    return data


class TestImageOperations:
    """Test individual image processing operations."""

    def test_mergecells_basic(self):
        """Test basic cell merging."""
        img = np.array([[[1, 1, 2], [1, 3, 2], [4, 4, 4]]], dtype=np.uint32)
        labels = [1, 2]

        result = mergecells(img, labels)

        # All 1s and 2s should become the first label (1)
        non_zero_values = result[result > 0]
        unique_values = np.unique(non_zero_values)

        # Should only have values 1, 3, 4 (1 and 2 merged to 1)
        assert set(unique_values) == {1, 3, 4}
        assert 3 in result  # 3 should remain unchanged
        assert 4 in result  # 4 should remain unchanged

    def test_ncells_basic(self):
        """Test keeping only N largest cells."""
        img = np.array([[[1, 1, 2], [1, 3, 2], [4, 4, 4]]], dtype=np.uint32)

        result = ncells(img, ncells=2)

        # Should keep only background (0) and the two largest cells (1 and 4)
        assert np.allclose(np.unique(result), np.array([0, 1, 4]))

    def test_ncells_with_keep_labels(self):
        """Test keeping specific cells regardless of size."""
        img = np.array([[[1, 1, 2], [1, 3, 2], [4, 4, 4]]], dtype=np.uint32)
        keep_labels = [2]

        result = ncells(img, ncells=1, keep_cell_labels=keep_labels)

        # Should only have background (0) and the kept label (2)
        assert np.allclose(np.unique(result), np.array([0, 2]))

    def test_removeislands_basic(self):
        """Test removing small islands."""
        # Create an image with small and large connected components
        img = np.zeros((10, 10, 10), dtype=np.uint32)
        img[2:4, 2:4, 2:4] = 1  # Small island (8 voxels)
        img[6:9, 6:9, 6:9] = 2  # Large island (27 voxels)

        result = removeislands(img, minsize=10)

        # Small island should be removed, large one should remain
        assert 1 not in np.unique(result)
        assert 2 in np.unique(result)


class TestImageProcessingIntegration:
    """Integration tests for image processing operations."""

    def test_dilate_operation(self):
        """Test dilation operation."""
        img = np.zeros((10, 10, 10), dtype=np.uint32)
        img[4:6, 4:6, 4:6] = 1

        # Mock nbmorph.dilate_labels_spherical to avoid dependency
        with patch("imagemesh.morphology.nbmorph") as mock_nbmorph:
            mock_nbmorph.dilate_labels_spherical.return_value = img  # Return same for simplicity

            dilate(img, radius=2)

            mock_nbmorph.dilate_labels_spherical.assert_called_once_with(img, radius=2)

    def test_erode_operation(self):
        """Test erosion operation."""
        img = np.ones((10, 10, 10), dtype=np.uint32)

        # Mock nbmorph.erode_labels_spherical to avoid dependency
        with patch("imagemesh.morphology.nbmorph") as mock_nbmorph:
            mock_nbmorph.erode_labels_spherical.return_value = img  # Return same for simplicity

            erode(img, radius=2)

            mock_nbmorph.erode_labels_spherical.assert_called_once_with(
                img, radius=2, struct_sequence="DDB"
            )

    def test_erode_radius1_separates_corners(self):
        """erode radius=1 leaves no cells touching at voxel corners."""
        # two cells separated by the plane x + y + z = 15
        x, y, z = np.indices((12, 12, 12))
        img = np.where(x + y + z < 15, 2, 3).astype(np.uint32)

        def contacts(img):
            n = 0
            for o in np.ndindex(3, 3, 3):
                o = np.array(o) - 1
                a = img[tuple(slice(max(0, -k), 12 - max(0, k)) for k in o)]
                b = img[tuple(slice(max(0, k), 12 - max(0, -k)) for k in o)]
                n += ((a > 0) & (b > 0) & (a != b)).sum()
            return n

        # a diamond step only separates faces and edges
        assert contacts(erode(img, radius=1, struct_sequence="D")) > 0
        assert contacts(erode(img, radius=1)) == 0

    def test_smooth_operation(self):
        """Test smoothing operation."""
        img = np.ones((10, 10, 10), dtype=np.uint32)

        # Mock nbmorph.smooth_labels_spherical to avoid dependency
        with patch("imagemesh.morphology.nbmorph") as mock_nbmorph:
            mock_nbmorph.smooth_labels_spherical.return_value = img  # Return same for simplicity

            smooth(img, iterations=5, radius=3)

            mock_nbmorph.smooth_labels_spherical.assert_called_once_with(
                img, radius=3, iterations=5, dilate_radius=3
            )


def test_get_lowest_point():
    mask = np.zeros((5, 5, 10), dtype=bool)
    mask[2, 2, 3] = True
    mask[2, 2, 7] = True
    pt = get_lowest_point(mask)
    assert pt[2] == 3


def _scipy_fill_6conn(img):
    """scipy reference with the same 6-connectivity used by binary_fill_holes."""
    structure = ndi.generate_binary_structure(3, 1)  # 6-connectivity in 3D
    return ndi.binary_fill_holes(img, structure=structure)


def test_binary_fill_holes_empty_volume():
    img = np.zeros((5, 6, 7), dtype=bool)
    np.testing.assert_array_equal(binary_fill_holes(img), img)


def test_binary_fill_holes_full_volume():
    img = np.ones((4, 5, 6), dtype=bool)
    np.testing.assert_array_equal(binary_fill_holes(img), img)


def test_binary_fill_holes_no_holes():
    """A foreground blob with no interior holes must be returned unchanged."""
    img = np.zeros((10, 10, 10), dtype=bool)
    img[2:8, 2:8, 2:8] = True
    np.testing.assert_array_equal(binary_fill_holes(img), img)


def test_binary_fill_holes_single_hole():
    img = np.zeros((10, 10, 10), dtype=bool)
    img[2:8, 2:8, 2:8] = True
    img[4, 4, 4] = False  # single interior hole
    out = binary_fill_holes(img)
    expected = img.copy()
    expected[4, 4, 4] = True
    np.testing.assert_array_equal(out, expected)


def test_binary_fill_holes_hole_touching_boundary_not_filled():
    """A "hole" that connects to the outside through 6-connectivity should not
    be filled — it isn't enclosed."""
    img = np.zeros((10, 10, 10), dtype=bool)
    img[2:8, 2:8, 2:8] = True
    # Tunnel the hole all the way out through a face
    img[3:8, 4, 4] = False  # carves a channel from the +x face inward
    np.testing.assert_array_equal(binary_fill_holes(img), img)


def test_binary_fill_holes_matches_scipy_simple():
    img = np.zeros((12, 12, 12), dtype=bool)
    img[2:10, 2:10, 2:10] = True
    img[5, 5, 5] = False
    img[6, 7, 4:6] = False
    np.testing.assert_array_equal(binary_fill_holes(img), _scipy_fill_6conn(img))


def test_binary_fill_holes_matches_scipy_random():
    rng = np.random.default_rng(7)
    # Build a connected hollow shell with random interior holes inside.
    img = np.zeros((15, 16, 17), dtype=bool)
    img[2:13, 2:14, 2:15] = True
    interior = np.zeros_like(img)
    interior[5:10, 5:10, 5:10] = rng.random((5, 5, 5)) > 0.5
    img &= ~interior
    np.testing.assert_array_equal(binary_fill_holes(img), _scipy_fill_6conn(img))


def test_binary_fill_holes_hollow_sphere(tiny_seg):
    """The CSF/WM-shell fixture has WM enclosed by CSF; punching a hole inside
    WM should be re-filled."""
    mask = tiny_seg > 0
    # Carve a single-voxel hole well inside the WM core
    wm_idx = np.argwhere(tiny_seg == 2)
    center = wm_idx[len(wm_idx) // 2]
    mask[tuple(center)] = False
    filled = binary_fill_holes(mask)
    assert filled[tuple(center)]
    # Whole-volume agreement with scipy
    np.testing.assert_array_equal(filled, _scipy_fill_6conn(mask))


def test_grow_into_region_fills_connected_region():
    seeds = np.zeros((10, 10, 10), np.uint16)
    seeds[1, 1, 1] = 5
    region = np.zeros(seeds.shape, bool)
    region[:5] = True
    grown = grow_into_region(seeds, region, radius=1)
    assert np.all(grown[:5] == 5)
    assert np.all(grown[5:] == 0)


def test_separate_labels_inserts_newlabel_between():
    data = np.zeros((12, 4, 4), np.uint8)
    data[:6] = 1
    data[6:] = 2
    out = separate_labels(data.copy(), [1], [2], 1, np.uint8(7))
    assert np.all(out[5:7] == 7)
    assert np.all(out[:4] == 1) and np.all(out[8:] == 2)
