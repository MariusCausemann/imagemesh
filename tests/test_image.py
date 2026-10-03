"""Tests for imagemesh.image."""

import numpy as np
import pyvista as pv

from imagemesh.image import get_cell_frequencies, np2pv


class TestNp2pv:
    """Test the np2pv function."""

    def test_np2pv_basic(self, sample_image_data, sample_resolution):
        """Test basic np2pv functionality."""
        grid = np2pv(sample_image_data, sample_resolution)

        assert isinstance(grid, pv.ImageData)
        assert np.array_equal(grid.dimensions, sample_image_data.shape + np.array([1, 1, 1]))
        assert np.array_equal(grid.spacing, sample_resolution)
        assert grid.origin == (0, 0, 0)
        assert "data" in grid.array_names
        assert grid["data"].shape == (sample_image_data.size,)

    def test_np2pv_with_roimask(self, sample_image_data, sample_resolution):
        """Test np2pv with roimask."""
        roimask = np.ones_like(sample_image_data, dtype=bool)
        roimask[10:20, 10:20, 10:20] = False

        grid = np2pv(sample_image_data, sample_resolution, roimask=roimask)

        assert "roimask" in grid.array_names
        assert grid["roimask"].shape == (roimask.size,)


class TestGetCellFrequencies:
    """Test the get_cell_frequencies function."""

    def test_get_cell_frequencies_basic(self, sample_image_data):
        """Test basic cell frequency calculation."""
        frequencies = get_cell_frequencies(sample_image_data)

        assert frequencies.shape[0] == 2  # labels and counts
        assert frequencies.shape[1] >= 3  # at least 3 unique values (0, 1, 2, 3)

        # Check that labels are sorted by frequency
        counts = frequencies[1]
        assert np.all(counts[:-1] <= counts[1:])

    def test_get_cell_frequencies_single_cell(self):
        """Test with single cell."""
        data = np.zeros((10, 10, 10), dtype=np.uint32)
        data[2:8, 2:8, 2:8] = 5

        frequencies = get_cell_frequencies(data)

        assert frequencies.shape[1] == 2  # background and cell 5
        assert 5 in frequencies[0]


def test_nibabel_to_pyvista_roundtrip():
    nib = __import__("pytest").importorskip("nibabel")
    from imagemesh.image import nibabel_to_pyvista

    data = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    affine = np.diag([0.5, 0.5, 0.5, 1.0])
    grid = nibabel_to_pyvista(nib.Nifti1Image(data, affine))
    assert tuple(grid.dimensions) == (3, 4, 5)
    np.testing.assert_allclose(grid.spacing, 0.5)
    np.testing.assert_array_equal(grid["data"], data.flatten(order="F"))
