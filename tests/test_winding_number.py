"""Tests for imagemesh.winding_number against libigl."""

import numpy as np
import pytest
import pyvista as pv

from imagemesh.winding_number import label_points, winding_number

igl = pytest.importorskip("igl")


@pytest.fixture
def overlapping_surfaces():
    """Sphere and torus that overlap, so the winding number is 0, 1 or 2."""
    surf = pv.Sphere(theta_resolution=80, phi_resolution=80) + pv.ParametricTorus().translate(
        (1.5, 0, 0)
    )
    surf = surf.triangulate().clean()
    Q = np.random.default_rng(0).uniform(-1.5, 3, (5000, 3))
    return np.array(surf.points, dtype=np.float64), surf.regular_faces.astype(np.int64), Q


@pytest.fixture
def labeled_image():
    """Label image with touching, nested and domain boundary cut regions,
    and its (unsmoothed) multi-label surface net."""
    img = np.zeros((20, 20, 20), np.int32)
    img[3:10, 3:10, 3:12] = 5
    img[10:16, 3:10, 3:12] = 7
    img[4:15, 12:17, 5:15] = 2
    img[6:9, 13:16, 7:10] = 9
    img[0:3, 0:20, 15:20] = 3
    grid = pv.ImageData(dimensions=np.array(img.shape) + 1)
    grid.cell_data["data"] = img.ravel(order="F")
    surf = grid.contour_labels(
        "all", smoothing=False, output_mesh_type="triangles", background_value=0
    )
    return grid, surf


def test_winding_number_matches_igl(overlapping_surfaces):
    V, F, Q = overlapping_surfaces
    exact = winding_number(V, F, Q, beta=np.inf)
    w = winding_number(V, F, Q)
    w_igl = igl.fast_winding_number(V, F, Q)

    assert set(np.unique(np.round(exact))) == {0, 1, 2}
    np.testing.assert_allclose(w, w_igl, atol=1e-2)
    # same or better accuracy than libigl
    assert np.abs(w - exact).max() <= np.abs(w_igl - exact).max()
    assert np.mean(np.abs(w - exact)) <= np.mean(np.abs(w_igl - exact))


def test_label_points_matches_image_and_igl(labeled_image):
    grid, surf = labeled_image
    V = np.array(surf.points, dtype=np.float64)
    F = surf.regular_faces.astype(np.int64)
    blabels = surf.cell_data["boundary_labels"]
    Q = np.array(grid.cell_centers().points)

    labels = label_points(V, F, blabels, Q)
    np.testing.assert_array_equal(labels, grid.cell_data["data"])

    # one libigl winding number per label, faces flipped where the label is
    # on the second side
    labels_igl = np.zeros(len(Q), dtype=labels.dtype)
    for cid in np.setdiff1d(np.unique(blabels), [0]):
        F_label = np.vstack((F[blabels[:, 0] == cid], F[blabels[:, 1] == cid][:, [0, 2, 1]]))
        inside = np.abs(igl.fast_winding_number(V, F_label, Q)) > 0.5
        labels_igl[inside & (labels_igl == 0)] = cid
    np.testing.assert_array_equal(labels, labels_igl)
