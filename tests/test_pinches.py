"""Tests for imagemesh.pinches (pinch removal before surface nets)."""

import numpy as np
import pytest
import pyvista as pv
from scipy import ndimage

from imagemesh.pinches import count_pinches, remove_pinches


def non_manifold(img):
    """
    Number of non-manifold edges and vertices of the surface net of img, after
    merging coincident points (VTK duplicates the points at a pinch).
    """
    grid = pv.ImageData(dimensions=np.array(img.shape) + 1)
    grid.cell_data["data"] = img.flatten(order="F")
    surf = grid.contour_labels("all", smoothing=False, output_mesh_type="triangles")
    surf = surf.clean(tolerance=1e-6)
    faces = surf.regular_faces
    edges = np.sort(np.stack([faces, np.roll(faces, -1, axis=1)], axis=2).reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    n_edges = np.sum(counts != 2)
    # a manifold vertex: the opposite edges of its faces form a single cycle
    n_vertices = 0
    for v in range(surf.n_points):
        link = [[a for a in f if a != v] for f in faces if v in f]
        comp = {a: a for e in link for a in e}

        def find(a):
            while comp[a] != a:
                a = comp[a]
            return a

        for a, b in link:
            comp[find(a)] = find(b)
        n_vertices += len({find(a) for a in comp}) > 1
    return n_edges + n_vertices


def touching_cells(img):
    """True if two different cells are 26-neighbours."""
    nx, ny, nz = img.shape
    pad = np.pad(img, 1)
    core = pad[1:-1, 1:-1, 1:-1]
    for d in np.ndindex(3, 3, 3):
        nb = pad[d[0] : d[0] + nx, d[1] : d[1] + ny, d[2] : d[2] + nz]
        if np.any((core != 0) & (nb != 0) & (core != nb)):
            return True
    return False


def pieces(img, label):
    """Number of face-connected pieces of label."""
    return ndimage.label(img == label)[1]


def edge_pinch(a=1, b=0):
    """Two boxes of label a touching along an edge (in z), b elsewhere."""
    img = np.full((8, 8, 6), b, np.int32)
    img[1:4, 1:4, 1:5] = a
    img[4:7, 4:7, 1:5] = a
    return img


def vertex_pinch(a=1, b=0):
    """Two boxes of label a touching at a corner, b elsewhere."""
    img = np.full((8, 8, 8), b, np.int32)
    img[1:4, 1:4, 1:4] = a
    img[4:7, 4:7, 4:7] = a
    return img


@pytest.mark.parametrize("make", [edge_pinch, vertex_pinch])
def test_remove_pinches_separates_cell(make):
    img = make()
    assert count_pinches(img) > 0
    out = remove_pinches(img)
    assert count_pinches(out) == 0
    assert non_manifold(out) == 0
    # a pinch is not a connection: the cell only loses a few voxels at the
    # contact, and its face-connected pieces stay as they were
    assert np.all(out[img == 0] == 0)
    assert 0 < np.sum(out != img) <= 4
    assert pieces(out, 1) == pieces(img, 1)


@pytest.mark.parametrize("make", [edge_pinch, vertex_pinch])
def test_surface_nets_pinch_detected(make):
    assert non_manifold(make()) > 0


def test_remove_pinches_keeps_cells_separated():
    # cells next to both background voxels that could thicken the pinch,
    # separated from cell 1 by one background layer
    img = edge_pinch()
    img[1:3, 5:8, 1:5] = 2
    img[5:8, 1:3, 1:5] = 3
    assert count_pinches(img) > 0 and not touching_cells(img)
    out = remove_pinches(img, separate_cells=True)
    assert count_pinches(out) == 0
    assert not touching_cells(out)
    assert non_manifold(out) == 0


def test_remove_pinches_cells_in_contact():
    # checkerboard of two cells (no background): both are pinched, and they are
    # separated by background where that splits neither
    img = edge_pinch(a=1, b=2)
    out = remove_pinches(img)
    assert count_pinches(out) == 0
    # the cells are separated by background voxels: three labels meet along
    # junction lines, so the surface of every label is tested on its own
    for label in (0, 1, 2):
        assert non_manifold(np.pad((out == label).astype(np.int32), 1)) == 0
    assert np.sum(out != img) <= 8  # at most 2 per layer of the 4-voxel contact
    for label in (1, 2):
        assert pieces(out, label) <= pieces(img, label)


def test_pinch_free_image_unchanged():
    img = np.zeros((10, 10, 10), np.int32)
    img[2:5, 2:8, 2:8] = 1
    img[6:9, 2:8, 2:8] = 2
    assert count_pinches(img) == 0
    assert np.array_equal(remove_pinches(img), img)
