"""Tests for imagemesh.surface and imagemesh.tetmesh on a box of three labels."""

import numpy as np
import pytest

from imagemesh.image import np2pv
from imagemesh.surface import (
    close_quad_holes,
    extract_surface,
    label_volumes,
    self_intersections,
)
from imagemesh.tetmesh import mesh_surface

DX = 10.0


def two_cells(pad=0):
    """Two touching boxes (labels 2 and 3) in label 1 filling a 20^3 voxel image;
    with pad, the outer pad voxel layers in x are background (0)."""
    img = np.ones((20, 20, 20), dtype=np.uint32)
    img[4:10, 5:15, 5:15] = 2
    img[10:16, 5:15, 5:15] = 3
    if pad:
        img[:pad] = 0
        img[-pad:] = 0
    return np2pv(img, (DX,) * 3)


def tet_volumes(mesh):
    vol = mesh.compute_cell_sizes(length=False, area=False)["Volume"]
    return {int(k): vol[mesh["label"] == k].sum() for k in np.unique(mesh["label"])}


@pytest.mark.parametrize("pad", [0, 2])
def test_extract_surface(pad):
    imggrid = two_cells(pad)
    surf = extract_surface(imggrid)
    assert surf.is_all_triangles
    assert surf.field_data["dx"][0] == DX

    labels = surf.cell_data["boundary_labels"]
    pairs = {tuple(sorted(p)) for p in labels}
    assert {(1, 2), (1, 3), (2, 3)} <= pairs
    # the outside (0) only borders label 1
    assert all(p[1] == 1 for p in pairs if p[0] == 0)

    # each label region is closed: every edge of its faces is used twice
    for label in (1, 2, 3):
        faces = surf.regular_faces[(labels == label).any(axis=1)]
        edges = np.sort(np.stack([faces, np.roll(faces, -1, axis=1)], -1).reshape(-1, 2), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        assert np.all(counts == 2)

    # points stay inside the image, the box faces are kept
    lo, hi = np.array(imggrid.bounds).reshape(3, 2).T
    lo[0], hi[0] = pad * DX, (20 - pad) * DX
    assert np.allclose(surf.points.min(axis=0), lo)
    assert np.allclose(surf.points.max(axis=0), hi)
    assert not self_intersections(surf.points, surf.regular_faces).any()


@pytest.mark.parametrize("pad", [0, 2])
def test_mesh_surface(pad):
    surf = extract_surface(two_cells(pad))
    mesh, dec = mesh_surface(surf, envelopsize=0.5, simplify_eps=0.05 * DX)
    assert dec.n_points <= surf.n_points

    vols = tet_volumes(mesh)
    assert set(vols) == {1, 2, 3}
    # simplification, meshing and labeling keep the volumes enclosed by the surface
    ref = label_volumes(
        surf.points.astype(float), surf.regular_faces, surf["boundary_labels"].astype(int)
    )
    for k in (1, 2, 3):
        assert vols[k] == pytest.approx(ref[k], rel=1e-2)
    # smoothing rounds off the edges of the thin boxes
    cell = 6 * 10 * 10 * DX**3
    assert vols[2] == pytest.approx(cell, rel=0.25)
    assert vols[3] == pytest.approx(cell, rel=0.25)
    box = (20 - 2 * pad) * 20 * 20 * DX**3
    assert sum(vols.values()) == pytest.approx(box, rel=1e-3)


def test_mesh_surface_label_name_and_tetwild_kwargs():
    surf = extract_surface(two_cells())
    mesh, _ = mesh_surface(surf, label_name="marker", edge_length_fac=0.1, stop_energy=20)
    assert "marker" in mesh.cell_data and "label" not in mesh.cell_data
    assert set(np.unique(mesh["marker"])) == {1, 2, 3}


def open_edges(faces):
    """Half-edges of faces without a reverse half-edge."""
    a, b = faces.ravel(), np.roll(faces, -1, axis=1).ravel()
    n = faces.max() + 1
    return ~np.isin(b * n + a, a * n + b)


def test_close_quad_holes():
    # a 3^3 block for which vtkSurfaceNets3D drops one quad between labels 1 and 2
    img = np.full((3, 3, 3), 2, dtype=np.uint32)
    for p in [(0, 2, 2), (1, 0, 0), (1, 0, 1), (2, 0, 0), (2, 1, 1), (2, 2, 2)]:
        img[p] = 1
    surf = np2pv(img, (DX,) * 3).contour_labels(
        "all", smoothing=False, output_mesh_type="quads", background_value=0, scalars="data"
    )
    if not open_edges(surf.regular_faces).any():
        pytest.skip("this VTK version leaves no hole")
    assert open_edges(surf.regular_faces).sum() == 4

    closed = close_quad_holes(surf)
    assert closed.n_cells == surf.n_cells + 1
    assert not open_edges(closed.regular_faces).any()
    assert np.array_equal(closed.regular_faces[:-1], surf.regular_faces)
    assert sorted(closed["boundary_labels"][-1]) == [1, 2]
    # nothing to close
    assert close_quad_holes(closed) is closed
