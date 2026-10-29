"""Tests for imagemesh.surface and imagemesh.tetmesh on a box of three labels."""

import numpy as np
import pytest

from imagemesh.image import np2pv
from imagemesh.surface import (
    close_quad_holes,
    extract_surface,
    label_volumes,
    self_intersections,
    surface_quads,
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


def label_surface(surf, label):
    """The faces of label's boundary, oriented out of label."""
    labels = surf["boundary_labels"]
    faces = surf.regular_faces[(labels == label).any(axis=1)]
    inward = labels[(labels == label).any(axis=1), 1] == label
    faces[inward] = faces[inward, ::-1]
    return faces


def pinched_edges(faces):
    """Edges of faces used by more than two of them."""
    edges = np.sort(np.stack([faces, np.roll(faces, -1, axis=1)], -1).reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return (counts > 2).sum()


def is_manifold(faces):
    """Closed 2-manifold: every half-edge once with its reverse, every vertex link one cycle."""
    if pinched_edges(faces) or open_edges(faces).any():
        return False
    nxt = {}  # around vertex v: edge (v, w) -> next edge (v, x)
    for q in faces:
        for i in range(4):
            nxt.setdefault(q[i], {})[q[i - 1]] = q[(i + 1) % 4]
    for around in nxt.values():
        start, w, steps = next(iter(around)), None, 0
        while w != start:
            w = around[start if w is None else w]
            steps += 1
        if steps != len(around):
            return False
    return True


def quads(img, background_value=0):
    grid = np2pv(img.astype(np.uint32), (DX,) * 3)
    return grid, surface_quads(grid, background_value=background_value)


def face_keys(surf):
    """Centroid, normal direction and labels of every face, sorted."""
    x = surf.points[surf.regular_faces]
    normal = np.sign(np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]))
    keys = np.column_stack([x.mean(axis=1), normal, surf["boundary_labels"]])
    return keys[np.lexsort(keys.T[::-1])]


@pytest.mark.parametrize("seed", range(5))
def test_surface_quads_matches_contour_labels(seed):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 4, size=(5, 4, 3))
    grid, surf = quads(img)
    ref = grid.contour_labels("all", smoothing=False, output_mesh_type="quads", background_value=0)
    # the same faces, labels and orientations as surface nets
    assert np.array_equal(face_keys(surf), face_keys(ref))
    assert surf["boundary_labels"].dtype == np.uint32
    # without the slits
    assert not open_edges(surf.regular_faces).any()


@pytest.mark.parametrize("flip", [1, -1])
def test_surface_quads_direction(flip):
    # a rotated (or reflected) grid: every face lies between the voxels of its
    # labels, normal from a to b (contour_labels shifts oriented grids by
    # (I - R) spacing / 2)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 3, size=(4, 3, 3))
    grid = np2pv(img.astype(np.uint32), (1.0, 2.0, 0.5))
    c, s = np.cos(0.3), np.sin(0.3)
    grid.direction_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, flip]])
    grid.origin = (1.0, -2.0, 3.0)
    surf = surface_quads(grid)
    inv = np.linalg.inv(np.asarray(grid.index_to_physical_matrix))

    def label_at(x):
        ijk = np.floor(x @ inv[:3, :3].T + inv[:3, 3]).astype(int)
        inside = np.all((ijk >= 0) & (ijk < img.shape), axis=1)
        out = np.zeros(len(x), int)
        out[inside] = img[tuple(ijk[inside].T)]
        return out

    x = surf.points[surf.regular_faces]
    n = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
    n /= np.linalg.norm(n, axis=1)[:, None]
    labels = surf["boundary_labels"]
    assert np.array_equal(label_at(x.mean(axis=1) - 0.1 * n), labels[:, 0])
    assert np.array_equal(label_at(x.mean(axis=1) + 0.1 * n), labels[:, 1])


def test_surface_quads_no_slit():
    # the 3^3 block where vtkSurfaceNets3D leaves a slit (test_close_quad_holes)
    img = np.full((3, 3, 3), 2)
    for p in [(0, 2, 2), (1, 0, 0), (1, 0, 1), (2, 0, 0), (2, 1, 1), (2, 2, 2)]:
        img[p] = 1
    _, surf = quads(img)
    assert not open_edges(surf.regular_faces).any()


def test_surface_quads_edge_pinch():
    # two voxels of label 2 sharing an edge, in label 1: label 2 is separated
    img = np.ones((4, 4, 3))
    img[1, 1, 1] = img[2, 2, 1] = 2
    _, surf = quads(img)
    # the box surface and 2 x 8 cube corners: the shared edge is split
    assert surf.n_points == (5 * 5 * 4 - 3 * 3 * 2) + 2 * 8
    assert is_manifold(label_surface(surf, 2))
    assert is_manifold(label_surface(surf, 1))
    assert len(np.unique(label_surface(surf, 2))) == 16  # two separate cubes


def test_surface_quads_vertex_pinch():
    img = np.ones((4, 4, 4))
    img[1, 1, 1] = img[2, 2, 2] = 2
    _, surf = quads(img)
    # the box surface and 2 x 8 cube corners: the shared corner is split
    assert surf.n_points == (5**3 - 3**3) + 2 * 8
    assert is_manifold(label_surface(surf, 2))
    assert is_manifold(label_surface(surf, 1))


def test_surface_quads_checkerboard():
    # a 2x2 checkerboard of labels 1 and 2 under a layer of label 2: label 2
    # continues diagonally through the end of the checkerboard and connects,
    # label 1 is separated (the tie rule would separate label 2)
    img = np.ones((2, 2, 2))
    img[1, 0, 0] = img[0, 1, 0] = 2
    img[:, :, 1] = 2
    _, surf = quads(img)
    assert surf.n_points == 3 * 3 * 3 + 1  # split at the top, joined in the background
    assert not pinched_edges(label_surface(surf, 1))
    assert not pinched_edges(label_surface(surf, 2))


def test_surface_quads_background_value():
    img = np.array([[3, 5], [5, 3]]).reshape(2, 2, 1)
    _, surf = quads(img, background_value=5)
    # background faces carry it second, and the background connects
    labels = surf["boundary_labels"]
    assert set(map(tuple, labels)) == {(3, 5)}
    assert is_manifold(label_surface(surf, 3))
