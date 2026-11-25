"""Tests for the fairing of surface nets (extract_surface(method="fair"))."""

import numpy as np
import pytest

from imagemesh.image import np2pv
from imagemesh.surface import extract_surface, foreign_points, label_volumes, surface_quads

DX = 10.0


def sphere_and_tube(n=24):
    """A sphere (2) and a thin tube along x (3) in label 1, one voxel apart."""
    c = np.indices((n, n, n)).transpose(1, 2, 3, 0) + 0.5
    img = np.ones((n, n, n), np.uint32)
    img[np.linalg.norm(c - [12, 15, 12], axis=-1) < 6] = 2
    img[3:21][np.linalg.norm(c[3:21, ..., 1:] - [6, 12], axis=-1) < 1.6] = 3
    return img, np2pv(img, (DX,) * 3)


@pytest.fixture(scope="module")
def case():
    img, grid = sphere_and_tube()
    return img, grid, extract_surface(grid, method="fair")


def volumes(surf):
    L = np.asarray(surf.cell_data["boundary_labels"]).astype(np.int64)
    return label_volumes(np.asarray(surf.points), surf.regular_faces, L)


def test_fair_keeps_volumes(case):
    img, _, surf = case
    vol = volumes(surf)
    for k in (2, 3):
        assert vol[k] == pytest.approx(np.sum(img == k) * DX**3, rel=0.02)


def test_fair_distance_bound(case):
    _, grid, surf = case
    assert surf.field_data["max_surface_distance"][0] <= DX * (1 + 1e-9)
    d = surf.compute_implicit_distance(surface_quads(grid).triangulate())["implicit_distance"]
    assert np.abs(d).max() <= DX * (1 + 1e-6)


def test_fair_no_foreign_points_and_box_kept(case):
    _, grid, surf = case
    assert foreign_points(surf, grid).sum() == 0
    assert np.allclose(surf.bounds, grid.bounds)


def test_fair_tighter_bound(case):
    _, grid, _ = case
    surf = extract_surface(grid, method="fair", max_distance=0.3, iterations=10)
    assert surf.field_data["max_surface_distance"][0] <= 0.3 * DX * (1 + 1e-9)


def test_laplace_shrinks_tube_fair_does_not(case):
    img, grid, fair = case
    lap = volumes(extract_surface(grid))[3] / (np.sum(img == 3) * DX**3)
    assert lap < 0.8 < volumes(fair)[3] / (np.sum(img == 3) * DX**3)


def test_unknown_method_raises(case):
    _, grid, _ = case
    with pytest.raises(ValueError):
        extract_surface(grid, method="nope")


def assert_in_box(surf, grid, margin=0.1):
    """Every coordinate on a box face or margin dx inside it (the dent floors keep volume)."""
    P = np.asarray(surf.points)
    lo, hi = np.asarray(grid.bounds).reshape(3, 2).T
    on = (P == lo) | (P == hi)
    m = margin * DX * (1 - 1e-12)
    assert ((P >= lo + m) & (P <= hi - m) | on).all()
    assert np.allclose(surf.bounds, grid.bounds)


def dented_face(dent, n=12):
    """A cell (2) filling a box; ECS (1) in a slab and in a one-voxel dent of the x = 0 face."""
    img = np.full((n, n + 5, n + 5), 2, np.uint32)
    img[-3:] = 1
    for y, z in dent:
        img[0, y, z] = 1
    return np2pv(img, (DX,) * 3)


@pytest.mark.parametrize(
    "dent, kwargs",
    [
        # the dents of a real case (flyhedra fb_2um_sep) whose floors slid out of the box
        ([(7, 7), (9, 7), (9, 8), (9, 10), (10, 7), (10, 8)], {}),
        ([(7, 7), (9, 7), (9, 8), (9, 10), (10, 7), (10, 8)], {"max_distance": 0.5}),
        ([(5, 5), (5, 6)], {"max_distance": 0.5, "influence_radius": 40}),
    ],
)
def test_fair_stays_in_box(dent, kwargs):
    """The fairing shrinks a dent in the box face; its floor must stay inside the box."""
    grid = dented_face(dent)
    surf = extract_surface(grid, method="fair", **kwargs)
    assert_in_box(surf, grid)
    md = kwargs.get("max_distance", 1.0) + 0.1  # + box_margin next to the box faces
    assert surf.field_data["max_surface_distance"][0] <= md * DX * (1 + 1e-9)


@pytest.mark.parametrize(
    "dent", [[(7, 7), (9, 7), (9, 8), (9, 10), (10, 7), (10, 8)], [(5, 5), (5, 6)], [(5, 5)]]
)
def test_laplace_stays_in_box(dent):
    """The Laplacian smoothing (method="laplace") keeps the dent floors in the box too."""
    grid = dented_face(dent)
    assert_in_box(extract_surface(grid), grid)


def test_constrained_smooth_clamps_to_bounds():
    """Weights that extrapolate push points out of the box; bounds clamp them."""
    import scipy.sparse as sp

    from imagemesh.smoothing import constrained_smooth

    x = np.array([[0.0, 0.5, 0.5], [0.2, 0.5, 0.5], [1.0, 0.5, 0.5]])
    # point 0 is pushed away from point 1: 2 x0 - x1
    A = sp.csr_matrix(np.array([[2.0, -1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 0.0, 1.0]]))
    box = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    free = constrained_smooth(x, A, iterations=4, relaxation=1.0, distance=np.inf)
    assert free[0, 0] < 0.0
    P = constrained_smooth(x, A, iterations=4, relaxation=1.0, distance=np.inf, bounds=box)
    assert (P >= 0.0).all() and (P <= 1.0).all()
    assert P[0, 0] == 0.0
    P = constrained_smooth(
        x, A, iterations=4, relaxation=1.0, distance=np.inf, bounds=box, margin=0.1
    )
    assert P[0, 0] == 0.1 and P[2, 0] == 0.9
