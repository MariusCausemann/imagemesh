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
