"""Tests for imagemesh.simplification."""

import numpy as np

from imagemesh.image import np2pv
from imagemesh.simplification import simplify_surface
from imagemesh.surface import extract_surface

DX = 10.0


def curve_segments(surf):
    """Endpoints (n, 2, 3) of the edges between faces of different label pairs."""
    F = surf.regular_faces
    L = np.sort(surf["boundary_labels"], axis=1)
    pair = np.repeat(L[:, 0] * (L.max() + 1) + L[:, 1], 3)
    e = np.sort(np.stack([F, np.roll(F, -1, axis=1)], -1).reshape(-1, 2), axis=1)
    key = e[:, 0] * surf.n_points + e[:, 1]
    _, inv = np.unique(key, return_inverse=True)
    lo = np.full(inv.max() + 1, pair.max() + 1)
    hi = np.full(inv.max() + 1, -1)
    np.minimum.at(lo, inv, pair)
    np.maximum.at(hi, inv, pair)
    _, first = np.unique(inv, return_index=True)
    return surf.points[e[first[lo < hi]]]


def hausdorff(S, T, n=5):
    """Largest distance of points sampled on the segments S to the segments T."""
    t = np.linspace(0, 1, n)[:, None, None]
    P = (S[None, :, 0] * (1 - t) + S[None, :, 1] * t).reshape(-1, 3)
    A, d = T[:, 0], T[:, 1] - T[:, 0]
    s = np.einsum("pij,ij->pi", P[:, None] - A, d) / np.einsum("ij,ij->i", d, d)
    Q = A + np.clip(s, 0, 1)[..., None] * d
    return np.linalg.norm(P[:, None] - Q, axis=2).min(axis=1).max()


def test_simplify_keeps_junction_curves():
    # a ball split into labels 2 and 3 in label 1: unsmoothed, the junction
    # curve is a staircase that must stay within epsilon both ways
    x, y, z = np.mgrid[:16, :16, :16] - 7.5
    img = np.ones((16, 16, 16), dtype=np.uint32)
    ball = x**2 + y**2 + z**2 < 6**2
    img[ball & (z < 0.3 * x)] = 2
    img[ball & (z >= 0.3 * x)] = 3
    surf = extract_surface(np2pv(img, (DX,) * 3), iterations=0)
    eps = 0.5 * DX
    dec = simplify_surface(surf, eps)
    assert dec.n_cells < surf.n_cells
    fine, coarse = curve_segments(surf), curve_segments(dec)
    assert hausdorff(coarse, fine) <= eps * (1 + 1e-9)
    assert hausdorff(fine, coarse) <= eps * (1 + 1e-9)
