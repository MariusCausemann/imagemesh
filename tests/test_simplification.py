"""Tests for imagemesh.simplification."""

import numpy as np
import pyvista as pv

from imagemesh.image import np2pv
from imagemesh.simplification import _patches, _select, _vertex_faces, simplify_surface
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


def _ball_surface():
    x, y, z = np.mgrid[:16, :16, :16] - 7.5
    img = np.ones((16, 16, 16), dtype=np.uint32)
    img[x**2 + y**2 + z**2 < 6**2] = 2
    return extract_surface(np2pv(img, (DX,) * 3), iterations=0)


def test_simplify_min_collapse_fraction():
    surf = _ball_surface()
    full = simplify_surface(surf, 0.5 * DX, min_collapse_fraction=0)
    early = simplify_surface(surf, 0.5 * DX, min_collapse_fraction=0.05)
    assert full.n_cells < early.n_cells < surf.n_cells


def test_select_matches_sequential_greedy():
    """The parallel _select takes the set of the sequential greedy pass in
    order of (cost, u)."""
    s = pv.Sphere(theta_resolution=60, phi_resolution=60).triangulate()
    F = np.ascontiguousarray(s.regular_faces, np.int64)
    nv = s.n_points
    vf_ptr, vf_idx = _vertex_faces(F, nv)
    rng = np.random.default_rng(0)
    for _ in range(5):
        cost = rng.random(nv)
        target = np.full(nv, -1, np.int64)
        for u in range(nv):
            nbrs = np.setdiff1d(F[vf_idx[vf_ptr[u] : vf_ptr[u + 1]]], [u])
            if len(nbrs) and rng.random() < 0.8:
                target[u] = rng.choice(nbrs)
        covered = np.zeros(nv, bool)
        want = []
        for u in sorted(np.flatnonzero(target >= 0), key=lambda u: (cost[u], u)):
            v = target[u]
            if covered[u] or covered[v]:
                continue
            want.append(u)
            for x in (u, v):
                covered[F[vf_idx[vf_ptr[x] : vf_ptr[x + 1]]]] = True
        np.testing.assert_array_equal(_select(target, cost, F, vf_ptr, vf_idx), sorted(want))


def test_patches_large_labels():
    """Label pairs are told apart for labels whose product overflows int64."""
    big = 5 * 2**40
    labels = np.array(
        [[big + 1, big + 2], [big + 2, big + 1], [big + 1, 2 * big + 3], [0, big + 1]]
    )
    p = _patches(labels)
    assert p[0] == p[1] and len(np.unique(p)) == 3
