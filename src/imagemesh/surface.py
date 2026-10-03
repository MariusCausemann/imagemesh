"""
Multi-label surfaces of label images: surface nets extraction with
constrained smoothing, label transfer and surface checks.

Conventions: cell data boundary_labels (a, b) per triangle, the normal points
out of region a into region b; 0 is the background (outside).
"""

import numba as nb
import numpy as np
from scipy.spatial import cKDTree

from imagemesh.smoothing import smooth_surface_net


def extract_surface(grid, smoothing_scale=1.2, iterations=16, scalars="data", background_value=0):
    """
    Smoothed, triangulated multi-label surface of the label image grid
    (pv.ImageData with the labels as cell data `scalars`), with cell data
    'boundary_labels' (n_faces, 2): surface nets (contour_labels without
    smoothing) followed by a constrained smoothing that keeps the points on
    the bounding box on their box face.
    The displacement of each point is limited to smoothing_scale * dx; dx is
    stored in the field data.
    """
    surf = grid.contour_labels(
        "all",
        smoothing=False,
        output_mesh_type="quads",
        background_value=background_value,
        scalars=scalars,
    )
    dx = np.min(grid.spacing)
    surf = smooth_surface_net(
        surf, iterations=iterations, distance=dx, scale=smoothing_scale, fix_bounds=True
    )
    surf.field_data["dx"] = [dx]
    return surf


def transfer_labels(source_mesh, target_mesh, label_name="boundary_labels"):
    """Copy the cell data label_name from source_mesh to the nearest cells of target_mesh."""
    tree = cKDTree(source_mesh.cell_centers().points)
    _, idx = tree.query(target_mesh.cell_centers().points, k=1)
    target_mesh[label_name] = source_mesh[label_name][idx]
    return target_mesh


def label_volumes(V, F, L):
    """Volume per label (divergence theorem): face (a, b) adds to a, subtracts from b."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    v = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6
    vol = np.zeros(L.max() + 1)
    np.add.at(vol, L[:, 0], v)
    np.add.at(vol, L[:, 1], -v)
    return vol


def flipped_faces(V0, V, F):
    """Faces whose normal turned by more than 90 degrees."""

    def nrm(P):
        return np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])

    return np.einsum("ij,ij->i", nrm(V0), nrm(V)) <= 0


@nb.njit(cache=True, inline="always")
def _seg_tri(p, q, a, b, c):
    """Proper intersection of segment pq with triangle abc (Moller-Trumbore)."""
    d0, d1, d2 = q[0] - p[0], q[1] - p[1], q[2] - p[2]
    e1 = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    e2 = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    h0 = d1 * e2[2] - d2 * e2[1]
    h1 = d2 * e2[0] - d0 * e2[2]
    h2 = d0 * e2[1] - d1 * e2[0]
    det = e1[0] * h0 + e1[1] * h1 + e1[2] * h2
    if abs(det) < 1e-12:
        return False
    inv = 1.0 / det
    s0, s1, s2 = p[0] - a[0], p[1] - a[1], p[2] - a[2]
    u = inv * (s0 * h0 + s1 * h1 + s2 * h2)
    if u <= 0 or u >= 1:
        return False
    r0 = s1 * e1[2] - s2 * e1[1]
    r1 = s2 * e1[0] - s0 * e1[2]
    r2 = s0 * e1[1] - s1 * e1[0]
    v = inv * (d0 * r0 + d1 * r1 + d2 * r2)
    if v <= 0 or u + v >= 1:
        return False
    t = inv * (e2[0] * r0 + e2[1] * r1 + e2[2] * r2)
    return t > 0 and t < 1


@nb.njit(cache=True)
def _tri_tri(V, F, f, g):
    """Triangles f and g intersect (some edge of one pierces the other)."""
    for i in range(3):
        if _seg_tri(V[F[f, i]], V[F[f, (i + 1) % 3]], V[F[g, 0]], V[F[g, 1]], V[F[g, 2]]):
            return True
        if _seg_tri(V[F[g, i]], V[F[g, (i + 1) % 3]], V[F[f, 0]], V[F[f, 1]], V[F[f, 2]]):
            return True
    return False


@nb.njit(parallel=True, cache=True)
def _intersections(V, F, check, order, cell_start, keys, lo, h, dims):
    hit = np.zeros(len(F), np.bool_)
    for f in nb.prange(len(F)):
        if not check[f]:
            continue
        mn = np.empty(3, np.int64)
        mx = np.empty(3, np.int64)
        for d in range(3):
            m0 = min(V[F[f, 0], d], V[F[f, 1], d], V[F[f, 2], d])
            m1 = max(V[F[f, 0], d], V[F[f, 1], d], V[F[f, 2], d])
            mn[d] = max(int((m0 - lo[d]) / h), 0)
            mx[d] = min(int((m1 - lo[d]) / h), dims[d] - 1)
        for i in range(mn[0], mx[0] + 1):
            for j in range(mn[1], mx[1] + 1):
                for k in range(mn[2], mx[2] + 1):
                    key = (i * dims[1] + j) * dims[2] + k
                    s = np.searchsorted(keys, key)
                    if s >= len(keys) or keys[s] != key:
                        continue
                    for t in range(cell_start[s], cell_start[s + 1]):
                        g = order[t]
                        adjacent = False
                        for a in range(3):
                            for b in range(3):
                                adjacent |= F[f, a] == F[g, b]
                        if g != f and not adjacent and _tri_tri(V, F, f, g):
                            hit[f] = True
                            break
                    if hit[f]:
                        break
                if hit[f]:
                    break
            if hit[f]:
                break
    return hit


def self_intersections(V, F, check=None):
    """
    Faces (among check) that intersect a face they share no vertex with.
    Every triangle is binned into the cells of a uniform grid (spacing above
    the largest triangle extent) that its bounding box overlaps.
    """
    V = np.ascontiguousarray(V, np.float64)
    F = np.ascontiguousarray(F, np.int64)
    if check is None:
        check = np.ones(len(F), bool)
    h = 1.01 * (V[F].max(1) - V[F].min(1)).max()
    lo = V.min(0) - h
    dims = np.ceil((V.max(0) - lo) / h).astype(np.int64) + 2
    tmin = np.floor((V[F].min(1) - lo) / h).astype(np.int64)
    tmax = np.floor((V[F].max(1) - lo) / h).astype(np.int64)
    ids, keys = [], []
    for corner in np.ndindex(2, 2, 2):
        c = tmin + corner
        ok = (c <= tmax).all(1)
        ids.append(np.flatnonzero(ok))
        keys.append((c[ok, 0] * dims[1] + c[ok, 1]) * dims[2] + c[ok, 2])
    ids, keys = np.concatenate(ids), np.concatenate(keys)
    o = np.argsort(keys, kind="stable")
    ids, keys = ids[o], keys[o]
    ukeys, start = np.unique(keys, return_index=True)
    start = np.append(start, len(keys))
    check = np.ascontiguousarray(check)
    return _intersections(V, F, check, ids, start, ukeys, lo, h, dims)
