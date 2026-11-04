"""
Volume-preserving fairing of surface nets, bounded by the voxel surface.

The unsmoothed surface net (surface_quads) has its points on the voxel
corners, i.e. on the voxel surface S0. fair_surface_net moves them to

    argmin_x  alpha |L x|^2 + sum_i w_i |x_i - t_i|^2

with L the smoothing stencils of build_stencils (x_i minus the mean of its
neighbours; junction curves are smoothed as curves) and targets t_i that are
updated in an outer loop:

* dead zone: within +- dead_zone of its start along the normal, a point is
  pulled to where it is (no force), so the surface can straighten the voxel
  staircase; tangential motion is pulled back to the start. Points on a
  junction (faces of more than one label pair) get a spherical dead zone.
  dead_zone=None drops the anchor: the targets are the current positions.
* volume: each label k (except the background) has an offset p_k that is
  updated by (V_voxels - V) / area until its volume is the voxel volume; a
  face between a and b moves by p_a - p_b along its normal. This removes the
  shrinkage of the curvature term, which otherwise pulls in small and thin
  regions.
* distance limit: targets further than max_distance from S0 are pulled back
  to max_distance (towards the closest point of S0, so tangential motion
  stays free), and points at the limit get a stiff spring. A final
  projection guarantees dist(x_i, S0) <= max_distance for every point.

S0 consists of axis-aligned voxel faces, so the closest point on a face is
the point clipped to the face's bounding box. Each point searches the faces
around its 2-ring in the unsmoothed net; if a point slid further, its true
distance is at most the computed one, so the bound holds.
"""

import numba as nb
import numpy as np
import pyvista as pv
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from imagemesh.smoothing import bounding_box_mask, build_stencils, triangulate_quads


@nb.njit(parallel=True, cache=True)
def _closest_on_faces(X, indptr, faces, fmin, fmax):
    """Closest point of X[i] on the axis-aligned faces faces[indptr[i]:indptr[i + 1]]
    (bounding boxes fmin, fmax) and its distance."""
    n = len(X)
    C = np.empty_like(X)
    D = np.empty(n)
    for i in nb.prange(n):
        best = np.inf
        for k in range(indptr[i], indptr[i + 1]):
            f = faces[k]
            c0 = min(max(X[i, 0], fmin[f, 0]), fmax[f, 0])
            c1 = min(max(X[i, 1], fmin[f, 1]), fmax[f, 1])
            c2 = min(max(X[i, 2], fmin[f, 2]), fmax[f, 2])
            d2 = (X[i, 0] - c0) ** 2 + (X[i, 1] - c1) ** 2 + (X[i, 2] - c2) ** 2
            if d2 < best:
                best = d2
                C[i, 0], C[i, 1], C[i, 2] = c0, c1, c2
        D[i] = np.sqrt(best)
    return C, D


def _candidate_faces(quads, n_points, rings=2):
    """CSR (indptr, indices): per point the quads incident to its k-ring."""
    m = len(quads)
    VQ = sp.csr_matrix(
        (np.ones(4 * m, np.int8), (quads.ravel(), np.repeat(np.arange(m), 4))),
        shape=(n_points, m),
    )
    adj = (VQ @ VQ.T).astype(bool).astype(np.int8)  # points sharing a quad, incl. itself
    R = adj
    for _ in range(rings - 1):
        R = (R @ adj).astype(bool).astype(np.int8)
    C = (R @ VQ).tocsr()
    C.sort_indices()
    return C.indptr.astype(np.int64), C.indices.astype(np.int64)


def _quad_area_normals(X, quads):
    """Area vector of each quad (half the cross product of its diagonals)."""
    return 0.5 * np.cross(X[quads[:, 2]] - X[quads[:, 0]], X[quads[:, 3]] - X[quads[:, 1]])


def _label_volumes(X, quads, labels, n_labels):
    """Volume per label of the closed quad surface (a adds, b subtracts)."""
    a, b, c, d = (X[quads[:, k]] for k in range(4))
    v = (np.einsum("ij,ij->i", a, np.cross(b, c)) + np.einsum("ij,ij->i", a, np.cross(c, d))) / 6
    return np.bincount(labels[:, 0], v, n_labels) - np.bincount(labels[:, 1], v, n_labels)


def fair_surface_net(
    surf,
    dx,
    alpha=30.0,
    dead_zone=0.5,
    max_distance=1.0,
    volume=True,
    iterations=40,
    background_value=0,
    label_name="boundary_labels",
    fix_bounds=True,
    fixed=None,
    stiffness=100.0,
    degenerate=None,
    rtol=1e-6,
):
    """
    Fair the unsmoothed surface net surf (quads from surface_quads, cell data
    label_name (a, b), normals out of a) and triangulate it, see the module
    docstring. dead_zone and max_distance are in units of dx; None switches
    them off. fix_bounds keeps points on the bounding box on their box face;
    fixed: optional (n_points, 3) mask of coordinates that must not move.
    Returns the triangulated surface with the cell data repeated per triangle
    and the field data 'dx' and 'max_surface_distance' (the largest distance of
    a point from the voxel surface, nm or whatever unit dx is in).
    """
    quads = np.asarray(surf.regular_faces, dtype=np.int64)
    X0 = np.asarray(surf.points, dtype=np.float64)
    n = len(X0)
    labels = np.asarray(surf.cell_data[label_name]).astype(np.int64)
    n_labels = int(labels.max()) + 1

    fixed = np.zeros((n, 3), bool) if fixed is None else np.asarray(fixed, bool).copy()
    if fix_bounds:
        fixed |= bounding_box_mask(X0)

    A = build_stencils(quads, n, degenerate=degenerate)
    L = (sp.eye(n, format="csr") - A).tocsr()
    M = (alpha * (L.T @ L)).tocsr()
    blocks = []
    for c in range(3):
        f = np.flatnonzero(~fixed[:, c])
        k = np.flatnonzero(fixed[:, c])
        blocks.append((f, M[f][:, f].tocsr(), M[f][:, k] @ X0[k, c]))

    # per point: incident quads (for normals and volume offsets) and whether it
    # lies on a single interface (one label pair) or on a junction
    pair = labels[:, 0] * n_labels + labels[:, 1]
    rows = quads.ravel()
    qid = np.repeat(np.arange(len(quads)), 4)
    P2Q = sp.csr_matrix((np.ones(len(rows)), (rows, qid)), shape=(n, len(quads)))
    pmin = np.full(n, np.iinfo(np.int64).max)
    pmax = np.full(n, -1)
    np.minimum.at(pmin, rows, np.repeat(pair, 4))
    np.maximum.at(pmax, rows, np.repeat(pair, 4))
    junction = pmin != pmax

    if max_distance is not None:
        r = max_distance * dx
        indptr, cand = _candidate_faces(quads, n)
        fmin = X0[quads].min(axis=1)
        fmax = X0[quads].max(axis=1)

        def to_tube(Y):
            C, D = _closest_on_faces(np.ascontiguousarray(Y), indptr, cand, fmin, fmax)
            s = np.minimum(1.0, r / np.maximum(D, 1e-300))
            return C + (Y - C) * s[:, None], D

    h = None if dead_zone is None else dead_zone * dx
    keep = np.arange(n_labels) != background_value
    V0 = _label_volumes(X0, quads, labels, n_labels)
    p = np.zeros(n_labels)
    X = X0.copy()
    w = np.ones(n)
    for _ in range(iterations):
        an = _quad_area_normals(X, quads)
        area = np.linalg.norm(an, axis=1)
        N = P2Q @ an
        N /= np.linalg.norm(N, axis=1)[:, None] + 1e-300
        U = X - X0
        if h is None:
            T = X.copy()
        else:
            un = np.einsum("ij,ij->i", U, N)
            T = X0 + np.clip(un, -h, h)[:, None] * N  # normal dead zone, tangential spring
            ru = np.linalg.norm(U[junction], axis=1)
            T[junction] = (
                X0[junction] + U[junction] * np.minimum(1, h / np.maximum(ru, 1e-300))[:, None]
            )
        if volume:
            V = _label_volumes(X, quads, labels, n_labels)
            A_lab = np.bincount(labels[:, 0], area, n_labels) + np.bincount(
                labels[:, 1], area, n_labels
            )
            ok = keep & (A_lab > 0)
            p[ok] += (V0[ok] - V[ok]) / A_lab[ok]
            dq = (p[labels[:, 0]] - p[labels[:, 1]])[:, None] * an  # area-weighted face shift
            T += (P2Q @ dq) / np.maximum(P2Q @ area, 1e-300)[:, None]
        if max_distance is not None:
            T, _ = to_tube(T)
            _, D = _closest_on_faces(X, indptr, cand, fmin, fmax)
            at_limit = D > 0.98 * r
            w = np.where(at_limit, np.minimum(np.maximum(3 * w, 10.0), stiffness), 1.0)
        for c, (f, Mff, Mfk) in enumerate(blocks):
            K = (sp.diags(w[f]) + Mff).tocsr()
            x, _ = spla.cg(
                K,
                w[f] * T[f, c] - Mfk,
                x0=X[f, c],
                rtol=rtol,
                maxiter=5000,
                M=sp.diags(1.0 / K.diagonal()),
            )
            X[f, c] = x

    if max_distance is not None:
        X, _ = to_tube(X)
        X[fixed] = X0[fixed]
        _, D = _closest_on_faces(X, indptr, cand, fmin, fmax)
        dmax = float(D.max())
    else:
        dmax = np.nan

    tris = triangulate_quads(X, quads)
    out = pv.PolyData(X, np.column_stack([np.full(len(tris), 3), tris]).ravel())
    for name, arr in surf.cell_data.items():
        out.cell_data[name] = np.repeat(arr, 2, axis=0)
    out.field_data["dx"] = [dx]
    out.field_data["max_surface_distance"] = [dmax]
    return out
