"""
Replica of the smoothing in vtkSurfaceNets3D (pyvista's contour_labels),
applied to an existing unsmoothed surface net, with optional per-point,
per-axis constraints.

vtkSurfaceNets3D builds one smoothing stencil per point and hands it to
vtkConstrainedSmoothingFilter:

* stencil: the edge neighbours of the point.
* optimized stencils (VTK default): if the point has an edge shared by more
  than two faces (a junction where >= 3 regions meet), only the neighbours
  along junction edges are used, so junction curves are smoothed as curves
  instead of being pulled into the adjacent sheets.
* a stencil with a single neighbour locks the point.
* Jacobi iterations x <- x + relax * (mean(stencil) - x); after each
  iteration the displacement from the original position is clamped to a
  sphere of radius `distance`.
"""

import numba as nb
import numpy as np
import pyvista as pv
import scipy.sparse as sp


def unique_faces(faces):
    """Faces with duplicates (same vertex set, any order) removed."""
    s = np.ascontiguousarray(np.sort(faces, axis=1))
    # one void scalar per row: 1d np.unique is much faster than axis=0
    key = s.view(np.dtype((np.void, s.dtype.itemsize * s.shape[1]))).ravel()
    _, idx = np.unique(key, return_index=True)
    return faces[np.sort(idx)]


def build_stencils(faces, n_points, optimized=True, degenerate="full"):
    """
    Row-normalized sparse averaging matrix A, so that A @ x is the stencil
    mean. faces must be the quads of the unsmoothed surface net; duplicate
    faces are counted once.

    degenerate: what to do with opposite corners of a quad whose (optimized)
    stencils are identical, or one contains the other, e.g. a face with
    junction edges on all four sides. Such corners are pulled to (nearly) the
    same point (their distance halves every iteration), and the quad
    degenerates into two slivers. "full" gives them their full edge
    neighbourhood, "lock" keeps them in place, None keeps the VTK behaviour.
    """
    faces = unique_faces(np.asarray(faces))
    edges = np.stack([faces, np.roll(faces, -1, axis=1)], axis=-1).reshape(-1, 2)
    edges = np.sort(edges, axis=1).astype(np.int64)
    key, valence = np.unique(edges[:, 0] * n_points + edges[:, 1], return_counts=True)
    edges = np.column_stack([key // n_points, key % n_points])

    i = np.concatenate([edges[:, 0], edges[:, 1]])
    j = np.concatenate([edges[:, 1], edges[:, 0]])
    val = np.concatenate([valence, valence])

    if optimized:
        # points touching a junction edge only use junction edges
        junction = np.zeros(n_points, dtype=bool)
        junction[i[val > 2]] = True
        keep = ~junction[i] | (val > 2)
        if degenerate is not None:
            bad = _degenerate_diagonals(faces, i[keep], j[keep], n_points)
            if degenerate == "full":
                keep |= bad[i]
            elif degenerate == "lock":
                keep &= ~bad[i]
                i = np.concatenate([i, np.flatnonzero(bad)])
                j = np.concatenate([j, np.flatnonzero(bad)])
                keep = np.concatenate([keep, np.ones(bad.sum(), dtype=bool)])
            else:
                raise ValueError(f"unknown degenerate={degenerate!r}")
        i, j = i[keep], j[keep]

    # a single neighbour locks the point: its stencil is the point itself
    locked = np.bincount(i, minlength=n_points) == 1
    keep = ~locked[i]
    lid = np.flatnonzero(locked)
    i = np.concatenate([i[keep], lid])
    j = np.concatenate([j[keep], lid])

    n_nb = np.bincount(i, minlength=n_points)
    return sp.csr_matrix((1.0 / n_nb[i], (i, j)), shape=(n_points, n_points))


def _degenerate_diagonals(faces, i, j, n_points):
    """Mask of the opposite quad corners where the stencil (neighbour set
    i -> j) of one contains the stencil of the other."""
    B = sp.csr_matrix((np.ones(len(i), dtype=np.int8), (i, j)), shape=(n_points, n_points))
    B.sum_duplicates()
    a = np.concatenate([faces[:, 0], faces[:, 1]])
    b = np.concatenate([faces[:, 2], faces[:, 3]])
    Ba, Bb = B[a], B[b]
    common = Ba.multiply(Bb).getnnz(axis=1)
    n_min = np.minimum(Ba.getnnz(axis=1), Bb.getnnz(axis=1))
    same = (common == n_min) & (n_min > 1)
    bad = np.zeros(n_points, dtype=bool)
    bad[a[same]] = True
    bad[b[same]] = True
    return bad


@nb.njit(parallel=True, cache=True)
def _smooth_step(indptr, indices, data, x0, x, xn, relaxation, d2, fixed):
    """One constrained Jacobi iteration xn <- x + relaxation * (A @ x - x) with A
    in CSR format; returns max |xn - x|."""
    moved = 0.0
    for i in nb.prange(len(x)):
        a0 = a1 = a2 = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            j, w = indices[k], data[k]
            a0 += w * x[j, 0]
            a1 += w * x[j, 1]
            a2 += w * x[j, 2]
        y0 = x[i, 0] + relaxation * (a0 - x[i, 0])
        y1 = x[i, 1] + relaxation * (a1 - x[i, 1])
        y2 = x[i, 2] + relaxation * (a2 - x[i, 2])
        if fixed[i, 0]:
            y0 = x0[i, 0]
        if fixed[i, 1]:
            y1 = x0[i, 1]
        if fixed[i, 2]:
            y2 = x0[i, 2]
        u0, u1, u2 = y0 - x0[i, 0], y1 - x0[i, 1], y2 - x0[i, 2]
        r2 = u0 * u0 + u1 * u1 + u2 * u2
        if r2 > d2:
            f = np.sqrt(d2 / r2)
            y0, y1, y2 = x0[i, 0] + u0 * f, x0[i, 1] + u1 * f, x0[i, 2] + u2 * f
        xn[i, 0], xn[i, 1], xn[i, 2] = y0, y1, y2
        m = max(abs(y0 - x[i, 0]), abs(y1 - x[i, 1]), abs(y2 - x[i, 2]))
        moved = max(moved, m)
    return moved


def constrained_smooth(
    points, A, iterations=16, relaxation=0.5, distance=1.0, convergence=0.0, fixed=None
):
    """
    vtkConstrainedSmoothingFilter with a sphere constraint.

    fixed: optional (n_points, 3) bool mask of coordinates that must not move,
    e.g. fixed[k, 2] = True keeps point k in its z-plane.
    """
    x0 = np.ascontiguousarray(points, dtype=np.float64)
    x = x0.copy()
    A = sp.csr_matrix(A)
    if fixed is None:
        fixed = np.zeros(x.shape, dtype=bool)
    fixed = np.ascontiguousarray(fixed, dtype=bool)
    xn = np.empty_like(x)
    for _ in range(iterations):
        moved = _smooth_step(A.indptr, A.indices, A.data, x0, x, xn, relaxation, distance**2, fixed)
        x, xn = xn, x
        if moved <= convergence:
            break
    return x


def bounding_box_mask(points, tol=1e-6):
    """(n_points, 3) mask of the coordinates lying on the bounding box faces."""
    p = np.asarray(points)
    lo, hi = p.min(axis=0), p.max(axis=0)
    atol = tol * np.max(hi - lo)
    return np.isclose(p, lo, rtol=0, atol=atol) | np.isclose(p, hi, rtol=0, atol=atol)


def triangulate_quads(points, quads):
    """
    Split each quad along its shorter diagonal, unless the two triangles would
    fold onto each other (the quad became non-convex during smoothing); then
    the other diagonal is used. Ties are broken by vertex id, so a quad shared
    by two surfaces is split the same way in both, independent of its vertex
    order and orientation.
    """
    return _triangulate_quads(
        np.ascontiguousarray(points, dtype=np.float64), np.ascontiguousarray(quads, dtype=np.int64)
    )


@nb.njit(cache=True)
def _folds(p, a, b, c, d):
    """Whether the triangles (a, b, c) and (a, c, d) point in opposite directions."""
    u0, u1, u2 = p[b, 0] - p[a, 0], p[b, 1] - p[a, 1], p[b, 2] - p[a, 2]
    v0, v1, v2 = p[c, 0] - p[a, 0], p[c, 1] - p[a, 1], p[c, 2] - p[a, 2]
    w0, w1, w2 = p[d, 0] - p[a, 0], p[d, 1] - p[a, 1], p[d, 2] - p[a, 2]
    n0, n1, n2 = u1 * v2 - u2 * v1, u2 * v0 - u0 * v2, u0 * v1 - u1 * v0
    m0, m1, m2 = v1 * w2 - v2 * w1, v2 * w0 - v0 * w2, v0 * w1 - v1 * w0
    return n0 * m0 + n1 * m1 + n2 * m2 < 0


@nb.njit(parallel=True, cache=True)
def _triangulate_quads(p, quads):
    m = len(quads)
    tris = np.empty((2 * m, 3), np.int64)
    for f in nb.prange(m):
        q0, q1, q2, q3 = quads[f, 0], quads[f, 1], quads[f, 2], quads[f, 3]
        d02 = np.sqrt(
            (p[q0, 0] - p[q2, 0]) ** 2 + (p[q0, 1] - p[q2, 1]) ** 2 + (p[q0, 2] - p[q2, 2]) ** 2
        )
        d13 = np.sqrt(
            (p[q1, 0] - p[q3, 0]) ** 2 + (p[q1, 1] - p[q3, 1]) ** 2 + (p[q1, 2] - p[q3, 2]) ** 2
        )
        use13 = d13 < d02 or (d13 == d02 and min(q1, q3) < min(q0, q2))
        fold02 = _folds(p, q0, q1, q2, q3)
        fold13 = _folds(p, q1, q2, q3, q0)
        if fold02 != fold13:
            use13 = fold02
        if use13:
            q0, q1, q2, q3 = q1, q2, q3, q0
        tris[2 * f, 0], tris[2 * f, 1], tris[2 * f, 2] = q0, q1, q2
        tris[2 * f + 1, 0], tris[2 * f + 1, 1], tris[2 * f + 1, 2] = q0, q2, q3
    return tris


def smooth_surface_net(
    surf,
    iterations=16,
    relaxation=0.5,
    distance=None,
    scale=1.0,
    spacing=None,
    optimized=True,
    fixed=None,
    fix_bounds=False,
    triangulate=True,
    degenerate="full",
):
    """
    Smooth a surface from contour_labels(smoothing=False, output_mesh_type="quads")
    like contour_labels(smoothing=True) does. Defaults match pyvista:
    distance = norm(spacing) * scale.

    fixed: (n_points, 3) bool mask of coordinates that must not move.
    fix_bounds: additionally keep points on the bounding box on their box face.
    degenerate: see build_stencils; None is the VTK behaviour.
    """
    quads = surf.regular_faces
    if distance is None:
        distance = np.linalg.norm(spacing)
    A = build_stencils(quads, surf.n_points, optimized=optimized, degenerate=degenerate)

    fixed = np.zeros((surf.n_points, 3), dtype=bool) if fixed is None else np.asarray(fixed)
    if fix_bounds:
        fixed = fixed | bounding_box_mask(surf.points)

    pts = constrained_smooth(surf.points, A, iterations, relaxation, distance * scale, fixed=fixed)
    if not triangulate:
        out = surf.copy()
        out.points = pts
        return out
    tris = triangulate_quads(pts, quads)
    out = pv.PolyData(pts, np.column_stack([np.full(len(tris), 3), tris]).ravel())
    for name, arr in surf.cell_data.items():
        out.cell_data[name] = np.repeat(arr, 2, axis=0)
    return out
