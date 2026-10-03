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
    quads = np.asarray(quads)
    p = np.asarray(points)
    d02 = np.linalg.norm(p[quads[:, 0]] - p[quads[:, 2]], axis=1)
    d13 = np.linalg.norm(p[quads[:, 1]] - p[quads[:, 3]], axis=1)
    min02 = np.minimum(quads[:, 0], quads[:, 2])
    min13 = np.minimum(quads[:, 1], quads[:, 3])
    use13 = (d13 < d02) | ((d13 == d02) & (min13 < min02))

    def folds(q):
        # normals of the triangles (0, 1, 2) and (0, 2, 3) point in opposite directions
        n1 = np.cross(p[q[:, 1]] - p[q[:, 0]], p[q[:, 2]] - p[q[:, 0]])
        n2 = np.cross(p[q[:, 2]] - p[q[:, 0]], p[q[:, 3]] - p[q[:, 0]])
        return np.einsum("ij,ij->i", n1, n2) < 0

    fold02, fold13 = folds(quads), folds(np.roll(quads, -1, axis=1))
    use13 = np.where(fold02 != fold13, fold02, use13)
    q = np.where(use13[:, None], np.roll(quads, -1, axis=1), quads)
    return np.stack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]], axis=1).reshape(-1, 3)


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
):
    """
    Smooth a surface from contour_labels(smoothing=False, output_mesh_type="quads")
    like contour_labels(smoothing=True) does. Defaults match pyvista:
    distance = norm(spacing) * scale.

    fixed: (n_points, 3) bool mask of coordinates that must not move.
    fix_bounds: additionally keep points on the bounding box on their box face.
    """
    quads = surf.regular_faces
    if distance is None:
        distance = np.linalg.norm(spacing)
    A = build_stencils(quads, surf.n_points, optimized=optimized)

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
