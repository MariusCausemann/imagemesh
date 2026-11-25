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

from typing import NamedTuple

import numba as nb
import numpy as np
import pyvista as pv
import scipy.sparse as sp


@nb.njit(cache=True)
def incident_faces(faces, n):
    """CSR (indptr, indices): per point the faces it is a corner of, in
    increasing order."""
    k = faces.shape[1]
    ptr = np.zeros(n + 1, np.int64)
    for f in range(len(faces)):
        for c in range(k):
            ptr[faces[f, c] + 1] += 1
    for i in range(n):
        ptr[i + 1] += ptr[i]
    idx = np.empty(ptr[n], np.int64)
    fill = ptr[:-1].copy()
    for f in range(len(faces)):
        for c in range(k):
            i = faces[f, c]
            idx[fill[i]] = f
            fill[i] += 1
    return ptr, idx


@nb.njit(parallel=True, cache=True)
def _duplicate_faces(faces, ptr, idx):
    """Mask of the faces with the same vertex set as an earlier face."""
    m, k = faces.shape
    dup = np.zeros(m, np.bool_)
    for f in nb.prange(m):
        u = faces[f, 0]
        for c in range(1, k):
            u = min(u, faces[f, c])
        for jj in range(ptr[u], ptr[u + 1]):
            g = idx[jj]
            if g >= f:
                break
            same = True
            for c in range(k):
                x, y = faces[f, c], faces[g, c]
                inf = ing = False
                for c2 in range(k):
                    ing |= faces[g, c2] == x
                    inf |= faces[f, c2] == y
                if not (ing and inf):
                    same = False
                    break
            if same:
                dup[f] = True
                break
    return dup


def unique_faces(faces):
    """Faces with duplicates (same vertex set, any order) removed, the first
    of each kept, in order."""
    faces = np.ascontiguousarray(faces, np.int64)
    if not len(faces):
        return faces
    ptr, idx = incident_faces(faces, int(faces.max()) + 1)
    return faces[~_duplicate_faces(faces, ptr, idx)]


@nb.njit(parallel=True, cache=True)
def _edge_pass(faces, ptr, idx, fval, fill, nchunk, cnt, off, eu, ev, valence, vmin, vmax):
    """Per point u, the edges (u, x), x > u, of the faces around it, in
    increasing x: counted (fill=False) or written from off[u]."""
    n, k = len(ptr) - 1, faces.shape[1]
    maxdeg = 0
    for i in range(n):
        maxdeg = max(maxdeg, ptr[i + 1] - ptr[i])
    nchunk = max(1, min(n, nchunk))
    B = (n + nchunk - 1) // nchunk
    for ch in nb.prange(nchunk):
        xs = np.empty(2 * maxdeg, np.int64)
        vs = np.empty(2 * maxdeg, np.int64)
        for u in range(ch * B, min(n, (ch + 1) * B)):
            m = 0
            for jj in range(ptr[u], ptr[u + 1]):
                f = idx[jj]
                for c in range(k):
                    if faces[f, c] != u:
                        continue
                    for x in (faces[f, (c + k - 1) % k], faces[f, (c + 1) % k]):
                        if x <= u:
                            continue
                        j = m  # insertion sort by x
                        while j > 0 and xs[j - 1] > x:
                            xs[j], vs[j] = xs[j - 1], vs[j - 1]
                            j -= 1
                        xs[j], vs[j] = x, fval[f]
                        m += 1
            e = off[u] if fill else 0
            i = 0
            while i < m:
                j = i
                lo = hi = vs[i]
                while j + 1 < m and xs[j + 1] == xs[i]:
                    j += 1
                    lo, hi = min(lo, vs[j]), max(hi, vs[j])
                if fill:
                    eu[e], ev[e], valence[e], vmin[e], vmax[e] = u, xs[i], j - i + 1, lo, hi
                e += 1
                i = j + 1
            if not fill:
                cnt[u] = e


def edge_table(faces, n, fval=None):
    """
    The edges of the polygons faces (corners in cyclic order) on n points,
    each once, sorted by (u, v) with u < v: (u, v, valence (the number of
    faces on the edge), and the smallest and largest per-face value fval on
    it).
    """
    faces = np.ascontiguousarray(faces, np.int64)
    fval = np.zeros(len(faces), np.int64) if fval is None else np.asarray(fval, np.int64)
    ptr, idx = incident_faces(faces, n)
    cnt = np.zeros(n, np.int64)
    e0 = np.empty(0, np.int64)
    nchunk = 64 * nb.get_num_threads()
    _edge_pass(faces, ptr, idx, fval, False, nchunk, cnt, cnt, e0, e0, e0, e0, e0)
    off = np.zeros(n + 1, np.int64)
    np.cumsum(cnt, out=off[1:])
    out = [np.empty(off[n], np.int64) for _ in range(5)]
    _edge_pass(faces, ptr, idx, fval, True, nchunk, cnt, off, *out)
    return tuple(out)


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
    eu, ev, valence, _, _ = edge_table(faces, n_points)
    edges = np.column_stack([eu, ev])

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
def _smooth_step(indptr, indices, data, x0, x, xn, relaxation, d2, fixed, lo, hi):
    """One constrained Jacobi iteration xn <- x + relaxation * (A @ x - x) with A
    in CSR format, the free coordinates clamped to the box [lo, hi]; returns
    max |xn - x|."""
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
        # the box contains x0: clamping keeps the sphere constraint
        if not fixed[i, 0]:
            y0 = min(max(y0, lo[0]), hi[0])
        if not fixed[i, 1]:
            y1 = min(max(y1, lo[1]), hi[1])
        if not fixed[i, 2]:
            y2 = min(max(y2, lo[2]), hi[2])
        xn[i, 0], xn[i, 1], xn[i, 2] = y0, y1, y2
        m = max(abs(y0 - x[i, 0]), abs(y1 - x[i, 1]), abs(y2 - x[i, 2]))
        moved = max(moved, m)
    return moved


def constrained_smooth(
    points, A, iterations=16, relaxation=0.5, distance=1.0, convergence=0.0, fixed=None,
    guard=None, bounds=None, margin=0.0,
):  # fmt: skip
    """
    vtkConstrainedSmoothingFilter with a sphere constraint.

    fixed: optional (n_points, 3) bool mask of coordinates that must not move,
    e.g. fixed[k, 2] = True keeps point k in its z-plane. guard: optional
    PlanarGuard (planar_guard) of quads that every step keeps valid.
    bounds: optional box (xmin, xmax, ymin, ymax, zmin, zmax) containing the
    points; every step clamps the free coordinates to it, margin inside its
    faces (the start points must lie in that interval). With an averaging A
    the steps stay in the box up to rounding anyway, not in the margin.
    """
    x0 = np.ascontiguousarray(points, dtype=np.float64)
    x = x0.copy()
    A = sp.csr_matrix(A)
    if fixed is None:
        fixed = np.zeros(x.shape, dtype=bool)
    fixed = np.ascontiguousarray(fixed, dtype=bool)
    xn = np.empty_like(x)
    if bounds is None:
        lo, hi = np.full(3, -np.inf), np.full(3, np.inf)
    else:
        lo, hi = np.asarray(bounds, float).reshape(3, 2).T
        lo, hi = np.ascontiguousarray(lo + margin), np.ascontiguousarray(hi - margin)
    for _ in range(iterations):
        moved = _smooth_step(
            A.indptr, A.indices, A.data, x0, x, xn, relaxation, distance**2, fixed, lo, hi
        )
        apply_guard(x, xn, guard)
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
    bounds=None,
    margin=0.0,
):
    """
    Smooth a surface from contour_labels(smoothing=False, output_mesh_type="quads")
    like contour_labels(smoothing=True) does. Defaults match pyvista:
    distance = norm(spacing) * scale.

    fixed: (n_points, 3) bool mask of coordinates that must not move.
    fix_bounds: additionally keep points on the bounding box of surf on
    their box face, and every point inside the box bounds (default: that
    bounding box; pass the image bounds) (clamped). build_stencils' stencils
    are averages, so a step is a convex combination and stays in the box up
    to rounding; the clamp makes it exact (an average of points on a box face
    may round past it). Given bounds clamp even without fix_bounds. margin:
    the free coordinates stay that far inside the box faces (a one-voxel dent
    in a box face shrinks towards the face; without it its floor may end on
    the face, a region of no volume).
    degenerate: see build_stencils; None is the VTK behaviour.
    """
    quads = surf.regular_faces
    if distance is None:
        distance = np.linalg.norm(spacing)
    A = build_stencils(quads, surf.n_points, optimized=optimized, degenerate=degenerate)

    fixed = np.zeros((surf.n_points, 3), dtype=bool) if fixed is None else np.asarray(fixed)
    box = surf.bounds if bounds is None else bounds
    if fix_bounds:
        fixed = fixed | bounding_box_mask(surf.points)

    pts = constrained_smooth(
        surf.points, A, iterations, relaxation, distance * scale, fixed=fixed,
        bounds=box if fix_bounds or bounds is not None else None, margin=margin,
    )  # fmt: skip
    if not triangulate:
        out = surf.copy()
        out.points = pts
        return out
    tris = triangulate_quads(pts, quads)
    out = pv.PolyData(pts, np.column_stack([np.full(len(tris), 3), tris]).ravel())
    for name, arr in surf.cell_data.items():
        out.cell_data[name] = np.repeat(arr, 2, axis=0)
    return out


# ---------------------------------------------------------------- planar guard
#
# Quads lying in an axis plane (e.g. the caps on the cut planes of a block
# surface) that must stay valid while their points move: the diagonal
# triangulate_quads picks has two triangles of the original orientation with
# at least the minimum area. A smoothing or fairing step that would break a
# guarded quad keeps the moving corners of that quad where they were before
# the step (repeated until every guarded quad is valid: the previous state
# was valid, so this ends). Valid at the start, valid after every step.


class PlanarGuard(NamedTuple):
    """Guarded planar quads, see planar_guard."""

    quads: np.ndarray  # (m, 4) int64, indices into the points then extra
    axis: np.ndarray  # (m,) normal axis of each quad's plane
    sign: np.ndarray  # (m,) orientation (+-1) of each quad in its plane
    min_area2: np.ndarray  # (m,) smallest twice-area of a triangle
    extra: np.ndarray  # (k, 3) fixed points the quads may use (index n + j)


def planar_guard(points, quads, axis, min_area, extra=None):
    """
    The guard of the planar quads (indices into points, then into extra:
    fixed points that never move) with plane normal axis per quad, valid at
    points: each keeps its orientation there and the two triangles of the
    diagonal triangulate_quads picks keep at least min_area (per quad or
    scalar) each. Raises if a quad is not valid at points.
    """
    P = np.asarray(points, np.float64)
    E = np.zeros((0, 3)) if extra is None else np.asarray(extra, np.float64)
    q = np.ascontiguousarray(quads, np.int64).reshape(-1, 4)
    ax = np.ascontiguousarray(np.broadcast_to(np.asarray(axis, np.int64), (len(q),)))
    X = np.vstack([P, E])
    b, c = (ax + 1) % 3, (ax + 2) % 3
    x, y = X[q, b[:, None]], X[q, c[:, None]]
    a2 = np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    sign = np.sign(a2).astype(np.int64)
    if (sign == 0).any():
        raise ValueError(f"planar_guard: {(sign == 0).sum()} quads of zero area")
    m2 = np.ascontiguousarray(np.broadcast_to(2.0 * np.asarray(min_area, np.float64), (len(q),)))
    g = PlanarGuard(q, ax, sign, m2, np.ascontiguousarray(E))
    bad = guard_violations(P, g)
    if len(bad):
        raise ValueError(f"planar_guard: {len(bad)} quads not valid at the given points")
    return g


def empty_guard():
    return PlanarGuard(
        np.zeros((0, 4), np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64),
        np.zeros(0), np.zeros((0, 3)),
    )  # fmt: skip


def guard_violations(points, guard):
    """Indices of the guarded quads not valid at points."""
    X = np.ascontiguousarray(points, np.float64)
    g = guard
    ok = _guard_ok_all(X, g.extra, g.quads, g.axis, g.sign, g.min_area2)
    return np.flatnonzero(~ok)


@nb.njit(cache=True)
def _gp(X, E, i, a):
    n = len(X)
    return X[i, a] if i < n else E[i - n, a]


@nb.njit(cache=True)
def _ga2(X, E, i, j, k, b, c):
    """Twice the signed area of triangle (i, j, k) in the plane of axes (b, c)."""
    xi, yi = _gp(X, E, i, b), _gp(X, E, i, c)
    xj, yj = _gp(X, E, j, b), _gp(X, E, j, c)
    xk, yk = _gp(X, E, k, b), _gp(X, E, k, c)
    return (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)


@nb.njit(cache=True)
def _quad_ok(X, E, q0, q1, q2, q3, ax, sg, amin2):
    """Whether the diagonal triangulate_quads picks gives two triangles of
    orientation sg with twice-area >= amin2 (on a tie of the diagonals'
    lengths: both, as the tie break depends on the global point ids)."""
    b, c = (ax + 1) % 3, (ax + 2) % 3
    a012 = sg * _ga2(X, E, q0, q1, q2, b, c)
    a023 = sg * _ga2(X, E, q0, q2, q3, b, c)
    a123 = sg * _ga2(X, E, q1, q2, q3, b, c)
    a130 = sg * _ga2(X, E, q1, q3, q0, b, c)
    ok02 = min(a012, a023) >= amin2
    ok13 = min(a123, a130) >= amin2
    fold02 = a012 * a023 < 0
    fold13 = a123 * a130 < 0
    if fold02 != fold13:
        return ok13 if fold02 else ok02
    d02 = np.sqrt(
        (_gp(X, E, q0, b) - _gp(X, E, q2, b)) ** 2 + (_gp(X, E, q0, c) - _gp(X, E, q2, c)) ** 2
    )
    d13 = np.sqrt(
        (_gp(X, E, q1, b) - _gp(X, E, q3, b)) ** 2 + (_gp(X, E, q1, c) - _gp(X, E, q3, c)) ** 2
    )
    if d13 < d02:
        return ok13
    if d02 < d13:
        return ok02
    return ok02 and ok13


@nb.njit(parallel=True, cache=True)
def _guard_ok_all(X, E, gq, gax, gsg, gmin):
    ok = np.empty(len(gq), np.bool_)
    for f in nb.prange(len(gq)):
        ok[f] = _quad_ok(X, E, gq[f, 0], gq[f, 1], gq[f, 2], gq[f, 3], gax[f], gsg[f], gmin[f])
    return ok


@nb.njit(cache=True)
def _guard(Xold, Xnew, E, gq, gax, gsg, gmin, held):
    """
    Keep the moving corners of every guarded quad that Xnew breaks at Xold,
    until all are valid (Xold valid: ends). held[i] is set for the points
    kept. Returns the number of quads still invalid (0 if Xold was valid).
    """
    if _guard_ok_all(Xnew, E, gq, gax, gsg, gmin).all():
        return 0  # the common case, checked in parallel
    n = len(Xnew)
    changed = True
    while changed:
        changed = False
        for f in range(len(gq)):
            q0, q1, q2, q3 = gq[f, 0], gq[f, 1], gq[f, 2], gq[f, 3]
            if _quad_ok(Xnew, E, q0, q1, q2, q3, gax[f], gsg[f], gmin[f]):
                continue
            for k in range(4):
                i = gq[f, k]
                if i < n and not held[i]:
                    held[i] = True
                    Xnew[i, 0], Xnew[i, 1], Xnew[i, 2] = Xold[i, 0], Xold[i, 1], Xold[i, 2]
                    changed = True
    bad = 0
    for f in range(len(gq)):
        if not _quad_ok(Xnew, E, gq[f, 0], gq[f, 1], gq[f, 2], gq[f, 3], gax[f], gsg[f], gmin[f]):
            bad += 1
    return bad


def apply_guard(Xold, Xnew, guard):
    """Xnew (in place) with the guarded quads valid (see _guard); returns the
    mask of the points kept at Xold."""
    held = np.zeros(len(Xnew), np.bool_)
    if guard is None or not len(guard.quads):
        return held
    g = guard
    bad = _guard(Xold, Xnew, g.extra, g.quads, g.axis, g.sign, g.min_area2, held)
    if bad:
        raise RuntimeError(f"planar guard: {bad} quads invalid before the step")
    return held
