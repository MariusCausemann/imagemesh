"""
Volume-preserving fairing of surface nets, bounded by the voxel surface, with
a guaranteed (bitwise) range of influence.

The unsmoothed surface net (surface_quads) has its points on the voxel
corners, i.e. on the voxel surface S0. fair_surface_net works in two passes.

Pass 1 (fairing) decreases

    alpha |L x|^2 + sum_i w_i |x_i - t_i|^2

with L the smoothing stencils of build_stencils (x_i minus the mean of its
neighbours; junction curves are smoothed as curves) by a fixed number of
local, accelerated Jacobi steps (heavy ball). Every step is computed from the
previous positions only:

* dead zone: within +- dead_zone of its start along the normal, a point is
  pulled to where it is (no force), so the surface can straighten the voxel
  staircase; tangential motion is pulled back to the start. Points on a
  junction (faces of more than one label pair) get a spherical dead zone.
  dead_zone=None drops the anchor: the targets are the current positions.
* distance limit: targets further than max_distance from S0 are pulled back
  to max_distance (towards the closest point of S0, so tangential motion
  stays free), and points at the limit get a stiff spring.

Pass 2 (volume, repeated volume_repeats times) undoes the shrinkage of the
curvature term: every face measures the volume it swept since the start;
this is averaged over the faces within volume_rings edge steps on its
interface (label pair), by two box averages in a row (all faces within
volume_rings // 2 steps, then within the remaining steps), and every point
is moved back along the normals of its faces by the averaged volume / area.
The swept volume is mostly the straightening of the voxel staircase, which
averages out over a step period, plus the shrinkage, which does not. The box
averages are wide for their reach (a diffusion with the same reach would
only be about sqrt(volume_rings) faces wide and leak the staircase back), so
the shift is a smooth field and the surface stays fair while every region
keeps its volume. A volume feedback inside pass 1 would act on lagged,
narrow measurements (see the range below) and leave dimples.

A final projection guarantees dist(x_i, S0) <= max_distance for every point
(+ box_margin near the box faces, see below).

Box. Every point stays inside the box `bounds` (the image bounds): the
fairing steps, the volume restoration and the final projection clamp each
free coordinate to it, box_margin dx inside its faces (the points on a box
face keep their face). The distance limit alone does not keep a point
inside: S0 includes the voxel faces on the box, and a point may slide along
S0 out of it (e.g. the floor of a one-voxel dent in the box face, which the
fairing shrinks). Without the margin such a floor would end on the box face,
on top of the box faces of the dent: a region of no volume. A free
coordinate starts on a voxel corner off the face, at least dx inside, so the
clamp interval contains its start. Such a floor slides along the box faces
(they are part of S0, so the distance limit allows it) and the projection
onto S0 would take it back next to the face, so the margin comes last: near
the box faces the distance limit is max_distance + box_margin (the clamp
moves a point by at most the margin from the box, which contains S0), and
exact elsewhere. Clamping reads the point only (the range below is
unchanged).

Range of influence. Measure the distance between a surface point and a
voxel in voxels, as the L-infinity distance from the point's voxel corner to
the voxel (the cube [i, i+1] x [j, j+1] x [k, k+1]). The static data of a
point (stencils, faces, labels, the candidate faces of the distance limit:
quads around the 2-ring) are functions of the voxels within distance 3. A
step of pass 1 reads the previous positions within distance 2 (L^T L x: 2,
the normals: 1), so after S steps a point depends on the voxels within
2 S + 1. A volume restoration moves a point by the smoothed volume of its
faces, averaged over the faces within k = volume_rings edge steps, whose
corners lie within 1 + k: it adds 1 + k. In total, a point depends on the voxels within

    R = 2 S + 1 + volume_repeats * (1 + volume_rings)

only, and changing any voxel further away leaves it bitwise unchanged.
fair_surface_net takes this range (influence_radius) and runs the largest S
that fits. The guarantee needs:

* the same grid (spacing, origin, direction) and bounds for both images: the
  points on the bounding box are fixed by `bounds`, not by the extent of the
  surface;
* no checkerboard edge run (2x2 checkerboard voxels along an edge) reaching
  out of the range: surface_quads decides how such an edge is split from the
  ends of the whole run. remove_pinches removes them.

It holds independently of the number of threads: every loop writes its own
entries, and every sum runs in index order, which is the raster order of the
voxel corners and faces in any image on the grid.

S0 consists of axis-aligned voxel faces, so the closest point on a face is
the point clipped to the face's bounding box. Each point searches the faces
around its 2-ring in the unsmoothed net; if a point slid further, its true
distance is at most the computed one, so the bound holds. The search is
skipped while |x_i - x0_i| <= max_distance (x0_i lies on S0), which is
exact and leaves out almost all points.
"""

import numba as nb
import numpy as np
import pyvista as pv
import scipy.sparse as sp

from imagemesh.smoothing import _guard, apply_guard, build_stencils, empty_guard, triangulate_quads


@nb.njit(cache=True)
def _closest(x0, x1, x2, quads, p_ptr, p_idx, fmin, fmax, i):
    """Closest point to (x0, x1, x2) on the quads around the 2-ring of point i
    (quads of the points sharing a quad with a point sharing a quad with i),
    and its distance. The quads are visited in a fixed order from i, with
    repetitions, so ties go to the first one reached."""
    best = np.inf
    c0 = c1 = c2 = 0.0
    for k1 in range(p_ptr[i], p_ptr[i + 1]):
        for v1 in range(4):
            j1 = quads[p_idx[k1], v1]
            for k2 in range(p_ptr[j1], p_ptr[j1 + 1]):
                for v2 in range(4):
                    j2 = quads[p_idx[k2], v2]
                    for k3 in range(p_ptr[j2], p_ptr[j2 + 1]):
                        f = p_idx[k3]
                        e0 = min(max(x0, fmin[f, 0]), fmax[f, 0])
                        e1 = min(max(x1, fmin[f, 1]), fmax[f, 1])
                        e2 = min(max(x2, fmin[f, 2]), fmax[f, 2])
                        d2 = (x0 - e0) ** 2 + (x1 - e1) ** 2 + (x2 - e2) ** 2
                        if d2 < best:
                            best = d2
                            c0, c1, c2 = e0, e1, e2
    return c0, c1, c2, np.sqrt(best)


@nb.njit(cache=True)
def _dist2(X0, i, x0, x1, x2):
    """|x - X0[i]|^2, an upper bound of the squared distance of x from the voxel
    surface (X0[i] is a corner of the candidate faces of i)."""
    return (x0 - X0[i, 0]) ** 2 + (x1 - X0[i, 1]) ** 2 + (x2 - X0[i, 2]) ** 2


@nb.njit(cache=True)
def _tube(x0, x1, x2, X0, quads, p_ptr, p_idx, fmin, fmax, i, r):
    """(x0, x1, x2) pulled back to distance r from the voxel surface. Skipped
    (no-op) when |x - X0[i]| <= r already bounds the distance."""
    if _dist2(X0, i, x0, x1, x2) <= r * r:
        return x0, x1, x2
    c0, c1, c2, d = _closest(x0, x1, x2, quads, p_ptr, p_idx, fmin, fmax, i)
    if d > r:
        s = r / d
        return c0 + (x0 - c0) * s, c1 + (x1 - c1) * s, c2 + (x2 - c2) * s
    return x0, x1, x2


@nb.njit(parallel=True, cache=True)
def _max_distance(X, X0, quads, p_ptr, p_idx, fmin, fmax):
    """max_i dist(X[i], voxel surface). The distance is evaluated only where its
    bound |X - X0| exceeds a threshold t, halving t until the maximum found
    is at least t (then no other point can exceed it)."""
    n = len(X)
    ub = np.empty(n)
    for i in nb.prange(n):
        ub[i] = np.sqrt(_dist2(X0, i, X[i, 0], X[i, 1], X[i, 2]))
    t = 0.5 * ub.max()
    lo = np.inf  # points with ub > lo are done
    D = np.zeros(n)
    while True:
        for i in nb.prange(n):
            if t < ub[i] <= lo:
                D[i] = _closest(X[i, 0], X[i, 1], X[i, 2], quads, p_ptr, p_idx, fmin, fmax, i)[3]
        best = D.max()
        if best >= t or t == 0.0:
            return best
        lo, t = t, 0.5 * t if t > 1e-12 * ub.max() else 0.0


def _csr(M):
    M = sp.csr_matrix(M)
    M.sum_duplicates()
    M.sort_indices()
    return M.indptr.astype(np.int64), M.indices.astype(np.int64), M.data.astype(np.float64)


@nb.njit(cache=True)
def _insert_sorted(buf, n, v):
    """Insert v into the sorted buf[:n] unless present; returns the new length."""
    j = n
    while j > 0 and buf[j - 1] > v:
        j -= 1
    if j > 0 and buf[j - 1] == v:
        return n
    for t in range(n, j, -1):
        buf[t] = buf[t - 1]
    buf[j] = v
    return n + 1


@nb.njit(cache=True)
def _incident(quads, n):
    """CSR (indptr, indices): per point its quads, in increasing order."""
    ptr = np.zeros(n + 1, np.int64)
    for q in range(len(quads)):
        for k in range(4):
            ptr[quads[q, k] + 1] += 1
    for i in range(n):
        ptr[i + 1] += ptr[i]
    idx = np.empty(ptr[n], np.int64)
    fill = ptr[:-1].copy()
    for q in range(len(quads)):
        for k in range(4):
            i = quads[q, k]
            idx[fill[i]] = q
            fill[i] += 1
    return ptr, idx


@nb.njit(parallel=True, cache=True)
def _edge_neighbours(quads, labels, p_ptr, p_idx):
    """CSR (indptr, indices), sorted: per quad the quads sharing an edge with
    it that have the same labels."""
    m = len(quads)
    nb_ = np.full((m, 12), -1, np.int64)  # at most 3 per edge (4 quads meet)
    cnt = np.zeros(m, np.int64)
    for q in nb.prange(m):
        buf = np.empty(16, np.int64)
        nn = 0
        for e in range(4):
            a, b = quads[q, e], quads[q, (e + 1) % 4]
            for k in range(p_ptr[a], p_ptr[a + 1]):
                g = p_idx[k]
                if g == q or labels[g, 0] != labels[q, 0] or labels[g, 1] != labels[q, 1]:
                    continue
                for c in range(4):
                    if quads[g, c] == b:
                        if nn < len(buf):
                            nn = _insert_sorted(buf, nn, g)
                        break
        cnt[q] = min(nn, 12)
        for t in range(min(nn, 12)):
            nb_[q, t] = buf[t]
    ptr = np.zeros(m + 1, np.int64)
    ptr[1:] = np.cumsum(cnt)
    idx = np.empty(ptr[m], np.int64)
    for q in nb.prange(m):
        idx[ptr[q] : ptr[q + 1]] = nb_[q, : cnt[q]]
    return ptr, idx


@nb.njit(parallel=True, cache=True)
def _fair(
    X0, quads, fixed, junction,
    gE, gq, gax, gsg, gmin,  # planar guard (smoothing.planar_guard)
    a_ptr, a_idx, a_val,  # stencils A (L = I - A)
    lt_ptr, lt_idx, lt_val,  # L^T
    bound,  # Gershgorin bound of the row sums of |L^T L|
    p_ptr, p_idx,  # incident quads
    fmin, fmax,  # bounding boxes of the quads (distance limit)
    lo, hi,  # the box: every coordinate is clamped to [lo, hi]
    alpha, h, r, stiffness, step, momentum,
    use_dead_zone, use_tube, iterations,
):  # fmt: skip
    """Pass 1: iterations heavy-ball Jacobi steps, without the volume term."""
    n, m = len(X0), len(quads)
    X = X0.copy()
    vel = np.zeros_like(X0)  # heavy-ball velocity
    Xn = np.empty_like(X0)
    Y = np.empty_like(X0)
    T = np.empty_like(X0)
    w = np.ones(n)
    an = np.empty((m, 3))
    r98 = 0.98 * r
    held = np.zeros(n, np.bool_)
    for _ in range(iterations):
        for q in nb.prange(m):
            _area_vector(X, quads, q, an[q])

        # per point: L x, targets, spring weights
        for i in nb.prange(n):
            for a in range(3):
                y = X[i, a]
                for k in range(a_ptr[i], a_ptr[i + 1]):
                    y -= a_val[k] * X[a_idx[k], a]
                Y[i, a] = y
            N0 = N1 = N2 = 0.0
            for k in range(p_ptr[i], p_ptr[i + 1]):
                q = p_idx[k]
                N0 += an[q, 0]
                N1 += an[q, 1]
                N2 += an[q, 2]
            nn = np.sqrt(N0**2 + N1**2 + N2**2) + 1e-300
            u0, u1, u2 = X[i, 0] - X0[i, 0], X[i, 1] - X0[i, 1], X[i, 2] - X0[i, 2]
            t0, t1, t2 = X[i, 0], X[i, 1], X[i, 2]
            if use_dead_zone:
                if junction[i]:
                    ru = u0**2 + u1**2 + u2**2
                    s = min(1.0, h / max(np.sqrt(ru), 1e-300))
                    t0, t1, t2 = X0[i, 0] + u0 * s, X0[i, 1] + u1 * s, X0[i, 2] + u2 * s
                else:
                    un = u0 * N0 / nn + u1 * N1 / nn + u2 * N2 / nn
                    un = min(max(un, -h), h)
                    t0 = X0[i, 0] + un * N0 / nn
                    t1 = X0[i, 1] + un * N1 / nn
                    t2 = X0[i, 2] + un * N2 / nn
            if use_tube:
                t0, t1, t2 = _tube(t0, t1, t2, X0, quads, p_ptr, p_idx, fmin, fmax, i, r)
                at_limit = False
                if _dist2(X0, i, X[i, 0], X[i, 1], X[i, 2]) > r98 * r98:
                    d = _closest(X[i, 0], X[i, 1], X[i, 2], quads, p_ptr, p_idx, fmin, fmax, i)[3]
                    at_limit = d > r98
                w[i] = min(max(3 * w[i], 10.0), stiffness) if at_limit else 1.0
            T[i, 0], T[i, 1], T[i, 2] = t0, t1, t2

        # per point: accelerated Jacobi step
        for i in nb.prange(n):
            z0 = z1 = z2 = 0.0
            for k in range(lt_ptr[i], lt_ptr[i + 1]):
                j, v = lt_idx[k], lt_val[k]
                z0 += v * Y[j, 0]
                z1 += v * Y[j, 1]
                z2 += v * Y[j, 2]
            d = w[i] + alpha * bound[i]
            for a in range(3):
                if fixed[i, a]:
                    Xn[i, a] = X0[i, a]
                else:
                    z = z0 if a == 0 else (z1 if a == 1 else z2)
                    g = w[i] * (X[i, a] - T[i, a]) + alpha * z
                    vel[i, a] = momentum * vel[i, a] - step * g / d
                    x = X[i, a] + vel[i, a]
                    if x < lo[a] or x > hi[a]:  # on the box: stop there
                        x = min(max(x, lo[a]), hi[a])
                        vel[i, a] = 0.0
                    Xn[i, a] = x
        if len(gq):
            held[:] = False
            if _guard(X, Xn, gE, gq, gax, gsg, gmin, held):
                raise RuntimeError("planar guard: invalid quads before a fairing step")
            for i in range(n):
                if held[i]:
                    vel[i, 0] = vel[i, 1] = vel[i, 2] = 0.0
        X, Xn = Xn, X
    return X


@nb.njit(cache=True)
def _area_vector(X, quads, q, out):
    """Area vector of quad q (half the cross product of its diagonals)."""
    for a in range(3):
        u, v = (a + 1) % 3, (a + 2) % 3
        du0 = X[quads[q, 2], u] - X[quads[q, 0], u]
        dv0 = X[quads[q, 2], v] - X[quads[q, 0], v]
        du1 = X[quads[q, 3], u] - X[quads[q, 1], u]
        dv1 = X[quads[q, 3], v] - X[quads[q, 1], v]
        out[a] = 0.5 * (du0 * dv1 - dv0 * du1)


@nb.njit(cache=True)
def _box_chunk(V, area, e_ptr, e_idx, k, lo, hi, outV, outA):
    """_box_filter for the quads lo:hi (serial; owns its work arrays)."""
    seen = np.full(len(V), -1, np.int64)  # the last quad that visited
    cap = 1024
    queue = np.empty(cap, np.int64)
    dist = np.empty(cap, np.int64)
    for q in range(lo, hi):
        seen[q] = q
        queue[0], dist[0] = q, 0
        nq, head = 1, 0
        sv = 0.0
        sa = 0.0
        while head < nq:
            g, d = queue[head], dist[head]
            head += 1
            sv += V[g]
            sa += area[g]
            if d == k:
                continue
            for kk in range(e_ptr[g], e_ptr[g + 1]):
                h = e_idx[kk]
                if seen[h] != q:
                    seen[h] = q
                    if nq == cap:
                        cap *= 2
                        qn = np.empty(cap, np.int64)
                        dn = np.empty(cap, np.int64)
                        qn[:nq] = queue[:nq]
                        dn[:nq] = dist[:nq]
                        queue, dist = qn, dn
                    queue[nq], dist[nq] = h, d + 1
                    nq += 1
        outV[q] = sv
        outA[q] = sa


@nb.njit(parallel=True, cache=True)
def _box_filter(V, area, e_ptr, e_idx, k, nchunk):
    """
    Per quad q: the sums of V and area over the quads within k edge steps of q
    (breadth-first over the edge neighbours, same label pair). Sums run in
    breadth-first order, which only depends on the quads within reach, not on
    the chunks.
    """
    m = len(V)
    outV = np.empty(m)
    outA = np.empty(m)
    B = (m + nchunk - 1) // nchunk
    for c in nb.prange(nchunk):
        _box_chunk(V, area, e_ptr, e_idx, k, c * B, min(m, (c + 1) * B), outV, outA)
    return outV, outA


@nb.njit(parallel=True, cache=True)
def _swept(X, X0, quads):
    """Per quad: area vector, area and the volume swept from X0 to X."""
    m = len(quads)
    an = np.empty((m, 3))
    area = np.empty(m)
    V = np.empty(m)
    for q in nb.prange(m):
        a0 = np.empty(3)
        _area_vector(X0, quads, q, a0)
        _area_vector(X, quads, q, an[q])
        area[q] = np.sqrt(an[q, 0] ** 2 + an[q, 1] ** 2 + an[q, 2] ** 2)
        s = 0.0
        for a in range(3):
            ub = 0.0
            for k in range(4):
                ub += X[quads[q, k], a] - X0[quads[q, k], a]
            s += 0.5 * (an[q, a] + a0[a]) * 0.25 * ub
        V[q] = s
    return an, area, V


@nb.njit(parallel=True, cache=True)
def _shift(X, an, area, off, fixed, p_ptr, p_idx, lo, hi):
    """
    Move every point back by the offsets off of its quads along their
    normals, clamped to the box [lo, hi].
    """
    Xn = X.copy()
    for i in nb.prange(len(X)):
        sh = np.zeros(3)
        ar = 0.0
        for k in range(p_ptr[i], p_ptr[i + 1]):
            q = p_idx[k]
            for a in range(3):
                sh[a] -= off[q] * an[q, a]
            ar += area[q]
        for a in range(3):
            if not fixed[i, a]:
                Xn[i, a] = min(max(Xn[i, a] + sh[a] / max(ar, 1e-300), lo[a]), hi[a])
    return Xn


def _restore_volume(X, X0, quads, fixed, p_ptr, p_idx, e_ptr, e_idx, rounds, lo, hi):
    """
    Pass 2: per quad the volume swept from X0 to X, averaged over the quads
    within `rounds` edge steps of the same interface (two box averages of
    radius rounds // 2 and rounds - rounds // 2 in a row, i.e. tent-like
    weights); every point is moved back along the normals of its quads by
    the averaged volume / area, clamped to the box [lo, hi].
    """
    an, area, V = _swept(X, X0, quads)
    nchunk = 4 * nb.get_num_threads()
    h = rounds // 2
    if h:
        V, A = _box_filter(V, area, e_ptr, e_idx, h, nchunk)
    else:
        A = area
    V, A = _box_filter(V, A, e_ptr, e_idx, rounds - h, nchunk)
    off = V / np.maximum(A, 1e-300)
    return _shift(X, an, area, off, fixed, p_ptr, p_idx, lo, hi)


@nb.njit(parallel=True, cache=True)
def _project(X, X0, fixed, quads, p_ptr, p_idx, fmin, fmax, r, lo, hi):
    """
    Final projection into the distance limit, then the free coordinates into
    the margin box [lo, hi] (in place; the clamp moves a point by at most the
    margin from the box, which contains S0).
    """
    for i in nb.prange(len(X)):
        t0, t1, t2 = _tube(X[i, 0], X[i, 1], X[i, 2], X0, quads, p_ptr, p_idx, fmin, fmax, i, r)
        X[i, 0] = X0[i, 0] if fixed[i, 0] else min(max(t0, lo[0]), hi[0])
        X[i, 1] = X0[i, 1] if fixed[i, 1] else min(max(t1, lo[1]), hi[1])
        X[i, 2] = X0[i, 2] if fixed[i, 2] else min(max(t2, lo[2]), hi[2])


@nb.njit(parallel=True, cache=True)
def _quad_boxes(X0, quads):
    """Per quad the coordinate-wise min and max of its corners."""
    m = len(quads)
    fmin = np.empty((m, 3))
    fmax = np.empty((m, 3))
    for q in nb.prange(m):
        for a in range(3):
            lo = hi = X0[quads[q, 0], a]
            for c in range(1, 4):
                x = X0[quads[q, c], a]
                lo, hi = min(lo, x), max(hi, x)
            fmin[q, a], fmax[q, a] = lo, hi
    return fmin, fmax


def _bounds_mask(X, bounds, tol=1e-6):
    """(n, 3) mask of the coordinates on the faces of the box bounds."""
    lo, hi = np.asarray(bounds, float).reshape(3, 2).T
    atol = tol * np.max(hi - lo)
    return np.isclose(X, lo, rtol=0, atol=atol) | np.isclose(X, hi, rtol=0, atol=atol)


def fair_surface_net(
    surf,
    dx,
    influence_radius=None,
    iterations=None,
    alpha=30.0,
    dead_zone=0.5,
    max_distance=1.0,
    volume=True,
    label_name="boundary_labels",
    fix_bounds=True,
    bounds=None,
    box_margin=0.1,
    fixed=None,
    guard=None,
    stiffness=100.0,
    degenerate=None,
    step=1.0,
    momentum=0.8,
    volume_rings=None,
    volume_repeats=None,
):
    """
    Fair the unsmoothed surface net surf (quads from surface_quads, cell data
    label_name (a, b), normals out of a) and triangulate it, see the module
    docstring.

    influence_radius: range of influence in voxels (L-infinity, see the module
    docstring): no point depends on a voxel further away, bitwise. It sets
    the number of fairing steps, (influence_radius - 1 - volume_repeats *
    (1 + volume_rings)) // 2 with volume, else (influence_radius - 1) // 2;
    iterations (if given) may be smaller. None (default): no limit,
    iterations steps (default 100); the range is still finite, see the
    field data.
    volume_rings: reach (in edge steps between faces) of the average of
    the volume change in each of the volume_repeats volume restorations;
    default influence_radius // 8 (32 without a limit), which leaves about
    3/4 of the range to the fairing steps. Its cost grows with its square.
    volume_repeats: default 1 (2 without a limit).
    dead_zone and max_distance are in units of dx; None switches them off.
    fix_bounds keeps points on the faces of the box bounds (default: the
    bounding box of surf; pass the image bounds for the range guarantee) on
    their box face, and every other coordinate box_margin dx inside the box
    faces (clamped; given bounds clamp even without fix_bounds); fixed:
    optional (n_points, 3) mask of coordinates that must not move. guard:
    optional smoothing.PlanarGuard (planar_guard, valid at the points of surf)
    of planar quads that every step keeps valid: a
    point whose move would break one keeps its position in that step (in the
    fairing steps, the volume restoration and the final projection; such a
    point may stay outside the distance limit). step, momentum: of the heavy-ball Jacobi steps.
    Returns the triangulated surface with the cell data repeated per triangle
    and the field data 'dx', 'influence_radius' (the range of influence of
    this run in voxels, 2 iterations + 1 + volume_repeats (1 + volume_rings)
    with volume, else 2 iterations + 1, at most the influence_radius asked) and
    'max_surface_distance' (the largest distance of a point from the voxel
    surface, in the unit of dx).
    """
    if volume_rings is None:
        volume_rings = 32 if influence_radius is None else max(1, int(influence_radius) // 8)
    if volume_repeats is None:
        volume_repeats = 2 if influence_radius is None else 1
    if influence_radius is None:
        iterations = 100 if iterations is None else int(iterations)
    else:
        reserved = 1 + int(volume_repeats) * (1 + int(volume_rings)) if volume else 1
        n_max = (int(influence_radius) - reserved) // 2
        if n_max < 1:
            raise ValueError(
                f"influence_radius={influence_radius} allows no step (>= {reserved + 2} needed)"
            )
        if iterations is not None and iterations > n_max:
            raise ValueError(
                f"iterations={iterations} exceeds the {n_max} steps that "
                f"influence_radius={influence_radius} allows"
            )
        iterations = n_max if iterations is None else int(iterations)

    quads = np.ascontiguousarray(surf.regular_faces, dtype=np.int64)
    X0 = np.ascontiguousarray(surf.points, dtype=np.float64)
    n = len(X0)
    labels = np.asarray(surf.cell_data[label_name]).astype(np.int64)

    fixed = np.zeros((n, 3), bool) if fixed is None else np.asarray(fixed, bool).copy()
    box = surf.bounds if bounds is None else bounds
    if fix_bounds:
        fixed |= _bounds_mask(X0, box)
    if fix_bounds or bounds is not None:
        m = float(box_margin) * dx
        lo, hi = np.asarray(box, float).reshape(3, 2).T
        box_lo, box_hi = np.ascontiguousarray(lo + m), np.ascontiguousarray(hi - m)
    else:
        box_lo, box_hi = np.full(3, -np.inf), np.full(3, np.inf)

    A = build_stencils(quads, n, degenerate=degenerate)
    L = sp.eye(n, format="csr") - A
    a_ptr, a_idx, a_val = _csr(A)
    lt_ptr, lt_idx, lt_val = _csr(L.T)
    # sum_j |(L^T L)_ij| <= sum_k |L_ki| sum_j |L_kj|; csr products sum in index order
    l_ptr, l_idx, l_val = _csr(L)
    rowabs = sp.csr_matrix((np.abs(l_val), l_idx, l_ptr), shape=(n, n)) @ np.ones(n)
    bound = sp.csr_matrix((np.abs(lt_val), lt_idx, lt_ptr), shape=(n, n)) @ rowabs

    p_ptr, p_idx = _incident(quads, n)
    junction = np.zeros(n, bool)  # faces of more than one label pair
    for c in range(2):
        lab = np.repeat(labels[:, c], 4)
        lo = np.full(n, np.iinfo(np.int64).max)
        hi = np.full(n, np.iinfo(np.int64).min)
        np.minimum.at(lo, quads.ravel(), lab)
        np.maximum.at(hi, quads.ravel(), lab)
        junction |= lo != hi

    fmin, fmax = _quad_boxes(X0, quads)
    r = np.inf if max_distance is None else max_distance * dx
    g = empty_guard() if guard is None else guard
    X = _fair(
        X0, quads, fixed, junction, g.extra, g.quads, g.axis, g.sign, g.min_area2,
        a_ptr, a_idx, a_val, lt_ptr, lt_idx, lt_val, bound,
        p_ptr, p_idx, fmin, fmax, box_lo, box_hi,
        float(alpha), 0.0 if dead_zone is None else dead_zone * dx, r, float(stiffness),
        float(step), float(momentum), dead_zone is not None, max_distance is not None,
        iterations,
    )  # fmt: skip
    if volume:
        e_ptr, e_idx = _edge_neighbours(quads, labels, p_ptr, p_idx)
        for _ in range(int(volume_repeats)):
            Xv = _restore_volume(
                X, X0, quads, fixed, p_ptr, p_idx, e_ptr, e_idx, int(volume_rings), box_lo, box_hi
            )
            apply_guard(X, Xv, guard)
            X = Xv
    if max_distance is not None:
        Xp = X.copy()
        _project(Xp, X0, fixed, quads, p_ptr, p_idx, fmin, fmax, r, box_lo, box_hi)
        apply_guard(X, Xp, guard)
        X = Xp

    if max_distance is not None:
        dmax = float(_max_distance(X, X0, quads, p_ptr, p_idx, fmin, fmax))
    else:
        dmax = np.nan

    tris = triangulate_quads(X, quads)
    out = pv.PolyData(X, np.column_stack([np.full(len(tris), 3), tris]).ravel())
    for name, arr in surf.cell_data.items():
        out.cell_data[name] = np.repeat(arr, 2, axis=0)
    out.field_data["dx"] = [dx]
    reach = 2 * iterations + 1 + (int(volume_repeats) * (1 + int(volume_rings)) if volume else 0)
    out.field_data["influence_radius"] = [reach]
    out.field_data["max_surface_distance"] = [dmax]
    return out
