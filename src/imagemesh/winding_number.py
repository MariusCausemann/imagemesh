"""
Fast generalized winding numbers of triangle meshes (Barill et al. 2018,
"Fast Winding Numbers for Soups and Clouds").

The triangles are stored in a median-split kd-tree. For each node, the
contribution to the solid angle is approximated by a third order Taylor
expansion around the node's area-weighted centroid if the query point is
farther away than beta times the node radius; in leaves close to the query
point, the solid angles of the triangles are computed exactly. Expansion
order and far field criterion are those of libigl's fast_winding_number for
triangle soups (UT_SolidAngle from Houdini, "order 2" there).

The solid angle is linear in the faces, so several weighted channels share
one traversal: each face carries a weight per channel. This is used by
label_points to classify points of a multi-label surface in one go, with one
channel per bit of the label index.
"""

import numba as nb
import numpy as np


@nb.njit(cache=True)
def _split(cent, perm, s, e):
    """Partition perm[s:e] at its median m along the longest axis of the
    centroids' bounding box (quickselect), and return m."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for i in range(s, e):
        for d in range(3):
            lo[d] = min(lo[d], cent[perm[i], d])
            hi[d] = max(hi[d], cent[perm[i], d])
    axis = np.argmax(hi - lo)
    m = s + (e - s) // 2
    a, b = s, e - 1
    while a < b:
        pivot = cent[perm[(a + b) // 2], axis]
        i, j = a, b
        while i <= j:
            while cent[perm[i], axis] < pivot:
                i += 1
            while cent[perm[j], axis] > pivot:
                j -= 1
            if i <= j:
                perm[i], perm[j] = perm[j], perm[i]
                i += 1
                j -= 1
        if m <= j:
            b = j
        elif m >= i:
            a = i
        else:
            break
    return m


@nb.njit(parallel=True, cache=True)
def _build_tree(cent, leaf_size):
    """Median-split kd-tree over the face centroids, built breadth first, so
    the nodes of each level have the contiguous ids levels[l]:levels[l + 1]
    (children after their parents). Node k holds the faces
    perm[start[k]:end[k]]; leaves have left[k] == -1."""
    n = len(cent)
    max_nodes = 2 * n
    perm = np.arange(n)
    start = np.empty(max_nodes, np.int64)
    end = np.empty(max_nodes, np.int64)
    left = np.full(max_nodes, -1, np.int64)
    levels = np.zeros(65, np.int64)  # depth <= log2(#faces) + 1
    start[0], end[0] = 0, n
    n_levels = 0
    a, b = 0, 1
    while b > a:
        mid = np.full(b - a, -1, np.int64)
        for k in nb.prange(a, b):
            if end[k] - start[k] > leaf_size:
                mid[k - a] = _split(cent, perm, start[k], end[k])
        n_nodes = b
        for k in range(a, b):
            s, e, m = start[k], end[k], mid[k - a]
            if m >= 0:
                start[n_nodes], end[n_nodes] = s, m
                start[n_nodes + 1], end[n_nodes + 1] = m, e
                left[k] = n_nodes
                n_nodes += 2
        n_levels += 1
        levels[n_levels] = b
        a, b = b, n_nodes
    return perm, start[:b], end[:b], left[:b], levels[: n_levels + 1]


@nb.njit(parallel=True, cache=True)
def _face_geometry(V, F):
    """Centroids and area vectors (b - a) x (c - a) / 2 of the faces."""
    cent = np.empty((len(F), 3))
    N = np.empty((len(F), 3))
    for f in nb.prange(len(F)):
        a, b, c = V[F[f, 0]], V[F[f, 1]], V[F[f, 2]]
        cent[f] = (a + b + c) / 3.0
        N[f] = 0.5 * np.cross(b - a, c - a)
    return cent, N


# Moments of a node per channel, around its centre P (y = x - P, n the unit
# normal): D_i = int n_i (order 1), M_ij = int n_i y_j (order 2) and
# T_ijk = int n_i y_j y_k (order 3, symmetric in jk, so only the pairs
# (_PJ[p], _PK[p]) are stored), flattened to [D (3), M (9), T (18)]
_PJ = np.array([0, 1, 2, 0, 0, 1])
_PK = np.array([0, 1, 2, 1, 2, 2])
_NMOM = 30


@nb.njit(cache=True)
def _face_moments(V, F, N, f, P, g):
    """Moments g of face f with area vector N[f] around P. The triangle's
    second moment is exact: int y y^T = A/12 (sum_v y_v y_v^T + s s^T), with
    s = sum_v y_v."""
    y = np.empty((3, 3))
    s = np.zeros(3)
    for v in range(3):
        for d in range(3):
            y[v, d] = V[F[f, v], d] - P[d]
            s[d] += y[v, d]
    for i in range(3):
        g[i] = N[f, i]
        for j in range(3):
            g[3 + 3 * i + j] = N[f, i] * s[j] / 3.0
        for p in range(6):
            j, k = _PJ[p], _PK[p]
            yy = y[0, j] * y[0, k] + y[1, j] * y[1, k] + y[2, j] * y[2, k]
            g[12 + 6 * i + p] = N[f, i] / 12.0 * (yy + s[j] * s[k])


@nb.njit(cache=True)
def _shift_add(src, e, dst):
    """Add the moments src around P + e to dst, around P."""
    for i in range(3):
        Di = src[i]
        dst[i] += Di
        for j in range(3):
            dst[3 + 3 * i + j] += src[3 + 3 * i + j] + Di * e[j]
        for p in range(6):
            j, k = _PJ[p], _PK[p]
            dst[12 + 6 * i + p] += (
                src[12 + 6 * i + p]
                + src[3 + 3 * i + j] * e[k]
                + src[3 + 3 * i + k] * e[j]
                + Di * e[j] * e[k]
            )


@nb.njit(parallel=True, cache=True)
def _node_moments(V, F, W, N, cent, perm, start, end, left, levels):
    """Per node: area-weighted centroid P, squared distance R2 from P to the
    farthest corner of the node's bounding box, and the weighted moments per
    channel. Leaves are summed directly, inner nodes are combined from their
    children, level by level from the bottom."""
    n_nodes = len(start)
    K = W.shape[1]
    P = np.zeros((n_nodes, 3))
    lo = np.full((n_nodes, 3), np.inf)
    hi = np.full((n_nodes, 3), -np.inf)
    area = np.zeros(n_nodes)
    mom = np.zeros((n_nodes, K, _NMOM))
    for lev in range(len(levels) - 2, -1, -1):
        for k in nb.prange(levels[lev], levels[lev + 1]):
            if left[k] < 0:
                for i in range(start[k], end[k]):
                    f = perm[i]
                    A = np.sqrt(N[f, 0] ** 2 + N[f, 1] ** 2 + N[f, 2] ** 2)
                    area[k] += A
                    for d in range(3):
                        P[k, d] += A * cent[f, d]
                        for v in range(3):
                            lo[k, d] = min(lo[k, d], V[F[f, v], d])
                            hi[k, d] = max(hi[k, d], V[F[f, v], d])
                if area[k] > 0.0:
                    P[k] /= area[k]
                else:  # degenerate faces only
                    P[k] = cent[perm[start[k]]]
                g = np.empty(_NMOM)
                for i in range(start[k], end[k]):
                    f = perm[i]
                    _face_moments(V, F, N, f, P[k], g)
                    for j in range(K):
                        w = W[f, j]
                        if w != 0.0:
                            for m in range(_NMOM):
                                mom[k, j, m] += w * g[m]
            else:
                cl, cr = left[k], left[k] + 1
                area[k] = area[cl] + area[cr]
                for d in range(3):
                    if area[k] > 0.0:
                        P[k, d] = (area[cl] * P[cl, d] + area[cr] * P[cr, d]) / area[k]
                    else:
                        P[k, d] = 0.5 * (P[cl, d] + P[cr, d])
                    lo[k, d] = min(lo[cl, d], lo[cr, d])
                    hi[k, d] = max(hi[cl, d], hi[cr, d])
                for c in (cl, cr):
                    e = P[c] - P[k]
                    for j in range(K):
                        _shift_add(mom[c, j], e, mom[k, j])
    d = np.maximum(P - lo, hi - P)
    R2 = (d**2).sum(axis=1)
    return P, R2, mom


@nb.njit(cache=True, inline="always")
def _solid_angle(V, F, f, q):
    """Signed solid angle of face f seen from q (Van Oosterom & Strackee
    1983); positive if q lies on the side the normal (b - a) x (c - a) points
    away from."""
    a0, a1, a2 = V[F[f, 0], 0] - q[0], V[F[f, 0], 1] - q[1], V[F[f, 0], 2] - q[2]
    b0, b1, b2 = V[F[f, 1], 0] - q[0], V[F[f, 1], 1] - q[1], V[F[f, 1], 2] - q[2]
    c0, c1, c2 = V[F[f, 2], 0] - q[0], V[F[f, 2], 1] - q[1], V[F[f, 2], 2] - q[2]
    la = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
    lb = np.sqrt(b0 * b0 + b1 * b1 + b2 * b2)
    lc = np.sqrt(c0 * c0 + c1 * c1 + c2 * c2)
    num = a0 * (b1 * c2 - b2 * c1) + a1 * (b2 * c0 - b0 * c2) + a2 * (b0 * c1 - b1 * c0)
    ab = a0 * b0 + a1 * b1 + a2 * b2
    bc = b0 * c0 + b1 * c1 + b2 * c2
    ca = c0 * a0 + c1 * a1 + c2 * a2
    den = la * lb * lc + ab * lc + bc * la + ca * lb
    return 2.0 * np.arctan2(num, den)


@nb.njit(cache=True)
def _expansion_coeffs(x, d2, c):
    """Coefficients c such that c . moments is the third order Taylor
    expansion of the solid angle int n . f(y) dA, f(y) = (x + y)/|x + y|^3,
    around y = 0: f_i + d_j f_i y_j + 1/2 d_jk f_i y_j y_k, with
    d_j f_i = delta_ij/r^3 - 3 x_i x_j/r^5 and
    d_jk f_i = -3 (delta_ij x_k + delta_ik x_j + delta_jk x_i)/r^5
               + 15 x_i x_j x_k/r^7."""
    r3 = 1.0 / (d2 * np.sqrt(d2))
    r5 = r3 / d2
    r7 = r5 / d2
    for i in range(3):
        c[i] = x[i] * r3
        for j in range(3):
            c[3 + 3 * i + j] = (i == j) * r3 - 3.0 * x[i] * x[j] * r5
        for p in range(6):
            j, k = _PJ[p], _PK[p]
            sym = (i == j) * x[k] + (i == k) * x[j] + (j == k) * x[i]
            c[12 + 6 * i + p] = (1.0 if j == k else 2.0) * (
                -1.5 * sym * r5 + 7.5 * x[i] * x[j] * x[k] * r7
            )


@nb.njit(parallel=True, cache=True)
def _winding_numbers(V, F, W, Q, perm, start, end, left, P, R2, mom, beta):
    nq = len(Q)
    K = W.shape[1]
    out = np.zeros((nq, K))
    beta2 = beta * beta
    for iq in nb.prange(nq):
        q = Q[iq]
        x = np.empty(3)
        c = np.empty(_NMOM)
        stack = np.empty(64, np.int64)  # depth <= log2(#faces) + 1
        stack[0] = 0
        top = 1
        while top > 0:
            top -= 1
            k = stack[top]
            for d in range(3):
                x[d] = P[k, d] - q[d]
            d2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
            if d2 > beta2 * R2[k]:
                _expansion_coeffs(x, d2, c)
                for j in range(K):
                    s = 0.0
                    for m in range(_NMOM):
                        s += c[m] * mom[k, j, m]
                    out[iq, j] += s
            elif left[k] < 0:
                for i in range(start[k], end[k]):
                    f = perm[i]
                    om = _solid_angle(V, F, f, q)
                    for j in range(K):
                        out[iq, j] += W[f, j] * om
            else:
                stack[top] = left[k]
                stack[top + 1] = left[k] + 1
                top += 2
    return out / (4.0 * np.pi)


def weighted_winding_numbers(V, F, W, Q, beta=2.5, leaf_size=8):
    """
    Generalized winding numbers of query points with respect to a triangle
    mesh, with per-face weights in several channels.

    Parameters:
        V: (nv, 3) vertex coordinates
        F: (nf, 3) triangles
        W: (nf, K) face weights, one column per channel
        Q: (nq, 3) query points
        beta: far field criterion, distance > beta * node radius (distance
            from the node centre to the farthest corner of its bounding box),
            as in libigl; larger is more accurate and slower. The default 2.5
            is more accurate than libigl's 2 with its 4-ary tree; np.inf gives the
            exact (slow) sum
        leaf_size: maximum number of faces in a leaf of the tree

    Returns:
        (nq, K) array, column j is sum_f W[f, j] * solid_angle_f(q) / (4 pi).
        For a closed, outward oriented surface and unit weights it is 1 inside
        and 0 outside.
    """
    V = np.ascontiguousarray(V, dtype=np.float64)
    F = np.ascontiguousarray(F, dtype=np.int64)
    W = np.ascontiguousarray(W, dtype=np.float64)
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    if len(F) == 0:
        return np.zeros((len(Q), W.shape[1]))
    cent, N = _face_geometry(V, F)
    perm, start, end, left, levels = _build_tree(cent, leaf_size)
    P, R2, mom = _node_moments(V, F, W, N, cent, perm, start, end, left, levels)
    return _winding_numbers(V, F, W, Q, perm, start, end, left, P, R2, mom, beta)


def winding_number(V, F, Q, **kwargs):
    """Generalized winding numbers (nq,) of the points Q with respect to the
    triangle mesh (V, F); see weighted_winding_numbers."""
    return weighted_winding_numbers(V, F, np.ones((len(F), 1)), Q, **kwargs)[:, 0]


def label_points(V, F, boundary_labels, Q, **kwargs):
    """
    Label of the region containing each query point, for a multi-label
    surface whose faces separate the regions boundary_labels[f, 0] and
    boundary_labels[f, 1] (0 is the background/outside), with a consistent
    orientation: the normals point either all from column 0 to column 1, or
    all the other way.

    The labels are mapped to indices 1..n (0 stays 0), and channel j carries
    the indicator of bit j of the index: face weight bit_j(a) - bit_j(b).
    Since the regions partition the domain, each channel is 0 or +-1, so all
    labels are found with ceil(log2(n + 1)) channels in a single traversal.

    Returns:
        (nq,) labels, 0 for points outside all regions.
    """
    boundary_labels = np.asarray(boundary_labels)
    lut = np.union1d([0], boundary_labels)
    code = np.searchsorted(lut, boundary_labels)
    K = max(1, int(code.max()).bit_length())
    bits = (code[:, :, None] >> np.arange(K)) & 1
    W = bits[:, 0] - bits[:, 1]
    wn = weighted_winding_numbers(V, F, W, Q, **kwargs)
    idx = (np.abs(wn) > 0.5) @ (1 << np.arange(K))
    # bits of all regions near a point can mix to an index that does not exist
    idx[idx >= len(lut)] = 0
    return lut[idx]
