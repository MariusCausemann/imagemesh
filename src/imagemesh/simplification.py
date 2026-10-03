"""
Feature preserving simplification of multi-label surfaces, e.g. from pyvista's
contour_labels, with a bounded deviation from the input.

The surface is simplified by half-edge collapses u -> v (v is an existing
vertex), applied in rounds of non-conflicting collapses, so that each round is
a parallel numba kernel on plain face arrays:

* vertex classes (computed once on the input): interior vertices (all incident
  edges have valence 2 and all incident faces the same label pair) collapse
  along any edge, curve vertices (two feature edges, i.e. edges of valence != 2
  or between different label pairs) only along their feature curve, corners
  (all other vertices) never move. Points on a bounding box plane only collapse
  into points on the same plane(s).
* a collapse only replaces a vertex index, so faces keep their winding and
  their labels. It is rejected if it violates the link condition, creates a
  duplicate face, rotates a face normal by more than max_angle, creates a bad
  triangle, an edge longer than max_edge_length or a sharp wedge (two faces
  meeting at an edge at less than min_wedge_angle).
* deviation: every input vertex (and optionally face centroid) is owned by an
  output face of the same label pair within epsilon, with a normal within
  max_angle of the input normal. A collapse is only accepted if all points
  owned by the faces around u can be reassigned to the new faces around v,
  and if sample points of each modified face (centroid, 3 points halfway to
  the vertices, midpoints of the new edges) are within epsilon of the
  input faces (of the same label pair) whose centroids were owned by the faces
  around u.
* orientation: normals are compared oriented by the label order (flipped if
  label[0] > label[1]), and the sample points of the modified faces must be
  within epsilon of an input face whose oriented normal is less than 90
  degrees apart, so no face flips the sides of its labels against the input.
* priority: quadric error (Garland-Heckbert) of placing u at v, plus a small
  edge length term, so that flat regions are coarsened uniformly.
* placement="qem": v is additionally moved to the position minimizing the
  quadric error of u and v, restricted to the feature curve (curve vertices
  move along the collapsed edge, corners never move) and to the bounding box
  planes of v, and clamped to one edge length around the edge midpoint. Both
  u -> v at the optimal position and at v are candidates, so this is never
  more restrictive than the endpoint placement. As v moves, all faces around v
  change as well, so the checks above then apply to all faces around u and v.
  Faces still only change a vertex index, so the winding (normal orientation)
  and the (ordered) labels of every face are kept.
* non-conflicting collapses: each vertex proposes its best valid collapse (in
  parallel). The proposals are then accepted greedily in order of increasing
  cost if their face star (faces around u and v) is disjoint from the stars
  of the proposals accepted before, so all accepted collapses can be applied
  at once.
"""

import time

import numba as nb
import numpy as np
import pyvista as pv

INTERIOR, CURVE, CORNER = 0, 1, 2


# --------------------------------------------------------------------------
# small geometric helpers
# --------------------------------------------------------------------------


@nb.njit(cache=True, inline="always")
def _cross(ax, ay, az, bx, by, bz):
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


@nb.njit(cache=True)
def _tri_normal(V, a, b, c):
    """Unnormalized normal (length = 2 * area) of triangle (a, b, c)."""
    return _cross(
        V[b, 0] - V[a, 0],
        V[b, 1] - V[a, 1],
        V[b, 2] - V[a, 2],
        V[c, 0] - V[a, 0],
        V[c, 1] - V[a, 1],
        V[c, 2] - V[a, 2],
    )


@nb.njit(cache=True)
def _sqdist(V, a, b):
    dx = V[a, 0] - V[b, 0]
    dy = V[a, 1] - V[b, 1]
    dz = V[a, 2] - V[b, 2]
    return dx * dx + dy * dy + dz * dz


@nb.njit(cache=True)
def _quality(V, a, b, c):
    """4 sqrt(3) area / sum of squared edge lengths, 1 for equilateral."""
    nx, ny, nz = _tri_normal(V, a, b, c)
    s = _sqdist(V, a, b) + _sqdist(V, b, c) + _sqdist(V, c, a)
    if s == 0.0:
        return 0.0
    return 2.0 * np.sqrt(3.0) * np.sqrt(nx * nx + ny * ny + nz * nz) / s


@nb.njit(cache=True)
def _max_opening_cos(V, a, x, T, n, Opp):
    """
    Cosine of the smallest angle between two of the faces T[:n] around the
    edge (a, x); -1 for less than two faces. Opp is (>= n, 3) scratch space.
    """
    for i in range(n):
        o = T[i, 0] + T[i, 1] + T[i, 2] - a - x
        Opp[i, 0], Opp[i, 1], Opp[i, 2] = V[o, 0], V[o, 1], V[o, 2]
    return _opening_cos(V[a], V[x], Opp, n)


@nb.njit(cache=True)
def _opening_cos(A, X, Opp, n):
    """
    Cosine of the smallest angle between two of the faces (A, X, Opp[i]),
    i < n, around the edge (A, X); -1 for less than two faces. Overwrites Opp.
    """
    tx, ty, tz = X[0] - A[0], X[1] - A[1], X[2] - A[2]
    tn = np.sqrt(tx * tx + ty * ty + tz * tz)
    if tn == 0.0:
        return 1.0
    tx, ty, tz = tx / tn, ty / tn, tz / tn
    W = Opp  # overwritten with the unit directions to the opposite points
    for i in range(n):
        wx, wy, wz = Opp[i, 0] - A[0], Opp[i, 1] - A[1], Opp[i, 2] - A[2]
        d = wx * tx + wy * ty + wz * tz
        wx, wy, wz = wx - d * tx, wy - d * ty, wz - d * tz
        wn = np.sqrt(wx * wx + wy * wy + wz * wz)
        if wn == 0.0:
            return 1.0
        W[i, 0], W[i, 1], W[i, 2] = wx / wn, wy / wn, wz / wn
    c = -1.0
    for i in range(n):
        for j in range(i + 1, n):
            c = max(c, W[i, 0] * W[j, 0] + W[i, 1] * W[j, 1] + W[i, 2] * W[j, 2])
    return c


@nb.njit(cache=True)
def _point_tri_sqdist(p, V, a, b, c):
    """Squared distance of point p to triangle (a, b, c) (Ericson, RTCD 5.1.5)."""
    ax, ay, az = V[a, 0], V[a, 1], V[a, 2]
    abx, aby, abz = V[b, 0] - ax, V[b, 1] - ay, V[b, 2] - az
    acx, acy, acz = V[c, 0] - ax, V[c, 1] - ay, V[c, 2] - az
    apx, apy, apz = p[0] - ax, p[1] - ay, p[2] - az
    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0.0 and d2 <= 0.0:
        return apx * apx + apy * apy + apz * apz
    bpx, bpy, bpz = p[0] - V[b, 0], p[1] - V[b, 1], p[2] - V[b, 2]
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    if d3 >= 0.0 and d4 <= d3:
        return bpx * bpx + bpy * bpy + bpz * bpz
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        t = d1 / (d1 - d3)
        qx, qy, qz = apx - t * abx, apy - t * aby, apz - t * abz
        return qx * qx + qy * qy + qz * qz
    cpx, cpy, cpz = p[0] - V[c, 0], p[1] - V[c, 1], p[2] - V[c, 2]
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz
    if d6 >= 0.0 and d5 <= d6:
        return cpx * cpx + cpy * cpy + cpz * cpz
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        t = d2 / (d2 - d6)
        qx, qy, qz = apx - t * acx, apy - t * acy, apz - t * acz
        return qx * qx + qy * qy + qz * qz
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        qx = bpx - t * (V[c, 0] - V[b, 0])
        qy = bpy - t * (V[c, 1] - V[b, 1])
        qz = bpz - t * (V[c, 2] - V[b, 2])
        return qx * qx + qy * qy + qz * qz
    s = va + vb + vc
    if s <= 0.0:  # degenerate triangle
        return min(
            apx * apx + apy * apy + apz * apz,
            bpx * bpx + bpy * bpy + bpz * bpz,
            cpx * cpx + cpy * cpy + cpz * cpz,
        )
    v = vb / s
    w = vc / s
    qx = apx - v * abx - w * acx
    qy = apy - v * aby - w * acy
    qz = apz - v * abz - w * acz
    return qx * qx + qy * qy + qz * qz


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------


@nb.njit(cache=True)
def _csr(keys, n):
    """CSR grouping of the indices of keys (values in [0, n))."""
    ptr = np.zeros(n + 1, np.int64)
    for k in keys:
        ptr[k + 1] += 1
    for i in range(n):
        ptr[i + 1] += ptr[i]
    pos = ptr[:-1].copy()
    idx = np.empty(len(keys), np.int64)
    for i in range(len(keys)):
        k = keys[i]
        idx[pos[k]] = i
        pos[k] += 1
    return ptr, idx


def _vertex_faces(F, nv):
    ptr, idx = _csr(F.ravel(), nv)
    return ptr, idx // 3


@nb.njit(parallel=True, cache=True)
def _quadrics(V, F, vf_ptr, vf_idx):
    """Area weighted plane quadrics, accumulated on the vertices."""
    nv = len(vf_ptr) - 1
    Q = np.zeros((nv, 4, 4))
    # gathered per vertex, in the same (face) order as a scatter over the faces
    for x in nb.prange(nv):
        for jj in range(vf_ptr[x], vf_ptr[x + 1]):
            f = vf_idx[jj]
            a, b, c = F[f, 0], F[f, 1], F[f, 2]
            nx, ny, nz = _tri_normal(V, a, b, c)
            n2 = np.sqrt(nx * nx + ny * ny + nz * nz)
            if n2 == 0.0:
                continue
            p0, p1, p2 = nx / n2, ny / n2, nz / n2
            p = (p0, p1, p2, -(p0 * V[a, 0] + p1 * V[a, 1] + p2 * V[a, 2]))
            w = 0.5 * n2
            for i in range(4):
                for j in range(4):
                    Q[x, i, j] += w * p[i] * p[j]
    return Q


@nb.njit(cache=True)
def _is_disk(u, F, vf_ptr, vf_idx):
    """True if the link of u is a single closed cycle (u is a manifold vertex)."""
    s, e = vf_ptr[u], vf_ptr[u + 1]
    m = e - s
    if m < 3:
        return False
    # link edges
    la = np.empty(m, np.int64)
    lb = np.empty(m, np.int64)
    for i in range(m):
        f = vf_idx[s + i]
        a, b, c = F[f, 0], F[f, 1], F[f, 2]
        if a == u:
            la[i], lb[i] = b, c
        elif b == u:
            la[i], lb[i] = c, a
        else:
            la[i], lb[i] = a, b
    # walk the cycle starting at the first edge, following la -> lb
    cur = lb[0]
    used = np.zeros(m, np.bool_)
    used[0] = True
    for _ in range(m - 1):
        nxt = -1
        for i in range(m):
            if not used[i] and la[i] == cur:
                nxt = i
                break
        if nxt < 0:
            return False
        used[nxt] = True
        cur = lb[nxt]
    return cur == la[0]


@nb.njit(parallel=True, cache=True)
def _classify(F, vf_ptr, vf_idx, feature_degree):
    nv = len(vf_ptr) - 1
    cls = np.empty(nv, np.int8)
    for u in nb.prange(nv):
        d = feature_degree[u]
        if d == 0:
            cls[u] = INTERIOR if _is_disk(u, F, vf_ptr, vf_idx) else CORNER
        elif d == 2:
            cls[u] = CURVE
        else:
            cls[u] = CORNER
    return cls


def _feature_degree(F, patch, nv):
    """Number of feature edges (valence != 2 or different label pairs) per vertex."""
    e = np.sort(np.stack([F, np.roll(F, -1, axis=1)], axis=-1).reshape(-1, 2), axis=1)
    key = e[:, 0] * nv + e[:, 1]
    ukey, inv, valence = np.unique(key, return_inverse=True, return_counts=True)
    fpatch = np.repeat(patch, 3)
    pmin = np.full(len(ukey), np.iinfo(np.int64).max)
    pmax = np.full(len(ukey), -1)
    np.minimum.at(pmin, inv, fpatch)
    np.maximum.at(pmax, inv, fpatch)
    feat = ukey[(valence != 2) | (pmin != pmax)]
    return np.bincount(feat // nv, minlength=nv) + np.bincount(feat % nv, minlength=nv)


def _patches(labels):
    """Label pair of each face, independent of the orientation."""
    s = np.sort(labels, axis=1).astype(np.int64)
    # a 1d key of the pair is much faster to unique than the rows
    _, patch = np.unique(s[:, 0] * (s[:, 1].max() + 1) + s[:, 1], return_inverse=True)
    return patch.ravel().astype(np.int64)


def _plane_masks(V, tol):
    """Bit mask of the bounding box planes each vertex lies on."""
    lo, hi = V.min(axis=0), V.max(axis=0)
    mask = np.zeros(len(V), np.uint8)
    for k in range(3):
        mask |= (np.abs(V[:, k] - lo[k]) <= tol).astype(np.uint8) << (2 * k)
        mask |= (np.abs(V[:, k] - hi[k]) <= tol).astype(np.uint8) << (2 * k + 1)
    return mask


@nb.njit(parallel=True, cache=True)
def _vertex_normals(V, F, sgn, vf_ptr, vf_idx):
    """Unit vertex normals of the faces oriented by sgn."""
    nv = len(vf_ptr) - 1
    N = np.zeros((nv, 3))
    for x in nb.prange(nv):
        for jj in range(vf_ptr[x], vf_ptr[x + 1]):
            f = vf_idx[jj]
            nx, ny, nz = _tri_normal(V, F[f, 0], F[f, 1], F[f, 2])
            N[x, 0] += sgn[f] * nx
            N[x, 1] += sgn[f] * ny
            N[x, 2] += sgn[f] * nz
        n = np.sqrt(N[x, 0] ** 2 + N[x, 1] ** 2 + N[x, 2] ** 2)
        if n > 0:
            N[x] /= n
    return N


@nb.njit(parallel=True, cache=True)
def _face_geometry(V, F, sgn):
    """
    Oriented unit normal, centroid and bounding sphere radius (around the
    centroid) of each face as rows of gin, and twice the face areas.
    """
    gin = np.empty((len(F), 7))
    area2 = np.empty(len(F))
    for f in nb.prange(len(F)):
        a, b, c = F[f, 0], F[f, 1], F[f, 2]
        nx, ny, nz = _tri_normal(V, a, b, c)
        n = np.sqrt(nx * nx + ny * ny + nz * nz)
        area2[f] = n
        n = max(n, 1e-300)
        gin[f, 0], gin[f, 1], gin[f, 2] = sgn[f] * nx / n, sgn[f] * ny / n, sgn[f] * nz / n
        for d in range(3):
            gin[f, 3 + d] = (V[a, d] + V[b, d] + V[c, d]) / 3.0
        r = 0.0
        for x in (a, b, c):
            dx, dy, dz = V[x, 0] - gin[f, 3], V[x, 1] - gin[f, 4], V[x, 2] - gin[f, 5]
            r = max(r, np.sqrt(dx * dx + dy * dy + dz * dz))
        gin[f, 6] = r
    return gin, area2


def _sample_points(V, F, patch, sgn, cls, vf_ptr, vf_idx, gin, centroids):
    """
    Points of the input that the output must approximate, with the label pair
    (patch) and unit normal (oriented by the labels) they must be
    approximated with. Points on feature
    curves have patch -1 and are only checked for their distance. pface is the
    input face of a centroid (-1 for vertices).
    """
    first_face = vf_idx[vf_ptr[:-1].clip(max=len(vf_idx) - 1)]
    used = np.diff(vf_ptr) > 0
    vid = np.flatnonzero(used)
    P = [V[vid]]
    owner = [first_face[vid]]
    ppatch = [np.where(cls[vid] == INTERIOR, patch[first_face[vid]], -1)]
    pn = [_vertex_normals(V, F, sgn, vf_ptr, vf_idx)[vid]]
    pface = [np.full(len(vid), -1)]
    if centroids:
        P.append(gin[:, 3:6])
        owner.append(np.arange(len(F)))
        ppatch.append(patch)
        pn.append(gin[:, :3])
        pface.append(np.arange(len(F)))
    return (
        np.ascontiguousarray(np.vstack(P)),
        np.concatenate(owner).astype(np.int64),
        np.concatenate(ppatch).astype(np.int64),
        np.ascontiguousarray(np.vstack(pn)),
        np.concatenate(pface).astype(np.int64),
    )


# --------------------------------------------------------------------------
# collapse check
# --------------------------------------------------------------------------


@nb.njit(cache=True)
def _check_collapse(
    u,
    v,
    p,
    V,
    F,
    patch,
    sgn,
    vf_ptr,
    vf_idx,
    pf_ptr,
    pf_idx,
    P,
    ppatch,
    pnormal,
    pface,
    Vin,
    Fin,
    patch_in,
    gin,
    cls,
    planes,
    eps2,
    cos_max,
    q_min,
    lmax2,
    cos_wedge,
    owner,
    write,
):
    """
    True if the half-edge collapse u -> v, with v moved to p, is valid. With
    write=True, the points owned by the modified faces are reassigned to the
    new faces around v.
    """
    if planes[u] & ~planes[v]:
        return False
    eps = np.sqrt(eps2)
    ru = vf_idx[vf_ptr[u] : vf_ptr[u + 1]]
    rv = vf_idx[vf_ptr[v] : vf_ptr[v + 1]]
    nu = len(ru)
    moved = p[0] != V[v, 0] or p[1] != V[v, 1] or p[2] != V[v, 2]

    # faces on edge uv, their opposite vertices and whether uv is a feature
    shared = np.zeros(nu, np.bool_)
    opp = np.empty(nu, np.int64)
    nsh = 0
    p0 = -1
    feature = False
    for i in range(nu):
        f = ru[i]
        a, b, c = F[f, 0], F[f, 1], F[f, 2]
        if a == v or b == v or c == v:
            shared[i] = True
            opp[nsh] = a + b + c - u - v
            nsh += 1
            if p0 == -1:
                p0 = patch[f]
            elif patch[f] != p0:
                feature = True
    if nsh != 2:
        feature = True
    if (cls[u] == CURVE) != feature:
        return False

    # link condition: common neighbours of u and v are the opposite vertices
    for i in range(nu):
        f = ru[i]
        for k in range(3):
            x = F[f, k]
            if x == u or x == v:
                continue
            is_opp = False
            for j in range(nsh):
                if opp[j] == x:
                    is_opp = True
                    break
            if is_opp:
                continue
            for g in rv:
                if F[g, 0] == x or F[g, 1] == x or F[g, 2] == x:
                    return False

    # the new faces around v: unchanged faces of v, then modified faces of u
    m = (len(rv) - nsh) + (nu - nsh)
    if m == 0:
        return False
    nf_id = np.empty(m, np.int64)
    nf_tri = np.empty((m, 3), np.int64)
    k = 0
    for g in rv:
        if F[g, 0] == u or F[g, 1] == u or F[g, 2] == u:
            continue
        nf_id[k] = g
        nf_tri[k] = F[g]
        k += 1
    n_unchanged = k
    new_k = np.full(nu, -1, np.int64)  # new face of each face around u
    for i in range(nu):
        if shared[i]:
            continue
        f = ru[i]
        nf_id[k] = f
        new_k[i] = k
        for j in range(3):
            nf_tri[k, j] = v if F[f, j] == u else F[f, j]
        k += 1
    # the faces whose geometry changes: with a moved v also the unchanged ones
    k0 = 0 if moved else n_unchanged

    # no duplicate faces
    for k in range(n_unchanged, m):
        for k2 in range(k):
            same = 0
            for j in range(3):
                x = nf_tri[k, j]
                if x == nf_tri[k2, 0] or x == nf_tri[k2, 1] or x == nf_tri[k2, 2]:
                    same += 1
            if same == 3:
                return False

    # local positions of the vertices of the new faces, with v at p
    loc = np.empty(3 * m, np.int64)
    nf_loc = np.empty((m, 3), np.int64)
    nl = 0
    vl = -1
    for k in range(m):
        for j in range(3):
            x = nf_tri[k, j]
            idx = -1
            for i in range(nl):
                if loc[i] == x:
                    idx = i
                    break
            if idx < 0:
                idx = nl
                loc[nl] = x
                nl += 1
                if x == v:
                    vl = idx
            nf_loc[k, j] = idx
    Vl = np.empty((nl, 3))
    for i in range(nl):
        Vl[i] = V[loc[i]]
    Vl[vl] = p

    # geometry of the new faces, with bounding spheres (centroid, radius)
    nrm = np.empty((m, 3))
    sph = np.empty((m, 4))
    for k in range(m):
        a, b, c = nf_loc[k, 0], nf_loc[k, 1], nf_loc[k, 2]
        r2 = 0.0
        for d in range(3):
            sph[k, d] = (Vl[a, d] + Vl[b, d] + Vl[c, d]) / 3.0
        for x in (a, b, c):
            dx, dy, dz = Vl[x, 0] - sph[k, 0], Vl[x, 1] - sph[k, 1], Vl[x, 2] - sph[k, 2]
            r2 = max(r2, dx * dx + dy * dy + dz * dz)
        sph[k, 3] = np.sqrt(r2)
        nx, ny, nz = _tri_normal(Vl, a, b, c)
        nn = np.sqrt(nx * nx + ny * ny + nz * nz)
        if nn == 0.0:
            return False
        nrm[k, 0], nrm[k, 1], nrm[k, 2] = nx / nn, ny / nn, nz / nn
        if k < k0:
            continue
        f = nf_id[k]
        ox, oy, oz = _tri_normal(V, F[f, 0], F[f, 1], F[f, 2])
        on = np.sqrt(ox * ox + oy * oy + oz * oz)
        if on == 0.0 or nx * ox + ny * oy + nz * oz < cos_max * nn * on:
            return False
        q = _quality(Vl, a, b, c)
        if q < q_min and q < _quality(V, F[f, 0], F[f, 1], F[f, 2]):
            return False
        for j in range(3):
            x = nf_tri[k, j]
            if x == v:
                continue
            l2 = _sqdist(Vl, vl, nf_loc[k, j])
            # edges of the unchanged faces may keep their length
            if l2 > lmax2 and (k >= n_unchanged or l2 > _sqdist(V, v, x)):
                return False

    # no sharp wedges at the new edges (v, x)
    nbuf = nu + len(rv)
    tris = np.empty((nbuf, 3), np.int64)
    buf = np.empty((nbuf, 3))
    for k in range(k0, m):
        for j in range(3):
            x = nf_tri[k, j]
            if x == v:
                continue
            first = True
            for k2 in range(k0, k):
                if x == nf_tri[k2, 0] or x == nf_tri[k2, 1] or x == nf_tri[k2, 2]:
                    first = False
                    break
            if not first:
                continue
            n = 0
            for k2 in range(m):
                if x == nf_tri[k2, 0] or x == nf_tri[k2, 1] or x == nf_tri[k2, 2]:
                    tris[n] = nf_loc[k2]
                    n += 1
            c_new = _max_opening_cos(Vl, vl, nf_loc[k, j], tris, n, buf)
            if c_new <= cos_wedge:
                continue
            # the edges (u, x) and (v, x) before the collapse
            n = 0
            for f in ru:
                if x == F[f, 0] or x == F[f, 1] or x == F[f, 2]:
                    tris[n] = F[f]
                    n += 1
            c_old = _max_opening_cos(V, u, x, tris, n, buf)
            n = 0
            for g in rv:
                if x == F[g, 0] or x == F[g, 1] or x == F[g, 2]:
                    tris[n] = F[g]
                    n += 1
            c_old = max(c_old, _max_opening_cos(V, v, x, tris, n, buf))
            if c_new > c_old:
                return False

    # with a moved v, also the edges (x, y) opposite v change their wedges
    if moved:
        for k in range(m):
            j = 0
            while nf_tri[k, j] != v:
                j += 1
            x, y = nf_tri[k, (j + 1) % 3], nf_tri[k, (j + 2) % 3]
            n = 0
            for jj in range(vf_ptr[x], vf_ptr[x + 1]):
                g = vf_idx[jj]
                if F[g, 0] == y or F[g, 1] == y or F[g, 2] == y:
                    n += 1
            if n > len(buf):
                buf = np.empty((n, 3))
            for old in range(2):  # after, then before the collapse
                n = 0
                for jj in range(vf_ptr[x], vf_ptr[x + 1]):
                    g = vf_idx[jj]
                    if F[g, 0] == y or F[g, 1] == y or F[g, 2] == y:
                        w = F[g, 0] + F[g, 1] + F[g, 2] - x - y
                        if old == 0 and (w == u or w == v):
                            buf[n, 0], buf[n, 1], buf[n, 2] = p[0], p[1], p[2]
                        else:
                            buf[n, 0], buf[n, 1], buf[n, 2] = V[w, 0], V[w, 1], V[w, 2]
                        n += 1
                c = _opening_cos(V[x], V[y], buf, n)
                if old == 0:
                    if c <= cos_wedge:
                        break
                    c_new = c
                elif c_new > c:
                    return False

    # the modified faces, whose owned points must be reassigned, with the new
    # face approximating them before (-1 for the collapsed faces)
    src_f = np.empty(nu + m, np.int64)
    src_k = np.empty(nu + m, np.int64)
    ns = 0
    for i in range(nu):
        src_f[ns], src_k[ns] = ru[i], new_k[i]
        ns += 1
    if moved:
        for k in range(n_unchanged):
            src_f[ns], src_k[ns] = nf_id[k], k
            ns += 1

    # deviation input -> output: every point owned by a modified face must be
    # within eps of a new face around v of the same patch with a similar normal
    npts = 0
    for i in range(ns):
        npts += pf_ptr[src_f[i] + 1] - pf_ptr[src_f[i]]
    pts = np.empty(npts, np.int64)
    pts_k = np.empty(npts, np.int64)  # new face approximating the point
    near = np.empty(npts, np.int64)  # input faces of the owned centroids
    near_k = np.empty(npts, np.int64)  # and the new faces approximating them
    npts = 0
    nnear = 0
    for i in range(ns):
        f = src_f[i]
        hint = src_k[i]
        for jj in range(pf_ptr[f], pf_ptr[f + 1]):
            j = pf_idx[jj]
            pp = ppatch[j]
            best = eps2
            best_d = eps
            best_k = -1
            # start with the new version of the current owner
            for kk in range(-1, m):
                k = hint if kk < 0 else kk
                if k < 0 or (kk >= 0 and k == hint):
                    continue
                if pp >= 0:
                    if patch[nf_id[k]] != pp:
                        continue
                    d = sgn[nf_id[k]] * (
                        nrm[k, 0] * pnormal[j, 0]
                        + nrm[k, 1] * pnormal[j, 1]
                        + nrm[k, 2] * pnormal[j, 2]
                    )
                    if d < cos_max:
                        continue
                # the face is at least |P - centroid| - radius away
                dx, dy, dz = P[j, 0] - sph[k, 0], P[j, 1] - sph[k, 1], P[j, 2] - sph[k, 2]
                rr = (sph[k, 3] + best_d) * (1.0 + 1e-12)
                if dx * dx + dy * dy + dz * dz > rr * rr:
                    continue
                d2 = _point_tri_sqdist(P[j], Vl, nf_loc[k, 0], nf_loc[k, 1], nf_loc[k, 2])
                if d2 <= best:
                    best = d2
                    best_d = np.sqrt(d2)
                    best_k = k
                    if not write:  # any face within eps will do for the check
                        break
            if best_k < 0:
                return False
            pts[npts] = j
            pts_k[npts] = best_k
            npts += 1
            if pface[j] >= 0:
                near[nnear] = pface[j]
                near_k[nnear] = best_k
                nnear += 1

    # deviation output -> input: sample points of the modified faces must be
    # within eps of the input faces they replace with the same patch and the
    # same side of the labels (oriented normals less than 90 degrees apart),
    # so no face flips against the input. Without centroid samples
    # (nnear == 0) this is skipped.
    if nnear > 0:
        q = np.empty(3)
        for k in range(k0, m):
            pk = patch[nf_id[k]]
            sk = sgn[nf_id[k]]
            for s in range(7):
                if 3 <= s < 6 and v != nf_tri[k, s - 3] and v != nf_tri[k, (s - 2) % 3]:
                    continue  # the edge opposite v is unchanged
                for d in range(3):
                    x = Vl[nf_loc[k, 0], d], Vl[nf_loc[k, 1], d], Vl[nf_loc[k, 2], d]
                    cen = (x[0] + x[1] + x[2]) / 3.0
                    if s < 3:  # halfway between the centroid and a vertex
                        q[d] = 0.5 * (cen + x[s])
                    elif s < 6:  # edge midpoint
                        q[d] = 0.5 * (x[s - 3] + x[(s - 2) % 3])
                    else:
                        q[d] = cen
                # first the input faces approximated by this face, then all
                found = False
                for sweep in range(2):
                    for i in range(nnear):
                        g = near[i]
                        if (near_k[i] == k) != (sweep == 0) or patch_in[g] != pk:
                            continue
                        dx, dy, dz = q[0] - gin[g, 3], q[1] - gin[g, 4], q[2] - gin[g, 5]
                        rr = (gin[g, 6] + eps) * (1.0 + 1e-12)
                        if dx * dx + dy * dy + dz * dz > rr * rr:
                            continue
                        if (
                            sk
                            * (
                                nrm[k, 0] * gin[g, 0]
                                + nrm[k, 1] * gin[g, 1]
                                + nrm[k, 2] * gin[g, 2]
                            )
                            <= 0.0
                        ):
                            continue
                        if _point_tri_sqdist(q, Vin, Fin[g, 0], Fin[g, 1], Fin[g, 2]) <= eps2:
                            found = True
                            break
                    if found:
                        break
                if not found:
                    return False

    if write:
        for i in range(npts):
            owner[pts[i]] = nf_id[pts_k[i]]
    return True


@nb.njit(cache=True)
def _placement(Q, V, u, v, cls, planes):
    """
    Position of v after the collapse u -> v minimizing the quadric error of u
    and v, restricted to the feature curve and the bounding box planes of v.
    A small Tikhonov term towards the edge midpoint keeps directions in which
    the quadric is (nearly) flat at the midpoint.
    """
    vx, vy, vz = V[v, 0], V[v, 1], V[v, 2]
    if cls[v] == CORNER or (cls[v] == CURVE and cls[u] != CURVE):
        return vx, vy, vz
    dx, dy, dz = V[u, 0] - vx, V[u, 1] - vy, V[u, 2] - vz
    L = np.sqrt(dx * dx + dy * dy + dz * dz)
    if L == 0.0:
        return vx, vy, vz
    # minimize p^T A p - 2 b^T p around the midpoint c: p = c + t
    a00 = Q[u, 0, 0] + Q[v, 0, 0]
    a01 = Q[u, 0, 1] + Q[v, 0, 1]
    a02 = Q[u, 0, 2] + Q[v, 0, 2]
    a11 = Q[u, 1, 1] + Q[v, 1, 1]
    a12 = Q[u, 1, 2] + Q[v, 1, 2]
    a22 = Q[u, 2, 2] + Q[v, 2, 2]
    cx, cy, cz = vx + 0.5 * dx, vy + 0.5 * dy, vz + 0.5 * dz
    if cls[v] == CURVE:  # along the collapsed curve edge
        if planes[u] != planes[v]:
            return vx, vy, vz
    else:
        # coordinates fixed by the bounding box planes of v
        fx = (planes[v] & 3) != 0
        fy = (planes[v] & 12) != 0
        fz = (planes[v] & 48) != 0
        if fx and fy and fz:
            return vx, vy, vz
        if fx:
            cx = vx
        if fy:
            cy = vy
        if fz:
            cz = vz
    # gradient -(b - A c)
    gx = -(Q[u, 0, 3] + Q[v, 0, 3]) - (a00 * cx + a01 * cy + a02 * cz)
    gy = -(Q[u, 1, 3] + Q[v, 1, 3]) - (a01 * cx + a11 * cy + a12 * cz)
    gz = -(Q[u, 2, 3] + Q[v, 2, 3]) - (a02 * cx + a12 * cy + a22 * cz)
    if cls[v] == CURVE:
        ex, ey, ez = dx / L, dy / L, dz / L
        h = (
            ex * (a00 * ex + a01 * ey + a02 * ez)
            + ey * (a01 * ex + a11 * ey + a12 * ez)
            + ez * (a02 * ex + a12 * ey + a22 * ez)
        )
        if h <= 0.0:
            return cx, cy, cz
        s = (ex * gx + ey * gy + ez * gz) / (h * (1.0 + 1e-2))
        s = min(max(s, -0.5 * L), 0.5 * L)  # stay on the curve edge
        return cx + s * ex, cy + s * ey, cz + s * ez
    # fixed coordinates: identity rows, zero right hand side
    if fx:
        a00, a01, a02, gx = 1.0, 0.0, 0.0, 0.0
    if fy:
        a01, a11, a12, gy = 0.0, 1.0, 0.0, 0.0
    if fz:
        a02, a12, a22, gz = 0.0, 0.0, 1.0, 0.0
    tr = (0.0 if fx else a00) + (0.0 if fy else a11) + (0.0 if fz else a22)
    if tr <= 0.0:
        return cx, cy, cz
    lam = 1e-2 * tr
    if not fx:
        a00 += lam
    if not fy:
        a11 += lam
    if not fz:
        a22 += lam
    # Cramer's rule
    c00 = a11 * a22 - a12 * a12
    c01 = a02 * a12 - a01 * a22
    c02 = a01 * a12 - a02 * a11
    det = a00 * c00 + a01 * c01 + a02 * c02
    if det <= 0.0:
        return cx, cy, cz
    c11 = a00 * a22 - a02 * a02
    c12 = a01 * a02 - a00 * a12
    c22 = a00 * a11 - a01 * a01
    tx = (c00 * gx + c01 * gy + c02 * gz) / det
    ty = (c01 * gx + c11 * gy + c12 * gz) / det
    tz = (c02 * gx + c12 * gy + c22 * gz) / det
    tn = np.sqrt(tx * tx + ty * ty + tz * tz)
    if tn > L:  # at most one edge length from the midpoint
        tx, ty, tz = tx * L / tn, ty * L / tn, tz * L / tn
    return cx + tx, cy + ty, cz + tz


@nb.njit(cache=True)
def _collapse_cost(Q, V, u, v, px, py, pz, length_weight):
    h = (px, py, pz, 1.0)
    c = 0.0
    for i in range(4):
        for j in range(4):
            c += (Q[u, i, j] + Q[v, i, j]) * h[i] * h[j]
    return max(c, 0.0) + length_weight * _sqdist(V, u, v)


@nb.njit(parallel=True, cache=True)
def _propose(
    V,
    F,
    patch,
    sgn,
    vf_ptr,
    vf_idx,
    pf_ptr,
    pf_idx,
    P,
    ppatch,
    pnormal,
    pface,
    Vin,
    Fin,
    patch_in,
    gin,
    cls,
    planes,
    Q,
    dirty,
    qem,
    eps2,
    cos_max,
    q_min,
    lmax2,
    cos_wedge,
    length_weight,
    owner,
    target,
    tpos,
    cost,
):
    """
    Update the best valid collapse target (or -1), the new position of the
    target and the cost of the dirty vertices in place.
    """
    todo = np.flatnonzero(dirty & (cls != CORNER))
    for t in nb.prange(len(todo)):
        u = todo[t]
        target[u] = -1
        cost[u] = np.inf
        s, e = vf_ptr[u], vf_ptr[u + 1]
        if s == e:
            continue
        nbrs = np.empty(3 * (e - s), np.int64)
        n = 0
        for i in range(s, e):
            f = vf_idx[i]
            for k in range(3):
                x = F[f, k]
                if x == u:
                    continue
                new = True
                for j in range(n):
                    if nbrs[j] == x:
                        new = False
                        break
                if new:
                    nbrs[n] = x
                    n += 1
        # candidates: each neighbour at its position and, with qem, at the
        # optimal position
        cv = np.empty(2 * n, np.int64)
        cp = np.empty((2 * n, 3))
        c = np.empty(2 * n)
        nc = 0
        for j in range(n):
            x = nbrs[j]
            cv[nc] = x
            cp[nc] = V[x]
            c[nc] = _collapse_cost(Q, V, u, x, V[x, 0], V[x, 1], V[x, 2], length_weight)
            nc += 1
            if qem:
                px, py, pz = _placement(Q, V, u, x, cls, planes)
                if px != V[x, 0] or py != V[x, 1] or pz != V[x, 2]:
                    cv[nc] = x
                    cp[nc, 0], cp[nc, 1], cp[nc, 2] = px, py, pz
                    c[nc] = _collapse_cost(Q, V, u, x, px, py, pz, length_weight)
                    nc += 1
        c = c[:nc]
        for j in np.argsort(c):
            if _check_collapse(
                u,
                cv[j],
                cp[j],
                V,
                F,
                patch,
                sgn,
                vf_ptr,
                vf_idx,
                pf_ptr,
                pf_idx,
                P,
                ppatch,
                pnormal,
                pface,
                Vin,
                Fin,
                patch_in,
                gin,
                cls,
                planes,
                eps2,
                cos_max,
                q_min,
                lmax2,
                cos_wedge,
                owner,
                False,
            ):
                target[u] = cv[j]
                tpos[u] = cp[j]
                cost[u] = c[j]
                break
    return target, cost


@nb.njit(cache=True)
def _select(target, cost, F, vf_ptr, vf_idx):
    """
    Greedy maximal set of proposals u -> target[u] with disjoint face stars
    (faces around u and v), in order of increasing cost.
    """
    cu = np.flatnonzero(target >= 0)
    # the stars of u' -> v' and of an accepted u -> v share a face iff u' or
    # v' is a vertex of a face around u or v, so only these are marked
    covered = np.zeros(len(vf_ptr) - 1, np.bool_)
    win = np.zeros(len(cu), np.bool_)
    for i in np.argsort(cost[cu]):
        u, v = cu[i], target[cu[i]]
        if covered[u] or covered[v]:
            continue
        win[i] = True
        for x in (u, v):
            for jj in range(vf_ptr[x], vf_ptr[x + 1]):
                g = vf_idx[jj]
                covered[F[g, 0]] = covered[F[g, 1]] = covered[F[g, 2]] = True
    return cu[win]


@nb.njit(parallel=True, cache=True)
def _apply(
    wu,
    wv,
    wp,
    V,
    F,
    patch,
    sgn,
    vf_ptr,
    vf_idx,
    pf_ptr,
    pf_idx,
    P,
    ppatch,
    pnormal,
    pface,
    Vin,
    Fin,
    patch_in,
    gin,
    cls,
    planes,
    Q,
    eps2,
    cos_max,
    q_min,
    lmax2,
    cos_wedge,
    owner,
):
    ok = np.empty(len(wu), np.bool_)
    for i in nb.prange(len(wu)):
        u, v = wu[i], wv[i]
        ok[i] = _check_collapse(
            u,
            v,
            wp[i],
            V,
            F,
            patch,
            sgn,
            vf_ptr,
            vf_idx,
            pf_ptr,
            pf_idx,
            P,
            ppatch,
            pnormal,
            pface,
            Vin,
            Fin,
            patch_in,
            gin,
            cls,
            planes,
            eps2,
            cos_max,
            q_min,
            lmax2,
            cos_wedge,
            owner,
            True,
        )
        Q[v] += Q[u]
    return ok


@nb.njit(parallel=True, cache=True)
def _dilate(mask, F, vf_ptr, vf_idx):
    """The vertices of the faces with a vertex in mask."""
    out = np.zeros(len(mask), np.bool_)
    for x in nb.prange(len(mask)):
        for jj in range(vf_ptr[x], vf_ptr[x + 1]):
            g = vf_idx[jj]
            if mask[F[g, 0]] or mask[F[g, 1]] or mask[F[g, 2]]:
                out[x] = True
                break
    return out


@nb.njit(cache=True)
def _cumsum(a):
    for i in range(len(a) - 1):
        a[i + 1] += a[i]


@nb.njit(parallel=True, cache=True)
def _collapse(wu, wv, F, patch, sgn, fid, owner, vf_ptr, vf_idx, pf_ptr, pf_idx, moved):
    """
    Substitute u -> v for the accepted collapses, drop the collapsed faces and
    renumber the faces (and the owners of the points, in place). The vertex ->
    face and face -> point CSR are updated instead of rebuilt: only the faces
    in the stars of u and v change. Also returns the vertices whose proposals
    must be recomputed.
    """
    nv, nf = len(vf_ptr) - 1, len(F)
    # role 1 for u, 2 for v, and the other vertex of the collapse
    role = np.zeros(nv, np.int8)
    partner = np.empty(nv, np.int64)
    # collapse of the faces in the stars of u and v (disjoint), else -1
    coll = np.empty(nf, np.int64)
    for f in nb.prange(nf):
        coll[f] = -1
    for i in nb.prange(len(wu)):
        u, v = wu[i], wv[i]
        role[u], role[v] = 1, 2
        partner[u], partner[v] = v, u
        for x in (u, v):
            for jj in range(vf_ptr[x], vf_ptr[x + 1]):
                coll[vf_idx[jj]] = i

    # the faces that are kept (not degenerate after the substitution)
    keep = np.empty(nf, np.bool_)
    for f in nb.prange(nf):
        a, b, c = F[f, 0], F[f, 1], F[f, 2]
        if coll[f] >= 0:
            a = partner[a] if role[a] == 1 else a
            b = partner[b] if role[b] == 1 else b
            c = partner[c] if role[c] == 1 else c
        keep[f] = a != b and b != c and c != a
    new_id = np.empty(nf, np.int64)
    n = 0
    for f in range(nf):
        new_id[f] = n if keep[f] else -1
        n += keep[f]
    F2 = np.empty((n, 3), np.int64)
    patch2 = np.empty(n, np.int64)
    sgn2 = np.empty(n)
    fid2 = np.empty(n, np.int64)
    for f in nb.prange(nf):
        g = new_id[f]
        if g < 0:
            continue
        for k in range(3):
            x = F[f, k]
            F2[g, k] = partner[x] if role[x] == 1 else x
        patch2[g], sgn2[g], fid2[g] = patch[f], sgn[f], fid[f]
    for j in nb.prange(len(owner)):
        owner[j] = new_id[owner[j]]

    # faces around x: the kept ones before, merged with those of u for x = v
    vf_ptr2 = np.empty(nv + 1, np.int64)
    vf_ptr2[0] = 0
    for x in nb.prange(nv):
        c = 0
        if role[x] != 1:
            for jj in range(vf_ptr[x], vf_ptr[x + 1]):
                c += keep[vf_idx[jj]]
            if role[x] == 2:
                y = partner[x]
                for jj in range(vf_ptr[y], vf_ptr[y + 1]):
                    c += keep[vf_idx[jj]]
        vf_ptr2[x + 1] = c
    _cumsum(vf_ptr2)
    vf_idx2 = np.empty(vf_ptr2[nv], np.int64)
    for x in nb.prange(nv):
        if role[x] == 1:
            continue
        i, ie = vf_ptr[x], vf_ptr[x + 1]
        j = je = 0
        if role[x] == 2:
            j, je = vf_ptr[partner[x]], vf_ptr[partner[x] + 1]
        pos = vf_ptr2[x]
        while i < ie or j < je:  # both are sorted
            if j >= je or (i < ie and vf_idx[i] <= vf_idx[j]):
                f = vf_idx[i]
                i += 1
            else:
                f = vf_idx[j]
                j += 1
            if keep[f]:
                vf_idx2[pos] = new_id[f]
                pos += 1

    # points per face: unchanged outside the stars, and the points of the
    # stars of u and v are owned by the new star of v (the order of the points
    # of a face does not matter)
    pf_ptr2 = np.zeros(n + 1, np.int64)
    for f in nb.prange(nf):
        if keep[f] and coll[f] < 0:
            pf_ptr2[new_id[f] + 1] = pf_ptr[f + 1] - pf_ptr[f]
    for i in nb.prange(len(wu)):
        u, v = wu[i], wv[i]
        for x in (u, v):
            for jj in range(vf_ptr[x], vf_ptr[x + 1]):
                f = vf_idx[jj]
                if (jj > vf_ptr[x] and vf_idx[jj - 1] == f) or (
                    x == v and (F[f, 0] == u or F[f, 1] == u or F[f, 2] == u)
                ):
                    continue  # counted before
                for q in range(pf_ptr[f], pf_ptr[f + 1]):
                    pf_ptr2[owner[pf_idx[q]] + 1] += 1
    _cumsum(pf_ptr2)
    pf_idx2 = np.empty(pf_ptr2[n], np.int64)
    pos = pf_ptr2[:-1].copy()
    for f in nb.prange(nf):
        if keep[f] and coll[f] < 0:
            g = new_id[f]
            for q in range(pf_ptr[f], pf_ptr[f + 1]):
                pf_idx2[pos[g] + q - pf_ptr[f]] = pf_idx[q]
    for i in nb.prange(len(wu)):
        u, v = wu[i], wv[i]
        for x in (u, v):
            for jj in range(vf_ptr[x], vf_ptr[x + 1]):
                f = vf_idx[jj]
                if (jj > vf_ptr[x] and vf_idx[jj - 1] == f) or (
                    x == v and (F[f, 0] == u or F[f, 1] == u or F[f, 2] == u)
                ):
                    continue
                for q in range(pf_ptr[f], pf_ptr[f + 1]):
                    g = owner[pf_idx[q]]
                    pf_idx2[pos[g]] = pf_idx[q]
                    pos[g] += 1

    # reactivate the vertices whose neighbourhood changed: all faces around v
    # changed (their owned points, and with a moved v their geometry)
    touched = _dilate(role == 2, F2, vf_ptr2, vf_idx2)
    if moved:
        # the checks of moved vertices also look at the faces next to the
        # modified ones
        touched = _dilate(touched, F2, vf_ptr2, vf_idx2)
    dirty = _dilate(touched, F2, vf_ptr2, vf_idx2)
    return F2, patch2, sgn2, fid2, vf_ptr2, vf_idx2, pf_ptr2, pf_idx2, dirty


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def _lap(timing, key, t):
    now = time.perf_counter()
    timing[key] += now - t
    return now


def simplify_surface(
    surf,
    epsilon,
    max_angle=60.0,
    min_quality=0.1,
    max_edge_length=np.inf,
    min_wedge_angle=15.0,
    length_weight=0.01,
    fixed=None,
    sample_centroids=True,
    label_name="boundary_labels",
    placement="endpoint",
    max_rounds=1000,
    verbose=False,
):
    """
    Simplify a triangulated multi-label surface by edge collapses.

    surf: triangulated pv.PolyData with cell data label_name (n_cells, 2).
    epsilon: maximal distance of the input points (vertices and, with
        sample_centroids, face centroids) to the output faces of the same
        label pair, and (with sample_centroids) of sample points of the output
        faces to the input faces.
    max_angle: maximal rotation (degrees) of a face normal per collapse, and
        maximal angle between an output face normal and the input normal of
        the points it approximates.
    min_quality: collapses may not create triangles with
        4 sqrt(3) area / sum(edge_length^2) below this (unless improving).
    max_edge_length: maximal length of edges created by collapses.
    min_wedge_angle: collapses may not create edges where two faces meet at an
        angle below this (degrees; 0 = folded onto each other), unless the
        edge was already that sharp. Sharp wedges keep fTetWild's mesh
        optimization from converging.
    length_weight: weight of the squared edge length in the collapse priority,
        relative to the quadric error (in units of the mean input face area).
    fixed: optional (n_points,) bool mask of vertices that must be kept.
    placement: "endpoint" collapses edges into one of their vertices
        (half-edge collapses, the output points are a subset of the input
        points), "qem" additionally moves the kept vertex to the position
        minimizing the quadric error (restricted to feature curves and
        bounding box planes), for a closer fit at the same number of faces.

    Returns the simplified pv.PolyData with the labels as cell data.
    """
    if placement not in ("endpoint", "qem"):
        raise ValueError(f"unknown placement {placement!r}")
    qem = placement == "qem"
    t0 = time.perf_counter()
    V = np.array(surf.points, dtype=np.float64)  # a copy, points may move
    F = np.ascontiguousarray(surf.regular_faces, dtype=np.int64)
    labels = np.asarray(surf.cell_data[label_name])
    nv = len(V)

    # label pair of each face, independent of the orientation
    patch = _patches(labels)

    vf_ptr, vf_idx = _vertex_faces(F, nv)
    cls = _classify(F, vf_ptr, vf_idx, _feature_degree(F, patch, nv))
    if fixed is not None:
        cls[np.asarray(fixed, dtype=bool)] = CORNER
    diag = np.linalg.norm(np.ptp(V, axis=0))
    planes = _plane_masks(V, 1e-9 * diag)
    Q = _quadrics(V, F, vf_ptr, vf_idx)
    # orientation of each face relative to its label pair
    sgn = np.where(labels[:, 0] > labels[:, 1], -1.0, 1.0)
    # input faces: oriented unit normal, centroid, bounding sphere radius
    gin, area2 = _face_geometry(V, F, sgn)
    P, owner, ppatch, pnormal, pface = _sample_points(
        V, F, patch, sgn, cls, vf_ptr, vf_idx, gin, sample_centroids
    )
    Vin, Fin, patch_in = (V.copy() if qem else V), F.copy(), patch.copy()
    mean_area = 0.5 * area2.mean()
    params = (
        float(epsilon) ** 2,
        float(np.cos(np.deg2rad(max_angle))),
        float(min_quality),
        float(max_edge_length) ** 2,
        float(np.cos(np.deg2rad(min_wedge_angle))),
    )
    lw = length_weight * mean_area
    # proposals only change if the neighbourhood changed, so they are kept
    # and only recomputed for dirty vertices
    dirty = np.ones(nv, dtype=bool)
    target = np.full(nv, -1, np.int64)
    tpos = V.copy()
    cost = np.full(nv, np.inf)
    if verbose:
        n_cls = np.bincount(cls, minlength=3)
        print(
            f"setup {time.perf_counter() - t0:.2f}s: {len(F)} faces, "
            f"{n_cls[INTERIOR]} interior / {n_cls[CURVE]} curve / {n_cls[CORNER]} corner vertices"
        )

    timing = dict(propose=0.0, select=0.0, apply=0.0, update=0.0)
    # input face of each face
    fid = np.arange(len(F))
    pf_ptr, pf_idx = _csr(owner, len(F))
    for it in range(max_rounds):
        tr = time.perf_counter()
        common = (
            V,
            F,
            patch,
            sgn,
            vf_ptr,
            vf_idx,
            pf_ptr,
            pf_idx,
            P,
            ppatch,
            pnormal,
            pface,
            Vin,
            Fin,
            patch_in,
            gin,
            cls,
            planes,
        )
        # dynamic scheduling, the work per vertex varies a lot
        with nb.parallel_chunksize(16):
            _propose(*common, Q, dirty, qem, *params, lw, owner, target, tpos, cost)
        tr = _lap(timing, "propose", tr)
        wu = _select(target, cost, F, vf_ptr, vf_idx)
        tr = _lap(timing, "select", tr)
        if len(wu) == 0:
            break
        wv = target[wu]
        wp = tpos[wu]
        with nb.parallel_chunksize(16):
            ok = _apply(wu, wv, wp, *common, Q, *params, owner)
        assert ok.all()
        moved = (wp != V[wv]).any()
        V[wv] = wp
        tr = _lap(timing, "apply", tr)

        # substitute u -> v, drop the collapsed faces and renumber
        F, patch, sgn, fid, vf_ptr, vf_idx, pf_ptr, pf_idx, dirty = _collapse(
            wu, wv, F, patch, sgn, fid, owner, vf_ptr, vf_idx, pf_ptr, pf_idx, moved
        )
        target[wu] = -1
        tr = _lap(timing, "update", tr)
        if verbose:
            print(
                f"round {it}: {len(wu)} collapses, {len(F)} faces, "
                f"{dirty.sum()} dirty, {time.perf_counter() - t0:.2f}s"
            )

    used = np.unique(F)
    vmap = np.full(nv, -1)
    vmap[used] = np.arange(len(used))
    out = pv.PolyData.from_regular_faces(V[used], vmap[F])
    out.cell_data[label_name] = labels[fid]
    if verbose:
        print(f"done in {time.perf_counter() - t0:.2f}s: {surf.n_cells} -> {out.n_cells} faces")
        print("time per step: " + ", ".join(f"{k} {t:.2f}s" for k, t in timing.items()))
    return out
