"""
Multi-label surfaces of label images: surface nets extraction with
constrained smoothing, label transfer and surface checks.

Conventions: cell data boundary_labels (a, b) per triangle, the normal points
out of region a into region b; 0 is the background (outside).
"""

import warnings

import numba as nb
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from imagemesh.smoothing import smooth_surface_net


def extract_surface(
    grid, smoothing_scale=1.2, iterations=16, scalars="data", background_value=0, degenerate=None
):
    """
    Smoothed, triangulated multi-label surface of the label image grid
    (pv.ImageData with the labels as cell data `scalars`), with cell data
    'boundary_labels' (n_faces, 2): surface nets (contour_labels without
    smoothing) followed by a constrained smoothing that keeps the points on
    the bounding box on their box face.
    The displacement of each point is limited to smoothing_scale * dx; dx is
    stored in the field data.
    degenerate: build_stencils' treatment of quads whose opposite corners have
    nested junction stencils. None (VTK) keeps junction curves smoothed as
    curves; "full" un-pins those corners at every voxel step of a one-voxel-wide
    strip between two junction curves (e.g. where the falx meets the outer
    boundary) and creases the strip by up to a voxel.
    """
    surf = surface_quads(grid, scalars=scalars, background_value=background_value)
    dx = np.min(grid.spacing)
    surf = smooth_surface_net(
        surf,
        iterations=iterations,
        distance=dx,
        scale=smoothing_scale,
        fix_bounds=True,
        degenerate=degenerate,
    )
    surf.field_data["dx"] = [dx]
    return surf


def surface_quads(grid, scalars="data", background_value=0, label_name="boundary_labels"):
    """
    Quad surface of the label image grid (pv.ImageData with the labels as cell
    data `scalars`): every voxel face between two labels, the points on the
    voxel corners, cell data label_name (a, b) as contour_labels(smoothing=False,
    output_mesh_type="quads"): b is the background or the larger label, the
    normal points from a into b.

    Where a label touches itself only along an edge or at a corner (a pinch),
    points are split into one copy per sheet. At an edge with one diagonal
    pair of equal labels, that label connects through the edge and the other
    two voxels get their own copies of the edge. With both diagonals equal (a
    checkerboard), one decision holds for the whole run of equal checkerboards
    along the edge: the label that continues diagonally through more ends of
    the run connects, then the background, then the smaller label. The copies
    of a corner are the groups of its faces joined at its edges, so both ends
    of an edge agree and the surface is closed (vtkSurfaceNets3D splits the two
    ends inconsistently and leaves slits, see close_quad_holes). A split edge
    whose sheets are joined at both ends (e.g. a neck of the connecting label
    that ends in a third label at both ends) stays a pinch.
    """
    dims = np.array(grid.dimensions) - 1
    img = np.asarray(grid.cell_data[scalars]).reshape(tuple(dims), order="F")
    pad = np.pad(img.astype(np.int64), 1, constant_values=background_value)
    corner, lface, labels = _voxel_faces(pad, background_value)
    ids, inv = np.unique(corner.ravel(), return_inverse=True)
    comp, ncomp = _corner_copies(pad, ids, background_value)
    offset = np.zeros(len(ids) + 1, np.int64)
    np.cumsum(ncomp, out=offset[1:])
    inv = inv.reshape(-1, 4)
    quads = offset[inv] + comp[inv, lface]
    idx = np.repeat(ids, ncomp)
    n = dims + 1  # corners per axis
    ijk = np.stack([idx % n[0], idx // n[0] % n[1], idx // (n[0] * n[1])], axis=1)
    M = np.asarray(grid.index_to_physical_matrix)  # with the direction matrix
    points = ijk @ M[:3, :3].T + M[:3, 3]
    if np.linalg.det(M[:3, :3]) < 0:  # a reflection turns the normals around
        quads = quads[:, ::-1]
    surf = pv.PolyData.from_regular_faces(points, quads)
    surf.cell_data[label_name] = labels.astype(img.dtype)
    return surf


@nb.njit(parallel=True, cache=True)
def _voxel_faces(pad, bg):
    """
    The faces between differently labelled voxels of the padded image pad: per
    face its 4 corners (linear corner ids, counterclockwise around the normal),
    its local face id at each corner (see _corner_copies) and its labels (a, b).
    Corner (i, j, k) lies between the padded voxels i, i + 1 (and j, k alike).
    """
    nx, ny, nz = pad.shape
    cn = (nx - 1, ny - 1, nz - 1)  # corners per axis
    count = np.zeros(nz + 1, np.int64)
    for z in nb.prange(nz):
        c = 0
        for y in range(ny):
            for x in range(nx):
                lab = pad[x, y, z]
                c += (x + 1 < nx and pad[x + 1, y, z] != lab) + (
                    y + 1 < ny and pad[x, y + 1, z] != lab
                )
                c += z + 1 < nz and pad[x, y, z + 1] != lab
        count[z + 1] = c
    for z in range(nz):
        count[z + 1] += count[z]
    m = count[nz]
    corner = np.empty((m, 4), np.int64)
    lface = np.empty((m, 4), np.int64)
    labels = np.empty((m, 2), np.int64)
    for zz in nb.prange(nz):
        z = np.int64(zz)
        f = count[z]
        for y in range(ny):
            for x in range(nx):
                for a in range(3):
                    if (x + 1, y + 1, z + 1)[a] == pad.shape[a]:
                        continue
                    A = pad[x, y, z]
                    B = pad[x + (a == 0), y + (a == 1), z + (a == 2)]
                    if A == B:
                        continue
                    flip = not (B == bg or (A != bg and A < B))
                    labels[f, 0], labels[f, 1] = (B, A) if flip else (A, B)
                    p, q = (a + 1) % 3, (a + 2) % 3
                    for r in range(4):
                        k = (4 - r) % 4 if flip else r  # (0, 3, 2, 1): reversed
                        s, t = int(k == 1 or k == 2), int(k >= 2)
                        cx = x + (s - 1) * (p == 0) + (t - 1) * (q == 0)
                        cy = y + (s - 1) * (p == 1) + (t - 1) * (q == 1)
                        cz = z + (s - 1) * (p == 2) + (t - 1) * (q == 2)
                        corner[f, r] = cx + cn[0] * (cy + cn[1] * cz)
                        lface[f, r] = 4 * a + (1 - s) + 2 * (1 - t)
                    f += 1
    return corner, lface, labels[:m]


@nb.njit(cache=True)
def _local_face(k1, a):
    """Local face id of the face between voxels k1 and k1 ^ (1 << a) of a corner."""
    k = k1 & ~(1 << a)
    return 4 * a + ((k >> ((a + 1) % 3)) & 1) + 2 * ((k >> ((a + 2) % 3)) & 1)


@nb.njit(cache=True)
def _root(parent, i):
    while parent[i] != i:
        i = parent[i]
    return i


@nb.njit(cache=True)
def _join(parent, i, j):
    i, j = _root(parent, i), _root(parent, j)
    parent[max(i, j)] = min(i, j)


@nb.njit(cache=True)
def _label(pad, x, y, z, bg):
    """Label of the padded voxel (x, y, z), bg outside."""
    if 0 <= x < pad.shape[0] and 0 <= y < pad.shape[1] and 0 <= z < pad.shape[2]:
        return pad[x, y, z]
    return bg


@nb.njit(cache=True)
def _run_end(pad, g, e, sh, X, Y, bg):
    """
    The 2x2 cross-section of voxels g (4, 3) in cyclic order is a checkerboard
    with X on g[0], g[2] and Y on g[1], g[3]. Walking along axis e in direction
    sh to the first layer that differs: whether X, and Y, continue diagonally
    through it.
    """
    q = np.empty(4, np.int64)
    k = sh
    while True:
        for j in range(4):
            q[j] = _label(
                pad, g[j, 0] + k * (e == 0), g[j, 1] + k * (e == 1), g[j, 2] + k * (e == 2), bg
            )
        if q[0] == X and q[2] == X and q[1] == Y and q[3] == Y:
            k += sh
            continue
        return int(q[0] == X and q[2] == X), int(q[1] == Y and q[3] == Y)


@nb.njit(parallel=True, cache=True)
def _corner_copies(pad, ids, bg):
    """
    The copies of the corners ids: per corner the copy of each of its 12 local
    faces (-1 if absent) and the number of copies. The 8 voxels around a corner
    are k = dx + 2 dy + 4 dz; local face 4 a + s + 2 t lies between the voxels
    differing in bit a whose other bits (a + 1, a + 2) % 3 are s, t.
    """
    cn0, cn1 = pad.shape[0] - 1, pad.shape[1] - 1
    comp = np.full((len(ids), 12), -1, np.int8)
    ncomp = np.zeros(len(ids), np.int64)
    for i in nb.prange(len(ids)):
        x, y, z = ids[i] % cn0, ids[i] // cn0 % cn1, ids[i] // (cn0 * cn1)
        lab = np.empty(8, np.int64)
        for k in range(8):
            lab[k] = pad[x + (k & 1), y + ((k >> 1) & 1), z + (k >> 2)]
        parent = np.arange(12)
        # the 6 edges (axis e, side d): their 4 voxels C in cyclic order and
        # the faces f[j] between C[j] and C[j + 1]
        C = np.empty(4, np.int64)
        f = np.empty(4, np.int64)
        lc = np.empty(4, np.int64)
        g = np.empty((4, 3), np.int64)
        for e in range(3):
            u, v = (e + 1) % 3, (e + 2) % 3
            for d in range(2):
                C[0] = d << e
                C[1] = C[0] | (1 << u)
                C[2] = C[1] | (1 << v)
                C[3] = C[0] | (1 << v)
                for j in range(4):
                    lc[j] = lab[C[j]]
                    f[j] = _local_face(C[j], u if j % 2 == 0 else v)
                if lc[0] == lc[2] and lc[1] == lc[3] and lc[0] != lc[1]:
                    # checkerboard: one decision for the whole run of equal
                    # checkerboards along e. The label whose diagonal continues
                    # through more end layers of the run connects (separating
                    # it fails at such an end), then the background, then the
                    # smaller label
                    for j in range(4):
                        g[j, 0] = x + (C[j] & 1)
                        g[j, 1] = y + ((C[j] >> 1) & 1)
                        g[j, 2] = z + (C[j] >> 2)
                    jx, jy = _run_end(pad, g, e, -1, lc[0], lc[1], bg)
                    kx, ky = _run_end(pad, g, e, 1, lc[0], lc[1], bg)
                    jx, jy = jx + kx, jy + ky
                    if jx != jy:
                        sep_odd = jx > jy  # the voxels C[1], C[3] are separated
                    else:
                        sep_odd = lc[0] == bg or (lc[1] != bg and lc[0] < lc[1])
                elif lc[0] == lc[2] and lc[1] != lc[0] and lc[3] != lc[0]:
                    sep_odd = True
                elif lc[1] == lc[3] and lc[0] != lc[1] and lc[2] != lc[1]:
                    sep_odd = False
                else:
                    first = -1
                    for j in range(4):
                        if lc[j] != lc[(j + 1) % 4]:
                            if first < 0:
                                first = f[j]
                            else:
                                _join(parent, first, f[j])
                    continue
                if sep_odd:  # faces around C[1] and around C[3]
                    _join(parent, f[0], f[1])
                    _join(parent, f[2], f[3])
                else:  # around C[0] and around C[2]
                    _join(parent, f[3], f[0])
                    _join(parent, f[1], f[2])
        nc = 0
        root_comp = np.full(12, -1, np.int64)
        for lf in range(12):
            a, s, t = lf // 4, lf & 1, (lf >> 1) & 1
            k1 = (s << ((a + 1) % 3)) | (t << ((a + 2) % 3))
            if lab[k1] == lab[k1 | (1 << a)]:
                continue
            r = _root(parent, lf)
            if root_comp[r] < 0:
                root_comp[r] = nc
                nc += 1
            comp[i, lf] = root_comp[r]
        ncomp[i] = nc
    return comp, ncomp


def close_quad_holes(surf, label_name="boundary_labels"):
    """
    Close the single-quad holes of a contour_labels quad mesh.

    vtkSurfaceNets3D (VTK 9.7) splits the corners of pinches into coincident
    copies, but not consistently at the two ends of an edge: the faces on
    either side of the edge then use different copies and a slit opens, e.g.
    for label 1 at (0, 2, 2), (1, 0, 0), (1, 0, 1), (2, 0, 0), (2, 1, 1) and
    (2, 2, 2) of a 3^3 block of label 2. The slit's rim is four open edges of
    faces with one label pair; each such loop is closed with a zero-area quad
    of that label pair, oriented like its neighbours, which opens up in the
    smoothing. Its triangulation can duplicate a neighbouring triangle; prefer
    surface_quads, which splits consistently and leaves no slits.
    """
    quads = surf.regular_faces
    labels = np.asarray(surf.cell_data[label_name])
    new_quads, new_faces, n_open = _close_quad_holes(quads.astype(np.int64), labels, surf.n_points)
    if n_open:
        warnings.warn(f"close_quad_holes: {n_open} open edges are not on a single-quad hole")
    if not len(new_quads):
        return surf
    out = pv.PolyData.from_regular_faces(
        np.asarray(surf.points), np.vstack([quads, new_quads.astype(quads.dtype)])
    )
    out.cell_data[label_name] = np.vstack([labels, labels[new_faces]])
    for name in surf.field_data:
        out.field_data[name] = surf.field_data[name]
    return out


@nb.njit(cache=True)
def _close_quad_holes(Q, L, n):
    """
    New quads closing the loops of four open edges of quads Q with one label
    pair L, the quad whose labels each new quad takes, and the number of open
    edges left.
    """
    m = 4 * len(Q)
    key = np.empty(m, np.int64)
    for f in range(len(Q)):
        for i in range(4):
            key[4 * f + i] = Q[f, i] * n + Q[f, (i + 1) % 4]
    skey = np.sort(key)
    # an open half-edge (a, b) has no reverse (b, a); the missing quad runs along (b, a)
    U, V, F = [], [], []
    for f in range(len(Q)):
        for i in range(4):
            a, b = Q[f, i], Q[f, (i + 1) % 4]
            s = np.searchsorted(skey, b * n + a)
            if s == m or skey[s] != b * n + a:
                U.append(b)
                V.append(a)
                F.append(f)
    U, V, F = np.array(U, np.int64), np.array(V, np.int64), np.array(F, np.int64)
    # open edges grouped by start point
    order = np.argsort(U, kind="mergesort")
    su = U[order]
    used = np.zeros(len(U), np.bool_)

    def out_edges(p):
        return order[np.searchsorted(su, p) : np.searchsorted(su, p, side="right")]

    def fits(e, f0):
        return not used[e] and L[F[e], 0] == L[f0, 0] and L[F[e], 1] == L[f0, 1]

    new_quads, new_faces = [], []
    for e0 in range(len(U)):
        if used[e0]:
            continue
        f0, p0, p1 = F[e0], U[e0], V[e0]
        found = False
        for e1 in out_edges(p1):
            p2 = V[e1]
            if found or not fits(e1, f0) or p2 == p0 or p2 == p1:
                continue
            for e2 in out_edges(p2):
                p3 = V[e2]
                if found or not fits(e2, f0) or p3 == p0 or p3 == p1 or p3 == p2:
                    continue
                for e3 in out_edges(p3):
                    if found or not fits(e3, f0) or V[e3] != p0:
                        continue
                    found = True
                    used[e0] = used[e1] = used[e2] = used[e3] = True
                    new_quads.append((p0, p1, p2, p3))
                    new_faces.append(f0)
    nq = np.empty((len(new_quads), 4), np.int64)
    for k in range(len(new_quads)):
        nq[k] = new_quads[k]
    return nq, np.array(new_faces, np.int64), len(U) - 4 * len(new_quads)


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
