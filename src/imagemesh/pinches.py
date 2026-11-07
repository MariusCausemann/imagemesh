"""
Removal of pinches from label images, before surface extraction with surface
nets (pyvista's contour_labels).

A label L is pinched where two of its voxels touch only along an edge or at a
corner. Surface nets then creates a non-manifold surface for L (an edge shared
by four faces of the same label pair, or a vertex shared by two separate fans),
which smoothing cannot resolve: the pinch is kept as a zero-thickness neck.

Per label L (L vs. not L), the critical configurations are (Latecki's
well-composedness):

* edge pinch: in a 2x2 window of any axis plane, one diagonal is L and the
  other two voxels are not.
* vertex pinch: in a 2x2x2 cube, two opposite corners are L and the other six
  are not, or vice versa (six are L and the two opposite corners are not).

Every 2x2 window is a face of some 2x2x2 cube, so the image is scanned cube by
cube, and a cube with a pinch is repaired by the change of one voxel, or of two
voxels of the same kind (a cube with two pinched faces needs two), that leaves
the whole cube pinch-free.

Cells count as connected through faces only, so a pinch is not a connection
(cuts that left only a pinch are joined beforehand, e.g. by
nbmorph.reconnect). The repairs, in order of preference:

1. a cell voxel becomes background (separates the voxels that touch);
2. a cell grows into a background voxel that touches no other cell;
3. a voxel is exchanged between two cells;
4. a cell grows into background next to another cell.

A voxel is only taken from a cell (1, 3) if that cannot split the cell (its
face neighbours in the cell stay face-connected within its 3x3x3
neighbourhood) and it is not the end of a line (exactly one face neighbour in
the cell), so that thin processes are not eaten from their ends. So
remove_pinches never splits a cell. A voxel that was changed
is never changed again, so the passes cannot cycle.

Blockwise: the cubes are repaired in 27 subfields (cube origins equal modulo
3, in whole-image coordinates), each in parallel. A repair reads the voxels
within 1 of its cube and writes only its cube, so the repairs of a subfield do
not see each other: the result does not depend on the number of threads, and
a block with a margin gets the same result as the whole image.
"""

import numba as nb
import numpy as np


def _cube_faces():
    """Voxel ids (k = x + 2y + 4z) of the 6 faces of a cube, in cyclic order."""
    faces = []
    for d in range(3):
        p, q = [b for b in range(3) if b != d]
        for s in range(2):
            faces.append(
                [(s << d) | (x << p) | (y << q) for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))]
            )
    return np.array(faces, np.int64)


FACES = _cube_faces()
D6 = np.array([(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)], np.int64)


@nb.njit(cache=True)
def _cube_ok(v):
    """True if the cube (8 labels, k = x + 2y + 4z) contains no pinch of any label."""
    for f in range(6):
        a, b, c, d = v[FACES[f, 0]], v[FACES[f, 1]], v[FACES[f, 2]], v[FACES[f, 3]]
        if a == c and b != a and d != a:
            return False
        if b == d and a != b and c != b:
            return False
    for k in range(4):
        c1, c2 = v[k], v[7 - k]
        # the other six voxels: all different from the corners' label, or all
        # equal to one label different from both corners
        n_same = 0
        n_eq = 0
        L = v[k ^ 1]
        for m in range(8):
            if m == k or m == 7 - k:
                continue
            if v[m] == c1:
                n_same += 1
            if v[m] == L:
                n_eq += 1
        if c1 == c2 and n_same == 0:
            return False
        if n_eq == 6 and c1 != L and c2 != L:
            return False
    return True


@nb.njit(cache=True)
def _touches_other_cell(img, i, j, k, label):
    """True if a voxel in the 26-neighbourhood of (i, j, k) is a cell other than label."""
    nx, ny, nz = img.shape
    for a in range(max(i - 1, 0), min(i + 2, nx)):
        for b in range(max(j - 1, 0), min(j + 2, ny)):
            for c in range(max(k - 1, 0), min(k + 2, nz)):
                w = img[a, b, c]
                if w != 0 and w != label:
                    return True
    return False


@nb.njit(cache=True)
def _removable(img, x, y, z, label):
    """True if voxel (x, y, z) can be taken from label without splitting it or
    shortening a line of it: it is not the end of a line (exactly one face
    neighbour of label), and its face neighbours of label stay face-connected
    within its 3x3x3 neighbourhood (without the voxel)."""
    nx, ny, nz = img.shape
    box = np.zeros((3, 3, 3), np.bool_)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                xx, yy, zz = x + a - 1, y + b - 1, z + c - 1
                if 0 <= xx < nx and 0 <= yy < ny and 0 <= zz < nz:
                    box[a, b, c] = img[xx, yy, zz] == label
    box[1, 1, 1] = False
    nface = 0
    start = -1
    for f in range(6):
        a, b, c = 1 + D6[f, 0], 1 + D6[f, 1], 1 + D6[f, 2]
        if box[a, b, c]:
            nface += 1
            start = f
    if nface == 0:
        return True  # an isolated voxel
    if nface == 1:
        return False  # the end of a line: taking it would shorten the line
    seen = np.zeros((3, 3, 3), np.bool_)
    stack = np.empty((27, 3), np.int64)
    a, b, c = 1 + D6[start, 0], 1 + D6[start, 1], 1 + D6[start, 2]
    seen[a, b, c] = True
    stack[0, 0], stack[0, 1], stack[0, 2] = a, b, c
    top = 1
    while top > 0:
        top -= 1
        a, b, c = stack[top, 0], stack[top, 1], stack[top, 2]
        for f in range(6):
            aa, bb, cc = a + D6[f, 0], b + D6[f, 1], c + D6[f, 2]
            if 0 <= aa < 3 and 0 <= bb < 3 and 0 <= cc < 3:
                if box[aa, bb, cc] and not seen[aa, bb, cc]:
                    seen[aa, bb, cc] = True
                    stack[top, 0], stack[top, 1], stack[top, 2] = aa, bb, cc
                    top += 1
    for f in range(6):
        a, b, c = 1 + D6[f, 0], 1 + D6[f, 1], 1 + D6[f, 2]
        if box[a, b, c] and not seen[a, b, c]:
            return False
    return True


# kinds of repair, in order of preference: separate first (a pinch is not a
# connection), then grow into background away from other cells, then exchange
# between cells, and last grow next to another cell. Rank 2 * kind for one
# voxel, 2 * kind + 1 for two.
SHRINK, GROW, EXCHANGE, GROW_NEXT = 0, 1, 2, 3
NO = 99


@nb.njit(parallel=True, cache=True)
def _pinched_cubes(img):
    nx, ny, nz = img.shape
    out = np.zeros((max(nx - 1, 0), max(ny - 1, 0), max(nz - 1, 0)), np.bool_)
    for i in nb.prange(nx - 1):
        v = np.empty(8, img.dtype)
        for j in range(ny - 1):
            for k in range(nz - 1):
                uniform = True
                for m in range(8):
                    v[m] = img[i + (m & 1), j + ((m >> 1) & 1), k + ((m >> 2) & 1)]
                    if v[m] != v[0]:
                        uniform = False
                if not uniform and not _cube_ok(v):
                    out[i, j, k] = True
    return out


@nb.njit(cache=True)
def _vox(i, j, k, m):
    return i + (m & 1), j + ((m >> 1) & 1), k + ((m >> 2) & 1)


@nb.njit(cache=True)
def _kind(img, x, y, z, old, label, separate_cells):
    """Kind of the change of voxel (x, y, z) from old to label."""
    if label == 0:
        return SHRINK
    if old != 0:
        return EXCHANGE
    if separate_cells and _touches_other_cell(img, x, y, z, label):
        return GROW_NEXT
    return GROW


@nb.njit(cache=True)
def _repair_cube(img, frozen, i, j, k, separate_cells):
    """Apply the preferred repair of the cube at (i, j, k): the change of one
    voxel, or of two voxels of the same kind to the same label (a cube in
    which two faces are pinched needs two), that leaves the cube pinch-free
    and takes no voxel from a cell that it could split. Returns the number of
    voxels changed."""
    v = np.empty(8, img.dtype)
    t = np.empty(8, img.dtype)
    cand = np.empty(9, img.dtype)
    uniform = True
    for m in range(8):
        x, y, z = _vox(i, j, k, m)
        v[m] = img[x, y, z]
        if v[m] != v[0]:
            uniform = False
    if uniform or _cube_ok(v):
        return 0
    n_cand = 1
    cand[0] = 0
    for m in range(8):
        new = True
        for c in range(n_cand):
            if cand[c] == v[m]:
                new = False
        if new:
            cand[n_cand] = v[m]
            n_cand += 1
    best = NO
    best_m1, best_m2 = -1, -1
    best_label = v[0]
    for m1 in range(8):
        x1, y1, z1 = _vox(i, j, k, m1)
        if frozen[x1, y1, z1]:
            continue  # changed before: never changed back
        for c in range(n_cand):
            label = cand[c]
            if label == v[m1]:
                continue
            k1 = _kind(img, x1, y1, z1, v[m1], label, separate_cells)
            # one voxel (m2 = -1), or two voxels of the same kind (m2 > m1)
            for m2 in range(m1, 8):
                if m2 == m1:
                    r = 2 * k1
                    m2s = -1
                else:
                    x2, y2, z2 = _vox(i, j, k, m2)
                    if frozen[x2, y2, z2] or v[m2] == label:
                        continue
                    k2 = _kind(img, x2, y2, z2, v[m2], label, separate_cells)
                    if (k1 == SHRINK) != (k2 == SHRINK) or (k1 == EXCHANGE) != (k2 == EXCHANGE):
                        continue  # not the same kind
                    r = 2 * max(k1, k2) + 1
                    m2s = m2
                if r >= best:
                    continue
                t[:] = v
                t[m1] = label
                if m2s >= 0:
                    t[m2s] = label
                if not _cube_ok(t):
                    continue
                # taking a voxel from a cell must not split it
                if v[m1] != 0 and not _removable(img, x1, y1, z1, v[m1]):
                    continue
                if m2s >= 0 and v[m2s] != 0:
                    img[x1, y1, z1] = label  # m2 is tested after m1 is changed
                    ok = _removable(img, x2, y2, z2, v[m2s])
                    img[x1, y1, z1] = v[m1]
                    if not ok:
                        continue
                best, best_m1, best_m2, best_label = r, m1, m2s, label
    if best_m1 < 0:
        return 0
    x, y, z = _vox(i, j, k, best_m1)
    img[x, y, z] = best_label
    frozen[x, y, z] = True
    if best_m2 < 0:
        return 1
    x, y, z = _vox(i, j, k, best_m2)
    img[x, y, z] = best_label
    frozen[x, y, z] = True
    return 2


@nb.njit(parallel=True, cache=True)
def _repair_subfield(img, frozen, cubes, idx, separate_cells):
    changed = np.zeros(idx.size, np.int64)
    for q in nb.prange(idx.size):
        c = idx[q]
        changed[q] = _repair_cube(
            img, frozen, cubes[c, 0], cubes[c, 1], cubes[c, 2], separate_cells
        )
    return changed.sum()


def count_pinches(img):
    """Number of 2x2x2 cubes of the label image that contain a pinch."""
    return int(_pinched_cubes(np.ascontiguousarray(img)).sum())


def remove_pinches(
    img,
    separate_cells=True,
    max_iterations=20,
    origin=(0, 0, 0),
    verbose=True,
):
    """
    Repair the pinches of all labels of img (0 is the background) without
    splitting any cell, see the module docstring. Returns a repaired copy.

    Args:
        img (np.ndarray): 3D label image.
        separate_cells (bool): Prefer growing a cell into background voxels
            that do not touch another cell; growing next to another cell is
            the last resort.
        max_iterations (int): Maximum number of passes.
        origin (tuple): Position of img[0, 0, 0] in the whole image, when img
            is a block of it: the subfields are taken in whole-image
            coordinates, so that a block with a margin gets the same result.
        verbose (bool): Print a line per pass.
    """
    img = np.array(img, copy=True, order="C")
    frozen = np.zeros(img.shape, np.bool_)  # changed voxels: never changed back
    for it in range(max_iterations):
        cubes = np.argwhere(_pinched_cubes(img)).astype(np.int64)
        if len(cubes) == 0:
            return img
        changed = 0
        g = cubes + np.asarray(origin, np.int64)
        sub = (g[:, 0] % 3) * 9 + (g[:, 1] % 3) * 3 + g[:, 2] % 3
        order = np.argsort(sub, kind="stable")
        bounds = np.searchsorted(sub[order], np.arange(28))
        for f in range(27):
            idx = order[bounds[f] : bounds[f + 1]]
            if idx.size:
                changed += int(_repair_subfield(img, frozen, cubes, idx, separate_cells))
        if verbose:
            print(
                f"remove_pinches, pass {it}: {len(cubes)} pinched cubes, {changed} voxels changed"
            )
        if changed == 0:
            break
    n_left = count_pinches(img)
    if n_left and verbose:
        print(f"remove_pinches: {n_left} pinched cubes left")
    return img
