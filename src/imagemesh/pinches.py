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
cube. A cube with a pinch is repaired by the single voxel change (to a label of
the cube or 0) that leaves the whole cube pinch-free, preferring to grow a cell
into the background (0), then (vertex pinches) to grow it into two background
voxels, then to exchange voxels between cells, and only then to shrink a
cell. With separate_cells, a cell does not grow into a background voxel that
touches (26-neighbourhood) another cell, so that cells separated by background
(e.g. after zero_edges) stay separated. Repairs can create new
pinches in neighbouring cubes, so the scan is repeated until no pinch is left.
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
def _sweep(img, separate_cells, repair):
    """
    One pass over all 2x2x2 cubes. Returns the number of cubes with a pinch
    and, if repair, the number of voxels changed (in place).
    """
    nx, ny, nz = img.shape
    v = np.empty(8, img.dtype)
    t = np.empty(8, img.dtype)
    cand = np.empty(9, img.dtype)
    n_pinched = 0
    n_changed = 0
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                uniform = True
                for m in range(8):
                    v[m] = img[i + (m & 1), j + ((m >> 1) & 1), k + ((m >> 2) & 1)]
                    if v[m] != v[0]:
                        uniform = False
                if uniform or _cube_ok(v):
                    continue
                n_pinched += 1
                if not repair:
                    continue
                # candidate labels: those of the cube and 0
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
                best_score = -1
                best_m = -1
                best_label = v[0]
                for m in range(8):
                    old = v[m]
                    for c in range(n_cand):
                        label = cand[c]
                        if label == old:
                            continue
                        if label == 0:
                            score = 0  # shrink a cell
                        elif old != 0:
                            score = 1  # exchange between cells
                        else:
                            score = 2  # grow a cell into the background
                        if score <= best_score:
                            continue
                        t[:] = v
                        t[m] = label
                        if not _cube_ok(t):
                            continue
                        x, y, z = i + (m & 1), j + ((m >> 1) & 1), k + ((m >> 2) & 1)
                        if (
                            separate_cells
                            and score == 2
                            and _touches_other_cell(img, x, y, z, label)
                        ):
                            continue
                        best_score, best_m, best_label = score, m, label
                if best_score < 2:
                    # a vertex pinch needs two background voxels to grow a
                    # face-connected path between the corners
                    done = False
                    for m1 in range(8):
                        if done or v[m1] != 0:
                            continue
                        for m2 in range(m1 + 1, 8):
                            if done or v[m2] != 0:
                                continue
                            for c in range(1, n_cand):
                                label = cand[c]
                                if label == 0:
                                    continue
                                t[:] = v
                                t[m1] = label
                                t[m2] = label
                                if not _cube_ok(t):
                                    continue
                                if separate_cells and (
                                    _touches_other_cell(
                                        img,
                                        i + (m1 & 1),
                                        j + ((m1 >> 1) & 1),
                                        k + ((m1 >> 2) & 1),
                                        label,
                                    )
                                    or _touches_other_cell(
                                        img,
                                        i + (m2 & 1),
                                        j + ((m2 >> 1) & 1),
                                        k + ((m2 >> 2) & 1),
                                        label,
                                    )
                                ):
                                    continue
                                for m in (m1, m2):
                                    img[i + (m & 1), j + ((m >> 1) & 1), k + ((m >> 2) & 1)] = label
                                n_changed += 2
                                done = True
                                break
                    if done:
                        continue
                if best_m >= 0:
                    img[i + (best_m & 1), j + ((best_m >> 1) & 1), k + ((best_m >> 2) & 1)] = (
                        best_label
                    )
                    n_changed += 1
    return n_pinched, n_changed


def count_pinches(img):
    """Number of 2x2x2 cubes of the label image that contain a pinch."""
    return _sweep(np.asarray(img), False, False)[0]


def remove_pinches(img, separate_cells=True, max_iterations=20):
    """
    Repair the pinches of all labels of img (0 is the background), see the
    module docstring. Returns a repaired copy.
    """
    img = np.array(img, copy=True)
    for it in range(max_iterations):
        n_pinched, n_changed = _sweep(img, separate_cells, True)
        print(f"remove_pinches, pass {it}: {n_pinched} pinched cubes, {n_changed} voxels changed")
        if n_pinched == 0:
            return img
        if n_changed == 0:
            break
    n_left = count_pinches(img)
    if n_left:
        print(f"remove_pinches: {n_left} pinched cubes left")
    return img
