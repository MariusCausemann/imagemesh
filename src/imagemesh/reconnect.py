"""
Local reconnection: undo the cuts an operation makes through a cell, up to a
length scale s, without global connected components.

Given labels (before) and target (after an operation), a cell counts as cut if
two of its voxels in target are connected through the cell in labels within a
tile, but not through the cell in target within the tile extended by s. Where
that happens, the cheapest path through the cell's voxels in labels is restored
and thickened by t through voxels the operation removed from the cell.

Tiles lie on a fixed grid of the whole image (cores of 2s voxels, extended by
s/2 on every side, so that they overlap by s). They are processed in 27
subfields (tile indices equal modulo 3) one after another, each in parallel:
the tiles of a subfield are far enough apart not to see each other's
restorations, and every later subfield sees those of the earlier ones, so a
cut that two overlapping tiles both contain is joined only once (joining it
twice along different paths would create a small tunnel). The result does not
depend on the number of threads, and a block with a margin gets the same result
as the whole image.

What this gives:

* a cut whose pieces are joined in labels within s/2 of it is always repaired
  (it lies in the core of some tile);
* a loop smaller than about s that the operation cuts stays cut (its pieces
  are connected in target within the extended tile): small handles and
  tunnels of a ragged segmentation can still be removed;
* a restored connection is at least 2t + 1 voxels thick where the cell was
  that thick in labels.

A path avoids voxels that another cell took over where it can; where it
cannot, they are given back to their cell in labels, and since that may cut the
other cell, the check is repeated (max_passes). Cuts that are longer than a
tile (a thin process that the operation removes over its whole length) are not
seen.
"""

import heapq

import numba
import numpy as np

FACES = np.array([(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)], np.int64)
OTHER = 1000  # cost of a voxel that another cell took over


@numba.njit(cache=True)
def _label(img, x0, x1, y0, y1, z0, z1):
    """Face-connected components of equal non-zero labels of img within the
    box [x0, x1) x [y0, y1) x [z0, z1). Returns the component array of the box,
    the number of components, their sizes, and whether each touches a side of
    the box that is not a side of the image."""
    nx, ny, nz = img.shape
    sx, sy, sz = x1 - x0, y1 - y0, z1 - z0
    comp = np.zeros((sx, sy, sz), np.int32)
    stack = np.empty(sx * sy * sz, np.int64)
    sizes = np.zeros(sx * sy * sz + 1, np.int64)
    touch = np.zeros(sx * sy * sz + 1, np.bool_)
    n = 0
    for i in range(sx):
        for j in range(sy):
            for k in range(sz):
                v = img[x0 + i, y0 + j, z0 + k]
                if v == 0 or comp[i, j, k] != 0:
                    continue
                n += 1
                comp[i, j, k] = n
                stack[0] = (i * sy + j) * sz + k
                top = 1
                while top > 0:
                    top -= 1
                    p = stack[top]
                    a = p // (sy * sz)
                    b = (p // sz) % sy
                    c = p % sz
                    sizes[n] += 1
                    if (
                        (a == 0 and x0 > 0)
                        or (a == sx - 1 and x1 < nx)
                        or (b == 0 and y0 > 0)
                        or (b == sy - 1 and y1 < ny)
                        or (c == 0 and z0 > 0)
                        or (c == sz - 1 and z1 < nz)
                    ):
                        touch[n] = True
                    for f in range(6):
                        aa, bb, cc = a + FACES[f, 0], b + FACES[f, 1], c + FACES[f, 2]
                        if not (0 <= aa < sx and 0 <= bb < sy and 0 <= cc < sz):
                            continue
                        if comp[aa, bb, cc] == 0 and img[x0 + aa, y0 + bb, z0 + cc] == v:
                            comp[aa, bb, cc] = n
                            stack[top] = (aa * sy + bb) * sz + cc
                            top += 1
    return comp, n, sizes[: n + 1], touch[: n + 1]


@numba.njit(cache=True)
def _repair(A, B, out, ca, c, label, ce, esize, K, E, t):
    """Join the pieces of the component c (of labels within the tile core K)
    that are not connected in target within E. Returns 1 if they could not
    all be joined."""
    kx0, kx1, ky0, ky1, kz0, kz1 = K
    ex0, ey0, ez0 = E[0], E[2], E[4]
    sx, sy, sz = kx1 - kx0, ky1 - ky0, kz1 - kz0
    n = sx * sy * sz
    ne = esize.size - 1
    ingroup = np.zeros(ne + 1, np.bool_)
    for i in range(sx):
        for j in range(sy):
            for k in range(sz):
                x, y, z = kx0 + i, ky0 + j, kz0 + k
                if ca[i, j, k] == c and B[x, y, z] == label:
                    ingroup[ce[x - ex0, y - ey0, z - ez0]] = True
    groups = np.flatnonzero(ingroup)
    nkeep = groups.size
    if nkeep < 2:
        return 0
    smallest = groups[np.argmin(esize[groups])]
    joined = np.zeros(ne + 1, np.bool_)
    joined[smallest] = True
    rest = np.zeros(n, np.bool_)  # restored path voxels (core-local)
    dist = np.empty(n, np.int64)
    parent = np.empty(n, np.int64)
    took = 0
    big = np.int64(1) << 60
    for _ in range(nkeep - 1):
        # cheapest path from the joined set to any other kept piece, through
        # the component's voxels: 0 within the cell in target, 1 through
        # background, OTHER through voxels another cell took over
        dist[:] = big
        parent[:] = -1
        heap = [(np.int64(0), np.int64(0))]
        heap.pop()
        for p in range(n):
            i, j, k = p // (sy * sz), (p // sz) % sy, p % sz
            if ca[i, j, k] != c:
                continue
            x, y, z = kx0 + i, ky0 + j, kz0 + k
            if rest[p] or (B[x, y, z] == label and joined[ce[x - ex0, y - ey0, z - ez0]]):
                dist[p] = 0
                heap.append((np.int64(0), np.int64(p)))
        found = -1
        while found < 0 and len(heap) > 0:
            d, p = heapq.heappop(heap)
            if d > dist[p]:
                continue
            i, j, k = p // (sy * sz), (p // sz) % sy, p % sz
            for f in range(6):
                a, b, cc = i + FACES[f, 0], j + FACES[f, 1], k + FACES[f, 2]
                if not (0 <= a < sx and 0 <= b < sy and 0 <= cc < sz):
                    continue
                if ca[a, b, cc] != c:
                    continue
                q = (a * sy + b) * sz + cc
                x, y, z = kx0 + a, ky0 + b, kz0 + cc
                v = B[x, y, z]
                if rest[q]:
                    step = 0
                elif v == label:
                    e = ce[x - ex0, y - ey0, z - ez0]
                    if not joined[e]:
                        parent[q] = p
                        found = q
                        break
                    step = 0
                elif v == 0:
                    step = 1
                else:
                    step = OTHER  # taken by another cell: only if needed
                nd = d + step
                if nd < dist[q]:
                    dist[q] = nd
                    parent[q] = p
                    heapq.heappush(heap, (nd, q))
        if found < 0:
            return 1
        i, j, k = found // (sy * sz), (found // sz) % sy, found % sz
        joined[ce[kx0 + i - ex0, ky0 + j - ey0, kz0 + k - ez0]] = True
        p = parent[found]
        while p >= 0:
            i, j, k = p // (sy * sz), (p // sz) % sy, p % sz
            v = B[kx0 + i, ky0 + j, kz0 + k]
            if v != label:
                rest[p] = True
                if v != 0:
                    took = 2
            p = parent[p]
    # restore the paths, thickened by t through voxels the operation removed
    # from the cell (within E)
    ex1, ey1, ez1 = E[1], E[3], E[5]
    tx, ty, tz = ex1 - ex0, ey1 - ey0, ez1 - ez0
    seen = np.zeros((tx, ty, tz), np.bool_)
    front = np.empty(tx * ty * tz, np.int64)
    nfront = 0
    for p in range(n):
        if rest[p]:
            i, j, k = p // (sy * sz), (p // sz) % sy, p % sz
            x, y, z = kx0 + i, ky0 + j, kz0 + k
            out[x, y, z] = label
            seen[x - ex0, y - ey0, z - ez0] = True
            front[nfront] = ((x - ex0) * ty + (y - ey0)) * tz + (z - ez0)
            nfront += 1
    start = 0
    for _ in range(t):
        end = nfront
        for m in range(start, end):
            p = front[m]
            a, b, cc = p // (ty * tz), (p // tz) % ty, p % tz
            for f in range(6):
                aa, bb, c2 = a + FACES[f, 0], b + FACES[f, 1], cc + FACES[f, 2]
                if not (0 <= aa < tx and 0 <= bb < ty and 0 <= c2 < tz):
                    continue
                if seen[aa, bb, c2]:
                    continue
                x, y, z = ex0 + aa, ey0 + bb, ez0 + c2
                if A[x, y, z] != label or B[x, y, z] != 0:
                    continue
                seen[aa, bb, c2] = True
                out[x, y, z] = label
                front[nfront] = (aa * ty + bb) * tz + c2
                nfront += 1
        start = end
    return took


@numba.njit(parallel=True, cache=True)
def _reconnect(A, B, out, g, h, s, t, unresolved, repaired, took, r, q, sub):
    nx, ny, nz = A.shape
    # tiles on the grid of the whole image: r = origin modulo g
    tx = (nx + r[0] + g - 1) // g
    ty = (ny + r[1] + g - 1) // g
    tz = (nz + r[2] + g - 1) // g
    for tile in numba.prange(tx * ty * tz):
        ix, iy, iz = tile // (ty * tz), (tile // tz) % ty, tile % tz
        if ((ix + q[0]) % 3) * 9 + ((iy + q[1]) % 3) * 3 + (iz + q[2]) % 3 != sub:
            continue  # another subfield (tile indices of the whole image mod 3)
        K = (
            max(ix * g - r[0] - h, 0),
            min((ix + 1) * g - r[0] + h, nx),
            max(iy * g - r[1] - h, 0),
            min((iy + 1) * g - r[1] + h, ny),
            max(iz * g - r[2] - h, 0),
            min((iz + 1) * g - r[2] + h, nz),
        )
        kx0, kx1, ky0, ky1, kz0, kz1 = K
        lost = False
        for x in range(kx0, kx1):
            for y in range(ky0, ky1):
                for z in range(kz0, kz1):
                    if A[x, y, z] != 0 and B[x, y, z] != A[x, y, z]:
                        lost = True
                        break
                if lost:
                    break
            if lost:
                break
        if not lost:
            continue  # nothing taken from a cell: nothing cut
        # pieces of a component of labels that fall apart in target within K
        ca, na, _, _ = _label(A, kx0, kx1, ky0, ky1, kz0, kz1)
        cb, _, _, _ = _label(B, kx0, kx1, ky0, ky1, kz0, kz1)
        first = np.zeros(na + 1, np.int32)
        cand = np.zeros(na + 1, np.bool_)
        lab = np.zeros(na + 1, A.dtype)
        ncand = 0
        for i in range(kx1 - kx0):
            for j in range(ky1 - ky0):
                for k in range(kz1 - kz0):
                    c = ca[i, j, k]
                    if c == 0 or B[kx0 + i, ky0 + j, kz0 + k] != A[kx0 + i, ky0 + j, kz0 + k]:
                        continue
                    e = cb[i, j, k]
                    if first[c] == 0:
                        first[c] = e
                        lab[c] = A[kx0 + i, ky0 + j, kz0 + k]
                    elif first[c] != e and not cand[c]:
                        cand[c] = True
                        ncand += 1
        if ncand == 0:
            continue
        # are they connected in target within K extended by s?
        E = (
            max(kx0 - s, 0),
            min(kx1 + s, nx),
            max(ky0 - s, 0),
            min(ky1 + s, ny),
            max(kz0 - s, 0),
            min(kz1 + s, nz),
        )
        ce, _, esize, _ = _label(B, E[0], E[1], E[2], E[3], E[4], E[5])
        for c in range(1, na + 1):
            if cand[c]:
                code = _repair(A, B, out, ca, c, lab[c], ce, esize, K, E, t)
                unresolved[tile] += code & 1
                took[tile] += code >> 1
                repaired[tile] += 1


def reconnect(
    labels,
    target,
    s=20,
    t=2,
    max_passes=8,
    origin=(0, 0, 0),
    return_stats=False,
):
    """
    The result target of an operation on labels, with the cuts it made through
    cells joined again, up to the length scale s (see the module docstring).

    Args:
        labels (np.ndarray): The input 3D labeled image (0 = background).
        target (np.ndarray): The result of an operation on labels.
        s (int): Length scale in voxels (default 20): cuts are found and joined within
            tiles of 3s voxels, and loops smaller than about s may stay cut.
        t (int): A restored connection is thickened by t voxels through voxels
            the operation removed from the cell.
        max_passes (int): A path through voxels another cell took over gives
            them back to their cell in labels, which may cut that cell; then
            another pass follows (every pass moves voxels back to labels, so
            this ends).
        origin (tuple): Position of labels[0, 0, 0] in the whole image, when
            labels is a block of it: the tiles lie on the grid of the whole
            image, so that a block with a margin gets the same result.
        return_stats (bool): Also return the number of passes, of candidate
            cuts checked, of cuts joined through another cell's voxels, and of
            cuts that could not be joined.

    Returns:
        np.ndarray: target with the restored connections.
    """
    A = np.ascontiguousarray(labels)
    B = np.ascontiguousarray(target, dtype=A.dtype)
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch: {A.shape} vs {B.shape}")
    s, t = int(s), int(t)
    if s < 2 or t < 0:
        raise ValueError("need s >= 2 and t >= 0")
    g, h = 2 * s, s // 2
    r = np.asarray(origin, np.int64) % g
    q = (np.asarray(origin, np.int64) - r) // g  # index of the first tile
    ntiles = int(np.prod([(n + rr + g - 1) // g for n, rr in zip(A.shape, r)]))
    stats = {"passes": 0, "checked": 0, "through_other": 0, "unresolved": 0}
    for _ in range(max_passes):
        unresolved = np.zeros(ntiles, np.int64)
        repaired = np.zeros(ntiles, np.int64)
        took = np.zeros(ntiles, np.int64)
        for sub in range(27):
            out = B.copy()
            _reconnect(A, B, out, g, h, s, t, unresolved, repaired, took, r, q, sub)
            B = out
        stats["passes"] += 1
        stats["checked"] += int(repaired.sum())
        stats["through_other"] += int(took.sum())
        stats["unresolved"] = int(unresolved.sum())
        if not took.any():
            break  # no cell lost voxels to a path: nothing new is cut
    return (out, stats) if return_stats else out
