"""
Removal of handles from label images: a cell with a handle (a tunnel through it,
genus > 0) gets its tunnels spanned by membranes, or, where a membrane would create
new handles, the handle cut.

1. Handle check: the Euler characteristic chi of every label
   (nbmorph.euler_characteristic) and its number b0 of 26-connected components
   (cc3d); chi < b0 means the cell has handles.
2. Windows grow from core min_window (halo core / 2) by factors of two until they
   contain every loop up to max_loop_length (a closed curve of length L lies within
   L / 4 of its bounding box centre). In each window the check is repeated, and
   only the cells that fail it are skeletonized, cropped to the window. The minimum
   cycle basis of the skeleton graph gives one short loop around each tunnel,
   running through the cell; a loop is handled by the window whose core contains
   its centre. Small handles are filled at the small scales, so the large windows
   only see the handles that are left.
3. Each loop is spanned by a harmonic membrane (Laplace solve on a polar grid with
   the loop as fixed boundary), voxelized with thickness N and assigned to the cell
   (overwriting the ECS and other cells) if that raises the cell's Euler
   characteristic. A local closing (fillet) blends the membrane into the cell, the
   cells it now touches are carved back (nbmorph.separate_labels_box, the filled
   cell has priority) so that cells stay separated, and pockets enclosed by the
   membranes are filled. If the membrane would create new handles (the loop is
   linked with another part of the cell), the handle is cut instead: the smallest
   cross-section of the cell normal to the loop that removes it is set to 0.

Handles with loops longer than max_loop_length are left. With workers > 1, the
windows are processed in a forkserver pool (numba's OpenMP layer is not fork-safe),
with one numba thread per worker.
"""

import itertools
import multiprocessing
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor

import cc3d
import fastremap
import nbmorph
import networkx as nx
import numba
import numpy as np
import scipy.sparse as sp
from scipy import ndimage as ndi
from scipy.sparse.linalg import splu
from skimage.morphology import skeletonize


def euler_per_label(img, nlabels):
    """Euler characteristic of every label (26-connected)."""
    return nbmorph.euler_characteristic(img, nlabels)


def components_per_label(img, nlabels):
    """Number of 26-connected components of every label."""
    cc = cc3d.connected_components(img, connectivity=26)
    b0 = np.zeros(nlabels + 1, np.int64)
    labs = np.fromiter(fastremap.component_map(cc, img).values(), np.int64)
    np.add.at(b0, labs, 1)
    b0[0] = 0
    return b0


def handle_counts(img, nlabels):
    """b0 - chi = b1 - b2 of every label: its number of handles if it has no cavities."""
    return components_per_label(img, nlabels) - euler_per_label(img, nlabels)


def labels_with_handles(img, nlabels):
    """Labels with chi < b0, i.e. b1 > b2 >= 0 (cells whose handles are cancelled by
    as many cavities are missed, which is rare)."""
    return np.nonzero(handle_counts(img, nlabels) > 0)[0]


def _pad(sl, shape, pad):
    return tuple(slice(max(s.start - pad, 0), min(s.stop + pad, n)) for s, n in zip(sl, shape))


def skeleton_loops(mask, min_nodes=8):
    """Ordered voxel loops (k x 3 arrays) of the minimum cycle basis of the skeleton."""
    skel = skeletonize(mask)
    pts = np.argwhere(skel)
    index = -np.ones(mask.shape, dtype=np.int64)
    index[tuple(pts.T)] = np.arange(len(pts))
    G = nx.Graph()
    G.add_nodes_from(range(len(pts)))
    offsets = [o for o in np.ndindex(3, 3, 3) if o > (1, 1, 1)]  # half of the 26-neighbourhood
    for o in offsets:
        o = np.array(o) - 1
        nb = pts + o
        ok = np.all((nb >= 0) & (nb < mask.shape), axis=1)
        j = np.full(len(pts), -1)
        j[ok] = index[tuple(nb[ok].T)]
        i = np.nonzero(j >= 0)[0]
        G.add_weighted_edges_from(
            zip(i.tolist(), j[i].tolist(), [float(np.linalg.norm(o))] * len(i))
        )
    # only the part of the skeleton graph that carries cycles
    core = nx.k_core(G, 2)
    # merge clusters of adjacent junction voxels into one node; they only add tiny cycles
    rep = {n: n for n in core}
    for cluster in nx.connected_components(core.subgraph(n for n, k in core.degree if k > 2)):
        r = min(cluster)
        rep.update(dict.fromkeys(cluster, r))
    Q = nx.Graph()
    for u, v, w in core.edges(data="weight"):
        a, b = rep[u], rep[v]
        if a != b and (not Q.has_edge(a, b) or Q[a][b]["weight"] > w):
            Q.add_edge(a, b, weight=w)
    core = nx.k_core(Q, 2)
    loops = []
    for comp in nx.biconnected_components(core):
        if len(comp) < min_nodes:
            continue
        H, chains = _contract_chains(core.subgraph(comp))
        for cyc in nx.minimum_cycle_basis(H, weight="weight"):
            order = _order_cycle(H.subgraph(cyc))
            nodes = []
            for a, b in zip(order, order[1:] + order[:1]):
                path = chains[frozenset((a, b))]
                nodes += (path if path[0] == a else path[::-1])[:-1]
            if len(nodes) >= min_nodes:
                loops.append(pts[nodes])
    return loops


def _contract_chains(G):
    """Replace chains of degree-2 nodes by single weighted edges; returns the graph and
    edge -> node path."""
    keep = {n for n, d in G.degree if d != 2} or {next(iter(G))}
    H, chains, seen = nx.Graph(), {}, set()

    def add(path, w):
        u, v = path[0], path[-1]
        if len(path) == 2 and H.has_edge(u, v):  # direct edge: split the existing chain instead
            old, w_old = chains.pop(frozenset((u, v))), H[u][v]["weight"]
            H.remove_edge(u, v)
            add(path, w)
            path, w = old, w_old
        if u == v or H.has_edge(u, v):  # split to keep H simple
            cuts = [len(path) // 3, 2 * len(path) // 3] if u == v else [len(path) // 2]
            for a, b in zip([0] + cuts, cuts + [len(path) - 1]):
                seg = path[a : b + 1]
                add(seg, sum(G[x][y]["weight"] for x, y in zip(seg, seg[1:])))
            return
        H.add_edge(u, v, weight=w)
        chains[frozenset((u, v))] = path

    for u in keep:
        for v in G[u]:
            if (u, v) in seen:
                continue
            path = [u, v]
            while path[-1] not in keep:
                a, b = path[-2], path[-1]
                path.append(next(x for x in G[b] if x != a))
            seen.update({(path[0], path[1]), (path[-1], path[-2])})
            add(path, sum(G[x][y]["weight"] for x, y in zip(path, path[1:])))
    return H, chains


def _order_cycle(H):
    """Node order of a cycle through all nodes of H (backtracking; cycles here are short)."""
    nodes = list(H.nodes)
    if all(d == 2 for _, d in H.degree):
        return [u for u, _ in nx.find_cycle(H, nodes[0])]
    start = min(nodes, key=H.degree)
    path, onpath = [start], {start}

    def extend():
        if len(path) == len(nodes):
            return start in H[path[-1]]
        for v in sorted(H[path[-1]], key=H.degree):
            if v not in onpath:
                path.append(v), onpath.add(v)
                if extend():
                    return True
                path.pop(), onpath.discard(v)
        return False

    if not extend():
        raise RuntimeError(f"no cycle through the {len(nodes)} nodes")
    return path


def span_loop(loop, kmax=128):
    """Harmonic membrane spanning a closed loop; returns grid points (R+1, K, 3), row R = loop."""
    closed = np.vstack([loop, loop[:1]]).astype(float)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    K = int(np.clip(np.ceil(s[-1] / 2), 24, kmax))
    t = np.linspace(0, s[-1], K, endpoint=False)
    bnd = np.stack([np.interp(t, s, closed[:, d]) for d in range(3)], axis=1)
    R = max(3, int(np.ceil(K / (2 * np.pi))))
    # unknowns: centre (0) and rings r = 1..R-1, angle k -> 1 + (r-1)*K + k
    r, k = np.meshgrid(np.arange(1, R), np.arange(K), indexing="ij")
    r, k = r.ravel(), k.ravel()
    idx = 1 + (r - 1) * K + k
    rows = [np.zeros(K + 1, int), idx, idx, idx, idx]
    cols = [
        np.r_[0, 1 + np.arange(K)],
        1 + (r - 1) * K + (k - 1) % K,
        1 + (r - 1) * K + (k + 1) % K,
        np.where(r == 1, 0, idx - K),
        idx,
    ]
    vals = [
        np.r_[K, -np.ones(K)],
        -np.ones(len(idx)),
        -np.ones(len(idx)),
        -np.ones(len(idx)),
        np.full(len(idx), 4.0),
    ]
    inner = r < R - 1  # outer neighbour is an unknown, otherwise it is on the loop
    rows.append(idx[inner]), cols.append(idx[inner] + K), vals.append(-np.ones(inner.sum()))
    n = 1 + (R - 1) * K
    A = sp.csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    )
    rhs = np.zeros((n, 3))
    rhs[idx[~inner]] = bnd[k[~inner]]
    x = splu(A).solve(rhs)
    grid = np.empty((R + 1, K, 3))
    grid[0] = x[0]
    grid[1:R] = x[1:].reshape(R - 1, K, 3)
    grid[R] = bnd
    return grid


def voxelize_membrane(grid, shape, thickness):
    """Voxels of the membrane, thickened to ~thickness, as (slice into shape, local mask)."""
    # resample the grid so that neighbouring samples are < 0.5 voxel apart
    gc = np.concatenate([grid, grid[:, :1]], axis=1)
    step = max(
        np.linalg.norm(np.diff(gc, axis=0), axis=2).max(),
        np.linalg.norm(np.diff(gc, axis=1), axis=2).max(),
    )
    f = int(np.ceil(step / 0.5))
    R, K = gc.shape[0] - 1, gc.shape[1] - 1
    ur, uk = np.linspace(0, R, R * f + 1), np.linspace(0, K, K * f + 1)
    coords = np.meshgrid(ur, uk, indexing="ij")
    pts = np.stack(
        [ndi.map_coordinates(gc[..., d], coords, order=1) for d in range(3)], axis=-1
    ).reshape(-1, 3)
    radius = (thickness - 1) // 2
    h = radius + 1
    lo = np.maximum(np.floor(pts.min(0)).astype(int) - h, 0)
    hi = np.minimum(np.ceil(pts.max(0)).astype(int) + h + 1, shape)
    surf = np.zeros(hi - lo, dtype=np.uint8)
    v = np.clip(np.rint(pts).astype(int) - lo, 0, hi - lo - 1)
    surf[tuple(v.T)] = 1
    if radius:
        surf = nbmorph.dilate_labels_spherical(surf, radius=radius, struct_sequence="B")
    return tuple(slice(a, b) for a, b in zip(lo, hi)), surf > 0


def _blocks(shape, core, halo):
    for start in itertools.product(*(range(0, n, core) for n in shape)):
        core_sl = tuple(slice(s, min(s + core, n)) for s, n in zip(start, shape))
        yield core_sl, _pad(core_sl, shape, halo)


def _block_loops(blk, core_sl, sl, labels, nlabels, max_loop_length):
    """Loops (global voxel coordinates) of the cells with handles in block blk = img[sl],
    with centre in its core."""
    t = time.time()
    present = np.intersect1d(fastremap.unique(blk), labels)
    if len(present) == 0:
        return [], [], 0.0
    sub = fastremap.mask_except(blk, present.tolist())  # other cells do not matter here
    cand = labels_with_handles(sub, nlabels)
    off = np.array([s.start for s in sl])
    lo = np.array([s.start for s in core_sl])
    hi = np.array([s.stop for s in core_sl])
    loops, skipped = [], []
    for lab in cand:
        m = blk == lab
        bb = _pad(ndi.find_objects(m.view(np.uint8))[0], m.shape, 1)
        for loop in skeleton_loops(m[bb]):
            loop = loop + off + [s.start for s in bb]
            centre = (loop.min(0) + loop.max(0)) / 2
            if not np.all((centre >= lo) & (centre < hi)):
                continue
            length = np.linalg.norm(np.diff(np.vstack([loop, loop[:1]]), axis=0), axis=1).sum()
            if length > max_loop_length:
                skipped.append((int(lab), round(length)))
                continue
            loops.append((int(lab), loop))
    return loops, skipped, time.time() - t


def _delta_chi(m, add):
    """Change of the Euler characteristic of mask m when adding the voxels add; chi is a
    sum of local terms, so the change is exact on a crop around add with margin 2."""
    sl = _pad(ndi.find_objects(add.view(np.uint8))[0], m.shape, 2)
    m, add = m[sl], add[sl]
    return euler_per_label((m | add).view(np.uint8), 1)[1] - euler_per_label(m.view(np.uint8), 1)[1]


def _apply_membrane(img, lab, gsl, box):
    """Assign the membrane to cell lab if that raises its Euler characteristic. Returns
    the taken voxels per label, or the (non-positive) change of chi if rejected."""
    reg = _pad(gsl, img.shape, 2)
    local = img[reg]
    m = local == lab
    mem = np.zeros_like(m)
    mem[tuple(slice(g.start - r.start, g.stop - r.start) for g, r in zip(gsl, reg))] = box
    mem &= ~m
    if not mem.any():
        return 0
    dchi = _delta_chi(m, mem)
    if dchi <= 0:
        return dchi
    taken = Counter(local[mem].tolist())
    local[mem] = lab
    return taken


def _separate(img, lab, gsl, margin):
    """Carve the other cells back from cell lab around gsl, so that cells stay separated."""
    reg = _pad(gsl, img.shape, margin)
    local = img[reg]
    sep = nbmorph.separate_labels_box(local, priority=local == lab)
    n = int((sep != local).sum())
    local[...] = sep
    return n


def _cut_handle(img, lab, loop, width=1, pads=(20, 60), step=3):
    """
    Cut the handle that loop runs around: at every step-th loop point, the connected
    cross-section of cell lab in a slab (thickness 2 * width + 1) normal to the loop is
    a candidate; the smallest one that the loop crosses once (so the cell stays
    connected along the rest of the loop) and that raises the Euler characteristic is
    removed. Cross-sections must lie inside the crop around the loop (margin pad);
    the pads are tried in turn. Returns the slice around the cut, or None.
    """
    for pad in pads:
        sl = _cut_handle_crop(img, lab, loop, width, pad, step)
        if sl is not None:
            return sl
    return None


def _cut_handle_crop(img, lab, loop, width, pad, step):
    sl = _pad(tuple(slice(a, b + 1) for a, b in zip(loop.min(0), loop.max(0))), img.shape, pad)
    local = img[sl]
    m = local == lab
    p = loop - [s.start for s in sl]
    grid = np.moveaxis(np.indices(m.shape), 0, -1)
    chi0 = euler_per_label(m.view(np.uint8), 1)[1]
    best = None
    for i in range(0, len(p), step):
        tangent = (p[(i + 2) % len(p)] - p[i - 2]).astype(float)
        tangent /= np.linalg.norm(tangent)
        slab = m & (np.abs((grid - p[i]) @ tangent) <= width)
        comp = cc3d.connected_components(slab.view(np.uint8), connectivity=26)
        slab = comp == comp[tuple(p[i])]  # the cross-section around the loop point
        if best is not None and slab.sum() >= best.sum():
            continue
        if np.any(slab[[0, -1]]) or np.any(slab[:, [0, -1]]) or np.any(slab[:, :, [0, -1]]):
            continue  # reaches the crop boundary
        inside = slab[tuple(p.T)]
        if np.count_nonzero(inside != np.roll(inside, 1)) != 2:
            continue
        if euler_per_label((m & ~slab).view(np.uint8), 1)[1] > chi0:
            best = slab
    if best is None:
        return None
    local[best] = 0
    return sl


def _fillet(img, lab, gsl, box, radius):
    """Closing of cell lab near a membrane, so that it blends into the cell; only adds ECS."""
    reg = _pad(gsl, img.shape, 2 * radius + 2)
    local = img[reg]
    m = (local == lab).view(np.uint8)
    closed = nbmorph.close_labels_spherical(m, radius) > 0
    near = np.zeros_like(m)
    near[tuple(slice(g.start - r.start, g.stop - r.start) for g, r in zip(gsl, reg))] = box
    near = nbmorph.dilate_labels_spherical(near, radius + 1) > 0
    add = closed & near & (local == 0)
    local[add] = lab
    return int(add.sum())


def _fill_local_cavities(img, lab, sl):
    """Assign pockets of the complement of cell lab that are enclosed within sl."""
    local = img[sl]
    comp = cc3d.connected_components((local != lab).view(np.uint8), connectivity=6)
    faces = [comp[[0, -1]], comp[:, [0, -1]], comp[:, :, [0, -1]]]
    border = fastremap.unique(np.concatenate([f.ravel() for f in faces]))
    cav = (comp > 0) & ~np.isin(comp, border)
    local[cav] = lab
    return int(cav.sum())


def _overlap(a, b):
    return all(x.start < y.stop and y.start < x.stop for x, y in zip(a, b))


def _bounded_map(ex, fn, jobs, workers):
    """ex.map(fn, *zip(*jobs)) with at most 2 * workers jobs (window copies) in flight;
    serial without ex."""
    if ex is None:
        yield from (fn(*job) for job in jobs)
        return
    pending = deque()
    for job in jobs:
        pending.append(ex.submit(fn, *job))
        if len(pending) >= 2 * workers:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def _pass(ex, img, nlabels, fills, core, max_len, max_loop_length, regions, known, workers, kw):
    """One pass with windows of core size `core` (halo core / 2), restricted to windows
    overlapping `regions` (if given) and to cells with more handles than `known`
    (label -> number of loops known to be too long); fills the loops up to max_len.
    Returns the slices of the applied membranes and the labels of the handles left
    unresolved (too long, or the membrane would create handles), or None if no cell
    has handles."""
    t = time.time()
    halo = core // 2
    h = handle_counts(img, nlabels)
    labels = np.array([lab for lab in np.nonzero(h > 0)[0] if h[lab] > known.get(lab, 0)])
    if len(labels) == 0:
        return None
    wanted = set(labels.tolist())
    bboxes = [sl for lab, sl in enumerate(ndi.find_objects(img), 1) if lab in wanted]
    jobs = [
        (c, sl)
        for c, sl in _blocks(img.shape, core, halo)
        if any(_overlap(bb, sl) for bb in bboxes)
        and (regions is None or any(_overlap(g, sl) for g in regions))
    ]
    t_check = time.time() - t
    results = list(
        _bounded_map(
            ex,
            _block_loops,
            ((img[sl], c, sl, labels, nlabels, max_len) for c, sl in jobs),
            workers,
        )
    )
    loops = [lp for res in results for lp in res[0]]
    too_long = sorted(s for res in results for s in res[1] if s[1] > max_loop_length)
    t_loops = time.time() - t - t_check
    t = time.time()
    touched, rejected, cuts, unresolved = [], 0, 0, []
    for lab, loop in loops:
        grid = span_loop(loop)
        gsl, box = voxelize_membrane(grid, img.shape, kw["thickness"])
        taken = _apply_membrane(img, lab, gsl, box)
        if not isinstance(taken, Counter):
            if taken < 0:  # the membrane would create handles: cut the handle instead
                cut = _cut_handle(img, lab, loop)
                if cut is not None:
                    cuts += 1
                    touched.append(cut)
                    fills.append((lab, loop, None, Counter()))
                    continue
                unresolved.append(lab)
            rejected += 1
            continue
        if kw["fillet_radius"]:
            taken["fillet"] = _fillet(img, lab, gsl, box, kw["fillet_radius"])
        taken["separated"] = _separate(img, lab, gsl, kw["fillet_radius"] + 3)
        touched.append(gsl)
        fills.append((lab, loop, grid, taken))
    # overlapping membranes can enclose pockets; look for them around the membranes
    cavities = sum(
        _fill_local_cavities(img, lab, _pad(gsl, img.shape, 2 * kw["fillet_radius"] + 3))
        for (lab, *_), gsl in zip(fills[len(fills) - len(touched) :], touched)
    )
    print(
        f"window {core}+{halo}: {len(labels)} cells with handles ({t_check:.1f}s), "
        f"{len(jobs)} windows, {len(loops)} loops ({t_loops:.1f}s wall, "
        f"{sum(r[2] for r in results):.1f}s in workers), applied {len(touched) - cuts}, "
        f"cut {cuts}, rejected {rejected}, {cavities} enclosed voxels filled "
        f"({time.time() - t:.1f}s)" + (f", too long: {too_long}" if too_long else "")
    )
    return touched, [lab for lab, _ in too_long] + unresolved


def fill_handles(
    img,
    thickness=3,
    max_loop_length=400,
    fillet_radius=2,
    min_window=64,
    cleanup_rounds=2,
    workers=1,
    return_fills=False,
):
    """
    Remove the handles of all labels of img (0 is the background), see the module
    docstring. Returns a repaired copy (and, with return_fills, a list of
    (label, loop, membrane grid or None for a cut, voxels taken per label)).

    Args:
        thickness: membrane thickness in voxels.
        max_loop_length: handles whose loop is longer (in voxels) are left.
        fillet_radius: radius of the closing that blends a membrane into the cell.
        min_window: core size of the smallest windows.
        cleanup_rounds: passes removing small handles created by the last membranes.
        workers: number of processes for the windows.
    """
    img = img.copy()
    nlabels = int(img.max())
    cores = [min_window]
    while 4 * (cores[-1] // 2 - 4) < max_loop_length:
        cores.append(2 * cores[-1])
    kw = dict(thickness=thickness, fillet_radius=fillet_radius)
    fills = []
    ex = None
    if workers > 1:
        method = (
            "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
        )
        ex = ProcessPoolExecutor(
            workers,
            mp_context=multiprocessing.get_context(method),
            initializer=numba.set_num_threads,
            initargs=(1,),
        )

    def run(core, regions, known):
        max_len = min(max_loop_length, 4 * (core // 2 - 4))  # loops centred in the core fit
        return _pass(
            ex, img, nlabels, fills, core, max_len, max_loop_length, regions, known, workers, kw
        )

    try:
        res = None
        for core in cores:
            res = run(core, None, {})
            if res is None:
                break
        # membranes of earlier scales are covered by the next scale; clean up after the
        # last one, apart from the handles it left unresolved
        if res is not None:
            touched, known = res[0], Counter(res[1])
            for _ in range(cleanup_rounds):
                if not touched:
                    break
                res = run(cores[0], touched, known)
                if res is None:
                    break
                touched = res[0]
    finally:
        if ex is not None:
            ex.shutdown()
    return (img, fills) if return_fills else img
