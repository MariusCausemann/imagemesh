import numpy as np
import pyvista as pv
import vtk

from .facets import _tet_face_topology


def convert_to_quadratic(tet_mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """Converts a linear tetrahedral mesh to a quadratic tetrahedral mesh."""
    filter_quad = vtk.vtkLinearToQuadraticCellsFilter()
    filter_quad.SetInputData(tet_mesh)
    filter_quad.Update()
    return pv.wrap(filter_quad.GetOutput())


def eval_shape_gradients(xi, eta, zeta):
    """
    Evaluates the derivatives of the 10-node tetrahedron shape functions
    with respect to the reference coordinates (xi, eta, zeta).
    Returns a (10, 3) numpy array.
    """
    L0 = 1.0 - xi - eta - zeta

    # dN / dxi
    dN_dxi = [
        1 - 4 * L0,
        4 * xi - 1,
        0,
        0,
        4 * (L0 - xi),
        4 * eta,
        -4 * eta,
        -4 * zeta,
        4 * zeta,
        0,
    ]

    # dN / deta
    dN_deta = [
        1 - 4 * L0,
        0,
        4 * eta - 1,
        0,
        -4 * xi,
        4 * xi,
        4 * (L0 - eta),
        -4 * zeta,
        0,
        4 * zeta,
    ]

    # dN / dzeta
    dN_dzeta = [
        1 - 4 * L0,
        0,
        0,
        4 * zeta - 1,
        -4 * xi,
        0,
        -4 * eta,
        4 * (L0 - zeta),
        4 * xi,
        4 * eta,
    ]

    return np.column_stack([dN_dxi, dN_deta, dN_dzeta])


# Reference points where the Jacobian is sampled: 4 vertices, 6 edge midpoints and the
# centre. detJ of a P2 tet is cubic, and its minimum often sits at an edge midpoint, so
# sampling only the vertices and the centre misses folds.
QUALITY_POINTS = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.5, 0.0, 0.0),
    (0.5, 0.5, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, 0.0, 0.5),
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5),
    (0.25, 0.25, 0.25),
]

# VTK quadratic tet: mid-edge node -> its two corner nodes
MIDSIDE_EDGES = [(4, 0, 1), (5, 1, 2), (6, 0, 2), (7, 0, 3), (8, 1, 3), (9, 2, 3)]


def compute_quadratic_quality(mesh):
    """
    Computes the minimum exact Jacobian determinant for every 2nd-order
    tetrahedron in a PyVista mesh.

    Returns: A 1D numpy array of the minimum Jacobian determinant per cell.
    """
    # 24 is the VTK cell type code for VTK_QUADRATIC_TETRA
    if 24 not in mesh.cells_dict:
        raise ValueError("No quadratic tetrahedra (type 24) found in the mesh!")

    # Get the physical (x, y, z) coordinates for all 10 nodes of every cell
    # Shape of tet_points: (N_cells, 10, 3)
    tet_nodes = mesh.cells_dict[24]
    tet_points = mesh.points[tet_nodes]

    integration_points = QUALITY_POINTS

    N_cells = len(tet_nodes)
    min_jacobians = np.full(N_cells, np.inf)

    for xi, eta, zeta in integration_points:
        # Get shape function gradients: Shape (10, 3)
        dNdX = eval_shape_gradients(xi, eta, zeta)
        J = np.einsum("nik,ij->nkj", tet_points, dNdX)

        # Unpack the 3 local basis vectors: Shape (N_cells, 3)
        v1, v2, v3 = J[:, :, 0], J[:, :, 1], J[:, :, 2]

        # 1. Determinant via Scalar Triple Product: det(J) = v1 · (v2 × v3)
        detJ = np.einsum("ni,ni->n", v1, np.cross(v2, v3))

        # 2. Base edge lengths (norms of v1, v2, v3 computed all at once)
        l0, l2, l3 = np.linalg.norm(J, axis=1).T

        # 3. Cross-edge lengths
        l1 = np.linalg.norm(v2 - v1, axis=1)
        l4 = np.linalg.norm(v3 - v1, axis=1)
        l5 = np.linalg.norm(v3 - v2, axis=1)

        # 4. Max edge-length product among the 4 corners
        max_length_product = np.max(
            [l0 * l2 * l3, l0 * l1 * l4, l1 * l2 * l5, l3 * l4 * l5], axis=0
        )

        # 5. Compute scaled Jacobian and update minimums in-place
        scaled_J = (np.sqrt(2.0) * detJ) / (max_length_product + 1e-14)
        np.minimum(min_jacobians, scaled_J, out=min_jacobians)

    return min_jacobians


def _pair_codes(lo, hi):
    """Encode unordered label pairs as int64 keys."""
    lo, hi = np.asarray(lo, dtype=np.int64), np.asarray(hi, dtype=np.int64)
    return (np.minimum(lo, hi) << 32) | np.maximum(lo, hi)


def boundary_facets(
    quad_mesh: pv.UnstructuredGrid, label_array: str = "marker", ignore_above: int | None = None
):
    """6-node boundary facets of a quadratic tet mesh and their label pairs.

    Returns ``(faces, codes)``: ``faces`` (F, 6) holds the external facets followed by
    the interfaces between cells of different ``label_array`` value (interfaces between
    two labels above ``ignore_above`` are skipped, as in ``mark_interface_facets``).
    ``codes`` encodes each facet's
    ``(region, region)`` pair (``0`` for the outside) with ``_pair_codes``, or is None
    when the mesh carries no ``label_array``.
    """
    topo = _tet_face_topology(quad_mesh)
    if label_array not in quad_mesh.cell_data:
        return topo["boundary_faces"], None
    markers = np.asarray(quad_mesh.cell_data[label_array]).astype(np.int64)
    a = markers[topo["interface_parents_a"]]
    b = markers[topo["interface_parents_b"]]
    keep = a != b
    if ignore_above is not None:
        keep &= np.minimum(a, b) <= ignore_above
    faces = np.vstack([topo["boundary_faces"], topo["interface_faces"][keep]])
    ext = markers[topo["boundary_parents"]]
    codes = np.concatenate([_pair_codes(np.zeros_like(ext), ext), _pair_codes(a[keep], b[keep])])
    return faces, codes


def project_to_labelled_sheets(
    points, faces, codes, target_surface, target_label_array="boundary_labels"
):
    """Closest point on the target for every node of ``faces``.

    Each node is projected only onto the target sheets whose ``target_label_array``
    pair matches the label pair of one of its facets; a node on several sheets (a
    junction) goes to the nearest of them. Nodes with no matching sheet keep their
    position and are flagged in ``found``. Without labels on either side, every node is
    projected onto the whole target.

    Returns ``(node_ids, closest, found)``.
    """
    node_ids = np.unique(faces)
    if codes is None or target_label_array not in target_surface.cell_data:
        _, closest = target_surface.find_closest_cell(points[node_ids], return_closest_point=True)
        return node_ids, closest, np.ones(len(node_ids), dtype=bool)

    labels = np.asarray(target_surface.cell_data[target_label_array])
    target_codes = _pair_codes(labels[:, 0], labels[:, 1])

    # unique (node, code) memberships
    pairs = np.unique(np.column_stack([faces.ravel(), np.repeat(codes, faces.shape[1])]), axis=0)
    pair_node = np.searchsorted(node_ids, pairs[:, 0])

    closest = points[node_ids].copy()
    best = np.full(len(node_ids), np.inf)
    order = np.argsort(target_codes, kind="stable")
    sorted_codes = target_codes[order]
    missing = []
    for code in np.unique(pairs[:, 1]):
        lo, hi = np.searchsorted(sorted_codes, [code, code + 1])
        sel = pair_node[pairs[:, 1] == code]
        if lo == hi:
            missing.append((int(code >> 32), int(code & 0xFFFFFFFF), len(sel)))
            continue
        sheet = target_surface.extract_cells(order[lo:hi])
        _, cp = sheet.find_closest_cell(points[node_ids[sel]], return_closest_point=True)
        dist = np.linalg.norm(cp - points[node_ids[sel]], axis=1)
        better = dist < best[sel]
        best[sel[better]] = dist[better]
        closest[sel[better]] = cp[better]

    found = np.isfinite(best)
    if missing:
        print(
            f"  -> {int((~found).sum())} boundary nodes have no matching target sheet and"
            f" stay put; missing label pairs (lo, hi, nodes): {missing}"
        )
    return node_ids, closest, found


def adaptive_snap_boundaries(
    quad_mesh: pv.UnstructuredGrid,
    target_surface: pv.PolyData,
    label_array: str = "marker",
    target_label_array: str = "boundary_labels",
    only_high_order: bool = False,
    floor_factor: float = 0.5,
    abs_floor: float = 0.1,
    n_steps: int = 10,
    max_steps: int | None = None,
    smooth_sweeps: int = 3,
    max_halvings: int = 3,
    relax_iters: int = 5,
    ignore_above: int | None = None,
):
    """Moves boundary nodes onto the target surface step by step, never breaking a cell.

    Every cell gets a quality floor ``min(q0, max(floor_factor * q0, abs_floor))``, where
    ``q0`` is its quality before snapping. In each step every boundary node that has not
    reached the target tries to advance by ``1 / n_steps`` of its displacement; a node
    whose advance would push an adjacent cell below its floor halves its increment, up to
    ``max_halvings`` times, and otherwise stays put and retries in the next step. After
    each step the interior displacement is smoothed so that interior nodes follow the
    boundary, and the free nodes of the cells that blocked a node are relaxed by a
    pattern search to make room for the next step. Stepping stops once no node can advance or after
    ``max_steps`` (default ``2 * n_steps``) steps. With ``only_high_order``, corner nodes
    never move, so only the mid-edge nodes curve. Interfaces between two labels above
    ``ignore_above`` are not snapped (see ``boundary_facets``).

    Returns the ids of all boundary nodes.
    """
    from .mesh_optimizer import (
        build_node_to_cell_map,
        cell_qualities,
        cells_of_nodes,
        quality_shape_grads,
        relax_nodes,
        smooth_interior_displacement,
    )

    print("Extracting boundary nodes...")
    faces, codes = boundary_facets(quad_mesh, label_array, ignore_above)
    if len(faces) == 0:
        return np.array([], dtype=int)

    cells = np.ascontiguousarray(quad_mesh.cells.reshape(-1, 11)[:, 1:], dtype=np.int64)
    n_cells, n_points = len(cells), quad_mesh.n_points
    points = np.asarray(quad_mesh.points, dtype=np.float64).copy()
    ref_points = points.copy()

    all_boundary_ids, closest, found = project_to_labelled_sheets(
        points, faces, codes, target_surface, target_label_array
    )
    move = found.copy()
    if only_high_order:
        move &= ~np.isin(all_boundary_ids, cells[:, :4])
    ids = all_boundary_ids[move]
    P0, D = points[ids], closest[move] - points[ids]

    grads = quality_shape_grads()
    offsets, data = build_node_to_cell_map(n_points, cells)
    q0 = cell_qualities(np.arange(n_cells), cells, points, grads)
    floor = np.minimum(q0, np.maximum(floor_factor * q0, abs_floor)) - 1e-9

    # Local corner-edge length, for the smoothing tolerance and the relaxation step
    corners = points[cells[:, :4]]
    h_cell = np.mean(
        [
            np.linalg.norm(corners[:, a] - corners[:, b], axis=1)
            for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        ],
        axis=0,
    )
    h_node = np.full(n_points, np.inf)
    for j in range(10):
        np.minimum.at(h_node, cells[:, j], h_cell)
    tol = 1e-3 * h_node
    is_free = np.ones(n_points, dtype=bool)
    is_free[all_boundary_ids] = False
    if only_high_order:
        is_free[cells[:, :4]] = False
    free_ids = np.nonzero(is_free)[0]

    alpha = np.zeros(len(ids))
    max_steps = 2 * n_steps if max_steps is None else max_steps
    for step in range(max_steps):
        cand = np.nonzero(alpha < 1.0)[0]
        if len(cand) == 0:
            break
        inc = np.minimum(1.0 / n_steps, 1.0 - alpha[cand])
        min_inc = inc * 0.5**max_halvings
        points[ids[cand]] = P0[cand] + (alpha[cand] + inc)[:, None] * D[cand]

        # Halve the increment of every advancing node of a cell that fell below its
        # floor; after max_halvings the node stays put. The state before the step
        # satisfied all floors, so this terminates.
        moving = np.ones(len(cand), dtype=bool)
        blocked = np.zeros(n_cells, dtype=bool)
        while moving.any():
            check = cells_of_nodes(ids[cand[moving]], offsets, data, n_cells)
            q = cell_qualities(check, cells, points, grads)
            bad = check[q < floor[check]]
            if len(bad) == 0:
                break
            blocked[bad] = True
            in_bad = np.zeros(n_points, dtype=bool)
            in_bad[cells[bad]] = True
            hold = moving & in_bad[ids[cand]]
            inc[hold] *= 0.5
            give_up = hold & (inc < min_inc * (1 - 1e-9))
            inc[give_up] = 0.0
            moving &= ~give_up
            h = cand[hold]
            points[ids[h]] = P0[h] + (alpha[h] + inc[hold])[:, None] * D[h]

        alpha[cand] += inc
        alpha[alpha > 1.0 - 1e-9] = 1.0
        n_moves = smooth_interior_displacement(
            points, ref_points, cells, free_ids, offsets, data, grads, floor, tol, smooth_sweeps
        )
        relax_ids = np.unique(cells[blocked])
        relax_ids = relax_ids[is_free[relax_ids]]
        n_relax = relax_nodes(
            points, cells, relax_ids, offsets, data, grads, floor, 0.05 * h_node, relax_iters
        )
        print(
            f"  -> Step {step + 1}: {int(moving.sum())} nodes advanced "
            f"({int((moving & (inc < 1.0 / n_steps - 1e-12)).sum())} partially), "
            f"{int((alpha == 1.0).sum())}/{len(ids)} fully snapped, "
            f"{n_moves} smoothing / {n_relax} relaxation moves."
        )
        if not moving.any():
            break

    quad_mesh.points = points

    print("\n  -> Final Alpha Distribution:")
    bins = [0.0, 0.25, 0.5, 0.75, 1.0 - 1e-9, 1.0 + 1e-9]
    counts, _ = np.histogram(alpha, bins=bins)
    names = ["[0, 0.25)", "[0.25, 0.5)", "[0.5, 0.75)", "[0.75, 1)", "1 (snapped)"]
    for name, count in zip(names[::-1], counts[::-1]):
        print(f"       Alpha {name:>11s}: {count:7d} nodes ({100 * count / len(ids):5.2f}%)")
    print("-" * 50)
    return all_boundary_ids


def straighten_inverted_cells(
    quad_mesh: pv.UnstructuredGrid, min_quality: float = 0.0, max_rounds: int = 10
):
    """Last-resort repair: reset the mid-edge nodes of every cell below `min_quality`
    to the midpoints of its corner edges (a straight-sided, constant-detJ cell).

    Mid-edge nodes are shared between cells, so a reset can degrade a neighbour; this
    repeats until no cell is below `min_quality` or `max_rounds` is reached. Returns
    the number of cells still below `min_quality`.
    """
    cells = quad_mesh.cells.reshape(-1, 11)[:, 1:]
    points = quad_mesh.points.copy()
    n_bad = 0
    for _ in range(max_rounds):
        q = compute_quadratic_quality(quad_mesh)
        bad = np.where(q < min_quality)[0]
        n_bad = len(bad)
        if n_bad == 0:
            break
        for m, a, b in MIDSIDE_EDGES:
            points[cells[bad, m]] = 0.5 * (points[cells[bad, a]] + points[cells[bad, b]])
        quad_mesh.points = points
        print(f"  -> Straightened mid-edge nodes of {n_bad} cells below quality {min_quality}.")
    else:
        n_bad = int((compute_quadratic_quality(quad_mesh) < min_quality).sum())
    return n_bad


def print_quality_stats(mesh: pv.UnstructuredGrid, mesh_name: str):
    """Computes and prints the Scaled Jacobian quality of the mesh."""

    if mesh.celltypes[0] == vtk.VTK_QUADRATIC_TETRA:
        q_arr = compute_quadratic_quality(mesh)

    else:
        mesh_with_quality = mesh.cell_quality(quality_measure="scaled_jacobian")
        q_arr = mesh_with_quality.cell_data["scaled_jacobian"]

    print(f"--- {mesh_name} Cell Quality (Scaled Jacobian) ---")
    print(f"  Min:  {q_arr.min():.4f} (values <= 0 indicate inverted cells)")
    print(f"  Mean: {q_arr.mean():.4f}")
    print(f"  Max:  {q_arr.max():.4f}\n")
    return q_arr
