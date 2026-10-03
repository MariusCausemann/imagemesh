"""
Facets of labelled tetrahedral meshes: interfaces between regions, outer
boundary facets and operations on facet (or cell) markers.
"""

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree


def smooth_cell_labels(surf, marker_name, target_labels, max_iter=20):
    """
    Smooths cell labels by enforcing that no triangle shares more than
    one edge with a different region, acting ONLY on the specified labels.
    """
    mesh = surf.copy()
    labels = mesh.cell_data[marker_name].copy()
    n_cells = mesh.n_cells

    # Convert to set for O(1) lookup speed inside the loop
    target_set = set(target_labels)

    # Pre-compute the edge neighbors
    neighbors_list = [mesh.cell_neighbors(i, "edges") for i in range(n_cells)]

    for iteration in range(max_iter):
        flipped = 0

        for i in range(n_cells):
            current_label = labels[i]

            # Constraint 1: Skip if the current cell isn't a target label
            if current_label not in target_set:
                continue

            neighbor_ids = neighbors_list[i]
            if len(neighbor_ids) == 0:
                continue

            neighbor_labels = labels[neighbor_ids]
            unique_vals, counts = np.unique(neighbor_labels, return_counts=True)
            majority_label = unique_vals[np.argmax(counts)]
            max_count = np.max(counts)

            # Constraint 2: Only allow flipping TO another target label
            if majority_label not in target_set:
                continue

            # The Magic Rule
            if majority_label != current_label and max_count >= 2:
                labels[i] = majority_label
                flipped += 1

        if flipped == 0:
            print(f"Selective smoothing converged in {iteration + 1} iterations.")
            break

    mesh.cell_data[marker_name] = labels
    return mesh


def mark_between_regions(ids_a, ids_b, da, db, seg, label_array):
    sega = seg.extract_cells(np.isin(seg.cell_data[label_array], ids_a))
    segb = seg.extract_cells(np.isin(seg.cell_data[label_array], ids_b))
    kd_tree_a = KDTree(sega.cell_centers().points)
    kd_tree_b = KDTree(segb.cell_centers().points)
    dista, _ = kd_tree_a.query(seg.cell_centers().points)
    distb, _ = kd_tree_b.query(seg.cell_centers().points)
    return np.logical_and(dista < da, distb < db)


def dilate_cell_marker(grid, marker, r=1):
    dil_marker = marker.copy()
    for _ in range(r):
        dil_marker = dilate_cell_marker_once(grid, dil_marker)
    return dil_marker


def dilate_cell_marker_once(grid, marker):
    all_cell_centers = grid.cell_centers().points
    dil_marker = marker.copy()
    for i in np.nonzero(marker == 0)[0]:
        neighbor_indices = grid.cell_neighbors(i, connections="points")
        neighbor_indices = [j for j in neighbor_indices if marker[j] > 0]
        if len(neighbor_indices) == 0:
            continue
        neighbor_markers = marker[neighbor_indices]
        neighbor_centers = all_cell_centers[neighbor_indices]
        neighbor_dist = np.linalg.norm(all_cell_centers[i] - neighbor_centers, axis=1)
        # possibly weight by distance?
        counts = np.bincount(neighbor_markers, weights=1 / neighbor_dist)
        dil_marker[i] = np.argmax(counts)
    return dil_marker


def erode_cell_marker(grid, marker, r=1):
    eroded_marker = marker.copy()
    for _ in range(r):
        new_marker = eroded_marker.copy()
        for i in range(grid.n_cells):
            neighbor_indices = grid.cell_neighbors(i, connections="points")
            if (eroded_marker[i] != eroded_marker[neighbor_indices]).any():
                new_marker[i] = 0
        eroded_marker = new_marker
    return eroded_marker


def remove_small_patches(grid, facet_marker, threshold, target_labels=None):
    new_marker = facet_marker.copy()
    if target_labels is None:
        target_labels = np.unique(facet_marker)
    for fm in target_labels:
        patch = grid.extract_cells(facet_marker == fm).connectivity()
        regionids, counts = np.unique(patch["RegionId"], return_counts=True)
        for ri, c in zip(regionids, counts):
            if c < threshold:
                new_marker[np.flatnonzero(facet_marker == fm)[patch["RegionId"] == ri]] = 0
    return new_marker


def _tet_face_topology(mesh):
    """
    Group all triangular (or quadratic triangular) faces of a tet mesh by their
    canonical (sorted) corner vertex triple, so adjacent tets sharing a face can be matched.

    Returns a dict with arrays:
      boundary_faces      (M, 3 or 6) outer-surface triangles (one parent tet)
      boundary_parents    (M,)        parent tet index for each boundary face
      interface_faces     (N, 3 or 6) triangles shared by two tets
      interface_parents_a (N,)        parent tet index for one side
      interface_parents_b (N,)        parent tet index for the other side
    """
    # Check for linear or quadratic tets
    cells_lin = mesh.cells_dict.get(pv.CellType.TETRA)
    cells_quad = mesh.cells_dict.get(pv.CellType.QUADRATIC_TETRA)

    if cells_quad is not None and len(cells_quad) > 0:
        cells = cells_quad
        # VTK Quadratic Tet Ordering:
        # Corners: 0,1,2,3. Edges: 4(0,1), 5(1,2), 6(2,0), 7(0,3), 8(1,3), 9(2,3)
        # We extract 6-node faces: [corner1, corner2, corner3, edge1, edge2, edge3]
        face_indices = np.array(
            [[1, 2, 3, 5, 9, 8], [0, 2, 3, 6, 9, 7], [0, 1, 3, 4, 8, 7], [0, 1, 2, 4, 5, 6]]
        )
        nodes_per_face = 6
    elif cells_lin is not None and len(cells_lin) > 0:
        cells = cells_lin
        face_indices = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
        nodes_per_face = 3
    else:
        raise ValueError("Input mesh has no tetrahedral or quadratic tetrahedral cells.")

    n_tets = len(cells)
    faces = cells[:, face_indices].reshape(-1, nodes_per_face)
    parents = np.repeat(np.arange(n_tets), 4)

    # Extract just the 3 corner nodes to sort and match.
    # This ensures exact matching without relying on mid-node permutations.
    corners = faces[:, :3]
    canonical = np.sort(corners, axis=1)

    order = np.lexsort(canonical.T[::-1])
    canonical = canonical[order]
    faces = faces[order]
    parents = parents[order]

    same_as_next = np.zeros(len(canonical), dtype=bool)
    same_as_next[:-1] = np.all(canonical[:-1] == canonical[1:], axis=1)
    same_as_prev = np.zeros(len(canonical), dtype=bool)
    same_as_prev[1:] = same_as_next[:-1]

    is_unique = ~same_as_next & ~same_as_prev
    pair_idx = np.where(same_as_next)[0]

    return {
        "boundary_faces": faces[is_unique],
        "boundary_parents": parents[is_unique],
        "interface_faces": faces[pair_idx],
        "interface_parents_a": parents[pair_idx],
        "interface_parents_b": parents[pair_idx + 1],
    }


def mark_interface_facets(mesh, label_array="marker", encoding_base=1000, ignore_above=None):
    """
    Build a facet mesh of the interfaces between regions with different
    markers in the input tet mesh.

    Each triangle is labelled with a unique ``interface_id`` encoded as
    ``min(a, b) * encoding_base + max(a, b)`` for the two adjacent
    region markers, and the original markers are also kept on
    ``region_a`` (lower) and ``region_b`` (higher).

    The resulting :class:`pyvista.PolyData` shares the parent's point
    array, so vertex indices line up with the parent tet mesh — ready to
    use as a FEniCS facet function.

    Parameters
    ----------
    ignore_above : int, optional
        If given, drop interfaces where *both* adjacent markers are larger
        than ``ignore_above`` (e.g. between the subdivisions of one region).
    """
    topo = _tet_face_topology(mesh)
    markers = np.asarray(mesh.cell_data[label_array])
    points = np.asarray(mesh.points)
    tet_centroids = np.asarray(mesh.cell_centers().points)

    faces = topo["interface_faces"]
    parents_a = topo["interface_parents_a"]
    parents_b = topo["interface_parents_b"]
    m_a = markers[parents_a]
    m_b = markers[parents_b]

    diff = m_a != m_b
    faces = faces[diff]
    parents_a = parents_a[diff]
    parents_b = parents_b[diff]
    m_a = m_a[diff]
    m_b = m_b[diff]

    lo = np.minimum(m_a, m_b).astype(np.int64)
    hi = np.maximum(m_a, m_b).astype(np.int64)

    if ignore_above is not None:
        keep = lo <= ignore_above
        faces = faces[keep]
        parents_a = parents_a[keep]
        parents_b = parents_b[keep]
        m_a = m_a[keep]
        m_b = m_b[keep]
        lo = lo[keep]
        hi = hi[keep]

    # Orient face winding so each normal points from the lo-marker region
    # toward the hi-marker region. The raw winding from _tet_face_topology
    # depends on which adjacent tet happened to be sorted first, so we
    # explicitly recompute and flip — same idea as mark_mesh flipping faces
    # whose label is on the "in" side.
    if len(faces) > 0:
        corners = points[faces[:, :3]]
        centroids = corners.mean(axis=1)
        e1 = corners[:, 1] - corners[:, 0]
        e2 = corners[:, 2] - corners[:, 0]
        normals = np.cross(e1, e2)

        lo_parents = np.where(m_a == lo, parents_a, parents_b)
        lo_centroids = tet_centroids[lo_parents]
        flip = np.einsum("ij,ij->i", normals, centroids - lo_centroids) < 0

        _flip_winding(faces, flip)

    interface_id = lo * encoding_base + hi

    return _build_facet_polydata(
        mesh,
        faces,
        {"interface_id": interface_id, "region_a": lo, "region_b": hi},
    )


def _build_facet_polydata(mesh, faces, scalars):
    """
    Builds a surface mesh containing the extracted interface facets.
    Dynamically creates a PolyData for linear faces (3 nodes) or an
    UnstructuredGrid for quadratic faces (6 nodes).
    """
    n = len(faces)
    points = np.asarray(mesh.points)

    if n == 0:
        grid = pv.PolyData()
        grid.points = points
    else:
        nodes_per_face = faces.shape[1]

        # VTK connectivity format: [n_nodes, p0, p1..., n_nodes, p0, p1...]
        cells = np.column_stack([np.full(n, nodes_per_face, dtype=np.int64), faces]).ravel()

        if nodes_per_face == 3:
            # Linear triangles are perfectly handled by PolyData
            grid = pv.PolyData(points, faces=cells)

        elif nodes_per_face == 6:
            # Quadratic triangles MUST be an UnstructuredGrid
            cell_types = np.full(n, pv.CellType.QUADRATIC_TRIANGLE, dtype=np.uint8)
            grid = pv.UnstructuredGrid(cells, cell_types, points)

        else:
            raise ValueError(f"Expected 3 or 6 nodes per face, got {nodes_per_face}.")

    # Attach the scalar data (interface IDs, markers, etc.)
    for name, arr in scalars.items():
        grid.cell_data[name] = np.asarray(arr)

    return grid


def _raw_faces(facet_mesh):
    """Raw connectivity of a facet mesh: (N, 3) for linear, (N, 6) for quadratic."""
    if isinstance(facet_mesh, pv.PolyData):
        return facet_mesh.faces.reshape(-1, 4)[:, 1:]
    cells = facet_mesh.cells_dict
    for ttype in (pv.CellType.QUADRATIC_TRIANGLE, pv.CellType.TRIANGLE):
        if ttype in cells:
            return cells[ttype]
    return np.empty((0, 6), dtype=np.int64)


def _flip_winding(faces, flip):
    """Reverse the winding of the rows selected by ``flip`` (in place)."""
    if faces.shape[1] == 3:
        faces[flip] = faces[flip][:, [0, 2, 1]]
    else:  # quadratic triangle: swap corners 1<->2 and their opposite edges
        faces[flip] = faces[flip][:, [0, 2, 1, 5, 4, 3]]
    return faces


def mark_boundary_facets(mesh, label_array="marker"):
    """
    Build a facet mesh of all outer boundary facets of the input tet mesh,
    each labelled by the marker of its single adjacent region.

    Face winding is oriented so each normal points outward (away from the
    parent tet centroid).

    The resulting :class:`pyvista.PolyData` shares the parent's point
    array, so vertex indices line up with the parent tet mesh.
    """
    topo = _tet_face_topology(mesh)
    markers = np.asarray(mesh.cell_data[label_array])
    points = np.asarray(mesh.points)
    tet_centroids = np.asarray(mesh.cell_centers().points)

    faces = topo["boundary_faces"]
    parents = topo["boundary_parents"]
    boundary = markers[parents].astype(np.int64)

    # Orient face winding so each normal points outward, away from the
    # parent tet centroid — same convention used in mark_spinal_boundary.
    if len(faces) > 0:
        corners = points[faces[:, :3]]
        centroids = corners.mean(axis=1)
        e1 = corners[:, 1] - corners[:, 0]
        e2 = corners[:, 2] - corners[:, 0]
        normals = np.cross(e1, e2)

        parent_centroids = tet_centroids[parents]
        flip = np.einsum("ij,ij->i", normals, centroids - parent_centroids) < 0

        _flip_winding(faces, flip)

    return _build_facet_polydata(mesh, faces, {"boundary": boundary})
