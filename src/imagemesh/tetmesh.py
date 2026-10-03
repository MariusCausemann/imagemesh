"""
Tetrahedral meshes of multi-label surfaces: fTetWild (pytetwild) meshing of
the whole surface, labeling of the tetrahedra by the generalized winding
number of the input surface, and tet mesh utilities.
"""

import time
from collections import defaultdict

import numpy as np
import pytetwild
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from imagemesh.facets import _raw_faces
from imagemesh.simplification import simplify_surface
from imagemesh.winding_number import label_points

# treat linear + quadratic tets as "volume", linear + quadratic tris as "surface"
VOL_TYPES = (pv.CellType.TETRA, pv.CellType.QUADRATIC_TETRA)
TRI_TYPES = (pv.CellType.TRIANGLE, pv.CellType.QUADRATIC_TRIANGLE)

TETWILD_DEFAULTS = dict(
    edge_length_fac=0.05,
    stop_energy=10,
    disable_filtering=True,
    coarsen=False,
    num_threads=1,
    quiet=True,
)


def mark_mesh(mesh, surf, label_name="label"):
    """
    Label each tetrahedron of mesh with the region of surf (cell data
    'boundary_labels') containing its center (cell data label_name), and drop
    the tetrahedra outside all regions (label 0).
    """
    marker = label_points(
        surf.points,
        surf.regular_faces,
        surf.cell_data["boundary_labels"],
        mesh.cell_centers().points,
    )
    mesh.cell_data[label_name] = marker
    return mesh.extract_cells(marker > 0)


def mesh_surface(surf, envelopsize=None, simplify_eps=None, label_name="label", **tetwild_kwargs):
    """
    Simplify the multi-label surface surf (simplify_eps: absolute distance
    tolerance, None to skip) and tetrahedralize it with fTetWild.

    envelopsize: absolute size of the fTetWild surface envelope; overrides the
    relative `epsilon` of pytetwild.
    tetwild_kwargs: forwarded to pytetwild.tetrahedralize_pv, on top of
    TETWILD_DEFAULTS (e.g. edge_length_fac, the target edge length relative to
    the bounding box diagonal, stop_energy or num_threads).

    Returns the tet mesh with cell data label_name and the simplified surface.
    """
    start = time.time()
    if simplify_eps:
        n = surf.n_points
        surf = simplify_surface(surf, epsilon=simplify_eps)
        print(f"simplified surface: {n} -> {surf.n_points} points")

    kwargs = {**TETWILD_DEFAULTS, **tetwild_kwargs}
    if envelopsize is not None:
        diag = np.linalg.norm(np.ptp(surf.points, axis=0))
        kwargs["epsilon"] = envelopsize / diag
    mesh = pytetwild.tetrahedralize_pv(surf, **kwargs)
    mesh = mark_mesh(mesh, surf, label_name=label_name)
    print("meshing finished!")
    mesh.field_data["runtime"] = time.time() - start
    mesh.field_data["threads"] = kwargs["num_threads"]
    return mesh, surf


def compute_surface_volume(mesh, labels, label_array="label"):
    """Volume and surface area of the regions of mesh with the given labels."""
    mesh = mesh.compute_cell_sizes()
    mesh["Volume"] = np.abs(mesh["Volume"])
    assert (mesh["Volume"] > 0).all()
    volumes, surface_areas = [], []
    for label in labels:
        cell = mesh.extract_cells(np.isin(mesh.cell_data[label_array], [label]))
        surf = cell.extract_surface()
        surface_areas.append(float(surf.compute_cell_sizes()["Area"].sum()))
        volumes.append(float(cell["Volume"].sum()))
    return volumes, surface_areas


def filter_by_mask(mesh, mask):
    """
    Extract the subset of cells with mask,
    while strictly preserving the original point array.
    """
    # Check if the mesh is homogeneous (every cell takes up the same flat length)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    if len(mesh.cells) % mesh.n_cells != 0:
        raise ValueError("Mesh contains mixed cell types. Cannot use 2D reshape method.")

    # Number of integers each cell takes up (e.g., 5 for linear tets: [4, p1, p2, p3, p4])
    stride = len(mesh.cells) // mesh.n_cells

    # Reshape, apply the mask to the rows, and flatten back to 1D
    cells_2d = mesh.cells.reshape(mesh.n_cells, stride)
    new_cells = cells_2d[mask].flatten()

    new_celltypes = mesh.celltypes[mask]

    # Rebuild the grid forcing it to use the full, unpruned points array
    retained_mesh = pv.UnstructuredGrid(new_cells, new_celltypes, mesh.points)

    # Transfer the cell data over
    for name in mesh.cell_data:
        retained_mesh.cell_data[name] = mesh.cell_data[name][mask]

    for name in mesh.point_data:
        retained_mesh.point_data[name] = mesh.point_data[name]

    return retained_mesh


def _gather(mesh, types, n_corner):
    """(corners (m, n_corner), global_ids (m,)) for the given cell types."""
    cdict, ct = mesh.cells_dict, mesh.celltypes
    corner_blocks, id_blocks = [], []
    for t in types:
        if t in cdict:
            gids = np.where(ct == t)[0]  # ascending, aligns with cdict[t]
            corner_blocks.append(cdict[t][:, :n_corner])
            id_blocks.append(gids)
    if not corner_blocks:
        return np.empty((0, n_corner), int), np.empty(0, int)
    return np.vstack(corner_blocks), np.concatenate(id_blocks)


def _tet_volumes(points, corners):
    p = points[corners]  # (m, 4, 3)
    a, b, c = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]
    return np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0


def largest_face_connected(mesh, by_volume=True, keep_surface_tris=True):
    tet_corners, tet_gids = _gather(mesh, VOL_TYPES, 4)
    n = len(tet_corners)
    if n == 0:
        raise ValueError("no tetrahedral cells found")

    combos = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    face_to_local = defaultdict(list)  # corner-face -> local tet indices
    for li, tet in enumerate(tet_corners):
        for a, b, c in combos:
            face_to_local[tuple(sorted((tet[a], tet[b], tet[c])))].append(li)

    rows, cols = [], []
    for cs in face_to_local.values():
        if len(cs) == 2:  # internal face shared by 2 tets
            x, y = cs
            rows += [x, y]
            cols += [y, x]

    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, labels = connected_components(adj, directed=False)

    if by_volume:
        weights = np.bincount(labels, weights=_tet_volumes(mesh.points, tet_corners))
    else:
        weights = np.bincount(labels)
    win = weights.argmax()

    keep_local = np.where(labels == win)[0]
    keep_global = tet_gids[keep_local].tolist()

    if keep_surface_tris:
        tri_corners, tri_gids = _gather(mesh, TRI_TYPES, 3)
        if len(tri_gids):
            kept_faces = set()
            for li in keep_local:
                tet = tet_corners[li]
                for a, b, c in combos:
                    kept_faces.add(tuple(sorted((tet[a], tet[b], tet[c]))))
            for ti, tri in zip(tri_gids, tri_corners):
                if tuple(sorted(tri)) in kept_faces:
                    keep_global.append(int(ti))

    return mesh.extract_cells(keep_global)


def extract_reduced_facets(reduced_tets, full_facets):
    """Facets of `full_facets` bounding `reduced_tets`, rebuilt on
    `reduced_tets.points` with its exact node ordering. Linear or quadratic.
    Assumes both objects share point_data['gid'] from the full mesh."""
    TET = (pv.CellType.TETRA, pv.CellType.QUADRATIC_TETRA)
    FACE_IDX = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

    def keys(rows):
        a = np.ascontiguousarray(np.sort(rows, axis=1))
        return a.view([("", a.dtype)] * a.shape[1]).ravel()

    # facet connectivity (single triangle type) + point->global map
    # (linear facet meshes are PolyData and have no cells_dict, hence _raw_faces)
    conn = _raw_faces(full_facets)  # (n, 3 or 6)
    ttype = pv.CellType.TRIANGLE if conn.shape[1] == 3 else pv.CellType.QUADRATIC_TRIANGLE
    fg = (
        np.asarray(full_facets.point_data["gid"])
        if "gid" in full_facets.point_data
        else np.arange(full_facets.n_points)
    )  # facet-pt -> global

    # corner faces of the reduced tets, as global ids
    csf_gid = np.asarray(reduced_tets.point_data["gid"]).astype(np.uint32)  # local -> global
    tet_corners = np.vstack(
        [reduced_tets.cells_dict[t][:, :4] for t in TET if t in reduced_tets.cells_dict]
    )
    csf_faces = csf_gid[tet_corners][:, FACE_IDX].reshape(-1, 3)

    keep = np.isin(keys(fg[conn[:, :3]]), keys(csf_faces))

    # remap kept facets straight to reduced_tets-local numbering
    g2l = np.full(int(csf_gid.max()) + 1, -1, np.int64)
    g2l[csf_gid] = np.arange(len(csf_gid), dtype=np.uint32)
    local = g2l[fg[conn[keep]]]
    assert (local >= 0).all(), "kept facet node missing from reduced_tets (gid mismatch)"

    out = pv.UnstructuredGrid({ttype: local}, reduced_tets.points)
    out.point_data.update(reduced_tets.point_data)  # shares csf_mesh point arrays
    for k in full_facets.cell_data:
        out.cell_data[k] = np.asarray(full_facets.cell_data[k])[keep]
    return out
