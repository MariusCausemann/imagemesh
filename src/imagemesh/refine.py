#!/usr/bin/env python3
"""
imagemesh.refine
================

Fix geometric locking in tetrahedral meshes.

A tetrahedron is "locked" when three of its four triangular facets lie on a
boundary -- either the *external* boundary of the mesh (a facet shared by no
other cell) or an *internal* boundary / interface (a facet shared by a cell
carrying a different region marker).  Such a tet has a single interior facet
and an apex node that is pinned by three boundary facets, which causes
volumetric/geometric locking in FEM computations.

This module repeatedly finds every locked tet and removes the locking by
inserting a node on the single interior facet and splitting *both* tets that
share that facet into three (a conforming 1->3 split radiating from the new
node).  Because the original boundary facets are preserved exactly, the mesh
stays conforming and no new locked tets are created, so the loop converges.

Region markers are read from a named cell-data array; all other cell-data
arrays are inherited by child cells and all point-data arrays are linearly
interpolated onto inserted nodes.

CLI
---
    python -m imagemesh.refine INPUT_MESH MARKER_NAME OUTPUT_MESH
"""

from __future__ import annotations

import sys
from collections import defaultdict
from itertools import combinations

import numpy as np

VTK_TETRA = 10


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _signed_volume(p0, p1, p2, p3):
    return float(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0


def _oriented(points, tet):
    """Return the tet connectivity reordered to have positive signed volume."""
    p0, p1, p2, p3 = (points[i] for i in tet)
    if _signed_volume(p0, p1, p2, p3) < 0.0:
        return [tet[0], tet[1], tet[3], tet[2]]
    return list(tet)


def _faces(tet):
    """The four facets of a tet as canonical (sorted) node tuples."""
    return list(combinations(sorted(tet), 3))


# --------------------------------------------------------------------------- #
# Core algorithm (pure numpy / python -- no PyVista dependency)
# --------------------------------------------------------------------------- #
def unlock_locked_tets(
    points,
    tets,
    cell_data,
    marker_name,
    point_data=None,
    max_passes=10_000,
    verbose=True,
):
    """Subdivide every locked tet until none remain.

    Parameters
    ----------
    points : (N, 3) array-like
        Node coordinates.
    tets : (M, 4) array-like of int
        Tetra connectivity.
    cell_data : dict[str, array-like]
        Per-cell arrays (length M).  Must contain ``marker_name``.
        Children inherit their parent's values.
    marker_name : str
        Name of the cell-data array holding region markers.  A facet shared by
        two cells with *different* markers is treated as an internal boundary.
    point_data : dict[str, array-like], optional
        Per-point arrays (length N).  Inserted nodes get the mean of the three
        facet nodes.
    max_passes : int
        Safety cap on the number of refinement passes.
    verbose : bool
        Print per-pass progress.

    Returns
    -------
    new_points : (N', 3) ndarray
    new_tets : (M', 4) ndarray
    new_cell_data : dict[str, ndarray]
    new_point_data : dict[str, ndarray]
    """
    if marker_name not in cell_data:
        raise KeyError(
            f"marker array {marker_name!r} not found; available cell arrays: {sorted(cell_data)}"
        )

    # Mutable working copies (points / point-data rows as numpy arrays).
    points = [np.asarray(p, dtype=float) for p in points]
    tets = [list(map(int, t)) for t in tets]
    cell_data = {k: list(np.asarray(v)) for k, v in cell_data.items()}
    point_data = {k: list(np.asarray(v)) for k, v in (point_data or {}).items()}
    markers = cell_data[marker_name]  # alias -- same list object
    alive = [True] * len(tets)

    def add_point(face):
        i, j, k = face
        points.append((points[i] + points[j] + points[k]) / 3.0)
        for arr in point_data.values():
            arr.append((arr[i] + arr[j] + arr[k]) / 3.0)
        return len(points) - 1

    def add_tet(nodes, parent):
        tets.append(_oriented(points, nodes))
        alive.append(True)
        for arr in cell_data.values():  # includes the marker array
            arr.append(arr[parent])

    def classify():
        """Return (face_to_cells, is_internal) for the currently alive cells."""
        face_to_cells = defaultdict(list)
        for ci, tet in enumerate(tets):
            if not alive[ci]:
                continue
            for f in _faces(tet):
                face_to_cells[f].append(ci)

        def is_internal(face):
            cs = face_to_cells[face]
            return len(cs) == 2 and markers[cs[0]] == markers[cs[1]]

        return face_to_cells, is_internal

    def find_locked(face_to_cells, is_internal):
        """Tets with exactly one interior facet (=> three boundary facets)."""
        locked = []
        for ci, tet in enumerate(tets):
            if not alive[ci]:
                continue
            interior = [f for f in _faces(tet) if is_internal(f)]
            if len(interior) == 1:  # the other three facets are boundary
                f = interior[0]
                cs = face_to_cells[f]
                cj = cs[0] if cs[1] == ci else cs[1]
                locked.append((ci, f, cj))
        return locked

    total_splits = 0
    for pass_no in range(1, max_passes + 1):
        face_to_cells, is_internal = classify()
        locked = find_locked(face_to_cells, is_internal)
        if not locked:
            break

        splits = 0
        for ci, f, cj in locked:
            # A tet (or its partner) may already have been consumed this pass.
            if not alive[ci] or not alive[cj]:
                continue
            a, b, c = f
            d = next(iter(set(tets[ci]) - {a, b, c}))  # apex of cell ci
            e = next(iter(set(tets[cj]) - {a, b, c}))  # apex of cell cj
            m = add_point(f)  # node inserted on the interior facet
            # 1 -> 3 split of each tet, radiating from m across the facet.
            add_tet([a, b, m, d], ci)
            add_tet([b, c, m, d], ci)
            add_tet([c, a, m, d], ci)
            add_tet([a, b, m, e], cj)
            add_tet([b, c, m, e], cj)
            add_tet([c, a, m, e], cj)
            alive[ci] = alive[cj] = False
            splits += 1

        total_splits += splits
        if verbose:
            print(
                f"  pass {pass_no}: {len(locked)} locked tet(s), {splits} facet split(s)",
                file=sys.stderr,
            )
        if splits == 0:
            # No progress possible (shouldn't happen) -- bail to avoid looping.
            break
    else:
        raise RuntimeError(f"did not converge within max_passes={max_passes}")

    # Report any tet we could not fix (all four facets on a boundary -> no
    # interior facet to insert a node on).
    face_to_cells, is_internal = classify()
    unfixable = sum(
        1
        for ci, tet in enumerate(tets)
        if alive[ci] and not any(is_internal(f) for f in _faces(tet))
    )
    if verbose:
        print(
            f"  done: {total_splits} facet split(s) total",
            file=sys.stderr,
        )
        if unfixable:
            print(
                f"  WARNING: {unfixable} tet(s) have all four facets on a "
                f"boundary and cannot be unlocked by interior insertion",
                file=sys.stderr,
            )

    # Compact: drop consumed cells, keep node ordering.
    keep = [i for i in range(len(tets)) if alive[i]]
    new_tets = np.asarray([tets[i] for i in keep], dtype=np.int64)
    new_points = np.asarray(points, dtype=float)
    new_cell_data = {k: np.asarray([v[i] for i in keep]) for k, v in cell_data.items()}
    new_point_data = {k: np.asarray(v) for k, v in point_data.items()}
    return new_points, new_tets, new_cell_data, new_point_data


# --------------------------------------------------------------------------- #
# PyVista wrapper
# --------------------------------------------------------------------------- #
def subdivide_grid(grid, marker_name, max_passes=10_000, verbose=True):
    """Unlock locked tets in a PyVista UnstructuredGrid of tetrahedra.

    Returns a new ``pyvista.UnstructuredGrid``.  All cell-data arrays are
    inherited by child cells; all point-data arrays are interpolated onto
    inserted nodes.
    """
    import pyvista as pv

    grid = grid.cast_to_unstructured_grid()
    if grid.n_cells and not np.all(grid.celltypes == VTK_TETRA):
        bad = sorted(set(np.unique(grid.celltypes).tolist()) - {VTK_TETRA})
        raise ValueError(
            f"grid must contain only tetrahedra (VTK type {VTK_TETRA}); "
            f"found other cell type(s): {bad}"
        )
    if marker_name not in grid.cell_data:
        raise KeyError(
            f"marker array {marker_name!r} not found in cell_data; "
            f"available: {list(grid.cell_data.keys())}"
        )

    # Connectivity in original cell order (so it aligns 1:1 with cell_data).
    cells = grid.cells.reshape(-1, 5)
    if not np.all(cells[:, 0] == 4):
        raise ValueError("unexpected cell connectivity layout for tetrahedra")
    conn = cells[:, 1:5]

    cell_data = {k: np.asarray(grid.cell_data[k]) for k in grid.cell_data.keys()}
    point_data = {k: np.asarray(grid.point_data[k]) for k in grid.point_data.keys()}

    new_points, new_tets, new_cell_data, new_point_data = unlock_locked_tets(
        grid.points,
        conn,
        cell_data,
        marker_name,
        point_data=point_data,
        max_passes=max_passes,
        verbose=verbose,
    )

    out = pv.UnstructuredGrid({VTK_TETRA: new_tets}, new_points)
    for name, arr in new_cell_data.items():
        out.cell_data[name] = arr
    for name, arr in new_point_data.items():
        out.point_data[name] = arr
    return out
