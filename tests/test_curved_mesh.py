"""Tests for ``imagemesh.curved`` — quadratic conversion, quality and adaptive snapping."""

import numpy as np
import pytest
import pyvista as pv

from imagemesh.curved import (
    adaptive_snap_boundaries,
    compute_quadratic_quality,
    convert_to_quadratic,
    print_quality_stats,
)

# ---------- helpers ----------------------------------------------------------


def _unit_linear_tet():
    """Single linear reference tet at (0,0,0)/(1,0,0)/(0,1,0)/(0,0,1)."""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    cells = np.array([4, 0, 1, 2, 3])
    ctypes = np.array([pv.CellType.TETRA], dtype=np.uint8)
    return pv.UnstructuredGrid(cells, ctypes, points)


def _unit_quadratic_tet(point_overrides=None):
    """Single quadratic tet with midnodes placed exactly at edge midpoints.

    VTK QUADRATIC_TETRA node order: corners 0..3 then edges
    4(0-1), 5(1-2), 6(0-2), 7(0-3), 8(1-3), 9(2-3).
    """
    points = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [0.0, 0.0, 1.0],  # 3
            [0.5, 0.0, 0.0],  # 4 = mid(0,1)
            [0.5, 0.5, 0.0],  # 5 = mid(1,2)
            [0.0, 0.5, 0.0],  # 6 = mid(0,2)
            [0.0, 0.0, 0.5],  # 7 = mid(0,3)
            [0.5, 0.0, 0.5],  # 8 = mid(1,3)
            [0.0, 0.5, 0.5],  # 9 = mid(2,3)
        ],
        dtype=float,
    )
    if point_overrides:
        for idx, coord in point_overrides.items():
            points[idx] = coord
    cells = np.concatenate([[10], np.arange(10)])
    ctypes = np.array([pv.CellType.QUADRATIC_TETRA], dtype=np.uint8)
    return pv.UnstructuredGrid(cells, ctypes, points)


# ---------- convert_to_quadratic --------------------------------------------


def test_convert_to_quadratic_single_tet():
    """A linear tet should become a 10-node quadratic tet with the same volume."""
    lin = _unit_linear_tet()
    quad = convert_to_quadratic(lin)

    assert quad.n_cells == lin.n_cells
    assert quad.n_points == 10
    assert quad.celltypes.tolist() == [pv.CellType.QUADRATIC_TETRA]

    # Cell connectivity stride: [10, p0..p9]
    cells = quad.cells.reshape(-1, 11)
    np.testing.assert_array_equal(cells[:, 0], 10)

    # Volume of the linear tet should be preserved (linear-to-quadratic only adds midnodes).
    np.testing.assert_allclose(quad.volume, lin.volume, rtol=1e-12)


def test_convert_to_quadratic_midnodes_lie_on_edges():
    """Mid-edge nodes must sit at the midpoints of their parent edges."""
    lin = _unit_linear_tet()
    quad = convert_to_quadratic(lin)
    pts = quad.points
    cells = quad.cells.reshape(-1, 11)[:, 1:]
    c = cells[0]

    edge_pairs = [(0, 1, 4), (1, 2, 5), (0, 2, 6), (0, 3, 7), (1, 3, 8), (2, 3, 9)]
    for a, b, m in edge_pairs:
        np.testing.assert_allclose(pts[c[m]], 0.5 * (pts[c[a]] + pts[c[b]]), atol=1e-12)


# ---------- compute_quadratic_quality ---------------------------------------


def test_quality_perfect_tet_is_positive():
    quad = _unit_quadratic_tet()
    q = compute_quadratic_quality(quad)
    assert q.shape == (1,)
    assert q[0] > 0.0


def test_quality_array_length_matches_cells():
    """One quality value per parent quadratic tet (worst over the 8 sub-tets)."""
    quad = _unit_quadratic_tet()
    points = np.vstack([quad.points, quad.points + np.array([2.0, 0.0, 0.0])])
    cells = np.concatenate(
        [
            [10],
            np.arange(10),
            [10],
            np.arange(10) + 10,
        ]
    )
    ctypes = np.array([pv.CellType.QUADRATIC_TETRA] * 2, dtype=np.uint8)
    two_tet_mesh = pv.UnstructuredGrid(cells, ctypes, points)

    q = compute_quadratic_quality(two_tet_mesh)
    assert q.shape == (2,)
    assert np.all(q > 0)


def test_quality_detects_inverted_midnode():
    """Pushing a midnode far off its edge inverts a sub-tet → quality goes negative."""
    bad = _unit_quadratic_tet(point_overrides={9: [-2.0, -2.0, 3.0]})
    q = compute_quadratic_quality(bad)
    assert q[0] < 0.0


def test_quality_chooses_shortest_octahedron_diagonal():
    """The internal slicing branch must pick whichever diagonal is shortest.

    We construct three quadratic tets where exactly one of the 4-9 / 5-7 / 6-8
    diagonals is much shorter than the other two; the routine should return a
    reasonable (positive) quality for all three configurations.
    """
    # Start from the perfect tet, then slightly compress the chosen diagonal
    # by nudging both endpoint midnodes towards the opposite face.
    qualities = []
    for short_diag in ("4-9", "5-7", "6-8"):
        overrides = {}
        if short_diag == "4-9":
            overrides[4] = [0.5, 0.05, 0.05]
            overrides[9] = [0.05, 0.45, 0.45]
        elif short_diag == "5-7":
            overrides[5] = [0.45, 0.45, 0.05]
            overrides[7] = [0.05, 0.05, 0.5]
        else:  # 6-8
            overrides[6] = [0.05, 0.5, 0.05]
            overrides[8] = [0.45, 0.05, 0.45]
        mesh = _unit_quadratic_tet(point_overrides=overrides)
        q = compute_quadratic_quality(mesh)
        qualities.append(q[0])

    # All three slicings should remain valid (positive quality).
    assert all(q > 0 for q in qualities), qualities


# ---------- print_quality_stats ----------------------------------------------


def test_print_quality_stats_returns_quality_array(capsys):
    quad = _unit_quadratic_tet()
    q = print_quality_stats(quad, "unit-tet")
    assert q.shape == (1,)
    assert q[0] > 0
    captured = capsys.readouterr().out
    assert "unit-tet" in captured


def test_print_quality_stats_linear_branch():
    """A linear mesh should be evaluated through PyVista's ``cell_quality``."""
    lin = _unit_linear_tet()
    q = print_quality_stats(lin, "lin-tet")
    expected = lin.cell_quality(quality_measure="scaled_jacobian").cell_data["scaled_jacobian"]
    np.testing.assert_allclose(q, expected)


# ---------- adaptive_snap_boundaries (slow, requires pytetwild) -------------


@pytest.fixture(scope="module")
def sphere_quad_mesh():
    """Coarse tetrahedralised unit sphere converted to quadratic."""
    pytetwild = pytest.importorskip("pytetwild")
    coarse_surf = pv.Sphere(radius=1.0, theta_resolution=8, phi_resolution=8).triangulate()
    lin = pytetwild.tetrahedralize_pv(
        coarse_surf,
        edge_length_fac=0.3,
        stop_energy=10,
        quiet=True,
    )
    return convert_to_quadratic(lin)


@pytest.mark.slow
def test_adaptive_snap_preserves_topology(sphere_quad_mesh):
    quad = sphere_quad_mesh.copy()
    target = pv.Sphere(radius=1.0, theta_resolution=64, phi_resolution=64).triangulate()

    n_cells_before = quad.n_cells
    n_points_before = quad.n_points
    cells_before = quad.cells.copy()

    adaptive_snap_boundaries(quad, target, only_high_order=True)

    assert quad.n_cells == n_cells_before
    assert quad.n_points == n_points_before
    np.testing.assert_array_equal(quad.cells, cells_before)


@pytest.mark.slow
def test_adaptive_snap_only_high_order_keeps_corners(sphere_quad_mesh):
    """``only_high_order=True`` may move only mid-edge nodes, never corners."""
    quad = sphere_quad_mesh.copy()
    target = pv.Sphere(radius=1.0, theta_resolution=64, phi_resolution=64).triangulate()

    corner_ids = np.unique(quad.cells.reshape(-1, 11)[:, 1:5])
    corners_before = quad.points[corner_ids].copy()

    adaptive_snap_boundaries(quad, target, only_high_order=True)

    np.testing.assert_array_equal(quad.points[corner_ids], corners_before)


@pytest.mark.slow
def test_adaptive_snap_pulls_midnodes_toward_target(sphere_quad_mesh):
    """Boundary mid-edge nodes should land closer to the target sphere on average."""
    quad = sphere_quad_mesh.copy()
    target = pv.Sphere(radius=1.0, theta_resolution=64, phi_resolution=64).triangulate()

    surf = quad.extract_surface(algorithm="dataset_surface")
    bnd_ids = set(surf["vtkOriginalPointIds"].tolist())

    edges = [(0, 1, 4), (1, 2, 5), (0, 2, 6), (0, 3, 7), (1, 3, 8), (2, 3, 9)]

    def boundary_mid_radius_errors(mesh):
        cells = mesh.cells.reshape(-1, 11)[:, 1:]
        errs = []
        for c in cells:
            for a, b, m in edges:
                if c[a] in bnd_ids and c[b] in bnd_ids:
                    errs.append(abs(np.linalg.norm(mesh.points[c[m]]) - 1.0))
        return np.array(errs)

    err_before = boundary_mid_radius_errors(quad)
    adaptive_snap_boundaries(quad, target, only_high_order=True)
    err_after = boundary_mid_radius_errors(quad)

    # On average, midnodes should sit closer to the unit sphere after snapping.
    assert err_after.mean() < err_before.mean()


@pytest.mark.slow
def test_adaptive_snap_no_boundary_nodes_is_noop():
    """A mesh with no surface (single embedded tet, no extracted boundary nodes
    in the target) should still leave the mesh untouched and not raise."""
    quad = _unit_quadratic_tet()
    # Target far away so corners would be pulled — but with only_high_order=True
    # and a single tet whose corner nodes are corners, the corners stay put.
    target = pv.Sphere(radius=10.0, center=(100, 100, 100)).triangulate()

    pts_before = quad.points.copy()
    adaptive_snap_boundaries(quad, target, only_high_order=True, n_steps=2)
    # only_high_order keeps corners frozen.
    np.testing.assert_array_equal(quad.points[:4], pts_before[:4])
