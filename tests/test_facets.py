"""Tests for imagemesh.facets and the tet mesh utilities on a split box."""

import numpy as np
import pytest
import pyvista as pv

from imagemesh.facets import mark_boundary_facets, mark_interface_facets
from imagemesh.tetmesh import filter_by_mask, largest_face_connected


@pytest.fixture(scope="module")
def split_box_mesh():
    """
    A pytetwild-meshed box [0,2]x[0,1]x[0,1] split at x=1.0 into two
    regions with markers 10 (left) and 20 (right).
    """
    import pytetwild

    surf = pv.Box(bounds=(0, 2, 0, 1, 0, 1)).triangulate().subdivide(2)
    mesh = pytetwild.tetrahedralize_pv(surf, edge_length_fac=0.1, stop_energy=10, quiet=True)
    centroids = mesh.cell_centers().points
    mesh.cell_data["marker"] = np.where(centroids[:, 0] < 1.0, 10, 20).astype(np.int32)
    return mesh


def test_split_box_has_many_tets(split_box_mesh):
    assert split_box_mesh.n_cells > 100
    assert np.unique(split_box_mesh.celltypes).tolist() == [pv.CellType.TETRA]


def test_mark_interface_facets_split_box(split_box_mesh):
    interfaces = mark_interface_facets(split_box_mesh)
    assert interfaces.n_cells > 0

    # Point array must match parent — required for FEniCS facet-function use
    assert interfaces.n_points == split_box_mesh.n_points
    np.testing.assert_array_equal(interfaces.points, split_box_mesh.points)

    np.testing.assert_array_equal(interfaces.cell_data["region_a"], 10)
    np.testing.assert_array_equal(interfaces.cell_data["region_b"], 20)
    np.testing.assert_array_equal(interfaces.cell_data["interface_id"], 10 * 1000 + 20)

    face_verts = interfaces.faces.reshape(-1, 4)[:, 1:]
    centroid_x = split_box_mesh.points[face_verts][..., 0].mean()
    assert abs(centroid_x - 1.0) < 0.05

    # normals point from the lower to the higher marker (+x): the x component of
    # the area weighted normals of the (jagged) interface is its projected area 1
    p = interfaces.points[face_verts]
    normal_sum = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]).sum(axis=0) / 2
    assert normal_sum[0] == pytest.approx(1.0)


def test_mark_interface_facets_ignore_above(split_box_mesh):
    assert mark_interface_facets(split_box_mesh, ignore_above=10).n_cells > 0
    assert mark_interface_facets(split_box_mesh, ignore_above=9).n_cells == 0


def test_mark_interface_facets_same_marker_yields_none(split_box_mesh):
    mesh = split_box_mesh.copy()
    mesh.cell_data["marker"][:] = 10
    interfaces = mark_interface_facets(mesh)
    assert interfaces.n_cells == 0
    assert interfaces.n_points == mesh.n_points


def test_mark_boundary_facets_split_box(split_box_mesh):
    boundaries = mark_boundary_facets(split_box_mesh)
    assert boundaries.n_points == split_box_mesh.n_points
    np.testing.assert_array_equal(boundaries.points, split_box_mesh.points)
    assert set(np.unique(boundaries.cell_data["boundary"])) <= {10, 20}
    assert np.isclose(boundaries.area, 10.0, rtol=0.01)


def test_facet_face_count_conservation(split_box_mesh):
    """4·n_tets == 2·n_interior + n_boundary; n_interface ≤ n_interior."""
    interfaces = mark_interface_facets(split_box_mesh)
    boundaries = mark_boundary_facets(split_box_mesh)
    n_total_face_uses = 4 * split_box_mesh.n_cells
    assert (n_total_face_uses - boundaries.n_cells) % 2 == 0
    assert (n_total_face_uses - boundaries.n_cells) // 2 >= interfaces.n_cells


def test_filter_by_mask_and_largest_component(split_box_mesh):
    left = filter_by_mask(split_box_mesh, split_box_mesh["marker"] == 10)
    assert left.n_points == split_box_mesh.n_points
    assert np.all(left["marker"] == 10)
    assert largest_face_connected(left.clean()).n_cells == left.n_cells
