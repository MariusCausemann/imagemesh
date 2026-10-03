"""Roundtrip tests for save_mesh, covering native VTK and meshio formats."""

import numpy as np
import pytest
import pyvista as pv

from imagemesh.io import read_mesh, save_mesh


def _single_tet():
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    cells = np.array([4, 0, 1, 2, 3], dtype=np.int64)
    cell_types = np.array([pv.CellType.TETRA], dtype=np.uint8)
    mesh = pv.UnstructuredGrid(cells, cell_types, points)
    mesh.cell_data["marker"] = np.array([7], dtype=np.int32)
    return mesh


def _triangle():
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    faces = np.array([3, 0, 1, 2], dtype=np.int64)
    mesh = pv.PolyData(points, faces)
    mesh.cell_data["marker"] = np.array([3], dtype=np.int32)
    return mesh


@pytest.mark.parametrize("suffix", [".vtk", ".vtu", ".xdmf"])
def test_save_mesh_tet_roundtrip(tmp_path, suffix):
    mesh = _single_tet()
    out = tmp_path / f"tet{suffix}"

    save_mesh(mesh, out)
    assert out.exists()

    m = read_mesh(out)
    assert m.n_points == 4
    assert m.n_cells == 1
    assert int(m.cell_data["marker"][0]) == 7


@pytest.mark.parametrize("suffix", [".vtk", ".vtp", ".xdmf"])
def test_save_mesh_polydata_roundtrip(tmp_path, suffix):
    mesh = _triangle()
    out = tmp_path / f"tri{suffix}"

    save_mesh(mesh, out)
    assert out.exists()

    m = read_mesh(out)
    assert m.n_points == 3
    assert m.n_cells == 1


def test_save_mesh_creates_parent_dir(tmp_path):
    mesh = _single_tet()
    out = tmp_path / "nested" / "subdir" / "tet.vtu"

    save_mesh(mesh, out)
    assert out.exists()
