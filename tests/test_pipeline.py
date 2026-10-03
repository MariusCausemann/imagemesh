"""End-to-end: label image -> surface -> simplified surface -> tet mesh -> facet tags."""

import numpy as np
import pytest

from imagemesh import (
    extract_surface,
    mark_boundary_facets,
    mark_interface_facets,
    mesh_surface,
    np2pv,
    read_mesh,
    save_mesh,
)
from imagemesh.surface import label_volumes


def nested_spheres(n=32):
    """Label 1 (box) containing two separate spheres, labels 2 and 3."""
    x = (np.indices((n, n, n)) + 0.5) / n
    img = np.ones((n, n, n), np.uint16)
    img[np.linalg.norm(x - np.array([0.3, 0.5, 0.5])[:, None, None, None], axis=0) < 0.18] = 2
    img[np.linalg.norm(x - np.array([0.7, 0.5, 0.5])[:, None, None, None], axis=0) < 0.18] = 3
    return np2pv(img, (1.0 / n,) * 3)


def test_pipeline_end_to_end(tmp_path):
    surf = extract_surface(nested_spheres())
    dx = surf.field_data["dx"][0]
    mesh, dec = mesh_surface(surf, envelopsize=0.1 * dx, simplify_eps=0.5 * dx, label_name="marker")
    assert dec.n_points < surf.n_points

    markers, counts = np.unique(mesh["marker"], return_counts=True)
    assert markers.tolist() == [1, 2, 3]
    assert counts.min() > 50

    vol = np.abs(mesh.compute_cell_sizes(length=False, area=False)["Volume"])
    assert vol.sum() == pytest.approx(1.0, rel=1e-3)
    # meshing and labelling keep the volumes enclosed by the simplified surface;
    # smoothing and simplification shrink the small spheres (radius 5.8 voxels)
    ref = label_volumes(dec.points.astype(float), dec.regular_faces, dec["boundary_labels"])
    sphere = 4 / 3 * np.pi * 0.18**3
    for m in (2, 3):
        assert vol[mesh["marker"] == m].sum() == pytest.approx(ref[m], rel=1e-2)
        assert 0.6 * sphere < ref[m] < sphere
    assert mesh.cell_quality("scaled_jacobian")["scaled_jacobian"].min() > 0

    interfaces = mark_interface_facets(mesh)
    ids = set(np.unique(interfaces["interface_id"]).tolist())
    assert ids == {1 * 1000 + 2, 1 * 1000 + 3}  # the spheres do not touch
    boundaries = mark_boundary_facets(mesh)
    assert set(np.unique(boundaries["boundary"]).tolist()) == {1}
    assert boundaries.area == pytest.approx(6.0, rel=1e-3)

    path = tmp_path / "mesh.xdmf"
    save_mesh(mesh, path)
    back = read_mesh(path)
    assert back.n_cells == mesh.n_cells
    np.testing.assert_array_equal(back["marker"], mesh["marker"])
