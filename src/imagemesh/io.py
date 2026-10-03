"""Reading and writing meshes, with meshio for the formats pyvista cannot write."""

import pathlib

import pyvista as pv

# Extensions PyVista's native .save() does not handle but meshio writes well.
# Kept explicit (rather than try/except) so behaviour is predictable.
_MESHIO_EXTENSIONS = {
    ".xdmf",
    ".xmf",  # XDMF (FEniCS, ParaView)
    ".msh",  # Gmsh
    ".med",  # Salome / Code_Aster
    ".inp",  # Abaqus
    ".h5m",  # MOAB
    ".mdpa",  # KratosMultiphysics
    ".off",  # Geomview OFF
}


def save_mesh(mesh, path):
    """Save a PyVista mesh, routing meshio-only extensions through pv.save_meshio.

    Parameters
    ----------
    mesh : pyvista.DataSet
        PolyData, UnstructuredGrid, or ImageData. ImageData is supported only
        for native VTK extensions (e.g. .vti); meshio formats require a mesh
        with explicit cell connectivity.
    path : str or pathlib.Path
        Output path. Format is inferred from the suffix.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in _MESHIO_EXTENSIONS:
        pv.save_meshio(str(path), mesh)
    else:
        mesh.save(str(path))


def read_mesh(path):
    """Read a mesh from disk, routing meshio-only extensions through pv.read_meshio.

    Mirrors :func:`save_mesh`. PyVista's native XDMF reader assumes time-series
    metadata that meshio does not write for single-snapshot meshes, so we go
    through meshio for those extensions.
    """
    path = pathlib.Path(path)
    if path.suffix.lower() in _MESHIO_EXTENSIONS:
        return pv.read_meshio(str(path))
    return pv.read(str(path))


def load_marked_mesh(path, label_array="marker"):
    """Load a labelled tetrahedral mesh (see tetmesh.mark_mesh) from disk."""
    mesh = read_mesh(path)
    if label_array not in mesh.cell_data:
        raise ValueError(
            f"{path} has no '{label_array}' cell-data array — was it produced by mark_mesh?"
        )
    return mesh
