"""imagemesh: tetrahedral meshes of multi-label images.

Pipeline: label image -> surface nets (surface.extract_surface) ->
simplification (simplification.simplify_surface) -> fTetWild + winding
number labeling (tetmesh.mesh_surface) -> facet tags (facets).
Label 0 is the background; no other label is reserved.
"""

from imagemesh.facets import mark_boundary_facets, mark_interface_facets
from imagemesh.image import nibabel_to_pyvista, np2pv
from imagemesh.io import read_mesh, save_mesh
from imagemesh.simplification import simplify_surface
from imagemesh.surface import extract_surface
from imagemesh.tetmesh import mark_mesh, mesh_surface
from imagemesh.winding_number import label_points

__all__ = [
    "extract_surface",
    "label_points",
    "mark_boundary_facets",
    "mark_interface_facets",
    "mark_mesh",
    "mesh_surface",
    "nibabel_to_pyvista",
    "np2pv",
    "read_mesh",
    "save_mesh",
    "simplify_surface",
]
