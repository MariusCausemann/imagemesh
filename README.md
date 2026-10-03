# imagemesh

Tetrahedral meshes of multi-label images. Shared meshing core of
[emimesh](https://github.com/MariusCausemann/emimesh) and
[brainmesh](https://github.com/MariusCausemann/brainmesh).

```
label image ──► surface nets + constrained smoothing   imagemesh.surface.extract_surface
            ──► feature/label preserving simplification imagemesh.simplification.simplify_surface
            ──► fTetWild + winding-number labelling     imagemesh.tetmesh.mesh_surface
            ──► interface / boundary facet tags         imagemesh.facets
```

Label `0` is the background; no other label is reserved. Domain-specific label
conventions (e.g. the ECS in emimesh, FreeSurfer ids in brainmesh) stay in the
downstream packages.

| module | contents |
|---|---|
| `image` | numpy / NIfTI ↔ `pv.ImageData` |
| `morphology` | label-image morphology and voxel helpers |
| `pinches`, `handles` | topology repair of label images before surface nets |
| `gaussian` | parallel recursive Gaussian filter |
| `surface`, `smoothing` | multi-label surface extraction, label volumes, self-intersections |
| `simplification` | parallel QEM edge-collapse simplification with bounded deviation |
| `winding_number` | fast (multi-label) generalized winding numbers |
| `tetmesh` | fTetWild meshing, tet labelling, tet mesh utilities |
| `facets` | interface and boundary facets of labelled tet meshes |
| `refine` | unlock tets with three boundary facets |
| `curved`, `mesh_optimizer` | quadratic (P2) meshes snapped to the surface |
| `io` | `save_mesh` / `read_mesh` (meshio formats such as XDMF) |

## Install

```bash
pip install -e ".[test]"   # or: conda env create -f environment.yml
python -m pytest
```
