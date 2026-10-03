"""Label images as pyvista.ImageData (labels stored as cell data) and NIfTI conversion."""

import fastremap
import numpy as np
import pyvista as pv


def np2pv(arr, resolution, roimask=None, as_point_data=False):
    """Image grid with arr (Fortran order) as cell data 'data' (or as point data)."""
    dimensions = np.array(arr.shape)
    if not as_point_data:
        dimensions += 1

    grid = pv.ImageData(dimensions=dimensions, spacing=resolution, origin=(0, 0, 0))
    grid["data"] = arr.flatten(order="F")
    if roimask is not None:
        grid["roimask"] = roimask.flatten(order="F")
    return grid


def get_cell_frequencies(img):
    """Labels of img and their voxel counts (2, n_labels), sorted by count."""
    cell_labels, cell_counts = fastremap.unique(img, return_counts=True)
    indices = np.argsort(cell_counts)
    return np.vstack([cell_labels[indices], cell_counts[indices]])


def get_bounding_box(mesh, eps=0):
    """Bounding box of mesh, enlarged by eps on each side."""
    eps = np.vstack([np.ones(3), -np.ones(3)]).T * eps
    bounds = np.array(mesh.bounds).reshape(3, 2)
    return pv.Box((bounds + eps).flatten())


def get_img(img):
    """Load a NIfTI image (path or nibabel image) in closest canonical (RAS) orientation."""
    import nibabel as nib

    if not isinstance(img, nib.Nifti1Image):
        img = nib.load(img)
    return nib.as_closest_canonical(img)


def upsample_nib(img, factor=2, order=0):
    """
    Upsamples a nibabel image by a given factor while preserving the
    physical bounding box and affine orientation.

    Parameters:
        img: nibabel.Nifti1Image
        factor: upsampling factor (e.g. 2 means double resolution)
        order: interpolation order (0=nearest, 1=linear, 3=cubic)
    """
    import nibabel.processing as nibp

    target_shape = list(img.shape)
    for i in range(min(3, len(target_shape))):
        target_shape[i] = int(np.round(target_shape[i] * factor))

    target_affine = img.affine.copy()
    target_affine[:3, :3] /= factor

    offset_multiplier = (1.0 / factor - 1.0) / 2.0
    offset = img.affine[:3, :3] @ np.array([offset_multiplier] * 3)
    target_affine[:3, 3] += offset

    return nibp.resample_from_to(
        img,
        to_vox_map=(tuple(target_shape), target_affine),
        order=order,
    )


def nibabel_to_pyvista(nib_img, scalar_name="data"):
    """
    Converts a nibabel image to a pyvista ImageData, with the image values
    as cell data scalar_name.
    """
    data = nib_img.get_fdata()
    if data.ndim > 3:
        data = data[..., 0]

    affine = nib_img.affine
    spacing = np.array(nib_img.header.get_zooms()[:3])

    grid = pv.ImageData()
    grid.dimensions = np.array(data.shape) + 1
    grid.spacing = spacing

    try:
        grid.direction_matrix = affine[:3, :3] / spacing
        grid.origin = affine[:3, 3] - (grid.direction_matrix @ spacing) / 2.0
    except AttributeError:
        grid.origin = affine[:3, 3] - spacing / 2.0

    grid.cell_data[scalar_name] = data.flatten(order="F")
    return grid
