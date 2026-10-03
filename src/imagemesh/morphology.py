"""
Operations on label images (3D integer arrays, 0 = background): morphology
on all or selected labels and voxel-level helpers (numba).
"""

import cc3d
import fastremap
import nbmorph
import numpy as np
from nbmorph import dilate_labels_spherical
from numba import njit, prange
from scipy.ndimage import zoom


def mergecells(img, labels):
    print(f"merging cells: {labels},  ({img.shape})")
    img = np.where(np.isin(img, labels), labels[0], img)
    return img


def ncells(img, ncells, keep_cell_labels=None):
    cell_labels, cell_counts = fastremap.unique(img, return_counts=True)
    cell_labels = cell_labels[np.argsort(cell_counts)][::-1]
    if keep_cell_labels is None:
        cois = set()
    else:
        cois = set(keep_cell_labels)
    for cid in cell_labels:
        if len(cois) >= ncells:
            break
        cois.add(cid)
    img = np.where(np.isin(img, list(cois)), img, 0)
    return img


def dilate(img, radius, labels=None):
    print(f"dilating cells,  ({img.shape})")
    if labels is None:
        img = nbmorph.dilate_labels_spherical(img, radius=radius)
    else:
        vipimg = np.where(np.isin(img, labels), img, 0)
        vipimg = dilate(vipimg, radius=radius)
        img = np.where(vipimg, vipimg, img)
    return img


def erode(img, radius, labels=None, struct_sequence=None):
    print(f"eroding cells,  ({img.shape})")
    if struct_sequence is None:
        # a single diamond step leaves cells touching at voxel corners
        struct_sequence = "B" if radius == 1 else "DDB"
    if labels is None:
        img = nbmorph.erode_labels_spherical(img, radius=radius, struct_sequence=struct_sequence)
    else:
        vipimg = np.where(np.isin(img, labels), img, 0)
        vipimg = erode(vipimg, radius=radius, struct_sequence=struct_sequence)
        orig_wo_vips = np.where(np.isin(img, labels), 0, img)
        img = np.where(orig_wo_vips > vipimg, orig_wo_vips, vipimg)
    return img


def mode(img, iterations=1):
    print(f"mode,  ({img.shape})")
    for i in range(iterations):
        img = nbmorph.mode_box(img)
    return img


def zero_edges(img):
    print(f"zero_edges,  ({img.shape})")
    img = nbmorph.zero_label_edges_box(img)
    return img


def upsample(img, factor):
    return zoom(img, factor, order=0)


def smooth(img, iterations, radius, labels=None):
    print(f"smoothing cells,  ({img.shape})")
    if labels is None:
        img = nbmorph.smooth_labels_spherical(
            img, radius=radius, iterations=iterations, dilate_radius=radius
        )
    else:
        vipimg = np.where(np.isin(img, labels), img, 0)
        vipimg = smooth(vipimg, iterations=iterations, radius=radius)
        # remove labelled cells from original image
        orig_wo_vips = np.where(np.isin(img, labels), 0, img)
        # insert smoothed labeled cells in original (overwrite original)
        img = np.where(vipimg, vipimg, orig_wo_vips)
    return img


def removeislands(img, minsize):
    return cc3d.dust(img, threshold=minsize, connectivity=6)


@njit(parallel=True, cache=True)
def set_mask_scalar(arr, mask, value):
    """Sets a scalar value wherever the 3D mask is True (Multi-threaded)."""
    # prange distributes the 2D slices (the 'i' dimension) across CPU cores
    for i in prange(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                if mask[i, j, k]:
                    arr[i, j, k] = value
    return arr


@njit(parallel=True, cache=True)
def copy_mask(dest, mask, src):
    """Copies values from src to dest wherever the 3D mask is True (Multi-threaded)."""
    for i in prange(dest.shape[0]):
        for j in range(dest.shape[1]):
            for k in range(dest.shape[2]):
                if mask[i, j, k]:
                    dest[i, j, k] = src[i, j, k]
    return dest


@njit(cache=True)
def binary_fill_holes(img):
    """
    Fills holes in a 3D binary volume using a stack-based flood fill with 6-connectivity.
    """
    depth, rows, cols = img.shape
    external_bg = np.zeros_like(img, dtype=np.bool_)

    # Cache optimization: Single (N, 3) matrix using int16
    # Halves memory usage and guarantees contiguous CPU cache hits
    max_pixels = depth * rows * cols
    stack = np.empty((max_pixels, 3), dtype=np.int16)
    top = 0

    # --- Initialization ---
    for r in range(rows):
        for c in range(cols):
            if not img[0, r, c]:
                stack[top, 0], stack[top, 1], stack[top, 2] = 0, r, c
                external_bg[0, r, c] = True
                top += 1
            if depth > 1 and not img[depth - 1, r, c]:
                stack[top, 0], stack[top, 1], stack[top, 2] = depth - 1, r, c
                external_bg[depth - 1, r, c] = True
                top += 1

    for d in range(1, depth - 1):
        for c in range(cols):
            if not img[d, 0, c]:
                stack[top, 0], stack[top, 1], stack[top, 2] = d, 0, c
                external_bg[d, 0, c] = True
                top += 1
            if rows > 1 and not img[d, rows - 1, c]:
                stack[top, 0], stack[top, 1], stack[top, 2] = d, rows - 1, c
                external_bg[d, rows - 1, c] = True
                top += 1

    for d in range(1, depth - 1):
        for r in range(1, rows - 1):
            if not img[d, r, 0]:
                stack[top, 0], stack[top, 1], stack[top, 2] = d, r, 0
                external_bg[d, r, 0] = True
                top += 1
            if cols > 1 and not img[d, r, cols - 1]:
                stack[top, 0], stack[top, 1], stack[top, 2] = d, r, cols - 1
                external_bg[d, r, cols - 1] = True
                top += 1

    # --- Flood Fill (Unrolled for pure speed) ---
    while top > 0:
        top -= 1
        d = stack[top, 0]
        r = stack[top, 1]
        c = stack[top, 2]

        # Unrolled 6-connectivity checks
        # d - 1
        if d > 0 and not img[d - 1, r, c] and not external_bg[d - 1, r, c]:
            external_bg[d - 1, r, c] = True
            stack[top, 0], stack[top, 1], stack[top, 2] = d - 1, r, c
            top += 1
        # d + 1
        if d < depth - 1 and not img[d + 1, r, c] and not external_bg[d + 1, r, c]:
            external_bg[d + 1, r, c] = True
            stack[top, 0], stack[top, 1], stack[top, 2] = d + 1, r, c
            top += 1
        # r - 1
        if r > 0 and not img[d, r - 1, c] and not external_bg[d, r - 1, c]:
            external_bg[d, r - 1, c] = True
            stack[top, 0], stack[top, 1], stack[top, 2] = d, r - 1, c
            top += 1
        # r + 1
        if r < rows - 1 and not img[d, r + 1, c] and not external_bg[d, r + 1, c]:
            external_bg[d, r + 1, c] = True
            stack[top, 0], stack[top, 1], stack[top, 2] = d, r + 1, c
            top += 1
        # c - 1
        if c > 0 and not img[d, r, c - 1] and not external_bg[d, r, c - 1]:
            external_bg[d, r, c - 1] = True
            stack[top, 0], stack[top, 1], stack[top, 2] = d, r, c - 1
            top += 1
        # c + 1
        if c < cols - 1 and not img[d, r, c + 1] and not external_bg[d, r, c + 1]:
            external_bg[d, r, c + 1] = True
            stack[top, 0], stack[top, 1], stack[top, 2] = d, r, c + 1
            top += 1

    return ~external_bg


@njit(parallel=True, cache=True)
def fill_from_neighbors(data, mask, neighbors, max_radius=100):
    if mask.sum() == 0:
        return data
    nb_mask = data.copy()
    set_mask_scalar(nb_mask, ~np.isin(data, neighbors) | mask, np.uint8(0))

    num_empty_voxels = ((nb_mask == 0) & mask).sum()
    for _ in range(max_radius):
        nb_mask = dilate_labels_spherical(nb_mask, radius=1)
        num_empty_voxels = ((nb_mask == 0) & mask).sum()
        if num_empty_voxels == 0:
            break

    copy_mask(data, mask, nb_mask)
    return data


@njit(cache=True, parallel=True)
def separate_labels(data, l1, l2, dist, newlabel, except_labels=None, except_region=None):
    m1 = nbmorph.dilate_labels_spherical(np.isin(data, l1), dist)
    m2 = nbmorph.dilate_labels_spherical(np.isin(data, l2), dist)
    if except_region is None:
        except_region = np.zeros(shape=data.shape, dtype=np.bool_)
    if except_labels is None:
        return set_mask_scalar(data, m1 & m2 & ~except_region, newlabel)
    return set_mask_scalar(data, m1 & m2 & ~except_region & ~np.isin(data, except_labels), newlabel)


@njit(cache=True, parallel=True)
def enforce_min_thickness(data, label, radius, s="B"):
    mask = data == label
    opened = nbmorph.erode_labels_spherical(mask, radius=radius, struct_sequence=s)
    opened = nbmorph.dilate_labels_spherical(opened, radius=radius, struct_sequence=s)
    diff = mask ^ opened
    dil = nbmorph.dilate_labels_spherical(diff, radius=radius, struct_sequence=s)
    set_mask_scalar(data, dil, label)
    return data


def get_lowest_point(mask):
    idx = np.argwhere(mask)
    ind = np.argsort(idx[:, 2])
    return idx[ind[0]]


@njit(cache=True, parallel=True)
def grow_into_region(seed_labels, region_mask, radius=1):
    grown_labels = np.copy(seed_labels)
    set_mask_scalar(grown_labels, ~region_mask, 0)
    n_voxels = (grown_labels > 0).sum()
    while True:
        grown_labels = dilate_labels_spherical(grown_labels, radius=radius, struct_sequence="D")
        set_mask_scalar(grown_labels, ~region_mask, 0)
        new_voxels = (grown_labels > 0).sum() - n_voxels
        if new_voxels == 0:
            break
        n_voxels += new_voxels
    return grown_labels
