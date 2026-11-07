import numpy as np
from scipy import ndimage

from imagemesh.reconnect import reconnect

FACE = ndimage.generate_binary_structure(3, 1)


def pieces(img, label):
    return ndimage.label(img == label, FACE)[1]


def dumbbell(n=60):
    """Two boxes joined by a 2 x 2 neck along z, cut at z = 30."""
    img = np.zeros((n, n, n), np.int32)
    img[10:30, 10:30, 5:22] = 1
    img[10:30, 10:30, 38:55] = 1
    img[19:21, 19:21, 22:38] = 1
    cut = img.copy()
    cut[:, :, 29:31] = 0
    return img, cut


def test_reconnect_restores_cut_neck():
    img, cut = dumbbell()
    assert pieces(cut, 1) == 2
    out = reconnect(img, cut, s=20, t=1)
    assert pieces(out, 1) == 1
    # only voxels of the cell in the input are restored
    changed = out != cut
    assert (img[changed] == 1).all()
    assert changed.sum() <= 2 * 2 * 2 * 9


def test_reconnect_keeps_small_loop_cut():
    """A ring cut at one place stays connected around the ring: not a cut."""
    img = np.zeros((40, 40, 12), np.int32)
    img[10:22, 10:22, 4:8] = 1
    img[13:19, 13:19, 4:8] = 0
    cut = img.copy()
    cut[15:17, 10:13, 4:8] = 0
    out = reconnect(img, cut, s=20, t=1)
    assert (out == cut).all()


def test_reconnect_block_with_margin():
    rng = np.random.default_rng(0)
    img = ndimage.zoom(rng.integers(1, 6, (12, 12, 12)), 6, order=0).astype(np.int32)
    img[rng.random(img.shape) < 0.02] = 0
    cut = img.copy()
    cut[(np.arange(72) % 9 == 0)[:, None, None] & (rng.random(img.shape) < 0.6)] = 0
    full = reconnect(img, cut, s=8, t=1)
    assert (full != cut).any()
    lo, hi, m = np.array([17, 5, 23]), np.array([60, 50, 66]), 20
    a, b = np.maximum(lo - m, 0), np.minimum(hi + m, img.shape)
    blk = tuple(slice(i, j) for i, j in zip(a, b))
    part = reconnect(img[blk], cut[blk], s=8, t=1, origin=a)
    core = tuple(slice(i - j, k - j) for i, j, k in zip(lo, a, hi))
    ref = full[tuple(slice(i, k) for i, k in zip(lo, hi))]
    assert (part[core] == ref).all()
