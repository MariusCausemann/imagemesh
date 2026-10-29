"""
Synthetic cases where surface_quads and contour_labels differ: the voxels,
contour_labels as it is, with close_quad_holes, and surface_quads, each after
a little smoothing.

    python plot_synth.py OUT.png
"""

import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

from imagemesh.image import np2pv  # noqa: E402
from imagemesh.smoothing import smooth_surface_net  # noqa: E402
from imagemesh.surface import close_quad_holes, surface_quads  # noqa: E402

warnings.simplefilter("ignore")
ITER = 4
LABEL_COLOUR = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
LIGHT = np.array([-0.4, -0.6, 0.7]) / np.linalg.norm([-0.4, -0.6, 0.7])


def shaded(polys, colours):
    """Mild shading: at most 20% darker than the label colour, so hues stay
    recognisable across panels (matplotlib's own shading turns them muddy)."""
    out = []
    for poly, c in zip(polys, colours):
        poly = np.asarray(poly)
        n = np.cross(poly[1] - poly[0], poly[2] - poly[0])
        n /= np.linalg.norm(n) + 1e-30
        out.append(np.array(matplotlib.colors.to_rgb(c)) * (0.8 + 0.2 * abs(n @ LIGHT)))
    return out


VIEW = (24, -52)
RED, ORANGE, VIOLET, INK, MUTED = "#e34948", "#eda100", "#6250d6", "#1f1f1e", "#8a8a85"


def flap_case():
    img = np.full((3, 3, 2), 2)
    img[0, 0, 0] = img[1, 0, 0] = img[0, 1, 0] = 1  # an L in the bottom layer
    img[1, 1, 1] = 1  # above its inner corner: touches two of them along edges, one at a corner
    return (
        "Flap: VTK leaves a slit, its cap duplicates a triangle",
        img,
        1,
        "Three voxels of 1 form an L; a fourth sits above its inner corner. contour_labels "
        "splits that corner\n"
        "inconsistently and leaves a slit (black); close_quad_holes caps it with a zero-area "
        "quad (violet)\n"
        "whose triangulation repeats a neighbour (red). surface_quads has no slit; here label 2 "
        "keeps its\n"
        "diagonal at the edge under the fourth voxel, so 1 is two pieces touching at a corner.",
        None,
        2,
        None,
    )


def checkerboard_case():
    core = np.array([[[3, 2], [2, 1]], [[2, 1], [1, 3]]])  # 1 at the voxels nearest the camera
    img = np.pad(core, 1, constant_values=3)  # inside label 3, away from the image border
    return (
        "Connectivity: a diagonal pair between two other labels has to connect",
        img,
        1,
        "A 2x2x2 checkerboard of single voxels inside label 3 (green); every panel is the "
        "horizontal plane z = 2.15,\n"
        "just above its centre corner, seen from above. The two voxels of 1 in this layer touch "
        "only along the vertical\n"
        "edge through the centre, between a 2 and a 3 (the layer below: the same with 1 and 2 "
        "swapped, turned 180\u00b0).\n"
        "contour_labels puts one vertex at the centre for all labels: 1 is pinched there (red) "
        "and every voxel is pulled\n"
        "into it. surface_quads gives each interface its own copy: 1 passes between 2 and 3 and "
        "stays one piece.",
        ((1, 1, 1), (3, 3, 3)),
        3,
        2.15,
    )


def pinch_case():
    img = np.full((2, 2, 3), 4)
    img[0, 0, 1] = img[1, 1, 1] = 1
    img[1, 0, 1], img[0, 1, 1] = 2, 3
    return (
        "Remaining pinch: neck of 1 between 2 and 3, label 4 below and above",
        img,
        1,
        "The neck of 1 would need a vertex in the middle of the edge (a lens); a single voxel "
        "edge has\n"
        "none, and both of its corners are held together by label 4's floor / ceiling. Same in "
        "all three.",
        ((0, 0, 1), (2, 2, 2)),
        4,
        None,
    )


def surfaces(img):
    grid = np2pv(img.astype(np.uint32), (1.0,) * 3)
    raw = grid.contour_labels(
        "all",
        smoothing=False,
        output_mesh_type="quads",
        background_value=0,
        scalars="data",
    )
    capped = close_quad_holes(raw)
    ncap = capped.n_cells - raw.n_cells
    out = []
    for q, nc in ((raw, 0), (capped, ncap), (surface_quads(grid), 0)):
        s = smooth_surface_net(q, iterations=ITER, distance=1.0, scale=1.2, fix_bounds=True)
        cap = np.zeros(s.n_cells, bool)
        if nc:
            cap[-2 * nc :] = True
        out.append((s, cap, nc))
    return out


def label_stats(tris, label_rows):
    """pieces (faces joined across edges), open and pinched edges of one label's faces"""
    t = tris[label_rows]
    e = np.sort(np.stack([t, np.roll(t, -1, axis=1)], -1).reshape(-1, 2), axis=1)
    key, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    face = np.repeat(np.arange(len(t)), 3)
    # faces sharing a manifold edge are one piece (a pinched edge does not join)
    order = np.argsort(inv, kind="stable")
    a, b = [], []
    for k in np.where(cnt == 2)[0]:
        f0, f1 = face[order][np.searchsorted(inv[order], k) : np.searchsorted(inv[order], k) + 2]
        a.append(f0)
        b.append(f1)
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(t), len(t)))
    pieces, piece = connected_components(g, directed=False)
    return pieces, piece, key[cnt == 1], key[cnt > 2]


CUBE = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float)
CUBE_FACES = [
    [0, 1, 3, 2],
    [4, 5, 7, 6],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [0, 2, 6, 4],
    [1, 3, 7, 5],
]
CUBE_EDGES = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [0, 2],
    [1, 3],
    [4, 6],
    [5, 7],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
]


def in_box(img, box):
    lo, hi = box
    return [ijk for ijk in np.ndindex(img.shape) if all(lo[a] <= ijk[a] < hi[a] for a in range(3))]


def draw_voxels(ax, img, box, fill):
    """Every voxel in the box as a cube in its label's colour, except the fill
    label (the one surrounding everything), which would only hide the others.
    Same box and view as the surface panels."""
    polys, colours = [], []
    for ijk in in_box(img, box):
        lab = int(img[ijk])
        if lab == fill:
            continue
        c = np.array(ijk) + 0.5 + 0.94 * (CUBE - 0.5)
        for q in CUBE_FACES:
            polys.append(c[q])
            colours.append(LABEL_COLOUR[lab])
    ax.add_collection3d(
        Poly3DCollection(
            polys,
            facecolors=shaded(polys, colours),
            edgecolors=(0.1, 0.1, 0.1, 0.7),
            linewidths=0.5,
        )
    )
    setup(ax, *box)
    # orientation triad in the panel's bottom-left corner, same view
    x0, y0, w, h = ax.get_position().bounds
    tri = ax.figure.add_axes((x0 - 0.01, y0 + 0.02 * h, 0.28 * w, 0.28 * h), projection="3d")
    for a, name in enumerate("xyz"):
        e = np.zeros(3)
        e[a] = 1
        tri.quiver(0, 0, 0, *e, color=INK, lw=1, arrow_length_ratio=0.25)
        tri.text(*(1.3 * e), name, color=INK, fontsize=9, ha="center", va="center")
    tri.set_xlim(-0.2, 1.2)
    tri.set_ylim(-0.2, 1.2)
    tri.set_zlim(-0.2, 1.2)
    tri.set_box_aspect((1, 1, 1))
    tri.view_init(*VIEW)
    tri.set_axis_off()
    tri.patch.set_alpha(0)
    tri.set_facecolor((1, 1, 1, 0))
    labs = [
        int(lab) for lab in np.unique(img[tuple(slice(a, b) for a, b in zip(*box))]) if lab != fill
    ]
    labs.append(fill)
    for k, lab in enumerate(labs):
        ax.text2D(
            0.02 + 0.11 * k,
            0.0,
            f"\u25a0 {lab}",
            transform=ax.transAxes,
            fontsize=9,
            color=LABEL_COLOUR[lab],
            va="top",
        )
    ax.text2D(
        0.02,
        -0.08,
        f"label {fill} fills the rest (no cubes drawn)",
        transform=ax.transAxes,
        fontsize=8.5,
        color=MUTED,
        va="top",
    )


def voxel_outlines(ax, img, label, box):
    """The named label's voxel edges (each once), to relate the surface to the voxels."""
    segs = set()
    for ijk in in_box(img, box):
        if img[ijk] == label:
            c = np.array(ijk) + CUBE
            segs |= {tuple(sorted((tuple(c[a]), tuple(c[b])))) for a, b in CUBE_EDGES}
    ax.add_collection3d(
        Line3DCollection(
            [np.array(e) for e in sorted(segs)],
            colors=(0.35, 0.35, 0.35, 0.6),
            linewidths=0.6,
        )
    )


def setup(ax, lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2 + 0.15
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1), zoom=1.12)
    ax.view_init(*VIEW)
    ax.computed_zorder = False  # draw order: surface, then voxel outlines, then defects on top
    ax.set_axis_off()


def draw_surface(ax, surf, cap, label, img, box):
    P = np.asarray(surf.points)
    T = surf.regular_faces
    bl = np.asarray(surf["boundary_labels"]).astype(int)
    rows = (bl == label).any(axis=1)
    lo, hi = np.asarray(box[0]), np.asarray(box[1])
    cen = P[T].mean(axis=1)
    shown = rows & np.all((cen >= lo - 1e-9) & (cen <= hi + 1e-9), axis=1)
    pieces, piece, opened, pinched = label_stats(T, rows)
    # each piece (faces joined across manifold edges) in its own shade of the label colour
    rid = np.where(rows)[0]
    rowpiece = np.full(len(T), -1)
    rowpiece[rid] = piece
    base = np.array(matplotlib.colors.to_rgb(LABEL_COLOUR[label]))
    shades = [base, 0.45 * base + 0.55, 0.55 * base, 0.7 * base + 0.3, 0.75 * base]
    polys = P[T[shown]]
    fc = [VIOLET if cap[i] else shades[rowpiece[i] % len(shades)] for i in np.where(shown)[0]]
    ax.add_collection3d(
        Poly3DCollection(
            polys, facecolors=shaded(polys, fc), edgecolors=(0.1, 0.1, 0.1, 0.35), linewidths=0.3
        )
    )
    voxel_outlines(ax, img, label, box)
    s = np.sort(T, axis=1)
    _, inv, cnt = np.unique(s, axis=0, return_inverse=True, return_counts=True)
    dup = np.where((cnt[inv.ravel()] > 1) & rows)[0]
    shown_dup = np.where((cnt[inv.ravel()] > 1) & shown)[0]
    if len(shown_dup):
        ax.add_collection3d(Poly3DCollection(P[T[shown_dup]], facecolors=RED, edgecolors=RED))
    for edges, col in ((opened, INK), (pinched, RED)):
        if len(edges):
            ax.add_collection3d(Line3DCollection(P[edges], colors=col, linewidths=3))
    setup(ax, lo, hi)
    return pieces, len(opened), len(pinched), len(dup) // 2


def slice_regions(surf, z, lo, hi, n=400):
    """Label per pixel of the plane z: crossing parity against each label's cut
    segments (each label's faces are a closed surface, so its cut is closed)."""
    sl = surf.slice(normal="z", origin=(0, 0, z), generate_triangles=True)
    P = np.asarray(sl.points)[:, :2]
    seg = sl.lines.reshape(-1, 3)[:, 1:]
    pairs = np.asarray(sl["boundary_labels"]).astype(int)
    X, Y = np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n))
    out = np.zeros(X.shape, int)
    for lab in np.unique(pairs):
        par = np.zeros(X.shape, bool)
        for a, b in P[seg[(pairs == lab).any(axis=1)]]:
            if a[1] != b[1]:
                par ^= ((a[1] > Y) != (b[1] > Y)) & (
                    X < a[0] + (Y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                )
        out[par & (out == 0)] = lab
    return out, P[seg]


PLANE = (0.6, 3.4)  # x and y range of the plane panels


def plane_frame(ax):
    """voxel grid, x / y arrows, limits of a plane panel"""
    lo, hi = PLANE
    for t in range(int(np.ceil(lo)), int(hi) + 1):
        ax.axhline(t, color=(1, 1, 1, 0.7), lw=0.6, zorder=1)
        ax.axvline(t, color=(1, 1, 1, 0.7), lw=0.6, zorder=1)
    kw = dict(
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=9,
        color=INK,
        arrowprops=dict(arrowstyle="->", color=INK, lw=0.8),
    )
    ax.annotate("x", xy=(0.2, -0.04), xytext=(0.0, -0.04), va="center", **kw)
    ax.annotate("y", xy=(-0.04, 0.2), xytext=(-0.04, 0.0), ha="center", **kw)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_axis_off()


def draw_voxel_plane(ax, img, z):
    """the labels of the voxel layer containing the plane z"""
    lo, hi = PLANE
    n = img.shape[0]
    layer = img[:, :, int(z)]
    rgb = np.array([[matplotlib.colors.to_rgb(LABEL_COLOUR[int(v)]) for v in row] for row in layer])
    ax.imshow(
        rgb.transpose(1, 0, 2),
        origin="lower",
        extent=(0, n, 0, img.shape[1]),
        interpolation="nearest",
    )
    for (i, j), v in np.ndenumerate(layer):
        if lo < i + 0.5 < hi and lo < j + 0.5 < hi:
            ax.text(
                i + 0.5,
                j + 0.5,
                str(v),
                ha="center",
                va="center",
                color="white",
                fontsize=14,
                weight="bold",
            )
    plane_frame(ax)


def draw_surface_plane(ax, surf, z, label):
    """The smoothed surfaces cut by the plane z, each region filled in its
    label's colour; red dots: where the label's pinched edges cross the plane.
    Returns the label's pieces and pinched edges."""
    lo, hi = PLANE
    lab, segs = slice_regions(surf, z, (lo, lo), (hi, hi), n=500)
    rgb = np.ones(lab.shape + (3,))
    for k, col in LABEL_COLOUR.items():
        rgb[lab == k] = matplotlib.colors.to_rgb(col)
    ax.imshow(rgb, origin="lower", extent=(lo, hi, lo, hi), interpolation="nearest", zorder=0)
    ax.add_collection(
        matplotlib.collections.LineCollection(segs, colors=INK, linewidths=1.0, zorder=3)
    )
    T = surf.regular_faces
    rows = (np.asarray(surf["boundary_labels"]).astype(int) == label).any(axis=1)
    P = np.asarray(surf.points)
    pieces, _, _, pinched = label_stats(T, rows)
    for a, b in P[pinched]:
        if (a[2] - z) * (b[2] - z) < 0:
            p = a + (z - a[2]) / (b[2] - a[2]) * (b - a)
            ax.scatter([p[0]], [p[1]], s=60, color=RED, edgecolor="white", lw=1.5, zorder=5)
    plane_frame(ax)
    return pieces, len(pinched)


def main(out):
    cases = [flap_case(), checkerboard_case(), pinch_case()]
    cols = [
        "voxels",
        "contour_labels",
        "contour_labels + close_quad_holes",
        "surface_quads",
    ]
    # explicit layout in inches: header (title + note), panel titles, panels, panel captions
    W, PANEL, CAPTION, GAP = 13.0, 2.9, 1.0, 0.2
    heads = [0.55 + 0.17 * (note.count("\n") + 1) for _, _, _, note, _, _, _ in cases]
    views = [(24, -52), (30, 45), (24, -52)]
    H = sum(heads) + len(cases) * (PANEL + CAPTION + GAP) + 0.45
    fig = plt.figure(figsize=(W, H))
    y = H
    global VIEW
    for (name, img, label, note, box, fill, plane), head, VIEW in zip(cases, heads, views):
        fig.text(0.01, (y - 0.1) / H, name, fontsize=11, color=INK, va="top", weight="bold")
        fig.text(0.01, (y - 0.33) / H, note, fontsize=8.5, color=MUTED, va="top")
        y -= head + PANEL
        surfs = surfaces(img)
        if box is None:  # the named label's voxels
            ijk = np.argwhere(img == label)
            box = (ijk.min(axis=0), ijk.max(axis=0) + 1)
        for j in range(4):
            if plane is not None:  # the plane z = plane, from above
                side = PANEL - 0.25
                ax = fig.add_axes((0.005 + j / 4 + 0.02, (y + 0.1) / H, side / W, side / H))
                ax.set_title(cols[j], fontsize=9, color=INK, loc="left", pad=4)
                if j == 0:
                    draw_voxel_plane(ax, img, plane)
                    ax.text(
                        0.0,
                        -0.09,
                        f"the voxel layer {int(plane)} \u2264 z \u2264 {int(plane) + 1}",
                        transform=ax.transAxes,
                        fontsize=8.5,
                        color=MUTED,
                        va="top",
                    )
                    continue
                pieces, npinch = draw_surface_plane(ax, surfs[j - 1][0], plane, label)
                lines = [
                    (
                        MUTED,
                        f"label {label} in 3D: {pieces} piece{'s' * (pieces != 1)}, "
                        + (f"{npinch} pinched edges" if npinch else "no pinched edges"),
                    )
                ]
                if npinch:
                    lines.append((RED, "\u25cf the pinched vertical edge: 1 touches itself"))
                for k, (col, text) in enumerate(lines):
                    ax.text(
                        0.0,
                        -0.09 - 0.075 * k,
                        text,
                        transform=ax.transAxes,
                        fontsize=8.5,
                        color=col,
                        va="top",
                    )
                continue
            ax = fig.add_axes((0.005 + j / 4, y / H, 0.24, PANEL / H), projection="3d")
            ax.set_title(cols[j], fontsize=9, color=INK, loc="left", pad=2)
            if j == 0:
                draw_voxels(ax, img, box, fill)
                continue
            s, cap, ncap = surfs[j - 1]
            pieces, nopen, npinch, ndup = draw_surface(ax, s, cap, label, img, box)
            lines = [
                (
                    MUTED,
                    f"label {label}: {pieces} piece{'s' * (pieces != 1)}"
                    + (" (one shade each)" if pieces > 1 else ""),
                )
            ]
            for n, col, what in (
                (ncap, VIOLET, "\u25a0 {} cap quad{}: zero area, close the slit"),
                (nopen, INK, "\u2501 {} open edge{}: rim of a slit (hole)"),
                (
                    npinch,
                    RED,
                    "\u2501 {} pinched edge{}: 4 faces of the label share it",
                ),
                (ndup, RED, "\u25a0 {} triangle{} stored twice, opposite orientation"),
            ):
                if n:
                    lines.append((col, what.format(n, "s" * (n != 1))))
            for k, (col, text) in enumerate(lines):
                ax.text2D(
                    0.02,
                    -0.075 * k,
                    text,
                    transform=ax.transAxes,
                    fontsize=8.5,
                    color=col,
                    va="top",
                )
        y -= CAPTION + GAP
    fig.text(
        0.01,
        0.05 / H,
        f"Surfaces: the named label's faces after {ITER} smoothing iterations, at their real "
        "positions; grey lines: its voxels. A piece = faces joined across manifold edges; pieces "
        "that touch only "
        "at a pinched edge or a corner are separate pieces.",
        fontsize=8,
        color=MUTED,
        va="bottom",
    )
    fig.savefig(out, dpi=120)


if __name__ == "__main__":
    main(sys.argv[1])
