"""Centerlines with radius by TubeTK ridge traversal on the consensus probability.

Alternative to extract_centerlines.py (VMTK on the binary mask). The binary mask
joins neighbouring vessels through thin low-probability necks; ridge traversal
follows the vessel direction in the grey-level consensus instead, so it only
crosses such a neck if the ridge itself turns into it.

1. SegmentTubes on the whole consensus volume (T1w grid), seeded from the voxel
   skeleton of network.nii.gz in order of decreasing probability. Each seed that
   is not already inside an extracted tube is traversed both ways until the ridge
   fails (roundness / ridgeness / revisit); the radius is optimised per point.
2. Tube tree (the Python wheel lacks ConvertTubesToTubeTree, so done here): each
   tube end is joined to the nearest point of another tube if it is within
   max(--join_min mm, --join_ratio x radius), and (if --link_prob > 0) the
   straight segment between them never drops below --link_prob consensus.

Output (results/sub-01/network/), scanner (T1w RAS) coordinates in mm:
  tubetk.tre               raw TubeTK tubes (voxel-frame mm, MetaIO text)
  tubetk_centerlines.vtp   polylines (one cell per branch between junctions);
                           point data: radius, ridgeness, intensity;
                           cell data: branch, length, mean_radius
  tubetk_tubes.vtp         tube around each branch with the varying radius
  tubetk.json              summary incl. loops (independent cycles), components
                           and coverage of the network skeleton

Run in the tubetk env:  pixi run -e tubetk python extract_tubetk.py
"""

import argparse
import json
import time

import itk
import networkx as nx
import nibabel as nib
import numpy as np
import pyvista as pv
from common import RESULTS
from itk import TubeTK as ttk
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

OUT = RESULTS / "network"


def to_itk(a, h):
    """numpy (i, j, k) -> itk image whose index (x, y, z) is (i, j, k)."""
    im = itk.image_view_from_array(np.ascontiguousarray(a.transpose(2, 1, 0)))
    im.SetSpacing([h] * 3)
    return im


def segment_tubes(cons, seeds, h, args, tre):
    img, seed_img = to_itk(cons, h), to_itk(seeds, h)
    s = ttk.SegmentTubes[itk.Image[itk.F, 3]].New()
    s.SetInput(img)
    s.SetDataMinMaxLimits(0.0, 1.0)
    s.SetRadiusInObjectSpace(args.radius_start)
    s.SetMinRoundness(args.min_roundness)
    s.SetMinCurvature(args.min_curvature)
    s.SetMinRidgeness(args.min_ridgeness)
    s.SetMinLength(round(args.min_length / args.step_x))  # in points
    s.SetVerbose(False)
    s.SetSeedMask(seed_img)
    s.SetUseSeedMaskAsProbabilities(True)
    # zero voxels would otherwise be seeds too (every voxel in the volume)
    s.SetSeedExtractionMinimumProbability(1e-3)
    s.ProcessSeeds()
    w = itk.SpatialObjectWriter[3].New()
    w.SetInput(s.GetTubeGroup())
    w.SetFileName(str(tre))
    w.Update()
    return read_tre(tre)


def read_tre(path):
    """Tubes from a MetaIO .tre: list of dicts of per-point arrays."""
    tubes, lines, i = [], path.read_text().splitlines(), 0
    while i < len(lines):
        if lines[i].startswith("PointDim"):
            cols = lines[i].split("=")[1].split()
            n = int(lines[i + 1].split("=")[1])
            a = np.array([ln.split() for ln in lines[i + 3 : i + 3 + n]], float)
            a = a[np.argsort(a[:, cols.index("id")])]
            xyz = a[:, [cols.index(c) for c in "xyz"]]
            tubes.append(
                dict(
                    xyz=xyz,
                    r=a[:, cols.index("r")],
                    ridgeness=a[:, cols.index("rn")],
                    intensity=a[:, cols.index("in")],
                )
            )
            i += 3 + n
        else:
            i += 1
    return tubes


def subsample(t, step):
    """Keep points about `step` mm apart along the tube (ends kept)."""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(t["xyz"], axis=0), axis=1))]
    keep = np.unique(np.r_[np.searchsorted(s, np.arange(0, s[-1], step)), len(s) - 1])
    return {k: v[keep] for k, v in t.items()}


def min_along(vol, a, b, h, n_per_mm=8):
    """Minimum of `vol` (trilinear) on the segment a-b (voxel-frame mm)."""
    n = max(int(np.linalg.norm(b - a) * n_per_mm), 2)
    p = np.linspace(a, b, n) / h
    return float(ndimage.map_coordinates(vol, p.T, order=1).min())


def build_graph(tubes, cons, h, args):
    """Tube points as graph nodes; consecutive points + end-to-tube joins as edges."""
    xyz = np.vstack([t["xyz"] for t in tubes])
    tid = np.concatenate([np.full(len(t["xyz"]), k) for k, t in enumerate(tubes)])
    r = np.concatenate([t["r"] for t in tubes])
    start = np.r_[0, np.cumsum([len(t["xyz"]) for t in tubes])]
    G = nx.Graph()
    G.add_nodes_from(range(len(xyz)))
    for k in range(len(tubes)):
        G.add_edges_from((j, j + 1) for j in range(start[k], start[k + 1] - 1))
    tree = cKDTree(xyz)
    n_join, n_rejected = 0, 0
    for k in range(len(tubes)):
        for e in (start[k], start[k + 1] - 1):
            reach = max(args.join_min, args.join_ratio * r[e])
            cand = [j for j in tree.query_ball_point(xyz[e], reach) if tid[j] != k]
            if not cand:
                continue
            j = min(cand, key=lambda j: np.linalg.norm(xyz[j] - xyz[e]))
            if (
                args.link_prob > 0
                and min_along(cons, xyz[e], xyz[j], h) < args.link_prob
            ):
                n_rejected += 1
                continue
            G.add_edge(e, j)
            n_join += 1
    return G, xyz, r, n_join, n_rejected


def split_branches(G):
    """Polylines (node ids) between nodes of degree != 2 (plus pure cycles)."""
    nodes = {n for n in G if G.degree(n) != 2}
    seen, out = set(), []
    for a in nodes:
        for b in G.neighbors(a):
            if (a, b) in seen:
                continue
            path, prev, cur = [a, b], a, b
            while cur not in nodes:
                prev, cur = cur, next(n for n in G.neighbors(cur) if n != prev)
                path.append(cur)
            seen.add((path[-1], path[-2]))
            seen.add((a, b))
            out.append(np.array(path))
    for comp in nx.connected_components(G):
        if not comp & nodes and len(comp) > 2:  # isolated loop
            out.append(np.array(nx.cycle_basis(G.subgraph(comp))[0]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sigma", type=float, default=0.0, help="Gaussian, voxels")
    p.add_argument("--radius_start", type=float, default=0.5, help="mm")
    p.add_argument("--min_roundness", type=float, default=0.15)
    p.add_argument("--min_curvature", type=float, default=0.005)
    p.add_argument("--min_ridgeness", type=float, default=0.85)
    p.add_argument("--min_length", type=float, default=2.0, help="mm")
    p.add_argument("--step_x", type=float, default=0.1, help="TubeTK step, mm")
    p.add_argument("--step", type=float, default=0.25, help="resampling, mm")
    p.add_argument("--join_min", type=float, default=1.0, help="mm")
    p.add_argument("--join_ratio", type=float, default=2.0, help="x radius")
    p.add_argument("--link_prob", type=float, default=0.0)
    p.add_argument("--min_component", type=float, default=2.0, help="mm")
    p.add_argument("--tube_sides", type=int, default=16)
    p.add_argument("--reuse", action="store_true", help="reuse tubetk.tre")
    p.add_argument("--tag", default="", help="suffix for the output files")
    args = p.parse_args()

    img = nib.load(OUT / "network.nii.gz")
    mask = np.asanyarray(img.dataobj) > 0
    h = float(np.mean(img.header.get_zooms()[:3]))
    cons = nib.load(OUT / "consensus.nii.gz").get_fdata(dtype=np.float32)
    if args.sigma > 0:
        cons = ndimage.gaussian_filter(cons, args.sigma)
    edt = ndimage.distance_transform_edt(mask, sampling=h)
    sk = skeletonize(mask).astype(bool)
    sk_idx = np.argwhere(sk)

    t0 = time.time()
    tre = OUT / f"tubetk{args.tag}.tre"
    if args.reuse and tre.exists():
        tubes = read_tre(tre)
    else:
        seeds = np.where(sk, np.maximum(cons, 1e-2), 0).astype(np.float32)
        tubes = segment_tubes(cons, seeds, h, args, tre)
    t_seg = time.time() - t0
    print(f"{len(tubes)} tubes in {t_seg:.0f} s", flush=True)
    tubes = [subsample(t, args.step) for t in tubes if len(t["xyz"]) > 1]

    G, xyz, r, n_join, n_rejected = build_graph(tubes, cons, h, args)
    ridge = np.concatenate([t["ridgeness"] for t in tubes])
    inten = np.concatenate([t["intensity"] for t in tubes])
    # drop tiny isolated pieces
    for comp in list(nx.connected_components(G)):
        pts = xyz[list(comp)]
        if np.ptp(pts, axis=0).max() < args.min_component:
            G.remove_nodes_from(comp)
    keep = np.array(sorted(G))
    new_id = np.full(len(xyz), -1)
    new_id[keep] = np.arange(len(keep))
    G = nx.relabel_nodes(G, dict(zip(keep, new_id[keep])))
    xyz, r, ridge, inten = xyz[keep], r[keep], ridge[keep], inten[keep]
    polys = split_branches(G)

    pts = nib.affines.apply_affine(img.affine, xyz / h)
    lengths = [np.linalg.norm(np.diff(pts[i], axis=0), axis=1).sum() for i in polys]
    net = pv.PolyData(pts, lines=np.concatenate([np.r_[len(i), i] for i in polys]))
    net["radius"] = r
    net["ridgeness"] = ridge
    net["intensity"] = inten
    net.cell_data["branch"] = np.arange(net.n_cells)
    net.cell_data["length"] = np.array(lengths)
    net.cell_data["mean_radius"] = np.array([r[i].mean() for i in polys])
    net.save(OUT / f"tubetk_centerlines{args.tag}.vtp")

    tubes_out = []
    for b, ids in enumerate(polys):
        line = pv.lines_from_points(pts[ids])
        line["radius"] = r[ids]
        tube = line.tube(
            scalars="radius", absolute=True, n_sides=args.tube_sides, capping=True
        )
        tube.cell_data["branch"] = np.full(tube.n_cells, b)
        tubes_out.append(tube)
    pv.merge(tubes_out).save(OUT / f"tubetk_tubes{args.tag}.vtp")

    # comparison with the mask: skeleton coverage, radius vs EDT, loops
    sk_xyz = sk_idx * h
    d, _ = cKDTree(xyz).query(sk_xyz)
    sk_r = edt[tuple(sk_idx.T)]
    covered = d <= sk_r + h
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    d_big, _ = cKDTree(xyz[list(comps[0])]).query(sk_xyz)
    vox = np.clip(np.round(xyz / h).astype(int), 0, np.array(mask.shape) - 1)
    in_mask = mask[tuple(vox.T)]
    edt_at = edt[tuple(vox.T)]
    loops = G.number_of_edges() - G.number_of_nodes() + len(comps)
    stats = dict(
        args=vars(args),
        segment_seconds=round(t_seg),
        tubes=len(tubes),
        joins=n_join,
        joins_rejected_low_prob=n_rejected,
        branches=int(net.n_cells),
        points=int(net.n_points),
        components=len(comps),
        largest_component_points_fraction=len(comps[0]) / G.number_of_nodes(),
        loops=int(loops),
        total_length_mm=float(np.sum(lengths)),
        length_outside_mask_fraction=float(1 - in_mask.mean()),
        skeleton_covered_fraction=float(covered.mean()),
        skeleton_covered_by_largest_fraction=float((d_big <= sk_r + h).mean()),
        radius_mm_percentiles_5_50_95=np.round(np.percentile(r, [5, 50, 95]), 3).tolist(),
        radius_over_edt_median=float(np.median(r[in_mask] / np.maximum(edt_at[in_mask], h / 2))),
    )
    (OUT / f"tubetk{args.tag}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
