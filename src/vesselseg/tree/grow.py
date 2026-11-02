"""Grow synthetic distal branches from the arterial tree through the subarachnoid space.

The extracted tree (extract -> fit-radius -> label) stops near the image
resolution (tip radius ~0.4 mm), so large parts of the cortical SAS lie 10-25 mm
from any vessel. This step adds the missing distal branches as a synthetic
forest, in the spirit of constrained constructive optimisation (CCO), but on
the voxel graph of the SAS, so that every branch stays in the CSF along the pia:

1. Domain: SAS voxels of the sasparc (ventricles and choroid plexus excluded),
   26-neighbour graph. Edges between left and right cortex labels (the falx)
   and between cerebellar and cerebral labels (the tentorium) are cut. Cost per
   mm of path

       c = (1 + w_ev * (1 - p)) / (1 + w_ev) + w_pia * min(max(d_pia - pia_band, 0), d_cap)

   with d_pia the distance (mm) to brain tissue and p the vessel probability
   (the risc-mi maps, also below the extraction threshold): branches hug the
   pia and follow faint evidence where there is some. Along the pia without
   evidence c = 1, so a cost-mm is a mm there (1 / (1 + w_ev) on p = 1).
2. Terminals: pia-adjacent SAS voxels (d_pia <= --pia_band), one random voxel
   per --spacing mm cube.
3. Territories (--mode budget, the default): the tips of the input network
   (trunk labels excluded) share the terminals further than --cover from it.
   A tip of radius r_i should feed a subtree in proportion to its flow,
   r_i^k (Murray, k = --murray), so the SAS is split into an additively
   weighted geodesic Voronoi, v -> argmin_i d(v, tip_i) + w_i, with the
   weights w iterated (--iters, --eta) until tip i holds ~r_i^k / sum r^k of
   those terminals. Edges between territories are cut, and edges over another
   input vessel cost --w_occ times more.
4. Growth: terminals in random order (--seed); one further than --cover
   (cost-mm) from the network -- the input plus the synthetic branches so far
   -- is joined by its least-cost path to its own tip's subtree (budget), or to
   the nearest attachable voxel of the network (--mode nearest: also mid-vessel,
   no trunks). Budget: a tip that got no branch is then joined to the farthest
   pia voxel of its territory, so that every tip continues.
5. Geometry: Laplacian smoothing of the whole synthetic forest (--smooth_iters
   steps of --smooth_lam towards the neighbour mean; junctions move, the
   attachments and leaves stay, a step out of the SAS is undone) pulls the
   voxel paths taut and the junctions into place, in place of a CCO-style
   optimisation; then the forest is cut into chains (attachment / junction /
   tip). A chain attached to
   the input network starts at an exact vertex of it, so a coordinate merge
   (proteoflow.pvs.read_centerlines) joins it to its host.
6. Radius: Murray's law r_parent^k = sum r_child^k. Nearest: --r_tip at the
   leaves, each chain capped by its parent. Budget: each tip's subtree is
   scaled so that it starts at the tip's radius (mean over its last 2 mm); the
   implied leaf radius per tip is the check of the budget; each leaf chain
   then tapers linearly to --r_tip (unless --no_taper). --trim mm are cut
   from every leaf afterwards (shorter leaves dropped). A chain's first
   point carries the parent radius, so the merged radius of the host is
   unchanged.

Synthetic chains inherit `segment` and `inlet` from their host; the blood
pressure is solved again over the whole network (label.blood_pressure).

Output (next to DIR/NAME, or --out):
  NAME_grown.vtp        NAME.vtp + the synthetic chains; point/cell data
                        `synthetic` (0/1), cell data `territory` (tip index)
  NAME_grown_tubes.vtp  tube around every branch (synthetic chains at their
                        own radius); cell data `branch`, `segment`, `synthetic`
  NAME_grown.json       arguments, counts, synthetic length per label, distance
                        from the SAS to the network per sasparc label, before/after;
                        budget: per tip radius, weight, target, count, leaf radius
  NAME_grown.png        three projections, input grey, synthetic red

usage: vesselseg grow DIR/NAME --sasparc SASPARC --seg SEG --evidence P.nii.gz
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
import scipy.sparse as sp
from scipy import ndimage
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from vesselseg.io import polylines, resample_like
from vesselseg.tree.label import ASEG, NAMES, blood_pressure

# CSF without pial arteries: ventricles, choroid plexus
NO_ARTERY = sorted({v for pair, t in ASEG.items() if t == "U" for v in pair})
CEREBELLUM = [7, 8, 46, 47]
# one direction of every 26-neighbour pair
OFFSETS = [o for o in itertools.product((-1, 0, 1), repeat=3) if o > (0, 0, 0)]


def hemisphere(lab):
    """1 for left cortex labels (1000s), 2 for right (2000s), 0 otherwise."""
    return np.where((lab >= 1000) & (lab < 2000), 1, np.where(lab >= 2000, 2, 0))


def barrier(la, lb):
    """Neighbour pairs across the falx (left/right cortex) or the tentorium
    (cerebellum/cerebral cortex)."""
    ha, hb = hemisphere(la), hemisphere(lb)
    ca, cb = np.isin(la, CEREBELLUM), np.isin(lb, CEREBELLUM)
    return (ha * hb == 2) | (ca & (hb > 0)) | (cb & (ha > 0))


def sas_graph(idx, lab, cost, zooms):
    """Symmetric CSR graph of the voxels with idx >= 0, edge weight = length x
    mean cost of its two voxels; falx / tentorium pairs left out."""
    n = int(idx.max()) + 1
    rows, cols, W = [], [], []
    for o in OFFSETS:
        a = tuple(slice(max(-k, 0), s - max(k, 0)) for k, s in zip(o, idx.shape))
        b = tuple(slice(max(k, 0), s - max(-k, 0)) for k, s in zip(o, idx.shape))
        ia, ib = idx[a], idx[b]
        ok = (ia >= 0) & (ib >= 0)
        keep = ~barrier(lab[a][ok], lab[b][ok])
        ia, ib = ia[ok][keep], ib[ok][keep]
        L = float(np.linalg.norm(np.multiply(o, zooms)))
        rows.append(ia)
        cols.append(ib)
        W.append(L * 0.5 * (cost[ia] + cost[ib]))
    rows, cols, W = np.concatenate(rows), np.concatenate(cols), np.concatenate(W)
    return sp.csr_matrix((np.r_[W, W], (np.r_[rows, cols], np.r_[cols, rows])), shape=(n, n))


def densify(P, polys, step):
    """Points every <= `step` mm along the polylines, with the nearest vertex of each."""
    pts, vid = [], []
    for ids in polys:
        for i0, i1 in zip(ids[:-1], ids[1:]):
            k = max(2, int(np.ceil(np.linalg.norm(P[i1] - P[i0]) / step)) + 1)
            t = np.linspace(0.0, 1.0, k)[:, None]
            pts.append(P[i0] * (1 - t) + P[i1] * t)
            vid.append(np.where(t[:, 0] < 0.5, i0, i1))
    return np.vstack(pts), np.concatenate(vid)


def seed_nodes(pts, vid, kd, node_xyz, reach):
    """Graph node of each densified tree point (nearest SAS voxel within `reach`
    mm); per node the vertex of its closest point. Returns (nodes, vertices)."""
    d, nd = kd.query(pts, distance_upper_bound=reach)
    ok = np.isfinite(d)
    d, nd, vid = d[ok], nd[ok], vid[ok]
    order = np.argsort(d)
    nodes, first = np.unique(nd[order], return_index=True)
    return nodes, vid[order][first]


def input_tips(P, polys, R, along=2.0):
    """Tips of the input network (degree-1 vertices after a coordinate merge) and
    the mean radius of each over its last `along` mm. Returns (vertices, radius)."""
    key = lambda i: tuple(np.round(P[i], 4))
    deg = {}
    for ids in polys:
        for a, b in zip(ids[:-1], ids[1:]):
            for k in (key(a), key(b)):
                deg[k] = deg.get(k, 0) + 1
    verts, radius = [], []
    for ids in polys:
        for oriented in (ids, ids[::-1]):
            v = oriented[0]
            if deg.get(key(v)) != 1:
                continue
            s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[oriented], axis=0), axis=1))]
            verts.append(int(v))
            radius.append(float(R[oriented][s <= along].mean()))
    return np.array(verts, np.int64), np.array(radius)


def territories(G, roots, demand, share, iters, eta):
    """Additively weighted geodesic Voronoi of G among the `roots` (nodes): node v
    belongs to argmin_i d(v, root_i) + w_i, the weights w tuned so that root i
    holds ~share_i of the `demand` nodes. This is the dual ascent of a
    semi-discrete transport problem, with a diminishing step eta / (1 + it / 10);
    the best iterate (median |log(count/target)|) is kept. Returns (label per
    node, -1 where unreachable; w; demand count per root)."""
    n, m = G.shape[0], len(roots)
    A = G.tocoo()
    # one extra node per root, with a single directed edge (weight w_i) to it
    rows = np.r_[A.row, n + np.arange(m)]
    cols = np.r_[A.col, roots]
    aug = sp.csr_matrix((np.r_[A.data, np.ones(m)], (rows, cols)), shape=(n + m, n + m))
    slot = aug.indptr[n : n + m]
    target = share / share.sum() * len(demand)
    w = np.zeros(m)
    best = (np.inf, None, None, None)
    for it in range(iters):
        aug.data[slot] = w + 1e-6
        _, _, src = dijkstra(aug, indices=n + np.arange(m), min_only=True, return_predecessors=True)
        label = np.where(src[:n] >= 0, src[:n] - n, -1)
        ld = label[demand]
        counts = np.bincount(ld[ld >= 0], minlength=m)
        err = np.clip(np.log((counts + 0.5) / (target + 0.5)), -2.0, 2.0)
        score = float(np.median(np.abs(err)))
        if score < best[0]:
            best = (score, label, w.copy(), counts)
        if it % 10 == 0 or it == iters - 1:
            print(f"  territories {it}: median |log(count/target)| {score:.2f}", flush=True)
        w += eta / (1 + it / 10) * err
        w -= w.min()
    print(f"  territories: best median |log(count/target)| {best[0]:.2f}", flush=True)
    return best[1], best[2], best[3]


def within(G, label, occupied, w_occ):
    """G without the edges between territories; edges touching an `occupied`
    voxel (another input vessel) cost w_occ times more."""
    A = G.tocoo()
    keep = (label[A.row] == label[A.col]) & (label[A.row] >= 0)
    data = A.data * np.where(occupied[A.row] | occupied[A.col], w_occ, 1.0)
    return sp.csr_matrix((data[keep], (A.row[keep], A.col[keep])), shape=G.shape)


def raster_distance(points, aff, shape, zooms):
    """Euclidean distance (mm) of every voxel of the grid to the nearest point."""
    ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(aff), points)).astype(int)
    ok = np.all((ijk >= 0) & (ijk < shape), axis=1)
    free = np.ones(shape, bool)
    free[tuple(ijk[ok].T)] = False
    return ndimage.distance_transform_edt(free, sampling=zooms)


def chains_of(parent, is_anchor):
    """Cut the synthetic forest (`parent`: node -> parent node) into chains from
    an anchor / junction to a junction / tip, each ordered parent -> child."""
    kids = {}
    for u, p in parent.items():
        kids.setdefault(p, []).append(u)
    out = []
    for u, p in parent.items():
        if not (is_anchor(p) or len(kids.get(p, ())) >= 2):
            continue
        ch = [p, u]
        while len(kids.get(ch[-1], ())) == 1:
            ch.append(kids[ch[-1]][0])
        out.append(ch)
    return out, kids


def smooth_forest(parent, X0, fixed, inside, iters, lam):
    """Laplacian smoothing of the synthetic forest (`parent`: node -> parent
    node, X0: node -> position): every free node, junctions included, moves
    `lam` of the way to the mean of its graph neighbours, `iters` times. Nodes
    in `fixed` stay; a move that leaves the domain (`inside`, positions ->
    bool) is undone. Pulls the voxel paths taut and the junctions into place.
    Returns {node: position}."""
    nodes = sorted(set(parent) | set(parent.values()))
    at = {v: i for i, v in enumerate(nodes)}
    i = np.array([at[u] for u in parent])
    j = np.array([at[p] for p in parent.values()])
    m = len(nodes)
    A = sp.csr_matrix((np.ones(2 * len(i)), (np.r_[i, j], np.r_[j, i])), shape=(m, m))
    A = sp.diags(1.0 / np.asarray(A.sum(axis=1)).ravel()) @ A
    X = np.array([X0[v] for v in nodes])
    free = np.array([v not in fixed for v in nodes])
    for _ in range(iters):
        Y = X + lam * (A @ X - X)
        ok = free & inside(Y)
        X[ok] = Y[ok]
    return dict(zip(nodes, X))


def plot(P, polys, synth, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax, (i, j, name) in zip(axes, [(0, 1, "axial"), (0, 2, "coronal"), (1, 2, "sagittal")]):
        for flag, color, lw in [(0, "0.45", 0.8), (1, "tab:red", 0.4)]:
            segs = [P[ids][:, [i, j]] for ids, s in zip(polys, synth) if s == flag]
            ax.add_collection(LineCollection(segs, colors=color, linewidths=lw))
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_title(name)
        ax.set_xlabel("RAS"[i] + " (mm)")
        ax.set_ylabel("RAS"[j] + " (mm)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parser():
    p = argparse.ArgumentParser(prog="vesselseg grow", description=__doc__.split("\n")[0])
    p.add_argument(
        "tree", help="path stem DIR/NAME of a `vesselseg label` output (NAME.vtp, .json)"
    )
    p.add_argument("--sasparc", required=True, help="sasparc labels (T1w space): the SAS")
    p.add_argument("--seg", required=True, help="seg_fixed (T1w space): the brain tissue")
    p.add_argument("--evidence", nargs="*", default=[], help="vessel probability maps (max)")
    p.add_argument("--out", help="output stem (default DIR/NAME_grown)")
    p.add_argument("--spacing", type=float, default=2.0, help="mm between terminals")
    p.add_argument("--cover", type=float, default=4.0, help="cost-mm to the network")
    p.add_argument("--pia_band", type=float, default=0.75, help="mm, terminal to tissue")
    p.add_argument("--w_pia", type=float, default=1.0, help="cost per mm off the pia")
    p.add_argument("--d_cap", type=float, default=3.0, help="mm, cap of d_pia in the cost")
    p.add_argument("--w_ev", type=float, default=1.0, help="cost of missing evidence")
    p.add_argument("--reach", type=float, default=1.5, help="mm, tree point to SAS voxel")
    p.add_argument("--limit", type=float, default=30.0, help="cost-mm of each update")
    p.add_argument("--no_attach", default="ICA-L,ICA-R,VB,CoW", help="trunk labels")
    p.add_argument("--mode", choices=["budget", "nearest"], default="budget")
    p.add_argument("--iters", type=int, default=150, help="territory weight iterations")
    p.add_argument("--eta", type=float, default=2.0, help="cost-mm per unit log error")
    p.add_argument("--w_occ", type=float, default=3.0, help="cost factor on input vessels")
    p.add_argument("--r_tip", type=float, default=0.1, help="mm, leaf (end) radius")
    p.add_argument("--no_taper", action="store_true", help="budget: constant leaf radius")
    p.add_argument("--trim", type=float, default=0.0, help="mm cut from every leaf")
    p.add_argument("--murray", type=float, default=3.0, help="Murray exponent k")
    p.add_argument("--smooth_iters", type=int, default=50, help="Laplacian smoothing steps")
    p.add_argument("--smooth_lam", type=float, default=0.5, help="step of each")
    p.add_argument("--min_tip", type=float, default=1.0, help="mm, shorter tip chains pruned")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tube_sides", type=int, default=12)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    t0 = time.time()
    stem = Path(args.tree)
    out = Path(args.out) if args.out else stem.parent / f"{stem.name}_grown"
    net = pv.read(f"{stem}.vtp")
    meta = json.loads(Path(f"{stem}.json").read_text())
    inlets = meta["inlets"]
    inlet_xyz = np.array(inlets["VA"] + [inlets[k] for k in ("ICA-L", "ICA-R") if k in inlets])
    P = np.asarray(net.points)
    polys = polylines(net)
    R = np.asarray(net.point_data["radius"])
    seg_pt = np.asarray(net.point_data["segment"])

    # 1. domain, cropped to the SAS
    img = nib.load(args.sasparc)
    lab = np.asarray(img.dataobj).astype(np.int32)
    seg_img = nib.load(args.seg)
    if seg_img.shape[:3] != img.shape[:3] or not np.allclose(seg_img.affine, img.affine):
        seg_img = resample_like(seg_img, img, 0)
    tissue = (np.asarray(seg_img.dataobj) > 0) & (lab == 0)
    sas = (lab > 0) & ~np.isin(lab, NO_ARTERY)
    nz = np.argwhere(sas)
    lo, hi = np.maximum(nz.min(0) - 2, 0), np.minimum(nz.max(0) + 3, sas.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    lab, sas, tissue = lab[box], sas[box], tissue[box]
    aff = img.affine.copy()
    aff[:3, 3] = img.affine[:3, :3] @ lo + img.affine[:3, 3]
    zooms = np.array(img.header.get_zooms()[:3], float)
    d_pia = ndimage.distance_transform_edt(~tissue, sampling=zooms)
    p_vol = np.zeros(sas.shape, np.float32)
    for f in args.evidence:
        e = np.asanyarray(resample_like(nib.load(f), (sas.shape, aff), 1).dataobj)
        p_vol = np.maximum(p_vol, np.clip(e, 0, 1))
    idx = np.full(sas.shape, -1, np.int32)
    n = int(sas.sum())
    idx[sas] = np.arange(n)
    ijk = np.argwhere(sas)  # node -> voxel (C order, as idx)
    node_xyz = nib.affines.apply_affine(aff, ijk)
    off_pia = np.minimum(np.maximum(d_pia[sas] - args.pia_band, 0.0), args.d_cap)
    cost = (1.0 + args.w_ev * (1.0 - p_vol[sas])) / (1.0 + args.w_ev) + args.w_pia * off_pia
    G = sas_graph(idx, lab, cost, zooms)
    print(f"SAS graph: {n} voxels, {G.nnz // 2} edges ({time.time() - t0:.0f} s)", flush=True)

    # seeds: the input tree; trunks cover but take no branch
    kd = cKDTree(node_xyz)
    pts, vid = densify(P, polys, 0.25)
    seeds, seed_vid = seed_nodes(pts, vid, kd, node_xyz, args.reach)
    no_attach = [NAMES.index(s) for s in args.no_attach.split(",") if s]
    attachable = np.zeros(n, bool)
    attachable[seeds] = ~np.isin(seg_pt[seed_vid], no_attach)
    anchor = np.full(n, -1, np.int64)  # node -> input vertex
    anchor[seeds] = seed_vid
    in_tree = np.zeros(n, bool)
    in_tree[seeds] = True
    print(f"seeds: {len(seeds)} voxels ({attachable.sum()} attachable)", flush=True)

    # 2. terminals
    rng = np.random.default_rng(args.seed)
    cand = np.flatnonzero(d_pia[sas] <= args.pia_band)
    cand = cand[rng.permutation(len(cand))]
    cell = np.floor(ijk[cand] * zooms / args.spacing).astype(np.int64)
    _, first = np.unique(cell, axis=0, return_index=True)
    terminals = cand[np.sort(first)]  # random order (the permutation above)
    print(f"terminals: {len(terminals)} of {len(cand)} pia voxels", flush=True)

    # 3. growth. dist: cost to the whole network (coverage); gdist / gpred: cost
    # to the tree a branch may join (nearest: the network; budget: its territory)
    dist, pred, src = dijkstra(G, indices=seeds, min_only=True, return_predecessors=True)
    dist0 = dist.copy()
    budget = None
    if args.mode == "budget":
        tip_v, tip_r = input_tips(P, polys, R)
        ok = ~np.isin(seg_pt[tip_v], no_attach)
        d_tip, tip_n = kd.query(P[tip_v], distance_upper_bound=args.reach)
        ok &= np.isfinite(d_tip)
        tip_v, tip_r, tip_n = tip_v[ok], tip_r[ok], tip_n[ok]
        tip_n, first = np.unique(tip_n, return_index=True)
        tip_v, tip_r = tip_v[first], tip_r[first]
        print(f"tips: {len(tip_n)} attachable input tips in the SAS", flush=True)
        demand = terminals[np.isfinite(dist0[terminals]) & (dist0[terminals] > args.cover)]
        share = tip_r**args.murray
        label, w, counts = territories(G, tip_n, demand, share, args.iters, args.eta)
        occupied = np.zeros(n, bool)
        occupied[seeds] = True
        occupied[tip_n] = False
        G_grow = within(G, label, occupied, args.w_occ)
        anchor[:] = -1
        anchor[tip_n] = tip_v
        in_tree[:] = False
        in_tree[tip_n] = True
        gdist, gpred, _ = dijkstra(G_grow, indices=tip_n, min_only=True, return_predecessors=True)
        budget = dict(tip_n=tip_n, tip_v=tip_v, tip_r=tip_r, w=w, counts=counts, demand=demand)
    else:
        label = np.zeros(n, np.int64)
        G_grow, gdist, gpred = G, dist, pred

    parent = {}
    joined = skipped_trunk = 0

    def join(t):
        """Add the least-cost path from node t to the tree; update both fields."""
        path = [t]
        while not in_tree[path[-1]] and gpred[path[-1]] >= 0:
            path.append(int(gpred[path[-1]]))
        if not in_tree[path[-1]]:
            return False
        new = np.array(path[:-1])
        for a, b in zip(path[:-1], path[1:]):
            parent[a] = b
        in_tree[new] = True
        attachable[new] = True
        d2, p2, s2 = dijkstra(
            G, indices=new, min_only=True, return_predecessors=True, limit=args.limit
        )
        better = d2 < dist
        dist[better], pred[better], src[better] = d2[better], p2[better], s2[better]
        if G_grow is not G:
            d2, p2, _ = dijkstra(
                G_grow, indices=new, min_only=True, return_predecessors=True, limit=args.limit
            )
            better = d2 < gdist
            gdist[better], gpred[better] = d2[better], p2[better]
        return True

    for t in terminals:
        if in_tree[t] or not dist[t] > args.cover or not np.isfinite(gdist[t]):
            continue
        if args.mode == "nearest" and not attachable[src[t]]:
            skipped_trunk += 1
            continue
        if join(t):
            joined += 1
            if joined % 200 == 0:
                t_s = time.time() - t0
                print(f"  {joined} branches, {len(parent)} voxels ({t_s:.0f} s)", flush=True)
    forced = 0
    if budget is not None:
        # every tip continues: one branch to the farthest pia voxel of its territory
        fed = {p for p in parent.values()}
        for i, v in enumerate(budget["tip_n"]):
            if v in fed:
                continue
            mine = cand[(label[cand] == i) & np.isfinite(gdist[cand]) & ~in_tree[cand]]
            if len(mine) and gdist[mine].max() > args.cover and join(mine[np.argmax(gdist[mine])]):
                forced += 1
    print(
        f"grown: {joined} branches (+{forced} forced tip continuations), {len(parent)} voxels; "
        f"{skipped_trunk} terminals left to trunks"
    )

    # 4. chains; prune short tips, then smooth
    def is_anchor(v):
        return anchor[v] >= 0 and v not in parent

    def length(ch):
        X = node_xyz[ch]
        return float(np.linalg.norm(np.diff(X, axis=0), axis=1).sum())

    chains, kids = chains_of(parent, is_anchor)
    pruned = 0
    for ch in chains:
        if not kids.get(ch[-1]) and not is_anchor(ch[0]) and length(ch) < args.min_tip:
            for v in ch[1:]:
                parent.pop(v, None)
            pruned += 1
    chains, kids = chains_of(parent, is_anchor)
    # smoothing: anchors (at the exact host vertex) and leaves fixed
    X0 = {v: P[anchor[v]] if is_anchor(v) else node_xyz[v] for v in set(parent) | set(kids)}
    fixed = {v for v in X0 if is_anchor(v) or not kids.get(v)}
    inv = np.linalg.inv(aff)

    def inside(Y):
        v = np.rint(nib.affines.apply_affine(inv, Y)).astype(int)
        ok = np.all((v >= 0) & (v < sas.shape), axis=1)
        ok[ok] = sas[tuple(v[ok].T)]
        return ok

    Xs = smooth_forest(parent, X0, fixed, inside, args.smooth_iters, args.smooth_lam)
    X_ch = [np.array([Xs[v] for v in ch]) for ch in chains]

    # 5. radius: Murray from the tips, capped by the parent
    start_of = {}
    for c, ch in enumerate(chains):
        start_of.setdefault(ch[0], []).append(c)
    r = np.full(len(chains), np.nan)

    def murray(c):
        stack = [c]
        while stack:  # post-order without recursion
            c = stack[-1]
            todo = [d for d in start_of.get(chains[c][-1], []) if np.isnan(r[d])]
            if todo:
                stack += todo
                continue
            ch_r = [r[d] for d in start_of.get(chains[c][-1], [])]
            r[c] = (np.sum(np.power(ch_r, args.murray)) ** (1 / args.murray)) if ch_r else r_leaf
            stack.pop()

    # nearest: r_tip at the leaves, capped by the parent. budget: unit leaves,
    # then each tip's subtree scaled so that it starts at the tip's radius (its
    # leaf radius r_leaf_i is the budget check)
    r_leaf = args.r_tip if budget is None else 1.0
    roots = [c for c, ch in enumerate(chains) if is_anchor(ch[0])]
    for c in roots:
        murray(c)
    r_first = np.zeros(len(chains))
    host = np.full(len(chains), -1, np.int64)
    scale = np.ones(len(chains))
    if budget is not None:
        tip_index = {int(v): i for i, v in enumerate(budget["tip_n"])}
        root_sum = {}
        for c in roots:
            root_sum[chains[c][0]] = root_sum.get(chains[c][0], 0.0) + r[c] ** args.murray
        leaf_r = np.full(len(budget["tip_n"]), np.nan)
        for v, s in root_sum.items():
            i = tip_index[int(v)]
            leaf_r[i] = budget["tip_r"][i] / s ** (1 / args.murray)
        budget["leaf_r"] = leaf_r
    stack = []
    for c in roots:
        v = anchor[chains[c][0]]
        r_first[c], host[c] = R[v], v
        if budget is None:
            r[c] = min(r[c], R[v])
        else:
            scale[c] = budget["leaf_r"][tip_index[int(chains[c][0])]]
            r[c] *= scale[c]
        stack.append(c)
    while stack:
        c = stack.pop()
        for d in start_of.get(chains[c][-1], []):
            if budget is None:
                r[d] = min(r[d], r[c])
            else:
                scale[d] = scale[c]
                r[d] *= scale[d]
            r_first[d], host[d] = r[c], host[c]
            stack.append(d)

    # leaves: trim the last --trim mm (a shorter leaf goes), then taper (budget)
    # from the chain radius to --r_tip; radii are those of the untrimmed tree
    def arc(X):
        return np.r_[0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]

    keep = np.ones(len(chains), bool)
    r_ch, ijk_ch = [], []
    for c, ch in enumerate(chains):
        X, V = X_ch[c], ijk[ch]
        leaf = ch[-1] not in start_of
        if leaf and args.trim > 0:
            s = arc(X)
            if s[-1] - args.trim < args.min_tip:
                keep[c] = False
                continue
            m = s <= s[-1] - args.trim
            X_ch[c], V = X[m], V[m]
        rr = np.full(len(X_ch[c]), r[c])
        if leaf and budget is not None and not args.no_taper:
            s = arc(X_ch[c])
            rr = r[c] + (min(args.r_tip, r[c]) - r[c]) * s / s[-1]
        rr[0] = r_first[c]
        r_ch.append(rr)
        ijk_ch.append(V)
    trimmed = int((~keep).sum())
    chains = [ch for ch, k in zip(chains, keep) if k]
    X_ch = [X for X, k in zip(X_ch, keep) if k]
    r, r_first, host = r[keep], r_first[keep], host[keep]

    # output network
    n0 = net.n_points
    offs = np.cumsum([0] + [len(X) for X in X_ch])
    new_pts = np.vstack(X_ch) if X_ch else np.zeros((0, 3))
    lines = [np.asarray(net.lines)]
    for c in range(len(chains)):
        lines.append(np.r_[len(X_ch[c]), n0 + np.arange(offs[c], offs[c + 1])])
    grown = pv.PolyData(np.vstack([P, new_pts]), lines=np.concatenate(lines))
    r_pt = np.concatenate(r_ch or [[]])
    host_pt = np.concatenate([np.full(len(X_ch[c]), host[c]) for c in range(len(chains))] or [[]])
    host_pt = host_pt.astype(np.int64)
    p_pt = np.concatenate([p_vol[tuple(V.T)] for V in ijk_ch] or [[]])
    syn = {
        "radius": r_pt,
        "radius_raw": r_pt,
        "evidence": p_pt,
        "node_type": np.zeros(len(new_pts)),
        "segment": seg_pt[host_pt] if len(host_pt) else np.zeros(0),
    }
    for k in net.point_data:
        a = np.asarray(net.point_data[k])
        if k == "blood_pressure":
            continue
        b = syn.get(k, np.zeros((len(new_pts),) + a.shape[1:]))
        grown.point_data[k] = np.concatenate([a, np.asarray(b, a.dtype)])
    grown.point_data["synthetic"] = np.r_[np.zeros(n0, np.uint8), np.ones(len(new_pts), np.uint8)]
    cell_of = np.zeros(n0, np.int64)
    for c, ids in enumerate(polys):
        cell_of[ids] = c
    host_cell = cell_of[host]
    ch_len = np.array([float(np.linalg.norm(np.diff(X, axis=0), axis=1).sum()) for X in X_ch])
    csyn = {
        "branch": len(polys) + np.arange(len(chains)),
        "length": ch_len,
        "mean_radius": np.array([rr[1:].mean() for rr in r_ch]),
        "origin": np.full(len(chains), -1),
    }
    for k in net.cell_data:
        a = np.asarray(net.cell_data[k])
        b = csyn.get(k, a[host_cell] if len(chains) else a[:0])
        grown.cell_data[k] = np.concatenate([a, np.asarray(b, a.dtype)])
    grown.cell_data["synthetic"] = np.r_[
        np.zeros(len(polys), np.uint8), np.ones(len(chains), np.uint8)
    ]
    terr = np.array([label[ch[0]] if budget is not None else -1 for ch in chains], np.int64)
    grown.cell_data["territory"] = np.r_[np.full(len(polys), -1, np.int64), terr]
    for k in net.field_data:
        grown.field_data[k] = net.field_data[k]
    grown.point_data["blood_pressure"] = blood_pressure(grown, inlet_xyz)
    grown.save(f"{out}.vtp")

    tubes = []
    all_polys = polylines(grown)
    synth = np.asarray(grown.cell_data["synthetic"])
    seg_cell = np.asarray(grown.cell_data["segment"])
    Pg, Rg = np.asarray(grown.points), np.asarray(grown.point_data["radius"])
    for b, ids in enumerate(all_polys):
        line = pv.lines_from_points(Pg[ids])
        if synth[b]:  # the chain's own radius at its first point too
            rr = r_ch[b - len(polys)]
            line["radius"] = np.r_[rr[1], rr[1:]]
        else:
            line["radius"] = Rg[ids]
        tube = line.tube(scalars="radius", absolute=True, n_sides=args.tube_sides, capping=True)
        terr_b = terr[b - len(polys)] if synth[b] else -1
        for k, v in (
            ("branch", b),
            ("segment", seg_cell[b]),
            ("synthetic", synth[b]),
            ("territory", terr_b),
        ):
            tube.cell_data[k] = np.full(tube.n_cells, v)
        tubes.append(tube)
    merged = pv.merge(tubes)
    merged.field_data["segment_names"] = np.array(NAMES)
    merged.save(f"{out}_tubes.vtp")
    plot(Pg, all_polys, synth, f"{out}.png")

    # coverage: Euclidean distance from each SAS label to the network, before / after
    syn_pts = (
        densify(Pg, [ids for ids, s in zip(all_polys, synth) if s], 0.25)[0]
        if len(chains)
        else np.zeros((0, 3))
    )
    d_before = raster_distance(pts, aff, sas.shape, zooms)
    d_after = (
        np.minimum(d_before, raster_distance(syn_pts, aff, sas.shape, zooms))
        if len(syn_pts)
        else d_before
    )
    coverage = {}
    for v in np.unique(lab[sas]):
        m = sas & (lab == v)
        b, a = d_before[m], d_after[m]
        coverage[int(v)] = dict(
            voxels=int(m.sum()),
            median_mm=[round(float(np.median(b)), 2), round(float(np.median(a)), 2)],
            p90_mm=[round(float(np.percentile(b, 90)), 2), round(float(np.percentile(a, 90)), 2)],
        )
    seg_syn = seg_cell[len(polys) :]
    stats = dict(
        args=vars(args),
        sas_voxels=n,
        terminals=int(len(terminals)),
        branches_joined=joined,
        terminals_left_to_trunks=skipped_trunk,
        tips_pruned=pruned,
        leaves_dropped_by_trim=trimmed,
        chains=len(chains),
        synthetic_length_mm=round(float(ch_len.sum()), 1),
        input_length_mm=round(float(np.asarray(net.cell_data["length"]).sum()), 1),
        synthetic_length_by_label={
            NAMES[k]: round(float(ch_len[seg_syn == k].sum()), 1) for k in np.unique(seg_syn)
        },
        radius_percentiles_5_50_95=np.percentile(r, [5, 50, 95]).round(3).tolist()
        if len(r)
        else [],
        cost_distance_median=[
            round(float(np.median(dist0[np.isfinite(dist0)])), 2),
            round(float(np.median(dist[np.isfinite(dist)])), 2),
        ],
        coverage_before_after=coverage,
        seconds=round(time.time() - t0, 1),
    )
    if budget is not None:
        b = budget
        target = b["tip_r"] ** args.murray
        target = target / target.sum() * len(b["demand"])
        ratio = (b["counts"] + 0.5) / (target + 0.5)
        fed = np.isfinite(b["leaf_r"])
        stats["budget"] = dict(
            tips=len(b["tip_n"]),
            tips_continued=int(fed.sum()),
            forced_continuations=forced,
            demand_terminals=int(len(b["demand"])),
            count_over_target_percentiles_5_50_95=np.percentile(ratio, [5, 50, 95])
            .round(2)
            .tolist(),
            leaf_radius_percentiles_5_50_95=np.percentile(b["leaf_r"][fed], [5, 50, 95])
            .round(3)
            .tolist()
            if fed.any()
            else [],
            per_tip=[
                dict(
                    xyz=P[v].round(1).tolist(),
                    segment=NAMES[seg_pt[v]],
                    radius=round(float(rr), 3),
                    weight=round(float(w_i), 2),
                    target=round(float(t_i), 1),
                    count=int(c_i),
                    leaf_radius=None if np.isnan(lr) else round(float(lr), 3),
                )
                for v, rr, w_i, t_i, c_i, lr in zip(
                    b["tip_v"], b["tip_r"], b["w"], target, b["counts"], b["leaf_r"]
                )
            ],
        )
    Path(f"{out}.json").write_text(json.dumps(stats, indent=2))
    print(
        json.dumps(
            {
                k: ({kk: vv for kk, vv in v.items() if kk != "per_tip"} if k == "budget" else v)
                for k, v in stats.items()
                if k not in ("args", "coverage_before_after")
            },
            indent=2,
        )
    )
    print(f"wrote {out}.vtp, {out}_tubes.vtp, {out}.json, {out}.png")


if __name__ == "__main__":
    main()
