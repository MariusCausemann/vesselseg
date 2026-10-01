"""Label the centerline tree by arterial territory, fully automatically.

Labels: ICA-L/R, VB (vertebrals + basilar), CoW (the Circle of Willis segments
the graph kept), ACA, MCA-L/R, PCA-L/R, cerebellar (SCA/AICA/PICA), unknown.
No atlas registration: the only anatomical input is the subject's own
parcellation (sasparc: Desikan-Killiany + aseg labels extended into the
subarachnoid space, where the vessels run).

Input: the arterial tree written by extract.py (NAME.vtp + NAME.json), e.g.
centerlines_tree_posterior_t2. It must be acyclic and in one piece, and NAME.json
must have `inlets_xyz`. The older VMTK-based centerlines (centerlines_posterior_t2)
have loops and no inlets, and are no longer supported.

1. Graph: one node per branch end, one edge per branch (cell of NAME.vtp); an
   error if it has a cycle or several components.
2. Inlets: NAME.json `inlets_xyz` (found by extract.py), each mapped to the
   nearest graph node. Vertebrals: the names starting with VA, rooted at the
   source (node_type 2, the lower vertebral inlet). ICAs: ICA-L / ICA-R by name.
   The mid-sagittal plane between the left and right cortex centroids is only
   used to resolve the side of midline labels.
3. Trunks. From each ICA inlet, and from the vertebral confluence, follow the
   thickest continuation until the first junction that splits into two
   children of comparable radius (ratio >= --split_ratio, measured 1-6 mm from
   the junction) heading to opposite sides (M1 vs A1 at the ICA terminus, the
   two P1 at the basilar tip), each feeding >= --min_territory mm of centerline
   next to its target cortex (ICA: MCA/ACA, basilar: PCA; this skips the SCA).
   ICA = inlet -> terminus, VB = vertebrals -> confluence -> basilar tip.
4. CoW: the rest of the backbone (the tree paths joining the inlets: A1-ACom-A1,
   PCom-P1, whichever the graph kept), each piece if it is at most --cow_max mm
   long. Longer pieces pass through a distal contact and are left to step 5.
5. Territories: every other branch gets ACA, MCA-L/R, PCA-L/R, cerebellar or
   unknown from an exact min-sum labelling of the tree (dynamic programming).
   Cost per mm of centerline: 0 if the label next to the point allows the
   class, 1 if not (0.25 where no label is within --prior_dist mm); plus a
   penalty at each junction where the label changes: --trunk_cost for a
   territory leaving its trunk (ICA-L -> MCA-L, VB -> PCA, ...), --unknown_cost
   for an unknown side branch (unknown only ends subtrees), --switch for
   anything implausible (MCA -> ACA, ICA-L -> MCA-R, ACA/MCA leaving the ICA
   below its terminus, PCA leaving the basilar below its tip, ...). A territory
   therefore only changes inside a subtree when more than ~--switch mm beyond
   disagree; those junctions are reported as suspected false contacts.

Output (next to DIR/NAME):
  NAME_labeled.vtp        NAME.vtp + cell/point data `segment` (index into the
                          field data `segment_names`) and point data
                          `blood_pressure` (Poiseuille, fitted radius: 1 at the
                          inlets, 0 at every other tip; blood.py), the blood
                          direction for a consumer such as proteoflow. Loops
                          inside one territory (extract.py's loop_returns) are
                          re-cut where their blood flows meet (recut_loops), so
                          its polylines can differ from NAME.vtp's
  NAME_tubes_labeled.vtp  NAME_tubes.vtp + cell data `segment`
  NAME_labeled.json       inlets, termini, CoW pieces, per-label length,
                          implausible junctions
  NAME_labeled.png        three projections coloured by label

The inlet tree of each branch (cell data `inlet`, decided by the extract.py
MILP) is not used for labelling; the agreement with the labels is printed.

usage: vesselseg label tree/centerlines --sasparc SASPARC
"""

import argparse
import json
from itertools import pairwise
from pathlib import Path

import networkx as nx
import nibabel as nib
import numpy as np
import pyvista as pv
from scipy import ndimage
from scipy.spatial import cKDTree

from vesselseg.io import midplane, polylines
from vesselseg.tree.blood import network_pressure

NAMES = [
    "unknown",
    "ICA-L",
    "ICA-R",
    "VB",
    "CoW",
    "ACA",
    "MCA-L",
    "MCA-R",
    "PCA-L",
    "PCA-R",
    "cerebellar",
]
UNK, ICA_L, ICA_R, VB, COW, ACA, MCA_L, MCA_R, PCA_L, PCA_R, CRB = range(len(NAMES))
TRUNK = [ICA_L, ICA_R, VB, COW]
TERRITORY = [UNK, ACA, MCA_L, MCA_R, PCA_L, PCA_R, CRB]
COLORS = [
    "lightgray",
    "saddlebrown",
    "sandybrown",
    "black",
    "tab:purple",
    "tab:green",
    "tab:blue",
    "tab:cyan",
    "tab:red",
    "tab:pink",
    "tab:orange",
]

# Desikan-Killiany index (label % 1000) -> territories it borders
# (A = ACA, M = MCA, P = PCA of the label's hemisphere)
DK = {
    **dict.fromkeys([2, 14, 17, 23, 26, 28, 32], "A"),
    **dict.fromkeys([10, 25], "AP"),  # isthmus cingulate, precuneus
    29: "AMP",  # superior parietal (watershed)
    **dict.fromkeys([5, 6, 7, 13, 16, 21], "P"),
    **dict.fromkeys([9, 11], "MP"),  # inferior temporal, lateral occipital
    **dict.fromkeys([1, 3, 8, 12, 15, 18, 19, 20, 22, 24, 27, 30, 31, 33, 34, 35], "M"),
}
# aseg (left, right) -> territories; C = cerebellar, U = not arterial
ASEG = {
    (7, 46): "C",
    (8, 47): "C",
    (16, 16): "CP",  # brainstem: SCA/AICA or P2 around the midbrain
    (10, 49): "P",  # thalamus (thalamoperforators)
    (17, 53): "P",
    (18, 54): "P",
    (11, 50): "MA",  # caudate, putamen, pallidum, accumbens (lenticulostriate)
    (12, 51): "MA",
    (13, 52): "MA",
    (26, 58): "MA",
    (4, 43): "U",  # ventricles, choroid plexus
    (5, 44): "U",
    (14, 14): "U",
    (15, 15): "U",
    (31, 63): "U",
}


def classes(label, side):
    """Classes a point next to `label` can belong to (None: no information).

    `side` (+1 right, -1 left of the mid-sagittal plane) resolves midline labels.
    """
    if label >= 1000:
        code, hemi = DK.get(label % 1000), 1 if label >= 2000 else -1
    else:
        code, hemi = None, side
        for (lft, rgt), c in ASEG.items():
            if label in (lft, rgt):
                code = c
                if lft != rgt:
                    hemi = -1 if label == lft else 1
    if code is None:
        return None
    right = hemi > 0
    out = {
        "A": [ACA],
        "M": [MCA_R if right else MCA_L],
        "P": [PCA_R if right else PCA_L],
        "C": [CRB],
        "U": [UNK],
    }
    return [k for ch in code for k in out[ch]]


def nearest_labels(points, img, max_dist):
    """Label of the nearest labelled voxel for each point (0 if > max_dist mm)."""
    zooms = img.header.get_zooms()[:3]
    ijk = nib.affines.apply_affine(np.linalg.inv(img.affine), points)
    pad = int(np.ceil(max_dist / min(zooms))) + 2
    lo = np.maximum(np.floor(ijk.min(0)).astype(int) - pad, 0)
    hi = np.minimum(np.ceil(ijk.max(0)).astype(int) + pad + 1, img.shape)
    lab = np.asarray(img.dataobj[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]])
    lab = lab.astype(np.int32)
    idx = ndimage.distance_transform_edt(
        lab == 0, sampling=zooms, return_distances=False, return_indices=True
    )
    v = np.clip(np.rint(ijk).astype(int) - lo, 0, np.array(lab.shape) - 1)
    near = idx[:, v[:, 0], v[:, 1], v[:, 2]]
    out = lab[tuple(near)]
    xyz = nib.affines.apply_affine(img.affine, near.T + lo)
    out[np.linalg.norm(xyz - points, axis=1) > max_dist] = 0
    return out


class Tree:
    """Branch graph with per-branch geometry seen from either end."""

    def __init__(self, net):
        self.P = np.asarray(net.points)
        self.R = np.asarray(net.point_data["radius"])
        self.polys = polylines(net)
        self.mean_r = np.asarray(net.cell_data["mean_radius"])
        self.length = np.asarray(net.cell_data["length"])
        key, self.ends = {}, []
        for ids in self.polys:
            a, b = (
                key.setdefault(tuple(np.round(self.P[i], 2)), len(key)) for i in (ids[0], ids[-1])
            )
            self.ends.append((a, b))
        self.X = np.zeros((len(key), 3))
        for (a, b), ids in zip(self.ends, self.polys):
            self.X[a], self.X[b] = self.P[ids[0]], self.P[ids[-1]]
        G = nx.MultiGraph()
        for c, (a, b) in enumerate(self.ends):
            G.add_edge(a, b, c=c)
        if not nx.is_connected(G):
            n_comp = nx.number_connected_components(G)
            raise SystemExit(f"error: branch graph has {n_comp} components, not one")
        if G.number_of_edges() != G.number_of_nodes() - 1:
            raise SystemExit(
                f"error: branch graph has {G.number_of_edges() - G.number_of_nodes() + 1} "
                "independent cycles; expected a tree (run extract.py)"
            )
        self.T = nx.Graph(G)

    def other(self, c, v):
        a, b = self.ends[c]
        return b if v == a else a

    def incident(self, v):
        return [d["c"] for *_, d in self.T.edges(v, data=True)]

    def oriented(self, c, v):
        """Point ids of branch c starting at its end node v."""
        ids = self.polys[c]
        return ids if self.ends[c][0] == v else ids[::-1]

    def arc(self, ids):
        return np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.P[ids], axis=0), axis=1))]

    def r_near(self, c, v, lo=1.0, hi=6.0):
        """Mean radius 1-6 mm from node v (the junction bulge excluded)."""
        ids = self.oriented(c, v)
        s = self.arc(ids)
        sel = (s >= lo) & (s <= hi)
        return float(self.R[ids][sel].mean() if sel.any() else self.R[ids].mean())

    def heading(self, c, v, n, at=8.0):
        """Displacement along n from node v to the point `at` mm along branch c."""
        ids = self.oriented(c, v)
        s = self.arc(ids)
        return float((self.P[ids[min(np.searchsorted(s, at), len(ids) - 1)]] - self.X[v]) @ n)

    def path_edges(self, u, v):
        nodes = nx.shortest_path(self.T, u, v)
        return nodes, [self.T[a][b]["c"] for a, b in pairwise(nodes)]


def downstream(tree, root, w):
    """Sum of w over each branch and everything beyond it, seen from root."""
    order = list(nx.bfs_edges(tree.T, root))
    into = {v: tree.T[u][v]["c"] for u, v in order}
    tot = {}
    for u, v in reversed(order):
        tot[into[v]] = w[into[v]] + sum(tot[into[x]] for x in tree.T[v] if x != u)
    return tot


def walk(tree, v, skip, n, feeds, args):
    """Thickest continuation from node v until the trunk splits: two children of
    comparable radius, heading to opposite sides, each feeding >= --min_territory
    mm of the trunk's target territory (feeds: branch -> mm downstream of v).
    Returns (branches, end node)."""
    path, e, total = [], None, 0.0
    while True:
        kids = [c for c in tree.incident(v) if c != e and c not in skip and c not in path]
        if not kids:
            return path, v
        kids.sort(key=lambda c: -tree.r_near(c, v))
        if len(kids) > 1 and path:
            c1, c2 = kids[:2]
            h1, h2 = tree.heading(c1, v, n), tree.heading(c2, v, n)
            if (
                tree.r_near(c2, v) >= args.split_ratio * tree.r_near(c1, v)
                and h1 * h2 < 0
                and min(abs(h1), abs(h2)) >= 1.0
                and min(feeds[c1], feeds[c2]) >= args.min_territory
            ):
                return path, v
        e = kids[0]
        path.append(e)
        total += tree.length[e]
        v = tree.other(e, v)
        if total > args.trunk_max:
            print(f"warning: trunk walk exceeded {args.trunk_max} mm", flush=True)
            return path, v


def label_tree(tree, U, TR, root, TRv):
    """Exact min-sum labelling of the rooted tree: unary U[c], pairwise
    TR[k_parent, k_child] (TRv[node] instead at the nodes it lists)."""
    parent = dict(nx.bfs_predecessors(tree.T, root))
    order = list(nx.bfs_edges(tree.T, root))
    into = {v: tree.T[u][v]["c"] for u, v in order}
    kids = {v: [w for w in tree.T[v] if parent.get(v) != w] for v in tree.T}
    M = {}
    for _, v in reversed(order):
        c = into[v]
        T = TRv.get(v, TR)
        M[c] = U[c] + sum(np.min(T + M[into[w]][None, :], axis=1) for w in kids[v])
    lab = {}
    for w in kids[root]:
        lab[into[w]] = int(np.argmin(M[into[w]]))
    for _, v in order:
        for w in kids[v]:
            lab[into[w]] = int(np.argmin(TRv.get(v, TR)[lab[into[v]]] + M[into[w]]))
    return lab, into, kids, parent


def point_edges(polys):
    """``(edges (m, 2), cell (m,))``: consecutive points of each polyline."""
    edges = np.vstack([np.c_[ids[:-1], ids[1:]] for ids in polys])
    cell = np.repeat(np.arange(len(polys)), [len(ids) - 1 for ids in polys])
    return edges, cell


def inlet_points(points, edges, xyz):
    """The tip (degree-1 point) nearest each inlet coordinate."""
    deg = np.bincount(edges.ravel(), minlength=len(points))
    tips = np.flatnonzero(deg == 1)
    return tips[cKDTree(points[tips]).query(np.asarray(xyz))[1]]


def recut_loops(net, seg, inlet_xyz):
    """Re-cut the loops of one territory where their blood flows meet.

    extract.py writes, as field data ``loop_returns``, the point pairs where
    its tree comes back to a node its own path from the inlet left (a copy of
    the node, offset by 0.03 mm). Each pair whose loop (the tree path between
    the two copies) lies inside one territory label (ACA, MCA-L/R, PCA-L/R,
    cerebellar) is joined; the Poiseuille pressure (blood.py, fitted radius,
    p = 1 at the inlets, 0 at the other tips) is solved, and the loop is cut
    again at its point of lowest pressure, detaching the loop edge there that
    carries less blood (it ends at a new copy of the point). So the vessels
    around the loop run the way the blood does, not the way the selection
    happened to close it. Loops touching a trunk or the CoW are left alone:
    there a return is as often a contact between two vessels (sub-01: the
    basilar against a thinner artery, whose re-cut moved the basilar tip).
    Labels are unchanged, every new polyline is a piece of an old one.

    Returns ``(net, seg, report)``, ``net`` a new PolyData (polylines between
    junctions, tips and the old polyline ends, pointing down the pressure; cell
    data copied from the old polyline, ``branch``/``length``/``mean_radius``
    recomputed) and ``seg`` its labels; ``net`` is the input when nothing
    changes. ``report`` lists the loops re-cut and skipped."""
    pairs = np.asarray(net.field_data.get("loop_returns", np.zeros((0, 2))), np.int64)
    report = dict(recut=[], skipped=0)
    if not len(pairs):
        return net, seg, report
    P = np.asarray(net.points, dtype=np.float64)
    R = np.asarray(net.point_data["radius"], dtype=np.float64)
    polys = polylines(net)
    E, cell = point_edges(polys)
    ends = {int(i) for ids in polys for i in (ids[0], ids[-1])}
    inlets = inlet_points(P, E, inlet_xyz)
    territory = set(TERRITORY) - {UNK}
    src = list(range(len(P)))  # point -> the point whose data it copies
    pos = [P]
    changed = False
    for pr, pu in pairs.tolist():
        G = nx.Graph()
        G.add_edges_from((int(a), int(b), {"e": k}) for k, (a, b) in enumerate(E))
        if pr not in G or pu not in G or pr == pu:
            continue
        path = nx.shortest_path(G, pu, pr)
        loop_e = [G.edges[a, b]["e"] for a, b in pairwise(path)]
        labels = {int(seg[cell[k]]) for k in loop_e}
        if len(path) < 3 or len(labels) != 1 or not labels <= territory:
            report["skipped"] += 1
            continue
        E[E == pr] = pu  # join the return to its node
        pts = np.vstack(pos)
        p, Gc = network_pressure(pts, E, R[src], inlets)
        cyc = path[:-1]  # pu ... (pr joined into pu)
        i = int(np.argmin(p[cyc]))
        m = cyc[i]
        e_in, e_out = loop_e[i - 1], loop_e[i]  # the loop edges at m (pu: last, first)
        flow = lambda k: Gc[k] * abs(p[E[k, 0]] - p[E[k, 1]])
        k = min((e_in, e_out), key=flow)
        new = len(src)
        src.append(src[m])
        pos.append(pts[m] + 0.03)
        E[k][E[k] == m] = new
        ends |= {m, new}
        changed = True
        report["recut"].append(
            dict(
                label=NAMES[labels.pop()],
                loop_mm=round(float(np.linalg.norm(np.diff(P[path], axis=0), axis=1).sum()), 1),
                closed_xyz=P[pu].round(1).tolist(),
                cut_xyz=pts[m].round(1).tolist(),
            )
        )
    if not changed:
        return net, seg, report

    # polylines of the new graph, each oriented down the pressure
    pts = np.vstack(pos)
    p, _ = network_pressure(pts, E, R[src], inlets)
    G = nx.Graph()
    G.add_edges_from((int(a), int(b), {"e": k}) for k, (a, b) in enumerate(E))
    special = {n for n in ends if n in G} | {n for n in G if G.degree(n) != 2}
    chains, seen = [], set()
    for s0 in sorted(special):
        for nb in G.neighbors(s0):
            k = G.edges[s0, nb]["e"]
            if k in seen:
                continue
            ids, es = [s0, nb], [k]
            seen.add(k)
            while ids[-1] not in special:
                nxt = next(m for m in G.neighbors(ids[-1]) if m != ids[-2])
                es.append(G.edges[ids[-1], nxt]["e"])
                seen.add(es[-1])
                ids.append(nxt)
            if p[ids[-1]] > p[ids[0]]:
                ids = ids[::-1]
            chains.append((ids, int(cell[es[0]])))
    used = np.unique(np.concatenate([ids for ids, _ in chains]))
    new_id = np.full(len(pts), -1)
    new_id[used] = np.arange(len(used))
    lines = [new_id[ids] for ids, _ in chains]
    out = pv.PolyData(pts[used], lines=np.concatenate([np.r_[len(i), i] for i in lines]))
    srcs = np.asarray(src)[used]
    for k in net.point_data:
        out.point_data[k] = np.asarray(net.point_data[k])[srcs]
    old = np.array([c for _, c in chains])
    for k in net.cell_data:
        out.cell_data[k] = np.asarray(net.cell_data[k])[old]
    out.cell_data["branch"] = np.arange(len(lines))
    out.cell_data["length"] = np.array(
        [np.linalg.norm(np.diff(out.points[i], axis=0), axis=1).sum() for i in lines]
    )
    out.cell_data["mean_radius"] = np.array([out["radius"][i].mean() for i in lines])
    for k in net.field_data:
        if k != "loop_returns":
            out.field_data[k] = net.field_data[k]
    return out, np.asarray(seg)[old], report


def blood_pressure(net, inlet_xyz):
    """Poiseuille blood pressure per point of the tree (:mod:`vesselseg.tree.blood`,
    fitted radius): 1 at the tips nearest the inlets, 0 at every other tip."""
    P = np.asarray(net.points)
    E, _ = point_edges(polylines(net))
    p, _ = network_pressure(P, E, net.point_data["radius"], inlet_points(P, E, inlet_xyz))
    return p


def main(argv=None):
    p = argparse.ArgumentParser(prog="vesselseg label", description=__doc__.split("\n")[0])
    p.add_argument("tree", help="path stem DIR/NAME of the `vesselseg extract` output")
    p.add_argument("--sasparc", required=True, help="sasparc labels (T1w space): territory prior")
    p.add_argument("--split_ratio", type=float, default=0.6, help="terminus / tip")
    p.add_argument("--trunk_max", type=float, default=200.0, help="mm")
    p.add_argument("--min_territory", type=float, default=20.0, help="mm fed")
    p.add_argument("--cow_max", type=float, default=60.0, help="mm per CoW piece")
    p.add_argument("--prior_dist", type=float, default=3.0, help="mm to a label")
    p.add_argument("--trunk_cost", type=float, default=2.0, help="mm-equivalents")
    p.add_argument("--unknown_cost", type=float, default=4.0, help="mm-equivalents")
    p.add_argument("--switch", type=float, default=15.0, help="mm-equivalents")
    args = p.parse_args(argv)
    outdir, stem = Path(args.tree).parent, Path(args.tree).name

    net = pv.read(outdir / f"{stem}.vtp")
    tree = Tree(net)
    img = nib.load(args.sasparc)
    centre, n = midplane(img)
    side = lambda x: float((np.asarray(x) - centre) @ n)
    print(f"midplane centre {np.round(centre, 1)}, normal {np.round(n, 3)}", flush=True)

    meta = json.loads((outdir / f"{stem}.json").read_text())
    if "inlets_xyz" not in meta:
        raise SystemExit(f"error: {stem}.json has no inlets_xyz (not from extract.py)")
    node_of = {}
    for name, xyz in meta["inlets_xyz"].items():
        d = np.linalg.norm(tree.X - np.asarray(xyz), axis=1)
        node_of[name] = int(np.argmin(d))
        if d.min() > 1.0:
            print(f"warning: inlet {name} is {d.min():.1f} mm from the nearest node")
    src_pt = np.flatnonzero(np.asarray(net.point_data["node_type"]) == 2)
    source = int(np.argmin(np.linalg.norm(tree.X - tree.P[src_pt[0]], axis=1)))
    va = [source] + [v for k, v in node_of.items() if k.startswith("VA") and v != source]
    ica = {s: node_of[NAMES[s]] for s in (ICA_R, ICA_L) if NAMES[s] in node_of}
    print(f"inlets: VA {[tree.X[v].round(1).tolist() for v in va]}", flush=True)
    for s, v in ica.items():
        print(f"        {NAMES[s]} {tree.X[v].round(1).tolist()}", flush=True)
    for s in (ICA_L, ICA_R):
        if s not in ica:
            print(f"warning: no {NAMES[s]} inlet in {stem}.json", flush=True)

    # prior: classes allowed next to each centerline point
    lab_pt = nearest_labels(tree.P, img, args.prior_dist)
    K = len(NAMES)
    allowed = np.zeros((len(tree.P), K), bool)
    informed = lab_pt > 0
    for i in np.flatnonzero(informed):
        ks = classes(int(lab_pt[i]), side(tree.P[i]))
        if ks is None:
            informed[i] = False
        else:
            allowed[i, ks] = True
    cost_pt = np.where(allowed, 0.0, 1.0)
    cost_pt[~informed] = 0.25
    cost_pt[~informed, UNK] = 0.0
    ds = []  # centerline length per point, per branch
    for ids in tree.polys:
        d = np.diff(tree.arc(ids))
        ds.append(np.r_[d, 0] / 2 + np.r_[0, d] / 2)

    def feeds(root, targets):
        # only points whose label allows the targets alone (not e.g. brainstem,
        # which allows cerebellar as well as PCA)
        others = [k for k in range(K) if k not in targets]
        hit = informed & allowed[:, targets].any(1) & ~allowed[:, others].any(1)
        w = [ds[c] @ hit[ids] for c, ids in enumerate(tree.polys)]
        return downstream(tree, root, w)

    # 3. trunks
    fixed, ends, interior = {}, {}, {}
    for s, v in ica.items():
        tgt = [ACA, MCA_L if s == ICA_L else MCA_R]
        path, end = walk(tree, v, set(), n, feeds(v, tgt), args)
        fixed.update(dict.fromkeys(path, s))
        ends[f"{NAMES[s]} terminus"] = end
        interior[s] = {tree.ends[c][k] for c in path for k in (0, 1)} - {end}
    vb = set()
    conf = va[0]
    if len(va) > 1:
        target = next(iter(ica.values()), int(np.argmax(tree.X[:, 2])))
        p12, _ = tree.path_edges(va[0], va[1])
        p1t, _ = tree.path_edges(va[0], target)
        i = 0
        while i + 1 < min(len(p12), len(p1t)) and p12[i + 1] == p1t[i + 1]:
            i += 1
        conf = p12[i]
    for v in va:
        vb.update(tree.path_edges(v, conf)[1])
    path, tip = walk(tree, conf, vb | set(fixed), n, feeds(va[0], [PCA_L, PCA_R]), args)
    vb.update(path)
    fixed.update(dict.fromkeys(vb, VB))
    ends["VA confluence"], ends["basilar tip"] = conf, tip
    interior[VB] = {tree.ends[c][k] for c in vb for k in (0, 1)} - {tip}

    # 4. CoW: backbone pieces between the trunks
    inlets = va + list(ica.values())
    backbone = set()
    for i, u in enumerate(inlets):
        for v in inlets[i + 1 :]:
            backbone.update(tree.path_edges(u, v)[1])
    rest = nx.Graph()
    for c in backbone - set(fixed):
        rest.add_edge(*tree.ends[c], c=c)
    cow_pieces = []
    for comp in nx.connected_components(rest):
        cs = [d["c"] for *_, d in rest.subgraph(comp).edges(data=True)]
        L = float(tree.length[cs].sum())
        cow_pieces.append(dict(branches=sorted(cs), length_mm=round(L, 1), cow=L <= args.cow_max))
        if L <= args.cow_max:
            fixed.update(dict.fromkeys(cs, COW))

    # 5. territories
    U = {}
    for c, ids in enumerate(tree.polys):
        u = np.full(K, np.inf)
        if c in fixed:
            u[fixed[c]] = 0.0
        else:
            u[TERRITORY] = ds[c] @ cost_pt[ids][:, TERRITORY]
        U[c] = u
    TR = np.full((K, K), args.switch)
    TR[np.ix_(TRUNK, TRUNK)] = 0.0
    feeds_of = {
        ICA_L: [ACA, MCA_L, PCA_L, UNK],
        ICA_R: [ACA, MCA_R, PCA_R, UNK],
        VB: [PCA_L, PCA_R, CRB, UNK],
        COW: [ACA, MCA_L, MCA_R, PCA_L, PCA_R, CRB, UNK],
    }
    for t, bs in feeds_of.items():
        TR[t, bs] = TR[bs, t] = args.trunk_cost
    # unknown only as a leaf: nothing hangs off it (rows are the parent, which is
    # upstream since the root is the VMTK inlet)
    TR[TERRITORY[1:], UNK] = args.unknown_cost
    np.fill_diagonal(TR, 0.0)
    # ACA/MCA only leave the ICA at its terminus, PCA only the basilar at its tip
    TRv = {}
    for t, late in [(ICA_L, [ACA, MCA_L]), (ICA_R, [ACA, MCA_R]), (VB, [PCA_L, PCA_R])]:
        Tt = TR.copy()
        Tt[t, late] = Tt[late, t] = args.switch
        for v in interior.get(t, ()):
            TRv[v] = Tt
    lab, into, kids, _ = label_tree(tree, U, TR, source, TRv)
    seg = np.array([lab[c] for c in range(len(tree.polys))], dtype=np.int8)

    switches = []  # junctions where the labelling paid --switch
    for v, ws in kids.items():
        if v not in into:
            continue
        a = seg[into[v]]
        for w in ws:
            b = seg[into[w]]
            if TRv.get(v, TR)[a, b] >= args.switch:
                switches.append(
                    dict(
                        xyz=tree.X[v].round(1).tolist(),
                        branches=[int(into[v]), int(into[w])],
                        labels=[NAMES[a], NAMES[b]],
                    )
                )

    # loops inside one territory, re-cut by blood pressure
    inlet_xyz = tree.X[inlets]
    out, seg_out, loops = recut_loops(net, seg, inlet_xyz)
    print(
        f"loops: {len(loops['recut'])} re-cut inside a territory, "
        f"{loops['skipped']} left (trunk/CoW or several labels)",
        flush=True,
    )

    # outputs
    out = out.copy()
    out.cell_data["segment"] = seg_out
    out.point_data["blood_pressure"] = blood_pressure(out, inlet_xyz)
    seg_pt = np.zeros(out.n_points, np.int8)
    for c, ids in enumerate(polylines(out)):
        seg_pt[ids] = seg_out[c]
    out.point_data["segment"] = seg_pt
    out.field_data["segment_names"] = np.array(NAMES)
    out.save(outdir / f"{stem}_labeled.vtp")
    tubes_path = outdir / f"{stem}_tubes.vtp"
    if tubes_path.exists():
        tubes = pv.read(tubes_path)
        # the tubes keep the branches of NAME.vtp (a re-cut loop only splits one)
        tubes.cell_data["segment"] = seg[np.asarray(tubes.cell_data["branch"])]
        tubes.field_data["segment_names"] = np.array(NAMES)
        tubes.save(outdir / f"{stem}_tubes_labeled.vtp")

    stats = dict(
        args=vars(args),
        midplane=dict(centre=centre.round(2).tolist(), normal=n.round(4).tolist()),
        inlets=dict(
            VA=[tree.X[v].round(1).tolist() for v in va],
            **{NAMES[s]: tree.X[v].round(1).tolist() for s, v in ica.items()},
        ),
        landmarks={k: tree.X[v].round(1).tolist() for k, v in ends.items()},
        cow_pieces=cow_pieces,
        segments={
            NAMES[k]: dict(
                branches=int((seg == k).sum()),
                length_mm=round(float(tree.length[seg == k].sum()), 1),
            )
            for k in range(K)
        },
        implausible_junctions=switches,
        loops=loops,
    )
    (outdir / f"{stem}_labeled.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: stats[k] for k in ("landmarks", "cow_pieces", "segments")}, indent=2))
    print(f"{len(switches)} implausible junctions (suspected false contacts)")
    report_inlet_trees(net, meta, seg, tree.length)
    plot(tree, seg, va, ica, ends, outdir / f"{stem}_labeled.png")


def report_inlet_trees(net, meta, seg, length):
    """Print the share of each label's length that lies in each inlet tree."""
    if "inlet" not in net.cell_data:
        return
    inlet = np.asarray(net.cell_data["inlet"])
    names = list(net.field_data["inlet_names"])
    print("label length (mm) by extract_tree inlet tree (join = -1):")
    print(f"  {'':11s}" + "".join(f"{nm:>8s}" for nm in names) + f"{'join':>8s}")
    for k in range(len(NAMES)):
        m = seg == k
        if not m.any():
            continue
        row = [length[m & (inlet == i)].sum() for i in [*range(len(names)), -1]]
        print(f"  {NAMES[k]:11s}" + "".join(f"{x:8.1f}" for x in row))


def plot(tree, seg, va, ica, ends, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    _, ax = plt.subplots(1, 3, figsize=(27, 9))
    segs, cols, widths = [], [], []
    for c, ids in enumerate(tree.polys):
        q = tree.P[ids]
        segs += list(pairwise(q))
        cols += [COLORS[seg[c]]] * (len(ids) - 1)
        widths += list(1.2 * tree.R[ids[:-1]])
    segs = np.array(segs)
    marks = [
        (va, "^", "inlet"),
        (list(ica.values()), "^", None),
        (list(ends.values()), "*", "terminus / tip"),
    ]
    for k, (i, j, title) in enumerate(
        [(0, 1, "axial (x, y)"), (0, 2, "coronal (x, z)"), (1, 2, "sagittal (y, z)")]
    ):
        ax[k].add_collection(LineCollection(segs[:, :, [i, j]], colors=cols, linewidths=widths))
        for nodes, m, lbl in marks:
            x = tree.X[nodes]
            ax[k].scatter(
                x[:, i],
                x[:, j],
                marker=m,
                s=120,
                c="gold",
                edgecolors="k",
                zorder=5,
                label=lbl,
            )
        ax[k].autoscale()
        ax[k].set_aspect("equal")
        ax[k].set_title(title)
        ax[k].grid(alpha=0.3)
    handles = [plt.Line2D([], [], color=c, lw=4, label=nm) for nm, c in zip(NAMES, COLORS)]
    ax[2].legend(handles=handles, loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=70)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
