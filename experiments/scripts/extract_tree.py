"""Arterial tree on the native T2w grid by one global selection (no local heuristics).

Pipeline: prep_inputs -> run_riscmi -> register_to_t1 -> build_evidence ->
extract_tree -> fit_radius -> label_centerlines. Replaces extract_centerlines.py's
contact separation, loop breaking and spur pruning, and build_network.py's
binarisation and gap bridging, by one integer program over an over-complete set
of candidate pieces (after Tuertken et al., TPAMI 2016).

0. Inputs (build_evidence.py output, results/sub-01/network/t2space/, one grid of
   ~0.35 x 0.33 x 0.33 mm, anisotropic: every length, EDT, arc length, bridge
   box and merge length below uses the voxel spacing). Evidence
   P = max(T2w, consensus_other) inside the head ROI (--t2_only: T2w alone):
   T2w is the densest and the only one that finds the posterior vessels, but
   ~75% of both ICA trunks are invisible in it (P < 0.005), which the
   consensus of the other contrasts covers. Candidate mask: hysteresis on P
   (seeds >= --seed, grown into >= --grow), components of >= --min_cc voxels;
   permissive on purpose, the selection decides what is a vessel.
   Radius = distance to the boundary of a stricter mask (EDT - h/2, h the mean
   spacing; at least h/2): the candidates with P >= --r_frac x the maximum of P
   within --r_window mm (0.5 mm: +-1-2 voxels), i.e. each vessel at the half
   maximum of its own peak. The permissive mask is a halo 2-4x too wide (median
   skeleton radius 0.9 mm); with the half-maximum mask the median radius at the
   skeleton points of the old T1-grid tree's radius classes 1.2-2 mm and > 2 mm is
   1.53 and 2.04 mm, and 1.34 mm on the basilar (old tree 1.32 mm). A wider window
   (2 mm) lets a bright neighbour raise the threshold: the basilar drops to
   0.89 mm and the thin vessels (P peak < 1) to the floor h/2 = 0.17 mm; with
   0.5 mm they get 0.30 mm. Vessels below ~0.3 mm are ~1-2 voxels wide and
   their radius is barely measurable (the old floor of 0.25 mm was a resolution
   artefact too); fit_radius.py then imposes Murray's law on them.
   --network NAME instead takes build_network.py's binary mask and consensus on
   the T1w grid (old input).
1. Candidate segments: voxel skeleton of the mask, split at junctions.
   Junction clusters joined by a segment shorter than their radius (what two
   vessels touching or crossing leave) are merged into one node, so a crossing
   becomes one node where the vessels can pass straight through.
2. Candidate bridges: from every free end, curvature-penalised minimal paths
   (Reeds-Shepp forward model, HamiltonFastMarching via agd) in a box of
   --max_gap mm, resampled to an isotropic grid of --hfm_step mm (the
   orientation sphere needs one grid scale), seeded with the end's outgoing
   direction, through the metric 1 + max(0, -w(P)). The --bridges cheapest
   paths to other segments that are a real shortcut (along the skeleton >= 2x
   farther, or another component) become candidate segments; the segment they
   reach is split there. Bridges may join separate skeleton components. The
   selection drops a bridge whose ends the skeleton already joins within its
   length + --bridge_detour mm: at the native T2w resolution the test above lets
   1-2 mm parallel shortcuts through (sub-01: a 2.5 mm bridge into a thick
   neighbouring fragment beside a faint ACA stretch, P 0.05-0.3, took the ACA
   off its real course and left the course as a dead end).
3. Selection: every segment can be used in either direction (an arc), and every
   used arc has exactly one parent arc ending at its start node (or is an inlet
   arc), so the result is a tree grown from the inlets. The mixed-integer
   program (HiGHS) minimises
     sum over used arcs      --alpha - reward(arc)   (+ --beta + excess for a bridge)
     sum over dead-end arcs  --w_leaf   (a used arc without a child)
     sum over parent->child  --w_turn (1 - cos(turn)) / 2
                             + growth(log(r_child / r_parent) - log --r_tol)
     + --w_cap rho x (subtree length beyond slack x c0 r^3, or short of
                   c0 r^3 / slack),  slack = --cap_slack^(sigma(r) / sigma0)
   with growth, sigma and rho from the radius measurement model of
   radius_error.py: the growth cost is
   --w_rad per unit log ratio for a resolved full-evidence pair, divided by
   the pair's relative error and bounded by the evidence (a faint or barely
   resolved parent says little about its child), and the capacity band widens
   with the radius error and weighs a faint segment less;
   reward = integral of w(P) = clip(log(P / --p0), -2, 1) along the arc,
   leaving out the first --bulge x the start node's radius (so skeleton spurs
   inside a thick vessel earn nothing). The turn is measured between the
   directions --reach mm beyond the junction bulge. Two arcs may end at the
   same node (the vessels touch there, or cross: each continues with its own
   child), and an arc has at most --max_children children. A flow from the
   inlets, of which each used arc consumes its length, rules out cycles and
   measures the centerline length each arc feeds: by Murray's law it should
   scale with r^3 (c0 = length of the candidates an inlet can reach / sum of the
   inlets' r^3). This is what keeps the ICA feeding the MCA through a 90 degree
   T-junction, rather than the thin ACom through the straight A1-M1 line.
   The dead-end term is there because --alpha (1 mm-equivalent) is repaid by any
   spur longer than 1-2 mm (w saturates at 1 per mm): without it the tree keeps ~23
   dead ends under 3 mm of radius < 0.25 mm (~60 mm), often the start of a parallel
   route ~1 mm beside the one used. A continuous variable leaf_a >= y_a - sum of the
   transitions out of a is 1 exactly for a used arc without a child (no binaries);
   a true vessel end pays it once too, repaid by its length. --w_leaf 1 (default 0:
   --prune_len does this job, see below) removes 22 of 23 such spurs (-180 mm,
   2%) and keeps the ACA link; 2 cuts it.
   Two exact reductions keep the program small. (i) An arc u->v is dropped if u
   cannot be reached from an inlet by a directed walk of kept arcs avoiding the
   arc's own segment (a used arc needs a parent chain back to an inlet, and a
   segment is used in one direction only); iterated to a fixed point. Outside
   the 2-core this forces the orientation away from the inlets: a bridge whose
   far side holds no inlet is only used pointing there, one with inlets on both
   sides keeps both directions. (ii) The flow of an arc is bounded by its
   own length plus the length of all segments reachable downstream of its head
   (not through its own segment), instead of the total candidate length; the
   capacity slacks are bounded accordingly.
   There is no loop breaking, spur pruning or contact separation step: every
   one of those decisions is this single program. Solve: HiGHS without the
   capacity terms (fast, to --mip_gap or --time_limit; or the tree of another
   run on the same candidates, --warm), completed (flows, slacks) as the start
   of a region search: boxes of --lns_box mm on a grid of half that stride,
   each solved exactly with every binary outside it fixed to the incumbent
   (all flows free), in parallel on --workers cores, improving boxes merged,
   until no box improves (region_search). The full MIP does not close its gap
   (sub-01: 1-4% after 600 s, and the outcome then depends on where it
   stopped); the region optimum is deterministic for a given start, except for
   the boxes that hit --lns_box_time (counted in the json). sub-01: 40 mm
   boxes, 30 s each, ~8 min on 64 genoaxq cores (20 + 40 mm boxes at 120 s:
   17 min, a worse objective).
   --prune_len L (default 3) cuts the short thin dead ends after the selection
   instead of --w_leaf inside it: a dead end, from its tip up to the arc it
   branches off, goes if shorter than L mm (along the voxel path) with median
   radius below --prune_r mm, repeated until none is left. Only dead ends go, so
   the routing stays as the selection chose it; the dead-end rows make the
   capacity program harder (sub-01: 4.1% gap after 600 s with --w_leaf 1
   --bridge_detour 5, and the left carotid siphon cut short, against 2.0%
   without; these numbers are from the earlier fixed-rule radius and full MIP).
4. Joins: the result is one tree per inlet (which inlet feeds what). They are
   joined into one piece where they meet at a node (thickest node first),
   else by the cheapest path of unused segments (bridge candidates included)
   between them (cost length / radius: the communicating artery the selection
   left out), once per pair of trees, so no loop is closed.
5. Output, mapped to T1w world through t2_to_t1.json (proteoflow, the sasparc
   labels and the mesh live there): one polyline per unbranched chain of arcs.
   Coordinates are smoothed over --smooth_window mm; the radius is the raw
   value per point (fit_radius.py smooths it). Where several arcs of one tree
   end at or pass through one node, each gets its own copy of the node point,
   offset by 0.03 mm (label_centerlines.py joins branches by rounded
   coordinates, proteoflow within 1e-6 mm), so the tree stays a tree.

Inlets (in T1w world): free ends of skeleton components of >= --inlet_min_comp mm,
one of each per side of the mid-sagittal plane (between the sasparc cortex
centroids, as in label_centerlines.py). Vertebrals (VA-L/R): where the head ROI
cuts them, i.e. free ends within --va_band mm of the bottom of the mask; the
thickest, by mean radius 1 to --va_reach mm along the thickest continuation, if
>= --r_va. The lower one is the source (node_type 2). ICAs: of the other free
ends that head upwards (--inlet_reach mm along the vessel is higher than the
end; the ICAs end in the skull base, well above the cut) and whose 90th
percentile radius 1 to --ica_reach mm along the vessel is >= --r_ica, the lowest
one per side (a low-evidence tail of the skeleton can continue the trunk below
the skull base, hence the percentile).

Output (results/sub-01/network/), same format as extract_centerlines.py:
  NAME.vtp        point data radius, node_type; cell data branch, length,
                  mean_radius, origin (1 = contains a bridge, 2 = joins two
                  inlet trees), inlet (index into field data inlet_names:
                  the inlet whose tree the branch belongs to, -1 for joins)
  NAME_tubes.vtp  tube around each branch
  NAME.json       inputs, candidate / selection counts, objective, bridges
                  used, shared nodes (contacts and crossings), length per
                  inlet tree, the joins, inlets_xyz, timings per stage
  NAME_candidates.pkl, NAME_selection.pkl   caches for --reuse (steps 0-2)
                  and --reuse_selection (step 3)

--p0 sets which P counts as evidence. The default 0.005 is insensitive at
native resolution: every candidate has P >= --grow = 0.01, w saturates at 1 from
P = 0.0136, and only 33 mm of the 8.5 m skeleton has a mean P < 0.02 (251 mm
< 0.05), so p0 acts on the bridge metric (P < p0 costs up to 3x) rather than
on which segments are kept.

usage: pixi run build-hfm   (once)
       pixi run python extract_tree.py                  (NAME centerlines_tree_t2native)
       pixi run python fit_radius.py --name centerlines_tree_t2native
       pixi run python label_centerlines.py --name centerlines_tree_t2native
"""

import argparse
import json
import multiprocessing
import os
import pickle
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import nibabel as nib
import numpy as np
import pyvista as pv
from common import RESULTS, ROOT, SASPARC
from label_centerlines import midplane
from radius_error import growth_cost, rel_var, validity
from scipy import ndimage, sparse
from scipy.spatial import cKDTree
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import skeletonize

OUT = RESULTS / "network"
HFM_BIN = ROOT / "external/HamiltonFastMarching/build"
CONN = np.ones((3, 3, 3))
OFFS = np.argwhere(CONN) - 1
OFFS = OFFS[np.any(OFFS != 0, axis=1)]


def path_length(pts, sp):
    """Length (mm) of a polyline in voxel coordinates on a grid of spacing sp."""
    return float(np.linalg.norm(np.diff(pts, axis=0) * sp, axis=1).sum())


def weight(p, p0):
    """Evidence per mm of centerline: > 0 where the vessel is likelier than p0."""
    return np.clip(np.log(np.maximum(p, 1e-6) / p0), -2.0, 1.0)


# ---------------------------------------------------------------------------
# 1. candidate segments


def skeleton_segments(mask, r_vox, sk=None, r_size=None):
    """Skeleton split at junctions: nodes (voxel position, radius), segments.

    Segments are voxel paths (int index arrays, including the two node voxels)
    between junction clusters (26-connected voxels with >= 3 neighbours) or free
    ends. Returns node_pos (voxel coords), node_r (mm, from r_vox), node_size
    (mm, from r_size, default r_vox), segments [(u, v, path)] and the skeleton.
    """
    sk = skeletonize(mask).astype(bool) if sk is None else sk
    deg = ndimage.convolve(sk.astype(np.uint8), CONN.astype(np.uint8), mode="constant")
    deg = (deg.astype(int) - 1) * sk
    jlab, nj = ndimage.label(deg >= 3, CONN)
    node_of = np.full(sk.shape, -1, np.int32)
    node_of[jlab > 0] = jlab[jlab > 0] - 1
    ends = np.argwhere(sk & (deg <= 1))
    node_of[tuple(ends.T)] = nj + np.arange(len(ends))
    n_nodes = nj + len(ends)
    node_vox = [[] for _ in range(n_nodes)]
    for v in np.argwhere(node_of >= 0):
        node_vox[node_of[tuple(v)]].append(v)
    node_pos = np.array([np.mean(v, axis=0) for v in node_vox])
    node_r = np.array([max(r_vox[tuple(x)] for x in v) for v in node_vox])
    r_size = r_vox if r_size is None else r_size
    node_size = np.array([max(r_size[tuple(x)] for x in v) for v in node_vox])

    shape = np.array(sk.shape)

    def nbrs(v):
        w = v + OFFS
        w = w[np.all((w >= 0) & (w < shape), axis=1)]
        return w[sk[tuple(w.T)]]

    # every interior voxel goes to one segment: where the skeleton has a small
    # knot, walks from its two sides can otherwise take overlapping routes
    seen, taken, segments = set(), set(), []
    for n in range(n_nodes):
        for v in node_vox[n]:
            for w in nbrs(v):
                if node_of[tuple(w)] == n or tuple(w) in taken:
                    continue
                path, prev, cur = [tuple(v)], tuple(v), tuple(w)
                while node_of[cur] < 0:
                    path.append(cur)
                    nxt = [
                        tuple(x)
                        for x in nbrs(np.array(cur))
                        if tuple(x) != prev
                        and tuple(x) not in path
                        and tuple(x) not in taken
                    ]
                    if not nxt:
                        break
                    # prefer a node voxel, then the face/edge neighbour
                    nxt.sort(
                        key=lambda x: (
                            node_of[x] < 0,
                            np.abs(np.subtract(x, cur)).sum(),
                        )
                    )
                    prev, cur = cur, nxt[0]
                if node_of[cur] < 0:
                    continue  # dangling (the rest of a knot another walk took)
                path.append(cur)
                key = (
                    frozenset(path[1:-1])
                    if len(path) > 2
                    else frozenset([tuple(path[0]), tuple(path[1])])
                )
                if key in seen:
                    continue
                seen.add(key)
                taken.update(path[1:-1])
                segments.append((n, int(node_of[cur]), np.array(path)))
    return node_pos, node_r, node_size, segments, sk


def merge_close_junctions(node_pos, node_r, node_size, segments, sp, min_len):
    """Contract segments between junctions shorter than max(min_len, r_u, r_v),
    r the junction's size (node_size)."""
    deg = np.zeros(len(node_pos), int)
    for u, v, _ in segments:
        deg[u] += 1
        deg[v] += 1
    parent = list(range(len(node_pos)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    keep = []
    for u, v, path in segments:
        length = path_length(path, sp)
        if (
            u != v
            and deg[u] >= 3
            and deg[v] >= 3
            and length <= max(min_len, node_size[u], node_size[v])
        ):
            parent[find(u)] = find(v)
        else:
            keep.append((u, v, path))
    root = np.array([find(a) for a in range(len(node_pos))])
    ids, new = np.unique(root, return_inverse=True)
    pos = np.array([node_pos[root == i].mean(axis=0) for i in ids])
    rad = np.array([node_r[root == i].max() for i in ids])
    size = np.array([node_size[root == i].max() for i in ids])
    out, n_loops = [], 0
    for u, v, path in keep:
        u, v = int(new[u]), int(new[v])
        length = path_length(path, sp)
        if u == v and length <= 2 * size[u] + 2 * sp.max():
            n_loops += 1  # a small loop the merge closed
            continue
        out.append((u, v, path))
    return pos, rad, size, out, len(node_pos) - len(pos), n_loops


# ---------------------------------------------------------------------------
# 2. candidate bridges (curvature-penalised minimal paths)


def _hfm_setup(workdir):
    from agd.Eikonal import FileIO, LibraryCall

    LibraryCall.binary_dir["FileHFM"] = str(HFM_BIN)
    if not getattr(FileIO, "_private_dir", False):
        orig = FileIO.WriteCallRead

        def wcr(inp, exe, binary_dir="", **kw):
            kw["working_dir"] = workdir
            return orig(inp, exe, binary_dir, **kw)

        FileIO.WriteCallRead = wcr
        FileIO._private_dir = True


def _hfm_box(cost, g, n_theta, xi):
    """dictIn on a cubic grid of step g; node m sits at physical m * g.

    The Reeds-Shepp model needs one gridScale for all axes (the orientation
    sphere is not anisotropic), hence the isotropic resampling in bridge_job.
    """
    from agd import Eikonal

    hfm = Eikonal.dictIn(
        {"model": "ReedsSheppForward3", "exportValues": 1, "verbosity": 0, "xi": xi}
    )
    sides = [[-0.5 * g, (n - 0.5) * g] for n in cost.shape]
    hfm.SetRect(sides=sides, dimx=cost.shape[0])
    hfm.nTheta = n_theta
    hfm["cost"] = np.ascontiguousarray(
        np.broadcast_to(cost[..., None, None], tuple(int(d) for d in hfm["dims"]))
    )
    return hfm


def angles(d):
    return np.arccos(np.clip(d[2], -1, 1)), np.arctan2(d[1], d[0]) % (2 * np.pi)


def bridge_job(job):
    """One free end: values of the curvature-penalised distance, then geodesics
    to the cheapest point of each of the best target segments.

    The cost box (voxels from `lo`) is resampled to a cubic grid of step g mm,
    in the frame physical = voxel * sp (a rigid image of the scanner frame, so
    directions and angles are preserved). Returns geodesics in voxel coords.
    """
    (e, x_e, d_e, cost, lo, targets, sp, g, n_theta, xi, n_keep, workdir) = job
    _hfm_setup(workdir)
    n = np.floor((np.array(cost.shape) - 1) * sp / g).astype(int) + 1
    grid = np.meshgrid(*[np.arange(k) * g / s_ for k, s_ in zip(n, sp)], indexing="ij")
    cost = ndimage.map_coordinates(cost, grid, order=1, mode="nearest")
    org = lo * sp  # physical position of the box corner
    th, ph = angles(d_e)
    seed = [*(x_e * sp - org), th, ph]
    hfm = _hfm_box(cost, g, n_theta, xi)
    hfm["seed"] = seed
    out = hfm.Run()
    val = np.asarray(out["values"])
    flat = val.reshape(*val.shape[:3], -1)
    best = []
    for seg, vox in targets.items():
        node = np.clip(np.round((vox * sp - org) / g).astype(int), 0, n - 1)
        v = flat[tuple(node.T)]  # (n_vox, n_orient)
        i, k = np.unravel_index(np.argmin(v), v.shape)
        if np.isfinite(v[i, k]):
            best.append((float(v[i, k]), seg, vox[i], node[i], k))
    best.sort(key=lambda b: b[0])
    best = best[:n_keep]
    if not best:
        return e, []
    ax = hfm.Axes()
    tips = []
    for _, _, _, node, k in best:
        ti, pi = np.unravel_index(k, val.shape[3:])
        tips.append([*(node * g), float(ax[3][ti]), float(ax[4][pi])])
    hfm = _hfm_box(cost, g, n_theta, xi)
    hfm["seed"] = seed
    hfm["tips"] = np.array(tips)
    hfm["exportValues"] = 0
    out = hfm.Run()
    res = []
    for (V, seg, q, _, _), geo in zip(best, out["geodesics"]):
        geo = (np.asarray(geo)[:3].T[::-1] + org) / sp  # voxel coords, e to q
        res.append((V, seg, q, geo))
    return e, res


# ---------------------------------------------------------------------------
# 3. selection


class Segment:
    """Geometry of one candidate segment (voxel coords, per-point P and radius)."""

    def __init__(self, u, v, pts, p, r, sp, bridge=False, excess=0.0):
        self.u, self.v, self.pts, self.p, self.r = u, v, pts, p, r
        self.bridge, self.excess = bridge, excess
        step = np.linalg.norm(np.diff(pts, axis=0) * sp, axis=1)
        self.s = np.r_[0, np.cumsum(step)]
        self.length = self.s[-1]


def arc_features(seg, forward, r_node, args):
    """Reward, start/end directions and radii of a segment traversed u->v or v->u."""
    pts, p, r, s = seg.pts, seg.p, seg.r, seg.s
    if not forward:
        pts, p, r, s = pts[::-1], p[::-1], r[::-1], seg.length - s[::-1]
    L = seg.length
    b0 = args.bulge * r_node[seg.u if forward else seg.v]
    b1 = args.bulge * r_node[seg.v if forward else seg.u]
    ds = np.gradient(s) if len(s) > 1 else np.zeros(1)
    reward = float(np.sum(weight(p, args.p0) * ds * (s > b0)))

    def window(lo, hi):
        sel = (s >= lo) & (s <= hi)
        return sel if sel.sum() >= 2 else np.ones_like(sel)

    ws = window(min(b0, L / 2), min(b0, L / 2) + args.reach)
    we = window(max(L - b1 - args.reach, 0), max(L - b1, L / 2))
    d0 = pts[ws].mean(axis=0) - pts[0]
    d1 = pts[-1] - pts[we].mean(axis=0)
    unit = lambda d: d / max(np.linalg.norm(d), 1e-9)
    return dict(
        reward=reward,
        d0=unit(d0),
        d1=unit(d1),
        r0=float(np.median(r[ws])),
        r1=float(np.median(r[we])),
        p0=float(np.median(p[ws])),
        p1=float(np.median(p[we])),
    )


def highs_model(c, M, lb, ub, ub_v, integer, cs=None):
    """HiGHS instance of the program; column j is solved in units of cs[j] (the
    flows and capacity slacks in --flow_unit mm: their big-M coefficients of up
    to ~1e4 mm otherwise slow every LP)."""
    import highspy

    cs = np.ones(len(c)) if cs is None else np.asarray(cs, float)
    inf = highspy.kHighsInf
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = len(c), M.shape[0]
    lp.col_cost_ = np.asarray(c) * cs
    lp.col_lower_ = np.zeros(len(c))
    lp.col_upper_ = np.asarray(ub_v) / cs
    lp.row_lower_ = np.clip(lb, -inf, inf)
    lp.row_upper_ = np.clip(ub, -inf, inf)
    Mc = (sparse.csr_matrix(M) @ sparse.diags(cs)).tocsc()
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = Mc.indptr
    lp.a_matrix_.index_ = Mc.indices
    lp.a_matrix_.value_ = Mc.data
    lp.integrality_ = [
        highspy.HighsVarType.kInteger if i else highspy.HighsVarType.kContinuous
        for i in integer
    ]
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.passModel(lp)
    return h, cs


def run_highs(h, cs, x0):
    """Run h from the start x0 (unscaled; None = none); x, objective, gap, status."""
    import highspy

    if x0 is not None:
        sol = highspy.HighsSolution()
        sol.col_value = list(np.asarray(x0) / cs)
        sol.value_valid = True
        h.setSolution(sol)
    h.run()
    status = h.modelStatusToString(h.getModelStatus())
    x = np.asarray(h.getSolution().col_value)
    if len(x) != len(cs):
        raise RuntimeError(status)
    info = h.getInfo()
    return x * cs, info.objective_function_value, info.mip_gap, status


def solve_milp(c, M, lb, ub, ub_v, integer, x0, args, cs=None, time_limit=None):
    """HiGHS MIP with an optional starting solution (scipy's milp takes none)."""
    h, cs = highs_model(c, M, lb, ub, ub_v, integer, cs)
    h.setOptionValue("output_flag", bool(args.verbose))
    h.setOptionValue("time_limit", float(time_limit or args.time_limit))
    h.setOptionValue("mip_rel_gap", float(args.mip_gap))
    h.setOptionValue("mip_heuristic_effort", float(args.mip_heuristic))
    x, obj, gap, status = run_highs(h, cs, x0)
    bound = h.getInfo().mip_dual_bound
    return x, obj, gap, status, bound


# Region search. Each worker process holds one HiGHS copy of the
# whole program; a task fixes every binary but those of one box to the incumbent
# and solves the rest (the box's binaries and all flows) to optimality.
_LNS = {}


def _lns_init(model, n_bin, box_time):
    h, cs = highs_model(*model)
    h.setOptionValue("threads", 1)  # deterministic, one core per box
    h.setOptionValue("mip_rel_gap", 1e-6)
    h.setOptionValue("mip_abs_gap", 1e-4)
    h.setOptionValue("time_limit", float(box_time))
    _LNS.update(h=h, cs=cs, n_bin=n_bin, ub=np.asarray(model[4])[:n_bin] / cs[:n_bin])


def _lns_box(task):
    free, x = task
    h, cs, nb = _LNS["h"], _LNS["cs"], _LNS["n_bin"]
    xb = np.round(x[:nb])
    lo = xb.copy()
    hi = xb.copy()
    lo[free] = 0.0
    hi[free] = _LNS["ub"][free]
    h.changeColsBounds(nb, np.arange(nb, dtype=np.int32), lo, hi)
    t = time.time()
    xn, _, _, status = run_highs(h, cs, x)
    return xn, status, time.time() - t


def region_boxes(A, trans, node_xyz, B):
    """Boxes of side B mm on a grid of stride B / 2 over the nodes: per box, the
    free binary columns (the arcs with an end in the box and the transitions into
    or out of them), as an array and as a set. Boxes with fewer than 2 arcs and
    duplicates are left out."""
    n_arc = len(A)
    u = np.array([a[2] for a in A])
    v = np.array([a[3] for a in A])
    lo = node_xyz.min(axis=0) - B / 2
    idx = np.floor((node_xyz - lo) / (B / 2)).astype(int)  # node in boxes idx-1, idx
    par = np.array([a for a, _ in trans])
    chi = np.array([b for _, b in trans])
    out, seen = [], set()
    for ijk in sorted({tuple(q - o) for q in idx for o in np.ndindex(2, 2, 2)}):
        inb = np.all((idx == ijk) | (idx == np.add(ijk, 1)), axis=1)
        fa = inb[u] | inb[v]
        if fa.sum() < 2:
            continue
        ft = fa[chi] | ((par >= 0) & fa[np.maximum(par, 0)])
        free = np.r_[np.flatnonzero(fa), n_arc + np.flatnonzero(ft)]
        key = free.tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append((free, set(free.tolist())))
    return out


def region_search(model, n_bin, scales, x, complete, feasible, args):
    """Large-neighbourhood search over boxes, to a local optimum that no single
    box improves (each box solved exactly, its binaries and all flows free).
    A pass solves every dirty box in parallel from the same incumbent, then
    applies the improving ones in box order: each box's own changes (binaries it
    set differently from the incumbent it started from) are merged into the
    current incumbent, completed (flows, slacks) and kept if feasible and better.
    A box whose merge fails, and every box sharing a binary with an accepted
    change, is dirty for the next pass. When none is left, a pass over all boxes
    confirms the optimum (the flows are global: a change elsewhere can open a
    move in an unchanged box). Deterministic for a given start, as long as no
    box solve hits --lns_box_time (counted in the stats). `scales` is a list of
    box sets, searched in turn (small boxes first: cheap, and they bring the
    start close for the larger ones)."""
    c = model[0]
    obj = float(c @ x)
    stats = {
        "boxes": [len(b) for b in scales],
        "box_free_max": max(len(f) for b in scales for f, _ in b),
        "box_solves": 0,
        "box_s_max": 0.0,
        "box_cpu_s": 0.0,
        "improved": 0,
        "rejected": 0,
        "passes": 0,
        "box_status": {},
    }
    workers = max(1, int(args.workers))
    pool = (
        ProcessPoolExecutor(  # spawn: HiGHS's thread pool does not survive a fork
            workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_lns_init,
            initargs=(model, n_bin, args.lns_box_time),
        )
        if workers > 1
        else None
    )
    if pool is None:
        _lns_init(model, n_bin, args.lns_box_time)
    t0 = time.time()
    try:
        for boxes in scales:
            x, obj = _region_passes(
                pool, boxes, x, obj, n_bin, c, complete, feasible, args, stats, t0
            )
    finally:
        if pool is not None:
            pool.shutdown()
    stats["box_cpu_s"] = round(stats["box_cpu_s"], 1)
    stats["s"] = round(time.time() - t0, 1)
    return x, obj, stats


def _region_passes(pool, boxes, x, obj, n_bin, c, complete, feasible, args, stats, t0):
    """Passes of region_search over one box set, to its local optimum."""
    n = len(boxes)
    dirty = np.ones(n, bool)
    size = np.array([len(f) for f, _ in boxes])
    final = False
    while True:
        stats["passes"] += 1
        todo = np.flatnonzero(dirty)
        dirty[:] = False
        # largest boxes first for the load balance; applied in box order below
        order = todo[np.argsort(-size[todo], kind="stable")]
        tasks = [(boxes[i][0], x) for i in order]
        res = list(pool.map(_lns_box, tasks)) if pool else map(_lns_box, tasks)
        res = dict(zip(order.tolist(), res))
        stats["box_solves"] += len(todo)
        x_snap, b_snap = x, np.round(x[:n_bin])
        changed = np.zeros(n_bin, bool)
        n_imp = 0
        for i in todo:
            xn, status, dt = res[i]
            bs = stats["box_status"]
            bs[status] = bs.get(status, 0) + 1
            stats["box_s_max"] = max(stats["box_s_max"], round(dt, 2))
            stats["box_cpu_s"] += dt
            if float(c @ xn) >= float(c @ x_snap) - 1e-6:
                continue
            if x is x_snap:
                cand = xn
            else:  # merge this box's own changes into the current incumbent
                own = np.flatnonzero(np.round(xn[:n_bin]) != b_snap)
                cand = x.copy()
                cand[own] = np.round(xn[own])
                cand = complete(cand)
                if not feasible(cand) or float(c @ cand) >= obj - 1e-6:
                    stats["rejected"] += 1
                    dirty[i] = True
                    continue
            changed |= np.round(cand[:n_bin]) != np.round(x[:n_bin])
            x, obj = cand, float(c @ cand)
            n_imp += 1
        if changed.any():
            ch = set(np.flatnonzero(changed).tolist())
            for j in range(n):
                if not boxes[j][1].isdisjoint(ch):
                    dirty[j] = True
        stats["improved"] += n_imp
        print(
            f"  region pass {stats['passes']}: {len(todo)} boxes, {n_imp} improved,"
            f" objective {obj:.2f}, {time.time() - t0:.0f} s",
            flush=True,
        )
        if not dirty.any():
            if final or not args.lns_confirm:
                break
            final = True  # one pass over every box confirms the optimum
            dirty[:] = True
        else:
            final = False
    return x, obj


def selection_program(segs, n_nodes, r_node, inlets, args, h):
    """The selection program (arcs, transitions, `build` of one Murray mode),
    without solving it; h = voxel size (mm) of the radius error model."""
    A = []  # arcs: (segment, forward, start node, end node, features)
    for k, sg in enumerate(segs):
        A.append((k, True, sg.u, sg.v, arc_features(sg, True, r_node, args)))
        A.append((k, False, sg.v, sg.u, arc_features(sg, False, r_node, args)))
    # capacity constants come from the full candidate set (unaffected by the pruning)
    L_seg = np.array([sg.length for sg in segs])
    r_seg = np.array([np.median(sg.r) for sg in segs])
    p_seg = np.array([np.median(sg.p) for sg in segs])
    r_in = [
        r_seg[k] for k, _, u, _, _ in A if u in inlets
    ]  # every arc leaving an inlet is kept below (its tail is trivially reachable)

    # Exact pruning of arcs that no feasible tree can use. A used arc u->v has a
    # chain of parents back to an inlet, i.e. a directed walk of kept arcs from an
    # inlet to u that does not use the segment itself (one direction per
    # segment). Iterated to a fixed point; outside the 2-core this forces the
    # orientation away from the inlets (bridge with inlets only on one side).
    A_full = len(A)
    alive = np.ones(len(A), bool)
    # A bridge must close a gap, not run parallel to a vessel the skeleton already
    # has: drop it if its ends are joined along the skeleton within its own length
    # + --bridge_detour mm (the candidate test, 2x / +3 mm from the free end, lets
    # 1-2 mm parallel shortcuts through at the native T2w resolution).
    n_detour = 0
    if args.bridge_detour > 0:
        Gs = nx.Graph()
        for sg in segs:
            if not sg.bridge and sg.u != sg.v:
                w = (
                    min(sg.length, Gs.edges[sg.u, sg.v]["w"])
                    if Gs.has_edge(sg.u, sg.v)
                    else sg.length
                )
                Gs.add_edge(sg.u, sg.v, w=w)
        for k, sg in enumerate(segs):
            if sg.bridge and sg.u in Gs:
                d = nx.single_source_dijkstra_path_length(
                    Gs, sg.u, cutoff=sg.length + args.bridge_detour, weight="w"
                )
                if sg.v in d:
                    alive[2 * k] = alive[2 * k + 1] = False
                    n_detour += 1
    while True:
        out_arcs = [[] for _ in range(n_nodes)]
        for a, (_, _, s0, _, _) in enumerate(A):
            if alive[a]:
                out_arcs[s0].append(a)

        def reach(sources, skip_seg):
            seen = set(sources)
            stack = list(seen)
            while stack:
                n = stack.pop()
                for a in out_arcs[n]:
                    if A[a][0] != skip_seg and A[a][3] not in seen:
                        seen.add(A[a][3])
                        stack.append(A[a][3])
            return seen

        changed = False
        for k in range(len(segs)):
            if not (alive[2 * k] or alive[2 * k + 1]):
                continue
            seen = reach(list(inlets), k)
            for a in (2 * k, 2 * k + 1):
                if alive[a] and A[a][2] not in seen:
                    alive[a] = False
                    changed = True
        if not changed:
            break
    # the candidate length that can be fed at all (components no inlet can reach,
    # even through bridges, do not count)
    L_reach = L_seg[sorted({A[a][0] for a in np.flatnonzero(alive)})].sum()
    c0 = L_reach / np.sum(np.power(r_in, 3))
    # downstream length of every kept arc: the segments reachable from its head
    # without its own segment (bounds the subtree it can feed)
    down = {}
    for a in np.flatnonzero(alive):
        seen = reach([A[a][3]], A[a][0])
        segs_ds = {A[c][0] for n in seen for c in out_arcs[n] if A[c][0] != A[a][0]}
        down[a] = L_seg[A[a][0]] + sum(L_seg[k] for k in segs_ds)
    keep = np.flatnonzero(alive)
    U_arc = np.array([down[a] for a in keep])  # bound of the subtree length
    A = [A[a] for a in keep]
    n_arc = len(A)
    print(
        f"arc pruning: {A_full} -> {n_arc} arcs ({n_detour} parallel bridges)",
        flush=True,
    )
    starts = [[] for _ in range(n_nodes)]
    ends = [[] for _ in range(n_nodes)]
    for a, (_, _, s0, s1, _) in enumerate(A):
        starts[s0].append(a)
        ends[s1].append(a)

    trans, t_cost = [], []  # (parent arc or -1 = inlet, child arc)
    # The growth cost is the negative log-likelihood of radius_error.py's
    # measurement model: a parent the image barely resolves (radius near the
    # voxel size, or faint evidence) says little about how much its child may
    # grow (sub-01: a faint ACA stretch at the floor radius, and a 7 mm ACA
    # connector at P 0.1, would otherwise pay to continue into their own lumen,
    # and the tree dropped that lumen).
    for n in range(n_nodes):
        for b in starts[n]:
            fb = A[b][4]
            for a in ends[n]:
                if A[a][0] == A[b][0]:
                    continue
                fa = A[a][4]
                turn = (1 - fa["d1"] @ fb["d0"]) / 2
                g_cost = growth_cost(
                    fa["r1"], fb["r0"], fa["p1"], fb["p0"], h, args.w_rad, args.r_tol
                )
                trans.append((a, b))
                t_cost.append(args.w_turn * turn + g_cost)
            if n in inlets:
                trans.append((-1, b))
                t_cost.append(0.0)
    n_tr = len(trans)
    y_cost = np.array(
        [
            args.alpha
            - f["reward"]
            + (args.beta + segs[k].excess if segs[k].bridge else 0.0)
            for k, _, _, _, f in A
        ]
    )

    def build():
        """The program: variables y (arc used), x (transition used), f (flow
        through a transition: the centerline length of the child's subtree,
        which also rules out cycles), e / d (subtree length above / below the
        arc's capacity band) and leaf (a used arc without a child)."""
        # capacity (Murray): the centerline length an arc feeds should scale with
        # r^3; c0 = candidate length / sum of r^3 over the inlets. The band's log
        # half-width scales with the radius error, and its penalty with the
        # validity of the segment's radius (median P)
        r_arc = np.array([r_seg[k] for k, *_ in A])
        L_arc = np.array([L_seg[k] for k, *_ in A])
        slack = args.cap_slack ** np.sqrt(rel_var(r_arc, h))
        cap_hi = slack * c0 * r_arc**3
        cap_lo = c0 * r_arc**3 / slack
        rho_arc = validity(p_seg[[k for k, *_ in A]])
        X, F, E, Dd = n_arc, n_arc + n_tr, n_arc + 2 * n_tr, 2 * n_arc + 2 * n_tr
        Lf = 3 * n_arc + 2 * n_tr
        nv = Lf + n_arc
        c = np.r_[
            y_cost,
            t_cost,
            np.zeros(n_tr),
            args.w_cap * np.r_[rho_arc, rho_arc],
            np.full(n_arc, args.w_leaf),
        ]
        rows, cols, vals, lb, ub = [], [], [], [], []

        def row(entries, lo, hi):
            i = len(lb)
            for j, w in entries:
                rows.append(i)
                cols.append(j)
                vals.append(w)
            lb.append(lo)
            ub.append(hi)

        by_seg = {}
        for a, arc in enumerate(A):
            by_seg.setdefault(arc[0], []).append(a)
        for pair in by_seg.values():  # one direction per segment
            if len(pair) == 2:
                row([(pair[0], 1), (pair[1], 1)], 0, 1)
        into = [[] for _ in range(n_arc)]
        out_of = [[] for _ in range(n_arc)]
        for t, (a, b) in enumerate(trans):
            into[b].append(t)
            if a >= 0:
                out_of[a].append(t)
                row([(X + t, 1), (a, -1)], -np.inf, 0)  # parent is used
            row([(F + t, 1), (X + t, -U_arc[b])], -np.inf, 0)  # flow only if used
            row([(F + t, -1), (X + t, L_arc[b])], -np.inf, 0)  # at least the arc itself
        for b in range(n_arc):
            f_in = [(F + t, 1) for t in into[b]]
            # exactly one parent per used arc; each used arc consumes its length
            row([(X + t, 1) for t in into[b]] + [(b, -1)], 0, 0)
            row(f_in + [(F + t, -1) for t in out_of[b]] + [(b, -L_arc[b])], 0, 0)
            row(f_in + [(b, -cap_hi[b]), (E + b, -1)], -np.inf, 0)
            row(
                [(F + t, -1) for t in into[b]] + [(b, cap_lo[b]), (Dd + b, -1)],
                -np.inf,
                0,
            )
            # a used arc without a child is a dead end (minimised: leaf = 1 exactly then)
            row([(Lf + b, 1), (b, -1)] + [(X + t, 1) for t in out_of[b]], 0, np.inf)
            if out_of[b]:
                row(
                    [(X + t, 1) for t in out_of[b]] + [(b, -args.max_children)],
                    -np.inf,
                    0,
                )
        M = sparse.csr_matrix((vals, (rows, cols)), shape=(len(lb), nv))
        U_tr = np.array([U_arc[b] for _, b in trans])
        # e <= U - cap_hi, d <= cap_lo - L_arc for a used arc (both vanish if unused)
        ub_v = np.r_[
            np.ones(n_arc + n_tr),
            U_tr,
            np.maximum(0, U_arc - cap_hi),
            np.maximum(0, cap_lo - L_arc),
            np.ones(n_arc),
        ]
        integer = np.r_[np.ones(n_arc + n_tr, bool), np.zeros(nv - n_arc - n_tr, bool)]
        # flows and capacity slacks are solved in units of --flow_unit mm
        cs = np.ones(nv)
        cs[F:Lf] = args.flow_unit
        tol = 1e-6 * (1 + np.abs(np.where(np.isfinite(lb), lb, 0)))

        def feasible(x):
            r = M @ x
            return bool(
                np.all(r >= np.asarray(lb) - tol - 1e-6 * np.abs(r))
                and np.all(r <= np.asarray(ub) + tol + 1e-6 * np.abs(r))
            )

        def complete(x):
            """Flow, capacity slacks and leaves of the tree given by the transitions of x."""
            x = np.where(integer, np.round(x), 0.0)
            kids = {}
            for t in np.flatnonzero(x[X:F] > 0.5):
                kids.setdefault(trans[t][0], []).append(t)
            sub = np.zeros(n_arc)
            order, stack = [], list(kids.get(-1, []))
            seen = set()  # a merged region-search candidate may hold a cycle
            while stack:  # arcs in parent-before-child order
                t = stack.pop()
                if trans[t][1] in seen:
                    continue  # its flow stays 0: infeasible, rejected
                seen.add(trans[t][1])
                order.append(t)
                stack.extend(kids.get(trans[t][1], []))
            for t in reversed(order):
                b = trans[t][1]
                sub[b] = L_arc[b] + sum(sub[trans[u][1]] for u in kids.get(b, []))
                x[F + t] = sub[b]
            y = x[:n_arc]
            n_kid = np.zeros(n_arc)
            for a, ts in kids.items():
                if a >= 0:
                    n_kid[a] = len(ts)
            x[Lf : Lf + n_arc] = np.maximum(0, y - n_kid)
            x[E:Dd] = np.maximum(0, sub - cap_hi * y)
            x[Dd : Dd + n_arc] = np.maximum(0, cap_lo * y - sub)
            return x

        return SimpleNamespace(
            c=c,
            M=M.tocsr(),
            lb=lb,
            ub=ub,
            ub_v=ub_v,
            integer=integer,
            cs=cs,
            feasible=feasible,
            complete=complete,
            E=E,
            Lf=Lf,
            nv=nv,
        )

    return SimpleNamespace(
        A=A, trans=trans, build=build, n_arc=n_arc, n_tr=n_tr, c0=c0, n_detour=n_detour
    )


def select_tree(segs, n_nodes, r_node, inlets, args, node_xyz, h=None):
    """Tree of oriented segments grown from `inlets`: the selection program,
    started from its solution without the capacity terms (or --warm), solved by
    region search to a local optimum no --lns_box box improves."""
    h = 2 * min(float(np.min(sg.r)) for sg in segs) if h is None else h  # floor h/2
    prog = selection_program(segs, n_nodes, r_node, inlets, args, h)
    A, trans, n_arc, n_tr = prog.A, prog.trans, prog.n_arc, prog.n_tr
    m = prog.build()
    X, F = n_arc, n_arc + n_tr
    t0 = time.time()
    if args.warm:  # the tree of another run (same candidates)
        A_w, parent_w, *_ = pickle.loads(
            (OUT / f"{args.warm}_selection.pkl").read_bytes()
        )
        idx = {(k, fwd): a for a, (k, fwd, *_) in enumerate(A)}
        old = {a: idx[A_w[a][0], A_w[a][1]] for a in parent_w}  # pruned like ours
        old[-1] = -1
        tr_idx = {tr: t for t, tr in enumerate(trans)}
        x_fast = np.zeros(m.nv)
        for b_, a_ in parent_w.items():
            x_fast[old[b_]] = 1
            x_fast[X + tr_idx[old[a_], old[b_]]] = 1
        obj_fast = np.nan
    else:  # the same selection without the capacity terms (fast); completed, it
        # is feasible for the full program (the terms are soft)
        c_fast = m.c.copy()
        c_fast[m.E : m.Lf] = 0
        x_fast, obj_fast, *_ = solve_milp(
            c_fast, m.M, m.lb, m.ub, m.ub_v, m.integer, None, args, cs=m.cs
        )
    x0 = m.complete(x_fast)
    t1 = time.time() - t0
    scales = [region_boxes(A, trans, node_xyz, b) for b in args.lns_box]
    model = (m.c, m.M, m.lb, m.ub, m.ub_v, m.integer, m.cs)
    x, obj, lns_info = region_search(
        model, n_arc + n_tr, scales, x0, m.complete, m.feasible, args
    )
    x = m.complete(x)
    obj = float(m.c @ x)
    lns_info["box_mm"] = args.lns_box
    info = dict(
        arcs=n_arc,
        transitions=n_tr,
        status="region optimum",
        objective=obj,
        objective_fast_no_capacity=round(float(obj_fast), 3),
        objective_warm_start=round(float(m.c @ x0), 1),
        solve_s=round(time.time() - t0, 1),
        warm_start_s=round(t1, 1),
        capacity_mm_per_mm3=round(float(prog.c0), 1),
        capacity_penalty=round(float(m.c[m.E : m.Lf] @ x[m.E : m.Lf]), 1),
        leaves=int(np.sum(x[m.Lf :] > 0.5)),
        region_search=lns_info,
        bridges_parallel_dropped=prog.n_detour,
    )
    parent = {b: a for a, b in (trans[t] for t in np.flatnonzero(x[X:F] > 0.5))}
    D = nx.DiGraph([(a, b) for b, a in parent.items()])
    assert set(parent) <= nx.descendants(D, -1), "selection is not a tree"
    return A, parent, info


# ---------------------------------------------------------------------------


def find_inlets(segs, n_skel, node_pos, world, side, z_bottom, args):
    """Free ends of the skeleton where the VAs and ICAs enter (see the module
    docstring). `world` maps voxel coordinates to T1w world (mm), as do `side`
    and `z_bottom`. Returns {node: name} and the source node (the lower VA)."""
    skel = segs[:n_skel]
    G = nx.Graph()
    G.add_edges_from((sg.u, sg.v) for sg in skel)
    comp = {n: i for i, c in enumerate(nx.connected_components(G)) for n in c}
    size = np.bincount([comp[sg.u] for sg in skel], [sg.length for sg in skel])
    # any skeleton piece of some size: the VA is not always connected to the ICAs
    # before the selection bridges them
    main = [n for n, i in comp.items() if size[i] >= args.inlet_min_comp]
    at = {}
    for sg in skel:
        at.setdefault(sg.u, []).append((sg, True))
        at.setdefault(sg.v, []).append((sg, False))

    def walk(e, reach, thick):
        # points (voxel coords), radii and arc length from free end e up to
        # reach mm, through the degree-2 nodes left by bridge splits; at a
        # junction stop, or (thick) continue on the thickest segment
        pts, r, s, n, prev = [], [], [], e, None
        while not s or s[-1] < reach:
            nxt = [(sg, f) for sg, f in at[n] if sg is not prev]
            if not nxt or (len(nxt) > 1 and not thick):
                break
            sg, f = max(nxt, key=lambda a: np.median(a[0].r))
            p_, r_, s_ = (
                (sg.pts, sg.r, sg.s)
                if f
                else (sg.pts[::-1], sg.r[::-1], sg.length - sg.s[::-1])
            )
            s0 = s[-1] if s else 0.0
            pts.extend(p_), r.extend(r_), s.extend(s0 + s_)
            n, prev = (sg.v if f else sg.u), sg
        return np.array(pts), np.array(r), np.array(s)

    def mean_r(r, s, hi, q=None):
        sel = (s >= 1.0) & (s <= hi)  # past the end (label_centerlines r_near)
        if sel.sum() < 2:
            return 0.0
        return float(r[sel].mean() if q is None else np.percentile(r[sel], q))

    free = [e for e in main if len(at[e]) == 1]
    x = {e: world(node_pos[e]) for e in free}
    va = {}
    for e in free:
        if x[e][2] > z_bottom + args.va_band:
            continue
        _, r, s = walk(e, args.va_reach, thick=True)
        r = mean_r(r, s, args.va_reach)
        s_ = "VA-R" if side(x[e]) > 0 else "VA-L"
        if r >= args.r_va and (s_ not in va or r > va[s_][1]):
            va[s_] = (e, r)
    ica = {}
    for e in free:
        if e in {v for v, _ in va.values()}:
            continue
        # a low-evidence tail (thin, P ~0.05) can continue the trunk below the
        # skull base: the 90th percentile over --ica_reach mm sees the trunk
        pts, r, s = walk(e, max(args.inlet_reach, args.ica_reach), thick=False)
        r = mean_r(r, s, args.ica_reach, 90)
        if r < args.r_ica or len(pts) < 2:
            continue
        i = min(np.searchsorted(s, args.inlet_reach), len(pts) - 1)
        if world(pts[i])[2] <= x[e][2]:
            continue  # free end on top: an outlet
        s_ = "ICA-R" if side(x[e]) > 0 else "ICA-L"
        if args.verbose:
            print(f"ICA candidate {s_} {np.round(x[e], 1)} r {r:.2f}", flush=True)
        if s_ not in ica or x[e][2] < x[ica[s_][0]][2]:
            ica[s_] = (e, r)  # the lowest: the ICA enters at the skull base
    names = {}
    for s_ in ("VA-L", "VA-R", "ICA-R", "ICA-L"):
        found = va.get(s_) or ica.get(s_)
        if found is None:
            print(f"warning: no {s_} inlet", flush=True)
            continue
        names[found[0]] = s_
        xs = np.round(x[found[0]], 1).tolist()
        print(f"inlet {s_}: {xs}, r {found[1]:.2f} mm", flush=True)
    if not va:
        raise SystemExit(
            f"no vertebral: no free end within --va_band {args.va_band} mm of the "
            f"bottom of the mask with radius >= --r_va {args.r_va} mm"
        )
    source = min((e for e, _ in va.values()), key=lambda e: x[e][2])
    return names, source


def smooth_along(values, s, window):
    out = values.copy()
    for i in range(1, len(values) - 1):
        sel = np.abs(s - s[i]) <= window / 2
        out[i] = values[sel].mean(axis=0)
    return out


def localmax(x, r_mm, sp):
    """Maximum of x over the box of half-width r_mm around every voxel."""
    for ax in range(3):
        x = ndimage.maximum_filter1d(x, size=2 * int(round(r_mm / sp[ax])) + 1, axis=ax)
    return x


def ridge(sk_vox, strict, edt_s, reach_mm, sp):
    """Axis of the strict mask next to each skeleton voxel: the strict voxel of
    largest EDT within reach_mm (per voxel) of it. Returns that voxel (float
    voxel coords) and its EDT (mm); a skeleton voxel with no strict voxel in
    reach keeps its own position and EDT."""
    sv = np.argwhere(strict)
    e = edt_s[tuple(sv.T)]
    pos = sk_vox.astype(float)
    best = edt_s[tuple(sk_vox.T)].astype(float)
    tree = cKDTree(sv * sp)
    for i, nb in enumerate(tree.query_ball_point(sk_vox * sp, reach_mm, workers=-1)):
        if nb:
            j = nb[int(np.argmax(e[nb]))]
            if e[j] > best[i]:
                best[i], pos[i] = e[j], sv[j]
    return pos, best


def load_inputs(args):
    """Evidence P, candidate mask and radius mask on one grid, cropped to the mask.

    Native mode: P = max(T2w, consensus_other) (T2w alone with --t2_only) inside
    the head ROI; candidates = hysteresis on P (seeds >= --seed, grown into
    >= --grow), components of >= --min_cc voxels. --network NAME instead takes
    build_network's binary mask and consensus on the T1w grid.

    The radius comes from a stricter mask than the candidates: candidates within
    --r_frac of the maximum of P in a box of half-width --r_window, i.e. each
    vessel at its half-maximum width (build_network's --trim rule), so that the
    permissive grow threshold does not widen the vessels. Returns a dict with
    P, mask, r_vox (mm: EDT of the strict mask - h/2, at least h/2), sp (mm per
    voxel), M (4x4 voxel -> T1w world, for the cropped voxel grid), off (crop
    offset in voxels) and the counts.

    The skeleton of the candidate mask is centred in the candidates, not in the
    strict mask, and off its axis (sub-01: the median skeleton voxel reads 2 EDT
    0.45 mm below the strict mask's width, the basilar 1.5-2 mm off over 7 mm).
    With --ridge_reach > 0 the radius at a skeleton voxel is therefore read at
    the strict voxel of largest EDT within min(candidate EDT there,
    --ridge_reach) mm (its own cross-section; the skeleton's mean ridge offset
    is returned). sub-01 with --ridge_reach 2.5 (the default): radii that match
    the T2w flow void (MCA 1.3 vs 1.3 mm, VB 3.1 vs 2.9-3.4 mm diameter); the
    selection keeps the ACA course and lumen with radius_error.py's model.
    """
    if args.network:
        img = nib.load(OUT / f"{args.network}.nii.gz")
        mask = np.asanyarray(img.dataobj) > 0
        P = nib.load(OUT / "consensus.nii.gz").get_fdata(dtype=np.float32)
        t2 = nib.load(OUT / "t1space" / "MS_t2_prob.nii.gz")
        P = np.maximum(P, t2.get_fdata(dtype=np.float32))
        to_t1 = np.eye(4)
    else:
        D = Path(args.evidence)
        img = nib.load(D / "MS_t2_prob.nii.gz")
        P = img.get_fdata(dtype=np.float32)
        if not args.t2_only:
            other = nib.load(D / "consensus_other.nii.gz")
            P = np.maximum(P, other.get_fdata(dtype=np.float32))
        P[np.asanyarray(nib.load(D / "roi.nii.gz").dataobj) <= 0] = 0
        mask = apply_hysteresis_threshold(P, args.grow, args.seed)
        to_t1 = np.array(json.loads((D / "t2_to_t1.json").read_text())["matrix"])
    sp = np.array(img.header.get_zooms()[:3], float)
    # crop to the mask plus room for the bridge boxes
    pad = np.ceil(args.max_gap / sp).astype(int) + 3
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(axis=0) - pad, 0)
    hi = np.minimum(idx.max(axis=0) + pad + 1, mask.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    P, mask = np.ascontiguousarray(P[sl]), mask[sl]
    n_raw = int(mask.sum())
    if not args.network:
        lab, _ = ndimage.label(mask, CONN)
        size = np.bincount(lab.ravel())
        size[0] = 0
        mask = np.isin(lab, np.flatnonzero(size >= args.min_cc))
    strict = mask
    if args.r_frac > 0 and not args.network:
        strict = mask & (P >= args.r_frac * localmax(P, args.r_window, sp))
    h = sp.mean()
    edt_s = ndimage.distance_transform_edt(strict, sampling=sp)
    r_vox = np.maximum(edt_s - h / 2, h / 2)
    # the junction size (merging, small loops, bulge) stays the EDT at the
    # voxel itself: the axis radius of a side branch's node can be the trunk's
    r_size = r_vox.copy()
    sk = skeletonize(mask).astype(bool)
    sk_vox = np.argwhere(sk)
    ridge_offset = 0.0
    if args.ridge_reach > 0:
        edt_c = ndimage.distance_transform_edt(mask, sampling=sp)
        reach = np.minimum(edt_c[tuple(sk_vox.T)], args.ridge_reach)
        pos, best = ridge(sk_vox, strict, edt_s, reach, sp)
        r_vox[tuple(sk_vox.T)] = np.maximum(best - h / 2, h / 2)
        ridge_offset = float(np.median(np.linalg.norm((pos - sk_vox) * sp, axis=1)))
    shift = np.eye(4)
    shift[:3, 3] = lo
    return dict(
        P=P,
        mask=mask,
        r_vox=r_vox.astype(np.float32),
        r_size=r_size.astype(np.float32),
        sk=sk,
        sp=sp,
        M=to_t1 @ img.affine @ shift,
        off=lo,
        counts=dict(
            grid_shape=list(img.shape),
            crop_shape=list(mask.shape),
            mask_voxels_hysteresis=n_raw,
            mask_voxels=int(mask.sum()),
            strict_voxels=int(strict.sum()),
            ridge_offset_median_mm=round(ridge_offset, 3),
        ),
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--evidence",
        default=str(OUT / "t2space"),
        help="build_evidence.py output directory",
    )
    p.add_argument("--t2_only", action="store_true", help="P = T2w alone")
    p.add_argument(
        "--network", default=None, help="T1w-grid mask NAME.nii.gz (old input)"
    )
    p.add_argument("--grow", type=float, default=0.01, help="P, candidate hysteresis")
    p.add_argument("--seed", type=float, default=0.5, help="P, candidate seeds")
    p.add_argument("--min_cc", type=int, default=30, help="voxels per candidate piece")
    p.add_argument("--r_frac", type=float, default=0.5, help="radius mask, x local max")
    p.add_argument("--r_window", type=float, default=0.5, help="mm, local maximum")
    p.add_argument("--r_va", type=float, default=0.6, help="mm, vertebral inlet")
    p.add_argument("--va_band", type=float, default=5.0, help="mm above mask bottom")
    p.add_argument("--va_reach", type=float, default=20.0, help="mm, VA radius")
    p.add_argument("--r_ica", type=float, default=1.3, help="mm, ICA inlet")
    p.add_argument("--inlet_reach", type=float, default=8.0, help="mm, for heading")
    p.add_argument("--ica_reach", type=float, default=40.0, help="mm, ICA radius")
    p.add_argument("--inlet_min_comp", type=float, default=30.0, help="mm of skeleton")
    p.add_argument("--p0", type=float, default=0.005, help="P with zero evidence")
    p.add_argument("--alpha", type=float, default=1.0, help="mm-equivalents per arc")
    p.add_argument(
        "--w_leaf", type=float, default=0.0, help="mm-equivalents per dead-end arc"
    )
    p.add_argument(
        "--prune_len",
        type=float,
        default=3.0,
        help="mm: cut dead ends shorter than this after the selection (0 = off)",
    )
    p.add_argument(
        "--prune_r", type=float, default=0.25, help="mm: ... and thinner than this"
    )
    p.add_argument(
        "--ridge_reach",
        type=float,
        default=2.5,
        help="mm: read the radius on the strict mask's axis within this (0 = off)",
    )
    p.add_argument("--bulge", type=float, default=1.5, help="x node radius")
    p.add_argument("--reach", type=float, default=2.0, help="mm, for directions")
    p.add_argument("--w_turn", type=float, default=4.0, help="mm-eq. for a U-turn")
    p.add_argument("--w_rad", type=float, default=4.0, help="mm-eq. per log radius")
    p.add_argument("--r_tol", type=float, default=1.25, help="child/parent radius")
    p.add_argument("--max_children", type=int, default=3)
    p.add_argument("--w_cap", type=float, default=0.01, help="per mm off capacity")
    p.add_argument("--cap_slack", type=float, default=2.0, help="capacity factor")
    p.add_argument("--merge_len", type=float, default=1.0, help="mm")
    p.add_argument("--max_gap", type=float, default=6.0, help="bridge box, mm")
    p.add_argument("--cone", type=float, default=70.0, help="deg from end direction")
    p.add_argument("--bridges", type=int, default=2, help="candidates per free end")
    p.add_argument("--beta", type=float, default=2.0, help="mm-eq. per bridge")
    p.add_argument(
        "--bridge_detour",
        type=float,
        default=5.0,
        help="mm: drop bridges whose ends the skeleton joins within length + this",
    )
    p.add_argument("--xi", type=float, default=2.0, help="curvature radius, mm")
    p.add_argument("--n_theta", type=int, default=16, help="angular resolution")
    p.add_argument("--hfm_step", type=float, default=0.5, help="mm, isotropic box")
    p.add_argument("--workers", type=int, default=min(64, len(os.sched_getaffinity(0))))
    p.add_argument(
        "--time_limit", type=float, default=600.0, help="s, start without capacity"
    )
    p.add_argument("--mip_gap", type=float, default=1e-3, help="relative, ditto")
    p.add_argument(
        "--lns_box",
        type=float,
        nargs="+",
        default=[40.0],
        help="mm, region box sides, searched in turn",
    )
    p.add_argument("--lns_box_time", type=float, default=30.0, help="s per box")
    p.add_argument(
        "--no_lns_confirm",
        dest="lns_confirm",
        action="store_false",
        help="skip the final pass over every box",
    )
    p.add_argument("--flow_unit", type=float, default=100.0, help="mm, flow scaling")
    p.add_argument(
        "--mip_heuristic", type=float, default=0.3, help="HiGHS mip_heuristic_effort"
    )
    p.add_argument("--verbose", action="store_true", help="HiGHS log")
    p.add_argument("--reuse", action="store_true", help="cached candidates")
    p.add_argument("--warm", default=None, help="NAME of a run to start from")
    p.add_argument("--reuse_selection", action="store_true", help="cached tree")
    p.add_argument(
        "--smooth_window", type=float, default=1.0, help="mm, coordinates only"
    )
    p.add_argument("--tube_sides", type=int, default=16)
    p.add_argument("--out", default=None, help="output NAME")
    return p


def main():
    args = parser().parse_args()
    name = args.out or (
        "centerlines_tree" + args.network.removeprefix("network")
        if args.network
        else "centerlines_tree_t2native"
    )
    stats = dict(args=vars(args))
    timings = stats["timings_s"] = {}

    def tick(label, t0):
        timings[label] = round(time.time() - t0, 1)
        print(f"{label}: {timings[label]} s", flush=True)

    cache = OUT / f"{name}_candidates.pkl"
    if args.reuse and cache.exists():
        cached = pickle.loads(cache.read_bytes())
        if len(cached) == 5:  # older cache: one node radius
            cached = (*cached[:3], cached[2], *cached[3:])
        segs, node_pos, node_r, node_size, n_skel, cand = cached
        stats.update(cand)
        M = np.array(cand["voxel_to_t1"])
        sp = np.array(cand["spacing_mm"])
    else:
        t0 = time.time()
        inp = load_inputs(args)
        P, mask, r_vox, sp, M = (inp[k] for k in ("P", "mask", "r_vox", "sp", "M"))

        stats.update(inp["counts"])
        tick("inputs_mask_radius", t0)
        print(inp["counts"], flush=True)

        # 1. candidate segments
        t0 = time.time()
        r_size = inp["r_size"]
        node_pos, node_r, node_size, raw, sk = skeleton_segments(
            mask, r_vox, inp["sk"], r_size
        )
        stats["skeleton_voxels"] = int(sk.sum())
        stats["segments_skeleton"] = len(raw)
        node_pos, node_r, node_size, raw, n_merged, n_small = merge_close_junctions(
            node_pos, node_r, node_size, raw, sp, args.merge_len
        )
        stats.update(junctions_merged=n_merged, small_loops_dropped=n_small)
        tick("skeleton_segments", t0)
        print(f"segments: {len(raw)} ({n_merged} nodes merged)", flush=True)

        def geometry(u, v, path):
            pts = path.astype(float)
            pts[0], pts[-1] = node_pos[u], node_pos[v]
            return pts, P[tuple(path.T)], r_vox[tuple(path.T)]

        # 2. candidate bridges
        t0 = time.time()
        deg = np.bincount(
            [n for u, v, _ in raw for n in (u, v)], minlength=len(node_pos)
        )
        G = nx.Graph()
        for k, (u, v, path) in enumerate(raw):
            L = path_length(path, sp)
            if not G.has_edge(u, v) or G.edges[u, v]["w"] > L:
                G.add_edge(u, v, w=L)
        seg_of_vox = {}
        arc_pos = {}
        for k, (u, v, path) in enumerate(raw):
            s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0) * sp, axis=1))]
            for i, q in enumerate(path[1:-1], 1):
                seg_of_vox[tuple(q)] = k
                arc_pos[tuple(q)] = (s[i], s[-1])
        vox_list = np.array(list(seg_of_vox))
        vox_tree = cKDTree(vox_list * sp)
        free = [n for n in range(len(node_pos)) if deg[n] == 1]
        seg_at = {}
        for k, (u, v, _) in enumerate(raw):
            seg_at.setdefault(u, []).append(k)
            seg_at.setdefault(v, []).append(k)

        work = tempfile.mkdtemp(prefix="hfm_")
        jobs = []
        half = np.ceil(args.max_gap / sp).astype(int) + 2
        wcost = 1 + np.maximum(0, -weight(P, args.p0))
        cos_cone = np.cos(np.radians(args.cone))
        for e in free:
            k_e = seg_at[e][0]
            u, v, path = raw[k_e]
            pts, _, _ = geometry(u, v, path)
            if v == e:
                pts = pts[::-1]
            s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0) * sp, axis=1))]
            sel = (s >= 0.5) & (s <= 0.5 + args.reach)
            if sel.sum() < 2:
                sel = s <= args.reach
            d = (pts[0] - pts[sel].mean(axis=0)) * sp
            if np.linalg.norm(d) < 1e-6:
                continue
            d /= np.linalg.norm(d)
            x_e = node_pos[e]
            dist = nx.single_source_dijkstra_path_length(
                G, e, cutoff=6 * args.max_gap, weight="w"
            )
            cand = vox_tree.query_ball_point(x_e * sp, args.max_gap)
            targets = {}
            for i in cand:
                q = vox_list[i]
                k = seg_of_vox[tuple(q)]
                if k == k_e:
                    continue
                off = (q - x_e) * sp
                eu = np.linalg.norm(off)
                if eu < sp.max() or off @ d < cos_cone * eu:
                    continue
                su, sv = raw[k][0], raw[k][1]
                sa, L = arc_pos[tuple(q)]
                # along the skeleton; infinite to another piece (always a bridge)
                gd = min(dist.get(su, np.inf) + sa, dist.get(sv, np.inf) + L - sa)
                if gd < max(2 * eu, eu + 3.0):
                    continue
                targets.setdefault(k, []).append(q)
            if not targets:
                continue
            lo = np.maximum(np.round(x_e).astype(int) - half, 0)
            hi = np.minimum(np.round(x_e).astype(int) + half + 1, mask.shape)
            targets = {k: np.array(q) for k, q in targets.items()}
            targets = {
                k: q[np.all((q >= lo) & (q < hi), axis=1)] for k, q in targets.items()
            }
            targets = {k: q for k, q in targets.items() if len(q)}
            box = wcost[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].astype(float)
            jobs.append(
                (e, x_e, d, box, lo, targets, sp, args.hfm_step)
                + (args.n_theta, args.xi, args.bridges, None)
            )
        print(
            f"bridge candidates: {len(jobs)} free ends with targets of {len(free)}",
            flush=True,
        )
        results = []
        with ProcessPoolExecutor(args.workers) as ex:
            futs = []
            for i, j in enumerate(jobs):
                wd = os.path.join(work, str(i))
                os.makedirs(wd)
                futs.append(ex.submit(bridge_job, (*j[:-1], wd)))
            for f in futs:
                results.append(f.result())
        tick("minimal_paths", t0)
        t0 = time.time()

        # split target segments at the bridge ends, then build all segments
        splits = {}  # segment -> list of (arc position, voxel)
        bridges = []
        for e, res in results:
            for V, k, q, g in res:
                splits.setdefault(k, []).append(tuple(q))
                bridges.append((e, k, tuple(q), g, V - path_length(g, sp)))
        segs = []
        node_pos = list(node_pos)
        node_r = list(node_r)
        node_size = list(node_size)
        split_node = {}
        for k, (u, v, path) in enumerate(raw):
            cut_at = sorted(
                {
                    i
                    for i, q in enumerate(map(tuple, path))
                    if q in set(splits.get(k, []))
                }
            )
            bounds = [0, *[i for i in cut_at if 0 < i < len(path) - 1], len(path) - 1]
            nodes = [u]
            for i in bounds[1:-1]:
                split_node[tuple(path[i])] = len(node_pos)
                nodes.append(len(node_pos))
                node_pos.append(path[i].astype(float))
                node_r.append(r_vox[tuple(path[i])])
                node_size.append(r_size[tuple(path[i])])
            nodes.append(v)
            for a, b, i0, i1 in zip(nodes[:-1], nodes[1:], bounds[:-1], bounds[1:]):
                sub = path[i0 : i1 + 1]
                pts = sub.astype(float)
                pts[0] = node_pos[a]
                pts[-1] = node_pos[b]
                segs.append(
                    Segment(a, b, pts, P[tuple(sub.T)], r_vox[tuple(sub.T)], sp)
                )
        n_skel = len(segs)
        for e, k, q, g, excess in bridges:
            b = split_node.get(q)
            if b is None:  # q was a node voxel already
                continue
            g = np.vstack([node_pos[e], g[1:-1], node_pos[b]])
            s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(g, axis=0) * sp, axis=1))]
            keep = np.r_[0, np.flatnonzero(np.diff(np.floor(s)) > 0) + 1, len(g) - 1]
            g = g[np.unique(keep)]
            pg = ndimage.map_coordinates(P, g.T, order=1)
            rg = np.full(len(g), min(node_r[e], node_r[b]))
            segs.append(Segment(e, b, g, pg, rg, sp, bridge=True, excess=excess))
        node_pos, node_r = np.array(node_pos), np.array(node_r)
        node_size = np.array(node_size)
        tick("segments_and_bridges", t0)
        z_bottom = float(nib.affines.apply_affine(M, np.argwhere(mask))[:, 2].min())
        stats.update(
            free_ends=len(free),
            bridge_candidates=len(segs) - n_skel,
            segments=len(segs),
            nodes=len(node_pos),
            skeleton_components=nx.number_connected_components(
                nx.Graph([(sg.u, sg.v) for sg in segs[:n_skel]])
            ),
            z_bottom_t1_mm=round(z_bottom, 2),
            voxel_to_t1=M.tolist(),
            spacing_mm=sp.tolist(),
        )

        cand = {k: stats[k] for k in stats if k not in ("args", "timings_s")}
        cache.write_bytes(
            pickle.dumps((segs, node_pos, node_r, node_size, n_skel, cand))
        )
    world = lambda x: nib.affines.apply_affine(M, x)  # voxels -> T1w world, mm

    # inlets, in T1w world (sasparc, z)
    centre, normal = midplane(nib.load(SASPARC))
    side = lambda x: float((np.asarray(x) - centre) @ normal)
    inlet_name, source = find_inlets(
        segs, n_skel, node_pos, world, side, stats["z_bottom_t1_mm"], args
    )
    inlets = set(inlet_name)
    in_names = list(inlet_name.values())
    stats["inlets_xyz"] = {
        nm: np.round(world(node_pos[n]), 1).tolist() for n, nm in inlet_name.items()
    }

    # 3. selection
    sel_cache = OUT / f"{name}_selection.pkl"
    cached = pickle.loads(sel_cache.read_bytes()) if sel_cache.exists() else None
    if args.reuse_selection and cached and cached[3] == inlets:
        A, parent, info, _ = cached
    else:
        if args.reuse_selection:
            print("cached selection has other inlets: solving again", flush=True)
        t0 = time.time()
        A, parent, info = select_tree(
            segs,
            len(node_pos),
            node_size,
            inlets,
            args,
            np.asarray(node_pos) * sp,
            h=float(np.mean(sp)),
        )
        tick("selection_milp", t0)
        sel_cache.write_bytes(pickle.dumps((A, parent, info, inlets)))
    stats["milp"] = info
    print(f"milp: {info}", flush=True)
    used = sorted(parent)
    children = {}
    for b, a in parent.items():
        children.setdefault(a, []).append(b)

    # 4. join the inlet trees into one piece, decided on the arcs: where they
    # meet at a node (thickest first), else by the cheapest path of unused
    # segments between them (cost length / radius: the communicating artery
    # the selection left out); once per pair, so no loop is closed
    trees = sorted(set(inlet_name.values()), key=in_names.index)
    tree_of = {}
    stack = [b for b in used if parent[b] == -1]
    for b in stack:
        tree_of[b] = trees.index(inlet_name[A[b][2]])
    while stack:
        b = stack.pop()
        for c_ in children.get(b, []):
            tree_of[c_] = tree_of[b]
            stack.append(c_)

    # 3b. spur cut (--prune_len > 0): a dead end, from its tip up to the arc it
    # branches off, is dropped if shorter than --prune_len mm and of median
    # radius below --prune_r mm; repeated until none is left. Only dead ends go,
    # so the routing and connectivity the selection chose are untouched. A chain
    # with a node that another inlet tree also uses is a join, and stays.
    pruned = []
    while args.prune_len > 0:
        touch = {}  # node -> inlet trees using it
        for b in parent:
            for n in (A[b][2], A[b][3]):
                touch.setdefault(n, set()).add(tree_of[b])
        cut = []
        for b in parent:
            if children.get(b):
                continue
            chain = [b]
            while parent[chain[-1]] != -1 and len(children[parent[chain[-1]]]) == 1:
                chain.append(parent[chain[-1]])
            if parent[chain[-1]] == -1:  # the whole tree of an inlet
                continue
            if any(len(touch[A[a][3]]) > 1 for a in chain):
                continue
            ks = [A[a][0] for a in chain]
            length = sum(segs[k].length for k in ks)
            r_med = float(np.median(np.concatenate([segs[k].r for k in ks])))
            if length < args.prune_len and r_med < args.prune_r:
                cut.append((chain, length, r_med))
        if not cut:
            break
        for chain, length, r_med in cut:
            ks = [A[a][0] for a in chain]
            children[parent[chain[-1]]].remove(chain[-1])
            pruned.append(
                dict(
                    tip_xyz=np.round(world(node_pos[A[chain[0]][3]]), 1).tolist(),
                    inlet=trees[tree_of[chain[0]]],
                    arcs=len(chain),
                    length_mm=round(float(length), 2),
                    radius_mm=round(r_med, 3),
                    p_median=round(
                        float(np.median(np.concatenate([segs[k].p for k in ks]))), 3
                    ),
                )
            )
            for a in chain:
                del parent[a], tree_of[a]
                children.pop(a, None)
    used = sorted(parent)
    stats["spurs_pruned"] = len(pruned)
    stats["spurs_pruned_mm"] = round(sum(s["length_mm"] for s in pruned), 1)
    stats["spurs_pruned_list"] = pruned
    if pruned:
        print(f"spurs pruned: {len(pruned)} ({stats['spurs_pruned_mm']} mm)")

    at_node = {}  # node -> trees with an arc starting or ending there
    for b in used:
        for n in (A[b][2], A[b][3]):
            at_node.setdefault(n, set()).add(tree_of[b])
    group = list(range(len(trees)))

    def find(a):
        while group[a] != a:
            a = group[a]
        return a

    node_joins = []  # (node, tree, tree)
    for n in sorted(at_node, key=lambda n: -node_r[n]):
        trs = sorted(at_node[n])
        for t2 in trs[1:]:
            ga, gb = find(trs[0]), find(t2)
            if ga != gb:
                group[gb] = ga
                node_joins.append((n, trs[0], t2))
    seg_used = {A[a][0] for a in used}
    H = nx.Graph()
    for k, sg in enumerate(segs):
        if k not in seg_used and sg.u != sg.v:
            w = sg.length / max(float(np.median(sg.r)), 0.1)
            if not H.has_edge(sg.u, sg.v) or H.edges[sg.u, sg.v]["w"] > w:
                H.add_edge(sg.u, sg.v, w=w, k=k)
    path_joins = []  # (node path, segments, tree at start, tree at end)
    while len({find(t) for t in range(len(trees))}) > 1:
        best = None
        for g in {find(t) for t in range(len(trees))}:
            src = [
                n for n in at_node if n in H and any(find(t) == g for t in at_node[n])
            ]
            if not src:
                continue
            dist, path = nx.multi_source_dijkstra(H, src, weight="w")
            for n, d in dist.items():
                other = n in at_node and any(find(t) != g for t in at_node[n])
                if other and (best is None or d < best[0]):
                    best = (d, path[n], g)
        if best is None:
            break
        _, path, g = best
        ta = next(t for t in at_node[path[0]] if find(t) == g)
        tb = next(t for t in at_node[path[-1]] if find(t) != g)
        group[find(tb)] = g
        ks = [H.edges[q0, q1]["k"] for q0, q1 in pairwise(path)]
        H.remove_edges_from(pairwise(path))
        path_joins.append((path, ks, ta, tb))
    n_groups = len({find(t) for t in range(len(trees))})
    if n_groups > 1:
        print(f"warning: the inlet trees form {n_groups} unconnected pieces")
    breaks = {n for n, *_ in node_joins}
    breaks |= {q for path, *_ in path_joins for q in (path[0], path[-1])}

    # 5. output: one polyline per chain of arcs with a single child (broken at
    # the join nodes)
    inst = {}  # node -> [(point id, inlet tree)]
    pts_out, rad_out, lines, origin, cell_len, cell_r = [], [], [], [], [], []
    p_out = []  # evidence P per point (fit_radius.py: faint = unmeasured radius)
    node_type, cell_tree = [], []

    uses = {}  # node -> copies of its point so far (ends and passes)

    def offset(n):
        k = uses.get(n, 0)
        uses[n] = k + 1
        return 0.03 * k

    def new_point(n, xyz, r, t, tree, p=1.0):
        inst.setdefault(n, [])
        pts_out.append(world(xyz) + offset(n))
        rad_out.append(r)
        p_out.append(p)
        node_type.append(t)
        inst[n].append((len(pts_out) - 1, tree))
        return len(pts_out) - 1

    def add_line(ids, org, length, r_mean, tree):
        lines.append(ids)
        origin.append(org)
        cell_len.append(float(length))
        cell_r.append(float(r_mean))
        cell_tree.append(tree)

    start_pt = {}
    for b in used:
        if parent[b] == -1:
            n = A[b][2]
            if n in inst:  # an inlet node with several root arcs
                start_pt[b] = inst[n][0][0]
            else:
                t = 2 if n == source else 1
                start_pt[b] = new_point(n, node_pos[n], node_r[n], t, tree_of[b])
    stack = [b for b in used if parent[b] == -1]
    while stack:
        b = stack.pop()
        tr = tree_of[b]
        chain = [b]
        while len(children.get(chain[-1], [])) == 1 and A[chain[-1]][3] not in breaks:
            chain.append(children[chain[-1]][0])
        xyz, rr, pp = [], [], []
        for a in chain:
            k, fwd = A[a][0], A[a][1]
            sg = segs[k]
            pts_, r_ = (sg.pts, sg.r) if fwd else (sg.pts[::-1], sg.r[::-1])
            pp.append((sg.p if fwd else sg.p[::-1])[1:])
            pts_ = world(pts_[1:])
            if a != chain[-1]:  # passes through a node: its own copy of the point
                pts_[-1] += offset(A[a][3])
            xyz.append(pts_)
            rr.append(r_[1:])
        xyz = np.vstack(xyz)
        rr = np.concatenate(rr)
        p0 = start_pt[b]
        allp = np.vstack([pts_out[p0], xyz])
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(allp, axis=0), axis=1))]
        sm = smooth_along(allp, s, args.smooth_window)
        r_all = np.r_[rad_out[p0], rr]  # raw radius (fit_radius.py smooths it)
        p_all = np.r_[p_out[p0], np.concatenate(pp)]
        n_end = A[chain[-1]][3]
        kids = children.get(chain[-1], [])
        ids = [p0]
        # windows of neighbouring points can coincide: drop repeated points
        dup = np.r_[False, np.linalg.norm(np.diff(sm, axis=0), axis=1) < 1e-6]
        for x_, r_, p_, d_ in zip(sm[1:-1], r_all[1:-1], p_all[1:-1], dup[1:-1]):
            if d_:
                continue
            pts_out.append(x_)
            rad_out.append(r_)
            p_out.append(p_)
            node_type.append(0)
            ids.append(len(pts_out) - 1)
        t_end = 0 if kids else 1
        ids.append(new_point(n_end, node_pos[n_end], r_all[-1], t_end, tr, p_all[-1]))
        for c_ in kids:
            start_pt[c_] = ids[-1]
            stack.append(c_)
        org = int(any(segs[A[a][0]].bridge for a in chain))
        add_line(ids, org, s[-1], np.mean(r_all), tr)

    def point_of(n, tree):
        return next(pid for pid, t in inst[n] if t == tree)

    remap, joins = {}, []
    for n, ta, tb in node_joins:
        pa, pb = point_of(n, ta), point_of(n, tb)
        remap[pb] = pa
        node_type[pa] = 0
        joins.append(
            dict(
                trees=[trees[ta], trees[tb]],
                xyz=np.round(world(node_pos[n]), 1).tolist(),
                radius=round(float(node_r[n]), 2),
            )
        )
    for path, ks, ta, tb in path_joins:
        ids, rr, length = [point_of(path[0], ta)], [], 0.0
        for (q0, q1), k in zip(pairwise(path), ks):
            sg = segs[k]
            fwd = sg.u == q0
            pts_, r_ = (sg.pts, sg.r) if fwd else (sg.pts[::-1], sg.r[::-1])
            p_ = sg.p if fwd else sg.p[::-1]
            last = q1 == path[-1]
            inner = zip(world(pts_[1 : -1 if last else None]), r_[1:], p_[1:])
            for x_, r_i, p_i in inner:
                pts_out.append(x_)
                rad_out.append(r_i)
                p_out.append(p_i)
                node_type.append(0)
                ids.append(len(pts_out) - 1)
                rr.append(r_i)
            length += sg.length
        ids.append(point_of(path[-1], tb))
        for i in (ids[0], ids[-1]):
            node_type[i] = 0
        if not rr:  # a single segment without interior points
            rr = [min(node_r[path[0]], node_r[path[-1]])]
        r_mean = np.mean(rr)
        add_line(ids, 2, length, r_mean, -1)
        joins.append(
            dict(
                trees=[trees[ta], trees[tb]],
                xyz=np.round(world(node_pos[path[len(path) // 2]]), 1).tolist(),
                radius=round(float(np.median(rr)), 2),
                path_mm=round(float(length), 1),
            )
        )

    def resolve(i):
        while i in remap:
            i = remap[i]
        return i

    lines = [[resolve(i) for i in ids] for ids in lines]
    keep = np.unique(np.concatenate(lines))
    new_id = np.full(len(pts_out), -1)
    new_id[keep] = np.arange(len(keep))
    lines = [new_id[ids] for ids in lines]
    pts_out = np.asarray(pts_out)[keep]
    rad_out = np.asarray(rad_out)[keep]
    p_out = np.asarray(p_out)[keep]
    node_type = np.asarray(node_type)[keep]

    net = pv.PolyData(pts_out, lines=np.concatenate([np.r_[len(i), i] for i in lines]))
    net["radius"] = np.array(rad_out)
    net["evidence"] = np.asarray(p_out, np.float32)
    net["node_type"] = np.array(node_type, np.uint8)
    net.cell_data["branch"] = np.arange(net.n_cells)
    net.cell_data["length"] = np.array(cell_len)
    net.cell_data["mean_radius"] = np.array(cell_r)
    net.cell_data["origin"] = np.array(origin)
    net.cell_data["inlet"] = np.array(cell_tree)
    net.field_data["inlet_names"] = np.array(trees)
    net.save(OUT / f"{name}.vtp")
    tubes = []
    for b, ids in enumerate(lines):
        line = pv.lines_from_points(pts_out[ids])
        line["radius"] = np.asarray(rad_out)[ids]
        tube = line.tube(
            scalars="radius", absolute=True, n_sides=args.tube_sides, capping=True
        )
        tube.cell_data["branch"] = np.full(tube.n_cells, b)
        tube.cell_data["origin"] = np.full(tube.n_cells, origin[b])
        tubes.append(tube)
    pv.merge(tubes).save(OUT / f"{name}_tubes.vtp")

    # summary
    skel_len = sum(sg.length for sg in segs[:n_skel])
    used_len = sum(segs[k].length for k in seg_used if k < n_skel)
    ends_by_node = {}
    for a in used:
        ends_by_node.setdefault(A[a][3], []).append(a)
    shared = [n for n, arcs in ends_by_node.items() if len(arcs) > 1]
    stats.update(
        inlets=len(inlets),
        arcs_used=len(used),
        bridges_used=int(sum(segs[k].bridge for k in seg_used)),
        skeleton_length_mm=round(float(skel_len), 1),
        skeleton_length_used_mm=round(float(used_len), 1),
        skeleton_segments_dropped=int(n_skel - sum(k < n_skel for k in seg_used)),
        shared_nodes=len(shared),
        trees={
            trees[t]: round(
                float(np.sum(np.array(cell_len)[np.array(cell_tree) == t])), 1
            )
            for t in range(len(trees))
        },
        tree_joins=joins,
        pieces=n_groups,
        shared_nodes_xyz=[np.round(world(node_pos[n]), 1).tolist() for n in shared],
        branches=int(net.n_cells),
        points=int(net.n_points),
        total_length_mm=round(float(np.sum(cell_len)), 1),
        radius_mm_percentiles_5_50_95=np.round(
            np.percentile(rad_out, [5, 50, 95]), 3
        ).tolist(),
    )
    (OUT / f"{name}.json").write_text(json.dumps(stats, indent=2))
    print(
        json.dumps(
            {k: v for k, v in stats.items() if k not in ("args", "shared_nodes_xyz")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
