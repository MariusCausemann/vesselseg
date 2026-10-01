"""Fit the vessel radius over the whole centerline tree.

The radius extract_tree.py writes is the mask distance-to-boundary: a noisy
measurement of a true radius that varies smoothly along each vessel and obeys
a branching law at junctions. It has local pinches (vessel contact, partial
volume) and collapses at the tips, and downstream ``u ~ 1/R^2`` at a conserved
flow turns each of those into a velocity spike. This step replaces it by one
regularized estimate over the whole tree (Huber data term + smoothing along
each chain + Murray's law at junctions), and is meant to run between
extract_tree.py and label_centerlines.py.

Graph. The polylines of NAME.vtp are merged into a point graph (points within
--merge_tol coincide: junctions are shared between branches; zero-length edges
vanish). Cycles (joins between inlet trees, loops) are cut at their thinnest
edge for the fit (maximum spanning forest by raw edge radius); the cut edges
keep their raw radius. Roots: the inlets -- the points where a branch starts
and no branch ends, i.e. the node_type 2 source and the other inlets -- so a
junction's parent edge is the one towards the nearest inlet (arclength),
which keeps Murray's law oriented towards the feeding vessel also for the
second inlet tree (ICAs) joined to the source tree. A component without an
inlet is rooted at its thickest point.

Model. With ``x_e = log r_e`` the unknown per edge and ``y_e`` the log of the
edge's raw sample (the mean of its end-point radii, leaving out a junction
end: a junction point carries the parent's radius), minimise::

    J(x) =   sum_e  w_e rho(x_e - y_e)                          data
           + 1/2 sum_{e,f} lam^2/ds_ef (x_e - x_f)^2            along a chain
           + 1/2 sum_j mu (x_p - 1/k log sum_d exp(k x_d))^2    Murray

    w_e   = l_e P_e r_obs^2 / (r_obs^2 + h^2)  edge length times reliability
            (radius_error.py): a radius near the voxel size h is
            partial-volume limited, and one at faint evidence P_e (mean over
            the edge's points, the `evidence` array of extract_tree.py; 1
            without it) may not measure a lumen at all
    rho   = Huber loss (quadratic up to --huber in log radius, then linear),
            so a local artefact (pinch, collapsing tip) costs little to
            override
    e, f  = consecutive edges of one chain (their shared point has degree 2),
            ds_ef the distance of their midpoints
    j     = a junction (degree > 2, not a root), p its parent edge (towards
            the nearest inlet), d its daughter edges (the edges at the
            junction)

All weights are lengths (mm): --lambda_mm is the smoothing length along a
chain (nothing is smoothed across a junction), --murray_mm how much
along-vessel evidence one junction's branching law is worth, --k the Murray
exponent (``r_p^k = sum r_d^k``). Solved by Gauss-Newton on the IRLS form of
the Huber loss (sparse normal equations, backtracking).

The fit is per edge; it is turned back into a per-point `radius` by a
regularized least squares, so that the mean of an edge's two end-point radii
(how a consumer such as proteoflow reads the edge radius) reproduces the fitted
edge radius.

Output (results/sub-01/network/), NAME.vtp rewritten in place:
  NAME.vtp             point data `radius` = fitted, `radius_raw` = the input
                       (kept if already present, so a rerun fits from the raw
                       radius and is idempotent); cell data mean_radius updated
  NAME_tubes.vtp       regenerated with the fitted radius
  NAME_radius_fit.json fit statistics and before/after comparison (median
                       |change| of the radius, junctions violating Murray's law
                       by more than 3x)

usage (after extract_tree.py, before label_centerlines.py):
  pixi run python fit_radius.py --name centerlines_tree_posterior_t2
"""

import argparse
import json

import networkx as nx
import numpy as np
import pyvista as pv
import scipy.sparse as sp
from common import RESULTS
from label_centerlines import polylines
from radius_error import rel_var, validity
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.special import logsumexp, softmax

OUT = RESULTS / "network"


def fit_tree_radius(
    tree, points, radius, roots, lam_mm=2.0, mu_mm=5.0, k=3.0, h_mm=0.5, huber=0.15,
    maxiter=100, tol=1e-8, floor_mm=0.0, floor_weight=0.02,
    root_radius=None, root_weight_mm=10.0, evidence=None,
):  # fmt: skip
    """Fit the edge ``radius`` of ``tree`` (a forest of nodes with edge
    attributes ``length``, mm) as one regularized estimate over the whole
    tree, in place; see the module docstring for the objective. ``points``
    are the node coordinates and ``radius`` the node radii (raw samples).
    ``roots`` are the inlet nodes (no Murray law at them); the parent of a
    junction is the incident edge towards the nearest root.

    The data weight is radius_error.py's: length / relative error variance
    (a radius near ``h_mm`` is partial-volume limited), times the edge's mean
    ``evidence`` P (node P of the extraction) if given. The fit is bounded
    below by ``floor_mm`` (the extraction's minimum radius). ``root_radius``
    ({root: mm}) anchors the edges at each root with ``root_weight_mm`` of data
    weight; samples below half the root's radius within its anchored reach
    (``root_reach``, {root: set of edges}) keep only ``floor_weight`` of their
    data weight (the crop at the edge of the head ROI thins the trunks).

    Returns ``{iterations, J0, J, n_junctions, murray_rms_before,
    murray_rms_after, median_abs_change}`` (log-radius units for the last
    three)."""
    fixed = {n for n in tree.nodes if tree.degree(n) > 2}
    elist = list(tree.edges)
    eidx = {}
    for i, (a, b) in enumerate(elist):
        eidx[a, b] = eidx[b, a] = i
    n = len(elist)
    lens = np.array([tree.edges[e]["length"] for e in elist])
    r_obs = np.array([
        np.mean([radius[m] for m in ([m for m in e if m not in fixed] or e)]) for e in elist
    ])  # fmt: skip
    y = np.log(r_obs)
    w = lens / rel_var(r_obs, h_mm)  # = l r^2 / (r^2 + h^2)
    p_e = (
        np.array([np.mean([evidence[m] for m in e]) for e in elist])
        if evidence is not None
        else None
    )
    if p_e is not None:
        w = w * validity(p_e)
    unmeasured = np.zeros(n, bool)
    root_radius = root_radius or {}
    for m, (r_root, reach) in root_radius.items():
        if m not in tree:
            continue
        for e in reach:
            i = eidx[e]
            unmeasured[i] |= r_obs[i] < 0.5 * r_root
    w = np.where(unmeasured, floor_weight * w, w)
    for m, (r_root, _) in root_radius.items():
        if m not in tree:
            continue
        for c in tree.neighbors(m):
            i = eidx[m, c]
            y[i] = np.log(r_root)
            w[i] = root_weight_mm
    x_lo = np.log(floor_mm) if floor_mm > 0 else -np.inf

    pairs, ds = [], []
    for m in tree.nodes:
        if tree.degree(m) == 2:
            a, b = tree.neighbors(m)
            i, j = eidx[m, a], eidx[m, b]
            pairs.append((i, j))
            ds.append(0.5 * (lens[i] + lens[j]))
    pairs = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    s_w = lam_mm / np.sqrt(np.array(ds))

    # parent of a node: its predecessor on the shortest path from the nearest root
    pred = {}
    paths = nx.multi_source_dijkstra_path(tree, list(roots), weight="length")
    for node, path in paths.items():
        if len(path) > 1:
            pred[node] = path[-2]
    root_set = set(roots)
    juncs = []
    for m in fixed:
        if m in root_set or m not in pred:
            continue
        par = eidx[m, pred[m]]
        kids = [eidx[m, c] for c in tree.neighbors(m) if c != pred[m]]
        juncs.append((par, np.array(kids)))

    def murray(x):
        return np.array([x[p] - logsumexp(k * x[d]) / k for p, d in juncs])

    def objective(x):
        rd = np.abs(x - y)
        rho = np.where(rd <= huber, 0.5 * rd**2, huber * (rd - 0.5 * huber))
        sm = s_w * (x[pairs[:, 0]] - x[pairs[:, 1]])
        return float(w @ rho + 0.5 * sm @ sm + 0.5 * mu_mm * np.sum(murray(x) ** 2))

    # constant structure of the smoothing rows
    rows_s = np.repeat(np.arange(len(pairs)), 2)
    cols_s = pairs.ravel()
    vals_s = np.column_stack([s_w, -s_w]).ravel()
    S = sp.csr_matrix((vals_s, (rows_s, cols_s)), shape=(len(pairs), n))
    StS = (S.T @ S).tocsr()

    x = np.maximum(y, x_lo)
    J0 = objective(x)
    Jx, it = J0, 0
    for it in range(1, maxiter + 1):
        rd = x - y
        irls = np.where(
            np.abs(rd) <= huber, 1.0, huber / np.maximum(np.abs(rd), 1e-300)
        )
        Wd = w * irls
        mres = murray(x)
        mr, mc, mv = [], [], []
        for q, (p, d) in enumerate(juncs):
            mr += [q] * (1 + len(d))
            mc += [p, *d.tolist()]
            mv += [1.0, *(-softmax(k * x[d])).tolist()]
        M = sp.csr_matrix((mv, (mr, mc)), shape=(len(juncs), n))
        H = sp.diags(Wd) + StS + mu_mm * (M.T @ M)
        g = Wd * rd + StS @ x + mu_mm * (M.T @ mres)
        dx = spsolve(H.tocsc(), -g)
        step = 1.0
        while step > 1e-4:  # projected onto x >= log floor_mm
            Jn = objective(np.maximum(x + step * dx, x_lo))
            if Jn <= Jx:
                break
            step *= 0.5
        x_new = np.maximum(x + step * dx, x_lo)
        dx, x = x_new - x, x_new
        step = 1.0
        conv = abs(Jx - Jn) <= tol * max(Jx, 1.0)
        Jx = Jn
        if conv or np.max(np.abs(step * dx)) < 1e-9:
            break

    for (a, b), val in zip(elist, np.exp(x)):
        tree.edges[a, b]["radius"] = float(val)
    rms = lambda v: float(np.sqrt(np.mean(v**2))) if len(v) else 0.0  # noqa: E731
    return {
        "iterations": it,
        "unmeasured_edges": int(unmeasured.sum()),
        "unmeasured_length_mm": float(lens[unmeasured].sum()),
        "J0": J0,
        "J": Jx,
        "n_junctions": len(juncs),
        "murray_rms_before": rms(murray(y)),
        "murray_rms_after": rms(murray(x)),
        "median_abs_change": float(np.median(np.abs(x - y))),
        "_juncs": juncs,
        "_elist": elist,
        "_y": y,
        "_x": x,
    }


def inlet_radius(tree, root, reach_mm):
    """Radius of the trunk an inlet feeds and the edges it is taken over: the
    90th percentile of the raw edge radius from 1 mm to ``reach_mm`` along the
    thickest continuation (what extract_tree's inlet detection measures; the
    head-ROI crop thins the first mm)."""
    edges, r, s, n, prev, s0 = [], [], [], root, None, 0.0
    while s0 < reach_mm:
        nxt = [c for c in tree.neighbors(n) if c != prev]
        if not nxt:
            break
        c = max(nxt, key=lambda c: tree.edges[n, c]["r_raw"])
        edges.append((n, c))
        r.append(tree.edges[n, c]["r_raw"])
        s0 += tree.edges[n, c]["length"]
        s.append(s0)
        n, prev = c, n
    r, s = np.array(r), np.array(s)
    sel = s >= 1.0
    r_in = float(np.percentile(r[sel] if sel.sum() >= 2 else r, 90))
    return r_in, set(edges)


def murray_violations(juncs, logr, k, factor=3.0):
    """Number of junctions whose parent ``r_p^k`` differs from the daughters'
    ``sum r_d^k`` by more than ``factor`` (either way)."""
    v = 0
    for p, d in juncs:
        ratio = np.exp(k * (logr[p] - logsumexp(k * logr[d]) / k))
        v += int(ratio > factor or ratio < 1 / factor)
    return v


def merge_points(points, tol):
    """Representative point id per point, merging points within ``tol``."""
    parent = np.arange(len(points))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in cKDTree(points).query_pairs(tol):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(len(points))])


def point_radius(n_nodes, edge_list, edge_r, r0, ridge=0.3):
    """Node radii whose edge means reproduce ``edge_r``: least squares on
    ``(p_u + p_v)/2 = r_e`` with a small smoothness ridge ``ridge * (p_u -
    p_v)`` (a path has one more node than edges, leaving an alternating
    null mode), anchored weakly at ``r0`` (the current node radii)."""
    e = np.asarray(edge_list)
    m = len(e)
    rows = np.repeat(np.arange(m), 2)
    B = sp.csr_matrix((np.full(2 * m, 0.5), (rows, e.ravel())), shape=(m, n_nodes))
    L = sp.csr_matrix(
        (np.tile([ridge, -ridge], m), (rows, e.ravel())), shape=(m, n_nodes)
    )
    A = (B.T @ B + L.T @ L + 1e-6 * sp.identity(n_nodes)).tocsc()
    return spsolve(A, B.T @ np.asarray(edge_r) + 1e-6 * r0)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--name", required=True, help="network file stem in results/*/network"
    )
    p.add_argument(
        "--lambda_mm",
        type=float,
        default=2.0,
        help="smoothing length along a chain (mm)",
    )
    p.add_argument(
        "--murray_mm",
        type=float,
        default=5.0,
        help="weight of one junction's branching law, as an equivalent vessel length (mm)",
    )
    p.add_argument("--k", type=float, default=3.0, help="Murray exponent")
    p.add_argument(
        "--h_mm",
        type=float,
        default=0.5,
        help="voxel size of the mask: radii near it are down-weighted",
    )
    p.add_argument(
        "--huber", type=float, default=0.15, help="Huber threshold (log radius)"
    )
    p.add_argument(
        "--merge_tol",
        type=float,
        default=1e-6,
        help="points closer than this coincide (mm)",
    )
    p.add_argument(
        "--floor_mm",
        type=float,
        default=None,
        help="extraction's minimum radius (default: the smallest raw radius)",
    )
    p.add_argument(
        "--floor_weight",
        type=float,
        default=0.02,
        help="data weight left to a thinned trunk sample at an inlet (x its weight)",
    )
    p.add_argument(
        "--inlet_reach_mm",
        type=float,
        default=40.0,
        help="inlet radius: 90th percentile over this reach from the inlet (mm)",
    )
    p.add_argument(
        "--inlet_weight_mm",
        type=float,
        default=10.0,
        help="data weight of the inlet radius, as an equivalent vessel length (mm)",
    )
    p.add_argument("--tube_sides", type=int, default=16)
    args = p.parse_args()

    path = OUT / f"{args.name}.vtp"
    net = pv.read(path)
    P = np.asarray(net.points, dtype=np.float64)
    raw = np.asarray(
        net.point_data["radius_raw" if "radius_raw" in net.point_data else "radius"]
    )
    raw = raw.astype(np.float64)
    polys = polylines(net)

    # merged point graph
    rep = merge_points(P, args.merge_tol)
    node_of = np.unique(rep, return_inverse=True)[1]
    n_nodes = node_of.max() + 1
    cnt = np.bincount(node_of, minlength=n_nodes)
    node_r = np.bincount(node_of, weights=raw, minlength=n_nodes) / cnt
    node_p = None
    if "evidence" in net.point_data:
        ev = np.asarray(net.point_data["evidence"], np.float64)
        node_p = np.bincount(node_of, weights=ev, minlength=n_nodes) / cnt
    else:
        print(f"warning: {path.name} has no 'evidence' array: radius weights without P")
    node_xyz = np.zeros((n_nodes, 3))
    node_xyz[node_of] = P
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for ids in polys:
        for a, b in zip(ids[:-1], ids[1:]):
            u, v = int(node_of[a]), int(node_of[b])
            if u != v and not G.has_edge(u, v):
                G.add_edge(
                    u, v, length=float(np.linalg.norm(node_xyz[u] - node_xyz[v]))
                )
    G.remove_nodes_from([m for m in list(G) if G.degree(m) == 0])
    for u, v, d in G.edges(data=True):
        d["r_raw"] = 0.5 * (node_r[u] + node_r[v])

    # inlets: a branch starts there and none ends
    starts = {int(node_of[ids[0]]) for ids in polys}
    ends = {int(node_of[ids[-1]]) for ids in polys}
    src_node = set(
        int(node_of[i]) for i in np.flatnonzero(np.asarray(net["node_type"]) == 2)
    )
    inlets = (starts - ends) | src_node
    names = [str(s) for s in net.field_data.get("inlet_names", [])]
    floor = float(raw.min()) if args.floor_mm is None else args.floor_mm

    # spanning forest (cycles cut at the thinnest edge), fit per component
    F = nx.maximum_spanning_tree(G, weight="r_raw")
    cut = [(u, v) for u, v in G.edges if not F.has_edge(u, v)]
    edge_fit = {(min(u, v), max(u, v)): d["r_raw"] for u, v, d in G.edges(data=True)}
    comps, before, after, juncs_all = [], 0, 0, 0
    abs_change = []
    for nodes in nx.connected_components(F):
        if len(nodes) < 2:
            continue
        sub = F.subgraph(nodes).copy()
        roots = [m for m in nodes if m in inlets]
        root_radius = {m: inlet_radius(sub, m, args.inlet_reach_mm) for m in roots}
        if not roots:
            roots = [max(nodes, key=lambda m: node_r[m])]
        st = fit_tree_radius(
            sub, node_xyz, node_r, roots,
            lam_mm=args.lambda_mm, mu_mm=args.murray_mm, k=args.k,
            h_mm=args.h_mm, huber=args.huber, floor_mm=floor,
            floor_weight=args.floor_weight,
            root_radius=root_radius, root_weight_mm=args.inlet_weight_mm,
            evidence=node_p,
        )  # fmt: skip
        st["inlets"] = [
            dict(xyz=np.round(node_xyz[m], 1).tolist(), radius_mm=round(r, 3))
            for m, (r, _) in root_radius.items()
        ]
        juncs = st.pop("_juncs")
        st.pop("_elist")
        y, x = st.pop("_y"), st.pop("_x")
        before += murray_violations(juncs, y, args.k)
        after += murray_violations(juncs, x, args.k)
        juncs_all += len(juncs)
        abs_change.append(np.abs(x - y))
        for u, v, d in sub.edges(data=True):
            edge_fit[min(u, v), max(u, v)] = d["radius"]
        st.update(n_nodes=len(nodes), n_roots=len(roots))
        comps.append(st)

    edge_list = list(edge_fit)
    edge_r = np.array([edge_fit[e] for e in edge_list])
    nr = point_radius(n_nodes, edge_list, edge_r, node_r)
    nr = np.maximum(nr, floor)  # the LS realization can undershoot the bound
    new = nr[node_of]
    back = 0.5 * (nr[[e[0] for e in edge_list]] + nr[[e[1] for e in edge_list]])
    realization = float(np.median(np.abs(np.log(back / edge_r))))

    mean_raw = np.asarray(net.cell_data["mean_radius"]).copy()
    net.point_data["radius_raw"] = raw
    net.point_data["radius"] = new
    net.cell_data["mean_radius"] = np.array([new[ids].mean() for ids in polys])
    net.save(path)

    tubes = []
    origin = np.asarray(net.cell_data["origin"])
    for b, ids in enumerate(polys):
        line = pv.lines_from_points(P[ids])
        line["radius"] = new[ids]
        tube = line.tube(
            scalars="radius", absolute=True, n_sides=args.tube_sides, capping=True
        )
        tube.cell_data["branch"] = np.full(tube.n_cells, b)
        tube.cell_data["origin"] = np.full(tube.n_cells, origin[b])
        tubes.append(tube)
    pv.merge(tubes).save(OUT / f"{args.name}_tubes.vtp")

    change = np.concatenate(abs_change) if abs_change else np.zeros(0)
    stats = {
        "params": {
            "floor_mm": floor,
            "floor_weight": args.floor_weight,
            "inlet_reach_mm": args.inlet_reach_mm,
            "inlet_weight_mm": args.inlet_weight_mm,
            "lambda_mm": args.lambda_mm,
            "murray_mm": args.murray_mm,
            "k": args.k,
            "h_mm": args.h_mm,
            "huber": args.huber,
            "merge_tol": args.merge_tol,
        },  # fmt: skip
        "n_points": int(len(P)),
        "n_nodes": int(n_nodes),
        "n_edges": int(G.number_of_edges()),
        "n_components": len(comps),
        "cut_edges": len(cut),
        "inlets": len(inlets),
        "inlet_names": names,
        "junctions": juncs_all,
        "murray_violations_3x": {"before": before, "after": after},
        "median_abs_log_change_edge": float(np.median(change)) if len(change) else 0.0,
        "median_abs_change_point_mm": float(np.median(np.abs(new - raw))),
        "median_rel_change_point": float(np.median(np.abs(new - raw) / raw)),
        "radius_median_mm": {
            "before": float(np.median(raw)),
            "after": float(np.median(new)),
        },
        "radius_min_mm": {"before": float(raw.min()), "after": float(new.min())},
        "mean_radius_changed_cells": int(
            np.sum(~np.isclose(mean_raw, net.cell_data["mean_radius"]))
        ),
        "point_realization_median_abs_log_err": realization,
        "components": comps,
    }
    (OUT / f"{args.name}_radius_fit.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "components"}, indent=1))


if __name__ == "__main__":
    main()
