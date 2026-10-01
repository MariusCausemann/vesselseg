"""Centerline graph with radius (VMTK) and a tube surface for the arterial network.

0. Contact separation: vessels that touch or cross in the mask are cut apart where
   the skeleton links them by a thin (--contact_radius), uncertain (consensus
   < --contact_prob) branch: short ones where the vessels at both ends pass by
   instead of branching, longer ones (bridges) at their weakest point (see
   separate_contacts). Only links whose removal keeps the network connected
   are cut, so only loops are broken.
1. Surface: marching cubes of the separated mask at 0.5, then windowed-sinc
   smoothing (volume preserving, so thin 1-2 voxel branches survive).
2. Seeds: voxel skeleton of the mask, terminal spurs shorter than
   max(--min_spur mm, --spur_ratio x radius at the junction) pruned. The lowest
   endpoint (neck) is the VMTK source, all other endpoints are targets.
3. vmtkcenterlines (Voronoi / maximum inscribed spheres) on a qhull Delaunay
   tessellation (~13x faster than VMTK's own), Laplacian smoothing, merged into
   one polyline per branch, sharing points at bifurcations. Skeleton pieces the
   VMTK tree misses are added and attached at their most probable contact, and
   at further contacts only where those are neither thin nor uncertain.
4. Loops: the graph is reduced to a maximum spanning tree over branch strength
   (min of consensus / --loop_prob and radius / --loop_radius along the branch);
   a loop is kept only if its weakest branch has strength >= 1, the others are
   cut at their weakest point (see break_loops).
   Terminal branches shorter than max(--min_spur, --spur_ratio x junction
   radius) are then pruned again, now measured from the parent centerline.
5. Radius smoothed along each branch (moving average over --radius_window mm).

Output (results/sub-01/network/), scanner (T1w RAS) coordinates in mm:
  centerlines.vtp          polylines (one cell per branch); point data: radius,
                           origin (0 = VMTK, 1 = skeleton loop closure),
                           node_type (2 = inlet, 1 = outlet, 0 = interior);
                           cell data: branch, length, mean_radius
  centerlines_tubes.vtp    tube around each branch with the varying radius
  centerlines_surface.vtp  the smoothed surface the centerlines were computed on
  centerlines_mask.nii.gz  network.nii.gz labels, 3 = contact voxels cut in step 0
  centerlines.json         summary + coverage check (skeleton voxels far from any
                           centerline, e.g. unused sides of Circle of Willis loops)
                           and the loops left in the graph (loops_kept: centre,
                           perimeter, strength)

With --network NAME the input is NAME.nii.gz and the outputs are named
centerlines<NAME minus "network">*, e.g. --network network_posterior_t2 ->
centerlines_posterior_t2.vtp. --t2_prob uses max(consensus, MS T2w probability)
wherever consensus is used above, so that T2w-only branches (build_network.py
--t2) are not treated as uncertain. Only the largest connected piece of the
network is used (build_network.py --t2_all keeps unconnected ones).

Run in the vmtk env:  pixi run -e vmtk python extract_centerlines.py
"""

import argparse
import json
from itertools import pairwise

import networkx as nx
import nibabel as nib
import numpy as np
import pyvista as pv
import vtk
from common import RESULTS
from scipy import ndimage
from scipy.spatial import Delaunay, cKDTree
from skimage.measure import marching_cubes
from skimage.morphology import skeletonize
from vmtk import vmtkscripts, vtkvmtk

OUT = RESULTS / "network"
RADIUS = "MaximumInscribedSphereRadius"
CONN = np.ones((3, 3, 3))


def outgoing(path, xyz, r, reach):
    """Direction in which a branch (path starting at a junction) leaves it.

    Uses the skeleton points between the junction radius and `reach` mm beyond,
    so the direction is not bent by the junction itself.
    """
    p = xyz[path]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    sel = (s >= r[path[0]]) & (s <= r[path[0]] + reach)
    if sel.sum() < 2:
        sel = s <= reach
    d = p[sel].mean(axis=0) - p[0]
    return d / max(np.linalg.norm(d), 1e-9)


def separate_contacts(mask, prob, G, idx, r, h, args):
    """Cut the mask apart where vessels touch, cross or are joined by a spurious link.

    Works on the pruned skeleton graph G (nodes index idx, radius r in mm).
    Candidates are branches between two junctions that are
    - thin: minimum radius <= --contact_radius mm, and
    - uncertain: minimum consensus < --contact_prob.
    Two kinds are cut:
    - contacts (<= --contact_max_len mm): what vessels touching side by side or
      crossing leave in the skeleton. Cut only if the vessel directions do not
      form a plausible bifurcation: at both ends the other branches contain a
      straight-through pair (outgoing directions at > arccos(-cos) degrees), i.e.
      a vessel passes the contact instead of branching from it. The mask voxels
      near both vessels go to the nearer one, and the layer of one vessel that
      touches the other is removed.
    - bridges (<= --bridge_max_len mm): thin, uncertain links between two
      vessels. The branch is cut at its least probable interior point (the mask
      voxels there that belong to this branch, not to the vessels it joins).
    Candidates are cut least probable first, and only if the skeleton stays
    connected without them, so only loops are broken. Cuts that still split off
    a piece of the mask are undone.
    Returns the separated mask, the removed voxels and counts.
    """
    polys, T = split_branches(len(idx), np.array(G.edges, int).reshape(-1, 2))
    xyz = idx * h
    arc = [
        np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyz[p], axis=0), axis=1))]
        for p in polys
    ]
    at = {}  # junction -> branches (oriented away from it)
    for p in polys:
        at.setdefault(p[0], []).append(p)
        at.setdefault(p[-1], []).append(p[::-1])

    def passes(path):
        """Some pair of the other branches at path[0] runs straight through."""
        others = [q for q in at[path[0]] if not np.array_equal(q, path)]
        dirs = [outgoing(q, xyz, r, args.contact_reach) for q in others]
        return any(
            dirs[i] @ dirs[j] < -args.contact_cos
            for i in range(len(dirs))
            for j in range(i + 1, len(dirs))
        )

    counts = dict(
        thin_uncertain=0, contacts=0, bridges=0, keeps_tree=0, undone=0, cut=0
    )
    cand = []
    for k, (p, s) in enumerate(zip(polys, arc)):
        if T.degree(p[0]) < 3 or T.degree(p[-1]) < 3 or p[0] == p[-1]:
            continue
        pp = prob[tuple(idx[p].T)]
        if r[p].min() > args.contact_radius or pp.min() >= args.contact_prob:
            continue
        counts["thin_uncertain"] += 1
        if s[-1] <= args.contact_max_len:
            if passes(p) and passes(p[::-1]):
                counts["contacts"] += 1
                cand.append((pp.min(), k, "contact"))
        elif s[-1] <= args.bridge_max_len:
            counts["bridges"] += 1
            cand.append((pp.min(), k, "bridge"))

    B = nx.MultiGraph()
    for k, p in enumerate(polys):
        B.add_edge(p[0], p[-1], key=k)
    nodes = np.array(list(G))
    sk_tree = cKDTree(xyz[nodes])
    removed = np.zeros(mask.shape, np.int32)  # cut number, 0 = kept
    n_cut = 0
    for _, k, kind in sorted(cand):
        p, s = polys[k], arc[k]
        B.remove_edge(p[0], p[-1], key=k)
        if not nx.has_path(B, p[0], p[-1]):
            B.add_edge(p[0], p[-1], key=k)
            counts["keeps_tree"] += 1
            continue
        if kind == "contact":
            # the two passing vessels: skeleton points near each end
            seeds = []
            for q in (p, p[::-1]):
                pts = []
                for o in at[q[0]]:
                    if not np.array_equal(o, q):
                        d = np.r_[
                            0,
                            np.cumsum(np.linalg.norm(np.diff(xyz[o], axis=0), axis=1)),
                        ]
                        pts.extend(o[d <= r[q[0]] + 2 * args.contact_reach])
                seeds.append(np.array(pts))
            a, b = seeds
            ext = idx[np.r_[a, b]]
        else:
            inner = (s > r[p[0]] + h) & (s < s[-1] - r[p[-1]] - h)
            if not inner.any():
                inner[1:-1] = True
            w = p[inner][np.argmin(prob[tuple(idx[p[inner]].T)])]
            ext = idx[[w]]
        pad = int(np.ceil(4 / h))
        lo = np.maximum(ext.min(axis=0) - pad, 0)
        box = tuple(slice(i, j) for i, j in zip(lo, ext.max(axis=0) + pad + 1))
        vox = np.argwhere(mask[box] & (removed[box] == 0))
        pos = (vox + lo) * h
        if kind == "contact":
            da, ia = cKDTree(xyz[a]).query(pos)
            db, ib = cKDTree(xyz[b]).query(pos)
            va = np.zeros(mask[box].shape, bool)
            vb = np.zeros_like(va)
            near_a = (da <= db) & (da <= r[a][ia] + 2 * h)
            near_b = (db < da) & (db <= r[b][ib] + 2 * h)
            va[tuple(vox[near_a].T)] = True
            vb[tuple(vox[near_b].T)] = True
            layer = va & ndimage.binary_dilation(vb, CONN)
        else:
            own = np.isin(nodes[sk_tree.query(pos)[1]], p[1:-1])
            near = np.linalg.norm(pos - xyz[w], axis=1) <= r[w] + 1.5 * h
            layer = np.zeros(mask[box].shape, bool)
            layer[tuple(vox[own & near].T)] = True
        n_cut += 1
        removed[box][layer] = n_cut

    # undo cuts that split off a piece of the mask, the most probable first
    comp, _ = ndimage.label(mask & (removed == 0), CONN)
    root = np.arange(comp.max() + 1)

    def find(a):
        while root[a] != a:
            root[a] = root[root[a]]
            a = root[a]
        return a

    objects = ndimage.find_objects(removed, max_label=n_cut)
    for c in range(n_cut, 0, -1):
        if objects[c - 1] is None:
            continue  # nothing left to remove (voxels taken by earlier cuts)
        box = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in objects[c - 1])
        cut = removed[box] == c
        # a cut can be several separate patches; only a patch that touches two
        # pieces joins them when restored
        parts, n_parts = ndimage.label(cut, CONN)
        joins = False
        for q in range(1, n_parts + 1):
            near = comp[box][
                ndimage.binary_dilation(parts == q, CONN) & (comp[box] > 0)
            ]
            roots = {find(a) for a in np.unique(near)}
            if len(roots) > 1:
                first, *rest = roots
                for a in rest:
                    root[a] = first
                joins = True
        if joins:
            removed[box][cut] = 0
            counts["undone"] += 1
    # neighbouring cuts can split off a piece together that neither touches on
    # both sides alone: undo such groups of touching cuts as a whole
    blobs, _ = ndimage.label(removed > 0, CONN)
    for b, box in enumerate(ndimage.find_objects(blobs), 1):
        box = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in box)
        cut = blobs[box] == b
        near = comp[box][ndimage.binary_dilation(cut, CONN) & (comp[box] > 0)]
        roots = {find(a) for a in np.unique(near)}
        if len(roots) > 1:
            first, *rest = roots
            for a in rest:
                root[a] = first
            counts["undone"] += len(np.unique(removed[box][cut]))
            removed[box][cut] = 0
    counts["cut"] = len(np.unique(removed)) - 1
    removed = removed > 0
    return mask & ~removed, removed, counts


def tessellation(surf):
    """Delaunay tetrahedra inside `surf`, the input vmtkcenterlines otherwise builds.

    qhull takes ~30 s here, vtkDelaunay3D inside vmtkcenterlines ~15 min. Point
    ids must match the surface vmtkcenterlines works on, which it passes through
    vtkCleanPolyData first, so `surf` must already be cleaned (see surface()).
    """
    pts = np.asarray(surf.points, float)
    grid = pv.UnstructuredGrid({vtk.VTK_TETRA: Delaunay(pts).simplices}, pts)
    grid.point_data["Normals"] = surf.point_data["Normals"]
    ext = vtkvmtk.vtkvmtkInternalTetrahedraExtractor()
    ext.SetInputData(grid)
    ext.SetOutwardNormalsArrayName("Normals")
    ext.Update()
    return ext.GetOutput()


def face_connected(mask, forbid):
    """Add voxels until the mask is 6-connected wherever it is 26-connected.

    Pieces touching only along an edge or at a corner become separate
    marching-cubes surfaces otherwise. Voxels in `forbid` are never added.
    """
    mask = mask.copy()
    n_added = 0
    six, full = ndimage.generate_binary_structure(3, 1), np.ones((3, 3, 3))
    while True:
        lab, n = ndimage.label(mask, six)
        if n == 1:
            return mask, n_added
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        big = lab == sizes.argmax()
        rest = mask & ~big
        # voxels face-adjacent to a small piece and touching the big one
        add = (
            ndimage.binary_dilation(rest, six)
            & ndimage.binary_dilation(big, full)
            & ~mask
            & ~forbid
        )
        if not add.any():
            return mask, n_added  # pieces not 26-connected to the big one
        mask |= add
        n_added += int(add.sum())


def surface(mask, affine, up, sigma, n_iter, pass_band):
    """Smoothed isosurface of the mask on a grid refined `up` times (trilinear).

    The refinement is what makes VMTK work here: at 0.5 mm many branches are only
    2-3 triangles across and the Voronoi diagram breaks up there.
    """
    sl = tuple(slice(max(a.min() - 4, 0), a.max() + 5) for a in np.nonzero(mask))
    lo = np.array([s.start for s in sl]) - 1
    f = ndimage.zoom(np.pad(mask[sl], 1).astype(np.float32), up, order=1)
    if sigma > 0:
        f = ndimage.gaussian_filter(f, sigma * up)
    verts, faces, _, _ = marching_cubes(f, 0.5)
    # zoom maps the first/last voxel centres onto each other
    n = np.array(mask[sl].shape) + 2
    verts = nib.affines.apply_affine(
        affine, verts * (n - 1) / (np.array(f.shape) - 1) + lo
    )
    faces = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
    surf = pv.PolyData(verts, faces).clean()
    n_regions = surf.connectivity().point_data["RegionId"].max() + 1
    if n_regions > 1:
        print(f"warning: surface has {n_regions} pieces, keeping the largest")
    surf = surf.connectivity("largest").extract_surface()
    surf = surf.smooth_taubin(n_iter=n_iter, pass_band=pass_band)
    surf = surf.compute_normals(auto_orient_normals=True).triangulate()
    # vtkCleanPolyData renumbers points; vmtkcenterlines applies it again, which
    # then keeps the order that tessellation() relies on
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(surf)
    clean.Update()
    return pv.wrap(clean.GetOutput())


def skeleton_graph(mask):
    sk = skeletonize(mask).astype(bool)
    idx = np.argwhere(sk)
    lookup = {tuple(v): i for i, v in enumerate(idx)}
    G = nx.Graph()
    G.add_nodes_from(range(len(idx)))
    offs = np.argwhere(np.ones((3, 3, 3))) - 1
    offs = offs[np.any(offs != 0, axis=1)]
    for i, v in enumerate(idx):
        for o in offs:
            j = lookup.get(tuple(v + o))
            if j is not None and j > i:
                G.add_edge(i, j, w=float(np.linalg.norm(o)))
    return G, idx


def prune_spurs(G, radius, h, min_len, ratio, n_pass=3, keep=()):
    """Remove terminal branches shorter than max(min_len, ratio * junction radius).

    Edge weight "w" times h is the length in mm; endpoints in `keep` stay.
    Returns G and the number of branches removed.
    """
    n_removed = 0
    for _ in range(n_pass):
        removed = 0
        for e in [n for n in G if G.degree(n) == 1]:
            if e not in G or G.degree(e) != 1 or e in keep:
                continue
            path, length, prev, cur = [e], 0.0, None, e
            while True:
                nxt = [n for n in G.neighbors(cur) if n != prev]
                if len(nxt) != 1:
                    break
                length += G.edges[cur, nxt[0]]["w"] * h
                prev, cur = cur, nxt[0]
                if G.degree(cur) != 2:
                    break
                path.append(cur)
            if G.degree(cur) > 2 and length < max(min_len, ratio * radius[cur]):
                G.remove_nodes_from(path)
                removed += 1
        n_removed += removed
        if not removed:
            break
    return G, n_removed


def run(script, **kw):
    s = getattr(vmtkscripts, script)()
    for k, v in kw.items():
        setattr(s, k, v)
    s.Execute()
    return s


def lines_of(poly):
    """Point ids of each polyline cell."""
    lines, out, i = poly.lines, [], 0
    while i < len(lines):
        n = lines[i]
        out.append(lines[i + 1 : i + 1 + n])
        i += n + 1
    return out


def build_tree(cl, tol, max_miss):
    """Merge VMTK source->target lines (which coincide on shared vessels) into a tree.

    The source is the line end shared by most lines (VMTK snaps the source seed
    to the surface, so it can differ from the requested point). The longest line
    goes in first. Every other line is walked from its target towards the source
    until it comes within max(tol, 0.5 * radius) of the tree; that tail is kept
    and joined to the nearest tree point (the bifurcation). Lines that never
    rejoin, or end > max_miss mm from the source (steepest descent failed), are
    dropped.
    Returns points, radius, edges (i, j), the source point and the number of
    dropped lines.
    """
    pts_all, r_all = np.asarray(cl.points), np.asarray(cl[RADIUS])
    all_lines = lines_of(cl)
    ends = np.array([pts_all[[i[0], i[-1]]] for i in all_lines]).reshape(-1, 3)
    n_near = cKDTree(ends).query_ball_point(ends, max_miss, return_length=True)
    source = ends[np.argmax(n_near)]
    lines = []
    for ids in all_lines:
        p = pts_all[ids]
        # orient target -> source
        if np.linalg.norm(p[0] - source) < np.linalg.norm(p[-1] - source):
            ids = ids[::-1]
        if len(ids) > 2 and np.linalg.norm(pts_all[ids[-1]] - source) <= max_miss:
            lines.append(ids)
    n_dropped = len(all_lines) - len(lines)
    lines.sort(key=len, reverse=True)
    first = lines[0]
    pts, rad = list(pts_all[first]), list(r_all[first])
    edges = [(k, k + 1) for k in range(len(first) - 1)]
    for ids in lines[1:]:
        tree = cKDTree(np.array(pts))
        d, nn = tree.query(pts_all[ids])
        near = d <= np.maximum(tol, 0.5 * r_all[ids])
        if not near.any():
            n_dropped += 1
            continue
        k = int(np.argmax(near))  # first point (from the target) back on the tree
        if k == 0:
            continue  # target already covered
        base = len(pts)
        pts.extend(pts_all[ids[:k]])
        rad.extend(r_all[ids[:k]])
        edges += [(base + j, base + j + 1) for j in range(k - 1)]
        edges.append((base + k - 1, int(nn[k])))
    return np.array(pts), np.array(rad), np.array(edges), source, n_dropped


def close_loops(G, sk_xyz, sk_r, sk_p, pts, h, min_len, attach_dist, r_weak, p_weak):
    """Skeleton pieces the VMTK tree misses (other side of loops, failed targets).

    Returns new points, edges (in the index space continuing after `pts`), the
    number of pieces added and of attachments skipped. Each piece is joined to
    the nearest tree point where it touches covered skeleton, at most once per
    `attach_dist` mm: always at its most probable contact, elsewhere only where
    the contact is not thin (radius <= r_weak mm) or uncertain (consensus sk_p
    < p_weak), since each further attachment closes a loop.
    """
    tree = cKDTree(pts)
    nodes = np.array(list(G))
    d, nn = tree.query(sk_xyz[nodes])
    missed = set(nodes[d > sk_r[nodes] + h])
    nearest = dict(zip(nodes, nn))
    new_pts, edges, n_pieces, n_skipped = [], [], 0, 0
    base = len(pts)
    for comp in nx.connected_components(G.subgraph(missed)):
        if len(comp) * h < min_len:
            continue
        loc = {u: base + len(new_pts) + k for k, u in enumerate(comp)}
        new_pts.extend(sk_xyz[list(comp)])
        edges += [(loc[a], loc[b]) for a, b in G.subgraph(comp).edges]
        contacts = [(u, v) for u in comp for v in G.neighbors(u) if v not in missed]
        contacts.sort(key=lambda e: -min(sk_p[e[0]], sk_p[e[1]]))
        joined = []
        for u, v in contacts:
            t = nearest[v]
            if any(np.linalg.norm(pts[t] - pts[j]) <= attach_dist for j in joined):
                continue
            weak = min(sk_r[u], sk_r[v]) <= r_weak or min(sk_p[u], sk_p[v]) < p_weak
            if joined and weak:
                n_skipped += 1
                continue
            joined.append(t)
            edges.append((loc[u], t))
        n_pieces += 1
    return np.array(new_pts).reshape(-1, 3), edges, n_pieces, n_skipped


def split_branches(n_points, edges):
    """Polylines (point ids) between nodes of degree != 2."""
    G = nx.Graph()
    G.add_nodes_from(range(n_points))
    G.add_edges_from(map(tuple, edges))
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
    return out, G


def merge_close_junctions(pts, edges, min_len):
    """Contract internal branches (junction to junction) shorter than min_len mm.

    Child lines rejoin the tree a few steps apart, which otherwise leaves
    sub-mm branches between nearly coincident bifurcations.
    Returns edges (unused points stay, as isolated nodes) and the count.
    """
    n_merged = 0
    while True:
        polys, T = split_branches(len(pts), edges)
        short = [
            ids
            for ids in polys
            if T.degree(ids[0]) > 2
            and T.degree(ids[-1]) > 2
            and ids[0] != ids[-1]
            and np.linalg.norm(np.diff(pts[ids], axis=0), axis=1).sum() < min_len
        ]
        if not short:
            return edges, n_merged
        target = np.arange(len(pts))
        done = set()
        for ids in short:
            if done & set(ids):
                continue  # neighbour already contracted this round
            done |= set(ids)
            pts[ids[0]] = pts[ids].mean(axis=0)
            target[ids] = ids[0]
            n_merged += 1
        e = np.sort(target[edges], axis=1)
        edges = np.unique(e[e[:, 0] != e[:, 1]], axis=0)


def break_loops(pts, edges, p_pt, r_pt, loop_prob, loop_radius):
    """Cut every loop of the centerline graph that is not strong throughout.

    Branch strength = min over its interior points of min(consensus / loop_prob,
    radius / loop_radius) (p_pt, r_pt per point). In a maximum spanning tree of the
    branch graph each non-tree branch is the weakest branch of the loop it closes,
    so a loop is kept only if that branch has strength >= 1 and the loop has more
    than two branches (self-loops and parallel branches are skeleton artifacts).
    Other non-tree branches are cut at their weakest point (the stubs left there
    are removed with the other short terminal branches, see main).
    Returns edges and a summary of the kept loops (centre, perimeter, strength)
    and the number cut.
    """
    polys, _ = split_branches(len(pts), edges)
    score = np.minimum(p_pt / loop_prob, r_pt / loop_radius)
    B = nx.MultiGraph()
    for k, ids in enumerate(polys):
        inner = ids[1:-1] if len(ids) > 2 else ids
        B.add_edge(int(ids[0]), int(ids[-1]), key=k, w=float(score[inner].min()))
    tree = nx.Graph()
    tree.add_nodes_from(B)
    in_tree = set()
    for u, v, k, _ in nx.maximum_spanning_edges(B, weight="w", keys=True, data=True):
        tree.add_edge(u, v, key=k)
        in_tree.add(k)
    arc = [np.linalg.norm(np.diff(pts[ids], axis=0), axis=1).sum() for ids in polys]
    E = nx.Graph()
    E.add_edges_from(map(tuple, edges))
    kept = []
    for u, v, k, d in B.edges(keys=True, data=True):
        if k in in_tree:
            continue
        path = nx.shortest_path(tree, u, v) if u != v else [u]
        loop = [tree.edges[a, b]["key"] for a, b in pairwise(path)] + [k]
        if d["w"] >= 1 and len(loop) > 2:
            ids = np.concatenate([polys[b] for b in loop])
            kept.append(
                dict(
                    centre_xyz=np.round(pts[ids].mean(axis=0), 1).tolist(),
                    perimeter_mm=round(float(sum(arc[b] for b in loop)), 1),
                    n_branches=len(loop),
                    strength=round(d["w"], 2),
                )
            )
            continue
        ids = polys[k]
        if len(ids) == 2:
            E.remove_edge(int(ids[0]), int(ids[1]))
            continue
        c = ids[1:-1][np.argmin(score[ids[1:-1]])]
        E.remove_node(int(c))
    n_cut = B.number_of_edges() - len(in_tree) - len(kept)
    return np.array(E.edges, int).reshape(-1, 2), kept, n_cut


def smooth_along(values, pts, window):
    """Moving average over arc length `window` (mm), ends kept."""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    out = values.copy()
    for k in range(1, len(values) - 1):
        sel = np.abs(s - s[k]) <= window / 2
        out[k] = values[sel].mean()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--contact_prob", type=float, default=0.5, help="consensus")
    p.add_argument("--contact_radius", type=float, default=1.0, help="mm")
    p.add_argument("--contact_max_len", type=float, default=1.5, help="mm")
    p.add_argument("--contact_reach", type=float, default=2.0, help="mm")
    p.add_argument("--bridge_max_len", type=float, default=10.0, help="mm")
    p.add_argument(
        "--contact_cos",
        type=float,
        default=0.7,
        help="-cos of the angle above which two branches run straight through",
    )
    p.add_argument("--up", type=int, default=2, help="surface grid refinement")
    p.add_argument("--sigma", type=float, default=0.0, help="Gaussian, voxels")
    p.add_argument("--smooth_iter", type=int, default=30)
    p.add_argument("--pass_band", type=float, default=0.1)
    p.add_argument("--min_spur", type=float, default=2.0, help="mm")
    p.add_argument("--spur_ratio", type=float, default=3.0, help="x junction radius")
    p.add_argument("--step", type=float, default=0.25, help="resampling, mm")
    p.add_argument("--cl_smooth_iter", type=int, default=50)
    p.add_argument("--cl_smooth_factor", type=float, default=0.1)
    p.add_argument("--join_tol", type=float, default=0.25, help="mm")
    p.add_argument("--min_branch", type=float, default=1.0, help="mm")
    p.add_argument("--min_loop", type=float, default=1.5, help="mm")
    p.add_argument("--attach_dist", type=float, default=1.0, help="mm")
    p.add_argument("--loop_prob", type=float, default=0.5, help="consensus")
    p.add_argument("--loop_radius", type=float, default=0.75, help="mm")
    p.add_argument("--radius_window", type=float, default=1.5, help="mm")
    p.add_argument("--tube_sides", type=int, default=16)
    p.add_argument(
        "--reuse",
        action="store_true",
        help="reuse centerlines_raw.vtp (skip the ~2 min surface + VMTK step)",
    )
    p.add_argument("--network", default="network", help="input NAME.nii.gz")
    p.add_argument("--t2_prob", action="store_true", help="max(consensus, T2w)")
    args = p.parse_args()
    name = "centerlines" + args.network.removeprefix("network")

    img = nib.load(OUT / f"{args.network}.nii.gz")
    labels = np.asanyarray(img.dataobj).astype(np.uint8)
    h = float(np.mean(img.header.get_zooms()[:3]))
    prob = nib.load(OUT / "consensus.nii.gz").get_fdata(dtype=np.float32)
    if args.t2_prob:
        t2 = nib.load(OUT / "t1space" / "MS_t2_prob.nii.gz")
        prob = np.maximum(prob, t2.get_fdata(dtype=np.float32))

    # one tree: other pieces (build_network.py --t2_all) are not on the surface,
    # and their radius (distance to it) would be meaningless
    comp, _ = ndimage.label(labels > 0, CONN)
    sizes = np.bincount(comp.ravel())
    sizes[0] = 0
    mask = comp == sizes.argmax()
    n_dropped_vox = int((labels > 0).sum() - mask.sum())
    print(f"kept the largest piece: {n_dropped_vox} voxels in others", flush=True)
    edt = ndimage.distance_transform_edt(mask, sampling=h)
    G, idx = skeleton_graph(mask)
    r = edt[tuple(idx.T)]
    G, _ = prune_spurs(G, r, h, args.min_spur, args.spur_ratio)
    mask, cut, contacts = separate_contacts(mask, prob, G, idx, r, h, args)
    labels[cut] = 3
    nib.save(nib.Nifti1Image(labels, img.affine), OUT / f"{name}_mask.nii.gz")
    print(f"contacts: {contacts}, {cut.sum()} voxels cut", flush=True)
    if ndimage.label(mask, CONN)[1] > 1:
        print("warning: the contact cuts split the mask", flush=True)

    edt = ndimage.distance_transform_edt(mask, sampling=h)
    G, idx = skeleton_graph(mask)
    n_ends0 = sum(1 for n in G if G.degree(n) == 1)
    G, _ = prune_spurs(G, edt[tuple(idx.T)], h, args.min_spur, args.spur_ratio)
    ends = np.array([n for n in G if G.degree(n) == 1])
    ends_xyz = nib.affines.apply_affine(img.affine, idx[ends])
    order = np.argsort(ends_xyz[:, 2])
    source, targets = ends_xyz[order[0]], ends_xyz[order[1:]]
    print(f"endpoints: {n_ends0} -> {len(ends)} after pruning", flush=True)

    raw = OUT / f"{name}_raw.vtp"
    surf_path = OUT / f"{name}_surface.vtp"
    if args.reuse and raw.exists():
        cl, surf = pv.read(raw), pv.read(surf_path)
    else:
        mask6, n_added = face_connected(mask, cut)
        print(f"face connectivity: {n_added} voxels added", flush=True)
        surf = surface(
            mask6, img.affine, args.up, args.sigma, args.smooth_iter, args.pass_band
        )
        surf.save(surf_path)
        print(
            f"surface: {surf.n_points} points, {surf.volume / 1000:.2f} ml", flush=True
        )
        cl = run(
            "vmtkCenterlines",
            Surface=surf,
            DelaunayTessellation=tessellation(surf),
            SeedSelectorName="pointlist",
            SourcePoints=source.tolist(),
            TargetPoints=targets.ravel().tolist(),
            AppendEndPoints=1,
            Resampling=1,
            ResamplingStepLength=args.step,
        ).Centerlines
        cl = pv.wrap(cl)
        cl.save(raw)
    cl = pv.wrap(
        run(
            "vmtkCenterlineSmoothing",
            Centerlines=cl,
            NumberOfSmoothingIterations=args.cl_smooth_iter,
            SmoothingFactor=args.cl_smooth_factor,
        ).Centerlines
    )
    pts, r_raw, edges, source, n_dropped = build_tree(cl, args.join_tol, 2 * h)
    n_vmtk = len(pts)
    sk_xyz = nib.affines.apply_affine(img.affine, idx)
    sk_r = edt[tuple(idx.T)]
    extra, extra_edges, n_loops, n_weak = close_loops(
        G,
        sk_xyz,
        sk_r,
        prob[tuple(idx.T)],
        pts,
        h,
        args.min_loop,
        args.attach_dist,
        args.contact_radius,
        args.contact_prob,
    )
    pts = np.vstack([pts, extra])
    edges = np.vstack([edges, np.array(extra_edges, int).reshape(-1, 2)])
    origin = np.r_[np.zeros(n_vmtk, np.uint8), np.ones(len(extra), np.uint8)]
    r_raw = np.r_[r_raw, np.zeros(len(extra))]
    edges, n_merged = merge_close_junctions(pts, edges, args.min_branch)
    vox = np.round(nib.affines.apply_affine(np.linalg.inv(img.affine), pts)).astype(int)
    vox = tuple(np.clip(vox, 0, np.array(mask.shape) - 1).T)
    edges, loops_kept, n_loops_cut = break_loops(
        pts,
        edges,
        prob[vox],
        edt[vox],
        args.loop_prob,
        args.loop_radius,
    )
    print(f"loops: {n_loops_cut} cut, {len(loops_kept)} kept", flush=True)
    # short terminal branches: skeleton pieces, VMTK lines rejoining the tree just
    # before their target, and the stubs of the loop cuts
    E = nx.Graph()
    for a, b in edges:
        E.add_edge(a, b, w=float(np.linalg.norm(pts[a] - pts[b])))
    i_source = int(np.argmin(np.linalg.norm(pts - source, axis=1)))
    E, n_stubs = prune_spurs(
        E, edt[vox], 1.0, args.min_spur, args.spur_ratio, keep={i_source}
    )
    edges = np.array(E.edges, int).reshape(-1, 2)
    print(f"stubs: {n_stubs} removed", flush=True)
    # drop the points that contraction left unused
    used = np.unique(edges)
    new_id = np.full(len(pts), -1)
    new_id[used] = np.arange(len(used))
    pts, r_raw, origin, edges = pts[used], r_raw[used], origin[used], new_id[edges]
    polys, T = split_branches(len(pts), edges)

    # skeleton points: smooth the voxel staircase, radius = distance to surface
    # (the same definition as VMTK's maximum inscribed sphere, at a centre point)
    surf_tree = cKDTree(np.asarray(surf.points))
    for ids in polys:
        inner = ids[1:-1][origin[ids[1:-1]] == 1]
        if len(inner):
            sm = np.column_stack(
                [
                    smooth_along(pts[ids, k], pts[ids], args.radius_window)
                    for k in range(3)
                ]
            )
            pts[inner] = sm[1:-1][origin[ids[1:-1]] == 1]
    r_raw[origin == 1] = surf_tree.query(pts[origin == 1])[0]

    radius = r_raw.copy()
    for ids in polys:
        radius[ids[1:-1]] = smooth_along(r_raw[ids], pts[ids], args.radius_window)[1:-1]
    lengths = [np.linalg.norm(np.diff(pts[ids], axis=0), axis=1).sum() for ids in polys]

    net = pv.PolyData(pts, lines=np.concatenate([np.r_[len(i), i] for i in polys]))
    net["radius"] = radius
    node_type = np.zeros(net.n_points, np.uint8)
    node_type[[n for n in T if T.degree(n) == 1]] = 1
    node_type[np.argmin(np.linalg.norm(pts - source, axis=1))] = 2
    net["node_type"] = node_type
    net["origin"] = origin
    net.cell_data["branch"] = np.arange(net.n_cells)
    net.cell_data["length"] = np.array(lengths)
    net.cell_data["mean_radius"] = np.array([radius[i].mean() for i in polys])
    net.cell_data["origin"] = np.array([origin[i[1:-1]].max(initial=0) for i in polys])
    net.save(OUT / f"{name}.vtp")

    tubes = []
    for b, ids in enumerate(polys):
        line = pv.lines_from_points(pts[ids])
        line["radius"] = radius[ids]
        tube = line.tube(
            scalars="radius", absolute=True, n_sides=args.tube_sides, capping=True
        )
        tube.cell_data["branch"] = np.full(tube.n_cells, b)
        tube.cell_data["origin"] = np.full(tube.n_cells, net.cell_data["origin"][b])
        tubes.append(tube)
    pv.merge(tubes).save(OUT / f"{name}_tubes.vtp")

    # coverage: pruned-skeleton voxels not within their radius + 1 voxel of a centerline
    nodes = np.array(list(G))
    d, _ = cKDTree(pts).query(sk_xyz[nodes])
    missed = d > sk_r[nodes] + h
    stats = dict(
        args=vars(args),
        contacts=contacts,
        contact_voxels_cut=int(cut.sum()),
        surface_points=int(surf.n_points),
        surface_vol_ml=float(surf.volume / 1000),
        mask_vol_ml=float(mask.sum() * h**3 / 1000),
        other_pieces_vox=n_dropped_vox,
        endpoints_skeleton=int(n_ends0),
        endpoints_pruned=len(ends),
        source_xyz=np.round(source, 2).tolist(),
        source_seed_xyz=np.round(ends_xyz[order[0]], 2).tolist(),
        vmtk_lines=int(cl.n_cells),
        lines_dropped=int(n_dropped),
        loop_pieces_added=int(n_loops),
        weak_attachments_skipped=int(n_weak),
        junctions_merged=int(n_merged),
        loops_cut=int(n_loops_cut),
        loops_kept=loops_kept,
        stubs_removed=int(n_stubs),
        branches=int(net.n_cells),
        points=int(net.n_points),
        components=int(nx.number_connected_components(T)),
        loops=int(
            T.number_of_edges()
            - T.number_of_nodes()
            + nx.number_connected_components(T)
        ),
        total_length_mm=float(np.sum(lengths)),
        radius_mm_percentiles_5_50_95=np.round(
            np.percentile(radius, [5, 50, 95]), 3
        ).tolist(),
        skeleton_missed_fraction=float(missed.mean()),
    )
    (OUT / f"{name}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
