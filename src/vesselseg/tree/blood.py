"""Steady blood pressure on a vessel network (Poiseuille), shared by extract.py and label.py.

Every edge e of a graph is a tube with hydraulic conductance

    G_e = 1 / int_e ds / r^4        (Poiseuille at unit viscosity, constants dropped)

and the pressure p solves the network's mass balance sum_e G_e (p_i - p_j) = 0 at
every free node, with p = 1 at the inlets and p = 0 at every other tip (the
outlets). Blood runs from high to low p, so p orients every edge, also around a
loop: where two flows round a loop meet, p has a local minimum.

This is the same model as proteoflow's blood_flow_direction (a CG1 line mesh
with coefficient R1^4), without the finite-element machinery. A component with
no fixed node (no tip) gets NaN.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve


def conductance(s, r, r_min=1e-3):
    """``1 / int ds / r^4`` along one polyline with arclength ``s`` and radius
    ``r`` per point (midpoint rule, r floored at ``r_min``)."""
    s, r = np.asarray(s, float), np.maximum(np.asarray(r, float), r_min)
    ds = np.diff(s)
    rm = 0.5 * (r[:-1] + r[1:])
    res = float(np.sum(ds / rm**4))
    return 1.0 / max(res, 1e-12)


def pressure(n, edges, G, fixed):
    """Node pressures of the network ``edges`` (m, 2) with conductances ``G``
    (m,) on ``n`` nodes, ``fixed`` a ``{node: pressure}`` dict. Nodes of a
    component without a fixed node get NaN."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    G = np.asarray(G, dtype=np.float64)
    i, j = edges.T
    W = sp.coo_matrix((np.r_[G, G], (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    Lap = (sp.diags(np.asarray(W.sum(axis=1)).ravel()) - W).tocsr()
    p = np.full(n, np.nan)
    if not fixed:
        return p
    fx = np.fromiter(fixed, dtype=np.int64)
    p[fx] = np.fromiter(fixed.values(), dtype=np.float64)
    _, comp = connected_components(W, directed=False)
    solvable = np.isin(comp, comp[fx])
    solvable[fx] = False
    free = np.flatnonzero(solvable)
    if len(free):
        rhs = -Lap[free][:, fx] @ p[fx]
        p[free] = spsolve(Lap[free][:, free].tocsc(), rhs)
    return p


def network_pressure(points, edges, radius, inlets, tol=1e-6):
    """Pressure per point of a centerline graph (``edges`` between consecutive
    points of the polylines, ``radius`` per point), p = 1 at the ``inlets``
    (point indices) and 0 at every other tip. Points within ``tol`` are one node
    (vesselseg offsets the copies of a crossing by 0.03 mm, so a crossing stays
    two vessels). Returns ``(p (n,), G (m,))``; an edge of zero length takes
    the pressure of its ends."""
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    radius = np.asarray(radius, dtype=np.float64)
    n = len(points)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in cKDTree(points).query_pairs(tol):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    rep = np.array([find(k) for k in range(n)])
    e = rep[edges]
    keep = e[:, 0] != e[:, 1]
    length = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    rm = np.maximum(0.5 * (radius[edges[:, 0]] + radius[edges[:, 1]]), 1e-3)
    G = rm**4 / np.maximum(length, 1e-12)
    deg = np.bincount(e[keep].ravel(), minlength=n)
    inlet_rep = set(rep[np.asarray(list(inlets), dtype=np.int64)].tolist())
    fixed = {int(v): 0.0 for v in np.flatnonzero(deg == 1)}
    fixed.update({v: 1.0 for v in inlet_rep})
    p = pressure(n, e[keep], G[keep], fixed)
    return p[rep], G
