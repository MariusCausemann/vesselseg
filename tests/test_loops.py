"""Blood pressure (blood.py), loop returns (extract.py) and their re-cut (label.py)."""

import networkx as nx
import numpy as np
import pyvista as pv

from vesselseg.io import polylines
from vesselseg.tree import label as lb
from vesselseg.tree.blood import conductance, network_pressure, pressure
from vesselseg.tree.extract import loop_returns


def test_pressure_by_hand():
    # inlet 0 -> 1 (G 1), then 1 -> 2 (G 1) and 1 -> 3 (G 3): p1 = 1 / 5;
    # 4 - 5 has no fixed node
    edges = [(0, 1), (1, 2), (1, 3), (4, 5)]
    p = pressure(6, edges, [1.0, 1.0, 3.0, 1.0], {0: 1.0, 2: 0.0, 3: 0.0})
    assert np.allclose(p[:4], [1.0, 0.2, 0.0, 0.0])
    assert np.isnan(p[4:]).all()


def test_conductance_uniform_tube():
    s = np.linspace(0, 4.0, 9)
    assert np.isclose(conductance(s, np.full(9, 0.5)), 0.5**4 / 4.0)


def test_network_pressure_merges_coincident_points():
    # a chain 0-1-2 whose middle point is stored twice (1 and 3, 1e-9 apart)
    pts = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 1 + 1e-9]], float)
    p, _ = network_pressure(pts, [(0, 1), (3, 2)], np.ones(4), inlets=[0])
    assert np.allclose(p, [1.0, 0.5, 0.0, 0.5])


def test_loop_returns():
    # arcs (segment, fwd, start node, end node): 0 -> 1, 1 -> 2, 2 -> 3, 3 -> 1
    # (back to the start node of arc 1); arc 4: 1 -> 4 is an ordinary child
    A = [(0, True, 0, 1), (1, True, 1, 2), (2, True, 2, 3), (3, True, 3, 1), (4, True, 1, 4)]
    parent = {0: -1, 1: 0, 2: 1, 3: 2, 4: 0}
    assert loop_returns(A, parent) == [(3, 1)]


def line(a, b, n=6):
    return np.linspace(a, b, n)


def loop_net(label=lb.ACA):
    """Trunk 0 -> J, a loop J -> A -> B -> C -> J' closed back at J (J' a copy
    of J, the way extract.py writes a return), and an outlet B -> D. The right
    side (B -> C -> J') is thinner, so B is fed from the left."""
    J, A, B, C, D = (np.array(x, float) for x in ([0, 0, 10], [-5, 0, 15], [0, 0, 20],
                                                   [5, 0, 15], [0, 0, 30]))  # fmt: skip
    trunk = line([0, 0, 0], J)
    left = np.vstack([line(J, A), line(A, B)[1:]])
    right = np.vstack([line(B, C)[1:], line(C, J + 0.03)[1:]])
    loop = np.vstack([left, right])
    out = line(B, D)
    pts = np.vstack([trunk, loop[1:], out[1:]])
    n_t, n_l = len(trunk), len(loop) - 1
    i_J = n_t - 1
    i_B = i_J + len(left) - 1
    i_Jc = i_J + n_l
    lines = [
        list(range(n_t)),
        [i_J, *range(n_t, n_t + n_l)],
        [i_B, *range(n_t + n_l, len(pts))],
    ]
    net = pv.PolyData(pts, lines=np.concatenate([[len(c), *c] for c in lines]))
    r = np.full(len(pts), 1.0)
    r[i_J + 1 : i_Jc + 1] = 0.8
    r[i_B + 1 : i_Jc + 1] = 0.5
    net.point_data["radius"] = r
    net.cell_data["origin"] = np.zeros(3, int)
    net.field_data["loop_returns"] = np.array([[i_Jc, i_J]])
    seg = np.array([lb.ACA, label, lb.ACA])
    return net, seg


def as_graph(net):
    G = nx.Graph()
    for ids in polylines(net):
        G.add_edges_from(zip(ids[:-1], ids[1:]))
    return G


def test_recut_moves_the_closure_to_where_the_flows_meet():
    net, seg = loop_net()
    out, seg_out, rep = lb.recut_loops(net, seg, [[0.0, 0.0, 0.0]])
    assert len(rep["recut"]) == 1 and rep["skipped"] == 0
    assert np.allclose(rep["recut"][0]["cut_xyz"], [0, 0, 20])  # at B
    G = as_graph(out)
    assert nx.is_tree(G)
    assert set(seg_out) == {lb.ACA}
    # every polyline runs down the pressure, and C is now fed from J: the
    # right side ends next to B instead of at the copy of J
    P, p = np.asarray(out.points), lb.blood_pressure(out, [[0.0, 0.0, 0.0]])
    for ids in polylines(out):
        assert p[ids[0]] >= p[ids[-1]]
    c = int(np.argmin(np.linalg.norm(P - [5, 0, 15], axis=1)))
    j = int(np.argmin(np.linalg.norm(P - [0, 0, 10], axis=1)))
    assert p[j] > p[c]
    tips = [n for n in G if G.degree(n) == 1]
    assert sum(np.linalg.norm(P[tips] - [0, 0, 20], axis=1) < 0.1) == 1  # B's copy
    assert "loop_returns" not in out.field_data


def test_recut_skips_loops_outside_one_territory():
    net, seg = loop_net(label=lb.COW)
    out, seg_out, rep = lb.recut_loops(net, seg, [[0.0, 0.0, 0.0]])
    assert out is net and rep == dict(recut=[], skipped=1)
