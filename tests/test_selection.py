"""The selection MILP (HiGHS + region search) on a toy candidate set."""

import numpy as np

from vesselseg.tree import extract as et

SP = np.array([0.5, 0.5, 0.5])  # mm per voxel


def toy_candidates():
    """Inlet trunk 0->1 splitting into 1->2 and 1->3; a faint spur 1->4; a loose
    piece 5->6 with evidence but no connection to the inlet."""
    node_pos = np.array(
        [[0, 0, 0], [0, 0, 40], [-20, 0, 60], [20, 0, 60], [0, 8, 46], [60, 0, 0], [60, 0, 30]],
        float,
    )
    spec = [  # (u, v, P, radius)
        (0, 1, 0.9, 1.26),
        (1, 2, 0.9, 1.0),
        (1, 3, 0.9, 1.0),
        (1, 4, 0.001, 0.3),
        (5, 6, 0.9, 1.0),
    ]
    segs = []
    for u, v, p, r in spec:
        n = int(np.ceil(np.linalg.norm(node_pos[v] - node_pos[u]))) + 1
        pts = np.linspace(node_pos[u], node_pos[v], n)
        segs.append(et.Segment(u, v, pts, np.full(n, p), np.full(n, r), SP))
    r_node = np.array([1.26, 1.26, 1.0, 1.0, 0.3, 1.0, 1.0])
    return segs, node_pos, r_node


def args_for(tmp_path, **kw):
    args = et.parser().parse_args(["--evidence", "-", "--sasparc", "-", "-o", str(tmp_path / "t")])
    args.workers, args.time_limit, args.lns_box_time = 1, 30.0, 10.0
    for k, v in kw.items():
        setattr(args, k, v)
    return args


def test_selection_is_inlet_tree(tmp_path):
    segs, node_pos, r_node = toy_candidates()
    A, parent, info = et.select_tree(
        segs, len(node_pos), r_node, {0}, args_for(tmp_path), node_pos * SP
    )
    used = {A[a][0]: A[a][1] for a in parent}  # segment -> forward
    assert used == {0: True, 1: True, 2: True}  # the spur and the loose piece go
    roots = [a for a, p in parent.items() if p == -1]
    assert [A[a][0] for a in roots] == [0]
    for b, a in parent.items():  # every used arc starts where its parent ends
        if a != -1:
            assert A[b][2] == A[a][3]
    assert info["status"] == "region optimum"


def test_weight_saturates():
    p0 = 0.005
    assert et.weight(np.array([1.0]), p0)[0] == 1.0
    assert et.weight(np.array([1e-6]), p0)[0] == -2.0
    assert et.weight(np.array([p0]), p0)[0] == 0.0


def test_spur_with_evidence_is_kept(tmp_path):
    segs, node_pos, r_node = toy_candidates()
    segs[3].p[:] = 0.9  # the spur is now a real (5 mm) side branch
    A, parent, _ = et.select_tree(
        segs, len(node_pos), r_node, {0}, args_for(tmp_path), node_pos * SP
    )
    assert {A[a][0] for a in parent} == {0, 1, 2, 3}
