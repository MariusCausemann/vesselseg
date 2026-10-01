import itertools

import nibabel as nib
import numpy as np
import pytest

from conftest import polyline_net
from vesselseg.tree import label as lb


def branch_tree():
    """A -0-> B, B -1-> C, B -2-> D, D -3-> E (branch ids), as a label.Tree."""
    X = dict(A=[0, 0, 0], B=[0, 0, 10], C=[-8, 0, 16], D=[8, 0, 16], E=[8, 0, 26])
    ends = [("A", "B"), ("B", "C"), ("B", "D"), ("D", "E")]
    chains = [np.linspace(X[a], X[b], 6) for a, b in ends]
    net = polyline_net(chains, [np.ones(6)] * 4)
    net.cell_data["mean_radius"] = np.ones(4)
    net.cell_data["length"] = np.array([np.linalg.norm(c[-1] - c[0]) for c in chains])
    tree = lb.Tree(net)
    node = {k: int(np.argmin(np.linalg.norm(tree.X - v, axis=1))) for k, v in X.items()}
    return tree, node


def brute_force(tree, U, TR, root, TRv):
    parent = dict(lb.nx.bfs_predecessors(tree.T, root))
    into = {v: tree.T[u][v]["c"] for u, v in lb.nx.bfs_edges(tree.T, root)}
    pairs = [(into[u], into[v], u) for v, u in parent.items() if u != root]
    best, arg = np.inf, None
    K = U.shape[1]
    for labs in itertools.product(range(K), repeat=len(U)):
        cost = sum(U[c, labs[c]] for c in range(len(U)))
        cost += sum(TRv.get(n, TR)[labs[p], labs[c]] for p, c, n in pairs)
        if cost < best:
            best, arg = cost, labs
    return best, arg


@pytest.mark.parametrize("seed", range(5))
def test_label_tree_is_exact(seed):
    tree, node = branch_tree()
    rng = np.random.default_rng(seed)
    K = 3
    U = rng.uniform(0, 3, (4, K))
    TR = rng.uniform(0, 2, (K, K))
    np.fill_diagonal(TR, 0)
    TRv = {node["D"]: rng.uniform(0, 2, (K, K))}  # an override at one junction
    lab, into, *_ = lb.label_tree(tree, U, TR, node["A"], TRv)
    labs = [lab[c] for c in range(4)]
    cost = sum(U[c, labs[c]] for c in range(4))
    cost += TR[labs[0], labs[1]] + TR[labs[0], labs[2]] + TRv[node["D"]][labs[2], labs[3]]
    best, _ = brute_force(tree, U, TR, node["A"], TRv)
    assert np.isclose(cost, best)


def test_tree_rejects_cycles():
    sq = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]]
    chains = [np.linspace(sq[i], sq[(i + 1) % 4], 4) for i in range(4)]
    net = polyline_net(chains, [np.ones(4)] * 4)
    net.cell_data["mean_radius"] = np.ones(4)
    net.cell_data["length"] = np.full(4, 10.0)
    with pytest.raises(SystemExit, match="cycles"):
        lb.Tree(net)


def test_downstream_lengths():
    tree, node = branch_tree()
    tot = lb.downstream(tree, node["A"], tree.length)
    L = tree.length
    assert np.isclose(tot[0], L.sum())
    assert np.isclose(tot[2], L[2] + L[3])


def test_classes():
    assert lb.classes(1028, -1) == [lb.ACA]  # left superior frontal
    assert lb.classes(2030, -1) == [lb.MCA_R]  # right superior temporal: hemisphere wins
    assert lb.classes(1011, +1) == [lb.MCA_L, lb.PCA_L]  # lateral occipital: watershed
    assert lb.classes(8, +1) == [lb.CRB]  # cerebellar cortex
    assert lb.classes(16, +1) == [lb.CRB, lb.PCA_R]  # brainstem: side from the midplane
    assert lb.classes(0, 1) is None


def test_nearest_labels():
    lab = np.zeros((20, 20, 20), np.int16)
    lab[5, 5, 5] = 1028
    lab[15, 15, 15] = 2030
    img = nib.Nifti1Image(lab, np.diag([0.5, 0.5, 0.5, 1]))
    pts = np.array([[2.5, 2.5, 3.0], [7.5, 7.5, 7.0], [5.0, 5.0, 5.0]])
    assert lb.nearest_labels(pts, img, 1.0).tolist() == [1028, 2030, 0]


def test_midplane():
    lab = np.zeros((20, 10, 10), np.int16)
    lab[2:6] = 1028  # left cortex at low x
    lab[14:18] = 2028
    c, n = lb.midplane(nib.Nifti1Image(lab, np.eye(4)))
    assert np.allclose(n, [1, 0, 0]) and np.isclose(c[0], 9.5)
