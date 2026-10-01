import numpy as np
import pytest
import pyvista as pv

R_TRUNK, R_KID = 2.0 ** (1 / 3), 1.0  # Murray: r_p^3 = 2 r_d^3


def polyline_net(chains, radius, extra_point=None):
    """PolyData with one polyline per chain of points (consecutive ids)."""
    pts, lines, ids = [], [], 0
    for c in chains:
        pts.append(c)
        lines += [len(c), *range(ids, ids + len(c))]
        ids += len(c)
    net = pv.PolyData(np.vstack(pts), lines=np.array(lines))
    net.point_data["radius"] = np.concatenate(radius)
    for k, v in (extra_point or {}).items():
        net.point_data[k] = v
    return net


@pytest.fixture
def y_tree():
    """Trunk up the z axis, splitting into two daughters; noisy radii, a pinch."""
    rng = np.random.default_rng(0)
    s = np.linspace(0, 20, 41)
    trunk = np.c_[np.zeros_like(s), np.zeros_like(s), s]
    t = np.linspace(0, 1, 31)[:, None]
    left = trunk[-1] + t * np.array([-10.0, 0, 10])
    right = trunk[-1] + t * np.array([10.0, 0, 10])
    r_trunk = R_TRUNK * (1 + 0.05 * rng.standard_normal(len(trunk)))
    r_trunk[18:20] = 0.5  # local pinch (vessel contact)
    radius = [
        r_trunk,
        R_KID * (1 + 0.05 * rng.standard_normal(len(left))),
        R_KID * (1 + 0.05 * rng.standard_normal(len(right))),
    ]
    radius[1][0] = radius[2][0] = r_trunk[-1]  # junction point carries the parent's
    net = polyline_net([trunk, left, right], radius)
    n = net.n_points
    node_type = np.zeros(n, np.uint8)
    node_type[0] = 2  # the source inlet
    net.point_data["node_type"] = node_type
    net.point_data["evidence"] = np.full(n, 0.9, np.float32)
    net.cell_data["mean_radius"] = np.array([r.mean() for r in radius])
    net.cell_data["length"] = np.array([20.0, np.sqrt(200), np.sqrt(200)])
    net.cell_data["origin"] = np.zeros(3, int)
    net.cell_data["branch"] = np.arange(3)
    return net
