import numpy as np

from vesselseg.tree.extract import bridge_job


def test_bridge_job_straight_tube():
    """A cheap straight channel along x: the geodesic from one end follows it to the target."""
    cost = np.full((16, 9, 9), 20.0)
    cost[:, 4, 4] = 1.0
    sp = np.ones(3)
    targets = {7: np.array([[13, 4, 4]])}
    job = (0, np.array([2.0, 4, 4]), np.array([1.0, 0, 0]), cost, np.zeros(3, int), targets)
    e, res = bridge_job((*job, sp, 1.0, 8, 0.4, 1))
    assert e == 0 and len(res) == 1
    value, seg, q, geo = res[0]
    assert seg == 7 and tuple(q) == (13, 4, 4)
    assert np.isfinite(value) and value >= 11.0  # at least the Euclidean length at cost 1
    assert np.allclose(geo[0], [2, 4, 4], atol=0.6) and np.allclose(geo[-1], [13, 4, 4], atol=0.6)
    assert np.abs(geo[:, 1:] - 4).max() < 1.0  # stays in the channel
