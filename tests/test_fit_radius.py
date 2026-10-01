import json

import numpy as np
import pyvista as pv

from conftest import R_KID, R_TRUNK
from vesselseg.io import polylines
from vesselseg.tree import fit_radius
from vesselseg.tree.radius_model import growth_cost, rel_var, validity


def test_rel_var_limits():
    assert np.isclose(rel_var(1e3, 0.5), 1.0)  # resolved vessel: relative error only
    assert np.isclose(rel_var(0.5, 0.5), 2.0)
    assert rel_var(0.2, 0.5) > rel_var(0.4, 0.5)  # thinner: less reliable


def test_validity_is_clipped_evidence():
    assert np.allclose(validity([-1, 0.3, 2]), [0, 0.3, 1])


def test_growth_cost():
    kw = dict(h=0.3, w=4.0, r_tol=1.25)
    assert growth_cost(1.0, 1.2, 1, 1, **kw) == 0.0  # within r_tol
    c2, c3 = growth_cost(1.0, 2.0, 1, 1, **kw), growth_cost(1.0, 3.0, 1, 1, **kw)
    assert 0 < c2 < c3
    assert c3 <= -np.log(np.exp(-4.0)) + 1e-9  # bounded by -log(1 - rho_max)
    assert growth_cost(1.0, 3.0, 0.1, 0.1, **kw) < c3  # faint evidence says little


def _fit(tmp_path, net, *args):
    path = tmp_path / "tree.vtp"
    net.save(path)
    json_path = tmp_path / "tree.json"
    json_path.write_text(json.dumps({"spacing_mm": [0.35, 1 / 3, 1 / 3]}))
    fit_radius.main([str(tmp_path / "tree"), *args])
    return pv.read(path), json.loads((tmp_path / "tree_radius_fit.json").read_text())


def test_fit_smooths_and_obeys_murray(tmp_path, y_tree):
    raw = np.asarray(y_tree.point_data["radius"]).copy()
    out, stats = _fit(tmp_path, y_tree)
    assert stats["params"]["h_mm"] == 0.34  # mean spacing from tree.json
    assert np.allclose(out.point_data["radius_raw"], raw)
    r = np.asarray(out.point_data["radius"])
    trunk, left, right = (r[ids] for ids in polylines(out))
    assert trunk[18:20].min() > 0.8 * R_TRUNK  # the pinch is overridden
    assert 0.85 * R_TRUNK < np.median(trunk) < 1.15 * R_TRUNK
    assert 0.85 * R_KID < np.median(left) < 1.15 * R_KID
    # Murray's law at the junction (k = 3)
    rp, r1, r2 = trunk[-5], left[5], right[5]
    assert abs(rp**3 / (r1**3 + r2**3) - 1) < 0.3
    assert stats["murray_violations_3x"]["after"] == 0
    assert (tmp_path / "tree_tubes.vtp").exists()


def test_fit_is_idempotent(tmp_path, y_tree):
    out1, _ = _fit(tmp_path, y_tree)
    out2, _ = _fit(tmp_path, out1)  # refits from radius_raw
    assert np.allclose(out1.point_data["radius"], out2.point_data["radius"])
    assert np.allclose(out1.point_data["radius_raw"], out2.point_data["radius_raw"])
