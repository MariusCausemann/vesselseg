"""`vesselseg label` end to end on a schematic arterial tree, and the output
contract proteoflow reads (src/proteoflow/pvs.py read_centerlines,
scripts/run_circulation.py read_inlets)."""

import json

import nibabel as nib
import numpy as np
import pyvista as pv

from conftest import polyline_net
from vesselseg.tree import label as lb

# T1w world, mm (x to the right); one entry per branch: (start, end, radius)
P = dict(
    va_l=(-6, -10, -60), va_r=(6, -10, -60), conf=(0, -10, -40), tip=(0, -10, -10),
    p1_l=(-10, -15, -8), p1_r=(10, -15, -8), pca_l=(-25, -50, 5), pca_r=(25, -50, 5),
    ica_l=(-20, 10, -40), ica_r=(20, 10, -40), term_l=(-20, 10, -10), term_r=(20, 10, -10),
    mca_l=(-60, 10, 0), mca_r=(60, 10, 0), aca_l=(-3, 45, 25), aca_r=(3, 45, 25),
)  # fmt: skip
BRANCHES = [
    ("va_l", "conf", 1.6), ("va_r", "conf", 1.4), ("conf", "tip", 1.5),
    ("tip", "p1_l", 1.0), ("tip", "p1_r", 1.0), ("p1_l", "pca_l", 0.9), ("p1_r", "pca_r", 0.9),
    ("ica_l", "term_l", 2.0), ("ica_r", "term_r", 2.0),
    ("term_l", "p1_l", 0.6), ("term_r", "p1_r", 0.6),  # PCom
    ("term_l", "mca_l", 1.3), ("term_r", "mca_r", 1.3),
    ("term_l", "aca_l", 1.0), ("term_r", "aca_r", 1.0),
]  # fmt: skip


def schematic_tree(tmp_path):
    chains, radius = [], []
    for a, b, r in BRANCHES:
        n = int(np.linalg.norm(np.subtract(P[b], P[a]))) + 2
        chains.append(np.linspace(P[a], P[b], n))
        radius.append(np.full(n, r))
    net = polyline_net(chains, radius)
    node_type = np.zeros(net.n_points, np.uint8)
    node_type[0] = 2  # VA-L start: the source
    net.point_data["node_type"] = node_type
    net.cell_data["mean_radius"] = np.array([r for *_, r in BRANCHES])
    net.cell_data["length"] = np.array([len(c) - 1.0 for c in chains])
    net.save(tmp_path / "tree.vtp")
    names = {"VA-L": "va_l", "VA-R": "va_r", "ICA-L": "ica_l", "ICA-R": "ica_r"}
    meta = {"inlets_xyz": {k: list(map(float, P[v])) for k, v in names.items()}}
    (tmp_path / "tree.json").write_text(json.dumps(meta))


def schematic_sasparc(tmp_path):
    """Cortex labels around the territories' ends, brainstem around the basilar."""
    aff = np.eye(4)
    aff[:3, 3] = [-80, -70, -70]
    lab = np.zeros((161, 141, 111), np.int16)

    def put(xyz, value, r=2):
        i, j, k = np.round(np.subtract(xyz, aff[:3, 3])).astype(int)
        lab[i - r : i + r, j - r : j + r, k - r : k + r] = value

    def along(a, b, value):  # the distal 70% of a branch, 2 mm beside it
        for t in np.linspace(0.3, 1, 40):
            put(np.add(P[a], t * np.subtract(P[b], P[a])) + [0, 0, 3], value)

    for side, off in (("l", 1000), ("r", 2000)):
        along(f"term_{side}", f"mca_{side}", off + 30)  # superior temporal: MCA
        along(f"term_{side}", f"aca_{side}", off + 28)  # superior frontal: ACA
        along(f"p1_{side}", f"pca_{side}", off + 5)  # cuneus: PCA
    nib.save(nib.Nifti1Image(lab, aff), tmp_path / "sasparc.nii.gz")


def read_inlets(path):
    """scripts/run_circulation.py read_inlets."""
    inlets = json.loads(path.read_text())["inlets"]
    out = {}
    for name, val in inlets.items():
        rows = val if isinstance(val[0], list) else [val]
        out[name] = np.asarray(rows, float)
    return out


def test_label_end_to_end(tmp_path):
    schematic_tree(tmp_path)
    schematic_sasparc(tmp_path)
    lb.main([str(tmp_path / "tree"), "--sasparc", str(tmp_path / "sasparc.nii.gz")])

    for f in ("tree_labeled.vtp", "tree_labeled.json", "tree_labeled.png"):
        assert (tmp_path / f).exists(), f
    out = pv.read(tmp_path / "tree_labeled.vtp")
    names = [str(s) for s in out.field_data["segment_names"]]
    assert names == lb.NAMES
    assert "radius" in out.point_data and "segment" in out.point_data
    seg = np.asarray(out.cell_data["segment"])
    assert len(seg) == out.n_cells
    got = {(a, b): names[s] for (a, b, _), s in zip(BRANCHES, seg)}
    assert got["ica_l", "term_l"] == "ICA-L" and got["ica_r", "term_r"] == "ICA-R"
    assert got["va_l", "conf"] == "VB" and got["conf", "tip"] == "VB"
    assert got["term_l", "mca_l"] == "MCA-L" and got["term_r", "mca_r"] == "MCA-R"
    assert got["term_l", "aca_l"] == "ACA"
    assert got["p1_l", "pca_l"] == "PCA-L" and got["p1_r", "pca_r"] == "PCA-R"

    inlets = read_inlets(tmp_path / "tree_labeled.json")
    assert set(inlets) == {"VA", "ICA-L", "ICA-R"}
    assert inlets["VA"].shape == (2, 3) and inlets["ICA-L"].shape == (1, 3)
    assert np.allclose(inlets["ICA-L"][0], P["ica_l"])
