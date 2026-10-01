"""Panel of coronal / axial / sagittal MIPs of probability maps.

Every map is brought into T1w space (rigid transforms from register_to_t1.py)
on a common 0.5 mm grid (the prepared T1w), so panels are directly comparable
across models, input series and scales.

usage: plot_mips.py OUT.png PROB.nii.gz [PROB.nii.gz ...]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from common import PREP, RESULTS, to_t1

REF = PREP / "vesselfm" / "t1_0.50mm.nii.gz"  # T1w space


def label(p):
    p = Path(p)
    return f"{p.parent.name}/{p.name.replace('_prob.nii.gz', '')}"


def mips(path, ref):
    a = to_t1(path, ref)
    return [np.rot90(a.max(1)), np.rot90(a.max(2)), np.rot90(a.max(0))]


def main(out, paths):
    ref = nib.load(REF)
    rows = [("T2 black top-hat", RESULTS / "t2_blacktophat_0.50mm.nii.gz")]
    rows = [r for r in rows if r[1].exists()] + [(label(p), p) for p in paths]
    fig, ax = plt.subplots(len(rows), 3, figsize=(13, 4.3 * len(rows)), squeeze=False)
    for i, (name, p) in enumerate(rows):
        vmax = 0.8 if "top-hat" in name else 1.0
        for j, (m, v) in enumerate(zip(mips(p, ref), ["coronal", "axial", "sagittal"])):
            ax[i, j].imshow(m, cmap="magma", vmin=0, vmax=vmax, interpolation="none")
            ax[i, j].set_title(f"{name} — {v}", fontsize=9)
            ax[i, j].axis("off")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
