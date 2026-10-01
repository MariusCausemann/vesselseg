"""anat2vessels (nrdg/anat2vessels, Gilmore et al. 2025) nnU-Net models on sub-01.

Preprocessing mirrors anat2vessels.preprocess.preprocess_img with
--skull_strip (rigid ANTs registration to the skull-stripped package reference,
resampling onto it, cropping), re-implemented because that module imports
ray/pybids at import time. Differences: the skull strip uses the proteoflow
brain+CSF segmentation dilated by --margin_mm instead of antspynet, and for
t1t2 one T1->ref transform is applied to the T2w already coregistered onto the
T1 grid, so both channels stay aligned. The transform is kept to map the
probabilities back to the native grid.

Models (installed in $nnUNet_results beforehand; this script does not download):
  t1 -> Dataset076, t2 -> Dataset086, t1t2 -> Dataset096 (channel 0 = T1, 1 = T2)

Output: results/sub-01/anat2vessels/{model}/{t1key}[_{t2key}]_prob.nii.gz on the
native grid of the first input.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import ants
import nibabel as nib
import numpy as np
from common import MODALITIES, RESULTS, ROOT, T2_ON_T1, head_roi

DATASETS = {"t1": "076", "t2": "086", "t1t2": "096"}
PLANS = ["-tr", "nnUNetTrainer", "-c", "3d_fullres", "-p", "nnUNetResEncUNetLPlans"]
REF = Path.home() / ".cache/anat2vessels/ref.nii.gz"


def installed(model):
    res = Path(os.environ.get("nnUNet_results", ""))
    return res.is_dir() and any(
        d.name.startswith(f"Dataset{DATASETS[model]}_") for d in res.iterdir()
    )


def brain_masked(path, margin_mm):
    """Image times the dilated brain+CSF mask (replaces anat2vessels' antspynet skull strip)."""
    img = nib.load(path)
    mask = head_roi(img, margin_mm)
    tmp = Path(tempfile.mkdtemp(dir=ROOT / "data")) / Path(path).name
    nib.save(nib.Nifti1Image(img.get_fdata(dtype=np.float32) * mask, img.affine), tmp)
    return ants.image_read(str(tmp))


def to_ref(img, ref, fwd=None):
    """Rigid to the reference (or apply `fwd`), resample onto it and crop, as upstream."""
    if fwd is None:
        fwd = ants.registration(fixed=ref, moving=img, type_of_transform="Rigid")[
            "fwdtransforms"
        ]
    out = ants.apply_transforms(
        fixed=ref, moving=img, transformlist=fwd, interpolator="linear"
    )
    return ants.crop_image(out, ref), fwd


def run(model, keys, folds, tta, margin_mm):
    ref = ants.image_read(str(REF))
    name = "_".join(keys)
    out = RESULTS / "anat2vessels" / model
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "data") as tmp:
        tin, tout = Path(tmp) / "in", Path(tmp) / "out"
        tin.mkdir()
        # all channels share the transform of the first one, so they stay co-registered;
        # T2 is taken from its pipeline coregistration onto the T1 grid
        fwd = None
        for ch, k in enumerate(keys):
            src = T2_ON_T1 if (k == "t2" and "t1" in keys) else MODALITIES[k][0]
            img, fwd = to_ref(brain_masked(src, margin_mm), ref, fwd)
            ants.image_write(img, str(tin / f"{name}_{ch:04d}.nii.gz"))
            ants.image_write(img, str(out / f"{name}_input{ch:04d}_refspace.nii.gz"))
        cmd = [
            "nnUNetv2_predict",
            "-d",
            DATASETS[model],
            "-i",
            str(tin),
            "-o",
            str(tout),
        ]
        cmd += ["-f", *folds, *PLANS, "--save_probabilities"]
        if not tta:
            cmd.append("--disable_tta")
        subprocess.run(cmd, check=True)

        seg = ants.image_read(str(tout / f"{name}.nii.gz"))
        prob = np.load(tout / f"{name}.npz")["probabilities"][1]
        # axis order of the .npz depends on the dataset's image reader (SimpleITK:
        # z,y,x; nibabel: x,y,z); take the one that agrees with the exported seg
        s = seg.numpy() > 0
        cands = [c for c in (prob.transpose(2, 1, 0), prob) if c.shape == s.shape]
        prob = max(cands, key=lambda c: ((c > 0.5) == s).mean())
        prob = seg.new_image_like(np.ascontiguousarray(prob).astype(np.float32))
        native = ants.image_read(str(MODALITIES[keys[0]][0]))
        back = ants.apply_transforms(
            fixed=native, moving=prob, transformlist=fwd, whichtoinvert=[True]
        )
        ants.image_write(back, str(out / f"{name}_prob.nii.gz"))
        ants.image_write(prob, str(out / f"{name}_refspace_prob.nii.gz"))
        print(
            f"{model}/{name}: frac>0.5 (ref space) = {(prob.numpy() > 0.5).mean():.5f}",
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models", nargs="+", default=["t1t2", "t2"], choices=list(DATASETS)
    )
    p.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4"])
    p.add_argument("--no_tta", action="store_true")
    p.add_argument(
        "--margin_mm", type=float, default=2.0, help="dilation of the brain mask"
    )
    args = p.parse_args()
    runs = {"t1": ["t1"], "t2": ["t2"], "t1t2": ["t1", "t2"]}
    missing = [m for m in args.models if not installed(m)]
    if missing or not REF.exists():
        sys.exit(
            f"anat2vessels models not installed in $nnUNet_results: {missing} (or ref image missing). "
            "Install them first, see README."
        )
    for m in args.models:
        run(m, runs[m], args.folds, not args.no_tta, args.margin_mm)
