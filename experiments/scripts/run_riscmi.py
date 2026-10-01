"""risc-mi cerebral-artery nnU-Net models (MC: MIDL 2024, MS: MIUA 2024a) on sub-01.

Inputs are the RAI-oriented images from prep_inputs.py. MS was exported with
file_ending .nrrd, so its inputs are converted. nnU-Net writes the vessel-class
probability as .npz on the input grid; it is converted to NIfTI with the
geometry of the predicted segmentation.

Output: results/sub-01/riscmi/{model}/{key}_prob.nii.gz (+ {key}_seg.nii.gz)
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from common import MODALITIES, PREP, RESULTS, ROOT

W = ROOT / "weights/riscmi"
MODELS = {
    "MC": (W / "MC/nnUNetTrainer__nnUNetPlans__3d_fullres", ".nii.gz"),
    "MS": (
        W
        / "MA_MS/Structural MRI/Dataset606_MRIVessels/nnUNetTrainerDA5__nnUNetPlans__3d_fullres",
        ".nrrd",
    ),
}


def run(model, keys, tta, folds, chk):
    mdir, ending = MODELS[model]
    out = RESULTS / "riscmi" / model
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "data") as tmp:
        tin, tout = Path(tmp) / "in", Path(tmp) / "out"
        tin.mkdir()
        for k in keys:
            src = PREP / "riscmi" / k / f"{k}_0000.nii.gz"
            dst = tin / f"{k}_0000{ending}"
            if ending == ".nii.gz":
                shutil.copy(src, dst)
            else:
                sitk.WriteImage(sitk.ReadImage(str(src)), str(dst))
        cmd = [
            "nnUNetv2_predict_from_modelfolder",
            "-m",
            str(mdir),
            "-i",
            str(tin),
            "-o",
            str(tout),
        ]
        cmd += [
            "-f",
            *folds,
            "-chk",
            chk,
            "--save_probabilities",
            "-npp",
            "2",
            "-nps",
            "2",
        ]
        if not tta:
            cmd.append("--disable_tta")
        subprocess.run(cmd, check=True)
        for k in keys:
            seg = sitk.ReadImage(str(tout / f"{k}{ending}"))
            prob = np.load(tout / f"{k}.npz")["probabilities"][1].astype(np.float32)
            p = sitk.GetImageFromArray(prob)
            p.CopyInformation(seg)
            sitk.WriteImage(p, str(out / f"{k}_prob.nii.gz"))
            sitk.WriteImage(seg, str(out / f"{k}_seg.nii.gz"))
            print(
                f"{model}/{k}: frac vessel = {sitk.GetArrayViewFromImage(seg).mean():.5f}",
                flush=True,
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--keys", nargs="+", default=list(MODALITIES))
    p.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4"])
    p.add_argument("--chk", default="checkpoint_final.pth")
    p.add_argument("--no_tta", action="store_true")
    args = p.parse_args()
    for m in args.models:
        run(m, args.keys, not args.no_tta, args.folds, args.chk)
