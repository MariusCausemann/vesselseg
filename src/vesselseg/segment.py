"""risc-mi MS artery probability map of one structural series, via the brainseg container.

risc-mi MS (Alshenoudy et al., MIUA 2024; RISC Software GmbH) is an nnU-Net v2
model (Dataset606_MRIVessels, nnUNetTrainerDA5, 3d_fullres, 5 folds) trained on
T1w / T2w / PD / FLAIR with TOF-derived labels. It runs inside brainseg's
`brainseg_riscmi.sif` (weights included; see the README for building and
copying it to an offline machine), found by brainseg's container lookup:
--container, ./, ./.containers/, $BRAINSEG_CONTAINER_DIR, ~/.brainseg_containers/.
It is never pulled from the registry from here.

The input is first reoriented to RAI and cast to float32 (the orientation the
model was trained in), so the probability map lives on that RAI grid: for the
T2w, this is the grid `vesselseg evidence` and the tree extraction work on.

Output: OUT (probability, float32) and OUT_mask (0/1, with --mask).

usage: vesselseg segment -i T2w.nii.gz -o riscmi/t2_prob.nii.gz [--gpu] [--no_tta]
"""

import argparse
import tempfile
from pathlib import Path

from vesselseg.io import nifti_stem

TOOL = "riscmi_arteries"


def find_riscmi_container(container=None):
    """Path of brainseg_riscmi.sif; FileNotFoundError if it is not on this machine."""
    if container is not None:
        container = Path(container)
        if not container.exists():
            raise FileNotFoundError(container)
        return container.resolve()
    from brainseg.utils import find_container

    return find_container(TOOL, build=False)


def to_rai(src, dst):
    """Reorient `src` to RAI, float32 (what risc-mi was trained on)."""
    import SimpleITK as sitk

    im = sitk.DICOMOrient(sitk.ReadImage(str(src)), "RAI")
    sitk.WriteImage(sitk.Cast(im, sitk.sitkFloat32), str(dst))
    return dst


def segment(
    image,
    out,
    *,
    container=None,
    gpu=False,
    tta=True,
    folds="0 1 2 3 4",
    workers=None,
    keep_mask=False,
):
    """Write the risc-mi MS probability map of `image` to `out` (RAI grid)."""
    from brainseg.tools import run_riscmi_arteries

    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    sif = find_riscmi_container(container)
    stem = nifti_stem(out)
    mask = out.with_name(f"{stem}_mask.nii.gz")
    # the input is bound read-only into the container from its own directory
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=f".{stem}_") as tmp:
        rai = to_rai(image, Path(tmp) / f"{stem}_rai.nii.gz")
        run_riscmi_arteries(
            rai,
            mask,
            sif,
            folds=folds,
            tta=tta,
            save_prob=True,
            prob_path=out,
            gpu=gpu,
            workers=workers,
        )
    if not out.exists():
        raise RuntimeError(f"risc-mi wrote no probability map {out}")
    if not keep_mask:
        mask.unlink(missing_ok=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="vesselseg segment",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-i", "--input", required=True, type=Path, help="structural MRI series")
    p.add_argument("-o", "--out", required=True, type=Path, help="probability map (.nii.gz)")
    p.add_argument("--container", type=Path, help="brainseg_riscmi.sif (default: lookup)")
    p.add_argument("--gpu", action="store_true", help="NVIDIA GPU (apptainer --nv)")
    p.add_argument("--no_tta", action="store_true", help="no mirroring TTA (8x faster)")
    p.add_argument("--folds", default="0 1 2 3 4", help="folds to ensemble")
    p.add_argument("--workers", type=int, help="CPU worker processes (default: auto)")
    p.add_argument("--mask", action="store_true", help="also keep the 0/1 mask")
    args = p.parse_args(argv)
    segment(
        args.input,
        args.out,
        container=args.container,
        gpu=args.gpu,
        tta=not args.no_tta,
        folds=args.folds,
        workers=args.workers,
        keep_mask=args.mask,
    )


if __name__ == "__main__":
    main()
