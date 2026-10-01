"""Prepare sub-01 inputs for vesselFM (cropped, isotropic, + head mask) and risc-mi (RAI).

Outputs
  data/prep/vesselfm/{key}_{scale}mm.nii.gz        cropped + isotropic image
  data/prep/vesselfm/{key}_{scale}mm_mask.nii.gz   dilated brain+CSF mask on that grid
  data/prep/riscmi/{key}/{key}_0000.nii.gz         full native image in RAI orientation
  data/prep/check/{key}.png                        overlay of image and resampled mask
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from common import MODALITIES, PREP, crop_to, head_roi, iso_grid, resample_like, save

SCALES = [0.33, 0.5, 0.75]


def prep_vesselfm(key, path):
    img = nib.as_closest_canonical(nib.load(path))
    mask = head_roi(img)
    img_c, mask_c = crop_to(img, mask)
    native = min(img.header.get_zooms()[:3])
    for s in SCALES:
        if s < native - 0.02:  # no upsampling beyond native resolution
            continue
        grid = iso_grid(img_c, s)
        im = resample_like(img_c, grid, order=1)
        mk = resample_like(
            nib.Nifti1Image(mask_c.astype(np.uint8), img_c.affine), grid, 0
        )
        tag = f"{key}_{s:.2f}mm"
        save(
            np.asanyarray(im.dataobj),
            im.affine,
            PREP / "vesselfm" / f"{tag}.nii.gz",
            np.float32,
        )
        save(
            np.asanyarray(mk.dataobj),
            mk.affine,
            PREP / "vesselfm" / f"{tag}_mask.nii.gz",
            np.uint8,
        )
        print(tag, im.shape)
    check_png(key, img, mask)


def prep_riscmi(key, path):
    im = sitk.DICOMOrient(sitk.ReadImage(str(path)), "RAI")
    out = PREP / "riscmi" / key / f"{key}_0000.nii.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.Cast(im, sitk.sitkFloat32), str(out))
    print(out.name, im.GetSize(), im.GetSpacing())


def check_png(key, img, mask):
    a = np.asanyarray(img.dataobj)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        sl = [slice(None)] * 3
        sl[i] = a.shape[i] // 2
        ax[i].imshow(np.rot90(a[tuple(sl)]), cmap="gray")
        ax[i].contour(
            np.rot90(mask[tuple(sl)]), levels=[0.5], colors="r", linewidths=0.6
        )
        ax[i].axis("off")
    fig.suptitle(f"{key}: image + resampled head ROI")
    out = PREP / "check" / f"{key}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--keys", nargs="+", default=list(MODALITIES))
    args = p.parse_args()
    for k in args.keys:
        path, _ = MODALITIES[k]
        prep_vesselfm(k, path)
        prep_riscmi(k, path)
