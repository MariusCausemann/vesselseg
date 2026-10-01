"""Shared paths and resampling helpers.

All resampling goes through the NIfTI affines (never voxel-wise), because the
proteoflow segmentations are stored in a different orientation (LIA) than the
images (RAS).
"""

import functools
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]  # experiments/scripts -> repo root
PF = Path("/global/D1/homes/mariusca/proteoflow")
DATA = ROOT / "data"
PREP = DATA / "prep"
REG = PREP / "reg"  # rigid {key}->T1w transforms from register_to_t1.py
RESULTS = ROOT / "results" / "sub-01"

SEG = PF / "results/sub-01/seg/sub-01_seg.nii.gz"  # FreeSurfer labels, LIA
# Desikan-Killiany + aseg labels extended into the subarachnoid space, RAS
SASPARC = PF / "results/sub-01/seg/sub-01_sasparc.nii.gz"
# brainmesh, RAS; the -uncl variant (mark_unclassified = true) is the only one with label 72
SEG_FIXED = PF / "results/sub-01-uncl/seg/sub-01-uncl_seg_fixed.nii.gz"

# key -> (path, vessels appear dark?)
MODALITIES = {
    "t2": (PF / "data/sub-01_ses-01_T2w.nii.gz", True),
    "t1": (PF / "data/sub-01_ses-01_T1w.nii.gz", False),
    "t1gd": (
        DATA / "sub-01_ses-02_T1w.nii.gz",
        True,
    ),  # ses-02, 4.3 h post intrathecal Gd
    "t1gd3": (DATA / "sub-01_ses-03_T1w.nii.gz", True),  # 23.8 h
    "t1gd4": (DATA / "sub-01_ses-04_T1w.nii.gz", True),  # 47.8 h
    "flair": (DATA / "sub-01_ses-01_FLAIR.nii.gz", True),
}
# T2w already coregistered onto the T1w grid by the proteoflow pipeline
T2_ON_T1 = PF / "data/sub-01_ses-01_T2w_registered.nii.gz"

# FreeSurfer labels
CSF_SAS = [24]
CSF_VENT = [4, 5, 14, 15, 43, 44]
UNCLASSIFIED = 72  # brainmesh label in seg_fixed


def input_key(path):
    """Which input series a result map was predicted from (t1, t2, flair, t1gd)."""
    path = Path(path)
    if "anat2vessels" in path.parts:
        return "t1"  # anat2vessels maps are written back onto the T1w grid
    if "blacktophat" in path.name:
        return "t2"
    return path.name.split("_")[0]


def to_t1(path, ref, order=1, key=None):
    """Resample a map predicted on series `key` onto `ref` (a T1w-space nibabel image).

    Goes through the rigid {key}->T1w registration; for T1w itself this is a plain
    affine resampling.
    """
    import tempfile

    import ants

    key = key or input_key(path)
    if key == "t1":
        return np.asanyarray(resample_like(nib.load(path), ref, order).dataobj)
    with tempfile.TemporaryDirectory() as tmp:
        fref = Path(tmp) / "ref.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros(ref.shape, np.float32), ref.affine), fref)
        out = ants.apply_transforms(
            fixed=ants.image_read(str(fref)),
            moving=ants.image_read(str(path)),
            transformlist=[str(REG / f"{key}_to_t1_0GenericAffine.mat")],
            interpolator="linear" if order else "nearestNeighbor",
        )
        fout = Path(tmp) / "out.nii.gz"
        ants.image_write(out, str(fout))
        return nib.load(fout).get_fdata(dtype=np.float32)


def resample_like(img, target, order):
    """Resample `img` onto the grid of `target` (a nibabel image or (shape, affine))."""
    if isinstance(target, tuple):
        target = nib.Nifti1Image(np.zeros(target[0], np.uint8), target[1])
    return resample_from_to(img, target, order=order, mode="constant", cval=0)


def head_roi(target, margin_mm=8.0):
    """Brain+CSF mask from the segmentation, dilated by `margin_mm`, on `target` grid."""
    mask = resample_like(_dilated_seg(margin_mm), target, 0)
    return np.asanyarray(mask.dataobj) > 0


@functools.cache
def _dilated_seg(margin_mm):
    seg = nib.load(SEG)
    m = np.asanyarray(seg.dataobj) > 0
    dist = ndimage.distance_transform_edt(~m, sampling=seg.header.get_zooms()[:3])
    return nib.Nifti1Image((dist <= margin_mm).astype(np.uint8), seg.affine)


def crop_to(img, mask, margin=4):
    """Crop image (and mask) to the bounding box of `mask`, keeping the affine consistent."""
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin + 1, mask.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    return img.slicer[sl], mask[sl]


def iso_grid(img, spacing):
    """(shape, affine) of an isotropic grid covering the field of view of `img`, same axes."""
    zooms = np.array(img.header.get_zooms()[:3])
    shape = np.round(np.array(img.shape[:3]) * zooms / spacing).astype(int)
    aff = img.affine.copy()
    aff[:3, :3] = img.affine[:3, :3] / zooms * spacing
    # keep the corner of the field of view fixed (voxel centres shift by half a voxel)
    aff[:3, 3] = img.affine[:3, :3] @ ((spacing / zooms - 1) / 2) + img.affine[:3, 3]
    return tuple(shape), aff


def save(arr, affine, path, dtype=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr)
    if dtype is not None:
        arr = arr.astype(dtype)
    img = nib.Nifti1Image(arr, affine)
    img.header.set_xyzt_units("mm")
    nib.save(img, path)
    return path
