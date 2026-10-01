"""Shared image and polyline helpers.

All resampling goes through the NIfTI affines (never voxel-wise), because a
FreeSurfer-style segmentation is typically stored in a different orientation
(LIA) than the images (RAS).
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage


def resample_like(img, target, order):
    """Resample `img` onto the grid of `target` (a nibabel image or (shape, affine))."""
    if isinstance(target, tuple):
        target = nib.Nifti1Image(np.zeros(target[0], np.uint8), target[1])
    return resample_from_to(img, target, order=order, mode="constant", cval=0)


def dilated_seg(seg, margin_mm):
    """Brain+CSF mask of the segmentation `seg`, dilated by `margin_mm`, on its grid."""
    m = np.asanyarray(seg.dataobj) > 0
    dist = ndimage.distance_transform_edt(~m, sampling=seg.header.get_zooms()[:3])
    return nib.Nifti1Image((dist <= margin_mm).astype(np.uint8), seg.affine)


def head_roi(seg, target, margin_mm=8.0):
    """Brain+CSF mask from the segmentation, dilated by `margin_mm`, on `target` grid."""
    mask = resample_like(dilated_seg(seg, margin_mm), target, 0)
    return np.asanyarray(mask.dataobj) > 0


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


def nifti_stem(path):
    """File name without .nii / .nii.gz."""
    name = Path(path).name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def midplane(img):
    """Centre and right-pointing unit normal of the mid-sagittal plane.

    `img` is a parcellation with Desikan-Killiany cortex labels (1000s left,
    2000s right), e.g. the sasparc; the plane lies between the two centroids.
    """
    lab = np.asarray(img.dataobj)
    cl = nib.affines.apply_affine(img.affine, np.argwhere((lab >= 1000) & (lab < 2000)))
    cr = nib.affines.apply_affine(img.affine, np.argwhere(lab >= 2000))
    cl, cr = cl.mean(0), cr.mean(0)
    return (cl + cr) / 2, (cr - cl) / np.linalg.norm(cr - cl)


def polylines(net):
    """Point ids of each polyline cell of a pyvista PolyData."""
    a, out, i = net.lines, [], 0
    while i < len(a):
        out.append(a[i + 1 : i + 1 + a[i]])
        i += a[i] + 1
    return out
