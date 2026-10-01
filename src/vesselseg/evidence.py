"""Vessel evidence on the NATIVE T2w grid, for the arterial tree extraction.

The T2w (sub-01: 0.35 x 0.333 x 0.333 mm) is the densest input, and a T1w grid
would throw that resolution away. The internal carotids, however, are largely
invisible in T2w (flow void merges with bone/dura), so the T1w / T1-Gd maps are
needed there. This step brings everything onto the T2w grid (the grid the T2w
probability map was predicted on, RAI after `vesselseg segment`; its affine is
kept exactly, the volume is only cropped to the head ROI):

  segment -> register -> evidence -> extract -> fit-radius -> label

Transforms: `vesselseg register` registers moving=key onto fixed=T1w. The ANTs
.mat maps points of the FIXED space to the MOVING space (LPS mm), so
  x_key = M_key x_t1,   x_t2 = M_t2 x_t1   =>   x_key = M_key M_t2^-1 x_t2.
Images are resampled on the T2 grid by pulling each T2 voxel through this chain
(linear, scipy.ndimage.affine_transform; identical to antsApplyTransforms up to
floating point). The nibabel frames are RAS, ANTs is LPS: M_ras = L M_lps L with
L = diag(-1, -1, 1, 1). The series do not share a world frame (sub-01: the T2w
is ~5 mm off the T1w), which is why t2_to_t1.json is needed for everything that
lives in T1w coordinates (the segmentation, the mesh, proteoflow).

Keys: `t1` is the T1w itself (no transform); every other key KEY needs
REG/KEY_to_t1_0GenericAffine.mat, including `t2`. Keys that differ only by a
trailing session number after `gd` (t1gd, t1gd3, t1gd4) are one contrast: they
are averaged before the consensus over contrasts.

Output (OUT/, all on the same cropped T2w grid):
  MS_t2_prob.nii.gz       T2w probability, native (cropped, float32)
  MS_{key}_prob.nii.gz    other contrasts warped into the T2w grid (linear)
  consensus_other.nii.gz  mean over contrasts of the non-T2 maps
  roi.nii.gz              head ROI: brain+CSF of --seg dilated by 20 mm
  t2_to_t1.json           4x4 world->world (RAS mm), T2w scanner frame -> T1w frame
  evidence.json           grid, crop, inputs

usage: vesselseg evidence --t2_prob riscmi/t2_prob.nii.gz --probs t1=riscmi/t1_prob.nii.gz \\
           t1gd=riscmi/t1gd_prob.nii.gz --reg reg/ --t1 T1w.nii.gz --seg seg.nii.gz -o evidence/
"""

import argparse
import json
import re
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.measure import block_reduce

from vesselseg.io import crop_to, head_roi, iso_grid, resample_like, save
from vesselseg.register import mat_path, parse_pairs

LPS = np.diag([-1.0, -1.0, 1.0, 1.0])  # RAS <-> LPS, its own inverse
ROI_MARGIN = 20.0  # mm around the brain+CSF segmentation
CROP_MARGIN = 4.0  # mm around the ROI bounding box
REF_SPACING = 0.5  # mm, T1w reference grid of the ROI


def contrast(key):
    return re.sub(r"(gd)\d+$", r"\1", key)  # t1gd3 -> t1gd


def reference_grid(t1, seg):
    """Cropped isotropic 0.5 mm T1w-space grid (RAS) around the head, as an empty image.

    The grid the ROI is computed on: the canonical T1w cropped to its 8 mm
    head mask (+4 voxels), resampled to REF_SPACING.
    """
    img = nib.as_closest_canonical(t1)
    img_c, _ = crop_to(img, head_roi(seg, img))
    shape, aff = iso_grid(img_c, REF_SPACING)
    return nib.Nifti1Image(np.zeros(shape, np.uint8), aff)


def coarse_roi(seg, ref, margin_mm, h=2.0):
    """Brain+CSF seg dilated by `margin_mm` on the grid of `ref`, computed on a coarse grid.

    Same as io.head_roi, but the full-resolution EDT there takes ~1 min and
    the margin only needs to be approximate.
    """
    m = np.asanyarray(resample_like(seg, ref, 0).dataobj) > 0
    f = int(round(h / REF_SPACING))
    c = block_reduce(m, (f, f, f), np.max)
    c = ndimage.distance_transform_edt(~c, sampling=f * REF_SPACING) <= margin_mm
    for ax in range(3):
        c = np.repeat(c, f, axis=ax)
    return c[: m.shape[0], : m.shape[1], : m.shape[2]]


def ras_matrix(mat):
    """4x4 RAS affine of an ANTs transform file (maps fixed -> moving points)."""
    import ants

    tx = ants.read_transform(str(mat))
    lps = np.eye(4)
    lps[:3, 3] = tx.apply_to_point((0.0, 0.0, 0.0))
    for i in range(3):
        e = np.zeros(3)
        e[i] = 1.0
        lps[:3, i] = np.array(tx.apply_to_point(tuple(e))) - lps[:3, 3]
    return LPS @ lps @ LPS


def apply(m, pts):
    """Apply a 4x4 to (n, 3) points."""
    return pts @ m[:3, :3].T + m[:3, 3]


def voxel_grid_map(src_aff, dst_aff, world_map):
    """(matrix, offset) for ndimage.affine_transform: dst voxel -> src voxel.

    world_map takes dst world (RAS) to src world (RAS).
    """
    m = np.linalg.inv(src_aff) @ world_map @ dst_aff
    return m[:3, :3], m[:3, 3]


def warp(data, src_aff, dst_shape, dst_aff, world_map, order=1):
    mat, off = voxel_grid_map(src_aff, dst_aff, world_map)
    return ndimage.affine_transform(
        data, mat, off, output_shape=dst_shape, order=order, mode="constant", cval=0
    )


def crop_box(t2, m_t2_to_t1, roi, ref, margin_mm):
    """Voxel bounding box on the T2w grid of the head ROI (on the T1w grid `ref`)."""
    idx = np.argwhere(roi[::2, ::2, ::2]) * 2
    world_t1 = apply(ref.affine, idx.astype(float))
    world_t2 = apply(np.linalg.inv(m_t2_to_t1), world_t1)
    vox = apply(np.linalg.inv(t2.affine), world_t2)
    pad = margin_mm / np.linalg.norm(t2.affine[:3, :3], axis=0)
    lo = np.maximum(np.floor(vox.min(0) - pad), 0).astype(int)
    hi = np.minimum(np.ceil(vox.max(0) + pad) + 1, t2.shape).astype(int)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def build_evidence(t2_prob, probs, reg, t1, seg, out):
    """Write the evidence directory `out` (see the module docstring).

    t2_prob: T2w probability map; probs: {key: path} of the other maps, each on
    its own series grid; reg: directory of the {key}_to_t1 transforms; t1, seg:
    the T1w image and its brain segmentation (paths).
    """
    t0 = time.time()
    out = Path(out)
    t2_img = nib.load(t2_prob)
    print("T2w grid:", t2_img.shape, nib.aff2axcodes(t2_img.affine), flush=True)
    m_t2 = ras_matrix(mat_path(reg, "t2"))  # T1 world -> T2 world
    m_t2_to_t1 = np.linalg.inv(m_t2)  # T2 world -> T1 world

    seg_img = nib.load(seg)
    ref = reference_grid(nib.load(t1), seg_img)
    roi_t1 = coarse_roi(seg_img, ref, ROI_MARGIN)
    sl = crop_box(t2_img, m_t2_to_t1, roi_t1, ref, CROP_MARGIN)
    lo = np.array([s.start for s in sl])
    shape = tuple(s.stop - s.start for s in sl)
    aff = t2_img.affine.copy()
    aff[:3, 3] = t2_img.affine[:3, :3] @ lo + t2_img.affine[:3, 3]
    print("crop", [(s.start, s.stop) for s in sl], shape, flush=True)

    t2 = np.asarray(t2_img.dataobj[sl], np.float32)
    save(t2, aff, out / "MS_t2_prob.nii.gz", np.float32)

    maps, inputs = {}, {"t2": str(t2_prob)}
    for k, src in probs.items():
        if k == "t2":
            raise ValueError("the T2w map goes in t2_prob, not in probs")
        img = nib.load(src)
        m_k = ras_matrix(mat_path(reg, k)) if k != "t1" else np.eye(4)
        # T2 world -> T1 world -> key world
        world_map = m_k @ m_t2_to_t1
        print(f"warping {k} to the T2w grid", flush=True)
        maps[k] = warp(img.get_fdata(dtype=np.float32), img.affine, shape, aff, world_map).astype(
            np.float32
        )
        save(maps[k], aff, out / f"MS_{k}_prob.nii.gz", np.float32)
        inputs[k] = str(src)
    if not maps:
        raise ValueError("no non-T2 probability maps: the consensus needs at least one")

    groups = {}
    for k, a in maps.items():
        groups.setdefault(contrast(k), []).append(a)
    print("contrasts:", {g: len(v) for g, v in groups.items()})
    cons = np.mean([np.mean(v, axis=0) for v in groups.values()], axis=0)
    save(cons, aff, out / "consensus_other.nii.gz", np.float32)

    # head ROI: T2 voxel -> T1 voxel, nearest neighbour
    roi = warp(roi_t1.astype(np.uint8), ref.affine, shape, aff, m_t2_to_t1, order=0).astype(
        np.uint8
    )
    save(roi, aff, out / "roi.nii.gz", np.uint8)

    (out / "t2_to_t1.json").write_text(
        json.dumps(
            dict(
                matrix=m_t2_to_t1.tolist(),
                convention="RAS mm (nibabel), T2w scanner frame -> T1w scanner frame",
                source=str(mat_path(reg, "t2")),
            ),
            indent=2,
        )
    )

    stats = dict(
        inputs=inputs,
        t1=str(t1),
        seg=str(seg),
        consensus_contrasts={g: len(v) for g, v in groups.items()},
        shape=list(shape),
        spacing_mm=np.linalg.norm(aff[:3, :3], axis=0).tolist(),
        crop_offset_voxels=lo.tolist(),
        full_shape=list(t2_img.shape),
        affine=aff.tolist(),
        axcodes="".join(nib.aff2axcodes(aff)),
        roi_margin_mm=ROI_MARGIN,
        crop_margin_mm=CROP_MARGIN,
        roi_fraction=float(roi.mean()),
    )
    stats["runtime_s"] = time.time() - t0
    (out / "evidence.json").write_text(json.dumps(stats, indent=2))
    print(f"done in {stats['runtime_s']:.0f} s")
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="vesselseg evidence",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--t2_prob", required=True, type=Path, help="T2w probability map")
    p.add_argument(
        "--probs",
        required=True,
        nargs="+",
        help="KEY=PATH of the other maps (t1, t1gd, ...)",
    )
    p.add_argument("--reg", required=True, type=Path, help="`vesselseg register` output")
    p.add_argument("--t1", required=True, type=Path, help="T1w image (the fixed series)")
    p.add_argument("--seg", required=True, type=Path, help="brain segmentation, T1w space")
    p.add_argument("-o", "--out", required=True, type=Path, help="output directory")
    args = p.parse_args(argv)
    build_evidence(args.t2_prob, parse_pairs(args.probs), args.reg, args.t1, args.seg, args.out)


if __name__ == "__main__":
    main()
