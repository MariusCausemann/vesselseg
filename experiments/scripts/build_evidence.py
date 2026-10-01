"""Vessel evidence on the NATIVE T2w grid, for the T2-space arterial tree extraction.

build_network.py works on the 0.5 mm T1w grid, but the T2w (0.35 x 0.333 x 0.333 mm)
is the densest input and the T1w grid throws that resolution away. The internal
carotids, however, are largely invisible in T2w (flow void merges with bone/dura),
so the T1w / T1-Gd maps are needed there. This script brings everything onto the
T2w grid (the RAI-reoriented native T2w that risc-mi predicted on; its affine is
kept exactly, the volume is only cropped to the head ROI):

  prep_inputs -> run_riscmi -> register_to_t1 -> build_evidence
      -> extract_tree (T2 space) -> fit_radius -> label_centerlines

Transforms: register_to_t1.py registers moving=key onto fixed=T1w. The ANTs .mat
maps points of the FIXED space to the MOVING space (LPS mm), so
  x_key = M_key x_t1,   x_t2 = M_t2 x_t1   =>   x_key = M_key M_t2^-1 x_t2.
Images are resampled on the T2 grid by pulling each T2 voxel through this chain
(linear, scipy.ndimage.affine_transform; identical to antsApplyTransforms up to
floating point). The nibabel frames are RAS, ANTs is LPS: M_ras = L M_lps L with
L = diag(-1, -1, 1, 1). The series do not share a world frame (the T2w is ~5 mm
off the T1w), which is why t2_to_t1.json is needed for everything that lives in
T1w coordinates (mesh, proteoflow, earlier centerlines).

Output (results/sub-01/network/t2space/, all on the same cropped T2w grid):
  MS_t2_prob.nii.gz     T2w probability, native (cropped, float32)
  MS_{key}_prob.nii.gz  other contrasts warped into the T2w grid (linear)
  consensus_other.nii.gz  build_network's consensus over the non-T2 contrasts
                        (post-Gd sessions averaged into one contrast first)
  roi.nii.gz            build_network's head ROI (nearest neighbour)
  t2_to_t1.json         4x4 world->world (RAS mm), T2w scanner frame -> T1w frame
  evidence.json         grid, crop, inputs, verification numbers

Usage: python build_evidence.py [--model MS] [--keys t1 t1gd t1gd3 t1gd4] [--no_verify]
"""

import argparse
import json
import time

import ants
import nibabel as nib
import numpy as np
import pyvista as pv
from build_network import REF, contrast, coarse_roi
from common import REG, RESULTS, save
from scipy import ndimage

OUT = RESULTS / "network" / "t2space"
LPS = np.diag([-1.0, -1.0, 1.0, 1.0])  # RAS <-> LPS, its own inverse
ROI_MARGIN = 20.0  # mm, as build_network
CROP_MARGIN = 4.0  # mm around the ROI bounding box


def ras_matrix(mat):
    """4x4 RAS affine of an ANTs transform file (maps fixed -> moving points)."""
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


def crop_box(t2, m_t2_to_t1, margin_mm):
    """Voxel bounding box on the T2w grid of the build_network head ROI (T1w grid)."""
    ref = nib.load(REF)
    roi = coarse_roi(ref, ROI_MARGIN)
    idx = np.argwhere(roi[::2, ::2, ::2]) * 2
    world_t1 = apply(ref.affine, idx.astype(float))
    world_t2 = apply(np.linalg.inv(m_t2_to_t1), world_t1)
    vox = apply(np.linalg.inv(t2.affine), world_t2)
    pad = margin_mm / np.linalg.norm(t2.affine[:3, :3], axis=0)
    lo = np.maximum(np.floor(vox.min(0) - pad), 0).astype(int)
    hi = np.minimum(np.ceil(vox.max(0) + pad) + 1, t2.shape).astype(int)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi)), roi, ref


def line_lengths(mesh):
    """Per-edge (point a, point b, length) of the polyline cells."""
    lines, edges = mesh.lines, []
    i = 0
    while i < len(lines):
        n = lines[i]
        ids = lines[i + 1 : i + 1 + n]
        edges += list(zip(ids[:-1], ids[1:]))
        i += n + 1
    e = np.array(edges)
    return e, np.linalg.norm(mesh.points[e[:, 0]] - mesh.points[e[:, 1]], axis=1)


def sample(vol, aff, world, order=1):
    vox = apply(np.linalg.inv(aff), world)
    return ndimage.map_coordinates(
        vol, vox.T, order=order, mode="constant", cval=0, prefilter=False
    )


def verify(out, m_t2_to_t1, t2, aff, maps, cons, ref, tree):
    res = {}
    t1space = RESULTS / "network" / "t1space"
    # 1. point transform: confident T2 voxels into the T1-space T2 map
    t1_t2 = nib.load(t1space / "MS_t2_prob.nii.gz")
    t1_t2d = t1_t2.get_fdata(dtype=np.float32)
    idx = np.argwhere(t2 > 0.5)
    idx = idx[:: max(len(idx) // 500000, 1)]
    w2 = apply(aff, idx.astype(float))
    mapped = sample(t1_t2d, t1_t2.affine, apply(m_t2_to_t1, w2))
    raw = sample(t1_t2d, t1_t2.affine, w2)
    res["point_transform"] = dict(
        n=len(idx),
        mean_T1space_T2_at_mapped=float(mapped.mean()),
        mean_T1space_T2_at_untransformed=float(raw.mean()),
        frac_above_0p5_mapped=float((mapped > 0.5).mean()),
        frac_above_0p5_untransformed=float((raw > 0.5).mean()),
    )
    # 2. image warp: T2-grid t1gd back to the T1 grid vs build_network's map
    key = "t1gd" if "t1gd" in maps else next(iter(maps))
    ref_map = nib.load(t1space / f"MS_{key}_prob.nii.gz").get_fdata(dtype=np.float32)
    ii = np.indices(ref.shape, dtype=np.float32).reshape(3, -1).T
    back = sample(
        maps[key], aff, apply(np.linalg.inv(m_t2_to_t1), apply(ref.affine, ii))
    )
    back = back.reshape(ref.shape)
    use = (back > 0) | (ref_map > 0)  # both maps are 0 outside the common FOV
    both = (back > 0) & (ref_map > 0)
    res["image_warp"] = dict(
        key=key,
        pearson_r=float(np.corrcoef(back[both], ref_map[both])[0, 1]),
        pearson_r_union=float(np.corrcoef(back[use], ref_map[use])[0, 1]),
        dice_0p5=float(
            2
            * ((back > 0.5) & (ref_map > 0.5)).sum()
            / max((back > 0.5).sum() + (ref_map > 0.5).sum(), 1)
        ),
        n_vox=int(both.sum()),
    )
    # 3. ICA coverage along the current tree (T1w world)
    mesh = pv.read(tree)
    seg = np.asarray(mesh.point_data["segment"])
    names = list(mesh.field_data["segment_names"])
    w = apply(np.linalg.inv(m_t2_to_t1), mesh.points)
    mx = lambda a: ndimage.maximum_filter(a, size=3)  # noqa: E731
    p_t2 = sample(mx(t2), aff, w, 0)
    p_ot = sample(mx(cons), aff, w, 0)
    e, ln = line_lengths(mesh)
    ica = {}
    for nm in ("ICA-L", "ICA-R"):
        s = names.index(nm)
        sel = (seg[e[:, 0]] == s) & (seg[e[:, 1]] == s)
        both_lo = (p_t2[e[:, 0]] < 0.005) & (p_t2[e[:, 1]] < 0.005)
        both_ot = (p_ot[e[:, 0]] > 0.3) & (p_ot[e[:, 1]] > 0.3)
        either = both_lo & ~both_ot
        ica[nm] = dict(
            length_mm=float(ln[sel].sum()),
            t2_below_0p005_mm=float(ln[sel & both_lo].sum()),
            other_above_0p3_mm=float(ln[sel & both_ot].sum()),
            t2_blind_and_other_below_0p3_mm=float(ln[sel & either].sum()),
        )
    res["ica"] = ica
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="MS", choices=["MS", "MC"])
    p.add_argument(
        "--keys",
        nargs="+",
        default=["t1", "t1gd", "t1gd3", "t1gd4"],
        help="non-T2 inputs (build_network's list; add flair to also warp it)",
    )
    p.add_argument("--no_verify", action="store_true")
    p.add_argument(
        "--tree",
        default=str(RESULTS / "network" / "centerlines_tree_posterior_t2_labeled.vtp"),
    )
    args = p.parse_args()
    t0 = time.time()

    t2_img = nib.load(RESULTS / "riscmi" / args.model / "t2_prob.nii.gz")
    print("T2w grid:", t2_img.shape, nib.aff2axcodes(t2_img.affine), flush=True)
    m_t2 = ras_matrix(REG / "t2_to_t1_0GenericAffine.mat")  # T1 world -> T2 world
    m_t2_to_t1 = np.linalg.inv(m_t2)  # T2 world -> T1 world

    sl, roi_t1, ref = crop_box(t2_img, m_t2_to_t1, CROP_MARGIN)
    lo = np.array([s.start for s in sl])
    shape = tuple(s.stop - s.start for s in sl)
    aff = t2_img.affine.copy()
    aff[:3, 3] = t2_img.affine[:3, :3] @ lo + t2_img.affine[:3, 3]
    print("crop", [(s.start, s.stop) for s in sl], shape, flush=True)

    t2 = np.asarray(t2_img.dataobj[sl], np.float32)
    save(t2, aff, OUT / f"{args.model}_t2_prob.nii.gz", np.float32)

    maps, inputs = {}, ["t2"]
    for k in args.keys:
        src = RESULTS / "riscmi" / args.model / f"{k}_prob.nii.gz"
        mat = REG / f"{k}_to_t1_0GenericAffine.mat"
        if not src.exists() or (k != "t1" and not mat.exists()):
            print(f"skip {k}: missing prob map or registration")
            continue
        img = nib.load(src)
        m_k = ras_matrix(mat) if k != "t1" else np.eye(4)
        # T2 world -> T1 world -> key world
        world_map = m_k @ m_t2_to_t1
        print(f"warping {k} to the T2w grid", flush=True)
        maps[k] = warp(
            img.get_fdata(dtype=np.float32), img.affine, shape, aff, world_map
        ).astype(np.float32)
        save(maps[k], aff, OUT / f"{args.model}_{k}_prob.nii.gz", np.float32)
        inputs.append(k)

    groups = {}
    for k, a in maps.items():
        groups.setdefault(contrast(k), []).append(a)
    print("contrasts:", {g: len(v) for g, v in groups.items()})
    cons = np.mean([np.mean(v, axis=0) for v in groups.values()], axis=0)
    save(cons, aff, OUT / "consensus_other.nii.gz", np.float32)

    # head ROI: T2 voxel -> T1 voxel, nearest neighbour
    roi = warp(
        roi_t1.astype(np.uint8), ref.affine, shape, aff, m_t2_to_t1, order=0
    ).astype(np.uint8)
    save(roi, aff, OUT / "roi.nii.gz", np.uint8)

    (OUT / "t2_to_t1.json").write_text(
        json.dumps(
            dict(
                matrix=m_t2_to_t1.tolist(),
                convention="RAS mm (nibabel), T2w scanner frame -> T1w scanner frame",
                source=str(REG / "t2_to_t1_0GenericAffine.mat"),
            ),
            indent=2,
        )
    )

    stats = dict(
        model=args.model,
        inputs=inputs,
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
    if not args.no_verify:
        stats["verification"] = verify(
            OUT, m_t2_to_t1, t2, aff, maps, cons, ref, args.tree
        )
    stats["runtime_s"] = time.time() - t0
    (OUT / "evidence.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats.get("verification", {}), indent=2))
    print(f"done in {stats['runtime_s']:.0f} s")


if __name__ == "__main__":
    main()
