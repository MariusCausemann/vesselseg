"""Ground-truth-free plausibility metrics for every probability map in results/sub-01.

The input series do not share a world frame (head motion between scans, and
t1gd is another session), so every map is first brought into T1w space with the
rigid transforms from register_to_t1.py, onto one common 0.5 mm grid (the
prepared T1w). The FreeSurfer seg and brainmesh seg_fixed are T1w-space and
are resampled onto the same grid through their affines.

Per run (threshold --thr):
  vol_ml, n_cc, largest_cc      volume, #26-connected components, largest-component fraction
  in_sas / in_vent / in_par /   fraction of the mask in SAS CSF (dilated 1 mm),
  outside_seg                   ventricles, parenchyma, unlabelled
  basal                         fraction in the lowest 35 % of the brain (CoW level and below)
  in_uncl / uncl_cov            fraction in brainmesh UNCLASSIFIED(72), and fraction of 72 covered
  skel_mm, r_med_mm             skeleton length and median radius (EDT at skeleton) of
                                components >= --min_cc voxels
Writes metrics.csv, dice.csv and an empty checklist.csv to fill in by eye.
"""

import argparse
import itertools

import nibabel as nib
import numpy as np
import pandas as pd
from common import (
    CSF_SAS,
    CSF_VENT,
    PREP,
    RESULTS,
    SEG,
    SEG_FIXED,
    UNCLASSIFIED,
    resample_like,
    to_t1,
)
from scipy import ndimage
from skimage.morphology import skeletonize

REF = PREP / "vesselfm" / "t1_0.50mm.nii.gz"  # T1w space
H = 0.5  # mm, reference spacing
SEGMENTS = ["ICA_L", "ICA_R", "BA", "VA_L", "VA_R", "M1_L", "M1_R", "M2_L", "M2_R"]
SEGMENTS += ["A1_L", "A1_R", "A2", "P1_L", "P1_R", "P2_L", "P2_R", "Pcom_L", "Pcom_R"]


def run_name(p):
    rel = p.relative_to(RESULTS)
    return str(rel).replace("_prob.nii.gz", "")


def metrics(mask, seg, uncl, basal, min_cc):
    n = mask.sum()
    lab, n_cc = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel())[1:] if n_cc else np.array([0])
    sas = ndimage.binary_dilation(np.isin(seg, CSF_SAS), iterations=2)
    vent = np.isin(seg, CSF_VENT)
    big = np.isin(lab, np.nonzero(sizes >= min_cc)[0] + 1)
    skel = skeletonize(big) > 0
    r = ndimage.distance_transform_edt(big, sampling=H)[skel]
    f = (lambda m: float((mask & m).sum() / n)) if n else (lambda m: np.nan)
    return {
        "vol_ml": n * H**3 / 1000,
        "n_cc": n_cc,
        "largest_cc": float(sizes.max() / n) if n else np.nan,
        "in_sas": f(sas & ~vent),
        "in_vent": f(vent),
        "in_par": f((seg > 0) & ~sas & ~vent),
        "outside_seg": f((seg == 0) & ~sas),
        "basal": f(basal),
        "in_uncl": f(uncl),
        "uncl_cov": float((mask & uncl).sum() / max(uncl.sum(), 1)),
        "skel_mm": skel.sum() * H,
        "r_med_mm": float(np.median(r)) if r.size else np.nan,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--min_cc", type=int, default=100)
    p.add_argument("--runs", nargs="+", help="only these runs, e.g. riscmi/MS/t1")
    p.add_argument("--tag", default="", help="suffix for the output csv names")
    args = p.parse_args()

    ref = nib.load(REF)
    seg = np.asanyarray(resample_like(nib.load(SEG), ref, 0).dataobj)
    uncl = (
        np.asanyarray(resample_like(nib.load(SEG_FIXED), ref, 0).dataobj)
        == UNCLASSIFIED
    )
    z = np.nonzero((seg > 0).any((0, 1)))[0]
    basal = np.zeros(seg.shape, bool)
    basal[:, :, : int(z.min() + 0.35 * (z.max() - z.min()))] = True

    # *_refspace_* maps live in the anat2vessels reference frame, not scanner space
    paths = sorted(
        p for p in RESULTS.rglob("*_prob.nii.gz") if "_refspace" not in p.name
    )
    if args.runs:
        paths = [p for p in paths if run_name(p) in args.runs]
    rows, masks = [], {}
    for path in paths:
        prob = to_t1(path, ref)
        m = prob > args.thr
        name = run_name(path)
        masks[name] = m
        rows.append(dict(run=name, **metrics(m, seg, uncl, basal, args.min_cc)))
        print(rows[-1], flush=True)

    out = RESULTS / "eval"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(out / f"metrics{args.tag}.csv", index=False)

    names = list(masks)
    dice = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for a, b in itertools.combinations(names, 2):
        s = masks[a].sum() + masks[b].sum()
        dice.loc[a, b] = dice.loc[b, a] = (
            2 * (masks[a] & masks[b]).sum() / s if s else np.nan
        )
    dice.round(3).to_csv(out / f"dice{args.tag}.csv")

    chk = out / "checklist.csv"
    if not chk.exists() and not args.runs:  # never overwrite hand-filled entries
        pd.DataFrame(index=names, columns=SEGMENTS).to_csv(chk, index_label="run")


if __name__ == "__main__":
    main()
