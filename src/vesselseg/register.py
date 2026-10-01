"""Rigid ANTs registration of every non-T1 series onto the T1w.

The series need NOT share a world frame: the head moves between acquisitions
(sub-01: T2w/FLAIR ~5 mm, a later session's T1w ~13-16 mm). Probability maps
are brought into T1w space with these transforms before they are combined
(`vesselseg evidence`).

Output: OUT/{key}_to_t1_0GenericAffine.mat for each --moving KEY=PATH (ANTs
convention: it maps points of the fixed T1w space to the moving space, LPS mm),
and OUT/{key}_to_t1.json with its rotation and shift.

usage: vesselseg register --fixed T1w.nii.gz --moving t2=T2w.nii.gz t1gd=ses-02_T1w.nii.gz -o reg/
"""

import argparse
import json
from pathlib import Path

import numpy as np


def mat_path(reg_dir, key):
    return Path(reg_dir) / f"{key}_to_t1_0GenericAffine.mat"


def offsets(mat):
    """Rotation (deg), shift at the centre and largest shift 6 cm out (mm) of a rigid .mat."""
    import ants

    tx = ants.read_transform(str(mat))
    rot = np.array(tx.parameters[:9]).reshape(3, 3)
    ang = np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1)))
    c = np.array(tx.fixed_parameters)
    pts = c + 60 * np.vstack([np.eye(3), -np.eye(3)])
    d = [np.linalg.norm(np.array(tx.apply_to_point(q)) - q) for q in pts]
    return ang, np.linalg.norm(np.array(tx.apply_to_point(c)) - c), max(d)


def register(fixed, moving, out):
    """Register each series in `moving` ({key: path}) rigidly onto `fixed` (the T1w)."""
    import ants

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t1 = ants.image_read(str(fixed))
    for k, path in moving.items():
        if k == "t1":
            raise ValueError("key 't1' is reserved for the fixed T1w")
        ants.registration(
            fixed=t1,
            moving=ants.image_read(str(path)),
            type_of_transform="Rigid",
            outprefix=str(out / f"{k}_to_t1_"),
        )
        ang, c, m = offsets(mat_path(out, k))
        print(f"{k}: rotation {ang:.2f} deg, shift at centre {c:.2f} mm, max at 6 cm {m:.2f} mm")
        stats = dict(
            fixed=str(fixed),
            moving=str(path),
            rotation_deg=round(float(ang), 3),
            shift_centre_mm=round(float(c), 3),
            shift_max_6cm_mm=round(float(m), 3),
        )
        (out / f"{k}_to_t1.json").write_text(json.dumps(stats, indent=2))
    return {k: mat_path(out, k) for k in moving}


def parse_pairs(items):
    """['k=path', ...] -> {k: Path}"""
    out = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep or not k or not v:
            raise argparse.ArgumentTypeError(f"expected KEY=PATH, got {it!r}")
        out[k] = Path(v)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="vesselseg register",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--fixed", required=True, type=Path, help="T1w image")
    p.add_argument("--moving", required=True, nargs="+", help="KEY=PATH per series")
    p.add_argument("-o", "--out", required=True, type=Path, help="output directory")
    args = p.parse_args(argv)
    register(args.fixed, parse_pairs(args.moving), args.out)


if __name__ == "__main__":
    main()
