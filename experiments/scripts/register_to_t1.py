"""Rigid ANTs registration of every non-T1 series onto the ses-01 T1w.

The series do NOT share a world frame: the head moves between acquisitions
(T2w/FLAIR ~5 mm, ses-02 T1w ~13-16 mm). Probability maps are brought into T1w
space with these transforms before any comparison with each other or with the
(T1w-space) segmentation.

Output: data/prep/reg/{key}_to_t1_0GenericAffine.mat (+ printed offsets)
"""

import argparse

import ants
import numpy as np
from common import MODALITIES, REG


def offsets(mat):
    tx = ants.read_transform(str(mat))
    rot = np.array(tx.parameters[:9]).reshape(3, 3)
    ang = np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1)))
    c = np.array(tx.fixed_parameters)
    pts = c + 60 * np.vstack([np.eye(3), -np.eye(3)])
    d = [np.linalg.norm(np.array(tx.apply_to_point(q)) - q) for q in pts]
    return ang, np.linalg.norm(np.array(tx.apply_to_point(c)) - c), max(d)


def main(keys):
    REG.mkdir(parents=True, exist_ok=True)
    t1 = ants.image_read(str(MODALITIES["t1"][0]))
    for k, (path, _) in MODALITIES.items():
        if k == "t1" or k not in keys:
            continue
        mov = ants.image_read(str(path))
        ants.registration(
            fixed=t1,
            moving=mov,
            type_of_transform="Rigid",
            outprefix=str(REG / f"{k}_to_t1_"),
        )
        ang, c, m = offsets(REG / f"{k}_to_t1_0GenericAffine.mat")
        print(
            f"{k}: rotation {ang:.2f} deg, shift at centre {c:.2f} mm, max at 6 cm {m:.2f} mm"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--keys", nargs="+", default=list(MODALITIES))
    main(p.parse_args().keys)
