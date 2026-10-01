"""Vessel diameter from the T2w flow void (FWHM across the centerline) vs the tree radius.

Per labelled branch, every 3rd point: six profiles across the centerline (T2w,
+-5 mm), the void FWHM between the minimum within 0.75 mm and the background
(90th percentile beyond 3 mm, each side; no contrast: skipped), and the width
of P above half its central maximum. Medians [IQR] over the cross-sections, against
the raw and fitted tree diameter 2r. The ICA trunks are mostly invisible in T2w.

usage: pixi run python profile_t2w.py NAME LABEL...   (NAME_labeled.vtp)
"""

import collections
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
from common import ROOT
from scipy import ndimage

V = str(ROOT) + "/"
ev = json.loads(Path(V + "results/sub-01/network/t2space/evidence.json").read_text())
t2 = nib.load(V + "data/prep/riscmi/t2/t2_0000.nii.gz")
I = np.asarray(t2.dataobj, np.float32)
Pimg = nib.load(V + "results/sub-01/network/t2space/MS_t2_prob.nii.gz")
P = np.asarray(Pimg.dataobj, np.float32)
off = np.array(ev["crop_offset_voxels"])
assert np.allclose(Pimg.affine[:3, :3], t2.affine[:3, :3])
assert np.allclose(
    Pimg.affine[:3, 3], t2.affine[:3, :3] @ off + t2.affine[:3, 3], atol=1e-3
)
to_t1 = np.array(
    json.loads(Path(V + "results/sub-01/network/t2space/t2_to_t1.json").read_text())[
        "matrix"
    ]
)
w2v = np.linalg.inv(to_t1 @ t2.affine)  # T1 world -> T2w voxel (full grid)
name = sys.argv[1]
t = pv.read(V + f"results/sub-01/network/{name}_labeled.vtp")
names = [str(x) for x in t.field_data["segment_names"]]
a = t.lines
i = 0
cells = []
while i < len(a):
    cells.append(a[i + 1 : i + 1 + a[i]])
    i += a[i] + 1
seg = t.point_data["segment"]
R = t.point_data["radius"]
Rr = t.point_data["radius_raw"]
s = np.linspace(-5, 5, 201)  # mm across


def vox(x):
    return (w2v[:3, :3] @ x.T).T + w2v[:3, 3]


rows = []
for c in cells:
    X = t.points[c]
    if len(c) < 5:
        continue
    for k in range(2, len(c) - 2, 3):
        lab = names[seg[c[k]]]
        if lab not in sys.argv[2:]:
            continue
        d = X[k + 2] - X[k - 2]
        d /= np.linalg.norm(d)
        e1 = np.cross(d, [1, 0, 0] if abs(d[0]) < 0.9 else [0, 1, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(d, e1)
        ws, wp = [], []
        for th in np.linspace(0, np.pi, 6, endpoint=False):
            u = np.cos(th) * e1 + np.sin(th) * e2
            q = vox(X[k] + s[:, None] * u)
            prof = ndimage.map_coordinates(I, q.T, order=1)
            pp = ndimage.map_coordinates(P, (q - off).T, order=1)
            c0 = len(s) // 2
            lo_i = (
                c0 + np.argmin(prof[c0 - 15 : c0 + 16]) - 15
            )  # void minimum within 0.75 mm
            bg = min(
                np.percentile(prof[: c0 - 60], 90), np.percentile(prof[c0 + 60 :], 90)
            )  # beyond 3 mm, each side
            half = (prof[lo_i] + bg) / 2
            if bg <= prof[lo_i] * 1.5:
                continue  # no contrast
            l = lo_i
            while l > 0 and prof[l] < half:
                l -= 1
            r_ = lo_i
            while r_ < len(s) - 1 and prof[r_] < half:
                r_ += 1
            if l == 0 or r_ == len(s) - 1:
                continue
            # sub-sample crossings
            fl = s[l] + (half - prof[l]) / (prof[l + 1] - prof[l]) * (s[l + 1] - s[l])
            fr = s[r_ - 1] + (half - prof[r_ - 1]) / (prof[r_] - prof[r_ - 1]) * (
                s[r_] - s[r_ - 1]
            )
            ws.append(fr - fl)
            above = np.flatnonzero(pp >= 0.5 * pp[c0 - 3 : c0 + 4].max())
            run = above[(above >= c0 - 60) & (above <= c0 + 60)]
            wp.append((s[1] - s[0]) * len(run) if len(run) else 0)
        if len(ws) >= 3:
            rows.append((lab, np.median(ws), np.median(wp), 2 * Rr[c[k]], 2 * R[c[k]]))
by = collections.defaultdict(list)
for r in rows:
    by[r[0]].append(r[1:])
print(f"{name}: diameter mm (median [IQR] over cross-sections)")
print(
    f"{'label':10s} {'n':>4s}  {'T2w void FWHM':>18s}  {'P half-max width':>18s}  {'tree raw 2r':>16s}  {'tree fitted 2r':>16s}"
)
for lab in sys.argv[2:]:
    v = np.array(by[lab])
    if not len(v):
        continue
    f = lambda x: (
        f"{np.median(x):5.2f} [{np.percentile(x, 25):.2f}-{np.percentile(x, 75):.2f}]"
    )
    print(
        f"{lab:10s} {len(v):4d}  {f(v[:, 0]):>18s}  {f(v[:, 1]):>18s}  {f(v[:, 2]):>16s}  {f(v[:, 3]):>16s}"
    )
