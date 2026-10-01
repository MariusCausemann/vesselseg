"""SeqSeg (github.com/numisveinsson/SeqSeg) centerlines of the arterial network.

SeqSeg traces vessels step by step from a few seeds: at each step it crops a
box around the current point (sized from the radius), segments it, extracts the
local surface/centerline and picks the next point(s), detecting bifurcations.
The public nnU-Net weights are aorta (CT/MR) and coronary CTA only, so it is run
in its SEGMENTATION mode: the local "prediction" is the crop of network.nii.gz
(the MS consensus tree, labels 1+2 binarized), i.e. SeqSeg acts as a tracker
over our mask rather than as a segmenter.

Seeds: the endpoints of centerlines.vtp (extract_centerlines.py) below
--seed_zmax mm with radius >= --seed_rmin mm (neck inlets: both ICAs and the
vertebral/basilar ends). Each seed points into its branch, --seed_step mm in.

SeqSeg works in SimpleITK physical coordinates (LPS); everything written here
is converted back to scanner RAS (T1w space, mm), like the other network files.

Output (results/sub-01/seqseg/):
  seqseg_tree.vtp        step graph: one polyline per traced branch, point data
                         radius, step, branch, bifurcation
  seqseg_centerline.vtp  SeqSeg's global centerline of its traced mask
  seqseg_seg.nii.gz      the traced mask (assembly >= 0.5), network grid
  seqseg.json            seeds, steps, branches, length, coverage vs network

Run in the seqseg env:  pixi run -e seqseg python run_seqseg.py
"""

import argparse
import json
import sys

import numpy as np
import SimpleITK as sitk
import vtk
import yaml
from common import RESULTS
from scipy.spatial import cKDTree
from seqseg import AlgorithmConfig
from seqseg.modules.assembly import calc_centerline_global
from seqseg.modules.initialization import initialize_json
from seqseg.modules.tracing import TracingContext, trace_centerline_from_context
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

NET = RESULTS / "network"
OUT = RESULTS / "seqseg"
LPS = np.array([-1.0, -1.0, 1.0])  # RAS <-> LPS (its own inverse)


def read_vtp(path):
    r = vtk.vtkXMLPolyDataReader()
    r.SetFileName(str(path))
    r.Update()
    return r.GetOutput()


def write_vtp(pd, path):
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(str(path))
    w.SetInputData(pd)
    w.Write()


def flip_points(pd):
    """Copy of pd with x, y negated (LPS <-> RAS)."""
    out = vtk.vtkPolyData()
    out.DeepCopy(pd)
    if out.GetNumberOfPoints():
        p = vtk_to_numpy(out.GetPoints().GetData()) * LPS
        out.GetPoints().SetData(numpy_to_vtk(p, deep=True))
    return out


def seeds_from_centerlines(args):
    """[[start, direction point, radius], ...] in LPS from the low wide endpoints."""
    cl = read_vtp(NET / "centerlines.vtp")
    P = vtk_to_numpy(cl.GetPoints().GetData())
    r = vtk_to_numpy(cl.GetPointData().GetArray("radius"))
    nt = vtk_to_numpy(cl.GetPointData().GetArray("node_type"))
    ends = np.where((nt > 0) & (P[:, 2] <= args.seed_zmax) & (r >= args.seed_rmin))[0]
    seeds = []
    for e in ends[np.argsort(P[ends, 2])]:
        ids = vtk.vtkIdList()
        cl.GetPointCells(e, ids)
        cell = cl.GetCell(ids.GetId(0))
        path = [cell.GetPointId(k) for k in range(cell.GetNumberOfPoints())]
        if path[0] != e:
            path = path[::-1]
        p = P[path]
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        k = min(np.searchsorted(s, args.seed_step), len(p) - 1)
        seeds.append(
            [
                (p[0] * LPS).tolist(),
                (p[k] * LPS).tolist(),
                float(args.seed_rscale * r[e]),
            ]
        )
    return seeds


def tree_polydata(tree):
    """One polyline per traced branch; branch[0] is the step it grows from."""
    pts, rad, step, br, lines = [], [], [], [], vtk.vtkCellArray()
    bif = set(tree.bifurcations)
    for b, branch in enumerate(tree.branches):
        if len(branch) < 2:
            continue
        lines.InsertNextCell(len(branch))
        for s in branch:
            lines.InsertCellPoint(len(pts))
            pts.append(np.asarray(tree.steps[s]["point"]) * LPS)
            rad.append(tree.steps[s]["radius"])
            step.append(s)
            br.append(b)
    pd = vtk.vtkPolyData()
    vp = vtk.vtkPoints()
    vp.SetData(numpy_to_vtk(np.array(pts, dtype=float), deep=True))
    pd.SetPoints(vp)
    pd.SetLines(lines)
    for name, a in [
        ("radius", np.array(rad, dtype=float)),
        ("step", np.array(step, dtype=np.int32)),
        ("branch", np.array(br, dtype=np.int32)),
        ("bifurcation", np.array([s in bif for s in step], dtype=np.int32)),
    ]:
        v = numpy_to_vtk(a, deep=True)
        v.SetName(name)
        pd.GetPointData().AddArray(v)
    return pd


def resample_mask(mask, spacing, margin=5.0):
    """Crop to the mask bbox + margin mm, then resample to `spacing` mm.

    Linear interpolation of the binary mask thresholded at 0.5, so 1-3 voxel
    wide vessels become rounder tubes. No pre-smoothing: even sigma = 0.25 mm
    splits off 17 % of the network at thin links.
    """
    ls = sitk.LabelShapeStatisticsImageFilter()
    ls.Execute(mask)
    x0, y0, z0, sx, sy, sz = ls.GetBoundingBox(1)
    m = np.ceil(margin / np.array(mask.GetSpacing())).astype(int)
    lo = np.maximum([x0, y0, z0] - m, 0)
    hi = np.minimum(np.array([x0 + sx, y0 + sy, z0 + sz]) + m, mask.GetSize())
    crop = sitk.RegionOfInterest(mask, (hi - lo).tolist(), lo.tolist())
    size = np.round(np.array(crop.GetSize()) * crop.GetSpacing() / spacing)
    ref = sitk.Image(size.astype(int).tolist(), sitk.sitkFloat32)
    ref.SetSpacing([spacing] * 3)
    ref.SetDirection(crop.GetDirection())
    # same outer corner as the crop
    ref.SetOrigin(crop.TransformContinuousIndexToPhysicalPoint([-0.5] * 3))
    ref.SetOrigin(ref.TransformContinuousIndexToPhysicalPoint([0.5] * 3))
    f = sitk.Resample(
        sitk.Cast(crop, sitk.sitkFloat32), ref, sitk.Transform(), sitk.sitkLinear
    )
    return sitk.Cast(f >= 0.5, sitk.sitkUInt8)


def polyline_length(pd):
    total = 0.0
    P = vtk_to_numpy(pd.GetPoints().GetData()) if pd.GetNumberOfPoints() else None
    for c in range(pd.GetNumberOfCells()):
        ids = [
            pd.GetCell(c).GetPointId(k)
            for k in range(pd.GetCell(c).GetNumberOfPoints())
        ]
        if len(ids) > 1:
            total += np.linalg.norm(np.diff(P[ids], axis=0), axis=1).sum()
    return float(total)


def coverage(ref_pd, pd, tol):
    """Fraction of ref points (radius-aware) within max(tol, radius) of pd's points."""
    if not pd.GetNumberOfPoints():
        return 0.0
    ref = vtk_to_numpy(ref_pd.GetPoints().GetData())
    rr = vtk_to_numpy(ref_pd.GetPointData().GetArray("radius"))
    d, _ = cKDTree(vtk_to_numpy(pd.GetPoints().GetData())).query(ref)
    return float(np.mean(d <= np.maximum(tol, rr)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="global_seg", help="packaged SeqSeg config")
    ap.add_argument("--seed_zmax", type=float, default=-25.0, help="mm (RAS z)")
    ap.add_argument("--seed_rmin", type=float, default=1.0, help="mm")
    ap.add_argument("--seed_step", type=float, default=2.0, help="mm into the branch")
    ap.add_argument("--seed_rscale", type=float, default=1.0, help="x seed radius")
    ap.add_argument(
        "--stop_radius", type=float, default=0.4, help="mm (global.yaml: 0.46)"
    )
    ap.add_argument("--min_radius", type=float, default=0.3, help="mm")
    ap.add_argument("--volume_ratio", type=float, default=None, help="box / radius")
    ap.add_argument(
        "--spacing",
        type=float,
        default=None,
        help="mm; resample the mask (cropped to its bbox) before tracing",
    )
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--max_branches", type=int, default=400)
    ap.add_argument("--max_steps_branch", type=int, default=200)
    ap.add_argument(
        "--restart_branch",
        action="store_true",
        help="re-trace a branch that ran into traced vessel (loops forever in seg mode)",
    )
    ap.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override SeqSeg config entries (YAML values), e.g. MERGE_NEXT=true",
    )
    ap.add_argument("--no_global_centerline", action="store_true")
    ap.add_argument(
        "--write_steps", action="store_true", help="dump per-step crops etc."
    )
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t = f"_{args.tag}" if args.tag else ""

    cfg = AlgorithmConfig.from_name(args.config)
    cfg["SEGMENTATION"] = True
    cfg["DEBUG"] = False  # True drops into pdb at DEBUG_STEP
    cfg["ANIMATION"] = False
    cfg["TREE_ANALYSIS"] = False
    cfg["STOP_RADIUS"] = args.stop_radius
    cfg["MIN_RADIUS"] = args.min_radius
    cfg["RESTART_BRANCH"] = args.restart_branch
    cfg.setdefault("ADD_RADIUS", 0)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    if args.volume_ratio:
        cfg["VOLUME_SIZE_RATIO"] = args.volume_ratio

    net = sitk.ReadImage(str(NET / "network.nii.gz"))
    seg = sitk.Cast(net > 0, sitk.sitkUInt8)
    if args.spacing:
        seg = resample_mask(seg, args.spacing)
        print(f"mask resampled to {args.spacing} mm, size {seg.GetSize()}")

    seeds = seeds_from_centerlines(args)
    print(f"{len(seeds)} seeds (LPS start, direction point, radius):")
    for s in seeds:
        print("  ", np.round(s[0], 1), np.round(s[1], 1), round(s[2], 2))
    case = "sub-01"
    branches, initial_seeds = initialize_json(
        {"name": case, "seeds": seeds}, str(OUT) + "/", None, None, "mm"
    )

    steps_dir = OUT / f"steps{t}"
    if args.write_steps:
        for d in ["points", "volumes", "predictions", "centerlines", "surfaces"]:
            (steps_dir / d).mkdir(parents=True, exist_ok=True)
    ctx = TracingContext(
        output_folder=str(steps_dir) + "/",
        image_file=seg,
        case=case,
        model_folder="",
        fold="all",
        potential_branches=branches,
        max_step_size=args.max_steps,
        max_n_branches=args.max_branches,
        max_n_steps_per_branch=args.max_steps_branch,
        global_config=cfg,
        unit="mm",
        seg_file=seg,
        write_samples=args.write_steps,
        disk_io=args.write_steps,
    )
    res = trace_centerline_from_context(ctx)
    tree = res.vessel_tree

    traced = sitk.BinaryThreshold(res.assembly.assembly, 0.5, 1e9)
    traced = sitk.Resample(traced, net, sitk.Transform(), sitk.sitkNearestNeighbor)
    sitk.WriteImage(
        sitk.Cast(traced, sitk.sitkUInt8), str(OUT / f"seqseg_seg{t}.nii.gz")
    )

    tree_pd = tree_polydata(tree)
    write_vtp(tree_pd, OUT / f"seqseg_tree{t}.vtp")

    ref = read_vtp(NET / "centerlines.vtp")
    a_net = sitk.GetArrayViewFromImage(net) > 0
    a_tr = sitk.GetArrayViewFromImage(traced) > 0
    summary = {
        "args": vars(args),
        "config": {k: cfg[k] for k in sorted(cfg.keys())},
        "seeds_lps": seeds,
        "steps": int(res.n_steps_taken),
        "branches": int(sum(len(b) > 1 for b in tree.branches)),
        "bifurcations": len(tree.bifurcations),
        "tree_length_mm": polyline_length(tree_pd),
        "traced_ml": float(a_tr.sum() * np.prod(net.GetSpacing()) / 1000),
        "network_voxels_traced": float((a_net & a_tr).sum() / a_net.sum()),
        "vmtk_centerline_covered_by_tree": coverage(ref, tree_pd, 1.0),
    }

    if not args.no_global_centerline:
        g, _, ok = calc_centerline_global(
            sitk.Cast(traced, sitk.sitkUInt8), initial_seeds
        )
        g = flip_points(g)
        write_vtp(g, OUT / f"seqseg_centerline{t}.vtp")
        summary |= {
            "global_centerline_success": bool(ok),
            "global_centerline_length_mm": polyline_length(g),
            "vmtk_centerline_covered_by_global": coverage(ref, g, 1.0),
        }

    (OUT / f"seqseg{t}.json").write_text(json.dumps(summary, indent=2, default=str))
    sys.stdout = sys.__stdout__
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("config", "args")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
