"""One connected arterial network (voxels) from the risc-mi probability maps.

1. Every {model}/{key}_prob map is warped into T1w space (0.5 mm prepared T1w grid)
   and cached under results/sub-01/network/t1space/.
2. Consensus: mean over contrasts, where the post-Gd sessions (t1gd, t1gd3, t1gd4)
   are first averaged into one contrast so that they don't outvote T1w/T2w.
3. Hysteresis threshold (seeds > --high, grown into > --low), restricted to the
   head ROI.
4. Gap bridging: starting from the largest component, minimum-cost paths
   (cost ~ 1 - consensus) connect every component of >= --min_cc voxels whose gap
   through non-vessel voxels is <= --max_gap mm. Components that cannot be reached
   are dropped.
5. --posterior: behind the brainstem (world y below the median y of FreeSurfer
   brainstem, 16, plus --post_margin), the mean in step 2 needs two of the three
   contrasts, and risc-mi finds little there in any of them. There the
   strongest contrast (max over the per-contrast means) is also accepted, with
   stricter thresholds (--post_low/--post_high), and it sets the bridging cost.
   Only voxels > 1 mm from the mean-consensus mask are added (the wider mask
   would otherwise mostly thicken vessels that are already there).
   This region holds the straight/transverse sinuses, vein of Galen and basal
   veins, so the added voxels are labelled separately for checking.
6. --t2: the T2w map on its own is the densest input (distal MCA/ACA/PCA in the
   SAS), and the mean dilutes vessels only it sees. Its own hysteresis mask
   (--t2_low/--t2_high) is accepted everywhere, beyond a --t2_shell mm shell
   (0 = the full T2w mask, including where it is wider than the network), and
   the bridging cost uses max(consensus, T2w). With --t2_all, every T2w voxel
   above --t2_high is kept, even in pieces the bridging cannot join (so the
   network is then no longer a single component).
7. --trim FRAC: a low --t2_low recovers faint branches but also a wide
   low-probability halo around every vessel. In the T2w-only voxels (label 6),
   keep a voxel only where P = max(consensus, T2w) >= FRAC x the maximum of P
   within --trim_r mm (FRAC = 0.5: each vessel at its half-maximum width), plus
   the skeleton of the untrimmed network, so no branch is lost or cut off.
   Faint branches have a narrow probability ridge, so the half-maximum alone
   leaves them ~1 voxel wide: label-6 voxels within --trim_min_r mm of the
   skeleton of a label-6 branch are kept as well (only inside the untrimmed mask).
   Pieces left without any skeleton voxel are dropped.

Output (results/sub-01/network/):
  network.nii.gz     1 = segmented, 2 = bridged gap (T1w grid)
  network_posterior.{nii.gz,vti,json}  with --posterior; 5 = added by the
                     posterior rule (not in the mean-consensus mask)
  network[_posterior]_t2.*  with --t2; 6 = added from T2w alone
  consensus.nii.gz   consensus probability
  network.vti        cropped to the network; cell data = voxels, in scanner (T1w)
                     coordinates via the image direction matrix. Arrays: network,
                     consensus, votes (#inputs > 0.5), radius_mm (EDT), and
                     prob_<key> (probabilities as uint8 percent, to keep the
                     file small). In ParaView: Threshold on `network` >= 1.
"""

import argparse
import json
import re

import nibabel as nib
import numpy as np
import pyvista as pv
from common import MODALITIES, PREP, REG, RESULTS, SEG, resample_like, save, to_t1
from scipy import ndimage
from skimage.filters import apply_hysteresis_threshold
from skimage.graph import MCP_Geometric
from skimage.measure import block_reduce
from skimage.morphology import ball, skeletonize

REF = PREP / "vesselfm" / "t1_0.50mm.nii.gz"  # T1w space, same grid as evaluate.py
H = 0.5  # mm
OUT = RESULTS / "network"
CONN = np.ones((3, 3, 3))


def contrast(key):
    return re.sub(r"(gd)\d+$", r"\1", key)  # t1gd3 -> t1gd


def load_probs(model, keys, ref):
    probs = {}
    for k in keys:
        src = RESULTS / "riscmi" / model / f"{k}_prob.nii.gz"
        if not src.exists():
            print(f"skip {k}: no {src.relative_to(RESULTS)}")
            continue
        if k != "t1" and not (REG / f"{k}_to_t1_0GenericAffine.mat").exists():
            print(f"skip {k}: not registered to T1w (run register_to_t1.py --keys {k})")
            continue
        cache = OUT / "t1space" / f"{model}_{k}_prob.nii.gz"
        if cache.exists() and cache.stat().st_mtime > src.stat().st_mtime:
            probs[k] = nib.load(cache).get_fdata(dtype=np.float32)
        else:
            print(f"warping {model}/{k} to T1w", flush=True)
            probs[k] = to_t1(src, ref, key=k).astype(np.float32)
            save(probs[k], ref.affine, cache, np.float32)
    return probs


def contrast_means(probs):
    groups = {}
    for k, p in probs.items():
        groups.setdefault(contrast(k), []).append(p)
    print("contrasts:", {g: len(v) for g, v in groups.items()})
    return np.stack([np.mean(v, axis=0) for v in groups.values()])


def posterior_region(ref, margin_mm):
    """World y below the brainstem's median y (+ margin), on the ref grid."""
    seg = np.asanyarray(resample_like(nib.load(SEG), ref, 0).dataobj)
    i, j, k = np.ogrid[tuple(slice(0, n) for n in seg.shape)]
    a = ref.affine
    y = a[1, 0] * i + a[1, 1] * j + a[1, 2] * k + a[1, 3]
    y_cut = float(np.median(np.broadcast_to(y, seg.shape)[seg == 16])) + margin_mm
    return y < y_cut, y_cut


def trim(net, prob, frac, r_mm, min_r):
    """Step 7: drop label-6 voxels below frac x the local maximum of prob."""
    sl = bbox(net > 0, int(r_mm / H) + 1, net.shape)
    n, p = net[sl], prob[sl]
    skel = skeletonize(n > 0)
    local = ndimage.maximum_filter(p, footprint=ball(int(round(r_mm / H))))
    # only around T2w-only branches, not the thin rims of consensus vessels
    near_skel = ndimage.distance_transform_edt(~(skel & (n == 6)), sampling=H) <= min_r
    keep = (n > 0) & ((n != 6) | (p >= frac * local) | near_skel)
    lab, _ = ndimage.label(keep, structure=CONN)
    keep &= np.isin(lab, np.unique(lab[skel]))
    out = np.zeros_like(net)
    out[sl] = np.where(keep, n, 0)
    return out


def bbox(mask, pad, shape):
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def bridge(mask, cons, min_cc, max_gap_mm, n_iter=3):
    """Connect components to the largest one along cheap paths; drop the rest."""
    lab, n = ndimage.label(mask, structure=CONN)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    net = lab == sizes.argmax()
    keep = np.isin(lab, np.nonzero(sizes >= min_cc)[0])
    path_vox = np.zeros_like(mask)
    # voxels inside any candidate component are (almost) free to traverse
    cost = np.where(keep, 0.01, 1.0 - 0.9 * np.clip(cons, 0, 1)).astype(np.float64)
    n_bridges = 0
    for _ in range(n_iter):
        cands = [c for c in np.unique(lab[keep & ~net]) if c]
        if not cands:
            break
        # only search within max_gap of a candidate: impassable (inf) elsewhere,
        # seeded from the network voxels inside that band
        band = ndimage.distance_transform_edt(~(keep & ~net), sampling=H)
        band = band <= max_gap_mm
        mcp = MCP_Geometric(np.where(band | net, cost, np.inf), fully_connected=True)
        seeds = np.argwhere(net & band)
        if not len(seeds):
            break  # nothing left within max_gap of the network
        cum, _ = mcp.find_costs(seeds)
        objs = ndimage.find_objects(lab)
        order = []
        for c in cands:
            sl = objs[c - 1]
            sub = np.where(lab[sl] == c, cum[sl], np.inf)
            i = np.unravel_index(np.argmin(sub), sub.shape)
            if np.isfinite(sub[i]):
                order.append((sub[i], c, tuple(s.start + j for s, j in zip(sl, i))))
        added = 0
        for _, c, end in sorted(order):
            if net[end]:
                continue  # already absorbed via an earlier path
            path = np.array(mcp.traceback(end))
            on_path = mask[tuple(path.T)]
            steps = np.linalg.norm(np.diff(path, axis=0), axis=1) * H
            gap = steps[~on_path[1:]].sum()
            if gap > max_gap_mm:
                continue
            touched = np.unique(lab[tuple(path.T)])
            net |= np.isin(lab, touched[touched > 0])
            gap_idx = path[~on_path]
            net[tuple(gap_idx.T)] = True
            path_vox[tuple(gap_idx.T)] = True
            n_bridges += 1
            added += 1
        print(f"  bridging pass: {added} components joined", flush=True)
        if not added:
            break
    # thicken 1-voxel bridges slightly so they read as vessels
    path_vox = ndimage.binary_dilation(path_vox, CONN) & ~mask & (cons > 0.05)
    path_vox |= net & ~mask
    net |= path_vox
    return net, path_vox, n_bridges, n


def coarse_roi(ref, margin_mm, h=2.0):
    """Brain+CSF seg dilated by `margin_mm`, computed on a coarse grid.

    Same as common.head_roi, but the full-resolution EDT there takes ~1 min and
    the margin only needs to be approximate.
    """
    seg = np.asanyarray(resample_like(nib.load(SEG), ref, 0).dataobj) > 0
    f = int(round(h / H))
    c = block_reduce(seg, (f, f, f), np.max)
    c = ndimage.distance_transform_edt(~c, sampling=f * H) <= margin_mm
    for ax in range(3):
        c = np.repeat(c, f, axis=ax)
    return c[: seg.shape[0], : seg.shape[1], : seg.shape[2]]


def write_vti(path, arrays, affine, sl):
    """Cropped cell-data ImageData in scanner coordinates (voxel = cell)."""
    lo = np.array([s.start for s in sl], float)
    shape = np.array([s.stop - s.start for s in sl])
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    grid = pv.ImageData(
        dimensions=tuple(shape + 1),
        spacing=tuple(spacing),
        origin=tuple(affine[:3, :3] @ (lo - 0.5) + affine[:3, 3]),
        direction_matrix=affine[:3, :3] / spacing,
    )
    for name, a in arrays.items():
        grid.cell_data[name] = a[sl].ravel(order="F")
    grid.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="MS", choices=["MS", "MC"])
    p.add_argument(
        "--keys",
        nargs="+",
        default=[k for k in MODALITIES if k != "flair"],
        help="inputs to use if available (missing / unregistered ones are skipped)",
    )
    p.add_argument("--low", type=float, default=0.2)
    p.add_argument("--high", type=float, default=0.5)
    p.add_argument("--min_cc", type=int, default=30, help="voxels")
    p.add_argument("--max_gap", type=float, default=5.0, help="mm")
    p.add_argument("--roi_margin", type=float, default=20.0, help="mm around brain")
    p.add_argument(
        "--posterior",
        action="store_true",
        help="also accept the strongest contrast behind the brainstem (step 5)",
    )
    p.add_argument("--post_low", type=float, default=0.3)
    p.add_argument("--post_high", type=float, default=0.7)
    p.add_argument(
        "--post_margin", type=float, default=0.0, help="mm added to the brainstem y"
    )
    p.add_argument("--t2", action="store_true", help="also accept T2w alone (step 6)")
    p.add_argument("--t2_low", type=float, default=0.3)
    p.add_argument("--t2_high", type=float, default=0.5)
    p.add_argument("--t2_shell", type=float, default=1.0, help="mm, see step 6")
    p.add_argument("--t2_all", action="store_true", help="see step 6")
    p.add_argument("--trim", type=float, default=0.0, help="see step 7; 0 = off")
    p.add_argument("--trim_r", type=float, default=2.0, help="mm, see step 7")
    p.add_argument("--trim_min_r", type=float, default=0.75, help="mm, step 7")
    args = p.parse_args()
    name = "network" + "_posterior" * args.posterior + "_t2" * args.t2

    ref = nib.load(REF)
    probs = load_probs(args.model, args.keys, ref)
    means = contrast_means(probs)
    cons = means.mean(axis=0)
    votes = sum((p > 0.5).astype(np.uint8) for p in probs.values())

    roi = coarse_roi(ref, args.roi_margin)
    mask = apply_hysteresis_threshold(cons, args.low, args.high) & roi
    # the extra masks below are wider than the mean one around the same vessels;
    # keep only new vessel beyond a 1 mm shell and let bridging join it
    shell = ndimage.binary_dilation(mask, CONN, iterations=2)
    added = np.zeros_like(mask)
    added_t2 = np.zeros_like(mask)
    cost_map = cons
    if args.posterior:
        post, y_cut = posterior_region(ref, args.post_margin)
        strongest = means.max(axis=0)
        extra = apply_hysteresis_threshold(strongest, args.post_low, args.post_high)
        added = extra & post & roi & ~shell
        cost_map = np.where(post, strongest, cons)
        print(
            f"posterior rule behind y = {y_cut:.1f} mm: "
            f"+{added.sum() * H**3 / 1000:.2f} mL",
            flush=True,
        )
    if args.t2:
        t2 = probs["t2"]
        added_t2 = apply_hysteresis_threshold(t2, args.t2_low, args.t2_high)
        n = int(round(args.t2_shell / H))
        t2_shell = ndimage.binary_dilation(mask, CONN, iterations=n) if n else mask
        added_t2 &= roi & ~t2_shell & ~added
        cost_map = np.maximum(cost_map, t2)
        print(f"T2w rule: +{added_t2.sum() * H**3 / 1000:.2f} mL", flush=True)
    mask |= added | added_t2

    sl = bbox(mask, int(2 * args.max_gap / H) + 2, mask.shape)
    net_c, path_c, n_bridges, n_cc0 = bridge(
        mask[sl], cost_map[sl], args.min_cc, args.max_gap
    )
    net = np.zeros(mask.shape, np.uint8)
    net[sl][net_c] = 1
    net[sl][path_c] = 2
    net[(net == 1) & added] = 5  # 3, 4 are taken by grow_network.py
    net[(net == 1) & added_t2] = 6
    if args.t2_all:
        net[(net == 0) & (probs["t2"] > args.t2_high) & roi] = 6

    vol_untrimmed = float((net > 0).sum() * H**3 / 1000)
    if args.trim:
        net = trim(net, cost_map, args.trim, args.trim_r, args.trim_min_r)
        print(f"trim: {vol_untrimmed:.1f} -> {(net > 0).sum() * H**3 / 1000:.1f} mL")

    radius = ndimage.distance_transform_edt(net > 0, sampling=H).astype(np.float32)
    save(net, ref.affine, OUT / f"{name}.nii.gz", np.uint8)
    save(cons, ref.affine, OUT / "consensus.nii.gz", np.float32)

    def pct(a):
        return np.round(100 * np.clip(a, 0, 1)).astype(np.uint8)

    arrays = {"network": net, "consensus": pct(cons), "votes": votes}
    arrays |= {"radius_mm": radius} | {f"prob_{k}": pct(v) for k, v in probs.items()}
    write_vti(OUT / f"{name}.vti", arrays, ref.affine, bbox(net > 0, 6, net.shape))

    kept = mask & (net > 0)
    stats = dict(
        inputs=list(probs),
        model=args.model,
        args=vars(args),
        cc_before=int(n_cc0),
        vol_before_ml=float(mask.sum() * H**3 / 1000),
        vol_ml=float((net > 0).sum() * H**3 / 1000),
        kept_fraction=float(kept.sum() / max(mask.sum(), 1)),
        bridges=n_bridges,
        bridge_vox=int((net == 2).sum()),
        posterior_ml=float((net == 5).sum() * H**3 / 1000),
        t2_ml=float((net == 6).sum() * H**3 / 1000),
        vol_untrimmed_ml=vol_untrimmed,
        cc_after=int(ndimage.label(net > 0, structure=CONN)[1]),
    )
    (OUT / f"{name}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
