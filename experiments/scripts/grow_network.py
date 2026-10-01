"""Grow the risc-mi network (build_network.py) into model-free vesselness.

risc-mi finds almost nothing above z ~ +60 mm or around the cerebellum, although
the T2w flow voids show tubes there. This extends the network with dark-tube
evidence the models do not see:

1. Vesselness: Frangi (dark ridges, sigma 0.5-1 mm) on the ses-01 T2w and the
   ses-03 post-Gd T1w (highest CSF Gd, so vessels are dark in bright CSF), both
   warped into T1w space, normalised within the SAS and combined as a geometric
   mean (each dilated by one voxel first to tolerate residual misregistration),
   so that both contrasts have to agree.
2. Growth: voxels in the SAS (FreeSurfer 24, dilated 1 mm) with vesselness above
   the --pct percentile, within --max_len mm geodesic distance (through such
   voxels) of the existing network.
3. Veins: each grown piece (component of grown & ~network) that comes within
   --sinus_mm of a dural-sinus zone (just outside the brain envelope, along the
   superior midline or posterior-inferiorly at the transverse sinus) is labelled
   separately as likely vein.
4. --posterior: starts from network_posterior.nii.gz (build_network.py
   --posterior), also grows behind the brainstem (see
   build_network.posterior_region), and there uses the T2w vesselness alone:
   risc-mi misses most of the posterior T2w flow voids, and the ses-03 contrast
   is weaker there. Outputs are network_grown_posterior.*.

Output (results/sub-01/network/):
  network_grown.nii.gz   1 segmented, 2 bridged, 3 grown, 4 grown + reaches sinus
                         (with --posterior also 5, from network_posterior)
  network_grown.vti      as network.vti, plus `vesselness` (percent)
  vesselness.nii.gz      combined vesselness (cached, with the per-contrast maps)
"""

import argparse
import json

import nibabel as nib
import numpy as np
from build_network import CONN, H, OUT, REF, bbox, posterior_region, write_vti
from common import MODALITIES, SEG, resample_like, save, to_t1
from scipy import ndimage
from skimage.filters import frangi
from skimage.graph import MCP_Geometric
from skimage.measure import block_reduce
from skimage.morphology import skeletonize

SIGMAS = (1.0, 1.5, 2.0)  # voxels = 0.5-1 mm
VESSEL_KEYS = ("t2", "t1gd3")
L_HEMI = [2, 3]  # FreeSurfer left cerebral WM / cortex
R_HEMI = [41, 42]
CEREB = [7, 8, 46, 47]


def vesselness(key, ref, roi):
    cache = OUT / "t1space" / f"vesselness_{key}.nii.gz"
    if cache.exists():
        return nib.load(cache).get_fdata(dtype=np.float32)
    print(f"warping {key} image and computing vesselness", flush=True)
    img = to_t1(MODALITIES[key][0], ref, key=key)
    sl = bbox(roi, 8, roi.shape)
    v = np.zeros(ref.shape, np.float32)
    v[sl] = frangi(img[sl], sigmas=SIGMAS, black_ridges=True)
    v /= np.percentile(v[roi], 99.5) + 1e-12
    v = np.clip(v, 0, 1) * roi
    save(v, ref.affine, cache, np.float32)
    return v


def coarse(mask, f):
    return block_reduce(mask, (f, f, f), np.max)


def upsample(c, f, shape):
    for ax in range(3):
        c = np.repeat(c, f, axis=ax)
    return c[: shape[0], : shape[1], : shape[2]]


def sinus_zone(seg, ref, shell_mm=8.0, h=2.0):
    """Dural-sinus zone: a shell just outside the brain envelope, restricted to
    the superior midline (SSS) and around the back of the cerebellum (transverse)."""
    f = int(round(h / H))
    brain = coarse(seg > 0, f)
    env = ndimage.binary_fill_holes(
        ndimage.binary_closing(brain, iterations=3, border_value=0)
    )
    d_env = ndimage.distance_transform_edt(~env, sampling=h)
    shell = (d_env > 0) & (d_env <= shell_mm)
    d_l = ndimage.distance_transform_edt(~coarse(np.isin(seg, L_HEMI), f), sampling=h)
    d_r = ndimage.distance_transform_edt(~coarse(np.isin(seg, R_HEMI), f), sampling=h)
    midline = np.abs(d_l - d_r) <= 10
    z = (np.indices(brain.shape).reshape(3, -1).T * f) @ ref.affine[2, :3]
    z = (z + ref.affine[2, 3]).reshape(brain.shape)
    zc = np.median(z[brain])
    sss = shell & midline & (z > zc)
    d_cb = ndimage.distance_transform_edt(~coarse(np.isin(seg, CEREB), f), sampling=h)
    trans = shell & (d_cb <= 15)
    return upsample(sss | trans, f, seg.shape)


def endpoints(net):
    skel = skeletonize(net) > 0
    nb = ndimage.convolve(skel.astype(np.uint8), np.ones((3, 3, 3), np.uint8))
    return skel & (nb == 2)  # itself + one neighbour


def blind_zone(seg, ref, z_min):
    """Where risc-mi finds (almost) nothing: above z_min, and the cerebellar
    surface away from the brainstem (whose cisterns hold the cranial nerves)."""
    i, j, k = np.ogrid[tuple(slice(0, n) for n in seg.shape)]
    zz = ref.affine[2, 0] * i + ref.affine[2, 1] * j + ref.affine[2, 2] * k
    zz = zz + ref.affine[2, 3]
    f = 4
    d_cb = ndimage.distance_transform_edt(
        ~block_reduce(np.isin(seg, [8, 47]), (f, f, f), np.max), sampling=f * H
    )
    d_bs = ndimage.distance_transform_edt(
        ~block_reduce(seg == 16, (f, f, f), np.max), sampling=f * H
    )
    cereb = upsample((d_cb <= 4) & (d_bs > 10), f, seg.shape)
    return (zz > z_min) | cereb


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pct", type=float, default=95.0, help="vesselness percentile in SAS"
    )
    p.add_argument("--max_len", type=float, default=60.0, help="mm geodesic growth")
    p.add_argument(
        "--z_min",
        type=float,
        default=40.0,
        help="grow only above this world z (mm), or on the cerebellar surface",
    )
    p.add_argument("--sinus_mm", type=float, default=2.0)
    p.add_argument(
        "--max_thick", type=float, default=2.0, help="mm, sd along thinnest axis"
    )
    p.add_argument("--only_vesselness", action="store_true")
    p.add_argument("--posterior", action="store_true", help="see step 4")
    p.add_argument(
        "--post_margin", type=float, default=0.0, help="mm added to the brainstem y"
    )
    args = p.parse_args()
    src, name = "network", "network_grown"
    if args.posterior:
        src, name = "network_posterior", "network_grown_posterior"

    ref = nib.load(REF)
    seg = np.asanyarray(resample_like(nib.load(SEG), ref, 0).dataobj)
    sas = ndimage.binary_dilation(seg == 24, CONN, iterations=2)

    vs = {k: vesselness(k, ref, sas) for k in VESSEL_KEYS}
    grown = {k: ndimage.grey_dilation(v, size=(3, 3, 3)) for k, v in vs.items()}
    V = np.sqrt(np.prod(list(grown.values()), axis=0)).astype(np.float32) * sas
    save(V, ref.affine, OUT / "vesselness.nii.gz", np.float32)
    if args.only_vesselness:
        return

    net0 = np.asanyarray(nib.load(OUT / f"{src}.nii.gz").dataobj)
    net = net0 > 0
    thr = np.percentile(V[sas], args.pct)
    zone_grow = blind_zone(seg, ref, args.z_min)
    if args.posterior:
        post, y_cut = posterior_region(ref, args.post_margin)
        V = np.where(post, grown["t2"] * sas, V)
        zone_grow |= post
        print(f"posterior: T2w-only vesselness behind y = {y_cut:.1f} mm")
    cand = (V > thr) & sas & ~net & zone_grow

    # grow only from the free ends: seeds near skeleton endpoints, and candidates
    # hugging a vessel shaft (the flow void is wider than the risc-mi mask) removed
    sl = bbox(net, int(args.max_len / H) + 2, net.shape)
    ends = endpoints(net[sl])
    near_end = ndimage.distance_transform_edt(~ends, sampling=H) <= 2.0
    shaft = ndimage.distance_transform_edt(~net[sl], sampling=H) <= 1.5
    c = cand[sl] & (~shaft | near_end)
    seeds = net[sl] & near_end
    print(
        f"vesselness thr {thr:.3f}: {c.sum() * H**3 / 1000:.1f} mL candidates, "
        f"{int(ends.sum())} network endpoints",
        flush=True,
    )

    # geodesic distance from the seeds, travelling only through candidates
    mcp = MCP_Geometric(np.where(c | seeds, 1.0, np.inf), fully_connected=True)
    cum, _ = mcp.find_costs(np.argwhere(seeds))
    add = np.zeros_like(net)
    add[sl] = c & (cum * H <= args.max_len)

    # drop grown pieces that are blobs or sheets rather than tubes (falx edge,
    # sulcal CSF): too thick along their thinnest principal axis
    lab, n = ndimage.label(add, structure=CONN)
    dropped = 0
    for c, sl_c in enumerate(ndimage.find_objects(lab), 1):
        ijk = np.argwhere(lab[sl_c] == c)
        if len(ijk) >= 20:
            sd = np.sqrt(np.linalg.eigvalsh(np.cov((ijk @ ref.affine[:3, :3].T).T)))
            if sd[0] > args.max_thick:
                add[sl_c][lab[sl_c] == c] = False
                dropped += len(ijk)
    print(f"dropped {dropped * H**3 / 1000:.2f} mL of non-tubular growth")

    zone = sinus_zone(seg, ref)
    near = ndimage.distance_transform_edt(~zone, sampling=H) <= args.sinus_mm
    lab, n = ndimage.label(add, structure=CONN)
    veinlike = np.unique(lab[near & add])
    vein = np.isin(lab, veinlike[veinlike > 0])

    out = net0.astype(np.uint8)
    out[add] = 3
    out[vein] = 4
    save(out, ref.affine, OUT / f"{name}.nii.gz", np.uint8)

    def pct(a):
        return np.round(100 * np.clip(a, 0, 1)).astype(np.uint8)

    cons = nib.load(OUT / "consensus.nii.gz").get_fdata(dtype=np.float32)
    radius = ndimage.distance_transform_edt(out > 0, sampling=H).astype(np.float32)
    arrays = {"network": out, "consensus": pct(cons), "radius_mm": radius}
    arrays |= {"vesselness": pct(V)} | {
        f"vesselness_{k}": pct(v) for k, v in vs.items()
    }
    arrays |= {"sinus_zone": zone.astype(np.uint8)}
    write_vti(OUT / f"{name}.vti", arrays, ref.affine, bbox(out > 0, 6, out.shape))

    stats = {
        "args": vars(args),
        "thr": float(thr),
        "vol_network_ml": float(net.sum() * H**3 / 1000),
        "vol_grown_ml": float((out == 3).sum() * H**3 / 1000),
        "vol_dropped_ml": float(dropped * H**3 / 1000),
        "vol_veinlike_ml": float((out == 4).sum() * H**3 / 1000),
        "grown_pieces": int(n),
        "veinlike_pieces": int((veinlike > 0).sum()),
        "cc_after": int(ndimage.label(out > 0, structure=CONN)[1]),
    }
    (OUT / f"{name}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
