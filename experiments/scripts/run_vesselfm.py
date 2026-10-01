"""vesselFM zero-shot inference on the prepared sub-01 inputs.

Re-implements the inference loop of external/vesselFM/vesselfm/seg/inference.py
(same model, sliding-window settings and TTA semantics) but
  * normalises with percentiles computed *inside the head mask* (the upstream
    ScaleIntensityRangePercentiles sees the zero padding),
  * optionally inverts dark-vessel contrasts (T2w, FLAIR, post-Gd T1w), or
    feeds a black top-hat of them (flow voids in bright CSF -> bright tubes),
  * writes the sigmoid probability map with the input affine (upstream drops
    origin/direction).

Output: results/sub-01/vesselfm/{ckpt}/{tag}_{raw|inv|tophat}[_tta]_prob.nii.gz
"""

import argparse
import itertools
import time

import hydra
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from common import MODALITIES, PREP, RESULTS, ROOT, save
from monai.inferers import SlidingWindowInfererAdapt
from omegaconf import OmegaConf
from scipy import ndimage

MODEL_CFG = ROOT / "external/vesselFM/vesselfm/seg/configs/model/dyn_unet_base.yaml"
TOPHAT_MM = 4.5  # widest vessel the closing fills
TOPHAT_CSF = 0.35  # closed intensity above which the surroundings count as CSF

CKPTS = {
    "base": "vesselFM_base.pt",
    "all": "vesselFM_all.pt",
    "all_ema": "vesselFM_all_ema.pt",
}


def load_model(ckpt, device):
    model = hydra.utils.instantiate(OmegaConf.load(MODEL_CFG))
    sd = torch.load(
        ROOT / "weights/vesselfm" / CKPTS[ckpt], map_location=device, weights_only=True
    )
    model.load_state_dict(sd)
    return model.to(device).eval()


def normalise(a, mask, lower, upper, contrast, spacing):
    lo, hi = np.percentile(a[mask], [lower, upper])
    x = np.clip((a - lo) / (hi - lo), 0, 1)
    if contrast == "inv":
        x = 1 - x
    elif contrast == "tophat":
        # angiography-like: bright where something narrow is dark inside bright CSF
        # (flow voids); parenchyma and CSF itself both map to ~0
        size = round(TOPHAT_MM / spacing) | 1
        closed = ndimage.grey_closing(x, size=(size,) * 3)
        x = (closed - x) * (closed > TOPHAT_CSF)
        x = x / np.percentile(x[mask], 99.9)
        x = np.clip(x, 0, 1)
    return np.where(mask, x, 0).astype(np.float32)


def resample(x, factor=None, shape=None):
    if factor == 1:
        return x
    size = shape[2:] if shape else [round(s / factor) for s in x.shape[2:]]
    return F.interpolate(x, size=size, mode="trilinear", align_corners=False)


@torch.no_grad()
def predict(model, x, inferer, scales, device):
    x = torch.from_numpy(x)[None, None].to(device)
    logits = []
    for s in scales:
        xs = resample(x, s)
        y = inferer(xs, model)
        logits.append(resample(y, shape=x.shape).cpu())
    return torch.stack(logits).mean(0).sigmoid().squeeze().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", default=["base", "all"], choices=list(CKPTS))
    p.add_argument("--keys", nargs="+", default=list(MODALITIES))
    p.add_argument("--scales", nargs="+", default=["0.33", "0.50", "0.75"])
    p.add_argument(
        "--contrast",
        nargs="+",
        default=["auto"],
        choices=["auto", "raw", "inv", "tophat"],
    )
    p.add_argument("--tta", action="store_true", help="multi-scale TTA [0.5, 1, 1.5]")
    p.add_argument("--lower", type=float, default=0.5)
    p.add_argument("--upper", type=float, default=99.5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    inferer = SlidingWindowInfererAdapt(
        roi_size=[128, 128, 128],
        sw_batch_size=args.batch_size,
        overlap=0.5,
        mode="constant",
        sigma_scale=0.125,
        padding_mode="constant",
    )
    tta_scales = [0.5, 1, 1.5] if args.tta else [1]

    for ckpt in args.ckpt:
        model = load_model(ckpt, device)
        for key, scale in itertools.product(args.keys, args.scales):
            f = PREP / "vesselfm" / f"{key}_{scale}mm.nii.gz"
            if not f.exists():
                continue
            img = nib.load(f)
            a = img.get_fdata(dtype=np.float32)
            mask = (
                np.asanyarray(
                    nib.load(str(f).replace(".nii.gz", "_mask.nii.gz")).dataobj
                )
                > 0
            )
            contrasts = args.contrast
            if contrasts == ["auto"]:
                contrasts = ["raw", "inv", "tophat"] if MODALITIES[key][1] else ["raw"]
            for c in contrasts:
                tag = f"{key}_{scale}mm_{c}" + ("_tta" if args.tta else "")
                out = RESULTS / "vesselfm" / ckpt / f"{tag}_prob.nii.gz"
                if out.exists() and not args.overwrite:
                    continue
                x = normalise(a, mask, args.lower, args.upper, c, float(scale))
                t0 = time.time()
                prob = predict(model, x, inferer, tta_scales, device) * mask
                save(prob, img.affine, out, np.float32)
                print(
                    f"{ckpt}/{tag}: {time.time() - t0:.0f}s, frac>0.5 = {(prob > 0.5).mean():.4f}",
                    flush=True,
                )
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
