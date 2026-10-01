> **Frozen exploratory code.** The pipeline that came out of this
> (risc-mi MS -> evidence -> extract -> fit-radius -> label) is the `vesselseg`
> package in `../src/vesselseg`, see `../README.md`; the copies of
> build_evidence.py, extract_tree.py, fit_radius.py, label_centerlines.py,
> radius_error.py and register_to_t1.py here are the pre-package versions.
> The `pixi run [-e env]` lines below predate the move: the legacy manifest is
> now `pixi.legacy.toml`, so call the env interpreters directly
> (`../../.pixi/envs/<env>/bin/python`, from `experiments/scripts`); data and
> results paths are still relative to the repo root.

# vesselseg — exploratory artery segmentation on non-angiographic MRI (sub-01)

Scratch repo testing whether the main cerebral arteries can be recovered from
proteoflow's sub-01 structural scans (no TOF/MRA/CTA; the Gd was intrathecal).
Inputs are read (never written) from `/global/D1/homes/mariusca/proteoflow`.

Models compared:

| model | source | trained on | notes |
|---|---|---|---|
| vesselFM (`base`, `all`) | github.com/bwittmann/vesselFM, HF `bwittmann/vesselFM` | synthetic + angiography/microscopy | zero-shot; weights RAIL++-M, research only |
| risc-mi MC, MS | github.com/risc-mi/cerebral-artery-segmentation | IXI/TubeTK T1/T2/PD/FLAIR with TOF pseudo-labels | nnU-Net v2; needs RAI orientation |
| anat2vessels t1, t2, t1t2 | github.com/nrdg/anat2vessels, HF `huggingbrain/AnatomicalVesselSeg` | T1/T2 with vessel labels | nnU-Net v2 ResEnc-L; rigid to package reference |

## Setup

```bash
pixi install -a                       # envs: default (prep/eval), vesselfm, nnunet, vmtk
git clone https://github.com/bwittmann/vesselFM external/vesselFM
git clone https://github.com/risc-mi/cerebral-artery-segmentation external/risc-mi
git clone https://github.com/nrdg/anat2vessels external/anat2vessels
```

Weights go in `weights/` (git-ignored):
- vesselFM: `hf_hub_download('bwittmann/vesselFM', f, local_dir='weights/vesselfm')`
- risc-mi: `gdown --folder` of the two Google Drive links in the risc-mi READMEs, into `weights/riscmi/MC` and `weights/riscmi/MA_MS`
- anat2vessels: **not installed yet.** Install the t2 and t1t2 models into `$nnUNet_results`
  (set by the nnunet env to `weights/nnunet_results`), e.g.
  `pixi run -e nnunet python -c "from anat2vessels.data.fetch import ensure_model_installed as e; e('t2'); e('t1t2')"`.
  The sha256 pinned upstream for `t1_model.zip` no longer matches the file on HF
  (the HF LFS oid is `bd93742f…`), so the t1-only model fails pooch's check; decide
  whether to trust it before installing.

## Run

```bash
cd experiments/scripts
pixi run python prep_inputs.py                        # crops, isotropic grids, RAI copies, check PNGs
pixi run python register_to_t1.py                     # rigid t2/flair/t1gd -> T1w (series are NOT co-registered)
pixi run -e vesselfm python run_vesselfm.py           # or sbatch slurm/vesselfm_sweep.sbatch
pixi run -e nnunet python run_riscmi.py               # or sbatch slurm/riscmi.sbatch
pixi run -e nnunet python run_anat2vessels.py         # once the models are installed
pixi run python evaluate.py                           # results/sub-01/eval/{metrics,dice,checklist}.csv
pixi run python build_network.py                      # results/sub-01/network/network.{vti,nii.gz}: one connected MS consensus tree
pixi run python build_network.py --posterior          # network_posterior.{vti,nii.gz,json}: + strongest-contrast vessels behind the brainstem (label 5)
pixi run python build_network.py --posterior --t2 --t2_low 0.005 --t2_shell 0 --t2_all --trim 0.5  # network_posterior_t2.*: + T2w vessels grown from prob > 0.5 down to 0.005 (label 6), trimmed to half-max width (min 0.75 mm radius)
pixi run python grow_network.py                       # + network_grown.vti: extends it above z=+40 mm with T2w/post-Gd vesselness (labels 3 grown, 4 likely vein)
pixi run python grow_network.py --posterior           # network_grown_posterior.*: same, from network_posterior, and also behind the brainstem with T2w-only vesselness
pixi run -e vmtk python extract_centerlines.py         # centerlines.vtp (+ radius) and centerlines_tubes.vtp from network.nii.gz, touching vessels cut apart, weak loops broken (~3 min; --reuse redoes only the graph step)
pixi run -e vmtk python extract_centerlines.py --network network_posterior_t2 --t2_prob  # centerlines_posterior_t2{,_tubes}.vtp (~5 min)
pixi run python fit_radius.py --name centerlines_tree_posterior_t2  # after extract_tree.py, before label_centerlines.py: fits the radius over the whole tree (Huber + chain smoothing + Murray's law), in place (raw kept as `radius_raw`), regenerates NAME_tubes.vtp, writes NAME_radius_fit.json (~7 s)
pixi run python label_centerlines.py --name centerlines_posterior_t2  # NAME_labeled.{vtp,json,png}, NAME_tubes_labeled.vtp: ICA/VB/CoW/ACA/MCA/PCA/cerebellar per branch, automatic (sasparc prior + tree DP, ~20 s)
pixi run python plot_mips.py ../results/sub-01/figs/x.png ../results/sub-01/riscmi/MC/t1_prob.nii.gz ...
```

GPU notes (ex3): `a40q` nodes are aarch64 (the x86 pixi envs don't run there);
`a100q`/n013 handed out jobs with no visible GPU, so the sbatch scripts use
`hgx2q` and abort if `nvidia-smi -L` sees nothing. The login-node T4 is enough
for everything here.
