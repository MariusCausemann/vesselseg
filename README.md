# vesselseg

Cerebral **artery trees from non-angiographic structural MRI**: no TOF, MRA or CTA,
only the T1w / T2w (and optionally post-contrast T1w) series that a brain-imaging
study already has. The output is one connected, labelled centerline tree with a
radius at every point, in the T1w scanner frame:

```
T1w, T2w, [T1w post-Gd ...]
   │  segment     risc-mi MS probability map per series (nnU-Net, in a container)
   │  register    rigid, each series -> T1w
   │  evidence    all maps on the native T2w grid; P = max(T2w, consensus of the rest)
   │  extract     candidate pieces -> one global selection (MILP) -> tree from the inlets
   │  fit-radius  radius as one regularised estimate over the tree (Murray's law)
   ▼  label       ICA-L/R, VB, CoW, ACA, MCA-L/R, PCA-L/R, cerebellar; blood pressure
centerlines_labeled.vtp + centerlines_labeled.json
   │  grow        (optional) synthetic distal branches through the SAS
   ▼
centerlines_labeled_grown.vtp
```

It was developed on one subject (sub-01: 0.33 mm T2w, 0.5 mm T1w, three
post-intrathecal-Gd T1w sessions), for building perivascular (PVS) meshes and
arterial blood-flow fields on the subject's own anatomy.

- [Install](#install)
- [Usage](#usage)
- [Method](#method)
- [Outputs](#outputs)
- [Running offline (air-gapped)](#running-offline-air-gapped)
- [Resources and runtime](#resources-and-runtime)
- [Limitations](#limitations)
- [Development](#development)
- [References and licences](#references-and-licences)

---

## Install

vesselseg is a pure Python package (`src/vesselseg`) with one console script,
`vesselseg`. One piece lives outside Python:

| piece | what | how it is found |
|---|---|---|
| risc-mi MS | nnU-Net model + weights, PyTorch | brainseg's `brainseg_riscmi.sif` (Apptainer) |

With [pixi](https://pixi.sh), `pixi install` in this repo gives the full env
(the `[tool.pixi]` tables of `pyproject.toml`). It expects
brainseg, which provides the
container lookup, checked out next to this repo (`../brainseg`). In any other
Python >= 3.11 env:

```bash
pip install -e ../brainseg -e .
vesselseg doctor                  # what is present, what is missing
```

HamiltonFastMarching comes from the `HFMpy` wheel (manylinux, no compiler
needed). To use vesselseg from another project, add it (and brainseg) as a
dependency of that project's env, e.g. as a pixi path or git dependency.

**The risc-mi container** comes from brainseg (`dockerfiles/Dockerfile.riscmi`).
It is private because the weights are proprietary (see below), so log in first:

```bash
apptainer registry login --username <github-user> docker://ghcr.io
apptainer build ~/.brainseg_containers/brainseg_riscmi.sif \
    docker://ghcr.io/mariuscausemann/brainseg-riscmi:latest
```

It is looked up in `--container`, `./`, `./.containers/`, `$BRAINSEG_CONTAINER_DIR`
and `~/.brainseg_containers/`, in that order. vesselseg never pulls it.

## Usage

Each step is a subcommand; `vesselseg <step> --help` lists all options.

```bash
# 1. probability maps (one call per series)
vesselseg segment -i T2w.nii.gz        -o work/riscmi/t2_prob.nii.gz  [--gpu] [--no_tta]
vesselseg segment -i T1w.nii.gz        -o work/riscmi/t1_prob.nii.gz
vesselseg segment -i ses-02_T1w.nii.gz -o work/riscmi/t1gd_prob.nii.gz

# 2. rigid registration of every other series onto the T1w
vesselseg register --fixed T1w.nii.gz --moving t2=T2w.nii.gz t1gd=ses-02_T1w.nii.gz -o work/reg

# 3. evidence on the T2w grid (seg: any brain segmentation in T1w space, e.g. FreeSurfer)
vesselseg evidence --t2_prob work/riscmi/t2_prob.nii.gz \
    --probs t1=work/riscmi/t1_prob.nii.gz t1gd=work/riscmi/t1gd_prob.nii.gz \
    --reg work/reg --t1 T1w.nii.gz --seg seg.nii.gz -o work/evidence

# 4.-6. the tree (sasparc: Desikan-Killiany + aseg labels extended into the SAS)
vesselseg extract --evidence work/evidence --sasparc sasparc.nii.gz -o work/tree/centerlines --workers 64
vesselseg fit-radius work/tree/centerlines
vesselseg label work/tree/centerlines --sasparc sasparc.nii.gz

# 7. optional: synthetic distal branches down to the cortex
vesselseg grow work/tree/centerlines_labeled --sasparc sasparc.nii.gz --seg seg.nii.gz \
    --evidence work/riscmi/*_prob.nii.gz
```

Series keys: `t1` is the fixed T1w and `t2` the grid of the tree. Any other key
is a further series. Keys that differ only in a trailing number after `gd`
(`t1gd`, `t1gd3`, `t1gd4`) count as one contrast in the consensus.

**Steps 1-6 from one YAML file** (skips steps whose outputs exist):

```bash
vesselseg run --config vessels.yml     # format: vesselseg run --help
```

**In a workflow manager** (Snakemake, Nextflow, ...), call the steps one by one,
one job per step and per series for `segment`. Each step reads only the files
written by the steps before it. `vesselseg.pipeline.to_argv` turns an option
dict into flags (`{workers: 8}` → `--workers 8`), so the per-step options can
live in the workflow's config.

## Method

### 1. Vessel probability: risc-mi MS

The MS model of [risc-mi](https://github.com/risc-mi/cerebral-artery-segmentation)
(Alshenoudy et al., MIUA 2024) is an nnU-Net v2 network trained to find arteries
on structural MRI (T1w, T2w, PD, FLAIR), using labels from TOF angiography of
the same subjects. The model is Dataset606_MRIVessels, `nnUNetTrainerDA5`,
3d_fullres. Inference ensembles 5 folds × 8 mirror flips. Each input is
reoriented to RAI and cast to float32 first, as in training. The probability map
stays on that RAI grid, and for the T2w this is the grid of everything after it.
Predictions run per series. The T2w is the sharpest of them and the only one that
shows the posterior vessels, but about 75% of both ICA trunks are invisible in
it, because the flow void merges with bone and dura. The T1w and post-Gd T1w
maps cover those.

### 2. Registration and evidence

The series need not share a world frame. In sub-01 the T2w is about 5 mm off
the T1w and a later session's T1w is 13-16 mm off. Each series is therefore
registered rigidly onto the T1w (ANTs). Each map is pulled onto a crop of the native T2w
grid through the transform chain
$x_k = M_k\,M_{t2}^{-1}\,x_{t2}$ (RAS mm). Here $M_k$ maps T1w points to
series $k$, and the ANTs LPS matrices are converted with
$M_\mathrm{RAS} = L\,M_\mathrm{LPS}\,L$, $L=\mathrm{diag}(-1,-1,1,1)$. The
evidence is

$$
P(x) = \max\bigl(P_{t2}(x),\ \overline{P}_\mathrm{other}(x)\bigr),
$$

where $\overline{P}_\mathrm{other}$ averages over contrasts (post-Gd sessions are
averaged first), and $P=0$ outside the brain+CSF segmentation dilated by 20 mm.

### 3. Candidates

- **Candidate mask:** a hysteresis threshold on $P$, seeded at $P \ge 0.5$ and
  grown into $P \ge 0.01$, keeping components of at least 30 voxels. It is
  permissive on purpose, because the selection decides what is a vessel.
- **Radius:** the distance to the boundary of a stricter half-maximum mask
  ($P \ge \tfrac12 \max_{0.5\,\mathrm{mm}} P$), read on that mask's own ridge
  within 2.5 mm of the skeleton. It is floored at $h/2$, where $h$ is the voxel size.
- **Candidate segments:** the voxel skeleton split at its junctions. Junction
  clusters closer than their radius are merged, so a crossing becomes one node.
- **Bridges:** from every free end, curvature-penalised minimal paths in a
  6 mm box. They use the Reeds-Shepp forward model of HamiltonFastMarching on an
  isotropic 0.5 mm grid, seeded with the end's direction, through the metric
  $1+\max(0,-w(P))$. The two cheapest real shortcuts per end become extra
  candidates. A free end of a skeleton piece that holds no inlet also searches
  15 mm (`--max_gap_far`), to the other pieces only, and keeps its four cheapest
  paths. Otherwise a piece behind a gap longer than 6 mm is dropped, or is
  hooked onto whatever happens to lie within 6 mm. In sub-01 that was a left
  PCA branch feeding the right medial occipital branches across the falx.

The evidence weight per mm of centerline is

$$
w(P) = \mathrm{clip}\bigl(\log(P/p_0),\,-2,\,1\bigr), \qquad p_0 = 0.005 .
$$

### 4. Global selection (MILP)

After Türetken et al. (TPAMI 2016), every candidate segment can be used in
either direction (an arc). Every used arc has exactly one parent arc ending at
its start node, or it leaves an inlet. A flow from the inlets, of which each arc
consumes its length, rules out cycles, so the result is a forest grown from the
inlets. The program minimises

$$
\sum_{a\ \mathrm{used}} \bigl(\alpha - R_a\bigr)
+ \sum_{\mathrm{bridges}} (\beta + \mathrm{excess} + w_\mathrm{falx}\,[\mathrm{across\ the\ falx}])
+ \sum_{a\to b} \Bigl[ w_\mathrm{turn}\,\tfrac{1-\cos\theta_{ab}}{2}
+ g\bigl(\log\tfrac{r_b}{r_a} - \log r_\mathrm{tol}\bigr) \Bigr]
+ w_\mathrm{cap}\,\rho\,\bigl|\,\ell_a - c_0 r_a^3\,\bigr|_\mathrm{slack}
$$

where the terms are:

- $R_a=\int_a w(P)\,ds$ is the evidence reward (it leaves out the junction bulge,
  so spurs inside a thick vessel earn nothing).
- A bridge is **across the falx** when its two ends lie next to the cortex of
  opposite hemispheres (sasparc labels within 2 mm, at least 80% one side). No
  cortical artery is fed from the other hemisphere. Only bridges are tested,
  because the A2s run side by side in the fissure. $w_\mathrm{falx}=20$
  mm-equivalents is a cost, not a ban.
- $\theta_{ab}$ is the turn angle at a junction.
- $g$ is the radius-growth cost of a child thicker than its parent.
- The last term is a **Murray capacity** band. The centerline length $\ell_a$
  that arc $a$ feeds should scale with $r_a^3$ (Murray 1926). This is what keeps
  the ICA feeding the MCA through a 90° T-junction rather than the thin ACom
  through the straight A1-M1 line.

$g$ and the band width come from one **radius measurement model**:

$$
\log r_\mathrm{obs} = \log r + \varepsilon,\quad
\sigma(r)^2 = \sigma_0^2\Bigl(1+\frac{h^2}{r^2}\Bigr),\quad
\text{valid with probability } \rho = P .
$$

A radius near the voxel size, or at faint evidence, says little, so the cost of
a mismatch is bounded by $-\log(1-\rho)$.

**Solve.** HiGHS first solves the program without the capacity terms, which is
fast. That solution is then improved by a **region search**: 40 mm boxes, each
solved exactly with every binary outside it fixed, run in parallel and merged
until no box improves. This local optimum is deterministic, while the full MIP
does not close its gap in reasonable time.

**Inlets.** These are found on the T1w-world tree, using the mid-sagittal plane
between the sasparc cortex centroids:
- The vertebrals are the thickest free ends where the head ROI cuts them.
- The ICAs are the lowest upward-heading free ends per side with a 90th
  percentile radius of at least 1.3 mm.

Short thin dead ends (< 3 mm, radius < 0.25 mm) are pruned, and the inlet trees
are joined where they meet.

### 5. Radius fit

The extracted radius is a noisy distance-to-boundary: it pinches at contacts
and collapses at tips. With edge unknowns $x_e = \log r_e$ and samples $y_e$:

$$
J(x) = \sum_e w_e\,\rho_\mathrm{Huber}(x_e-y_e)
+ \tfrac12\sum_{e\sim f}\frac{\lambda^2}{ds_{ef}}(x_e-x_f)^2
+ \tfrac12\sum_j \mu\Bigl(x_p - \tfrac1k\log\sum_{d}e^{k x_d}\Bigr)^2
$$

- $w_e = l_e\,P_e\,r^2/(r^2+h^2)$, the edge length times its reliability under
  the measurement model.
- $e\sim f$ are consecutive edges of one chain (no smoothing across junctions).
- The last term is Murray's law $r_p^k=\sum_d r_d^k$, with $k=3$, at every
  junction $j$, oriented towards the nearest inlet.

It is solved by Gauss-Newton on the IRLS form. The raw radius is kept as
`radius_raw`, so a rerun is idempotent. $h$ defaults to the mean spacing of
the evidence grid.

### 6. Territory labels

There is no atlas. The only anatomical input is the subject's own parcellation
(sasparc), and labelling runs in four steps:

1. **Inlets:** taken from `inlets_xyz` of the extraction.
2. **Trunks:** from each ICA inlet, and from the vertebral confluence, follow the
   thickest continuation to the first balanced split. That split has two
   children of comparable radius heading to opposite sides, each feeding at
   least 20 mm of centerline next to its target cortex. This gives the ICA
   terminus and the basilar tip.
3. **Circle of Willis:** the remaining backbone between the inlets (pieces of at
   most 60 mm).
4. **Territories:** every other branch gets ACA, MCA-L/R, PCA-L/R, cerebellar or
   unknown from an **exact min-sum labelling of the tree** (dynamic programming).
   - The unary cost is the length of centerline next to cortex that does not
     allow the class.
   - The pairwise costs penalise a territory leaving its trunk lightly, and
     implausible switches (MCA → ACA, a PCA below the basilar tip, ...) heavily.
     The junctions that pay a switch are reported as suspected false contacts.

**Loops and blood pressure.** The selection can come back to a node its own
path already left (a real loop, or a false contact). Extraction keeps the
output a tree by storing that node twice, as a copy offset by 0.03 mm, and
records these pairs (`loop_returns`, and every duplicated node in
`node_copies`). The labelling then solves a steady Poiseuille flow on the
network, with conductance $G_e = 1/\int_e r^{-4}\,ds$, pressure 1 at the inlets
and 0 at every other tip. A loop inside one territory is re-cut where its two
flows meet (the local pressure minimum), not wherever the selection happened to
close it. The final pressure is written as `blood_pressure`, which orients
every branch for downstream flow models.

With `--rejoin_copies` (opt-in), each implausible junction is also cut, against
the node copy that feeds its piece from a territory the transition costs allow.
Then these copies go through the same re-cut. Only copies where a branch ends
count, because two passes through one node may be a crossing. The result has
the same vessels and labels and is still a tree. In sub-01 this moves 131 mm of
PCA-R from behind a faint cerebellar branch back onto the PCA-R.

### 7. Synthetic distal branches (optional)

The extracted tree stops near the image resolution (tip radius ~0.4 mm), so
much of the cortical SAS lies 10-25 mm from any vessel. `vesselseg grow` adds
the missing distal branches as a synthetic forest, in the spirit of constrained
constructive optimisation (CCO), but as least-cost paths on the voxel graph of
the SAS:

- **Paths.** They hug the pia and follow faint evidence where there is some.
  Edges across the falx and the tentorium are cut.
- **Terminals.** One terminal is placed per `--spacing` mm along the pia. They
  are shared among the tips of the extracted tree in proportion to $r^3$
  (Murray's law), as a weighted geodesic Voronoi partition.
- **Smoothing and radius.** The forest is smoothed by Laplacian steps that stay
  in the SAS. Radii follow Murray's law from each tip's own radius down to
  `--r_tip`.

Synthetic chains inherit `segment` and `inlet` from their host and are marked
`synthetic = 1`. The blood pressure is solved again over the whole network.
They are a plausible filling, not a measurement.

## Outputs

Everything is in **T1w world coordinates** (RAS mm, the scanner frame of the T1w).

`NAME_labeled.vtp`, one polyline per unbranched branch:

| array | kind | meaning |
|---|---|---|
| `radius` | point | fitted lumen radius (mm) |
| `radius_raw` | point | radius as extracted |
| `evidence` | point | $P$ at the point |
| `node_type` | point | 2 at the source (lower vertebral) |
| `blood_pressure` | point | Poiseuille pressure, 1 at the inlets, 0 at the other tips |
| `segment` | point + cell | label index into `segment_names` |
| `branch`, `length`, `mean_radius` | cell | branch id, mm, mm |
| `origin` | cell | 1 = contains a bridge, 2 = joins two inlet trees |
| `inlet` | cell | inlet tree (index into `inlet_names`), −1 for joins |
| `segment_names` | field | `unknown, ICA-L, ICA-R, VB, CoW, ACA, MCA-L, MCA-R, PCA-L, PCA-R, cerebellar` |
| `inlet_names` | field | names of the inlet trees |
| `loop_returns`, `node_copies` | field | point pairs (copy, original) of the nodes stored twice |

A node stored twice sits 0.03 mm from its original. A consumer that merges
coincident points (to within ~1e-6 mm) therefore keeps the tree a tree.

`NAME_labeled.json` contains:
- `inlets`: `VA` is a list of points and `ICA-L` / `ICA-R` are single points,
  the blood inlets of a flow model.
- `landmarks`: the ICA termini, VA confluence and basilar tip.
- `cow_pieces`, the per-label `segments` lengths, `implausible_junctions`, and
  `loops` (loops re-cut, plus contacts swapped or kept with `--rejoin_copies`).

Each step also writes the following files:

| step | file | contents |
|---|---|---|
| extract | `NAME.vtp`, `NAME_tubes.vtp` | the unlabelled tree and its tube surface |
| extract | `NAME.json` | inputs, timings, the objective and region-search stats, `inlets_xyz`, `spacing_mm` |
| extract | `NAME_candidates.pkl`, `NAME_selection.pkl` | caches for `--reuse` / `--reuse_selection` |
| fit-radius | `NAME_radius_fit.json` | fit statistics (rewrites `NAME.vtp` in place) |
| label | `NAME_tubes_labeled.vtp` | tubes with `segment` |
| label | `NAME_labeled.png` | three projections coloured by label |
| grow | `NAME_labeled_grown.vtp`, `_tubes.vtp`, `.json`, `.png` | the tree plus synthetic chains (`synthetic`, `territory`), stats per label and per tip |

## Running offline (air-gapped)

Nothing in the pipeline downloads at run time. Three things have to be carried
over from a connected machine:

1. **The container**:
   ```bash
   apptainer build brainseg_riscmi.sif docker://ghcr.io/mariuscausemann/brainseg-riscmi:latest
   ```
   It holds PyTorch, nnU-Net and the 5 fold checkpoints (stripped to inference
   keys). Copy it to `$BRAINSEG_CONTAINER_DIR` (or `~/.brainseg_containers/`) on
   the target. An Apptainer runtime is part of the pixi env.
2. **The Python env**: pack the conda and PyPI-wheel part of the env (this
   repo's, or that of a project that depends on vesselseg) with
   [pixi-pack](https://github.com/Quantco/pixi-pack). It does not pack source,
   path or git dependencies, so install those afterwards from copies of the
   repos (vesselseg, brainseg, and any of the host project's own).
   ```bash
   pixi exec pixi-pack --ignore-pypi-non-wheel -o environment.tar .
   # target, next to the copied repos:
   pixi exec pixi-unpack environment.tar && source activate.sh
   for r in vesselseg brainseg; do
       pip install --no-deps --no-build-isolation -e "$r"
   done
   ```
3. **HamiltonFastMarching**: the `HFMpy` wheel is part of the pixi-pack'd env, nothing
   to build or copy.

Then `vesselseg doctor` checks everything: the Python modules, that HFMpy imports,
the container runtime, and that the `.sif` is found.

## Resources and runtime

On sub-01 (native T2w grid 0.35 × 0.33 × 0.33 mm):

| step | resources | time |
|---|---|---|
| segment, CPU | 128 cores, ~110 GB | ~80 min per series |
| segment, GPU | A100 (`--gpu`) | ~5 min per series |
| segment `--no_tta` | | 8× faster, Dice ~0.92 against TTA |
| register | 8 cores | minutes per series |
| evidence | 8 cores | ~3 min |
| extract | 64 cores (genoaxq/milanq) | selection ~8 min (region search), plus candidates and bridges |
| fit-radius | 1 core | ~10 s |
| label | 1 core | ~20 s |

The region search waits for its slowest box, so prefer fast cores. The env is
built for linux-64 (x86) only.

## Limitations

- **Arteries only.** risc-mi learned from TOF labels. Veins and sinuses are
  not targeted, but the dark-vessel contrasts pick up some of them.
- **Where there is no evidence, there is no tree.** In sub-01 the posterior falx
  and the cistern of Galen have no centerlines. The occipital cortical PCA
  branches are sparse too (~70 / 150 mm of candidates, against ~1550 mm frontal
  per side): there is ~4x less evidence per ml there in the T2w, none in the
  pre-contrast T1w, and the post-Gd sessions disagree (Dice 0.2 against 0.6-0.7
  frontally). Thresholds and the consensus are not what lose them.
- **A tree has no loops.** The Circle of Willis is cut somewhere: one A1 or
  PCom ends as a dead end. Loops inside one territory are cut where the blood
  flows meet, but the cut between territories follows the selection.
- **Radius below ~0.3 mm is not measured.** A vessel there is 1-2 voxels wide,
  and its radius comes from Murray's law rather than the data. A radius jump of
  more than 3× at a junction is usually a segmentation gap, not a topology error.
- **Tuned on one subject.** All defaults were tuned on sub-01. The steps all
  check their inputs, but check `NAME_labeled.png` and the `implausible_junctions`
  of a new subject.

## Development

```
src/vesselseg/        the package (cli, segment, register, evidence, pipeline, io)
src/vesselseg/tree/   extract (MILP), radius_model, fit_radius, label, blood
                      (Poiseuille pressure), grow
tests/                unit tests, synthetic data only (pixi run test)
experiments/          the exploratory scripts this grew from (vesselFM, risc-mi MC,
                      anat2vessels, VMTK, TubeTK, SeqSeg, the T1w-grid network,
                      sub-01 benchmarks), frozen with their own legacy pixi manifest
```

```bash
pixi run test     # pytest
pixi run lint     # ruff check + ruff format --check
```

## References and licences

- **risc-mi:** Alshenoudy, A., Sabrowsky-Hirsch, B., Scharinger, J., Thumfart, S.,
  Giretzlehner, M. (2024). *Towards Segmenting Cerebral Arteries from Structural
  MRI.* MIUA 2024, Springer Nature.
  - The model weights are proprietary to **RISC Software GmbH** and licensed for
    research only. Do not publish the container image.
- **Selection MILP:** Türetken, E., Benmansour, F., Andres, B., Głowacki, P.,
  Pfister, H., Fua, P. (2016). *Reconstructing curvilinear networks using path
  classifiers and integer programming.* IEEE TPAMI 38(12).
- **HamiltonFastMarching:** Mirebeau, J.-M., Portegies, J. (2019). *Hamiltonian
  Fast Marching: a numerical solver for anisotropic and non-holonomic eikonal
  PDEs.* IPOL 9.
- **Murray's law:** Murray, C. D. (1926). *The physiological principle of minimum
  work.* PNAS 12(3).
- **Solver:** [HiGHS](https://highs.dev) (MIT).
