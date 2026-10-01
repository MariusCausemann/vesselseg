"""The whole chain from the structural series to the labeled tree, from one config file.

  segment (each series) -> register -> evidence -> extract -> fit-radius -> label

A step is skipped when its outputs exist (delete them, or use --force, to
redo). For use inside a workflow manager, call the single steps instead (the
proteoflow Snakefile does).

Config (YAML):

  workdir: results/sub-01/vessels       # everything is written below this
  t1: data/sub-01_ses-01_T1w.nii.gz     # fixed series, T1w world = output frame
  t2: data/sub-01_ses-01_T2w.nii.gz     # the tree is extracted on this grid
  seg: results/sub-01/seg/sub-01_seg.nii.gz          # brain segmentation (T1w)
  sasparc: results/sub-01/seg/sub-01_sasparc.nii.gz  # DK + aseg parcellation (T1w)
  series:                               # optional further series KEY: PATH
    t1gd: data/sub-01_ses-02_T1w.nii.gz
  name: centerlines                     # tree file stem
  segment: {gpu: false, tta: true}      # `vesselseg segment` options
  extract: {workers: 64}                # `vesselseg extract` options
  fit_radius: {}
  label: {}

Option dicts map to command-line flags: {workers: 64} -> --workers 64,
{t2_only: true} -> --t2_only, {lns_box: [20, 40]} -> --lns_box 20 40.

usage: vesselseg run --config vessels.yml
"""

import argparse
from pathlib import Path


def to_argv(opts):
    argv = []
    for k, v in (opts or {}).items():
        flag = f"--{k}"
        if v is True:
            argv.append(flag)
        elif v is False or v is None:
            continue
        elif isinstance(v, (list, tuple)):
            argv += [flag, *map(str, v)]
        else:
            argv += [flag, str(v)]
    return argv


def run(cfg, force=False):
    from vesselseg import evidence, register, segment
    from vesselseg.tree import extract, fit_radius, label

    work = Path(cfg["workdir"])
    series = {"t1": Path(cfg["t1"]), "t2": Path(cfg["t2"])}
    series.update({k: Path(v) for k, v in (cfg.get("series") or {}).items()})
    if set(cfg.get("series") or {}) & {"t1", "t2"}:
        raise ValueError("series: t1 and t2 are given at the top level")
    todo = lambda *paths: force or not all(Path(p).exists() for p in paths)  # noqa: E731

    seg_opts = dict(cfg.get("segment") or {})
    probs = {}
    for k, path in series.items():
        probs[k] = work / "riscmi" / f"{k}_prob.nii.gz"
        if todo(probs[k]):
            segment.segment(
                path,
                probs[k],
                container=seg_opts.get("container"),
                gpu=seg_opts.get("gpu", False),
                tta=seg_opts.get("tta", True),
                folds=seg_opts.get("folds", "0 1 2 3 4"),
                workers=seg_opts.get("workers"),
            )

    reg = work / "reg"
    moving = {k: p for k, p in series.items() if k != "t1"}
    if todo(*(register.mat_path(reg, k) for k in moving)):
        register.register(series["t1"], moving, reg)

    evid = work / "evidence"
    if todo(evid / "evidence.json"):
        others = {k: p for k, p in probs.items() if k != "t2"}
        evidence.build_evidence(probs["t2"], others, reg, cfg["t1"], cfg["seg"], evid)

    stem = work / "tree" / cfg.get("name", "centerlines")
    sasparc = str(cfg["sasparc"])
    if todo(f"{stem}.vtp", f"{stem}.json"):
        extract.main(
            ["--evidence", str(evid), "--sasparc", sasparc, "-o", str(stem)]
            + to_argv(cfg.get("extract"))
        )
    if todo(f"{stem}_radius_fit.json"):
        fit_radius.main([str(stem)] + to_argv(cfg.get("fit_radius")))
    if todo(f"{stem}_labeled.vtp", f"{stem}_labeled.json"):
        label.main([str(stem), "--sasparc", sasparc] + to_argv(cfg.get("label")))
    print(f"labeled tree: {stem}_labeled.vtp")
    return Path(f"{stem}_labeled.vtp")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="vesselseg run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--force", action="store_true", help="redo every step")
    args = p.parse_args(argv)
    import yaml

    run(yaml.safe_load(args.config.read_text()), force=args.force)


if __name__ == "__main__":
    main()
