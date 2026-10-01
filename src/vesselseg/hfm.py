"""The HamiltonFastMarching FileHFM executable used for the curvature-penalised bridges.

`agd` (PyPI) is only the Python front end; its CPU solver is a compiled
FileHFM_<model> executable, built here for the one model the tree extraction
uses (ReedsSheppForward3). Lookup order:

  1. an explicit directory (extract --hfm_bin)
  2. $VESSELSEG_HFM_BIN (the directory, or the executable itself)
  3. $PIXI_PROJECT_ROOT/.vesselseg/hfm/build  (pixi workspaces)
  4. ~/.cache/vesselseg/hfm/build

`vesselseg build-hfm` builds into 3 (inside a pixi workspace) or 4. It needs
cmake and a C++17 compiler, and the HFM sources: a git clone (online), or
--src pointing at a copied source directory or .tar.gz (offline).
"""

import argparse
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

EXE = "FileHFM_ReedsSheppForward3"
REPO = "https://github.com/Mirebeau/HamiltonFastMarching.git"
REV = "420533c742d46f1d17145fa31b8167529be7182d"  # the revision the pipeline was tuned with


def default_root():
    root = os.environ.get("PIXI_PROJECT_ROOT")
    if root:
        return Path(root) / ".vesselseg" / "hfm"
    return Path.home() / ".cache" / "vesselseg" / "hfm"


def candidates(explicit=None):
    out = []
    if explicit:
        out.append(Path(explicit))
    env = os.environ.get("VESSELSEG_HFM_BIN")
    if env:
        env = Path(env)
        out.append(env.parent if env.name == EXE else env)
    if os.environ.get("PIXI_PROJECT_ROOT"):
        out.append(Path(os.environ["PIXI_PROJECT_ROOT"]) / ".vesselseg" / "hfm" / "build")
    out.append(Path.home() / ".cache" / "vesselseg" / "hfm" / "build")
    return out


def find_hfm(explicit=None):
    """Directory holding the FileHFM_ReedsSheppForward3 executable."""
    tried = candidates(explicit)
    for d in tried:
        if (d / EXE).is_file() and os.access(d / EXE, os.X_OK):
            return d.resolve()
    raise FileNotFoundError(
        f"{EXE} not found in " + ", ".join(map(str, tried)) + ". Run `vesselseg build-hfm` "
        "(offline: `vesselseg build-hfm --src <HamiltonFastMarching source dir or .tar.gz>`)"
        " or set VESSELSEG_HFM_BIN."
    )


def _sources(src, tmp):
    """Source directory: a given directory, an unpacked tarball, or a git clone."""
    if src is None:
        if shutil.which("git") is None:
            raise RuntimeError("no git: pass --src with a copy of the HFM sources")
        dst = Path(tmp) / "HamiltonFastMarching"
        subprocess.run(["git", "clone", "--quiet", REPO, str(dst)], check=True)
        subprocess.run(["git", "-C", str(dst), "checkout", "--quiet", REV], check=True)
        return dst
    src = Path(src)
    if src.is_dir():
        return src
    with tarfile.open(src) as tf:
        tf.extractall(tmp, filter="data")
    tops = [p for p in Path(tmp).iterdir() if p.is_dir()]
    if len(tops) != 1:
        raise RuntimeError(f"{src}: expected one top-level directory")
    return tops[0]


def build_hfm(src=None, root=None, jobs=8):
    """Configure and build FileHFM_ReedsSheppForward3; returns its directory."""
    root = Path(root) if root else default_root()
    build = root / "build"
    with tempfile.TemporaryDirectory() as tmp:
        source = _sources(src, tmp)
        cmake_src = source / "Interfaces" / "FileHFM"
        if not cmake_src.is_dir():
            raise RuntimeError(f"{source} is not a HamiltonFastMarching checkout")
        # the build tree points at its sources: keep a copy next to it
        kept = root / "src"
        if kept.resolve() != source.resolve():
            shutil.rmtree(kept, ignore_errors=True)
            shutil.copytree(source, kept, ignore=shutil.ignore_patterns(".git", "build"))
        shutil.rmtree(build, ignore_errors=True)
    cmd = [
        "cmake",
        "-S",
        str(kept / "Interfaces" / "FileHFM"),
        "-B",
        str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DIncludeStandardModels=OFF",
        "-DIncludeExperimentalModels=OFF",
        "-DCustomModelNames=ReedsSheppForward3",
    ]
    subprocess.run(cmd, check=True)
    subprocess.run(["cmake", "--build", str(build), "-j", str(jobs)], check=True)
    if not (build / EXE).is_file():
        raise RuntimeError(f"build finished without {build / EXE}")
    print(f"built {build / EXE}")
    return build


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="vesselseg build-hfm",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--src", type=Path, help="HFM source directory or .tar.gz (offline)")
    p.add_argument("--root", type=Path, help=f"install root (default: {default_root()})")
    p.add_argument("-j", "--jobs", type=int, default=8)
    args = p.parse_args(argv)
    build_hfm(args.src, args.root, args.jobs)


if __name__ == "__main__":
    main()
