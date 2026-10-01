"""`vesselseg <step> ...`: one subcommand per pipeline step (each has its own --help)."""

import importlib
import sys

COMMANDS = {
    "segment": (
        "vesselseg.segment",
        "risc-mi MS probability map of one series (container)",
    ),
    "register": ("vesselseg.register", "rigid registration of the series onto the T1w"),
    "evidence": ("vesselseg.evidence", "evidence maps on the native T2w grid"),
    "extract": ("vesselseg.tree.extract", "arterial tree by global selection (MILP)"),
    "fit-radius": (
        "vesselseg.tree.fit_radius",
        "radius fit over the tree (Murray's law)",
    ),
    "label": (
        "vesselseg.tree.label",
        "territory labels (ICA, VB, CoW, ACA, MCA, PCA, ...)",
    ),
    "run": ("vesselseg.pipeline", "all steps from a YAML config"),
    "doctor": ("vesselseg.cli", "check that everything needed offline is present"),
}


def usage():
    from vesselseg import __version__

    lines = [f"vesselseg {__version__}\n", "usage: vesselseg <command> [options]\n"]
    lines += [f"  {name:<11} {help_}" for name, (_, help_) in COMMANDS.items()]
    lines.append("\nvesselseg <command> --help for the options of each.")
    return "\n".join(lines)


def doctor(argv=None):
    """Report what the pipeline needs and whether this machine has it; exit 1 if not."""
    import argparse
    import shutil

    p = argparse.ArgumentParser(prog="vesselseg doctor", description=doctor.__doc__)
    p.add_argument("--container", help="brainseg_riscmi.sif to check instead of the lookup")
    p.add_argument("--no_segment", action="store_true", help="skip the container checks")
    args = p.parse_args(argv)
    ok = True

    def report(name, good, detail):
        nonlocal ok
        ok &= good
        print(f"[{'ok' if good else 'MISSING'}] {name}: {detail}")

    for mod in (
        "highspy",
        "agd",
        "ants",
        "SimpleITK",
        "pyvista",
        "networkx",
        "skimage",
    ):
        try:
            importlib.import_module(mod)
            report(f"python {mod}", True, "importable")
        except ImportError as e:
            report(f"python {mod}", False, str(e))

    try:
        hfmpy = importlib.import_module("HFMpy.HFM_ReedsSheppForward3")
        report("HFM", True, hfmpy.__file__)
    except ImportError as e:
        report("HFM", False, f"{e} (pip install HFMpy)")

    if not args.no_segment:
        runtime = shutil.which("apptainer") or shutil.which("singularity")
        report(
            "container runtime",
            runtime is not None,
            runtime or "no apptainer/singularity",
        )
        try:
            from vesselseg.segment import find_riscmi_container

            report("risc-mi container", True, str(find_riscmi_container(args.container)))
        except (FileNotFoundError, ImportError) as e:
            report("risc-mi container", False, str(e))
    sys.exit(0 if ok else 1)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return
    if argv[0] in ("-V", "--version"):
        from vesselseg import __version__

        print(__version__)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        sys.exit(f"vesselseg: unknown command {cmd!r}\n\n{usage()}")
    if cmd == "doctor":
        return doctor(rest)
    return importlib.import_module(COMMANDS[cmd][0]).main(rest)


if __name__ == "__main__":
    main()
