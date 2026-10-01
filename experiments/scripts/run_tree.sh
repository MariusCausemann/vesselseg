#!/bin/bash
# Full tree run on a compute node: extract_tree.py -> fit_radius.py ->
# label_centerlines.py -> tree_bench.py check (genoaxq: fastest cores; the region search waits on its slowest box).
# usage: srun -p genoaxq -c 64 --time 60 bash run_tree.sh NAME [extract_tree options]
#   (--reuse reads results/sub-01/network/NAME_candidates.pkl: copy a frozen set there)
set -e
cd "$(dirname "$0")"
N=$1
shift
pixi run python extract_tree.py --out "$N" "$@"
pixi run python fit_radius.py --name "$N" --h_mm 0.34
pixi run python label_centerlines.py --name "$N"
pixi run python tree_bench.py check "$N"
