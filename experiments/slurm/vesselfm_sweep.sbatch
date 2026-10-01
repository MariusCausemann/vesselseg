#!/bin/bash
#SBATCH --job-name=vesselfm
#SBATCH --partition=hgx2q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
nvidia-smi -L || { echo "no GPU visible in job"; exit 1; }
cd /global/D1/homes/mariusca/vesselseg/experiments/scripts
pixi run -e vesselfm python run_vesselfm.py "$@"
