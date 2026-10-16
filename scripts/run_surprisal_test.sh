#!/bin/bash
#SBATCH --partition=gpu-single          # <-- SET ME (e.g. gpu)
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1                        # one GPU; size to your model (8B in bf16 needs ~20 GB)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


# NOTE: Slurm will not create the logs/ directory for you. Run `mkdir -p logs` once before
# submitting with:  sbatch run_surprisal_test.sh

set -euo pipefail

# ---- SET ME ------------------------------------------------------------------------------
PROJECT_DIR="~/candor/"                                   # folder containing the script + data, currently assumes that the job is submitted from the top level project directory
RESULTS_DIR="~/candor/results_llms"
INPUT_CSV="${PROJECT_DIR}/src/dense_subset1_and_subset2_processed.csv"
MODEL="meta-llama/Llama-3.1-8B-Instruct"                         # HF name or local path
# HF_HOME_DIR=""  set as needed                             # where model weights are stored
OUTPUT="${RESULTS_DIR}/surprisals_test_noTurn0.csv.gz"
N_TRANSCRIPTS=155
# Environment: pick one and edit/remove the other
# module load python/3.11 cuda/12.1
# source /path/to/venv/bin/activate
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate surprisal
# -------------------------------------------------------------------------------------------

cd "${PROJECT_DIR}/src"

# Load secrets and settings from .env (set -a exports everything defined while it is on)
ENV_FILE="${PROJECT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
    set -a
    source "${ENV_FILE}"
    set +a
else
    echo "Missing ${ENV_FILE}" >&2
    exit 1
fi

# fail early (without printing the token) if something required is missing
: "${HF_TOKEN:?HF_TOKEN is not set in .env}"
HF_HOME_DIR="${HF_HOME:-}"

# Load conda
module load devel/miniforge

conda deactivate
# Activate the conda environment
conda activate olmo_env

echo "Host: $(hostname) | Job: ${SLURM_JOB_ID} | Start: $(date)"

python conversation_surprisal.py \
    --input "${INPUT_CSV}" \
    --model "${MODEL}" \
    --output "${OUTPUT}" \
    --hf-home "${HF_HOME_DIR}" \
    --device cuda:0 \
    --dtype bfloat16 \
    --limit-transcripts ${N_TRANSCRIPTS} \
    --context both \
    --skip-turn-zero

echo "Done: $(date)"

# Quick sanity check of the output
python - <<EOF
import pandas as pd
d = pd.read_csv("${OUTPUT}")
print(d.groupby("transcript_id").agg(turns=("turn_id", "nunique"), tokens=("token", "size"),
                                      mean_surprisal=("surprisal", "mean")))
print(d.head())
EOF
