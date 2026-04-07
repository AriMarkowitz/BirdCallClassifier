#!/bin/bash
#SBATCH --job-name=nmf_analysis
#SBATCH --output=nmf_analysis/slurm-%j.out
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1

# Run from project root: sbatch nmf_analysis/run_pipeline.sh

set -euo pipefail

PROJECT_DIR="$HOME/BirdCallClassifier"
ENV_NAME="birdcallclassifier"

module purge
module load miniconda3/latest

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

cd "$PROJECT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Python: $(which python)"
echo ""

echo "=== Step 1: Build spectrogram matrix ==="
python nmf_analysis/build_spectrogram_matrix.py \
    --data-dir data \
    --output-dir nmf_analysis/output \
    --min-per-species 10 \
    --max-per-species 50 \
    --n-soundscape-files 66

echo ""
echo "=== Step 2: Run NMFk rank selection ==="
python nmf_analysis/run_nmfk.py \
    --input-dir nmf_analysis/output \
    --k-min 20 \
    --k-max 80 \
    --k-step 5 \
    --n-runs 10 \
    --algo hals \
    --perturb-std 0.01

echo ""
echo "=== Step 3: Project all clips into NMF basis ==="
python nmf_analysis/project_clips.py \
    --data-dir data \
    --nmf-dir nmf_analysis/output \
    --output-dir nmf_analysis/output \
    --source all
