#!/bin/bash
#SBATCH --job-name=birdclef-s2
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/birdclef_%j.log
#SBATCH --error=logs/birdclef_%j.log

# Stage 2: Train temporal MLP on frozen HTSAT backbone.
# Requires a stage 1 checkpoint.
#
# Usage:
#   STAGE1_CKPT=checkpoints/309752_fold0/birdclef-htsat-epoch=37-val_macro_auc=0.9996.ckpt \
#     sbatch scripts/train_stage2.sh
#
#   # Or run all folds:
#   bash scripts/train_stage2_ensemble.sh

# ── Project paths ──
PROJECT_DIR="$HOME/BirdCallClassifier"
ENV_NAME="birdcallclassifier"

# ── Create log directory ──
mkdir -p "$PROJECT_DIR/logs"

# ── Load modules ──
module purge
module load cuda/12.8.0-dh6b
module load miniconda3/latest

# ── Activate conda ──
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# ── Print environment info ──
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Stage: 2 (temporal MLP)"
echo "---"

cd "$PROJECT_DIR"

# ── Config ──
FOLD="${FOLD:-0}"
SEED="${SEED:-42}"
STAGE1_CKPT="${STAGE1_CKPT:?ERROR: STAGE1_CKPT must be set (path to stage 1 checkpoint)}"

JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${JOB_ID}_s2_fold${FOLD}"

echo "Stage 1 checkpoint: $STAGE1_CKPT"
echo "Fold: $FOLD"

python src/train.py \
    --data_dir "$PROJECT_DIR/data" \
    --stage 2 \
    --stage1_ckpt "$STAGE1_CKPT" \
    --batch_size 128 \
    --num_workers 8 \
    --max_epochs 20 \
    --lr 1e-3 \
    --save_dir "$PROJECT_DIR/checkpoints" \
    --seed "$SEED" \
    --label_smoothing 0.1 \
    --n_folds 5 \
    --fold "$FOLD" \
    --use_wandb \
    --wandb_project birdclef-2026 \
    --run_name "htsat-s2-${RUN_ID}" \
    --run_id "$RUN_ID"

echo "Stage 2 training complete (run=$RUN_ID, fold=$FOLD)."
