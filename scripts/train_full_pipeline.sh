#!/bin/bash
#SBATCH --job-name=birdclef-full
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/birdclef_%j.log
#SBATCH --error=logs/birdclef_%j.log

# Full pipeline: Stage 1 (HTSAT + zeroed temporal MLP) → Stage 2 (temporal MLP only)
# Runs both stages in a single SLURM job per fold.
#
# Usage:
#   sbatch scripts/train_full_pipeline.sh                    # fold 0
#   FOLD=2 sbatch scripts/train_full_pipeline.sh             # fold 2
#   bash scripts/train_full_pipeline_ensemble.sh              # all 5 folds

set -e

# ── Project paths ──
PROJECT_DIR="$HOME/BirdCallClassifier"
ENV_NAME="birdcallclassifier"
CKPT_DIR="$PROJECT_DIR/checkpoints"

# ── Create log directory ──
mkdir -p "$PROJECT_DIR/logs"

# ── Load modules ──
module purge
module load cuda/12.8.0-dh6b
module load miniconda3/latest

# ── Activate conda ──
eval "$(conda shell.bash hook)"

if ! conda env list | grep -q "$ENV_NAME"; then
    echo "Creating conda environment: $ENV_NAME"
    conda create -n "$ENV_NAME" python=3.10 -y
    conda activate "$ENV_NAME"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install -r "$PROJECT_DIR/requirements.txt"
else
    conda activate "$ENV_NAME"
fi

pip install -q wandb

# ── Print environment info ──
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "---"

cd "$PROJECT_DIR"

# ── Download pretrained checkpoint if not present ──
CKPT_PATH="$CKPT_DIR/HTSAT_AudioSet_Saved_1.ckpt"
mkdir -p "$CKPT_DIR"

if [ ! -f "$CKPT_PATH" ]; then
    echo "Downloading AudioSet pretrained HTSAT checkpoint..."
    pip install -q gdown
    gdown "https://drive.google.com/uc?id=1OK8a5XuMVLyeVKF117L8pfxeZYdfSDZv" \
        -O "$CKPT_PATH" || echo "WARNING: Checkpoint download failed. Will train from scratch."
fi

# ── Config ──
FOLD="${FOLD:-0}"
SEED="${SEED:-42}"
S1_EPOCHS="${S1_EPOCHS:-30}"
S2_EPOCHS="${S2_EPOCHS:-20}"

JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
S1_RUN_ID="${JOB_ID}_s1_fold${FOLD}"
S2_RUN_ID="${JOB_ID}_s2_fold${FOLD}"

echo "========================================="
echo "  Full pipeline: fold=$FOLD"
echo "  Stage 1: ${S1_EPOCHS} epochs (full model)"
echo "  Stage 2: ${S2_EPOCHS} epochs (temporal MLP)"
echo "========================================="

# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: Full model with zeroed temporal inputs
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== STAGE 1: Full model training ==="

python src/train.py \
    --data_dir "$PROJECT_DIR/data" \
    --checkpoint "$CKPT_PATH" \
    --batch_size 128 \
    --num_workers 8 \
    --max_epochs "$S1_EPOCHS" \
    --lr 1e-4 \
    --save_dir "$CKPT_DIR" \
    --seed "$SEED" \
    --soundscape_weight 3.0 \
    --label_smoothing 0.1 \
    --n_folds 5 \
    --fold "$FOLD" \
    --stage 1 \
    --use_wandb \
    --wandb_project birdclef-2026 \
    --run_name "htsat-${S1_RUN_ID}" \
    --run_id "$S1_RUN_ID"

echo "Stage 1 complete."

# ── Find best stage 1 checkpoint ──
S1_DIR="$CKPT_DIR/$S1_RUN_ID"
BEST_S1=$(ls "$S1_DIR"/birdclef-htsat-*.ckpt 2>/dev/null \
    | sed 's/.*val_macro_auc[=_]\([0-9.]*\).*/\1 &/' \
    | sort -rn | head -1 | cut -d' ' -f2)

if [ -z "$BEST_S1" ]; then
    echo "ERROR: No stage 1 checkpoints found in $S1_DIR"
    exit 1
fi
echo "Best stage 1 checkpoint: $BEST_S1"

# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: Train temporal MLP on soundscapes only
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== STAGE 2: Temporal MLP training ==="

python src/train.py \
    --data_dir "$PROJECT_DIR/data" \
    --batch_size 128 \
    --num_workers 8 \
    --max_epochs "$S2_EPOCHS" \
    --lr 1e-3 \
    --save_dir "$CKPT_DIR" \
    --seed "$SEED" \
    --label_smoothing 0.1 \
    --n_folds 5 \
    --fold "$FOLD" \
    --stage 2 \
    --stage1_ckpt "$BEST_S1" \
    --use_wandb \
    --wandb_project birdclef-2026 \
    --run_name "htsat-${S2_RUN_ID}" \
    --run_id "$S2_RUN_ID"

echo ""
echo "========================================="
echo "  Full pipeline complete!"
echo "  Stage 1: $S1_RUN_ID"
echo "  Stage 2: $S2_RUN_ID"
echo "  Fold: $FOLD"
echo "========================================="
