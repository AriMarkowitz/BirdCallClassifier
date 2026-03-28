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

# Full pipeline: BirdSet EfficientNet training + optional pseudo-labeling retrain
# Runs in a single SLURM job per fold.
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

# Reduce allocator fragmentation on long variable-length batches
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ── Preprocess: detect active vocal regions (one-time, cached) ──
VALID_REGIONS="$PROJECT_DIR/data/valid_regions.json"
if [ ! -f "$VALID_REGIONS" ]; then
    echo "Detecting active vocal regions in train_audio (one-time preprocessing)..."
    python scripts/preprocess_activity.py \
        --data-dir "$PROJECT_DIR/data" \
        --output "$VALID_REGIONS"
fi

# ── Config ──
FOLD="${FOLD:-0}"
SEED="${SEED:-42}"
MAX_EPOCHS="${MAX_EPOCHS:-40}"
BIRDSET_MODEL="${BIRDSET_MODEL:-DBD-research-group/EfficientNet-B1-BirdSet-XCL}"

JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${JOB_ID}_fold${FOLD}"

echo "========================================="
echo "  BirdSet EfficientNet training: fold=$FOLD"
echo "  Epochs: ${MAX_EPOCHS}"
echo "  Backbone: ${BIRDSET_MODEL}"
echo "========================================="

# ══════════════════════════════════════════════════════════════════════════════
# Train BirdSet EfficientNet
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Training BirdSet EfficientNet ==="

python src/train.py \
    --data_dir "$PROJECT_DIR/data" \
    --birdset_model_name "$BIRDSET_MODEL" \
    --batch_size 16 \
    --num_workers 8 \
    --max_epochs "$MAX_EPOCHS" \
    --lr 5e-5 \
    --precision bf16 \
    --save_dir "$CKPT_DIR" \
    --seed "$SEED" \
    --loss focal \
    --label_smoothing 0.05 \
    --n_folds 5 \
    --fold "$FOLD" \
    --min_duration 5.0 \
    --max_duration 5.0 \
    --no_full_files \
    --distill \
    --distill_weight 0.15 \
    --distill_temperature 2.0 \
    --balance_alpha 0.5 \
    --use_wandb \
    --wandb_project birdclef-2026 \
    --run_name "enet-${RUN_ID}" \
    --run_id "$RUN_ID" \
    --valid_regions "$VALID_REGIONS"

echo ""
echo "========================================="
echo "  Training complete!"
echo "  Run ID: $RUN_ID"
echo "  Fold: $FOLD"
echo "========================================="
