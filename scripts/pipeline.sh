#!/bin/bash
#SBATCH --job-name=birdclef-pipeline
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pipeline_%j.log
#SBATCH --error=logs/pipeline_%j.log

# ── Pseudo-labeling pipeline ──
#
# Usage:
#   # Pseudo-label from existing checkpoint, then retrain:
#   CKPT=checkpoints/xxx/birdclef-birdset-epoch=38-val_macro_auc=0.8877.ckpt sbatch scripts/pipeline.sh
#
#   # Full pipeline (train from scratch → pseudo-label → retrain):
#   sbatch scripts/pipeline.sh
#
#   # Customize threshold/fold:
#   CKPT=path/to/best.ckpt THRESHOLD=0.7 FOLD=2 sbatch scripts/pipeline.sh

PROJECT_DIR="$HOME/BirdCallClassifier"
ENV_NAME="birdcallclassifier"
CKPT_DIR="$PROJECT_DIR/checkpoints"

mkdir -p "$PROJECT_DIR/logs"

# ── Load modules ──
module purge
module load cuda/12.8.0-dh6b
module load miniconda3/latest

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

cd "$PROJECT_DIR"

# ── Config (override via env vars) ──
FOLD="${FOLD:-0}"
SEED="${SEED:-42}"
THRESHOLD="${THRESHOLD:-0.8}"
CKPT="${CKPT:-}"  # path to existing checkpoint; if empty, trains from scratch first
JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
MIN_DURATION="${MIN_DURATION:-5.0}"
MAX_DURATION="${MAX_DURATION:-5.0}"

VALID_REGIONS="$PROJECT_DIR/data/valid_regions.json"
PSEUDO_CSV="$PROJECT_DIR/data/pseudo_labels.csv"

echo "=== Pipeline: pseudo-label → retrain ==="
echo "Job: $JOB_ID, Fold: $FOLD, Seed: $SEED, Threshold: $THRESHOLD"
echo "Checkpoint: ${CKPT:-'(will train from scratch first)'}"
echo "---"

# ── Preprocess: detect active vocal regions (cached) ──
if [ ! -f "$VALID_REGIONS" ]; then
    echo "Step 0: Detecting active vocal regions..."
    python scripts/preprocess_activity.py \
        --data-dir "$PROJECT_DIR/data" \
        --output "$VALID_REGIONS"
fi

# ── Step 1: Get a checkpoint (train if none provided) ──
if [ -z "$CKPT" ]; then
    RUN_ID_R1="${JOB_ID}_fold${FOLD}_r1"
    echo ""
    echo "=== Step 1: Training from scratch (run=$RUN_ID_R1) ==="

    python src/train.py \
        --data_dir "$PROJECT_DIR/data" \
        --batch_size 16 \
        --num_workers 8 \
        --max_epochs 40 \
        --lr 5e-5 \
        --precision bf16 \
        --save_dir "$CKPT_DIR" \
        --seed "$SEED" \
        --loss focal \
        --label_smoothing 0.05 \
        --n_folds 5 \
        --fold "$FOLD" \
        --min_duration "$MIN_DURATION" \
        --max_duration "$MAX_DURATION" \
        --full_files \
        --distill \
        --balance_alpha 0.5 \
        --use_wandb \
        --wandb_project birdclef-2026 \
        --run_name "enet-${RUN_ID_R1}" \
        --run_id "$RUN_ID_R1" \
        --valid_regions "$VALID_REGIONS"

    # Find best checkpoint from round 1
    CKPT=$(ls -t "$CKPT_DIR/$RUN_ID_R1"/birdclef-birdset-*.ckpt 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "ERROR: No checkpoint found in $CKPT_DIR/$RUN_ID_R1/"
        exit 1
    fi
    echo "Round 1 complete. Best checkpoint: $CKPT"
else
    echo ""
    echo "=== Step 1: Using provided checkpoint: $CKPT ==="
fi

# ── Step 2: Pseudo-label ──
echo ""
echo "=== Step 2: Pseudo-labeling (threshold=$THRESHOLD) ==="

python scripts/pseudo_label.py \
    --checkpoint "$CKPT" \
    --data-dir "$PROJECT_DIR/data" \
    --output "$PSEUDO_CSV" \
    --threshold "$THRESHOLD" \
    --batch-size 64 \
    --use_wandb \
    --wandb_project birdclef-2026 \
    --run_name "pseudo-label-${JOB_ID}_fold${FOLD}"

N_PSEUDO=$(tail -n +2 "$PSEUDO_CSV" 2>/dev/null | wc -l)
echo "Pseudo-labeled segments: $N_PSEUDO"

if [ "$N_PSEUDO" -eq 0 ]; then
    echo "WARNING: No pseudo-labels generated. Skipping retrain."
    exit 0
fi

# ── Step 3: Retrain with pseudo-labels ──
RUN_ID_R2="${JOB_ID}_fold${FOLD}_r2"
echo ""
echo "=== Step 3: Retraining with pseudo-labels (run=$RUN_ID_R2) ==="

python src/train.py \
    --data_dir "$PROJECT_DIR/data" \
    --batch_size 16 \
    --num_workers 8 \
    --max_epochs 40 \
    --lr 5e-5 \
    --precision bf16 \
    --save_dir "$CKPT_DIR" \
    --seed "$SEED" \
    --loss focal \
    --label_smoothing 0.05 \
    --n_folds 5 \
    --fold "$FOLD" \
    --min_duration "$MIN_DURATION" \
    --max_duration "$MAX_DURATION" \
    --full_files \
    --distill \
    --balance_alpha 0.5 \
    --use_wandb \
    --wandb_project birdclef-2026 \
    --run_name "enet-${RUN_ID_R2}" \
    --run_id "$RUN_ID_R2" \
    --valid_regions "$VALID_REGIONS" \
    --pseudo_labels "$PSEUDO_CSV"

echo ""
echo "=== Pipeline complete ==="
echo "Source checkpoint: $CKPT"
echo "Pseudo-labels: $PSEUDO_CSV ($N_PSEUDO segments)"
echo "Retrained checkpoints: $CKPT_DIR/$RUN_ID_R2/"
