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
# Round 1: Train from scratch (or resume)
# Round 2: Pseudo-label unlabeled soundscapes with round-1 checkpoint
# Round 3: Retrain with pseudo-labels added to training data

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
JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"

CKPT_PATH="$CKPT_DIR/HTSAT_AudioSet_Saved_1.ckpt"
VALID_REGIONS="$PROJECT_DIR/data/valid_regions.json"
PSEUDO_CSV="$PROJECT_DIR/data/pseudo_labels.csv"

echo "=== Pipeline: train → pseudo-label → retrain ==="
echo "Job: $JOB_ID, Fold: $FOLD, Seed: $SEED, Threshold: $THRESHOLD"
echo "---"

# ── Preprocess: detect active vocal regions (cached) ──
if [ ! -f "$VALID_REGIONS" ]; then
    echo "Step 0: Detecting active vocal regions..."
    python scripts/preprocess_activity.py \
        --data-dir "$PROJECT_DIR/data" \
        --output "$VALID_REGIONS"
fi

# ── Round 1: Initial training ──
RUN_ID_R1="${JOB_ID}_fold${FOLD}_r1"
echo ""
echo "=== Round 1: Initial training (run=$RUN_ID_R1) ==="

python src/train.py \
    --data_dir "$PROJECT_DIR/data" \
    --checkpoint "$CKPT_PATH" \
    --batch_size 128 \
    --num_workers 8 \
    --max_epochs 12 \
    --lr 1e-4 \
    --save_dir "$CKPT_DIR" \
    --seed "$SEED" \
    --loss focal \
    --label_smoothing 0.1 \
    --n_folds 5 \
    --fold "$FOLD" \
    --use_wandb \
    --wandb_project birdclef-2026 \
    --run_name "htsat-${RUN_ID_R1}" \
    --run_id "$RUN_ID_R1" \
    --valid_regions "$VALID_REGIONS"

echo "Round 1 complete."

# ── Find best checkpoint from round 1 ──
R1_BEST=$(ls -t "$CKPT_DIR/$RUN_ID_R1"/birdclef-htsat-*.ckpt 2>/dev/null | head -1)
if [ -z "$R1_BEST" ]; then
    echo "ERROR: No round-1 checkpoint found in $CKPT_DIR/$RUN_ID_R1/"
    exit 1
fi
echo "Best round-1 checkpoint: $R1_BEST"

# ── Pseudo-labeling ──
echo ""
echo "=== Pseudo-labeling unlabeled soundscapes (threshold=$THRESHOLD) ==="

python scripts/pseudo_label.py \
    --checkpoint "$R1_BEST" \
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
    echo "WARNING: No pseudo-labels generated. Skipping round 2."
    exit 0
fi

# ── Round 2: Retrain with pseudo-labels ──
RUN_ID_R2="${JOB_ID}_fold${FOLD}_r2"
echo ""
echo "=== Round 2: Retraining with pseudo-labels (run=$RUN_ID_R2) ==="

python src/train.py \
    --data_dir "$PROJECT_DIR/data" \
    --checkpoint "$CKPT_PATH" \
    --batch_size 128 \
    --num_workers 8 \
    --max_epochs 12 \
    --lr 1e-4 \
    --save_dir "$CKPT_DIR" \
    --seed "$SEED" \
    --loss focal \
    --label_smoothing 0.1 \
    --n_folds 5 \
    --fold "$FOLD" \
    --use_wandb \
    --wandb_project birdclef-2026 \
    --run_name "htsat-${RUN_ID_R2}" \
    --run_id "$RUN_ID_R2" \
    --valid_regions "$VALID_REGIONS" \
    --pseudo_labels "$PSEUDO_CSV"

echo ""
echo "=== Pipeline complete ==="
echo "Round 1 checkpoint: $R1_BEST"
echo "Pseudo-labels: $PSEUDO_CSV ($N_PSEUDO segments)"
echo "Round 2 checkpoints: $CKPT_DIR/$RUN_ID_R2/"
