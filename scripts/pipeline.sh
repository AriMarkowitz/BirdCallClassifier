#!/bin/bash
#SBATCH --job-name=birdclef-pipeline
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/pipeline_%j.log
#SBATCH --error=logs/pipeline_%j.log

# ── Multi-iteration self-training pipeline (Babych'25 1st-place) ──
#
# Each iteration: pseudo-label with current teacher → retrain student → promote best
# student as next teacher. Default 3 iterations.
#
# Usage:
#   # Multi-iteration self-training from existing baseline checkpoint:
#   CKPT=checkpoints/xxx/birdclef-birdset-epoch=25-val_macro_auc=0.9686.ckpt \
#     sbatch scripts/pipeline.sh
#
#   # Full pipeline (train baseline from scratch → N iterations of self-training):
#   sbatch scripts/pipeline.sh
#
#   # Reuse existing iter1 pseudo-labels (skip Step 2 of iteration 1 only):
#   CKPT=path/to/ckpt SKIP_PSEUDO=1 sbatch scripts/pipeline.sh
#
#   # Customize iterations / threshold / LR:
#   CKPT=path/to/ckpt N_ITERATIONS=4 THRESHOLD=0.7 RETRAIN_LR=5e-5 sbatch scripts/pipeline.sh

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
DISTILL_MANIFEST="${DISTILL_MANIFEST:-$PROJECT_DIR/data/distill_manifest.csv}"
HARD_NEGATIVES="${HARD_NEGATIVES:-1}"
MAX_PER_SPECIES="${MAX_PER_SPECIES:-500}"
SKIP_PSEUDO="${SKIP_PSEUDO:-0}"  # set to 1 to reuse existing pseudo_labels.csv/.npz
PSEUDO_DISTILL_WEIGHT="${PSEUDO_DISTILL_WEIGHT:-1.0}"
PSEUDO_POWER_T="${PSEUDO_POWER_T:-2.0}"  # T>1 sharpens pseudo soft labels; suppresses mid-range noise
PSEUDO_MIXUP_ALPHA="${PSEUDO_MIXUP_ALPHA:-0.4}"  # cross-domain MixUp pseudo↔labeled (Babych'25)
RETRAIN_EPOCHS="${RETRAIN_EPOCHS:-20}"
RETRAIN_LR="${RETRAIN_LR:-2e-5}"  # 10x lower than baseline; warm-start was destabilizing at 2e-4
N_ITERATIONS="${N_ITERATIONS:-3}"  # Babych'25 ran 4; gains taper after 3

VALID_REGIONS="$PROJECT_DIR/data/valid_regions.json"
PSEUDO_CSV="$PROJECT_DIR/data/pseudo_labels.csv"

HARD_NEG_ARG="--hard_negatives"
if [ "$HARD_NEGATIVES" = "0" ]; then
    HARD_NEG_ARG="--no_hard_negatives"
fi

DISTILL_MANIFEST_ARG=""
if [ -n "$DISTILL_MANIFEST" ] && [ -f "$DISTILL_MANIFEST" ]; then
    DISTILL_MANIFEST_ARG="--distill_manifest $DISTILL_MANIFEST"
fi

echo "=== Pipeline: pseudo-label → retrain ==="
echo "Job: $JOB_ID, Fold: $FOLD, Seed: $SEED, Threshold: $THRESHOLD, MaxPerSpecies: $MAX_PER_SPECIES"
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
        --batch_size 64 \
        --num_workers 8 \
        --max_epochs 40 \
        --lr 1.5e-4 \
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
        --no_distill \
        --balance_alpha 0.5 \
        --use_wandb \
        --wandb_project birdclef-2026 \
        --run_name "enet-${RUN_ID_R1}" \
        --run_id "$RUN_ID_R1" \
        --valid_regions "$VALID_REGIONS" \
        $DISTILL_MANIFEST_ARG \
        $HARD_NEG_ARG

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

# ── Self-training loop: N iterations of (pseudo-label → retrain) ──
# Each iteration uses the previous iteration's best checkpoint as teacher.
# Babych'25 1st-place ran 4 iterations; gains: 0.909 → 0.918 → 0.927 → 0.930.

CURRENT_CKPT="$CKPT"
echo ""
echo "=== Self-training loop: $N_ITERATIONS iterations ==="
echo "Initial teacher: $CURRENT_CKPT"

for ITER in $(seq 1 "$N_ITERATIONS"); do
    echo ""
    echo "════════════════════════════════════════════════"
    echo "  Iteration $ITER / $N_ITERATIONS"
    echo "  Teacher checkpoint: $CURRENT_CKPT"
    echo "════════════════════════════════════════════════"

    ITER_PSEUDO_CSV="${PSEUDO_CSV%.csv}_iter${ITER}.csv"
    RUN_ID_ITER="${JOB_ID}_fold${FOLD}_iter${ITER}"

    # ── Step 2: Pseudo-label with current teacher ──
    # SKIP_PSEUDO only applies to iteration 1 (reuse the canonical pseudo_labels.csv).
    if [ "$ITER" = "1" ] && [ "$SKIP_PSEUDO" = "1" ] \
        && [ -f "$PSEUDO_CSV" ] && [ -f "${PSEUDO_CSV%.csv}.npz" ]; then
        echo ""
        echo "--- Step 2 (iter $ITER): Reusing existing $PSEUDO_CSV (SKIP_PSEUDO=1) ---"
        cp "$PSEUDO_CSV" "$ITER_PSEUDO_CSV"
        cp "${PSEUDO_CSV%.csv}.npz" "${ITER_PSEUDO_CSV%.csv}.npz"
        N_PSEUDO=$(tail -n +2 "$ITER_PSEUDO_CSV" 2>/dev/null | wc -l)
    else
        echo ""
        echo "--- Step 2 (iter $ITER): Pseudo-labeling (threshold=$THRESHOLD) ---"
        python scripts/pseudo_label.py \
            --checkpoint "$CURRENT_CKPT" \
            --data-dir "$PROJECT_DIR/data" \
            --output "$ITER_PSEUDO_CSV" \
            --threshold "$THRESHOLD" \
            --max-per-species "$MAX_PER_SPECIES" \
            --match-train-distribution \
            --batch-size 64 \
            --use_wandb \
            --wandb_project birdclef-2026 \
            --run_name "pseudo-label-${RUN_ID_ITER}"

        N_PSEUDO=$(tail -n +2 "$ITER_PSEUDO_CSV" 2>/dev/null | wc -l)
    fi
    echo "Pseudo-labeled segments (iter $ITER): $N_PSEUDO"

    if [ "$N_PSEUDO" -eq 0 ]; then
        echo "WARNING: No pseudo-labels generated at iteration $ITER. Stopping loop."
        break
    fi

    # ── Step 3: Retrain (warm-started from current teacher) ──
    echo ""
    echo "--- Step 3 (iter $ITER): Retraining (run=$RUN_ID_ITER) ---"
    echo "  Warm-starting from: $CURRENT_CKPT"
    echo "  Retrain LR: $RETRAIN_LR, Epochs: $RETRAIN_EPOCHS"
    echo "  Pseudo-distill weight: $PSEUDO_DISTILL_WEIGHT"
    echo "  Pseudo power-T: $PSEUDO_POWER_T, mixup α: $PSEUDO_MIXUP_ALPHA"

    python src/train.py \
        --data_dir "$PROJECT_DIR/data" \
        --batch_size 64 \
        --num_workers 8 \
        --max_epochs "$RETRAIN_EPOCHS" \
        --lr "$RETRAIN_LR" \
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
        --no_distill \
        --balance_alpha 0.5 \
        --use_wandb \
        --wandb_project birdclef-2026 \
        --run_name "enet-${RUN_ID_ITER}" \
        --run_id "$RUN_ID_ITER" \
        --valid_regions "$VALID_REGIONS" \
        --pseudo_labels "$ITER_PSEUDO_CSV" \
        --pseudo_distill_weight "$PSEUDO_DISTILL_WEIGHT" \
        --pseudo_power_t "$PSEUDO_POWER_T" \
        --pseudo_mixup_alpha "$PSEUDO_MIXUP_ALPHA" \
        --warmstart "$CURRENT_CKPT" \
        $DISTILL_MANIFEST_ARG \
        $HARD_NEG_ARG

    # ── Promote best checkpoint from this iteration as next teacher ──
    # Pick the ckpt with highest val_macro_auc (parsed from filename).
    NEXT_CKPT=$(ls "$CKPT_DIR/$RUN_ID_ITER"/birdclef-birdset-*.ckpt 2>/dev/null \
        | awk -F'val_macro_auc=' '{print $2"\t"$0}' \
        | sort -k1,1 -gr \
        | head -1 | cut -f2-)
    if [ -z "$NEXT_CKPT" ]; then
        echo "ERROR: No checkpoint found in $CKPT_DIR/$RUN_ID_ITER/ — stopping loop"
        break
    fi
    echo ""
    echo "Iteration $ITER complete. Best ckpt: $NEXT_CKPT"
    CURRENT_CKPT="$NEXT_CKPT"
done

echo ""
echo "=== Pipeline complete ==="
echo "Initial teacher: $CKPT"
echo "Final teacher:   $CURRENT_CKPT"
