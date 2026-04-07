#!/bin/bash
#SBATCH --job-name=birdclef-enet
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/birdclef_%j.log
#SBATCH --error=logs/birdclef_%j.log

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

# ── Create conda env if it doesn't exist ──
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "Creating conda environment: $ENV_NAME"
    conda create -n "$ENV_NAME" python=3.10 -y
    conda activate "$ENV_NAME"

    # PyTorch with CUDA 12.1
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # Remaining dependencies
    pip install -r "$PROJECT_DIR/requirements.txt"
else
    conda activate "$ENV_NAME"
fi

# Install wandb if not present
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

# ── Fold, seed, and resume (override via environment) ──
FOLD="${FOLD:-0}"
SEED="${SEED:-42}"
RESUME_FROM="${RESUME_FROM:-}"
PSEUDO_LABELS="${PSEUDO_LABELS:-}"
BIRDSET_MODEL="${BIRDSET_MODEL:-DBD-research-group/EfficientNet-B1-BirdSet-XCL}"
MIN_DURATION="${MIN_DURATION:-5.0}"
MAX_DURATION="${MAX_DURATION:-5.0}"
FULL_FILES="${FULL_FILES:-0}"
MIX_PROB="${MIX_PROB:-0.0}"
MIXUP_ALPHA="${MIXUP_ALPHA:-0.0}"
DISTILL="${DISTILL:-0}"
DISTILL_WEIGHT="${DISTILL_WEIGHT:-0.15}"
DISTILL_TEMP="${DISTILL_TEMP:-2.0}"
DISTILL_MANIFEST="${DISTILL_MANIFEST:-$PROJECT_DIR/data/distill_manifest.csv}"
HARD_NEGATIVES="${HARD_NEGATIVES:-1}"
BG_MIX_PROB="${BG_MIX_PROB:-0.5}"
BG_SNR_MIN="${BG_SNR_MIN:-3.0}"
BG_SNR_MAX="${BG_SNR_MAX:-15.0}"
MAX_TIME_FRAMES="${MAX_TIME_FRAMES:-768}"
CHUNK_HOP_FRAMES="${CHUNK_HOP_FRAMES:-512}"
PRECISION="${PRECISION:-bf16}"
BATCH_SIZE="${BATCH_SIZE:-16}"

# ── Run training ──
JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${JOB_ID}_fold${FOLD}"
CKPT_DIR="$PROJECT_DIR/checkpoints"

RESUME_ARG=""
if [ -n "$RESUME_FROM" ]; then
    echo "Resuming from checkpoint: $RESUME_FROM"
    RESUME_ARG="--resume_from $RESUME_FROM"
fi

PSEUDO_ARG=""
if [ -n "$PSEUDO_LABELS" ]; then
    echo "Using pseudo-labels: $PSEUDO_LABELS"
    PSEUDO_ARG="--pseudo_labels $PSEUDO_LABELS"
fi

FULL_FILES_ARG=""
if [ "$FULL_FILES" = "1" ]; then
    echo "Full-file mode enabled: training on complete train_audio recordings"
    FULL_FILES_ARG="--full_files"
fi

DISTILL_ARG="--no_distill"
if [ "$DISTILL" = "1" ]; then
    DISTILL_ARG="--distill"
fi

DISTILL_MANIFEST_ARG=""
if [ -n "$DISTILL_MANIFEST" ] && [ -f "$DISTILL_MANIFEST" ]; then
    echo "Using distill data: $DISTILL_MANIFEST"
    DISTILL_MANIFEST_ARG="--distill_manifest $DISTILL_MANIFEST"
fi

HARD_NEG_ARG="--hard_negatives"
if [ "$HARD_NEGATIVES" = "0" ]; then
    HARD_NEG_ARG="--no_hard_negatives"
fi

python src/train.py \
    --data_dir "$PROJECT_DIR/data" \
    --birdset_model_name "$BIRDSET_MODEL" \
    --batch_size "$BATCH_SIZE" \
    --num_workers 8 \
    --max_epochs 40 \
    --lr 5e-5 \
    --precision "$PRECISION" \
    --max_time_frames "$MAX_TIME_FRAMES" \
    --chunk_hop_frames "$CHUNK_HOP_FRAMES" \
    --save_dir "$CKPT_DIR" \
    --seed "$SEED" \
    --loss focal \
    --label_smoothing 0.05 \
    --n_folds 5 \
    --fold "$FOLD" \
    --mix_prob "$MIX_PROB" \
    --mixup_alpha "$MIXUP_ALPHA" \
    --distill_weight "$DISTILL_WEIGHT" \
    --distill_temperature "$DISTILL_TEMP" \
    $DISTILL_ARG \
    --balance_alpha 0.5 \
    --min_duration "$MIN_DURATION" \
    --max_duration "$MAX_DURATION" \
    --use_wandb \
    --wandb_project birdclef-2026 \
    --run_name "enet-${RUN_ID}" \
    --run_id "$RUN_ID" \
    --valid_regions "$VALID_REGIONS" \
    $FULL_FILES_ARG \
    $RESUME_ARG \
    $PSEUDO_ARG \
    $DISTILL_MANIFEST_ARG \
    $HARD_NEG_ARG \
    --bg_mix_prob "$BG_MIX_PROB" \
    --bg_snr_min "$BG_SNR_MIN" \
    --bg_snr_max "$BG_SNR_MAX"

echo "Training complete (run=$RUN_ID, fold=$FOLD)."
