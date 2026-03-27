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
BACKBONE="${BACKBONE:-tf_efficientnet_b0_ns}"
MIN_DURATION="${MIN_DURATION:-3.0}"
MAX_DURATION="${MAX_DURATION:-30.0}"

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

python src/train.py \
    --data_dir "$PROJECT_DIR/data" \
    --backbone "$BACKBONE" \
    --batch_size 64 \
    --num_workers 8 \
    --max_epochs 25 \
    --lr 1e-4 \
    --save_dir "$CKPT_DIR" \
    --seed "$SEED" \
    --loss focal \
    --label_smoothing 0.1 \
    --n_folds 5 \
    --fold "$FOLD" \
    --mix_prob 0.3 \
    --balance_alpha 0.5 \
    --min_duration "$MIN_DURATION" \
    --max_duration "$MAX_DURATION" \
    --use_wandb \
    --wandb_project birdclef-2026 \
    --run_name "enet-${RUN_ID}" \
    --run_id "$RUN_ID" \
    --valid_regions "$VALID_REGIONS" \
    $RESUME_ARG \
    $PSEUDO_ARG

echo "Training complete (run=$RUN_ID, fold=$FOLD)."
