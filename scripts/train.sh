#!/bin/bash
#SBATCH --job-name=birdclef-htsat
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

# ── Download pretrained checkpoint if not present ──
CKPT_DIR="$PROJECT_DIR/checkpoints"
CKPT_PATH="$CKPT_DIR/HTSAT_AudioSet_Saved_1.ckpt"
mkdir -p "$CKPT_DIR"

if [ ! -f "$CKPT_PATH" ]; then
    echo "Downloading AudioSet pretrained HTSAT checkpoint..."
    pip install -q gdown
    gdown "https://drive.google.com/uc?id=1OK8a5XuMVLyeVKF117L8pfxeZYdfSDZv" \
        -O "$CKPT_PATH" || echo "WARNING: Checkpoint download failed. Will train from scratch."
fi

# ── Ensemble seed (override via environment: SEED=123 sbatch train.sh) ──
SEED="${SEED:-42}"

# ── Run training ──
JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${JOB_ID}_seed${SEED}"

python src/train.py \
    --data_dir "$PROJECT_DIR/data" \
    --checkpoint "$CKPT_PATH" \
    --batch_size 128 \
    --num_workers 8 \
    --max_epochs 40 \
    --lr 1e-4 \
    --save_dir "$CKPT_DIR" \
    --seed "$SEED" \
    --soundscape_weight 3.0 \
    --use_wandb \
    --wandb_project birdclef-2026 \
    --run_name "htsat-${RUN_ID}" \
    --run_id "$RUN_ID"

echo "Training complete (run=$RUN_ID)."
