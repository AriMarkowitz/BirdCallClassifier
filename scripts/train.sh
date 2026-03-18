#!/bin/bash
#SBATCH --job-name=birdclef-htsat
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/birdclef_%j.out
#SBATCH --error=logs/birdclef_%j.err

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
    # HTSAT-tiny AudioSet checkpoint from the original repo
    gdown "https://drive.google.com/uc?id=1HE9KQBGLFWMZ0CwG_USbpMc7o4lEzRiN" \
        -O "$CKPT_PATH" || echo "WARNING: Checkpoint download failed. Will train from scratch."
fi

# ── Run training ──
python src/train.py \
    --data_dir "$PROJECT_DIR/data" \
    --checkpoint "$CKPT_PATH" \
    --batch_size 32 \
    --num_workers 8 \
    --max_epochs 30 \
    --lr 1e-4 \
    --warmup_epochs 1 \
    --gpus 1 \
    --save_dir "$CKPT_DIR" \
    --seed 42

echo "Training complete."
