#!/bin/bash
#SBATCH --job-name=birdclef-mae
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/birdclef_%j.log
#SBATCH --error=logs/birdclef_%j.log

# Fine-tune Bird-MAE-Base on BirdCLEF 2026
# Usage:
#   sbatch scripts/train_birdmae.sh
#   FOLD=2 sbatch scripts/train_birdmae.sh

PROJECT_DIR="$HOME/BirdCallClassifier"
ENV_NAME="birdcallclassifier"

mkdir -p "$PROJECT_DIR/logs"

module purge
module load cuda/12.8.0-dh6b
module load miniconda3/latest

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Install safetensors if needed
pip install -q safetensors

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Model: Bird-MAE-Base"
echo "---"

cd "$PROJECT_DIR"

# Bird-MAE pretrained weights
CKPT_PATH="$PROJECT_DIR/checkpoints/models--DBD-research-group--Bird-MAE-Base/snapshots/6cc416d1a7ae2af29b6b866499b3b047a8f01304/model.safetensors"

if [ ! -f "$CKPT_PATH" ]; then
    echo "Downloading Bird-MAE-Base pretrained weights..."
    pip install -q huggingface_hub
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('DBD-research-group/Bird-MAE-Base', 'model.safetensors',
                cache_dir='$PROJECT_DIR/checkpoints')
"
fi

FOLD="${FOLD:-0}"
SEED="${SEED:-42}"
MAX_EPOCHS="${MAX_EPOCHS:-40}"

JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${JOB_ID}_mae_fold${FOLD}"

python src/train.py \
    --data_dir "$PROJECT_DIR/data" \
    --checkpoint "$CKPT_PATH" \
    --batch_size 64 \
    --num_workers 8 \
    --max_epochs "$MAX_EPOCHS" \
    --lr 1e-4 \
    --save_dir "$PROJECT_DIR/checkpoints" \
    --seed "$SEED" \
    --soundscape_weight 3.0 \
    --label_smoothing 0.1 \
    --n_folds 5 \
    --fold "$FOLD" \
    --use_wandb \
    --wandb_project birdclef-2026 \
    --run_name "birdmae-${RUN_ID}" \
    --run_id "$RUN_ID"

echo "Training complete (run=$RUN_ID, fold=$FOLD)."
