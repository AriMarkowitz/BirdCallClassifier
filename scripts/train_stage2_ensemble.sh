#!/bin/bash
# Launch stage 2 (temporal MLP) training for all 5 folds.
# Automatically finds the best stage 1 checkpoint per fold.
#
# Usage:
#   bash scripts/train_stage2_ensemble.sh <fold0_job> <fold1_job> <fold2_job> <fold3_job> <fold4_job>
#   Example:
#   bash scripts/train_stage2_ensemble.sh 309752_fold0 309753_fold1 309754_fold2 309755_fold3 309756_fold4

set -e

PROJECT_DIR="$HOME/BirdCallClassifier"
N_FOLDS=5

if [ $# -ne $N_FOLDS ]; then
    echo "Usage: $0 <fold0_run_id> <fold1_run_id> ... <fold4_run_id>"
    echo ""
    echo "Available checkpoint dirs:"
    ls -d "$PROJECT_DIR"/checkpoints/*/ 2>/dev/null | xargs -I{} basename {}
    exit 1
fi

for fold in $(seq 0 $((N_FOLDS - 1))); do
    run_id="${!((fold + 1))}"  # positional arg for this fold
    run_dir="$PROJECT_DIR/checkpoints/$run_id"

    if [ ! -d "$run_dir" ]; then
        echo "ERROR: $run_dir not found"
        exit 1
    fi

    # Pick best checkpoint by val_macro_auc
    best=$(ls "$run_dir"/birdclef-htsat-*.ckpt 2>/dev/null \
        | sed 's/.*val_macro_auc[=_]\([0-9.]*\).*/\1 &/' \
        | sort -rn | head -1 | cut -d' ' -f2)

    if [ -z "$best" ]; then
        echo "ERROR: No checkpoints in $run_dir"
        exit 1
    fi

    echo "Fold $fold: $best"
    FOLD=$fold STAGE1_CKPT="$best" sbatch scripts/train_stage2.sh
done

echo "All $N_FOLDS stage 2 jobs submitted."
