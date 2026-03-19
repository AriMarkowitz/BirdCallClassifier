#!/bin/bash
# Launch 5 training jobs with different seeds for ensemble.
# Usage: bash scripts/train_ensemble.sh

SEEDS=(42 123 456 789 2026)

for seed in "${SEEDS[@]}"; do
    echo "Submitting seed=$seed ..."
    SEED=$seed sbatch scripts/train.sh
done

echo "All ${#SEEDS[@]} ensemble jobs submitted."
