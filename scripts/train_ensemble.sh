#!/bin/bash
# Launch 5-fold CV ensemble training.
# Each job trains on 4/5 of soundscape data, validates on the remaining 1/5.
# Usage: bash scripts/train_ensemble.sh

N_FOLDS=5

for fold in $(seq 0 $((N_FOLDS - 1))); do
    echo "Submitting fold=$fold ..."
    FOLD=$fold sbatch scripts/train.sh
done

echo "All $N_FOLDS fold jobs submitted."
