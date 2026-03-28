#!/bin/bash
# Launch BirdSet EfficientNet training for all folds.
# Each fold runs as a single SLURM job.
#
# Usage:
#   bash scripts/train_full_pipeline_ensemble.sh
#   MAX_EPOCHS=50 bash scripts/train_full_pipeline_ensemble.sh

N_FOLDS="${N_FOLDS:-4}"

for fold in $(seq 0 $((N_FOLDS - 1))); do
    echo "Submitting fold=$fold ..."
    FOLD=$fold sbatch scripts/train_full_pipeline.sh
done

echo "All $N_FOLDS pipeline jobs submitted."
