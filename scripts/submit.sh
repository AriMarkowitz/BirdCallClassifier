#!/bin/bash
#
# Upload best checkpoints to Kaggle, run inference notebook, and submit.
#
# Usage:
#   bash scripts/submit.sh                          # uses all checkpoints in kaggle_dataset/
#   bash scripts/submit.sh 309600_seed42 309601_seed123   # pick specific runs
#
# Requirements: kaggle CLI configured (kaggle.json or ~/.kaggle/kaggle.json)

set -e

PROJECT_DIR="$HOME/BirdCallClassifier"
KAGGLE_DS="$PROJECT_DIR/kaggle_dataset"
NOTEBOOK_DIR="$PROJECT_DIR/notebooks"
KERNEL_SLUG="arimarkowitz/birdclef-2026-inference"
COMPETITION="birdclef-2026"

# ── Step 1: Copy checkpoints ──────────────────────────────────────────────────
echo "=== Step 1: Preparing checkpoints ==="

# Clear old checkpoints
rm -f "$KAGGLE_DS"/birdclef-htsat-*.ckpt
echo "Cleared old checkpoints from kaggle_dataset/"

if [ $# -gt 0 ]; then
    # Specific run IDs provided as arguments
    for run_id in "$@"; do
        run_dir="$PROJECT_DIR/checkpoints/$run_id"
        if [ ! -d "$run_dir" ]; then
            echo "ERROR: $run_dir not found"
            exit 1
        fi
        # Pick checkpoint with highest val_macro_auc from filename
        best=$(ls "$run_dir"/birdclef-htsat-*.ckpt 2>/dev/null | sed 's/.*val_macro_auc[=_]\([0-9.]*\).*/\1 &/' | sort -rn | head -1 | cut -d' ' -f2)
        if [ -z "$best" ]; then
            echo "ERROR: No checkpoints in $run_dir"
            exit 1
        fi
        cp "$best" "$KAGGLE_DS/"
        echo "  Copied: $(basename "$best") (from $run_id)"
    done
else
    echo "No run IDs specified — using existing checkpoints in kaggle_dataset/"
    # Check there's at least one
    if ! ls "$KAGGLE_DS"/birdclef-htsat-*.ckpt &>/dev/null; then
        echo "ERROR: No checkpoints found in $KAGGLE_DS/"
        echo "Usage: bash scripts/submit.sh <run_id1> [run_id2] ..."
        echo "Available runs:"
        ls -d "$PROJECT_DIR"/checkpoints/*/ 2>/dev/null | xargs -I{} basename {}
        exit 1
    fi
fi

echo ""
echo "Checkpoints to upload:"
ls -lh "$KAGGLE_DS"/birdclef-htsat-*.ckpt
echo ""

# ── Step 2: Upload dataset ────────────────────────────────────────────────────
echo "=== Step 2: Uploading dataset to Kaggle ==="
CKPT_COUNT=$(ls "$KAGGLE_DS"/birdclef-htsat-*.ckpt | wc -l)
kaggle datasets version -p "$KAGGLE_DS" -m "Ensemble: ${CKPT_COUNT} checkpoints" --dir-mode zip
echo ""

# Wait for dataset to process
echo "Waiting for dataset to finish processing..."
for i in $(seq 1 30); do
    sleep 10
    STATUS=$(kaggle datasets status arimarkowitz/birdclef-2026-model 2>/dev/null || echo "unknown")
    if echo "$STATUS" | grep -qi "ready"; then
        echo "Dataset ready."
        break
    fi
    echo "  Still processing... (${i}/30)"
done
echo ""

# ── Step 3: Push and run notebook ─────────────────────────────────────────────
echo "=== Step 3: Pushing notebook (triggers run) ==="
kaggle kernels push -p "$NOTEBOOK_DIR"
echo ""

# Wait for notebook to complete
echo "Waiting for notebook to complete..."
for i in $(seq 1 60); do
    sleep 30
    STATUS=$(kaggle kernels status "$KERNEL_SLUG" 2>&1)
    echo "  [$i] $STATUS"
    if echo "$STATUS" | grep -qi "complete"; then
        echo "Notebook finished successfully."
        break
    fi
    if echo "$STATUS" | grep -qi "error\|cancel"; then
        echo "ERROR: Notebook failed. Check Kaggle for details."
        exit 1
    fi
done
echo ""

# ── Step 4: Download output and submit ────────────────────────────────────────
echo "=== Step 4: Downloading output and submitting ==="
SUBMISSION_DIR=$(mktemp -d)
kaggle kernels output "$KERNEL_SLUG" -p "$SUBMISSION_DIR"

SUBMISSION_FILE="$SUBMISSION_DIR/submission.csv"
if [ ! -f "$SUBMISSION_FILE" ]; then
    echo "ERROR: submission.csv not found in notebook output"
    ls "$SUBMISSION_DIR/"
    exit 1
fi

ROWS=$(wc -l < "$SUBMISSION_FILE")
echo "Submission has $ROWS rows"

kaggle competitions submit -c "$COMPETITION" -f "$SUBMISSION_FILE" -m "Ensemble: ${CKPT_COUNT} models"
echo ""
echo "=== Done! Check your score at: ==="
echo "https://www.kaggle.com/competitions/$COMPETITION/leaderboard"

# Cleanup
rm -rf "$SUBMISSION_DIR"
