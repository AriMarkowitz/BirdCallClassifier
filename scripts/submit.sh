#!/bin/bash
#
# Upload best checkpoints to Kaggle, run inference notebook, and submit.
#
# Usage:
#   bash scripts/submit.sh                          # uses all checkpoints in kaggle_dataset/
#   bash scripts/submit.sh 309600_seed42 309601_seed123   # pick best checkpoint per run
#   USE_LATEST=1 bash scripts/submit.sh 309600_seed42     # pick most recent checkpoint per run
#
# Requirements: kaggle CLI configured (kaggle.json or ~/.kaggle/kaggle.json)

set -e

PROJECT_DIR="$HOME/BirdCallClassifier"
KAGGLE_DS="$PROJECT_DIR/kaggle_dataset"
NOTEBOOK_DIR="$PROJECT_DIR/notebooks"
KERNEL_SLUG="arimarkowitz/birdclef-2026-inference"
COMPETITION="birdclef-2026"
USE_LATEST="${USE_LATEST:-0}"

# ── Step 1: Copy checkpoints ──────────────────────────────────────────────────
echo "=== Step 1: Preparing checkpoints ==="
if [ "$USE_LATEST" = "1" ]; then
    echo "Mode: most recent checkpoint per run"
else
    echo "Mode: best val_macro_auc checkpoint per run"
fi

# Always sync model code so inference uses the current architecture
mkdir -p "$KAGGLE_DS/src"
cp "$PROJECT_DIR/src/model.py" "$KAGGLE_DS/src/model.py"
echo "Synced src/model.py to kaggle_dataset/"

# Clear old checkpoints
rm -f "$KAGGLE_DS"/birdclef-birdset-*.ckpt "$KAGGLE_DS"/birdclef-htsat-*.ckpt
echo "Cleared old checkpoints from kaggle_dataset/"

if [ $# -gt 0 ]; then
    # Specific run IDs provided as arguments
    FOLD_IDX=0
    for run_id in "$@"; do
        run_dir="$PROJECT_DIR/checkpoints/$run_id"
        if [ ! -d "$run_dir" ]; then
            echo "ERROR: $run_dir not found"
            exit 1
        fi

        if [ "$USE_LATEST" = "1" ]; then
            # Pick most recently modified checkpoint
            picked=$(ls -t "$run_dir"/birdclef-birdset-*.ckpt 2>/dev/null | head -1)
        else
            # Pick checkpoint with highest val_macro_auc from filename
            picked=$(ls "$run_dir"/birdclef-birdset-*.ckpt 2>/dev/null | sed 's/.*val_macro_auc[=_]\([0-9.]*\).*/\1 &/' | sort -rn | head -1 | cut -d' ' -f2)
        fi

        if [ -z "$picked" ]; then
            echo "ERROR: No checkpoints in $run_dir"
            exit 1
        fi
        # Use unique name per fold to avoid overwrites when filenames collide
        ext="${picked##*.}"
        base="$(basename "$picked" ".$ext")"
        dest="$KAGGLE_DS/${base}_fold${FOLD_IDX}.${ext}"
        cp "$picked" "$dest"
        echo "  Copied: $(basename "$dest") (from $run_id)"
        FOLD_IDX=$((FOLD_IDX + 1))
    done
else
    echo "No run IDs specified — using existing checkpoints in kaggle_dataset/"
    # Check there's at least one
    if ! ls "$KAGGLE_DS"/birdclef-birdset-*.ckpt &>/dev/null; then
        echo "ERROR: No checkpoints found in $KAGGLE_DS/"
        echo "Usage: bash scripts/submit.sh <run_id1> [run_id2] ..."
        echo "Available runs:"
        ls -d "$PROJECT_DIR"/checkpoints/*/ 2>/dev/null | xargs -I{} basename {}
        exit 1
    fi
fi

echo ""
echo "Checkpoints to upload:"
ls -lh "$KAGGLE_DS"/birdclef-birdset-*.ckpt
echo ""

# ── Step 2: Upload dataset ────────────────────────────────────────────────────
echo "=== Step 2: Uploading dataset to Kaggle ==="
CKPT_COUNT=$(ls "$KAGGLE_DS"/birdclef-birdset-*.ckpt | wc -l)
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

# Wait for notebook to complete and capture version number
echo "Waiting for notebook to complete..."
NOTEBOOK_COMPLETE=false
for i in $(seq 1 60); do
    sleep 30
    STATUS=$(kaggle kernels status "$KERNEL_SLUG" 2>&1)
    echo "  [$i] $STATUS"
    if echo "$STATUS" | grep -qi "complete"; then
        echo "Notebook finished successfully."
        NOTEBOOK_COMPLETE=true
        break
    fi
    if echo "$STATUS" | grep -qi "error\|cancel"; then
        echo "ERROR: Notebook failed. Check Kaggle for details."
        exit 1
    fi
done

if [ "$NOTEBOOK_COMPLETE" != "true" ]; then
    echo "ERROR: Notebook did not complete within 30 minutes."
    exit 1
fi

# Get the latest version number
VERSION=$(kaggle kernels list --mine --search "birdclef-2026-inference" --csv 2>/dev/null \
    | grep "birdclef-2026-inference" | head -1 | cut -d',' -f5)
if [ -z "$VERSION" ]; then
    VERSION=1
    echo "WARNING: Could not detect version number, defaulting to $VERSION"
fi
echo "Notebook version: $VERSION"
echo ""

# ── Step 4: Submit to competition ─────────────────────────────────────────────
# BirdCLEF is a code competition — the notebook is re-run on hidden test data.
# API submission via `kaggle competitions submit` doesn't work (403 Forbidden).
# Must submit through the Kaggle UI.
echo "=== Step 4: Submit via Kaggle UI ==="
echo ""
echo "Notebook version $VERSION ran successfully. Submit to competition:"
echo ""
echo "  1. Go to: https://www.kaggle.com/code/arimarkowitz/birdclef-2026-inference"
echo "  2. Click 'Output' tab → latest version"
echo "  3. Click 'Submit to Competition'"
echo ""
echo "=== Done! ==="
