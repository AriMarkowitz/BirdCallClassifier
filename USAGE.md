# BirdCLEF 2026 — Usage Guide

## A) Training

### Single model
```bash
sbatch scripts/train.sh
```
Trains one model with seed 42 on the Easley L40S partition. Checkpoints saved to `checkpoints/seed42/`. Logs appear in `logs/birdclef_<jobid>.out` and in Weights & Biases under project `birdclef-2026`.

### 5-model ensemble
```bash
bash scripts/train_ensemble.sh
```
Submits 5 independent SLURM jobs (seeds 42, 123, 456, 789, 2026). Each saves to its own subdirectory: `checkpoints/seed42/`, `checkpoints/seed123/`, etc. Jobs may run in parallel if GPUs are available.

### Monitor training
```bash
squeue -u $USER                        # check job status
tail -f logs/birdclef_<jobid>.out      # live log output
```
Or open Weights & Biases — run `wandb login` on the login node before submitting if you haven't already.

### Key hyperparameters (in `scripts/train.sh`)
| Flag | Default | Notes |
|---|---|---|
| `--max_epochs` | 50 | |
| `--lr` | 3e-5 | cosine restarts, floor 1e-6 |
| `--batch_size` | 128 | uses ~24GB of 48GB VRAM |
| `--soundscape_weight` | 3.0 | upweight soundscape segments |
| `--val_sites` | S22 S23 | sites held out for validation |

---

## B) Inference (local test)

Run the inference script directly against local data:
```bash
python notebooks/inference.py
```
This expects Kaggle-style paths (`/kaggle/input/...`). For local testing you'd need to edit the path constants at the top of the file. On Kaggle it runs automatically — see section C.

---

## C) Uploading to Kaggle

### What lives in `kaggle_dataset/`
This directory is the Kaggle dataset that gets uploaded. It contains:
- `htsat/` — vendored HTSAT model code
- `taxonomy.csv` — class label map
- `birdclef-htsat-*.ckpt` — trained checkpoint(s)
- `dataset-metadata.json` — dataset ID (`arimarkowitz/birdclef-2026-model`)

### Step 1 — Copy new checkpoint(s) in
After training, copy the best checkpoint from each seed into `kaggle_dataset/`. For a single model:
```bash
cp checkpoints/seed42/birdclef-htsat-<epoch>-<auc>.ckpt kaggle_dataset/
```
For the ensemble, copy the best checkpoint from each seed:
```bash
for seed in 42 123 456 789 2026; do
    best=$(ls -t checkpoints/seed${seed}/birdclef-htsat-*.ckpt 2>/dev/null | head -1)
    [ -n "$best" ] && cp "$best" kaggle_dataset/
done
```
Remove old checkpoints from `kaggle_dataset/` if you don't want them included (they bloat the upload and slow inference):
```bash
ls kaggle_dataset/*.ckpt   # verify what's there before deleting
rm kaggle_dataset/birdclef-htsat-<old-checkpoint>.ckpt
```

### Step 2 — Upload dataset (only when checkpoints change)
```bash
kaggle datasets version -p kaggle_dataset/ -m "v2: ensemble 5 seeds" --dir-mode zip
```
This creates a **new version** of the existing dataset — it does **not** create a duplicate dataset. Skip this step if the checkpoints haven't changed since the last upload.

### Step 3 — Push the inference notebook
```bash
kaggle kernels push -p notebooks/
```
This updates the notebook code. You only need to re-push when `notebooks/inference.py` changes.

### Step 4 — Run the notebook and submit
1. Go to [kaggle.com](https://www.kaggle.com) → your notebook → **Run All**
2. Once complete, go to the **Output** tab → **Submit to Competition**

### Avoiding redundant uploads
| Changed | Action needed |
|---|---|
| Only `inference.py` | `kaggle kernels push` only |
| Only checkpoints | `kaggle datasets version` + `kaggle kernels push` |
| Neither | Just re-run the existing notebook on Kaggle |

The dataset version number increments each time you run `kaggle datasets version`. The notebook automatically uses the latest version of the attached dataset.
