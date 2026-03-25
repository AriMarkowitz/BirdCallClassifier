# BirdCLEF 2026 — Usage Guide

## A) Training

### Single model
```bash
sbatch scripts/train.sh
```
Trains one model with seed 42 on the Easley L40S partition. Checkpoints saved to `checkpoints/<jobid>/`. Logs appear in `logs/birdclef_<jobid>.log` and in Weights & Biases under project `birdclef-2026`.

### 5-model ensemble
```bash
bash scripts/train_ensemble.sh
```
Submits 5 independent SLURM jobs (seeds 42, 123, 456, 789, 2026). Each gets a different 75/25 soundscape val split and saves to `checkpoints/<jobid>/`.

### Monitor training
```bash
squeue -u $USER                        # check job status
tail -f logs/birdclef_<jobid>.log      # live log output
```
Or open Weights & Biases — run `wandb login` on the login node before submitting if you haven't already.

### Key hyperparameters (in `scripts/train.sh`)
| Flag | Default | Notes |
|---|---|---|
| `--max_epochs` | 12 | |
| `--lr` | 1e-4 | constant LR |
| `--batch_size` | 128 | uses ~24GB of 48GB VRAM |
| `--soundscape_weight` | 3.0 | initial weight (curriculum ramps 0.5 → 3.0) |
| `--label_smoothing` | 0.1 | soft labels |
| `--loss` | bce | `bce` or `focal` |
| `--focal_alpha` | 0.25 | positive class weight (only with `--loss focal`) |
| `--focal_gamma` | 2.0 | focusing parameter (only with `--loss focal`) |
| `--n_folds` | 5 | k-fold CV |
| `--fold` | 0 | which fold to hold out |

### Curriculum training
Soundscape weight ramps linearly from 0.5 (mostly clean audio) to 3.0 (heavy soundscape focus) over the training run. Progress is logged each epoch.

### NMF feature integration
The HTSAT model includes an NMF latent feature branch. A fixed global dictionary (`nmf_analysis/output/W_k56.npy`, 64 mel bins × 56 components) is loaded at model init. During forward pass, the model computes a power mel spectrogram, solves for NMF activations H via multiplicative updates (50 iterations, no gradient), summarizes H via mean+max pooling → 112-dim feature vector, and projects this through `nmf_proj` (a linear layer) to produce class logits that are added to the HTSAT classification output. The NMF dictionary is frozen; only `nmf_proj` is trained.

---

## B) Inference (local test)

Run the inference script directly against local data:
```bash
python notebooks/inference.py
```
This expects Kaggle-style paths (`/kaggle/input/...`). For local testing you'd need to edit the path constants at the top of the file. On Kaggle it runs automatically — see section C.

---

## C) Uploading & Submitting to Kaggle

### What lives in `kaggle_dataset/`
This directory is the Kaggle dataset that gets uploaded. It contains:
- `htsat/` — vendored HTSAT model code
- `taxonomy.csv` — class label map
- `birdclef-htsat-*.ckpt` — trained checkpoint(s)
- `dataset-metadata.json` — dataset ID (`arimarkowitz/birdclef-2026-model`)

### One-command submit (recommended)
```bash
# Submit best checkpoint from specific training runs (picks highest val_macro_auc):
bash scripts/submit.sh 309600_seed42 309601_seed123 309602_seed456

# Submit whatever checkpoints are already in kaggle_dataset/:
bash scripts/submit.sh
```
This handles the full pipeline: clears old checkpoints, copies best from each run, uploads dataset, pushes notebook, waits for completion, downloads output, and submits.

### Manual step-by-step

#### Step 1 — Copy new checkpoint(s) in
```bash
rm kaggle_dataset/birdclef-htsat-*.ckpt  # clear old ones

# Copy best from each run (replace <run_id> with e.g. 309600_seed42)
for run_id in <run1> <run2> <run3>; do
    best=$(ls checkpoints/${run_id}/birdclef-htsat-*.ckpt | sed 's/.*val_macro_auc[=_]\([0-9.]*\).*/\1 &/' | sort -rn | head -1 | cut -d' ' -f2)
    cp "$best" kaggle_dataset/
done
```

#### Step 2 — Upload dataset (only when checkpoints change)
```bash
kaggle datasets version -p kaggle_dataset/ -m "description of changes" --dir-mode zip
```

#### Step 3 — Run notebook and submit (CLI)
```bash
kaggle kernels push -p notebooks/                          # triggers fresh run
kaggle kernels status arimarkowitz/birdclef-2026-inference  # check progress

# Once status is "complete":
kaggle kernels output arimarkowitz/birdclef-2026-inference -p /tmp/submission
kaggle competitions submit -c birdclef-2026 -f /tmp/submission/submission.csv -m "description"
```

#### Step 3 (alternative) — Via Kaggle UI
1. Open notebook on kaggle.com → click **Edit**
2. **Save Version** → **Save & Run All (Commit)**
3. Wait for run to complete
4. Go to **Output** tab → **Submit to Competition**

### Avoiding redundant work
| Changed | Action needed |
|---|---|
| Only `inference.py` | `kaggle kernels push` (triggers re-run) |
| Only checkpoints | `kaggle datasets version` + `kaggle kernels push` |
| Neither | Just re-run: `kaggle kernels push` |

The dataset version number increments each time you run `kaggle datasets version`. The notebook automatically uses the latest version of the attached dataset.
