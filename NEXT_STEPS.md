# BirdCLEF 2026 — Next Steps

## Current status (2026-04-27)
- **Single-B1 baseline**: val_macro_auc 0.9686 (job 358438), Kaggle LB **0.871**.
- **Single-model self-training: failed 5 distinct ways.** Hard labels, soft logits + KL, warm-start at lr=2e-4, distribution-match + power-T + cross-domain MixUp at lr=2e-5, multi-iteration loop. All degrade val_auc 1-3pp; iteration *compounds* the regression.
- **Diagnosis**: single B1 teacher is miscalibrated on Pantanal soundscapes (frog detections dominate). Each pseudo-label round amplifies that bias.
- **Now running (job 369185)**: 4-backbone ensemble baseline (efficientnet_b0, mobilenetv3_small, resnet18, convnext_tiny on folds 0-3). New code: `src/backbones.py`, `scripts/ensemble_pipeline.sh`. First model finished at val_auc 0.9443 (~2.4pp below B1 due to no bird-domain pretraining).
- **Gap to 1st place**: 0.93 vs 0.87 on '25 LB equivalent.

## Key problem: domain shift
Model overfits clean focal recordings; generalizes poorly to noisy polyphonic soundscapes. Tier-1 items are ordered by expected impact on this gap.

---

## Tier 1 — Active

### 1. Multi-backbone ensemble pipeline (PIVOTAL — currently running)
`scripts/ensemble_pipeline.sh` trains 4 models (default: efficientnet_b0, mobilenetv3_small, resnet18, convnext_tiny) on 4 different CV folds, then optionally runs N iterations of cross-pollinated self-training (each model retrains using pseudo-labels from the *other 3*). Cross-pollination prevents the self-reinforcing bias that killed the single-model loop.

- [ ] **Submit ensemble baseline to Kaggle** — once 4-model run finishes, measure ensemble-only lift over single-B1 (this is the experimental control).
- [ ] **Compare per-backbone scores** — informs which backbones to keep / replace for round 2.
- [ ] **Run full pipeline (`N_ITERATIONS=3`)** — baselines + 3 cross-pollinated iterations. Pivotal: if this beats single-B1's 0.871 LB, pseudo-labels are finally working.
- [ ] **Consider swapping ConvNeXt-Tiny (31M)** for `mobilenetv3_large_100` (5M) or `tf_efficientnetv2_b0` (~7M) if Kaggle CPU inference is too slow.

### 2. OOF (out-of-fold) ensemble validation
Each ensemble member trains on a different fold, so per-model val_macro_auc values aren't directly comparable. Currently no unified ensemble-quality signal short of Kaggle submission.

OOF stitching: each model `i` predicts on its own held-out fold; concatenate per-fold predictions → one prediction per training sample → global macro-AUC. Mathematically valid (no leakage) and gives trustworthy ensemble-quality estimate without burning Kaggle submission budget.

- [ ] **Add `scripts/oof_eval.py`** — takes ensemble manifest, runs each model on its own val fold, stitches predictions, reports unified macro-AUC + per-class AUC.
- [ ] **Use OOF macro-AUC** as early-stopping signal for the iteration loop and as comparison metric across hyperparameter sweeps.

### 3. SED head with GRU (head architecture upgrade)
Current `TemporalAttentionPool` collapses (B, C, H, W) to (B, C) with a single CLS query — clip-level only. SED-style head: feed (B, C, T) sequence into a biGRU + per-frame classifier; aggregate via attention pooling for clip-level loss but keep frame-level scores for SED loss / inference.

Backbone-agnostic: every backbone produces (B, C, H, W) with W = time. Mean-pool H → (B, T, C), exactly what GRU expects.

- [ ] **Add `src/sed.py` with `SEDHead` module**: biGRU + linear → (B, T, num_classes); attention pooler → (B, num_classes) for clip-level loss. PANN-style.
- [ ] **Add `--head {attn_clip, sed_gru}` flag** to train.py (default: current `attn_clip` behavior).
- [ ] **Frame-level loss when timestamps available** — supervise frame predictions on labeled soundscape segments (which have start/end times). Clip-level pool for focal recordings.
- [ ] **Train one ensemble member with SED head** as the comparison. Compare val_macro_auc and especially worst-class-AUC vs attn-pool head.
- [ ] **30s window training** — once SED head works, retrain with 30s windows so the GRU sees real temporal context. Bigger refactor (dataloader chunking, loss, inference).

### 4. Tune ensemble pseudo-label hyperparameters (only after Tier-1 #1 improves baseline)
- [ ] **Tune `PSEUDO_POWER_T`** — sweep T ∈ {1.5, 2.0, 3.0, 4.0}. Babych tuned against LB feedback.
- [ ] **Tune `PSEUDO_MIXUP_ALPHA`** — try 0.2, 0.4, 0.6.
- [ ] **Tune `THRESHOLD`** — pseudo-label confidence cutoff.
- [ ] **Tune `RETRAIN_LR`** — currently 2e-5 for warm-start.
- [ ] **Pseudo-label stability filter** — track per-segment prediction stability across iterations; drop samples where predictions flip-flop.

### 5. Pseudo-label our own train soundscapes
Babych pseudo-labels train soundscapes too, not just external unlabeled data. Adds free domain-matched data with reliable signal.

- [ ] **Extend `scripts/pseudo_label.py` to predict on `train_soundscapes/`** — filter out chunks overlapping labeled regions.

### 6. Audio quality as a training signal
`train.csv` has `rating` column (0-5). Focal recordings rated 5 are clean; rating 0 are noisy unrated iNat.

- [ ] **Quality-weighted loss** — `loss_weight = 0.5 + 0.5·rating/5`. Trivial.
- [ ] **Quality-stratified validation** — report val AUC by rating bucket to diagnose noisy-input failures.
- [ ] **Quality-based sampling for rare species** — oversample high-quality clips for species with <10 samples.

### 7. Multi-channel mel input (better RGB-pretrain transfer + richer signal)
ImageNet-pretrained timm backbones expect 3-channel input. We currently average their 3→1 conv weights (~0.5pp loss). Stacking 3 different "mel-shaped" representations as RGB-equivalent channels both preserves pretrained weight quality AND gives genuinely new signal per channel. All channels share `(n_mels, T_frames)` shape.

Implementation: `--mel_channels {1, 3}` flag + `--mel_recipe` selector. timm backbones load with `in_chans=3`; BirdSet B1 stays at 1 (was trained on 1 channel).

Channel recipes, ranked by expected ROI:

- [ ] **delta + delta-delta** (DCASE / SED standard, proven 0.5-1pp lift)
  - Ch1: log-mel; Ch2: 1st-order time deriv (call onset / freq-sweep direction); Ch3: 2nd-order (call concavity)
  - Cheap: `torchaudio.functional.compute_deltas`
- [ ] **mel + PCEN + ΔPCEN** (problem-domain specific)
  - Ch2: per-channel-energy-normalized mel suppresses stationary noise via learnable AGC. BirdNET uses PCEN.
  - `torchaudio.functional.pcen`. Most relevant for our Pantanal soundscape noise.
- [ ] **mel + CQT-mel + Δmel** (tonal + temporal)
  - Ch2: constant-Q transform on mel scale (log freq, better low-freq harmonic resolution)
  - `nnAudio` package or hand-rolled.
- [ ] **mel + modulation-mel + ΔΔ** (rhythm-aware)
  - Ch2: spectrogram of mel envelope along time (captures trills, woodpecker drumming, frog pulse rates)
- [ ] **mel + reassigned-mel + Δmel** (transient-sharp)
  - Ch2: reassigned spectrogram — sharpens energy onto true T-F location. Better for clicks, tics.
- [ ] **mel + gammatone + Δmel** (perceptually motivated)
  - Ch2: cochlear bandpass filters. Older bird-audio literature standard.
- [ ] **mel + magnitude/phase derivatives**
  - Ch2: instantaneous frequency / group delay. Phase carries real info; rarely used in supervised, could help in semi-supervised.
- [ ] **mel triplet at different time offsets** (cheapest)
  - Same mel offset by ±2.5s — forces time-shift invariance. No new transforms.

Watch out for:
- BirdSet pretrained backbones (B1) expect 1 channel — must stay `in_chans=1` regardless of `--mel_channels`.
- SpecAugment on a stack: mask the same time-freq region across all 3 channels. Update `_apply_spec_aug` for (B, 3, mel, T).
- Δ / PCEN should be computed on *clean* mel pre-SpecAugment, otherwise derivatives encode the augmentation noise.

### 8. Speed-up levers
- [ ] **`torch.compile` flag** — already added (`--compile`). Test on ensemble runs; expect 20-40% on L40s.
- [ ] **Channels-last memory format** — separate flag, stacks with compile.
- [ ] **Pre-resampled audio cache** — eliminates librosa fallback; ~10× dataloader speedup. Tight on disk space.

---

## Tier 2 — Augmentation & data expansion

### Augmentation (noise-robust training)
Already in: multi-species mixing, MixUp/SuMix, SpecAugment.

- [ ] **Soundscape background injection** — mix crops from 10,593 unlabeled soundscapes as background noise behind clean train_audio. Most realistic noise source possible. ~5GB free in `data/train_soundscapes/`. `SoundscapeBackgroundMix` class already exists in dataset.py.
- [ ] **FilterAugment** — random band-wise spectral shaping; mimics mic/habitat/distance variation.
- [ ] **Time shift** — cheap, mirrors test-time offset variability.
- [ ] **Pitch shift (light)** — ±0.5-1.0 semitones only.
- [ ] **Gaussian noise / gain variation** — random SNR + volume scaling.

### Regional hard-negative training
Pantanal has ~650 bird species, we train on 234. The other 400+ show up as unlabeled vocalizations and become false positives. Already have `DistillAudioDataset` + `distill_manifest.csv` + hard-negative class infrastructure.

- [ ] **Curate Pantanal species list** — eBird/GBIF for MT do Sul, cross-reference with recording availability.
- [ ] **Pull regional audio from iNat Sounds** — https://github.com/visipedia/inat_sounds. Existing iNat files have `iNat` prefix for dedup.
- [ ] **Pull regional audio from Xeno-canto** — CC-licensed. Filter by lat -16 to -22, lon -54 to -58.
- [ ] **Pull from prior BirdCLEF datasets (2021-2025)** — South American species overlap. Already in competition format.

### Supplemental audio for data-starved species
14 species have ≤5 training samples and zero soundscape labels (mostly frogs + marmoset + nightjar + titi monkey + feral horse). Effectively unlearnable from current data.

Priority targets (1 sample each): Hooded Capuchin (516975), Waxy Monkey Tree Frog (23724), Southern Spectacled Caiman (116570), Central Dwarf Frog (23150).

Secondary (2-3 samples): Mato Grosso Snouted Tree Frog (24321), Cei's White-lipped Frog (70711), Feral Horse (209233), Cuyaba Dwarf Frog (476521), Muller's Termite Frog (25214), Yungas de la Paz Poison Frog (64898), Black-tailed Marmoset (74580), Usina Tree Frog (555123), Spot-tailed Nightjar (sptnig1), Cope's Swamp Froglet (23176).

Sources: Xeno-canto, iNaturalist, Fonoteca Neotropical, YouTube field recordings, iNat Sounds dataset.

### BirdSet-pretrained backbones (conditional)
Available: `ConvNeXT-Base-BirdSet-XCL`, `Wav2Vec2-Base-BirdSet-XCL`, `AST-BirdSet-XCL`. Each starts ~3pp higher per-model than ImageNet-pretrained timm models (already seen bird audio).

- [ ] Wire into `backbones.py` IF the cheap ensemble's per-model floor turns out to be the bottleneck after running. Most of their value is faster convergence — improved augmentation + cross-pollination diversity may close the same gap. Defer unless evidence demands.

### BirdSet pretraining stages (separate from above)
- [ ] **BirdSet XCM pretraining** — 89k recordings, 409 species, 89GB. Verify benefit beyond XCL-pretrained backbones.
- [ ] **BirdSet XCL full** — 528k recordings, 9.7k species, 484GB. Maximum but expensive.

---

## Tier 3 — Analysis & post-processing

### Per-class error analysis
Already in: per-class AUC every epoch.

- [ ] **Performance vs training frequency** — join per-label metrics with sample count, soundscape positive count, site coverage. Separate "weak due to low data" from "weak due to confusion / domain shift."
- [ ] **Confusion and co-occurrence review** — which labels are over/under-predicted? Which pairs confuse? Use to guide augmentation + thresholds.
- [ ] **Track worst-class AUC across self-training iterations** — do rare species improve with pseudo-labeling?

### Post-processing (free wins, no retraining)
- [ ] **Per-class threshold tuning** — optimize on val (or OOF), not 0.5 cutoff.
- [ ] **Spatial/site priors** — species × location from train.csv; downweight species never observed near Pantanal.
- [ ] **Test-time augmentation (TTA)** — average predictions over time-shifted segments. Tight on Kaggle inference budget.

### Event detection / clip filtering (if focal label noise is the bottleneck)
A random 5s crop from a 30s focal file may not contain the labeled species.

- [ ] **Model-based crop filtering** — run best checkpoint over train_audio in 5s windows, keep windows with high confidence for labeled species.
- [ ] **Framewise max pooling (full-file training)** — train on full 30s recordings with per-frame class scores, aggregate via max over time. Most principled; overlaps with Tier-1 #3 (SED head + 30s windows).
- [ ] **BirdNET pre-filtering** — covers ~6000 species, limited for rare/regional classes.

---

## Tier 4 — Research direction: alternative SSL recipes

These were considered as alternatives to Babych-style cross-pollinated self-training. Saved here so we don't re-litigate the decision tree.

### Pure online self-training (use current student as teacher every N steps) — REJECTED
Teacher=student loop amplifies errors without a stabilizing gap. In multi-label settings one wrong confident prediction poisons the pool fast. Our teacher is already miscalibrated (frog-biased on Pantanal); labeling more data with the same miscalibrated teacher accelerates the bias, not fixes it.

### Mean Teacher / EMA teacher — RESERVED
Teacher = exponential moving average of student weights. EMA lags student, stabilizing the pseudo-label signal while still tracking improvements. Tarvainen & Valpola 2017.

- [ ] Consider if cross-pollinated ensemble (Tier-1 #1) underperforms.

### FixMatch-style consistency regularization — RESERVED
Each unlabeled sample gets weak+strong augmentations; weak view's prediction (if confident) supervises strong view. Same forward pass, no separate pseudo-label step.

- [ ] Consider if Noisy Student + ensemble teacher plateaus below target. Orthogonal SSL recipe.

### Escalation order if current direction stalls
1. Finish ensemble pipeline (Tier-1 #1).
2. If LB improves → iterate Babych-style. Add OOF eval (Tier-1 #2). Consider SED head (Tier-1 #3).
3. If LB degrades → diagnose: is per-model floor the issue (try BirdSet-pretrained backbones, Tier-2) or is calibration still bad (try EMA teacher / FixMatch, this section)?

---

## Archive — tried-and-removed / blocked

- HTSAT backbone (replaced by BirdSet EfficientNet-B1)
- Two-stage temporal fusion (overfits small soundscape training set)
- NMF latent basis (deprioritized with BirdSet backbone)
- Bird-MAE backbone prototype (no notable benefit over BirdSet B1)
- Single-model self-training (failed 5 ways; replaced by cross-pollinated ensemble)
- Google Perch v2 — blocked: TF-only, GPU-only.
- AudioProtoPNet — 0.3B params, too large for Kaggle CPU.
- Multi-scale inference — adds inference cost.

---

## Implemented infrastructure (context for future work)
- **Modular backbone factory** (`src/backbones.py`): birdset_b1, birdset_b0, efficientnet_b0, mobilenetv3_small, resnet18, convnext_tiny, plus `hf:Org/Model` for arbitrary HF audio models.
- **Multi-backbone ensemble pipeline** (`scripts/ensemble_pipeline.sh`): sequential training with cross-pollinated self-training option.
- **Ensemble pseudo-labeling** (`scripts/pseudo_label.py --checkpoints A,B,C`): logit-averaged multi-model pseudo-labels.
- **Wandb logging** (train/val/per-class AUC).
- **k-fold CV** (`--n_folds`, `--fold`).
- **Multi-hot labels** (primary + secondary), label smoothing, focal loss.
- **All taxonomy classes** (birds, insects, amphibians, mammals, reptiles).
- **BirdSet-compatible mel frontend** (32kHz, n_fft=2048, hop=256, 256 mels) with SpecAugment.
- **Validation**: stratified train_audio holdout + soundscape fold.
- **Multi-species mixing + MixUp (SuMix) + class-balanced sampler**.
- **Hard-negative classes** via `DistillAudioDataset` + `distill_manifest.csv`.
- **Pseudo-label distillation**: logit-based KL, cross-domain MixUp, power transform, Σ-label-weighted sampler, distribution-matched per-species caps.
- **Warm-start** with cosine-only LR schedule.
- **`save_top_k=1` default** + `--compile` flag in train.py.
