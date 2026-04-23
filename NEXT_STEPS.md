# BirdCLEF 2026 — Next Steps

## Current status (2026-04-23)
- Baseline EfficientNet-B1 BirdSet backbone reproduced: val_macro_auc **0.9686** (epoch 25, job 358438), Kaggle LB **0.871** (prior 334486).
- Babych BirdCLEF'25 1st-place self-training pieces are wired in and about to be tested on top of the 358438 baseline:
  - Power transform on pseudo soft labels (`PSEUDO_POWER_T=2.0`)
  - Cross-domain MixUp pseudo↔labeled (`PSEUDO_MIXUP_ALPHA=0.4`)
  - Sampler weighted by Σ soft-labels per pseudo chunk
  - Distribution-matched per-species pseudo caps
- Gap to 1st place: **0.93 vs 0.87** on '25 LB equivalent. Closing that gap is the point.

## Key problem: domain shift
Model overfits clean focal recordings and generalizes poorly to noisy polyphonic soundscapes. Every tier-1 item below targets this gap, ordered by expected impact.

---

## Tier 1 — Active / immediate next steps

### 1. Validate the current self-training pipeline
- [ ] **Run pipeline with 358438 epoch=25 ckpt** — does retrained model beat 0.9686 val? Does it beat 0.871 Kaggle? This is the pivotal experiment. If yes → iterate. If no → escalate to ensemble teacher (#2).
- [ ] **Tune `PSEUDO_POWER_T`** — sweep T ∈ {1.5, 2.0, 3.0, 4.0} once a single setting works. Babych's team tuned T against LB feedback; for us val_macro_auc is the signal.
- [ ] **Tune `PSEUDO_MIXUP_ALPHA`** — try 0.2, 0.4, 0.6. Higher α = more aggressive mixing.
- [ ] **Tune `THRESHOLD`** — pseudo-label confidence cutoff, currently 0.8. Balance between coverage and noise.
- [ ] **Tune `RETRAIN_LR`** — currently 2e-4 for bs=64. If unstable at start, drop to 1e-4.

### 2. Diverse-backbone ensemble teacher (most likely next lever)
The Babych diagram says "select best *ensemble* from current iteration → new teacher" and rotates backbones across iterations. A single-backbone B1 teacher propagates its own biases into pseudo-labels — directly explains the frog-flood problem we've observed (67% Amphibia in pseudo-labels vs 1.3% in training).

- [ ] **Add EfficientNet-B0 training support** — same BirdSet family, smaller and faster, fits in Kaggle CPU budget.
- [ ] **Add MobileNetV3-Small training support** — different inductive bias than EfficientNet.
- [ ] **Ensemble prediction at pseudo-label step** — average logits across 2-3 backbones before thresholding. Should calibrate pseudo-labels meaningfully.
- [ ] **Rotate backbones per iteration** — iteration 1 uses A+B, iteration 2 uses B+C, iteration 3 uses A+C (or similar) to avoid architectural lock-in.
- [ ] **Final Kaggle submission IS the ensemble** — not separate work; the teacher ensemble doubles as the submission. Target Kaggle CPU budget: 3-5 small models.

### 3. Multi-iteration Noisy Student loop
Babych gained ~1pp per iteration (0.909 → 0.918 → 0.927 → 0.930) across 4 iterations. We're doing 1. Once one iteration works:

- [ ] **Add iteration loop to pipeline.sh** — after Step 3 retrain, re-pseudo-label with the retrained model and retrain again.
- [ ] **Track val_macro_auc across iterations** — confirm monotone improvement or stop early.
- [ ] **Pseudo-label stability filter** — track per-segment prediction stability across iterations; drop samples where predictions flip-flop.

### 4. Pseudo-label our own train soundscapes too
Babych pseudo-labels "train soundscapes," not just external unlabeled data. We currently only pseudo-label Pantanal. Adding train soundscape chunks with teacher predictions gives free domain-matched data with a reliable signal.

- [ ] **Extend `scripts/pseudo_label.py` to also predict on `train_soundscapes/`** — filter out chunks that overlap labeled regions.

### 5. Audio quality as a training signal
`train.csv` has a `rating` column (0-5). Focal recordings rated 5 are clean; rating 0 are noisy unrated iNat. Lots of headroom.

- [ ] **Quality-weighted loss** — `loss_weight = 0.5 + 0.5·rating/5`. Trivial to add.
- [ ] **Quality-stratified validation** — report val AUC by rating bucket to diagnose noisy-input failures.
- [ ] **Quality-based sampling for rare species** — oversample high-quality clips for species with <10 training samples.

---

## Tier 2 — Research direction: online self-training variants

Considered but deferred. Saving the reasoning here so we don't lose it.

### Pure online self-training (use current student as teacher every N steps) — REJECTED
Appeal: pseudo-labels keep improving as model improves; no full re-pseudo-label pass between iterations.

Why rejected: teacher=student loop amplifies errors without a stabilizing gap. In multi-label settings one wrong confident prediction poisons the pool fast. Our teacher is already miscalibrated (frog-biased on Pantanal); labeling more data with the same miscalibrated teacher accelerates the bias, not fixes it.

### Mean Teacher / EMA teacher — RESERVED
Teacher is an exponential moving average of student weights. EMA lags the student, stabilizing the pseudo-label signal while still tracking improvements. Well-studied (Tarvainen & Valpola 2017).

- [ ] **Consider if diverse-backbone ensemble (#2) underperforms** — EMA teacher addresses similar concerns but with one model instead of several.

### FixMatch-style consistency regularization — RESERVED
Each unlabeled sample gets weak+strong augmentations; weak view's prediction (if confident) supervises strong view's. Same forward pass, no separate pseudo-label step. Well-studied baseline for semi-supervised learning.

- [ ] **Consider if Noisy Student + ensemble teacher plateaus below target** — FixMatch is an orthogonal SSL recipe.

### Recommended escalation order
1. Finish current pipeline run (tier 1 #1)
2. If it improves → iterate Babych-style (#3) and add diverse backbones (#2)
3. If it degrades → go straight to diverse-backbone ensemble teacher (#2) — addresses the root calibration issue
4. Only after all of the above → explore EMA teacher or FixMatch

---

## Tier 3 — Augmentation & data expansion (steady wins)

### Augmentation (noise-robust training)
- [x] Multi-species mixing (`MultiSpeciesMixDataset`)
- [x] MixUp / SuMix (`_sumix`)
- [x] SpecAugment (via BirdSet frontend)
- [ ] **Soundscape background injection** — mix crops from 10,593 unlabeled soundscapes as background noise behind clean train_audio. Most realistic noise source possible. ~5GB of free noise in `data/train_soundscapes/`.
- [ ] **FilterAugment** — random band-wise spectral shaping, mimics mic/habitat/distance variation.
- [ ] **Time shift** — cheap, mirrors test-time offset variability.
- [ ] **Pitch shift (light)** — ±0.5-1.0 semitone only.
- [ ] **Gaussian noise / gain variation** — random SNR + volume scaling.

### Regional hard-negative training (PRIORITY)
Pantanal has ~650 bird species, we train on 234. The other 400+ show up as unlabeled vocalizations and become false positives. Already have infrastructure: `DistillAudioDataset` + `distill_manifest.csv` + hard-negative classes.

- [ ] **Curate Pantanal species list** — eBird/GBIF checklist for MT do Sul, cross-reference with recording availability.
- [ ] **Pull regional audio from iNat Sounds** — https://github.com/visipedia/inat_sounds. Our existing iNat files have `iNat` prefix for deduplication.
- [ ] **Pull regional audio from Xeno-canto** — CC-licensed, best bird coverage. Filter by lat -16 to -22, lon -54 to -58.
- [ ] **Pull from prior BirdCLEF datasets (2021-2025)** — South American species overlap. Already in competition format.

### Supplemental audio for data-starved species
14 species have ≤5 training samples and zero soundscape labels (mostly frogs + marmoset + nightjar + titi monkey + feral horse). Effectively unlearnable from our data alone.

Priority targets (1 training sample each):
- Hooded Capuchin (516975), Waxy Monkey Tree Frog (23724), Southern Spectacled Caiman (116570), Central Dwarf Frog (23150)

Secondary (2-3 samples):
- Mato Grosso Snouted Tree Frog (24321), Cei's White-lipped Frog (70711), Feral Horse (209233), Cuyaba Dwarf Frog (476521), Muller's Termite Frog (25214), Yungas de la Paz Poison Frog (64898), Black-tailed Marmoset (74580), Usina Tree Frog (555123), Spot-tailed Nightjar (sptnig1), Cope's Swamp Froglet (23176)

Sources: Xeno-canto, iNaturalist, Fonoteca Neotropical, YouTube field recordings, iNat Sounds dataset.

### Larger-context training
- [ ] **Train on longer windows** — 30s windows with frame-level SED predictions + temporal max-pooling. Past winners found temporal context helps the model learn call patterns within a soundscape.

### BirdSet pretraining stages (if we're not already benefiting from it)
- [ ] **BirdSet XCM pretraining** — 89k recordings, 409 species, 89GB. Verify we actually benefit beyond the XCL-pretrained B1 backbone.
- [ ] **BirdSet XCL full** — 528k recordings, 9.7k species, 484GB. Maximum but expensive.

---

## Tier 4 — Analysis & post-processing (incremental)

### Per-class error analysis
- [x] Per-class AUC logged every epoch.
- [ ] **Performance vs training frequency** — join per-label metrics with sample count, soundscape positive count, site coverage. Separate "weak due to low data" from "weak due to confusion/domain-shift."
- [ ] **Confusion and co-occurrence review** — which labels are over/under-predicted? Which pairs confuse? Use to guide augmentation + thresholds.
- [ ] **Track worst-class AUC across self-training iterations** — do rare species improve with pseudo-labeling?

### Post-processing (free wins, no retraining)
- [ ] **Per-class threshold tuning** — optimize on val, not 0.5 cutoff.
- [ ] **Spatial/site priors** — species × location from train.csv, downweight species never observed near Pantanal.
- [ ] **Test-time augmentation (TTA)** — average predictions over time-shifted segments. Tight on Kaggle inference budget.

### Event detection / clip filtering (if label noise in train_audio is the bottleneck)
A random 5s crop from a 30s focal file may not contain the labeled species. Options:
- [ ] **Model-based crop filtering** — run best checkpoint over train_audio in 5s windows, keep windows with high confidence for the labeled species.
- [ ] **Framewise max pooling (full-file training)** — train on full 30s recordings with per-frame class scores, aggregate via max over time. Most principled.
- [ ] **BirdNET pre-filtering** — BirdNET covers ~6000 species, limited for rare/regional classes.

---

## Tier 5 — Deprioritized / archived

- [x] HTSAT backbone (replaced by BirdSet EfficientNet-B1)
- [x] Two-stage temporal fusion (removed — overfits small soundscape training set)
- [x] NMF latent basis (deprioritized with BirdSet backbone)
- [x] Bird-MAE backbone prototype (no notable benefit over BirdSet B1)
- [ ] Google Perch v2 (blocked: TF-only, GPU-only)
- [ ] AudioProtoPNet (0.3B params, too large for Kaggle CPU)
- [ ] Multi-scale inference (adds inference cost)

---

## Completed infrastructure (context for future work)
- Wandb logging (train_loss, val_loss, val_macro_auc, per-class AUC)
- Optional k-fold ensemble (`--n_folds`, `--fold`)
- Secondary labels (multi-hot targets)
- Label smoothing
- All taxonomy classes (birds, insects, amphibians, mammals, reptiles)
- Focal loss (`--loss focal`)
- BirdSet EfficientNet-B1 backbone with SpecAugment
- Validation includes stratified train_audio holdout + soundscape fold
- Multi-species mixing + MixUp (SuMix) + class-balanced sampler
- Distill manifest + hard-negative classes (`DistillAudioDataset`)
- Pseudo-labeling with logit-based teacher distillation, cross-domain MixUp, power transform, Σ-label-weighted sampler, distribution-matched caps
- Warm-start loading with cosine-only LR schedule
