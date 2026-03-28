# BirdCLEF 2026 — Next Steps

## Done (current training)
- [x] Wandb logging (train_loss, val_loss, val_macro_auc)
- [x] optional k-fold ensemble
- [x] Secondary labels in train_audio (multi-hot targets)
- [x] Label smoothing (ε=0.1)
- [x] All taxonomy classes trained (birds, insects, amphibians, mammals, reptiles)
- [x] NMF latent feature integration (global dictionary W_k56, nmf_proj head) — superseded by BirdSet EfficientNet-B1 backbone
## Key problem: domain shift
Val AUC was 0.999 but Kaggle score is 0.848. The inflated val AUC was because validation only used soundscape segments (most species absent → trivially high AUC). **Fixed:** validation now includes a stratified holdout of train_audio + soundscape fold, so val AUC reflects all species. The model memorizes clean focal recordings but cannot generalize to noisy, polyphonic test soundscapes. Past BirdCLEF winners confirm this is THE problem to solve. Everything below is ordered by expected impact on closing this gap.

---

## Tier 1 — Proven winning strategies (from past BirdCLEF solutions)

### 1. Noise-robust training with strong augmentation
The model trains on clean single-species focal recordings but tests on noisy multi-species soundscapes. Augmentation bridges this domain gap.

- [x] **Multi-species mixing** — implemented as `MultiSpeciesMixDataset` in dataset.py. Overlays 1-4 random train_audio clips at gains 0.1-0.7 with probability 0.3 per sample. Union of multi-hot labels. Enabled by default (`--multi_mix`, `--mix_prob 0.3`).
- [x] **MixUp / SuMix** — implemented as batch-level SuMix in train.py `_sumix()`. Shuffles batch and additively mixes waveforms with Beta-distributed lambda, soft-union labels. Enabled by default (`--mixup_alpha 0.4`).
- [ ] **Soundscape background injection** — mix a random crop from the 10,593 unlabeled soundscape files as background noise behind clean train_audio clips. This is the most realistic noise source possible — same equipment, same sites, same ambient conditions as test data. ~5GB of free noise available in `data/train_soundscapes/`.
- [x] **SpecAugment** — applied via `FrequencyMasking` and `TimeMasking` in BirdSet EfficientNet-B1 frontend during training.
- [ ] **FilterAugment** — apply random band-wise spectral shaping / filtering to mimic microphone, habitat, and distance variation. Useful for domain robustness without changing labels.
- [ ] **Time shift** — randomly roll or offset clips within the 5s window so detections are less position-dependent. Cheap augmentation and also useful to mirror test-time offset variability.
- [ ] **Pitch shift (light)** — small semitone perturbations only (e.g. ±0.5 to ±1.0 semitones) to improve robustness while avoiding unrealistic bird vocal transformations.
- [ ] **Gaussian noise / gain variation** — random SNR noise injection and volume scaling for additional robustness.

### 2. Iterative pseudo-labeling of unlabeled soundscapes
Only 66 of 10,658 soundscape files have labels (1,478 segments). The other 10,593 files are unlabeled — this is a massive untapped resource. Past winners treated this as the #1 lever.

- [x] **Round 1: Generate pseudo-labels** — run current best model on all unlabeled soundscapes and all training data (for additiona secondar labels), segment into 5s windows, predict species probabilities. Filter by confidence threshold (e.g. keep predictions > 0.6).
  - [ ] tune above paramter
- [ ] **Round 2: Retrain with pseudo-labels** — add high-confidence pseudo-labeled segments to training set. Use soft labels (raw probabilities) instead of hard labels to reduce noise propagation.
- [ ] **Round 3+: Iterate** — retrain → relabel → retrain. Each round improves the model's soundscape predictions, producing better pseudo-labels for the next round. Typically 2-3 rounds before diminishing returns.
- [ ] **Pseudo-label filtering** — track prediction confidence across rounds. Remove samples where the model flip-flops (unstable predictions). Weight pseudo-labeled samples lower than ground-truth labeled samples in the loss.
- [ ] **Train on Raw pseudo labels** - use raw probabilities for training

### 3. Focal loss
- [x] **Replace BCELoss with focal loss** — implemented as `--loss focal` flag in train.py. Down-weights easy negatives (230+ absent species per segment), focuses gradients on hard positives. Default α=0.25, γ=2.0.
- [ ] **Focal loss ablation** — compare BCE vs focal loss under the current BirdSet EfficientNet setup, especially once stronger augmentation is enabled. Check whether focal improves rare/quiet species recall without destabilizing calibration.

### 4. Regional hard-negative training with supplemental Pantanal species (PRIORITY)
The Pantanal hosts ~650 bird species alone, but we only train on 234 classes. The other 400+ species are present in test soundscapes as unlabeled vocalizations. Without hearing them during training, the model either ignores them (lucky) or misclassifies them as one of our 234 targets (false positives). This is likely a major source of score loss.

**Approach:** Expand the training set with audio from non-target Pantanal species, trained as additional classes. At inference, only predict the 234 target classes — the extra classes act as "hard negatives" that teach the model to distinguish our targets from acoustically similar non-target species.

- [ ] **Curate Pantanal species list** — use eBird/GBIF checklists for the Pantanal region (Mato Grosso do Sul, Brazil) to identify which of the ~650 species are NOT in our 234-class taxonomy. Cross-reference with recording availability.
- [ ] **Pull regional audio from iNat Sounds** — https://github.com/visipedia/inat_sounds has large-scale iNat audio. Filter by Pantanal-region species, deduplicate against existing train_audio (our iNat files have `iNat` prefix). This is the easiest source since format/quality matches our data.
- [ ] **Pull regional audio from Xeno-canto** — filter by geographic bounding box (lat -16 to -22, lon -54 to -58) or by species list. CC-licensed. Xeno-canto has the best bird coverage.
- [ ] **Pull from prior BirdCLEF competition data** — previous years' BirdCLEF datasets (2021-2025) cover South American species. Many will overlap with Pantanal fauna. Already in competition-compatible format.
- [ ] **Expand classification head** — increase `num_classes` from 234 to 234 + N_extra. Train on all classes, but at inference time slice predictions to the original 234. The extra classes share the backbone so they improve feature learning without affecting the output format.
- [ ] **Alternative: train extra species as a single "other" class** — simpler than adding hundreds of classes. All non-target species map to class 235 ("other"). The model learns "this is a bird but not one of ours." Less discriminative but much easier to implement.

### 5. Fix worst per-class AUC species (PRIORITY)
Per-class AUC analysis (run 313637, epoch 21) reveals two distinct failure modes:

**Data-starved species (AUC < 0.6) — need more data urgently:**
| Species | Common Name | Samples | Best AUC |
|---|---|---|---|
| 70711 | Cei's White-lipped Frog | 2 | 0.19 |
| 209233 | Feral Horse | 2 | 0.24 |
| 74113 | Highland (cattle) | 10 | 0.51 |
| 476521 | Cuyaba Dwarf Frog | 3 | 0.87 |
| 1595929 | Uruguay Harlequin Frog | 5 | 0.56 |
| 555123 | Usina Tree Frog | 3 | 0.83 |

These are nearly unlearnable with current data. Fixes: pseudo-labeling (in progress), supplemental audio from Xeno-canto/iNaturalist/Fonoteca Neotropical (see section 7).

**Acoustically confused species (AUC 0.8-0.9, plenty of data) — need better representations:**
| Species | Common Name | Samples | Best AUC |
|---|---|---|---|
| strowl1 | Striped Owl | 184 | 0.82 |
| grekis | Great Kiskadee | 482 | 0.87 |
| epaori4 | Variable Oriole | 188 | 0.86 |
| ruther1 | Rufescent Tiger Heron | 86 | 0.84 |

These have enough data but still underperform — likely acoustic confusion with similar species or domain shift from clean recordings to noisy soundscapes. Fixes: confusion matrix analysis to identify confusable pairs, targeted augmentation, multi-species mixing (in progress).

- [ ] **Investigate Great Kiskadee confusion** — 482 samples but only 0.87 AUC. Likely confused with acoustically similar species. Check confusion matrix once available.
- [ ] **Track worst-class AUC across rounds** — compare R1 vs R2 (pseudo-labeled) to see if rare species improve. If not, pseudo-label threshold may be too high for rare species.

### 5. Per-label error analysis + class imbalance audit
- [x] **Best / worst label analysis** — per-class AUC now logged every epoch (see section 4 above for initial results).
- [ ] **Performance vs training frequency** — join per-label metrics with number of training examples, number of positive soundscape segments, and site coverage. Check whether weak labels are mostly just low-resource labels or whether some abundant classes are still failing due to confusion/domain shift.
- [ ] **Confusion and co-occurrence review** — inspect which labels are commonly over-predicted, under-predicted, or confused with acoustically similar species / taxa. Use this to guide augmentation, thresholding, and targeted data collection.
- [ ] **Rare-label validation slice** — maintain a reporting view focused on low-resource classes so leaderboard improvements are not driven only by already-common species.

### 5. Bird-MAE-Base backbone (deprioritized)
- [x] **Replace HTSAT with Bird-MAE-Base** — NO NOTABLE BENEFIT - self-supervised ViT-B/16 (85M params, 768-dim embeddings) pretrained via masked autoencoder on BirdSet's 9.7k species ([model](https://huggingface.co/DBD-research-group/Bird-MAE-Base)). Previously prototyped on `bird-mae-backbone` branch but deprioritized in favor of BirdSet EfficientNet-B1.

---

## Tier 2 — High potential, more effort

### 6. Event detection clip filtering (DEFERRED)
Core problem: a random 5s crop from a 30s training file may not contain the labeled species at all, creating label noise. The current energy-based VAD (`preprocess_activity.py`) is not species-aware and can't reliably solve this. Options ranked by trustworthiness:
- [ ] **Use trained model to filter crops** — run the best checkpoint over all train_audio files in 5s windows, keep only windows where model confidence > threshold for the labeled species. Species-specific, no external dependency. Mild confirmation bias risk but strictly better than energy VAD.
- [ ] **Framewise max pooling (full-file training)** — train on full 30s recordings with per-frame class scores, aggregate via `max` over time. Supervision pushes the model to find real call frames; silence frames get low scores naturally. Most principled solution, no filtering needed.
- [ ] **BirdNET pre-filtering** — run BirdNET over train_audio, keep windows with high BirdNET confidence for the labeled species. Fast, species-aware, but BirdNET only covers ~6000 species and may miss rare/regional classes — fall back to random crop for unrecognized species.

### 7. Longer-context SED-style training
- [ ] **Train on longer windows** — instead of 5s clips, train on 30s windows with frame-level (SED) predictions and temporal max pooling. Past winners found that longer context helps the model learn temporal patterns of species calls within a soundscape.

### 7. Transfer learning from larger bird-audio sources
- [ ] **BirdSet XCM pretraining (stage 0)** — 89k focal recordings, 409 species, 89GB. Fine-tune backbone on XCM first, then on BirdCLEF data. Stays within current architecture. Need to check species overlap with our 234 classes.
- [ ] **BirdSet XCL (full dataset)** — 528k recordings, 9.7k species, 484GB. Maximum pretraining data but significant storage/compute requirements.
- [ ] **Add geographically relevant non-bird data** — incorporate insects / amphibians (and possibly other taxa present in the competition taxonomy) from similar South American / Pantanal-adjacent soundscapes. Goal: improve negative class modeling, reduce false bird positives, and better match the real acoustic background distribution at test time.
- [ ] **Supplemental audio for undersampled species** — 14 species have ≤5 training samples and zero soundscape labels (mostly frogs, plus a marmoset, a nightjar, a titi monkey, and feral horse). These are effectively unlearnable from our data alone. Sources to pull from:
  - **Xeno-canto** — largest open bird/wildlife sound archive; some frog coverage but spotty for rare Neotropical species
  - **iNaturalist** — has audio observations, especially for herps; search by species + Pantanal/Mato Grosso region
  - **Fonoteca Neotropical (FN)** — specialized Neotropical animal sound archive hosted by Instituto Humboldt; best coverage for South American frogs and mammals
  - **YouTube field recordings** — search species common name + "call" or "vocalization"; extract audio clips and manually verify
  - **iNat Sounds dataset** (https://github.com/visipedia/inat_sounds) — large-scale iNaturalist audio dataset with species labels. Check if our worst-performing species are represented and if the recordings are distinct from what's already in train_audio (our existing iNat files have `iNat` prefix — deduplicate by comparing filenames/URLs). High priority since our train_audio already comes partly from iNat, so format/quality will be consistent.
  - Priority targets (1 training sample each): Hooded Capuchin (516975), Waxy Monkey Tree Frog (23724), Southern Spectacled Caiman (116570), Central Dwarf Frog (23150)
  - Secondary targets (2-3 samples): Mato Grosso Snouted Tree Frog (24321), Cei's White-lipped Frog (70711), Feral Horse (209233), Cuyaba Dwarf Frog (476521), Muller's Termite Frog (25214), Yungas de la Paz Poison Frog (64898), Black-tailed Marmoset (74580), Usina Tree Frog (555123), Spot-tailed Nightjar (sptnig1), Cope's Swamp Froglet (23176)

### 8. Audio quality as a training signal
train.csv has a `rating` column (0-5 scale from Xeno-canto/iNaturalist): 6,845 files rated 5.0, 8,018 rated 4.0, and 12,849 rated 0.0 (unrated iNat). High-rated recordings are clean focal recordings; low-rated ones are noisy, distant, or have multiple species. This quality signal could be exploited in several ways:

- [ ] **Quality-weighted loss** — weight each sample's contribution to the loss by its rating (e.g. `loss_weight = 0.5 + 0.5 * rating/5`). Clean recordings get full weight; noisy ones contribute less, reducing gradient noise from mislabeled or ambiguous clips. Simple to implement: just multiply the per-sample loss in `training_step`.
- [ ] **Quality as an input feature** — concatenate a scalar quality embedding (or binned quality level) with the NMF latent features before the classification head. At inference time on soundscapes, use a fixed "soundscape quality" value (e.g. 2.0-3.0). This lets the model learn that low-quality inputs are more ambiguous and should produce softer predictions. Could also condition the model to be more conservative on noisy inputs.
- [ ] **Quality-aware curriculum** — train first on high-quality recordings (rating ≥ 4) for clean gradient signal, then gradually mix in lower-quality recordings. The model learns clean species signatures first, then adapts to noisy conditions. Risk: may overfit to clean recordings early.
- [ ] **Quality-stratified validation** — report val AUC separately for high/low quality clips to understand whether the model is failing on noisy inputs specifically.
- [ ] **Quality-based sampling** — during training, oversample high-quality recordings for rare species (where every clean sample matters) while keeping natural quality distribution for common species.

### 9. Class-balanced sampling / reweighting
- [ ] **Balanced label sampling** — test a sampler that increases exposure of underrepresented labels so all classes are seen more uniformly during training. Important for multi-label data: balance by positive-label coverage, not just by clip count.
- [ ] **Per-class loss weighting** — compare sampler-based balancing against per-class positive weights in the loss. Some labels may benefit more from loss reweighting than aggressive resampling.
- [ ] **Hybrid sampling scheme** — keep some natural-frequency batches for calibration while reserving part of each epoch for rare-label upsampling. Goal: reduce imbalance without making the train distribution too unrealistic.
- [ ] **Taxon-aware sampling** — ensure insects / amphibians / mammals / reptiles are not drowned out by abundant bird labels when constructing batches.

### 9. NMF latent basis — deprioritized
Steps 1-2 were done under the old HTSAT backbone. With BirdSet EfficientNet-B1, the NMF branch is no longer integrated. Keeping for reference only.

- [x] **Step 1: Learn global dictionary W** — done. W_k56.npy in `nmf_analysis/output/`, learned from representative soundscape spectrograms via NMFk.
- [x] **Step 2: Project clips into fixed basis** — done under HTSAT. Not ported to BirdSet EfficientNet-B1.
- [ ] **Steps 3-5** — deprioritized. BirdSet EfficientNet-B1 + temporal attention pooling achieves better val AUC without NMF.

### 10. Post-processing
- [ ] **Threshold tuning** — optimize per-class thresholds on validation set instead of using 0.5 cutoff. Quick win, no retraining.
- [ ] **Spatial/site priors** — build species×location co-occurrence from train.csv, downweight predictions for species never observed near the Pantanal. No retraining needed.
- [ ] **Test-time augmentation (TTA)** — average predictions over original + time-shifted segments. Free accuracy but tight on inference budget.

---

## Tier 3 — Deprioritized / blocked

- [x] **Two-stage temporal fusion** — implemented and removed. Temporal MLP overfits on small soundscape training set, did not improve Kaggle score. Code removed from train.py and dataset.py.
- [ ] **Cross-validation ensemble diversity** — vary hyperparameters across folds. Diminishing returns.
- [ ] **Google Perch v2** — blocked by TensorFlow-only, GPU-only. Worth revisiting if PyTorch port appears.
- [ ] **AudioProtoPNet** — 0.3B params, too large for Kaggle CPU inference.
- [ ] **Knowledge distillation** — no clear teacher model yet.
- [ ] **Multi-scale inference** — experimental, adds inference cost.
