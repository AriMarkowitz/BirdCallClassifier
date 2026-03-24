# NMF Global Latent Basis Analysis

Learn a fixed global NMF dictionary W from representative spectrogram data,
then use it to generate per-clip latent targets for training a CNN encoder.

## Pipeline

1. `build_spectrogram_matrix.py` — stratified sampling of train_audio + soundscapes,
   compute mel spectrograms, concatenate into matrix V (f x T).
2. `run_nmfk.py` — run NMFk on V to select optimal k and learn W.
3. `project_clips.py` — project all training clips into the fixed basis (solve for H given W).
4. `extract_latent_features.py` — summarize H into fixed-size feature vectors per clip.

## Requirements

Additional to the main project:
- NMFk (pip install nmfk) or scikit-learn NMF as fallback
