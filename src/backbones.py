"""
Backbone factory for BirdCLEFModel.

Each backbone takes a log-mel spectrogram (B, 1, n_mels, T_frames) and returns
a 4-D feature map (B, C, H, W) suitable for TemporalAttentionPool. Frequency
structure H is collapsed later; the only contract is: (B, C, H, W).

Optionally, a backbone may expose `birdset_logits(feat_map) -> (B, K)` for
teacher-student distillation against the BirdSet XCL classifier. Backbones
without BirdSet weights return None and the `--distill` flag becomes a no-op.

Adding a new backbone:
  1. Write `_build_<name>()` returning (module, out_channels, birdset_logits_fn_or_None)
  2. Register it in BACKBONES
  3. Done — train.py picks it up via `--backbone <name>`

For HuggingFace audio models: use `hf:Org/Model` as the backbone name;
`_build_hf()` will try to load via `AutoModel.from_pretrained(...)` and return
its last hidden state reshaped to 4-D.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _find_local_birdset_config(hf_model_name):
    """Look for a sidecar config.json next to backbones.py (for offline Kaggle inference).

    submit.sh ships configs to kaggle_dataset/configs/<sanitized-name>/config.json so
    inference works without internet. Returns dir path if found, else None.
    """
    import os
    safe = hf_model_name.replace("/", "__")
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, "configs", safe)
    if os.path.isfile(os.path.join(candidate, "config.json")):
        return candidate
    return None


class BirdsetEfficientNet(nn.Module):
    """BirdSet EfficientNet backbone (B0 or B1) — exposes XCL classifier for distillation."""

    def __init__(self, hf_model_name, pretrained=True):
        super().__init__()
        from transformers import EfficientNetForImageClassification, EfficientNetConfig
        if pretrained:
            self.hf_model = EfficientNetForImageClassification.from_pretrained(
                hf_model_name, num_channels=1, ignore_mismatched_sizes=True,
            )
        else:
            # Inference path: prefer local sidecar config (offline-safe for Kaggle).
            local = _find_local_birdset_config(hf_model_name)
            if local is not None:
                cfg = EfficientNetConfig.from_pretrained(local, local_files_only=True)
            else:
                cfg = EfficientNetConfig.from_pretrained(hf_model_name, local_files_only=True)
            cfg.num_channels = 1
            self.hf_model = EfficientNetForImageClassification(cfg)
        self.out_channels = int(self.hf_model.config.hidden_dim)
        self.birdset_num_classes = int(self.hf_model.config.num_labels)

    def forward(self, mel):
        enc = self.hf_model.efficientnet.encoder(
            self.hf_model.efficientnet.embeddings(mel),
            return_dict=True,
        )
        return enc.last_hidden_state  # (B, C, H, W)

    def birdset_logits(self, feat_map):
        pooled = self.hf_model.efficientnet.pooler(feat_map)
        pooled = pooled.reshape(pooled.shape[:2])
        return self.hf_model.classifier(self.hf_model.dropout(pooled))


class TimmBackbone(nn.Module):
    """Generic timm CNN backbone, 1-channel input, returns feature map.

    Works for: efficientnet_b0, mobilenetv3_small_100, resnet18, convnext_tiny, etc.
    Fails for ViT-style models that need per-token sequence handling (add a
    separate backbone class for those).
    """

    def __init__(self, timm_name, pretrained=True):
        super().__init__()
        import timm
        self.model = timm.create_model(
            timm_name,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,           # drop classifier
            features_only=False,     # we want the last feature map, not the pyramid
            global_pool="",          # disable global pooling, keep spatial dims
        )
        self.out_channels = int(self.model.num_features)

    def forward(self, mel):
        return self.model.forward_features(mel)

    def birdset_logits(self, feat_map):
        return None


def _build_birdset_b1(pretrained=True):
    return BirdsetEfficientNet("DBD-research-group/EfficientNet-B1-BirdSet-XCL", pretrained)


def _build_birdset_b0(pretrained=True):
    # DBD-research-group may not publish B0 — will fall back to timm if it fails.
    try:
        return BirdsetEfficientNet("DBD-research-group/EfficientNet-B0-BirdSet-XCL", pretrained)
    except Exception as exc:
        logger.warning(
            f"BirdSet B0 load failed ({exc}); falling back to timm efficientnet_b0 "
            f"(no BirdSet distillation for this model)"
        )
        return TimmBackbone("efficientnet_b0", pretrained)


def _build_efficientnet_b0(pretrained=True):
    return TimmBackbone("efficientnet_b0", pretrained)


def _build_mobilenetv3_small(pretrained=True):
    return TimmBackbone("mobilenetv3_small_100", pretrained)


def _build_mobilenetv3_large(pretrained=True):
    return TimmBackbone("mobilenetv3_large_100", pretrained)


def _build_resnet18(pretrained=True):
    return TimmBackbone("resnet18", pretrained)


def _build_convnext_tiny(pretrained=True):
    return TimmBackbone("convnext_tiny", pretrained)


def _build_regnety_002(pretrained=True):
    return TimmBackbone("regnety_002", pretrained)


def _build_efficientnetv2_b0(pretrained=True):
    return TimmBackbone("tf_efficientnetv2_b0", pretrained)


def _build_hf(hf_name, pretrained=True):
    """Generic HuggingFace audio model fallback. Best-effort."""
    from transformers import AutoModel, AutoConfig
    if pretrained:
        model = AutoModel.from_pretrained(hf_name)
    else:
        cfg = AutoConfig.from_pretrained(hf_name)
        model = AutoModel.from_config(cfg)

    class _HFWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            # Best effort: pick hidden size from config
            cfg = m.config
            self.out_channels = int(
                getattr(cfg, "hidden_size", None)
                or getattr(cfg, "hidden_dim", None)
                or getattr(cfg, "embed_dim", None)
                or 768
            )

        def forward(self, mel):
            out = self.m(mel) if mel.ndim == 2 else self.m(pixel_values=mel)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            if hidden.ndim == 3:
                # (B, T, C) -> (B, C, 1, T) so TemporalAttentionPool can consume it
                hidden = hidden.transpose(1, 2).unsqueeze(2)
            return hidden

        def birdset_logits(self, feat_map):
            return None

    return _HFWrapper(model)


BACKBONES = {
    "birdset_b1": _build_birdset_b1,
    "birdset_b0": _build_birdset_b0,
    "efficientnet_b0": _build_efficientnet_b0,
    "efficientnetv2_b0": _build_efficientnetv2_b0,
    "mobilenetv3_small": _build_mobilenetv3_small,
    "mobilenetv3_large": _build_mobilenetv3_large,
    "resnet18": _build_resnet18,
    "convnext_tiny": _build_convnext_tiny,
    "regnety_002": _build_regnety_002,
}


def build_backbone(name, pretrained=True):
    """Build a backbone by name. Returns nn.Module with `.forward(mel)`,
    `.out_channels`, and optionally `.birdset_logits(feat_map)`.

    Supports: birdset_b1, birdset_b0, efficientnet_b0, mobilenetv3_small,
    resnet18, convnext_tiny. Use `hf:<Org>/<Model>` for arbitrary HF models.
    """
    if name.startswith("hf:"):
        return _build_hf(name[3:], pretrained=pretrained)
    if name not in BACKBONES:
        raise ValueError(
            f"Unknown backbone '{name}'. "
            f"Supported: {sorted(BACKBONES)} (or 'hf:<Org>/<Model>')"
        )
    return BACKBONES[name](pretrained=pretrained)
