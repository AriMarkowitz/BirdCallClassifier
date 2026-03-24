"""Bird-MAE configuration — standalone version (no HuggingFace dependency)."""

from dataclasses import dataclass


@dataclass
class BirdMAEConfig:
    img_size_x: int = 512
    img_size_y: int = 128
    patch_size: int = 16
    in_chans: int = 1
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    pos_trainable: bool = False
    qkv_bias: bool = True
    qk_norm: bool = False
    init_values: float = None
    drop_rate: float = 0.0
    norm_layer_eps: float = 1e-6
    global_pool: str = "mean"

    # Feature extractor settings
    sampling_rate: int = 32000
    num_mel_bins: int = 128
    target_length: int = 512
    fbank_mean: float = -7.2
    fbank_std: float = 4.43

    @property
    def num_patches_x(self):
        return self.img_size_x // self.patch_size

    @property
    def num_patches_y(self):
        return self.img_size_y // self.patch_size

    @property
    def num_patches(self):
        return self.num_patches_x * self.num_patches_y

    @property
    def pos_drop_rate(self):
        return self.drop_rate

    @property
    def attn_drop_rate(self):
        return self.drop_rate

    @property
    def drop_path_rate(self):
        return self.drop_rate

    @property
    def proj_drop_rate(self):
        return self.drop_rate
