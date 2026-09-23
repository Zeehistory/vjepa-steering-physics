"""Config loading and schema definitions.

Configs are OmegaConf YAML files. We keep lightweight dataclass schemas so that defaults are explicit
and discoverable in code, while still allowing free-form experiment overrides from YAML and the CLI.

The canonical entry point is :func:`load_config`, which merges (in order):

1. dataclass defaults (:class:`ExperimentConfig`),
2. the YAML file,
3. any ``key=value`` dotlist overrides (e.g. from ``argparse`` remainder args).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass
class EncoderConfig:
    name: str = "mock"
    # HuggingFace id used by the real wrappers, e.g. "facebook/vjepa2-vitl-fpc64-256".
    hf_model_id: str | None = None
    hidden_dim: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    patch_size: int = 16
    tubelet_size: int = 2
    image_size: int = 256
    num_frames: int = 16
    frozen: bool = True
    # Which transformer block outputs to extract. "all" or a list of ints.
    layers: Any = "all"
    # Where inside each block to tap. "block" = the block's output, i.e. the residual stream after the
    # block (our default). A dotted submodule path such as "mlp.fc2" taps that submodule's output
    # instead -- the MLP write rather than the running residual. Musa et al. (ICML 2026) hook
    # `encoder.layer.N.mlp.fc2`, and the two differ sharply: the residual stream carries a large
    # near-constant component that dominates any per-clip PCA, while the MLP write does not.
    hook_site: str = "block"
    extract_attention: bool = False
    device: str = "auto"
    dtype: str = "float32"


@dataclass
class DecoderConfig:
    name: str = "transformer"
    mode: str = "reconstruct"  # reconstruct | future | state | diagram
    hidden_dim: int = 512
    depth: int = 8
    heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    num_query_tokens_per_frame: int = 64  # learned video query tokens per output frame
    use_layer_embedding: bool = True
    gradient_checkpointing: bool = False
    out_image_size: int = 64
    out_num_frames: int = 16
    out_channels: int = 3
    # Frame head. "patch" = the original per-query linear -> pixel block + 2-conv refine (default, so
    # every existing checkpoint loads). "convup" = token grid -> progressive x2 PixelShuffle upsampler
    # with residual blocks: no patch boundaries, wide receptive field per output pixel.
    frame_head: str = "patch"
    head_width: int = 256          # convup only: channel width at the token-grid resolution
    # mode == "state" head configuration
    state_dim: int = 0  # filled in from the dataset's state vector length
    # mode == "future": number of context frames whose latents are visible to the decoder
    context_frames: int = 8


@dataclass
class DataConfig:
    name: str = "synthetic_physics"
    root: str | None = None
    image_size: int = 64
    num_frames: int = 16
    fps: int = 8
    categories: Any = "all"
    split: str = "train"
    # synthetic generator knobs
    num_clips: int = 64
    scenarios: Any = field(default_factory=lambda: ["bouncing_ball", "projectile"])
    seed: int = 0
    ball_radius: float = 0.14  # mujoco_physics ball size
    # moving_ball (Step 2 velocity-first) knobs
    scenario: str = "constant_velocity"
    speed_range: Any = field(default_factory=lambda: [0.010, 0.035])
    radius_range: Any = field(default_factory=lambda: [0.07, 0.10])
    accel_range: Any = field(default_factory=lambda: [0.0015, 0.0035])  # scene_accel2d |a| range
    gravity_range: Any = field(default_factory=lambda: [0.0015, 0.0040])  # scene_gravity |g| (down) range
    omega_range: Any = field(default_factory=lambda: [0.06, 0.20])  # scene_angvel2d |omega| (rad/frame) range
    omega0_range: Any = field(default_factory=lambda: [-0.06, 0.06])  # scene_angaccel2d initial omega range
    alpha_range: Any = field(default_factory=lambda: [0.005, 0.014])  # scene_angaccel2d |alpha| (rad/frame^2)
    fixed_speed: float = 0.022
    camera_rotation: bool = False
    clips_per_scene: int = 4  # scene_velocity: clips per scene (shared first frame, only speed varies)
    shape: str = "disk"  # rendered object: "disk" (default) or "square" (cross-object control)
    # spin_ball3d (the velocity x spin CROSSTALK scene) knobs. Per scene the clips form a complete
    # n_vel x n_spin factorial over two INDEPENDENT quantities, so clips_per_scene is DERIVED
    # (= n_vel*n_spin) rather than set: an incomplete grid would leave the SV/VS commutation square
    # without a ground-truth corner. omega_range above is reused for the |spin| range.
    n_vel: int = 4    # distinct translation velocity vectors per scene
    n_spin: int = 4   # distinct signed spin rates per scene
    # paddle_strike (action -> outcome) knobs. These must exist HERE, not only in
    # configs/data/paddle_strike.yaml: DataConfig is a STRUCTURED config, so omegaconf validates
    # against these fields and a key present in the YAML but absent here raises ConfigKeyError as soon
    # as it is passed as an override (data.v_in_range=...) -- which is exactly how extract_latents is
    # driven per split.
    v_in_range: Any = field(default_factory=lambda: [0.85, 1.15])   # WORLD inflow speed, m/s
    ratio_range: Any = field(default_factory=lambda: [0.5, 3.0])    # |v_out| / v_in, certified envelope
    embodiment: str = "paddle"        # "paddle" (abstract striker) or "franka" (arm holding it)
    mixed_appearance: bool = False    # resample ball/table shade per scene (appearance nuisance)


@dataclass
class LossConfig:
    charbonnier: float = 1.0
    foreground: float = 0.0          # weight on foreground-weighted charbonnier (small-object scenes)
    foreground_gamma: float = 50.0   # darkness up-weight factor: w = 1 + gamma*(1-target)
    # Pixel-target compression: map targets into [target_lo, target_hi] before the pixel/structure
    # losses. Defaults (0,1) are a no-op. For pure-white/black scenes a sigmoid-output decoder can only
    # match target 1.0 by driving its logit to +inf, where it saturates (grad ~1e-3) and the
    # (small-object) foreground signal can't pull it back -> uniform collapse. Compressing the white
    # background to e.g. 0.95 puts the optimum at a finite logit (~2.9, grad ~0.05) so gradients stay
    # alive and the ball is actually rendered.
    target_lo: float = 0.0
    target_hi: float = 1.0
    ssim: float = 0.0
    ms_ssim: float = 0.0
    lpips: float = 0.0
    temporal_consistency: float = 0.0
    state: float = 0.0
    trajectory: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0
    collision: float = 0.0
    # Per-frame rendered-ball motion faithfulness (kills temporal-average smear): tie the decoded
    # ball's soft centroid to GT position every frame; penalize dark mass spread beyond a disk.
    frame_position: float = 0.0
    frame_spread: float = 0.0
    frame_spread_max_var: float = 0.004
    # Per-frame rendered-ORIENTATION faithfulness (angular analog of frame_position): tie the rendered
    # marker's unit direction to GT (cos,sin)theta every frame so decoded angular velocity is faithful.
    frame_orientation_projected: float = 0.0
    frame_orientation_body_thresh: float = 0.5
    frame_orientation_marker_thresh: float = 0.18
    marker: float = 0.0
    marker_redness_thresh: float = 0.12
    frame_orientation: float = 0.0
    frame_orientation_body_power: float = 1.0


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 0.05
    betas: Any = field(default_factory=lambda: [0.9, 0.95])
    grad_clip: float = 1.0
    warmup_steps: int = 50
    max_steps: int = 200
    scheduler: str = "cosine"  # cosine | constant | linear
    grad_accum: int = 1
    ema_decay: float = 0.999


@dataclass
class TrainConfig:
    batch_size: int = 4
    num_workers: int = 0
    # Cap on decoded latent shards held in RAM (LRU). LatentDataset itself defaults to UNBOUNDED,
    # which quietly OOM-kills any full-split training run: a 4000-clip / 4-layer fp32 cache is ~172 GB.
    max_cached_shards: int = 16
    mixed_precision: str = "no"  # no | fp16 | bf16
    log_every: int = 10
    ckpt_every: int = 100
    eval_every: int = 100
    resume: str | None = None
    # Warm-start WEIGHTS ONLY from another run's checkpoint: no optimizer, no scheduler, step stays 0.
    # `resume` cannot do this -- it restores the stored step, so pointing it at a finished 6000-step run
    # with max_steps 6000 exits immediately. Needed to add a loss term to an already-good model instead
    # of retraining from scratch with that term dominating from step 0. Ignored when `resume` is set.
    init_from: str | None = None
    seed: int = 0
    deterministic: bool = True


@dataclass
class ExperimentConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "outputs/run"
    latent_dir: str | None = None
    wandb: bool = False
    tags: Any = field(default_factory=list)


def _defaults() -> DictConfig:
    cfg = OmegaConf.structured(ExperimentConfig())
    assert isinstance(cfg, DictConfig)
    return cfg


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Merge dataclass defaults, an optional YAML file, and CLI dotlist overrides."""
    cfg = _defaults()
    if path is not None:
        file_cfg = OmegaConf.load(str(path))
        cfg = OmegaConf.merge(cfg, file_cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    assert isinstance(cfg, DictConfig)
    return cfg


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Snapshot a resolved config to disk (used for every run)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=str(path))


def to_container(cfg: DictConfig) -> dict[str, Any]:
    out = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(out, dict)
    return out
