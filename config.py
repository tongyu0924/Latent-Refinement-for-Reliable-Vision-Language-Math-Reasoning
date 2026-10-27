import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, fields
from typing import Optional


@dataclass
class Config:
    # model / data
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    dataset_name: str = "AI4Math/MathVista"
    dataset_split: str = "testmini"
    hf_token: Optional[str] = None
    max_samples: Optional[int] = None
    val_ratio: float = 0.2
    seed: int = 42
    min_pixels: int = 64 * 28 * 28
    max_pixels: int = 512 * 28 * 28
    max_target_tokens: int = 32

    # latent refinement / RL
    beta: float = 0.1             # KL penalty
    max_depth: int = 3
    depth_cost: float = 0.02
    entropy_bonus: float = 0.01
    depth_rl_weight: float = 0.15
    critic_weight: float = 0.2
    action_dim: int = 64
    state_dim: int = 256
    ablation: str = "none"        # "none" | "sft_only" (depth fixed to 0, no RL)

    # training
    use_lora: bool = False
    lora_r: int = 16
    epochs: int = 3
    batch_size: int = 4
    lr: float = 1e-4
    output_dir: str = "./relar_vlm_mathvista"

    # evaluation
    eval_max_samples: int = 200
    max_gen_tokens: int = 24

    def save(self, path):
        d = asdict(self)
        d.pop("hf_token", None)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path, **overrides):
        raw = json.load(open(path))
        raw = {k.lower(): v for k, v in raw.items()}   # also accepts upper-case keys from the Colab script
        names = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in raw.items() if k in names})
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg


def _parse_value(s: str, default):
    if s.lower() == "none":
        return None
    if isinstance(default, bool):
        return s.lower() in ("1", "true", "yes", "y")
    if isinstance(default, int):
        return int(s)
    if isinstance(default, float):
        return float(s)
    if default is None:
        try:
            return int(s)
        except ValueError:
            return s
    return s


def add_config_args(parser: argparse.ArgumentParser):
    """Expose every Config field as --field_name."""
    for f in fields(Config):
        parser.add_argument(f"--{f.name}", type=str, default=None)
    return parser


def config_from_args(args, base: Optional[Config] = None) -> Config:
    cfg = base or Config()
    defaults = Config()
    for f in fields(Config):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(cfg, f.name, _parse_value(v, getattr(defaults, f.name)))
    return cfg
