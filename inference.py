import os
import random
import logging
import argparse
from pathlib import Path

import numpy as np
import torch

from config import Config, add_config_args, config_from_args
from trainer import ReLAR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")


if __name__ == "__main__":
    parser = add_config_args(argparse.ArgumentParser(description="Evaluate / run a trained checkpoint"))
    parser.add_argument("--ckpt", default="./relar_vlm_mathvista/final_model")
    parser.add_argument("--split", default="val")
    parser.add_argument("--image", default=None, help="single-image mode")
    parser.add_argument("--question", default=None)
    parser.add_argument("--choices", nargs="*", default=None)
    parser.add_argument("--answer_type", default="integer")
    args = parser.parse_args()

    # Load the training config so the val split matches training exactly.
    cfg_path = Path(args.ckpt) / "config.json"
    cfg = Config.load(cfg_path) if cfg_path.exists() else Config()
    cfg = config_from_args(args, base=cfg)
    cfg.hf_token = cfg.hf_token or os.environ.get("HF_TOKEN")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    model = ReLAR(cfg, ckpt_dir=args.ckpt)

    if args.image:
        for refine in (False, True):
            out = model.predict(args.image, args.question, args.choices, args.answer_type, refine=refine)
            print(f"[{'ReLAR' if refine else 'Base '}] {out}")
    else:
        model.evaluate(args.split, max_samples=None, save_name="inference_results.json")
