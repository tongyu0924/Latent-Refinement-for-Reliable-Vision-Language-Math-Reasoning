import os
import random
import logging
import argparse

import numpy as np
import torch

from config import add_config_args, config_from_args
from trainer import ReLAR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = add_config_args(argparse.ArgumentParser(description="Train latent refinement on MathVista"))
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    cfg = config_from_args(args)
    cfg.hf_token = cfg.hf_token or os.environ.get("HF_TOKEN")
    set_seed(cfg.seed)

    model = ReLAR(cfg)
    model.train()
    if not args.skip_eval:
        model.evaluate("val", max_samples=cfg.eval_max_samples)
