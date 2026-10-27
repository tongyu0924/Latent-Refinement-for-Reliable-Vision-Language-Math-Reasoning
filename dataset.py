import re
import math
import random
import logging
from typing import Dict

import torch
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset

logger = logging.getLogger(__name__)
LETTERS = "ABCDEFGHIJ"


def _is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def safe_image(img: Image.Image, factor: int = 28, max_ratio: float = 180.0) -> Image.Image:
    """Upscale tiny images and pad extreme aspect ratios (Qwen2-VL needs side >= 28, ratio < 200)."""
    img = img.convert("RGB")
    w, h = img.size
    if min(w, h) < 2 * factor:
        s = 2 * factor / max(1, min(w, h))
        w, h = int(math.ceil(w * s)), int(math.ceil(h * s))
        img = img.resize((w, h), Image.BICUBIC)
    if max(w, h) / min(w, h) > max_ratio:
        nw = max(w, int(math.ceil(h / max_ratio)) + 1)
        nh = max(h, int(math.ceil(w / max_ratio)) + 1)
        canvas = Image.new("RGB", (nw, nh), (255, 255, 255))
        canvas.paste(img, ((nw - w) // 2, (nh - h) // 2))
        img = canvas
    return img


def build_prompt(item) -> str:
    q = str(item["question"]).strip()
    if item["question_type"] == "multi_choice" and item.get("choices"):
        opts = "\n".join(f"({LETTERS[i]}) {str(c).strip()}" for i, c in enumerate(item["choices"]))
        return (f"Solve the following visual math problem.\n\nQuestion: {q}\nChoices:\n{opts}\n\n"
                "Respond with only the correct option letter in the form 'Answer: <letter>'.")
    hint = {"integer": "an integer", "float": "a decimal number", "text": "a word or phrase",
            "list": "a Python list"}.get(item.get("answer_type", ""), "a value")
    if item.get("answer_type") == "float" and not _is_missing(item.get("precision")):
        hint += f" with {int(item['precision'])} decimal place(s)"
    unit = item.get("unit")
    unit_s = f" (unit: {unit})" if unit and not _is_missing(unit) else ""
    return (f"Solve the following visual math problem.\n\nQuestion: {q}{unit_s}\n\n"
            f"The final answer should be {hint}. "
            "Respond with only the final answer in the form 'Answer: <answer>'.")


def gold_answer(item) -> str:
    ans = str(item["answer"]).strip()
    if item["question_type"] == "multi_choice" and item.get("choices"):
        choices = [str(c).strip() for c in item["choices"]]
        if ans in choices:
            return LETTERS[choices.index(ans)]
    return ans


def encode_prompt(processor, image, prompt: str):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[safe_image(image)], return_tensors="pt")


class MathVistaDataset(Dataset):
    """MathVista testmini with a fixed-seed train/val split. Each item = prompt (image+question) + 'Answer: X<eos>'."""

    def __init__(self, processor, cfg, split: str = "train"):
        self.processor = processor
        self.tok = processor.tokenizer
        self.max_target_tokens = cfg.max_target_tokens

        raw = load_dataset(cfg.dataset_name, split=cfg.dataset_split,
                           **({"token": cfg.hf_token} if cfg.hf_token else {}))
        idxs = list(range(len(raw)))
        random.Random(cfg.seed).shuffle(idxs)
        if cfg.max_samples is not None:
            idxs = idxs[: cfg.max_samples]
        val_n = max(1, int(round(len(idxs) * cfg.val_ratio)))
        chosen = idxs[:val_n] if split == "val" else idxs[val_n:]
        self.data = raw.select(sorted(chosen))
        logger.info(f"MathVista split={split}, samples={len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        gold = gold_answer(item)
        enc = encode_prompt(self.processor, item["decoded_image"], build_prompt(item))
        prompt_ids = enc["input_ids"][0]

        target = self.tok(f"Answer: {gold}", add_special_tokens=False, truncation=True,
                          max_length=self.max_target_tokens)["input_ids"] + [self.tok.eos_token_id]
        return {
            "input_ids": torch.cat([prompt_ids, torch.tensor(target, dtype=torch.long)]),
            "prompt_len": len(prompt_ids),
            "vision": {k: v for k, v in enc.items() if k not in ("input_ids", "attention_mask")},
            "answer": gold,
            "meta": {
                "pid": item["pid"],
                "question_type": item["question_type"],
                "answer_type": item.get("answer_type"),
                "precision": item.get("precision"),
                "choices": item.get("choices"),
                "task": (item.get("metadata") or {}).get("task", "unknown"),
            },
        }


def collate_list(batch):
    return batch


def extract_answer(text: str, meta: Dict) -> str:
    t = text.strip()
    m = re.search(r"answer\s*[:：]\s*(.*)", t, flags=re.I | re.S)
    s = (m.group(1) if m else t).strip().split("\n")[0].strip().rstrip(".")
    if meta["question_type"] == "multi_choice":
        mm = re.match(r"^\(?([A-J])\)?(?:[\s\.\):,]|$)", s)
        if mm:
            return mm.group(1)
        for i, c in enumerate(meta.get("choices") or []):
            if str(c).strip().lower() == s.lower():
                return LETTERS[i]
        mm = re.search(r"\(([A-J])\)", t)
        return mm.group(1) if mm else s[:1].upper()
    if meta.get("answer_type") in ("integer", "float"):
        nums = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", "")) or \
               re.findall(r"-?\d+(?:\.\d+)?", t.replace(",", ""))
        return nums[0] if nums else s
    return s


def is_correct(pred: str, gold: str, meta: Dict) -> bool:
    if meta["question_type"] == "multi_choice":
        return pred.strip().upper() == gold.strip().upper()
    at = meta.get("answer_type")
    try:
        if at == "integer":
            return int(round(float(pred))) == int(round(float(gold)))
        if at == "float":
            prec = 2 if _is_missing(meta.get("precision")) else int(meta["precision"])
            return round(float(pred), prec) == round(float(gold), prec)
    except (ValueError, TypeError):
        return False
    norm = lambda x: re.sub(r"\s+", "", str(x).lower())
    return norm(pred) == norm(gold)
