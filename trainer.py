import csv
import json
import time
import logging
import contextlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from tqdm.auto import tqdm
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from models import (InitialStateExtractor, DepthController, ActionController,
                    HiddenRefiner, DecodeBridge, ValueCritic, RefinedLMHead)
from dataset import (MathVistaDataset, collate_list, extract_answer, is_correct,
                     build_prompt, encode_prompt)

logger = logging.getLogger(__name__)


def load_vlm(model_name: str, dtype, hf_token=None):
    kw = {"torch_dtype": dtype, **({"token": hf_token} if hf_token else {})}
    name = model_name.lower()
    if "qwen2.5-vl" in name:
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
    elif "qwen2-vl" in name:
        from transformers import Qwen2VLForConditionalGeneration as Cls
    else:
        try:
            from transformers import AutoModelForImageTextToText as Cls
        except ImportError:
            from transformers import AutoModelForVision2Seq as Cls
    try:
        return Cls.from_pretrained(model_name, attn_implementation="sdpa", **kw)
    except Exception:
        return Cls.from_pretrained(model_name, **kw)


class ReLAR:
    """Latent refinement on top of a (frozen or LoRA-tuned) VLM. Handles training, evaluation and inference."""

    MODULES = ["state_extractor", "depth_controller", "action_controller", "refiner", "decode_bridge", "critic"]

    def __init__(self, cfg: Config, ckpt_dir: Optional[str] = None):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        self.dtype = torch.bfloat16 if bf16 else torch.float32   # fp16 overflows on Qwen2-VL
        logger.info(f"device={self.device} dtype={self.dtype} model={cfg.model_name} ablation={cfg.ablation}")

        pkw = {"min_pixels": cfg.min_pixels, "max_pixels": cfg.max_pixels} if "qwen" in cfg.model_name.lower() else {}
        self.processor = AutoProcessor.from_pretrained(
            cfg.model_name, **pkw, **({"token": cfg.hf_token} if cfg.hf_token else {}))
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = load_vlm(cfg.model_name, self.dtype, cfg.hf_token).to(self.device)
        model.config.use_cache = False
        gc = model.generation_config
        gc.temperature, gc.top_p, gc.top_k = None, None, None
        for p in model.parameters():
            p.requires_grad = False

        if cfg.use_lora:
            if ckpt_dir is not None:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(Path(ckpt_dir) / "lora_adapter"))
            else:
                from peft import LoraConfig, get_peft_model
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model.enable_input_require_grads()
                model = get_peft_model(model, LoraConfig(
                    r=cfg.lora_r, lora_alpha=cfg.lora_r * 2, lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))   # LLM only
                model.print_trainable_parameters()
        self.vlm = model.to(self.device)

        H = self._core().get_output_embeddings().weight.shape[1]
        S, A = cfg.state_dim, cfg.action_dim
        self.state_extractor = InitialStateExtractor(H, S).to(self.device)
        self.depth_controller = DepthController(S, cfg.max_depth).to(self.device)
        self.action_controller = ActionController(S, A).to(self.device)
        self.refiner = HiddenRefiner(H, S, A).to(self.device)
        self.decode_bridge = DecodeBridge(S, H).to(self.device)
        self.critic = ValueCritic(S).to(self.device)

        if ckpt_dir is not None:
            for name in self.MODULES:
                getattr(self, name).load_state_dict(torch.load(Path(ckpt_dir) / f"{name}.pt", map_location=self.device))
            logger.info(f"loaded checkpoint: {ckpt_dir}")

        self._step = 0
        self._hist = {k: [] for k in ["step", "total", "lm", "kl", "rl", "critic", "depth", "entropy", "reward"]}

    # ---------------------------------------------------------------- helpers
    def _core(self):
        return self.vlm.get_base_model() if hasattr(self.vlm, "get_base_model") else self.vlm

    def _base_ctx(self):
        """Context in which the VLM behaves as the original (reference) model."""
        return self.vlm.disable_adapter() if self.cfg.use_lora else contextlib.nullcontext()

    def _set_train(self, flag: bool):
        for n in self.MODULES:
            getattr(self, n).train(flag)
        self.vlm.train(flag and self.cfg.use_lora)

    def _to_device(self, inp: Dict, full: bool = True) -> Dict:
        ids = inp["input_ids"] if full else inp["input_ids"][: inp["prompt_len"]]
        out = {"input_ids": ids.unsqueeze(0).to(self.device),
               "attention_mask": torch.ones(1, len(ids), dtype=torch.long, device=self.device)}
        for k, v in inp["vision"].items():
            v = v.to(self.device)
            out[k] = v.to(self.dtype) if v.is_floating_point() else v
        return out

    def _decode_logits(self, st, h):
        head = self._core().get_output_embeddings()
        return head(self.decode_bridge(st, h).to(head.weight.dtype))

    @staticmethod
    def _logp_and_kl(logits, ref_logits, targets):
        plp = F.log_softmax(logits.float(), -1)
        rlp = F.log_softmax(ref_logits.float(), -1)
        lp = plp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()
        kl = (plp.exp() * (plp - rlp)).sum(-1).mean()
        lp = torch.where(torch.isfinite(lp), lp, torch.full_like(lp, -5.0))
        kl = torch.where(torch.isfinite(kl), kl, torch.full_like(kl, 0.1))
        return lp, kl

    def _shaped_return(self, logp_seq: List[float], final_kl: float, total_H: float, d: int) -> float:
        g = -self.cfg.beta * final_kl + self.cfg.entropy_bonus * total_H
        if d == 0:
            return g
        return sum((logp_seq[t + 1] - logp_seq[t] - self.cfg.depth_cost) / (t + 1) + g / d for t in range(d))

    # ------------------------------------------------------- latent refinement
    def refine(self, h0, mask, deterministic: bool, step_cb=None):
        """s0 -> sample depth d -> d steps of (action, refine h, update s). Returns s0, s_d, d, log-prob, entropy."""
        s0 = self.state_extractor(h0, mask)
        d_s, d_lp, _, d_H = self.depth_controller(s0.unsqueeze(0), deterministic)
        d = 0 if self.cfg.ablation == "sft_only" else max(0, min(int(d_s.item()), self.cfg.max_depth))
        st, ht = s0, h0
        logp, H = d_lp[0], d_H[0]
        for _ in range(d):
            gamma, beta, v, a_lp, a_H = self.action_controller(st, deterministic)
            logp, H = logp + a_lp, H + a_H
            ht = self.refiner.refine(ht, st, gamma, beta, v, mask)
            st = self.refiner.update_state(st, ht, mask)
            if step_cb is not None:
                step_cb(st)
        return s0, st, d, logp, H

    def compute_sample(self, inp: Dict, is_train: bool = True) -> Optional[Dict]:
        inputs = self._to_device(inp, full=True)
        L, P = inputs["input_ids"].shape[1], inp["prompt_len"]
        targets = inputs["input_ids"][0, P:]
        pos = torch.arange(P - 1, L - 1, device=self.device)      # positions predicting the answer
        mask = torch.zeros(L, dtype=torch.long, device=self.device)
        mask[:P] = 1                                               # state only sees image + question

        with torch.set_grad_enabled(is_train and self.cfg.use_lora):
            out = self.vlm(**inputs, output_hidden_states=True, use_cache=False)
        h0 = out.hidden_states[-1][0]
        if self.cfg.use_lora:
            with torch.no_grad(), self._base_ctx():
                ref_logits = self.vlm(**inputs, use_cache=False).logits[0, pos].float()
        else:
            ref_logits = out.logits[0, pos].detach().float()
        del out
        h0_pos = h0[pos]

        with torch.no_grad():
            base_logp = float(F.log_softmax(ref_logits, -1).gather(-1, targets.unsqueeze(-1)).mean())

        logp_seq: List[float] = []

        def _cb(st):
            with torch.no_grad():
                logp_seq.append(float(self._logp_and_kl(self._decode_logits(st.detach(), h0_pos.detach()),
                                                        ref_logits, targets)[0]))

        _cb(self.state_extractor(h0, mask))
        s0, st, d, joint_logp, ent = self.refine(h0, mask, deterministic=not is_train, step_cb=_cb)

        lm_logp, lm_kl = self._logp_and_kl(self._decode_logits(st, h0_pos), ref_logits, targets)
        lm_loss = -lm_logp + self.cfg.beta * lm_kl
        if not torch.isfinite(lm_loss):
            return None

        R = self._shaped_return(logp_seq, float(lm_kl), float(ent), d)
        v = self.critic(s0.detach())
        adv = torch.tensor(float(np.clip(R - float(v.detach()), -5.0, 5.0)), device=self.device)
        rl_loss = -(adv * joint_logp) - self.cfg.entropy_bonus * ent
        critic_loss = F.mse_loss(v, torch.tensor(float(R), device=self.device))

        return {"lm_loss": lm_loss, "rl_loss": rl_loss, "critic_loss": critic_loss,
                "lm_logp": float(lm_logp), "base_logp": base_logp, "lm_kl": float(lm_kl),
                "depth": d, "entropy": float(ent), "reward": float(R)}

    # ---------------------------------------------------------------- training
    def train(self):
        cfg = self.cfg
        loader = DataLoader(MathVistaDataset(self.processor, cfg, "train"), batch_size=cfg.batch_size,
                            shuffle=True, collate_fn=collate_list, num_workers=2)
        groups = [(self.state_extractor, 2.0), (self.depth_controller, 3.0), (self.action_controller, 3.0),
                  (self.refiner, 2.0), (self.decode_bridge, 2.0), (self.critic, 3.0)]
        param_groups = [{"params": list(m.parameters()), "lr": cfg.lr * k, "weight_decay": 0.01} for m, k in groups]
        if cfg.use_lora:
            param_groups.append({"params": [p for p in self.vlm.parameters() if p.requires_grad],
                                 "lr": cfg.lr * 0.1, "weight_decay": 0.0})
        opt = AdamW(param_groups)
        params = [p for g in param_groups for p in g["params"]]
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for ep in range(cfg.epochs):
            self._set_train(True)
            ep_loss, nb = 0.0, 0
            pbar = tqdm(loader, desc=f"Epoch {ep + 1}/{cfg.epochs}")
            for batch in pbar:
                res = []
                for inp in batch:
                    try:
                        r = self.compute_sample(inp, is_train=True)
                    except torch.cuda.OutOfMemoryError:
                        logger.warning("OOM, sample skipped")
                        torch.cuda.empty_cache()
                        r = None
                    if r is not None:
                        res.append(r)
                if not res:
                    continue

                lm = torch.stack([r["lm_loss"] for r in res]).mean()
                rl = torch.stack([r["rl_loss"] for r in res]).mean()
                cr = torch.stack([r["critic_loss"] for r in res]).mean()
                total = lm if cfg.ablation == "sft_only" else lm + cfg.depth_rl_weight * rl + cfg.critic_weight * cr
                if not torch.isfinite(total):
                    continue

                opt.zero_grad(set_to_none=True)
                total.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()

                ep_loss += float(total)
                nb += 1
                self._step += 1
                avg = lambda k: float(np.mean([r[k] for r in res]))
                self._record(float(total), float(lm), avg("lm_kl"), float(rl), float(cr),
                             avg("depth"), avg("entropy"), avg("reward"))
                pbar.set_postfix(loss=f"{float(total):.3f}", logp=f"{avg('lm_logp'):.3f}",
                                 base=f"{avg('base_logp'):.3f}", R=f"{avg('reward'):.3f}",
                                 d=f"{avg('depth'):.2f}", g=f"{float(gnorm):.2f}")

            logger.info(f"epoch {ep + 1} avg_loss={ep_loss / max(nb, 1):.4f}")
            self.save(out_dir / f"checkpoint-epoch-{ep + 1}")

        self._save_history(out_dir)
        self.save(out_dir / "final_model")

    # -------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, inp: Dict, refine: bool = True, max_new_tokens: int = 24) -> Tuple[str, float, int]:
        """Greedy decoding. refine=False -> original VLM; refine=True -> decode from f_decode(s_d, h)."""
        inputs = self._to_device(inp, full=False)
        core = self._core()
        base_head = core.lm_head
        wrapper = RefinedLMHead(base_head, self.decode_bridge)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0, d = time.time(), 0
        if refine:
            h0 = self.vlm(**inputs, output_hidden_states=True, use_cache=False).hidden_states[-1][0]
            mask = torch.ones(h0.shape[0], dtype=torch.long, device=self.device)
            _, wrapper.st, d, _, _ = self.refine(h0, mask, deterministic=True)
        try:
            core.lm_head = wrapper
            with (contextlib.nullcontext() if refine else self._base_ctx()):
                gen = self.vlm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                        use_cache=True, pad_token_id=self.tokenizer.pad_token_id)
        finally:
            core.lm_head = base_head
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        text = self.tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text, time.time() - t0, d

    def predict(self, image, question: str, choices: Optional[List[str]] = None,
                answer_type: str = "integer", refine: bool = True) -> str:
        if isinstance(image, str):
            image = Image.open(image)
        item = {"question": question, "choices": choices, "answer_type": answer_type,
                "question_type": "multi_choice" if choices else "free_form", "precision": None, "unit": None}
        enc = encode_prompt(self.processor, image, build_prompt(item))
        inp = {"input_ids": enc["input_ids"][0], "prompt_len": enc["input_ids"].shape[1],
               "vision": {k: v for k, v in enc.items() if k not in ("input_ids", "attention_mask")}}
        self._set_train(False)
        return self.generate(inp, refine, self.cfg.max_gen_tokens)[0]

    # -------------------------------------------------------------- evaluation
    def evaluate(self, split: str = "val", max_samples: Optional[int] = None,
                 save_name: str = "eval_results.json") -> Dict:
        cfg = self.cfg
        ds = MathVistaDataset(self.processor, cfg, split)
        n = min(len(ds), max_samples or len(ds))
        self._set_train(False)
        st = defaultdict(float)
        by_cat = defaultdict(lambda: [0, 0, 0])
        depth_hist = defaultdict(int)
        records, n_err = [], 0

        for i in tqdm(range(n), desc=f"Eval [{split}]"):
            try:
                item = ds[i]
                with torch.no_grad():
                    r = self.compute_sample(item, is_train=False)
                base_txt, t_b, _ = self.generate(item, refine=False, max_new_tokens=cfg.max_gen_tokens)
                relar_txt, t_r, d = self.generate(item, refine=True, max_new_tokens=cfg.max_gen_tokens)
            except Exception as e:
                n_err += 1
                logger.warning(f"[{i}] failed: {type(e).__name__}: {e} (counted as wrong)")
                records.append({"pid": ds.data[i]["pid"], "error": str(e),
                                "base_correct": False, "relar_correct": False})
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            meta, gold = item["meta"], item["answer"]
            pb, pr = extract_answer(base_txt, meta), extract_answer(relar_txt, meta)
            cb, cr = is_correct(pb, gold, meta), is_correct(pr, gold, meta)
            st["base"] += cb; st["relar"] += cr; st["ok"] += 1
            st["t_base"] += t_b; st["t_relar"] += t_r; st["depth"] += d
            if r is not None:
                st["lp_base"] += r["base_logp"]; st["lp_relar"] += r["lm_logp"]
            depth_hist[d] += 1
            for key in (meta["task"], meta["question_type"]):
                by_cat[key][0] += 1; by_cat[key][1] += cb; by_cat[key][2] += cr
            records.append({"pid": meta["pid"], "task": meta["task"], "question_type": meta["question_type"],
                            "gold": gold, "base_pred": pb, "relar_pred": pr, "base_raw": base_txt,
                            "relar_raw": relar_txt, "base_correct": bool(cb), "relar_correct": bool(cr),
                            "depth": d})

        c, ok = max(n, 1), max(st["ok"], 1)
        summary = {
            "split": split, "n": n, "n_errors": n_err, "ablation": cfg.ablation,
            "acc_base_vlm": st["base"] / c, "acc_relar": st["relar"] / c,
            "answer_logp_base": st["lp_base"] / ok, "answer_logp_relar": st["lp_relar"] / ok,
            "avg_depth": st["depth"] / ok, "depth_distribution": dict(sorted(depth_hist.items())),
            "relative_time": st["t_relar"] / max(st["t_base"], 1e-8),
            "by_category": {k: {"n": v[0], "acc_base": v[1] / v[0], "acc_relar": v[2] / v[0]}
                            for k, v in sorted(by_cat.items())},
        }
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / save_name, "w") as f:
            json.dump({"summary": summary, "records": records}, f, indent=2, ensure_ascii=False)
        self._print_summary(summary)
        return summary

    @staticmethod
    def _print_summary(s: Dict):
        print("=" * 72)
        print(f"MathVista [{s['split']}] n={s['n']} errors={s['n_errors']} ablation={s['ablation']}")
        print(f"  Base VLM accuracy : {s['acc_base_vlm'] * 100:6.2f}%")
        print(f"  ReLAR accuracy    : {s['acc_relar'] * 100:6.2f}%  "
              f"(delta {100 * (s['acc_relar'] - s['acc_base_vlm']):+.2f})")
        print(f"  answer log-p      : base={s['answer_logp_base']:.4f}  relar={s['answer_logp_relar']:.4f}")
        print(f"  avg depth         : {s['avg_depth']:.3f}  dist={s['depth_distribution']}")
        print(f"  relative time     : {s['relative_time']:.2f}x")
        for k, v in s["by_category"].items():
            print(f"    {k:<30s} n={v['n']:<4d} base={v['acc_base'] * 100:6.2f}%  relar={v['acc_relar'] * 100:6.2f}%")
        print("=" * 72)

    # --------------------------------------------------------------- utilities
    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for name in self.MODULES:
            torch.save(getattr(self, name).state_dict(), path / f"{name}.pt")
        if self.cfg.use_lora:
            self.vlm.save_pretrained(str(path / "lora_adapter"))
        self.cfg.save(path / "config.json")

    def _record(self, total, lm, kl, rl, critic, depth, entropy, reward):
        self._hist["step"].append(self._step)
        for k, v in zip(["total", "lm", "kl", "rl", "critic", "depth", "entropy", "reward"],
                        [total, lm, kl, rl, critic, depth, entropy, reward]):
            self._hist[k].append(v)

    def _save_history(self, out_dir: Path):
        with open(out_dir / "loss_history.csv", "w", newline="") as f:
            w = csv.writer(f)
            keys = list(self._hist)
            w.writerow(keys)
            w.writerows(zip(*[self._hist[k] for k in keys]))
        n = len(self._hist["step"])
        if n == 0:
            return
        w = max(1, min(50, n // 5 or 1))
        ma = lambda x: np.concatenate([np.full(w - 1, np.nan), np.convolve(np.array(x, float), np.ones(w) / w, "valid")])
        steps = np.array(self._hist["step"])
        fig, ax = plt.subplots(3, 1, figsize=(12, 12))
        for k in ["total", "lm", "rl", "critic"]:
            ax[0].plot(steps, self._hist[k], alpha=0.3)
            ax[0].plot(steps, ma(self._hist[k]), lw=2, label=k)
        ax[1].plot(steps, ma(self._hist["depth"]), lw=2, label="depth")
        ax[2].plot(steps, ma(self._hist["reward"]), lw=2, label="shaped return")
        for a, t in zip(ax, ["Losses", "Refinement depth", "Shaped reward"]):
            a.set_title(t); a.legend(); a.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curve.png", dpi=150)
        plt.close()
