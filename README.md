# Latent Refinement for Reliable Vision-Language Math Reasoning

Refines VLM hidden states in latent space with an RL-trained controller before decoding, improving vision-language math reasoning without generating chain-of-thought tokens.

---

## Overview

Vision-language models (VLMs) often fail on visual math problems. Their errors come from unstable internal reasoning rather than from a lack of knowledge. Chain-of-thought prompting can help, but it requires generating many extra tokens.

This project instead **reasons in latent space**. After the VLM encodes the image and question, a lightweight policy iteratively refines a compact reasoning state built from the VLM's hidden states. The refined state is then injected into decoding. The policy is trained with reinforcement learning, using a reward defined by the improvement in ground-truth answer likelihood. No intermediate text is generated.

```
Image + Question ──► VLM (frozen) ──► h0  (last-layer hidden states)
                                         │
                                  f_extract (prompt tokens only)
                                         │
                                         s0
                              ┌──────────┴──────────┐
                     Depth policy π_d(d|s0)   Action policy π_a(a_t|s_t)
                              └──────────┬──────────┘
                     h0 → h1 → … → h_d ,  s0 → s1 → … → s_d
                                         │
                          f_decode(s_d, h) ──► lm_head ──► Answer
```

## Algorithm

**1. Initial state.** The VLM is run once on the image and question. The last-layer hidden states $h_0 \in \mathbb{R}^{L\times D}$ are attention-pooled over the prompt positions and projected to a reasoning state:

$$s_0 = f_{\text{extract}}(h_0)$$

**2. Depth policy.** A categorical policy chooses how many refinement steps to take:

$$d \sim \pi_d(d \mid s_0), \quad d \in \{0,\dots,d_{\max}\}$$

**3. Latent refinement.** At each step, a Gaussian action policy outputs a scale $\gamma_t$, a shift $\beta_t$ and a unit direction $v_t$:

$$a_t = (\gamma_t, \beta_t, v_t) \sim \pi_a(a_t \mid s_t)$$

The hidden states are updated with a FiLM-style transform, and the state is updated from the pooled result:

$$h_{t+1} = \mathrm{LN}(h_t)\odot(1+W_\gamma\gamma_t) + W_\beta\beta_t + \epsilon\,\hat v_t$$

$$s_{t+1} = s_t + g\big(s_t,\ \mathrm{pool}(h_{t+1})\big)$$

**4. Decoding.** A gated bridge mixes the final state into every hidden state before the frozen `lm_head`. The gate is initialised near 0, so the model starts as the original VLM:

$$\tilde h = (1-\sigma(s_d))\odot h + \sigma(s_d)\odot \mathrm{proj}(s_d)$$

At inference, $s_d$ is computed once from the prompt, and every generated token is decoded from $\tilde h$.

**5. Reward.** Each step is rewarded by the gain in log-likelihood of the gold answer $y^{\star}$, minus a per-step cost $c_d$:

$$r_t = \Delta_t - c_d, \quad \Delta_t = \log p(y^{\star}\mid x, s_{t+1}) - \log p(y^{\star}\mid x, s_t)$$

The shaped return adds a KL penalty to the original VLM and an entropy bonus:

$$R = \sum_{t=1}^{d}\Big[\tfrac{r_t}{t} + \tfrac{1}{d}\big(-\beta\,\mathrm{KL} + \lambda_H \mathcal{H}\big)\Big]$$

**6. Objective.** The training loss combines a language-modelling term, a REINFORCE term with a learned baseline $V(s_0)$, and a critic loss:

$$\mathcal{L} = \underbrace{-\log p(y^{\star}\mid x, s_d) + \beta\,\mathrm{KL}}_{\mathcal{L}_{\text{LM}}} \;+\; \alpha_{\text{RL}}\underbrace{\big[-(R - V(s_0))\log\pi(d, a_{1:d}) - \lambda_H\mathcal{H}\big]}_{\mathcal{L}_{\text{RL}}} \;+\; \alpha_V\,(V(s_0)-R)^2$$

---

## Results

**Setup:**
- Backbone: Qwen2-VL-2B-Instruct (frozen, bf16).
- Data: MathVista *testmini*, split 80/20 with seed 42 (800 training / 200 validation problems).
- Training: 3 epochs, $d_{\max}=3$.
- Decoding: greedy for both models.

| Method | Overall Acc. | Answer log-p |
|---|---|---|
| Qwen2-VL-2B (base) | 48.00 | −1.177 |
| **+ Latent Refinement (ours)** | **55.00** (+7.00) | **−0.346** |

**By category** (validation set, n = 200):

| Category | n | Base | Ours |
|---|---|---|---|
| Figure question answering | 51 | 62.75 | **66.67** |
| Geometry problem solving | 42 | 33.33 | **47.62** |
| Math word problem | 40 | 45.00 | **57.50** |
| Textbook question answering | 29 | 48.28 | **55.17** |
| Visual question answering | 38 | **47.37** | 44.74 |
| *Multiple choice* | 104 | 57.69 | **64.42** |
| *Free form* | 96 | 37.50 | **44.79** |

The relative inference time is 1.88× the base model. The learned depth policy chose $d=3$ for every problem.

**SFT-only ablation.** The SFT-only baseline trains the decode bridge with depth fixed to 0 and no RL. Its results are not reported yet. Run it with:
```bash
python train.py --ablation sft_only --output_dir ./outputs/sft_only
```

---

## Project Structure

```
├── config.py      # Config dataclass + CLI overrides
├── models.py      # StateExtractor, DepthController, ActionController, HiddenRefiner, DecodeBridge, ValueCritic
├── dataset.py     # MathVista loading, prompting, answer extraction / scoring
├── trainer.py     # ReLAR: refinement, RL training loop, generation, evaluation
├── train.py       # training entry point
└── inference.py   # evaluate a checkpoint or run a single image
```

## Usage

```bash
pip install -r requirements.txt
```

**Train** (also evaluates on the validation split when training finishes):
```bash
python train.py                                   # defaults: Qwen2-VL-2B, 3 epochs
python train.py --epochs 5 --depth_cost 0.05      # any Config field can be overridden
python train.py --use_lora true                   # LoRA on the LLM (needs more VRAM)
```

**Evaluate a checkpoint** (re-uses the checkpoint's config, so the validation split matches training):
```bash
python inference.py --ckpt ./relar_vlm_mathvista/final_model
```

**Single image:**
```bash
python inference.py --ckpt ./relar_vlm_mathvista/final_model \
    --image figure.png --question "What is the area of the triangle?" --answer_type float
```

**Colab:**
```python
!git clone https://github.com/<user>/Latent-Refinement-for-Reliable-Vision-Language-Math-Reasoning.git
%cd Latent-Refinement-for-Reliable-Vision-Language-Math-Reasoning
!pip install -q -r requirements.txt
!python train.py
```

## Notes

- Only the refinement modules are trained by default (~6M parameters). The VLM stays frozen and also serves as the KL reference model.
- The state is built from the prompt positions only, so there is no answer leakage under teacher forcing.
- Tiny images (side < 28 px) are upscaled automatically to satisfy the Qwen2-VL processor.
- MathVista only releases answers for *testmini*, so both training and validation come from it. Reported numbers are on the 200 held-out problems and are not directly comparable to official MathVista leaderboard scores.
