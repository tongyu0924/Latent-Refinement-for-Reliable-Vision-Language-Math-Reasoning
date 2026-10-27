from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class InitialStateExtractor(nn.Module):
    """s0 = f_extract(h0): attention pooling over prompt tokens + MLP."""

    def __init__(self, hidden_size: int, state_dim: int):
        super().__init__()
        self.attn_score = nn.Linear(hidden_size, 1, bias=False)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.GELU(),
            nn.Linear(hidden_size // 2, state_dim),
            nn.LayerNorm(state_dim),
        )

    def forward(self, h0: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = h0.float()
        scores = self.attn_score(h).squeeze(-1).masked_fill(~mask.bool(), float("-inf"))
        w = torch.softmax(scores, dim=0)
        return self.proj((h * w.unsqueeze(-1)).sum(0))


class DepthController(nn.Module):
    """pi_d(d | s0): categorical policy over depths {0, ..., max_depth}."""

    def __init__(self, state_dim: int, max_depth: int = 3):
        super().__init__()
        self.max_depth = max_depth
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, state_dim), nn.GELU(),
            nn.Linear(state_dim, state_dim // 2), nn.GELU(),
            nn.Linear(state_dim // 2, max_depth + 1),
        )

    def forward(self, s0: torch.Tensor, deterministic: bool = False):
        probs = F.softmax(self.mlp(s0.float()), dim=-1)
        dist = torch.distributions.Categorical(probs)
        depth = probs.argmax(-1) if deterministic else dist.sample()
        return depth, dist.log_prob(depth), probs, dist.entropy()


class ActionController(nn.Module):
    """pi_a(a_t | s_t): Gaussian policy over (gamma, beta, v)."""

    def __init__(self, state_dim: int, action_dim: int = 64, max_scale: float = 1.2):
        super().__init__()
        self.max_scale = max_scale
        self.shared_enc = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2), nn.GELU(),
            nn.Linear(state_dim * 2, state_dim), nn.LayerNorm(state_dim),
        )
        self.gamma_mlp = nn.Linear(state_dim, action_dim * 2)
        self.beta_mlp = nn.Linear(state_dim, action_dim * 2)
        self.dir_mlp = nn.Linear(state_dim, action_dim * 2)

    def _sample(self, out: torch.Tensor, normalize: bool, deterministic: bool):
        mean, log_std = out.chunk(2, dim=-1)
        dist = torch.distributions.Normal(mean, log_std.clamp(-3.0, 1.0).exp())
        raw = mean if deterministic else dist.rsample()
        vec = F.normalize(raw, dim=-1) if normalize else torch.tanh(raw) * self.max_scale
        return vec, dist.log_prob(raw).sum(-1), dist.entropy().sum(-1)

    def forward(self, st: torch.Tensor, deterministic: bool = False):
        feat = self.shared_enc(st.float())
        gamma, g_lp, g_h = self._sample(self.gamma_mlp(feat), False, deterministic)
        beta, b_lp, b_h = self._sample(self.beta_mlp(feat), False, deterministic)
        v, v_lp, v_h = self._sample(self.dir_mlp(feat), True, deterministic)
        return gamma, beta, v, g_lp + b_lp + v_lp, g_h + b_h + v_h


class HiddenRefiner(nn.Module):
    """h_{t+1} = FiLM(LN(h_t); gamma, beta) + eps * v ;  s_{t+1} = g(s_t, pool(h_{t+1}))."""

    def __init__(self, hidden_size: int, state_dim: int, action_dim: int = 64, dir_scale: float = 0.05):
        super().__init__()
        self.hidden_size = hidden_size
        self.dir_scale = dir_scale
        self.layernorm = nn.LayerNorm(hidden_size)
        self.gamma_proj = nn.Linear(action_dim, hidden_size, bias=False)
        self.beta_proj = nn.Linear(action_dim, hidden_size, bias=True)
        self.dir_proj = nn.Linear(action_dim, hidden_size, bias=False)
        self.h_agg_proj = nn.Linear(hidden_size, state_dim, bias=False)
        self.state_update = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim * 2), nn.GELU(),
            nn.Linear(state_dim * 2, state_dim),
            nn.LayerNorm(state_dim),
        )

    def refine(self, ht, st, gamma, beta, vt, mask):
        g = self.gamma_proj(gamma.float())
        b = self.beta_proj(beta.float())
        v = F.normalize(self.dir_proj(vt.float()), dim=-1)
        h = self.layernorm(ht.float()) * (1.0 + g.unsqueeze(0)) + b.unsqueeze(0)
        return h + self.dir_scale * v.unsqueeze(0).expand_as(h)

    def update_state(self, st, h_next, mask):
        h_valid = h_next[mask.bool()].float()
        h_agg = h_valid.mean(0) if len(h_valid) > 0 else torch.zeros(self.hidden_size, device=h_next.device)
        return self.state_update(torch.cat([st.float(), self.h_agg_proj(h_agg)], dim=-1)) + st.float()


class DecodeBridge(nn.Module):
    """f_decode(s, h) = (1 - gate(s)) * h + gate(s) * proj(s). Gate starts near 0 (identity)."""

    def __init__(self, state_dim: int, hidden_size: int, use_layernorm: bool = False, gate_bias_init: float = -4.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gate_net = nn.Sequential(nn.Linear(state_dim, hidden_size), nn.Sigmoid())
        nn.init.constant_(self.gate_net[0].bias, gate_bias_init)
        self.layernorm = nn.LayerNorm(hidden_size) if use_layernorm else nn.Identity()

    def forward(self, st: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gate = self.gate_net(st.float())
        out = (1.0 - gate) * h.float() + gate * self.proj(st.float()).unsqueeze(0)
        return self.layernorm(out)


class ValueCritic(nn.Module):
    """Baseline V(s0) for the policy gradient."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, state_dim), nn.GELU(),
            nn.Linear(state_dim, state_dim // 2), nn.GELU(),
            nn.Linear(state_dim // 2, 1),
        )

    def forward(self, s0: torch.Tensor) -> torch.Tensor:
        return self.net(s0.float()).squeeze(-1)


class RefinedLMHead(nn.Module):
    """Wraps lm_head so that model.generate() decodes from f_decode(s_d, h)."""

    def __init__(self, base_head: nn.Module, bridge: DecodeBridge):
        super().__init__()
        self.base_head = base_head
        self.bridge = bridge
        self.st: Optional[torch.Tensor] = None

    @property
    def weight(self):
        return self.base_head.weight

    def forward(self, h):
        if self.st is None:
            return self.base_head(h)
        return self.base_head(self.bridge(self.st, h).to(h.dtype))
