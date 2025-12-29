from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask


@dataclass(frozen=True)
class SpectralBiasConfig:
    enabled: bool = True
    head_dim: int = 64
    num_heads: int = 16
    L_train: int = 4096
    L_max: int = 1_000_000
    K: int = 6
    M: int = 2
    beta: float = 0.5
    ramp_lambda: float = 0.2
    tau: float = 64.0
    gate_kind: str = "softplus"  # "none"|"softplus"|"relu"
    share_across_heads: bool = True
    use_slope: bool = True
    c_scale: float = 0.01
    width_min: float = 32.0
    width_max: float = 256.0
    delta_star_max: int | None = None
    detach_q: bool = False
    subtract_b0: bool = True
    eps: float = 1e-9
    # regularizers (added to training loss)
    lambda_omega: float = 1e-5
    lambda_zero_mean: float = 1e-4
    lambda_entropy: float = 1e-4
    # pointer-driven blockmask (KV selection)
    use_pointer_mask: bool = False
    pointer_local_blocks: int = 16
    pointer_half_blocks: int = 4
    pointer_global_blocks: int = 0
    # Total pointer KV budget (local + centers + extra), per (head, q_block).
    # This is used only when `use_pointer_mask=True` and `pointer_budget_blocks > 0`.
    pointer_budget_blocks: int = 0
    pointer_budget_is_total: bool = True
    pointer_budget_temp: float = 1.0
    # "fixed" -> each peak gets the same scheduled radius (`half_blocks_active`)
    # "pi" -> per-peak radii derived from π + `pointer_budget_blocks`
    pointer_radius_mode: str = "pi"
    pointer_qblock_rep: str = "last"  # "last"|"mean"
    pointer_doc_clamp: bool = True
    gumbel_topk: bool = False
    gumbel_topk_k: int = 0  # 0 -> use M (no masking)
    gumbel_topk_tau: float = 1.0
    gumbel_topk_seed: int = 0
    pointer_reset_len: int = 0
    # debug/telemetry (rank0 recommended)
    debug_stats: bool = False
    # π entropy regularization: only penalize collapse below this floor
    # (fraction of log(M)), instead of pushing toward uniform.
    pi_entropy_floor_frac: float = 0.25
    # Δ* anti-saturation: penalize predictions near 0 / cap (optional).
    lambda_delta_edge: float = 0.0
    delta_edge_eps: float = 0.0


class SpectralBias(nn.Module):
    """
    Query-conditioned Fourier bias over relative distances Δ = q_idx - kv_idx.

    Designed to be used via FlexAttention score_mod without materializing [B,H,L,L].
    """

    def __init__(
        self,
        *,
        head_dim: int,
        num_heads: int,
        L_train: int,
        L_max: int = 1_000_000,
        K: int = 6,
        M: int = 2,
        beta: float = 0.5,
        ramp_lambda: float = 0.2,
        tau: float = 64.0,
        w_min: float | None = None,
        w_max: float | None = None,
        share_across_heads: bool = True,
        use_slope: bool = True,
        gate_kind: str = "softplus",
        c_scale: float = 0.01,
        width_min: float = 32.0,
        width_max: float = 256.0,
        delta_star_max: int | None = None,
        detach_q: bool = False,
        subtract_b0: bool = True,
        eps: float = 1e-9,
        use_pointer_mask: bool = False,
        pointer_local_blocks: int = 16,
        pointer_half_blocks: int = 4,
        pointer_global_blocks: int = 0,
        pointer_budget_blocks: int = 0,
        pointer_budget_is_total: bool = True,
        pointer_budget_temp: float = 1.0,
        pointer_radius_mode: str = "pi",
        pointer_qblock_rep: str = "last",
        pointer_doc_clamp: bool = True,
        gumbel_topk: bool = False,
        gumbel_topk_k: int = 0,
        gumbel_topk_tau: float = 1.0,
        gumbel_topk_seed: int = 0,
        pointer_reset_len: int = 0,
        lambda_omega: float = 1e-5,
        lambda_zero_mean: float = 1e-4,
        lambda_entropy: float = 1e-4,
        pi_entropy_floor_frac: float = 0.25,
        lambda_delta_edge: float = 0.0,
        delta_edge_eps: float = 0.0,
        debug_stats: bool = False,
        enabled: bool = True,
    ):
        super().__init__()
        if K <= 0:
            raise ValueError("K must be positive")
        if M <= 0:
            raise ValueError("M must be positive")
        if gate_kind not in {"none", "softplus", "relu"}:
            raise ValueError("gate_kind must be one of: none|softplus|relu")
        if not share_across_heads and num_heads <= 0:
            raise ValueError("num_heads must be set when share_across_heads=False")

        self.enabled = enabled
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.K = K
        self.M = M
        self.beta = float(beta)
        self.ramp_lambda = float(ramp_lambda)
        self.tau = float(tau)
        self.gate_kind = gate_kind
        self.share_across_heads = share_across_heads
        self.use_slope = use_slope
        self.c_scale = float(c_scale)
        self.width_min = float(width_min)
        self.width_max = float(width_max)
        self.delta_star_max = int(delta_star_max) if delta_star_max is not None else int(L_train - 1)
        self.detach_q = detach_q
        self.subtract_b0 = subtract_b0
        self.eps = float(eps)
        self.use_pointer_mask = bool(use_pointer_mask)
        self.pointer_local_blocks = int(pointer_local_blocks)
        self.pointer_half_blocks = int(pointer_half_blocks)
        self.pointer_global_blocks = int(pointer_global_blocks)
        self.pointer_budget_blocks = int(pointer_budget_blocks)
        self.pointer_budget_is_total = bool(pointer_budget_is_total)
        self.pointer_budget_temp = float(pointer_budget_temp)
        self.pointer_radius_mode = str(pointer_radius_mode).strip().lower()
        if self.pointer_radius_mode not in {"fixed", "pi"}:
            raise ValueError("pointer_radius_mode must be one of: fixed|pi")
        self.pointer_qblock_rep = str(pointer_qblock_rep)
        if self.pointer_qblock_rep not in {"last", "mean"}:
            raise ValueError("pointer_qblock_rep must be one of: last|mean")
        self.pointer_doc_clamp = bool(pointer_doc_clamp)
        self.gumbel_topk = bool(gumbel_topk)
        self.gumbel_topk_k = int(gumbel_topk_k)
        self.gumbel_topk_tau = float(gumbel_topk_tau)
        self.gumbel_topk_seed = int(gumbel_topk_seed)
        self.pointer_reset_len = int(pointer_reset_len)
        self.debug_stats = bool(debug_stats)
        self.register_buffer(
            "_pointer_mask_active",
            torch.tensor(1 if self.use_pointer_mask else 0, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "_pointer_half_blocks_active",
            torch.tensor(self.pointer_half_blocks if self.use_pointer_mask else 0, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "_pointer_global_blocks_active",
            torch.tensor(self.pointer_global_blocks if self.use_pointer_mask else 0, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "_pointer_reset_active",
            torch.tensor(0, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "_pointer_reset_len_active",
            torch.tensor(max(0, self.pointer_reset_len), dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "_gumbel_seed",
            torch.tensor(self.gumbel_topk_seed, dtype=torch.int64),
            persistent=False,
        )
        # Active clamp on Δ* in tokens (scheduled by the training loop).
        self.register_buffer(
            "_delta_star_max_active",
            torch.tensor(self.delta_star_max, dtype=torch.int32),
            persistent=False,
        )

        self.lambda_omega = float(lambda_omega)
        self.lambda_zero_mean = float(lambda_zero_mean)
        self.lambda_entropy = float(lambda_entropy)
        self.pi_entropy_floor_frac = float(pi_entropy_floor_frac)
        self.lambda_delta_edge = float(lambda_delta_edge)
        self.delta_edge_eps = float(delta_edge_eps)
        self.register_buffer(
            "_delta_edge_eps_active",
            torch.tensor(self.delta_edge_eps, dtype=torch.float32),
            persistent=False,
        )

        w_min = float(w_min) if w_min is not None else (2 * math.pi / L_max)
        w_max = float(w_max) if w_max is not None else (2 * math.pi / L_train)
        if not (w_min > 0 and w_max > 0 and w_min < w_max):
            raise ValueError(f"invalid (w_min, w_max)=({w_min}, {w_max})")

        omegas = torch.logspace(math.log10(w_min), math.log10(w_max), K, dtype=torch.float32)
        self.register_buffer("omegas", omegas, persistent=False)
        self.register_buffer("log_omegas", omegas.log(), persistent=False)
        self.register_buffer("omega_sq", omegas.square(), persistent=False)
        self.register_buffer("logw_min", torch.tensor(math.log(w_min), dtype=torch.float32), persistent=False)
        self.register_buffer("logw_max", torch.tensor(math.log(w_max), dtype=torch.float32), persistent=False)

        # Trig caches for distances 0..L-1 (lazily initialized on-device).
        # Keep these in bf16 to reduce memory/register pressure inside FlexAttention score_mod.
        self.register_buffer("_cos_wD", torch.empty(0, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("_sin_wD", torch.empty(0, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("_cos_mean", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("_sin_mean", torch.empty(0, dtype=torch.float32), persistent=False)

        # ---- SCOPE debug stats (rank0 recommended) ----
        token_edges = torch.tensor(
            [0, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535, 131071, 262143, 524287, 1048575],
            dtype=torch.int64,
        )
        block_edges = torch.tensor([0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191], dtype=torch.int64)
        self.register_buffer("_dbg_token_edges", token_edges, persistent=False)
        self.register_buffer("_dbg_block_edges", block_edges, persistent=False)
        self.register_buffer("_dbg_delta_tokens_hist", torch.zeros(token_edges.numel() + 1, dtype=torch.int32), persistent=False)
        self.register_buffer("_dbg_delta_blocks_hist", torch.zeros(block_edges.numel() + 1, dtype=torch.int32), persistent=False)
        self.register_buffer("_dbg_ptr_center_hist", torch.zeros(block_edges.numel() + 1, dtype=torch.int32), persistent=False)
        max_kv = int(self.pointer_local_blocks + (2 * self.pointer_half_blocks + 1) * self.M + self.pointer_global_blocks)
        self.register_buffer("_dbg_kv_unique_hist", torch.zeros(max_kv + 1, dtype=torch.int32), persistent=False)
        self.register_buffer("_dbg_ptr_outside_local_frac", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_dbg_ptr_block0_frac", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_dbg_ptr_block0_frac_excl", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_dbg_ptr_docstart_frac", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_dbg_ptr_radius_hist", torch.zeros(self.pointer_half_blocks + 1, dtype=torch.int32), persistent=False)
        max_extra = max(1, int(self.pointer_half_blocks * self.M))
        self.register_buffer("_dbg_ptr_extra_sum_hist", torch.zeros(max_extra + 1, dtype=torch.int32), persistent=False)
        self.register_buffer("_dbg_ptr_radius0_frac", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_dbg_pi_entropy_mean", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_dbg_reg_omega", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_dbg_reg_entropy", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_dbg_reg_delta_edge", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_dbg_reg_zero_mean", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_dbg_reg_total", torch.zeros((), dtype=torch.float32), persistent=False)

        hidden = max(128, 2 * head_dim)
        out = 5 * M + (1 if use_slope else 0)  # Δ*, μ, σ, π_logit, budget_logit, (optional) slope

        if share_across_heads:
            self.mlp = nn.Sequential(nn.Linear(head_dim, hidden), nn.SiLU(), nn.Linear(hidden, out))
        else:
            self.w1 = nn.Parameter(torch.empty(num_heads, hidden, head_dim))
            self.b1 = nn.Parameter(torch.zeros(num_heads, hidden))
            self.w2 = nn.Parameter(torch.empty(num_heads, out, hidden))
            self.b2 = nn.Parameter(torch.zeros(num_heads, out))
            self._init_per_head_weights()
        self._init_outputs()

    def _init_per_head_weights(self):
        std1 = 0.5 * (self.head_dim ** -0.5)
        bound1 = (3**0.5) * std1
        std2 = 0.5 * ((self.w1.size(1)) ** -0.5)
        bound2 = (3**0.5) * std2
        with torch.no_grad():
            self.w1.uniform_(-bound1, bound1)
            self.w2.uniform_(-bound2, bound2)

    def _init_outputs(self):
        # Initialize the output head to start near "no-op" + local Δ* (stability).
        # Break symmetry across pointers so π can get gradients (otherwise identical
        # components make π irrelevant and it stays exactly uniform).
        delta_biases = torch.linspace(-6.0, -4.0, steps=self.M)  # small Δ* initially (all within local band)
        with torch.no_grad():
            if self.share_across_heads:
                out: nn.Linear = self.mlp[-1]
                out.weight.zero_()
                out.bias.zero_()
                out.bias[: self.M].copy_(delta_biases.to(device=out.bias.device, dtype=out.bias.dtype))  # delta_raw
            else:
                self.w2.zero_()
                self.b2.zero_()
                biases = delta_biases.to(device=self.b2.device, dtype=self.b2.dtype).view(1, self.M).expand(self.num_heads, -1)
                self.b2[:, : self.M].copy_(biases)

    def _mlp_forward(self, q: Tensor) -> Tensor:
        if self.detach_q:
            q = q.detach()
        if self.share_across_heads:
            # Keep MLP compute in the same dtype as q (bf16 in this project) to
            # avoid matmul dtype mismatches without forcing q -> fp32 (large).
            q_dtype = q.dtype
            fc1: nn.Linear = self.mlp[0]
            fc2: nn.Linear = self.mlp[2]
            x = F.linear(q, fc1.weight.to(dtype=q_dtype), None if fc1.bias is None else fc1.bias.to(dtype=q_dtype))
            x = F.silu(x)
            y = F.linear(x, fc2.weight.to(dtype=q_dtype), None if fc2.bias is None else fc2.bias.to(dtype=q_dtype))
            return y
        # q: [B,H,L,D]
        q_dtype = q.dtype
        x = torch.einsum("bhld,hkd->bhlk", q, self.w1.to(dtype=q_dtype)) + self.b1.to(dtype=q_dtype)[None, :, None, :]
        x = F.silu(x)
        y = torch.einsum("bhlk,hok->bhlo", x, self.w2.to(dtype=q_dtype)) + self.b2.to(dtype=q_dtype)[None, :, None, :]
        return y

    def _get_trig_tables(self, L: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self._cos_wD.numel() != 0 and self._cos_wD.shape[-1] == L and self._cos_wD.device == device:
            return self._cos_wD, self._sin_wD, self._cos_mean, self._sin_mean

        D = torch.arange(L, device=device, dtype=torch.float32)
        w = self.omegas.to(device=device)
        cos_wD_f32 = torch.cos(w[:, None] * D[None, :])
        sin_wD_f32 = torch.sin(w[:, None] * D[None, :])
        self._cos_mean = cos_wD_f32.mean(dim=-1)
        self._sin_mean = sin_wD_f32.mean(dim=-1)
        cos_wD = cos_wD_f32.to(dtype=torch.bfloat16)
        sin_wD = sin_wD_f32.to(dtype=torch.bfloat16)
        self._cos_wD = cos_wD
        self._sin_wD = sin_wD
        return cos_wD, sin_wD, self._cos_mean, self._sin_mean

    def set_pointer_mask_state(self, *, enabled: bool, half_blocks: int, global_blocks: int | None = None):
        half_blocks = int(half_blocks)
        if half_blocks < 0:
            raise ValueError("half_blocks must be >= 0")
        half_blocks = min(half_blocks, self.pointer_half_blocks)
        self._pointer_mask_active.fill_(1 if enabled else 0)
        self._pointer_half_blocks_active.fill_(half_blocks)
        if global_blocks is not None:
            gb = int(global_blocks)
            if gb < 0:
                raise ValueError("global_blocks must be >= 0")
            gb = min(gb, self.pointer_global_blocks)
            self._pointer_global_blocks_active.fill_(gb)

    def set_pointer_reset_state(self, *, enabled: bool, reset_len: int):
        reset_len = int(reset_len)
        if reset_len < 0:
            raise ValueError("reset_len must be >= 0")
        self._pointer_reset_active.fill_(1 if enabled else 0)
        self._pointer_reset_len_active.fill_(reset_len if enabled else 0)

    def set_gumbel_seed(self, *, seed: int):
        self._gumbel_seed.fill_(int(seed))

    def set_delta_star_max_active(self, *, max_tokens: int):
        max_tokens = int(max_tokens)
        if max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        max_tokens = min(max_tokens, int(self.delta_star_max))
        self._delta_star_max_active.fill_(max_tokens)

    def set_delta_edge_eps_active(self, *, eps: float):
        eps = float(eps)
        if eps < 0:
            raise ValueError("eps must be >= 0")
        self._delta_edge_eps_active.fill_(eps)

    def forward(
        self,
        q: Tensor,
        *,
        docs: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        """
        q: [B, H, L, d_head] (projected per-head query vectors, before attention)

        Returns:
          coeff_cos: [B,H,L,K]
          coeff_sin: [B,H,L,K]
          slope: [B,H,L] (zeros if disabled)
          b0: [B,H,L] value of b(Δ=0) (zeros if subtract_b0=False)
          delta_star: [B,H,L,M] per-peak Δ*
          pi: [B,H,L,M] mixture weights (softmax)
          budget_logits: [B,H,L,M] logits for budget allocation
          delta_main: [B,H,L] (main pointer Δ*)
          width: [B,H,L] (window width for optional trough)
          cos_wD: [K,L]
          sin_wD: [K,L]
          reg_loss: [] scalar (0 in eval or if all lambdas are 0)
        """
        B, H, L, Dh = q.shape
        if H != self.num_heads:
            raise ValueError(f"expected H={self.num_heads}, got {H}")
        if Dh != self.head_dim:
            raise ValueError(f"expected head_dim={self.head_dim}, got {Dh}")

        device = q.device
        cos_wD, sin_wD, cos_mean, sin_mean = self._get_trig_tables(L, device)

        if not self.enabled:
            z_bhlk = q.new_zeros((B, H, L, self.K), dtype=torch.float32)
            z_bhl = q.new_zeros((B, H, L), dtype=torch.float32)
            z_bhlm = q.new_zeros((B, H, L, self.M), dtype=torch.float32)
            z_pi = q.new_full((B, H, L, self.M), 1.0 / self.M, dtype=torch.float32)
            z = q.new_zeros((), dtype=torch.float32)
            return z_bhlk, z_bhlk, z_bhl, z_bhl, z_bhlm, z_pi, z_bhlm, z_bhl, z_bhl, cos_wD, sin_wD, z

        y = self._mlp_forward(q).float()
        M = self.M
        K = self.K
        offs = 0
        delta_raw = y[..., offs : offs + M]
        offs += M
        mu_raw = y[..., offs : offs + M]
        offs += M
        sigma_raw = y[..., offs : offs + M]
        offs += M
        pi_logits = y[..., offs : offs + M]
        offs += M
        budget_logits = y[..., offs : offs + M]
        offs += M
        if self.use_slope:
            c_raw = y[..., offs : offs + 1]
            offs += 1

        # Δ* is a distance in tokens.
        delta_cap = self._delta_star_max_active.to(device=device, dtype=torch.float32)
        delta_cap = torch.clamp(delta_cap, min=0.0, max=float(L - 1))
        delta_star = torch.sigmoid(delta_raw) * delta_cap  # [B,H,L,M]

        # μ is mean of log(ω); bound to [log(w_min), log(w_max)] for stability.
        logw_min = self.logw_min.to(device=device)
        logw_max = self.logw_max.to(device=device)
        mu = logw_min + (logw_max - logw_min) * torch.sigmoid(mu_raw)  # [B,H,L,M]

        sigma = 0.25 + 1.75 * torch.sigmoid(sigma_raw)  # [B,H,L,M], in [0.25, 2.0]
        pi = torch.softmax(pi_logits, dim=-1)  # [B,H,L,M]

        if self.use_slope:
            slope = self.c_scale * torch.tanh(c_raw).squeeze(-1)  # [B,H,L]
        else:
            slope = q.new_zeros((B, H, L), dtype=torch.float32)

        logw = self.log_omegas.to(device=device).view(1, 1, 1, 1, K)  # [1,1,1,1,K]
        mu_e = mu.unsqueeze(-1)  # [B,H,L,M,1]
        sigma_e = sigma.unsqueeze(-1)  # [B,H,L,M,1]
        a_mk = torch.exp(-((logw - mu_e) ** 2) / (2 * (sigma_e**2)))  # [B,H,L,M,K]
        a_mk = a_mk / (a_mk.sum(dim=-1, keepdim=True) + self.eps)

        w = self.omegas.to(device=device).view(1, 1, 1, 1, K)
        phase = w * delta_star.unsqueeze(-1)  # [B,H,L,M,K]
        cos_wDelta = torch.cos(phase)
        sin_wDelta = torch.sin(phase)

        pi_e = pi.unsqueeze(-1)  # [B,H,L,M,1]
        coeff_cos = (pi_e * a_mk * cos_wDelta).sum(dim=3)  # [B,H,L,K]
        coeff_sin = (pi_e * a_mk * sin_wDelta).sum(dim=3)  # [B,H,L,K]

        if self.subtract_b0:
            b0 = coeff_cos.sum(dim=-1)  # b(0) = Σ_k coeff_cos[k]
        else:
            b0 = q.new_zeros((B, H, L), dtype=torch.float32)

        # Main pointer + a simple derived width (placeholder; can be learned later).
        main_idx = pi.argmax(dim=-1)  # [B,H,L]
        delta_main = torch.gather(delta_star, dim=-1, index=main_idx.unsqueeze(-1)).squeeze(-1)  # [B,H,L]
        width = self.width_min + (self.width_max - self.width_min) * torch.sigmoid(mu.mean(dim=-1))  # [B,H,L]

        # Regularizers (only during training).
        reg_omega = q.new_zeros((), dtype=torch.float32)
        reg_entropy = q.new_zeros((), dtype=torch.float32)
        reg_delta_edge = q.new_zeros((), dtype=torch.float32)
        reg_zero_mean = q.new_zeros((), dtype=torch.float32)
        if self.training:
            if self.lambda_omega > 0:
                omega_sq = self.omega_sq.to(device=device).view(1, 1, 1, 1, K)
                reg_omega = self.lambda_omega * (omega_sq * a_mk.square()).sum(dim=-1).mean()
            if self.lambda_entropy > 0:
                entropy = -(pi * (pi + self.eps).log()).sum(dim=-1)  # [B,H,L]
                h_floor = float(self.pi_entropy_floor_frac) * math.log(M)
                if h_floor > 0:
                    reg_entropy = self.lambda_entropy * F.relu(entropy.new_tensor(h_floor) - entropy).mean()
            if self.lambda_zero_mean > 0:
                # mean_Δ b(Δ) using trig means (ignoring optional gate for now)
                cos_mean = cos_mean.to(device=device)  # [K]
                sin_mean = sin_mean.to(device=device)  # [K]
                mean_delta = 0.5 * float(L - 1)
                mean_b = (coeff_cos * cos_mean).sum(dim=-1) + (coeff_sin * sin_mean).sum(dim=-1) + slope * mean_delta - b0
                reg_zero_mean = self.lambda_zero_mean * mean_b.mean().square()
            if self.lambda_delta_edge > 0:
                edge_eps = self._delta_edge_eps_active.to(device=device, dtype=torch.float32)
                denom = torch.clamp(delta_cap, min=1.0)
                u = (delta_star / denom).clamp(min=0.0, max=1.0)
                edge = F.relu(edge_eps - u) + F.relu(u - (1.0 - edge_eps))
                reg_delta_edge = self.lambda_delta_edge * edge.mean()

        reg_loss = reg_omega + reg_entropy + reg_delta_edge + reg_zero_mean

        if self.debug_stats:
            # Representative pointer stats per query-block.
            block_size = 128
            if B == 1 and L % block_size == 0:
                num_blocks = L // block_size
                q_blocks = torch.arange(num_blocks, device=device, dtype=torch.int64)
                t_rep = (q_blocks + 1) * block_size - 1
                if self.pointer_qblock_rep == "last":
                    delta_rep = delta_star[0, :, t_rep, :]  # [H,QB,M]
                    pi_rep = pi[0, :, t_rep, :]  # [H,QB,M]
                else:
                    delta_rep = delta_star[0].reshape(H, num_blocks, block_size, M).mean(dim=2)  # [H,QB,M]
                    pi_rep = pi[0].reshape(H, num_blocks, block_size, M).mean(dim=2)  # [H,QB,M]

                delta_tok_i = delta_rep.to(torch.int64)
                dt_idx = torch.bucketize(delta_tok_i.flatten(), self._dbg_token_edges, right=False)
                dt_hist = torch.bincount(dt_idx, minlength=self._dbg_delta_tokens_hist.numel()).to(torch.int32)
                self._dbg_delta_tokens_hist.copy_(dt_hist)

                delta_blk = torch.floor_divide(delta_tok_i, block_size)
                db_idx = torch.bucketize(delta_blk.flatten(), self._dbg_block_edges, right=False)
                db_hist = torch.bincount(db_idx, minlength=self._dbg_delta_blocks_hist.numel()).to(torch.int32)
                self._dbg_delta_blocks_hist.copy_(db_hist)

                key_tok = t_rep[None, :, None].to(delta_rep.dtype) - delta_rep
                key_tok = key_tok.clamp(min=0.0)
                ptr_center_blk = torch.floor_divide(key_tok.to(torch.int64), block_size)
                ptr_center_blk = torch.minimum(ptr_center_blk, q_blocks[None, :, None])
                reset_active = self._pointer_reset_active.to(device=device, dtype=torch.int32)
                reset_len = int(self._pointer_reset_len_active.item())

                # Doc-aware clamp: avoid wasting pointer budget pointing before the current doc
                # (those entries are masked out anyway). This reduces apparent block-0 collapse.
                if docs is not None:
                    # docs can change within blocks, so estimate doc starts at token granularity
                    doc_change_tok = torch.ones((L,), device=device, dtype=torch.bool)
                    doc_change_tok[1:] = docs[1:] != docs[:-1]
                    tok_idx = torch.arange(L, device=device, dtype=torch.int64)
                    doc_start_token = torch.where(doc_change_tok, tok_idx, torch.zeros_like(tok_idx)).cummax(dim=0).values

                    doc_start_blk_ptr = torch.floor_divide(doc_start_token[t_rep.to(torch.long)], block_size).to(
                        torch.int64
                    )  # [QB]
                    block_start_tok = (q_blocks * int(block_size)).to(torch.long)  # [QB]
                    doc_start_blk_local = torch.floor_divide(doc_start_token[block_start_tok], block_size).to(
                        torch.int64
                    )  # [QB]

                    if not self.pointer_doc_clamp:
                        doc_start_blk_ptr = torch.zeros_like(doc_start_blk_ptr)
                        doc_start_blk_local = torch.zeros_like(doc_start_blk_local)

                    if reset_active.item() != 0 and reset_len > 0:
                        seg_tok_ptr = (t_rep // reset_len) * reset_len
                        seg_blk_ptr = torch.floor_divide(seg_tok_ptr, block_size).to(torch.int64)
                        seg_tok_local = (block_start_tok // reset_len) * reset_len
                        seg_blk_local = torch.floor_divide(seg_tok_local, block_size).to(torch.int64)
                        doc_start_blk_ptr = torch.maximum(doc_start_blk_ptr, seg_blk_ptr)
                        doc_start_blk_local = torch.maximum(doc_start_blk_local, seg_blk_local)

                    ptr_center_blk = torch.maximum(ptr_center_blk, doc_start_blk_ptr[None, :, None])
                    if self.pointer_doc_clamp or reset_active.item() != 0:
                        self._dbg_ptr_docstart_frac.copy_(
                            (ptr_center_blk == doc_start_blk_ptr[None, :, None]).float().mean()
                        )
                    else:
                        self._dbg_ptr_docstart_frac.zero_()
                else:
                    self._dbg_ptr_docstart_frac.zero_()

                pc_idx = torch.bucketize(ptr_center_blk.flatten(), self._dbg_block_edges, right=False)
                pc_hist = torch.bincount(pc_idx, minlength=self._dbg_ptr_center_hist.numel()).to(torch.int32)
                self._dbg_ptr_center_hist.copy_(pc_hist)

                local_blocks = min(self.pointer_local_blocks, num_blocks)
                local_start = (q_blocks - (local_blocks - 1)).clamp(min=0)
                if docs is not None and (self.pointer_doc_clamp or reset_active.item() != 0):
                    local_start = torch.maximum(local_start, doc_start_blk_local)
                outside = ptr_center_blk < local_start[None, :, None]
                self._dbg_ptr_outside_local_frac.copy_(outside.float().mean())
                self._dbg_ptr_block0_frac.copy_((ptr_center_blk == 0).float().mean())
                # Exclude the first few query blocks where block-0 is often unavoidable.
                excl = 4
                if num_blocks > excl:
                    sub = ptr_center_blk[:, excl:, :]
                    self._dbg_ptr_block0_frac_excl.copy_((sub == 0).float().mean())
                else:
                    self._dbg_ptr_block0_frac_excl.zero_()

                entropy = -(pi_rep * (pi_rep + self.eps).log()).sum(dim=-1)  # [H,QB]
                self._dbg_pi_entropy_mean.copy_(entropy.mean())

            self._dbg_reg_omega.copy_(reg_omega.detach())
            self._dbg_reg_entropy.copy_(reg_entropy.detach())
            self._dbg_reg_delta_edge.copy_(reg_delta_edge.detach())
            self._dbg_reg_zero_mean.copy_(reg_zero_mean.detach())
            self._dbg_reg_total.copy_(reg_loss.detach())

        return (
            coeff_cos,
            coeff_sin,
            slope,
            b0,
            delta_star,
            pi,
            budget_logits,
            delta_main,
            width,
            cos_wD,
            sin_wD,
            reg_loss,
        )

    @torch.no_grad()
    def build_pointer_blockmask(
        self,
        *,
        input_seq: Tensor | None = None,
        docs: Tensor | None = None,
        delta_star: Tensor,
        pi: Tensor,
        budget_logits: Tensor,
        block_size: int = 128,
        local_blocks: int | None = None,
        ptr_half_blocks: int | None = None,
        pointer_budget_blocks: int | None = None,
        global_blocks: int | None = None,
        eod_token_id: int = 50256,
    ) -> BlockMask:
        """
        input_seq: [T] int tokens (optional if `docs` provided)
        docs: [T] document ids (optional if `input_seq` provided)
        delta_star: [B,H,T,M] float token offsets (Δ*), clipped already   
        budget_logits: [B,H,T,M] logits for budget allocation

        Returns a doc-aware causal BlockMask that is the union of:        
          - local sliding window in blocks
          - pointer windows around each Δ* (per head, per query block)
          - optional global anchors (first N blocks)
        """
        assert (input_seq is None) != (docs is None), "pass exactly one of: input_seq|docs"
        if docs is None:
            assert input_seq is not None and input_seq.ndim == 1
            docs = (input_seq == eod_token_id).cumsum(0)
        assert docs is not None and docs.ndim == 1
        device = docs.device
        T = int(docs.numel())
        assert T % block_size == 0
        B, H, TT, M = delta_star.shape
        assert B == 1
        assert TT == T
        assert H == self.num_heads
        assert M == self.M
        assert pi.shape == (B, H, T, M)
        assert budget_logits.shape == (B, H, T, M)

        local_blocks = self.pointer_local_blocks if local_blocks is None else int(local_blocks)
        ptr_half_blocks = self.pointer_half_blocks if ptr_half_blocks is None else int(ptr_half_blocks)
        local_blocks = max(1, local_blocks)
        ptr_half_blocks = max(0, ptr_half_blocks)

        num_blocks = T // block_size
        local_blocks = min(local_blocks, num_blocks)

        def document_causal(b, h, q_idx, kv_idx):
            q = q_idx.to(torch.long)
            k = kv_idx.to(torch.long)
            causal_mask = q >= k
            document_mask = docs[q] == docs[k]
            return causal_mask & document_mask

        q_blocks = torch.arange(num_blocks, device=device, dtype=torch.int32)   
        t_rep = (q_blocks + 1) * block_size - 1  # [QB]
        t_rep_l = t_rep.to(torch.long)
        # Doc-start estimation (token-level): docs can change within blocks, so we
        # derive a per-token `doc_start_token` via change points + cummax, then gather
        # at representative tokens.
        # doc_start_token[t] = most recent index where docs changed (start of current doc)
        doc_change_tok = torch.ones((T,), device=device, dtype=torch.bool)
        doc_change_tok[1:] = docs[1:] != docs[:-1]
        tok_idx = torch.arange(T, device=device, dtype=torch.int32)
        doc_start_token = torch.where(doc_change_tok, tok_idx, torch.zeros_like(tok_idx)).cummax(dim=0).values  # [T]
        # Doc start block for the representative token's doc (used for pointer clamp).
        doc_start_blk_ptr = torch.floor_divide(doc_start_token[t_rep_l], block_size).to(torch.int32)  # [QB]
        # Doc start block for the *first* token in the query block (used for the local band).
        # This avoids collapsing the whole local band when doc boundaries occur inside a block.
        block_start_tok = (q_blocks.to(torch.int64) * int(block_size)).to(torch.long)  # [QB]
        doc_start_blk_local = torch.floor_divide(doc_start_token[block_start_tok], block_size).to(torch.int32)  # [QB]
        if not self.pointer_doc_clamp:
            doc_start_blk_ptr = torch.zeros_like(doc_start_blk_ptr)
            doc_start_blk_local = torch.zeros_like(doc_start_blk_local)
        reset_active = self._pointer_reset_active.to(device=device, dtype=torch.int32)
        reset_len = int(self._pointer_reset_len_active.item())
        if reset_active.item() != 0 and reset_len > 0:
            seg_tok_ptr = (t_rep_l // reset_len) * reset_len
            seg_blk_ptr = torch.floor_divide(seg_tok_ptr, block_size).to(torch.int32)
            seg_tok_local = (block_start_tok // reset_len) * reset_len
            seg_blk_local = torch.floor_divide(seg_tok_local, block_size).to(torch.int32)
            doc_start_blk_ptr = torch.maximum(doc_start_blk_ptr, seg_blk_ptr)
            doc_start_blk_local = torch.maximum(doc_start_blk_local, seg_blk_local)
        ptr_enabled = self._pointer_mask_active.to(device=device, dtype=torch.int32)
        ptr_half_active = self._pointer_half_blocks_active.to(device=device, dtype=torch.int32)

        # Representative Δ* per query block.
        if self.pointer_qblock_rep == "last":
            delta_rep = delta_star[0, :, t_rep_l]  # [H,QB,M]
            budget_rep = budget_logits[0, :, t_rep_l]  # [H,QB,M]
        else:
            delta_rep = delta_star[0].reshape(H, num_blocks, block_size, M).mean(dim=2)  # [H,QB,M]
            budget_rep = budget_logits[0].reshape(H, num_blocks, block_size, M).mean(dim=2)  # [H,QB,M]

        if self.pointer_radius_mode == "fixed":
            r_m = ptr_half_active.view(1, 1, 1).expand(H, num_blocks, M)  # [H,QB,M]
        else:
            budget_val = self.pointer_budget_blocks if pointer_budget_blocks is None else int(pointer_budget_blocks)
            budget_val = max(0, int(budget_val))
            if self.pointer_budget_is_total:
                budget_total_req = int(budget_val)
                budget_total_cap = int(local_blocks + M + (2 * M * int(ptr_half_blocks)))
                budget_total = min(budget_total_req, budget_total_cap)
                budget_extra_blocks = max(budget_total - int(local_blocks) - int(M), 0)
            else:
                budget_extra_blocks = int(budget_val)
            budget_rem = int(budget_extra_blocks) // 2
            budget_rem = min(int(budget_rem), int(M * int(ptr_half_blocks)))
            if budget_rem <= 0:
                r_m = delta_rep.new_zeros(delta_rep.shape, dtype=torch.int32)  # [H,QB,M]
            else:
                temp = max(float(self.pointer_budget_temp), 1e-6)
                u = (budget_rep / temp).to(dtype=torch.float32)
                rem = delta_rep.new_full((H, num_blocks), float(budget_rem), dtype=torch.float32)
                alloc = delta_rep.new_zeros(delta_rep.shape, dtype=torch.float32)
                for m in range(M - 1):
                    v = torch.sigmoid(u[..., m])
                    b = v * rem
                    alloc[..., m] = b
                    rem = rem - b
                alloc[..., M - 1] = rem

                floor = torch.floor(alloc)
                frac = alloc - floor
                floor_sum = floor.sum(dim=-1).to(torch.int64)
                rem_int = int(budget_rem) - floor_sum
                rem_int = rem_int.clamp(min=0, max=M).to(torch.int64)
                order = torch.argsort(frac, dim=-1, descending=True)
                rank = torch.argsort(order, dim=-1)
                add = (rank < rem_int[..., None]).to(floor.dtype)
                extra = floor + add
                r_m = torch.clamp(extra.to(torch.int32), min=0, max=int(ptr_half_blocks))

        r_m = torch.minimum(r_m, ptr_half_active)  # respect schedule cap
        r_m = torch.where(ptr_enabled.bool(), r_m, torch.zeros_like(r_m))

        if self.training and self.gumbel_topk and ptr_half_blocks > 0:
            k = int(self.gumbel_topk_k)
            if k <= 0 or k > M:
                k = M
            if k < M:
                gen = torch.Generator(device=device)
                gen.manual_seed(int(self._gumbel_seed.item()))
                u = torch.rand((H, num_blocks, M), device=device, generator=gen, dtype=torch.float32)
                g = -torch.log(-torch.log(u + self.eps) + self.eps)
                tau = max(float(self.gumbel_topk_tau), 1e-6)
                scores = budget_rep.to(torch.float32) / tau + g
                topk_idx = torch.topk(scores, k=k, dim=-1).indices
                peak_mask = torch.zeros_like(r_m, dtype=torch.bool)
                peak_mask.scatter_(dim=-1, index=topk_idx, value=True)
                r_m = torch.where(peak_mask, r_m, torch.zeros_like(r_m))

        key_tok = t_rep[None, :, None].to(delta_rep.dtype) - delta_rep
        key_tok = key_tok.clamp(min=0.0)
        ptr_center_blk = torch.floor_divide(key_tok.to(torch.int64), block_size).to(torch.int32)  # [H,QB,M]
        ptr_center_blk = torch.minimum(ptr_center_blk, q_blocks[None, :, None])
        ptr_center_blk = torch.where(ptr_enabled.bool(), ptr_center_blk, q_blocks[None, :, None].expand(H, -1, M))
        ptr_center_blk = torch.maximum(ptr_center_blk, doc_start_blk_ptr[None, :, None])

        pieces = []

        # (1) local window: [qb-(local_blocks-1) ... qb]
        local_blocks = min(local_blocks, num_blocks)
        offs_local = torch.arange(local_blocks, device=device, dtype=torch.int32)
        local = (q_blocks[None, :, None] - offs_local[None, None, :]).clamp(min=0)
        local = torch.maximum(local, doc_start_blk_local[None, :, None])
        pieces.append(local.expand(H, -1, -1))

        # (2) pointer windows around each Δ*_m: [center-p ... center+p]
        if ptr_half_blocks > 0:
            offs_ptr = torch.arange(-ptr_half_blocks, ptr_half_blocks + 1, device=device, dtype=torch.int32)
            offs_abs = offs_ptr.abs()
            active = offs_abs[None, None, None, :] <= r_m[:, :, :, None]  # [H,QB,M,2p+1]
            ptr = ptr_center_blk[:, :, :, None] + offs_ptr[None, None, None, :]
            ptr = ptr.clamp(min=0, max=num_blocks - 1)
            ptr = torch.minimum(ptr, q_blocks[None, :, None, None])
            ptr = torch.maximum(ptr, doc_start_blk_ptr[None, :, None, None])
            ptr = torch.where(active, ptr, ptr_center_blk[:, :, :, None])
            pieces.append(ptr.flatten(2, 3))  # [H,QB,M*(2p+1)]
        else:
            pieces.append(ptr_center_blk)  # [H,QB,M]

        # (3) global anchors: first N blocks
        global_max = min(self.pointer_global_blocks, num_blocks)
        if global_max > 0:
            # Doc-local anchors: doc_start_blk + [0..G-1], clamped by causality.
            g = torch.arange(global_max, device=device, dtype=torch.int32)
            g = doc_start_blk_ptr[None, :, None] + g[None, None, :]
            g = torch.minimum(g, q_blocks[None, :, None]).expand(H, -1, -1)
            if global_blocks is None:
                gb_active = self._pointer_global_blocks_active.to(device=device, dtype=torch.int32)
            else:
                gb_active = torch.tensor(int(global_blocks), device=device, dtype=torch.int32)
            gb_active = torch.clamp(gb_active, min=0, max=global_max)
            active = torch.arange(global_max, device=device, dtype=torch.int32) < gb_active
            g = torch.where(active[None, None, :], g, q_blocks[None, :, None].expand(H, num_blocks, global_max))
            pieces.append(g)

        kv_blocks = torch.cat(pieces, dim=-1)  # [H,QB,MAX_KV]
        max_kv = kv_blocks.size(-1)

        # De-duplicate per (head, q_block) without building a dense [QB,QB] mask.
        kv_sorted, _ = kv_blocks.sort(dim=-1)
        keep = torch.ones_like(kv_sorted, dtype=torch.bool)
        keep[..., 1:] = kv_sorted[..., 1:] != kv_sorted[..., :-1]
        kv_num_blocks = keep.sum(dim=-1, dtype=torch.int32)  # [H,QB]

        big = num_blocks + 1
        key = (~keep).to(torch.int32) * big + kv_sorted
        _, order = key.sort(dim=-1)
        kv_packed = kv_sorted.gather(dim=-1, index=order)

        if self.debug_stats:
            kv_hist = torch.bincount(kv_num_blocks.flatten().to(torch.int64), minlength=self._dbg_kv_unique_hist.numel())
            self._dbg_kv_unique_hist.copy_(kv_hist.to(torch.int32))
            pr_hist = torch.bincount(r_m.flatten().to(torch.int64), minlength=self._dbg_ptr_radius_hist.numel())
            self._dbg_ptr_radius_hist.copy_(pr_hist.to(torch.int32))
            extra_sum = r_m.sum(dim=-1)
            max_extra = self._dbg_ptr_extra_sum_hist.numel() - 1
            extra_sum = torch.clamp(extra_sum, max=max_extra)
            extra_hist = torch.bincount(extra_sum.flatten().to(torch.int64), minlength=self._dbg_ptr_extra_sum_hist.numel())
            self._dbg_ptr_extra_sum_hist.copy_(extra_hist.to(torch.int32))
            self._dbg_ptr_radius0_frac.copy_((r_m == 0).float().mean())

        # IMPORTANT: `BlockMask.from_kv_blocks` infers `kv_len` from the last dimension of
        # `kv_indices` times `BLOCK_SIZE`. We pass full-length K/V tensors into FlexAttention,
        # so `kv_len` must equal `T`, i.e. the list length must be `num_blocks`.
        kv_list_len = num_blocks
        if max_kv >= kv_list_len:
            kv_indices_h = kv_packed[..., :kv_list_len]
        else:
            kv_indices_h = kv_packed.new_empty((H, num_blocks, kv_list_len))
            kv_indices_h[..., :max_kv] = kv_packed
            kv_indices_h[..., max_kv:] = kv_packed[..., :1]

        kv_indices = kv_indices_h[None].contiguous().to(torch.int32)  # [1,H,QB,num_blocks]
        kv_num_blocks = kv_num_blocks[None].contiguous()  # [1,H,QB]
        zeros_num = torch.zeros_like(kv_num_blocks)
        zeros_idx = torch.zeros_like(kv_indices)

        return BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            zeros_num,
            zeros_idx,
            BLOCK_SIZE=block_size,
            mask_mod=document_causal,
        )
