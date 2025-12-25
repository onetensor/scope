import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import copy
import json
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
# Ensure each rank initializes CUDA on its own device (avoids every process creating a context on cuda:0).
if torch.cuda.is_available():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from model.attn_bias import SpectralBias
#torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min

# Optionally prevent torch.compile from capturing FlexAttention (useful when Inductor hits a FlexAttention lowering bug).
try:
    import torch._dynamo as _dynamo  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    _dynamo = None  # type: ignore[assignment]

def _flex_attention_eager(q, k, v, *, block_mask, score_mod=None, scale=None):
    return flex_attention(q, k, v, block_mask=block_mask, score_mod=score_mod, scale=scale)

@lru_cache(maxsize=None)
def _get_compiled_flex_attention(backend: str, mode: str | None, fullgraph: bool):
    # Compile FlexAttention as a standalone callable so graph-breaking around it
    # does not fall back to the unfused reference path (which materializes full
    # score matrices and is extremely slow/memory-heavy).
    return torch.compile(
        _flex_attention_eager,
        dynamic=False,
        backend=backend,
        mode=mode,
        fullgraph=fullgraph,
    )

def _flex_attention_call(q, k, v, *, block_mask, score_mod=None, scale=None):
    # NOTE: this function is typically executed under `_dynamo.disable` when
    # `FLEXATTN_DYNAMO_DISABLE=1`, so we do not assume any outer compilation.
    # If enabled, we run a separately-compiled FlexAttention to retain the
    # fused kernel (and avoid dense score materialization).
    use_compile = bool(globals().get("args", None) is not None and getattr(args, "flexattn_compile", True))
    if use_compile and hasattr(torch, "compile"):
        backend = str(getattr(args, "torch_compile_backend", "inductor"))
        mode = str(getattr(args, "torch_compile_mode", "default")).strip()
        if mode.lower() in {"", "none"}:
            mode = None  # type: ignore[assignment]
        return _get_compiled_flex_attention(backend, mode, False)(
            q, k, v, block_mask=block_mask, score_mod=score_mod, scale=scale
        )
    return _flex_attention_eager(q, k, v, block_mask=block_mask, score_mod=score_mod, scale=scale)

flex_attention_nocompile = _dynamo.disable(_flex_attention_call) if _dynamo is not None else _flex_attention_call

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng

@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    # Match the real kernel's layout: grad_w is returned as a transposed view
    # with strides (1, out_features), not contiguous (in_features, 1).
    grad_x = x_f8.to(torch.bfloat16)
    grad_w = w_f8.to(torch.float32).T.contiguous().T
    return grad_x, grad_w

def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(backward, setup_context=setup_context)

# -----------------------------------------------------------------------------
# Muon optimizer

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.mul_(1 - group["lr"] * group["weight_decay"])
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8: bool = False, x_s: float = 1.0, w_s: float = 1.0, grad_s: float = 1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128, spectral_cfg: dict | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.12
        self.spectral_bias = None
        if spectral_cfg is not None and spectral_cfg.get("enabled", False):
            cfg = dict(spectral_cfg)
            cfg.pop("enabled", None)
            self.spectral_bias = SpectralBias(head_dim=head_dim, num_heads=num_heads, **cfg)

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask, docs: Tensor) -> tuple[Tensor, Tensor]:
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        fa = flex_attention_nocompile if getattr(args, "flexattn_dynamo_disable", False) else flex_attention
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q_for_bias = q.transpose(1, 2)  # [B,H,T,D] before rotary
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v) # @KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = self.lambdas[0] * v
        reg_loss = x.new_zeros((), dtype=torch.float32)
        if self.spectral_bias is not None:
            (
                coeff_cos,
                coeff_sin,
                slope,
                b0,
                delta_star,
                _pi,
                delta_main,
                width,
                cos_wD,
                sin_wD,
                reg_loss,
            ) = self.spectral_bias(q_for_bias)
            beta = float(self.spectral_bias.beta)
            use_slope = bool(self.spectral_bias.use_slope)
            subtract_b0 = bool(self.spectral_bias.subtract_b0)
            gate_kind = self.spectral_bias.gate_kind
            ramp_lambda = float(self.spectral_bias.ramp_lambda)
            tau = float(self.spectral_bias.tau)
            K = int(self.spectral_bias.K)

            if self.spectral_bias.use_pointer_mask:
                block_mask = self.spectral_bias.build_pointer_blockmask(
                    docs=docs,
                    delta_star=delta_star,
                    block_size=128,
                )

            spectral_impl = str(getattr(args, "spectral_impl", "score_mod")).strip().lower()
            if spectral_impl == "qk_aug":
                # Implement beta * b_q(i-j) by augmenting Q/K with position-dependent features, avoiding score_mod.
                # b_q(Δ)=Σ_k A_k(q)cos(ω_kΔ)+B_k(q)sin(ω_kΔ) can be rewritten using cos(a-b), sin(a-b) identities.
                s = math.sqrt(beta / float(self.attn_scale))
                # cos/sin tables are [K,T]; transpose to [T,K] for broadcasting.
                cos_t = cos_wD.transpose(0, 1).to(dtype=torch.float32)  # [T,K]
                sin_t = sin_wD.transpose(0, 1).to(dtype=torch.float32)  # [T,K]
                cos_bhtk = cos_t[None, None].expand(B, self.num_heads, T, K)
                sin_bhtk = sin_t[None, None].expand(B, self.num_heads, T, K)

                Qcos = coeff_cos * cos_bhtk + coeff_sin * sin_bhtk  # [B,H,T,K]
                Qsin = coeff_cos * sin_bhtk - coeff_sin * cos_bhtk  # [B,H,T,K]
                q_pos = torch.cat([Qcos, Qsin], dim=-1)  # [B,H,T,2K]
                k_pos = torch.cat([cos_bhtk, sin_bhtk], dim=-1)  # [B,H,T,2K]

                if use_slope:
                    # slope*(i-j) differs from -slope*j by a per-row constant (slope*i), which is softmax-invariant.
                    pos = torch.arange(T, device=x.device, dtype=torch.float32)
                    q_pos = torch.cat([q_pos, (-slope).unsqueeze(-1)], dim=-1)
                    k_pos = torch.cat([k_pos, pos[None, None, :, None].expand(B, self.num_heads, T, 1)], dim=-1)

                q_pos = (q_pos * s).to(dtype=q.dtype)
                k_pos = (k_pos * s).to(dtype=q.dtype)

                q_bhtd = q.transpose(1, 2)
                k_bhtd = k.transpose(1, 2)
                v_bhtd = v.transpose(1, 2)
                pad_dim = int(q_pos.size(-1))
                q_aug = torch.cat([q_bhtd, q_pos], dim=-1)
                k_aug = torch.cat([k_bhtd, k_pos], dim=-1)
                v_aug = torch.cat([v_bhtd, v_bhtd.new_zeros((B, self.num_heads, T, pad_dim))], dim=-1)
                # Some kernels are much happier when the head dim is a small multiple (e.g. 8/16).
                align = int(getattr(args, "spectral_qk_aug_align", 16))
                if align > 1:
                    d_aug = int(q_aug.size(-1))
                    d_aligned = ((d_aug + align - 1) // align) * align
                    extra = d_aligned - d_aug
                    if extra > 0:
                        z = q_aug.new_zeros((B, self.num_heads, T, extra))
                        q_aug = torch.cat([q_aug, z], dim=-1)
                        k_aug = torch.cat([k_aug, z], dim=-1)
                        v_aug = torch.cat([v_aug, z], dim=-1)
                y = fa(q_aug, k_aug, v_aug, block_mask=block_mask, scale=self.attn_scale).transpose(1, 2)
                y = y[..., : self.head_dim]
            elif spectral_impl == "score_mod":
                def score_mod(score, b, h, q_idx, kv_idx):
                    # FlexAttention may call score_mod with scalar indices during fake/compile; normalize to 1-D.
                    q = q_idx.to(torch.long).reshape(-1)  # [Q]
                    k = kv_idx.to(torch.long).reshape(-1)  # [Kv]
                    # NOTE: max clamp is unnecessary for valid token indices (0..T-1) and
                    # adds extra work / temporaries during compilation.
                    delta = (q[:, None] - k[None, :]).clamp(min=0)  # [Q,Kv]

                    A = coeff_cos[b, h, q]  # [Q,K]
                    Bc = coeff_sin[b, h, q]  # [Q,K]

                    # Accumulate per-frequency to avoid materializing [Q,Kv,K] intermediates (shared-mem blowups).
                    bias = score.new_zeros(delta.shape, dtype=torch.float32)
                    for kk in range(K):
                        bias = bias + A[:, kk : kk + 1] * cos_wD[kk, delta] + Bc[:, kk : kk + 1] * sin_wD[kk, delta]

                    delta_f = delta.to(dtype=bias.dtype)
                    if subtract_b0:
                        bias = bias - b0[b, h, q][:, None]
                    if use_slope:
                        bias = bias + slope[b, h, q][:, None] * delta_f
                    if gate_kind != "none" and ramp_lambda > 0:
                        dm = delta_main[b, h, q][:, None]
                        w = width[b, h, q][:, None]
                        x = (delta_f - dm).abs()
                        x = (x - w) / tau
                        if gate_kind == "relu":
                            bias = bias - ramp_lambda * torch.relu(x)
                        else:
                            bias = bias - ramp_lambda * F.softplus(x)

                    bias = bias.reshape_as(score)
                    return score + beta * bias.to(dtype=score.dtype)

                y = fa(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    block_mask=block_mask,
                    score_mod=score_mod,
                    scale=self.attn_scale,
                ).transpose(1, 2)
            else:
                raise ValueError(f"unknown SCOPE spectral_impl={spectral_impl!r} (expected: qk_aug|score_mod)")
        else:
            y = fa(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                block_mask=block_mask,
                scale=self.attn_scale,
            ).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y, reg_loss

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int, spectral_cfg: dict | None = None):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len, spectral_cfg=spectral_cfg) if layer_idx != 7 else None
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask, docs: Tensor) -> tuple[Tensor, Tensor]:
        reg_loss = x.new_zeros((), dtype=torch.float32)
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            attn_out, attn_reg_loss = self.attn(norm(x), ve, block_mask, docs)
            x = x + attn_out
            reg_loss = reg_loss + attn_reg_loss
        x = x + self.mlp(norm(x))
        return x, reg_loss

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int, spectral_cfg: dict | None = None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i, spectral_cfg=spectral_cfg) for i in range(num_layers)])
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128), use_fp8=True, x_s=0.5, w_s=2**-9, grad_s=2**-19)
        self.lm_head.weight.detach().zero_() # @Grad62304977
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        self.skip_weights = nn.Parameter(torch.ones(num_layers//2))

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor, docs: Tensor | None = None):
        BLOCK_SIZE = 128
        if docs is None:
            docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            q = q_idx.to(torch.long)
            k = kv_idx.to(torch.long)
            causal_mask = q >= k
            document_mask = docs[q] == docs[k]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        # manual block mask creation by @YouJiacheng
        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)
        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )
        # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1
        docs = (input_seq == 50256).cumsum(0)

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks, docs=docs)
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
        assert len(block_masks) == len(self.blocks)

        x = x0 = norm(self.embed(input_seq)[None]) # use of norm here by @Grad62304977

        # U-net design by @brendanh0gan
        reg_loss = x.new_zeros((), dtype=torch.float32)
        skip_connections = []
        n = len(self.skip_weights)
        for i in range(len(self.blocks)):
            if i >= n:
                x = x + self.skip_weights[i - n] * skip_connections.pop()
            x, block_reg_loss = self.blocks[i](x, ve[i], x0, block_masks[i], docs)
            if self.training:
                reg_loss = reg_loss + block_reg_loss
            if i < n:
                skip_connections.append(x)

        x = norm(x)
        logits = self.lm_head(x)
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits.float() / 7.5)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq)
        if self.training:
            loss = loss + reg_loss
        return loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank : int, world_size : int):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
        pos += batch_size
        yield inputs, targets

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    # NOTE: very long seq lengths can trip FlexAttention/Inductor issues on some PyTorch nightlies.
    # Use env vars `TRAIN_SEQ_LEN` / `VAL_SEQ_LEN` to override.
    train_seq_len = 16*1024 # FlexAttention sequence length
    val_seq_len = train_seq_len # FlexAttention sequence length for validation
    # optimization
    num_iterations = 7050 # number of iterations to run
    cooldown_frac = 0.4 # fraction of training spent cooling down the learning rate
    seed = 1337 # for reproducible ablations
    # architecture
    vocab_size = 50257
    # SCOPE: spectral bias (see SPEC.md)
    spectral_bias = True
    spectral_impl = "qk_aug"  # "qk_aug" (no score_mod) | "score_mod"
    spectral_qk_aug_align = 16  # pad augmented head dim up to a multiple of this (e.g. 8/16) for kernel friendliness
    spectral_K = 6
    spectral_M = 2
    spectral_beta = 0.5
    spectral_L_train = 4096
    spectral_L_max = 1_000_000
    spectral_ramp_lambda = 0.2
    spectral_tau = 64.0
    spectral_gate_kind = "softplus"  # "none"|"softplus"|"relu"
    spectral_share_across_heads = True
    spectral_use_slope = True
    spectral_detach_q = False
    spectral_subtract_b0 = True
    spectral_width_min = 32.0
    spectral_width_max = 256.0
    spectral_delta_star_max = None  # None -> use L_max
    spectral_use_pointer_mask = True
    spectral_pointer_local_blocks = 16
    spectral_pointer_half_blocks = 4
    spectral_pointer_global_blocks = 2
    spectral_pointer_qblock_rep = "last"  # "last"|"mean"
    spectral_pointer_schedule = True
    spectral_pointer_schedule_disable_steps = 0 # temporary change to test short ablation runs, change back to 300 or something else
    spectral_pointer_schedule_mid_steps = 1500
    spectral_pointer_schedule_half_blocks_mid = 2
    # Validation override for pointer mask (default: use scheduled state).
    spectral_pointer_val_force = False
    spectral_pointer_val_half_blocks = -1  # <0 -> use spectral_pointer_half_blocks
    spectral_lambda_omega = 1e-5
    spectral_lambda_zero_mean = 1e-4
    spectral_lambda_entropy = 1e-4
    # evaluation and logging
    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end
    scope_log_every = 200 # rank0-only SCOPE debug stats
    save_checkpoint = False
    # logging / diagnostics
    log_all_ranks = False  # write one logfile per rank (debugging)       
    flexattn_dynamo_disable = False  # graph-break around FlexAttention (compiler workaround)
    flexattn_compile = True  # compile FlexAttention standalone when graph-broken (keeps fused kernel)
    # torch.compile controls (FlexAttention still JITs its own kernels)   
    torch_compile = True
    torch_compile_backend = "inductor"
    torch_compile_mode = "default"  # "default"|"reduce-overhead"|"max-autotune"
    torch_compile_fullgraph = False
    torch_dynamo_verbose = False
    torch_dynamo_suppress_errors = False  # fallback to eager inside torch.compile
    torch_compile_fallback_to_eager = True  # if compile errors escape, retry eager
args = Hyperparameters()

def _env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    return default if v is None else str(v)

def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    return default if v is None else int(v)

def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    return default if v is None else float(v)

def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean env var {key}={v!r}")

# Lightweight env overrides (for ablations / Vast.ai runs).
args.train_files = _env_str("TRAIN_FILES", args.train_files)
args.val_files = _env_str("VAL_FILES", args.val_files)
args.val_tokens = _env_int("VAL_TOKENS", args.val_tokens)
args.train_seq_len = _env_int("TRAIN_SEQ_LEN", args.train_seq_len)
args.val_seq_len = _env_int("VAL_SEQ_LEN", args.val_seq_len)
args.num_iterations = _env_int("NUM_ITERATIONS", args.num_iterations)
args.val_loss_every = _env_int("VAL_LOSS_EVERY", args.val_loss_every)
args.scope_log_every = _env_int("SCOPE_LOG_EVERY", args.scope_log_every)
args.save_checkpoint = _env_bool("SAVE_CHECKPOINT", args.save_checkpoint)
args.log_all_ranks = _env_bool("LOG_ALL_RANKS", args.log_all_ranks)       
args.flexattn_dynamo_disable = _env_bool("FLEXATTN_DYNAMO_DISABLE", args.flexattn_dynamo_disable)
args.flexattn_compile = _env_bool("FLEXATTN_COMPILE", args.flexattn_compile)
args.cooldown_frac = _env_float("COOLDOWN_FRAC", args.cooldown_frac)      
args.seed = _env_int("SEED", args.seed)

# SCOPE toggles/knobs.
args.spectral_bias = _env_bool("SPECTRAL_BIAS", args.spectral_bias)
args.spectral_impl = _env_str("SPECTRAL_IMPL", args.spectral_impl)
args.spectral_qk_aug_align = _env_int("SPECTRAL_QK_AUG_ALIGN", args.spectral_qk_aug_align)
args.spectral_K = _env_int("SPECTRAL_K", args.spectral_K)
args.spectral_M = _env_int("SPECTRAL_M", args.spectral_M)
args.spectral_beta = _env_float("SPECTRAL_BETA", args.spectral_beta)
args.spectral_use_slope = _env_bool("SPECTRAL_USE_SLOPE", args.spectral_use_slope)
args.spectral_ramp_lambda = _env_float("SPECTRAL_RAMP_LAMBDA", args.spectral_ramp_lambda)
args.spectral_gate_kind = _env_str("SPECTRAL_GATE_KIND", args.spectral_gate_kind)
args.spectral_use_pointer_mask = _env_bool("SPECTRAL_USE_POINTER_MASK", args.spectral_use_pointer_mask)
args.spectral_pointer_schedule = _env_bool("SPECTRAL_POINTER_SCHEDULE", args.spectral_pointer_schedule)
args.spectral_pointer_schedule_disable_steps = _env_int("SPECTRAL_POINTER_SCHEDULE_DISABLE_STEPS", args.spectral_pointer_schedule_disable_steps)
args.spectral_pointer_schedule_mid_steps = _env_int("SPECTRAL_POINTER_SCHEDULE_MID_STEPS", args.spectral_pointer_schedule_mid_steps)
args.spectral_pointer_schedule_half_blocks_mid = _env_int("SPECTRAL_POINTER_SCHEDULE_HALF_BLOCKS_MID", args.spectral_pointer_schedule_half_blocks_mid)
args.spectral_pointer_val_force = _env_bool("SPECTRAL_POINTER_VAL_FORCE", args.spectral_pointer_val_force)
args.spectral_pointer_val_half_blocks = _env_int("SPECTRAL_POINTER_VAL_HALF_BLOCKS", args.spectral_pointer_val_half_blocks)
args.spectral_pointer_local_blocks = _env_int("SPECTRAL_POINTER_LOCAL_BLOCKS", args.spectral_pointer_local_blocks)
args.spectral_pointer_half_blocks = _env_int("SPECTRAL_POINTER_HALF_BLOCKS", args.spectral_pointer_half_blocks)
args.spectral_pointer_global_blocks = _env_int("SPECTRAL_POINTER_GLOBAL_BLOCKS", args.spectral_pointer_global_blocks)

# torch.compile toggles/knobs.
args.torch_compile = _env_bool("TORCH_COMPILE", args.torch_compile)
args.torch_compile_backend = _env_str("TORCH_COMPILE_BACKEND", args.torch_compile_backend)
args.torch_compile_mode = _env_str("TORCH_COMPILE_MODE", args.torch_compile_mode)
args.torch_compile_fullgraph = _env_bool("TORCH_COMPILE_FULLGRAPH", args.torch_compile_fullgraph)
args.torch_dynamo_verbose = _env_bool("TORCHDYNAMO_VERBOSE", args.torch_dynamo_verbose)
args.torch_dynamo_suppress_errors = _env_bool("TORCHDYNAMO_SUPPRESS_ERRORS", args.torch_dynamo_suppress_errors)
args.torch_compile_fallback_to_eager = _env_bool("TORCH_COMPILE_FALLBACK_TO_EAGER", args.torch_compile_fallback_to_eager)

# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert world_size == 8 # this code is designed for 8xH100
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# Best-effort cleanup on exceptions (avoids NCCL resource leak warnings).
import atexit
def _destroy_process_group():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
atexit.register(_destroy_process_group)

# Deterministic init across ablation runs (rank0 params are broadcast anyway).
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
import numpy as np
np.random.seed(args.seed)

# begin logging
log_dir = os.environ.get("LOG_DIR", "logs")
run_name = os.environ.get("RUN_NAME", "").strip()
run_uuid = str(uuid.uuid4()) if master_process else ""
_uuid_list = [run_uuid]
dist.broadcast_object_list(_uuid_list, src=0)
run_uuid = _uuid_list[0]
run_prefix = f"{run_name}_" if run_name else ""
run_id = f"{run_prefix}{run_uuid}"
os.makedirs(log_dir, exist_ok=True)
logfile = None
if args.log_all_ranks:
    logfile = f"{log_dir}/{run_id}_rank{rank}.txt"
    if master_process:
        print(f"{log_dir}/{run_id}_rank*.txt")
else:
    logfile = f"{log_dir}/{run_id}.txt" if master_process else None
    if master_process:
        print(logfile)

def _write_log(s: str):
    if logfile is None:
        return
    with open(logfile, "a") as f:
        print(s, file=f)
def print0(s, console=False):
    if master_process:
        _write_log(s)
        if console:
            print(s)
def print_rank(s, console=False):
    _write_log(s)
    if console:
        print(f"[rank{rank}] {s}")

# Ensure uncaught exceptions are captured in logs (useful for compiler failures).
_orig_excepthook = sys.excepthook
def _logging_excepthook(exc_type, exc, tb):
    import traceback
    msg = "".join(traceback.format_exception(exc_type, exc, tb))
    try:
        print_rank(msg, console=True)
    except Exception:
        print(msg)
    _orig_excepthook(exc_type, exc, tb)
sys.excepthook = _logging_excepthook

# begin by printing this file (the Python code)
print0(code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)
print0(
    json.dumps(
        dict(
            tag="RUN_CFG",
            seed=int(args.seed),
            train_seq_len=int(args.train_seq_len),
            val_seq_len=int(args.val_seq_len),
            num_iterations=int(args.num_iterations),
            val_loss_every=int(args.val_loss_every),
            world_size=int(world_size),
            torch_compile=dict(
                enabled=bool(args.torch_compile),
                backend=str(args.torch_compile_backend),
                mode=str(args.torch_compile_mode),
                fullgraph=bool(args.torch_compile_fullgraph),
                dynamo_verbose=bool(args.torch_dynamo_verbose),
                dynamo_suppress_errors=bool(args.torch_dynamo_suppress_errors),
                fallback_to_eager=bool(args.torch_compile_fallback_to_eager),
            ),
            flexattn=dict(
                dynamo_disable=bool(args.flexattn_dynamo_disable),
                compile=bool(args.flexattn_compile),
            ),
            scope=dict(
                enabled=bool(args.spectral_bias),
                impl=str(args.spectral_impl),
                qk_aug_align=int(getattr(args, "spectral_qk_aug_align", 0)),
                K=int(args.spectral_K),
                M=int(args.spectral_M),
                beta=float(args.spectral_beta),
                use_slope=bool(args.spectral_use_slope),
                ramp_lambda=float(args.spectral_ramp_lambda),
                gate_kind=str(args.spectral_gate_kind),
                use_pointer_mask=bool(args.spectral_use_pointer_mask),
                pointer_schedule=bool(args.spectral_pointer_schedule),
                pointer_local_blocks=int(args.spectral_pointer_local_blocks),
                pointer_half_blocks=int(args.spectral_pointer_half_blocks),
                pointer_global_blocks=int(args.spectral_pointer_global_blocks),
                val_force=bool(args.spectral_pointer_val_force),
                val_half_blocks=int(args.spectral_pointer_val_half_blocks),
            ),
        )
    ),
    console=True,
)

########################################
#    Construct model and optimizer     #
########################################

model: nn.Module = GPT(vocab_size=args.vocab_size, num_layers=16, num_heads=8, model_dim=1024,
                       max_seq_len=max(args.train_seq_len, args.val_seq_len),
                       spectral_cfg=dict(
                           enabled=args.spectral_bias,
                           K=args.spectral_K,
                           M=args.spectral_M,
                           beta=args.spectral_beta,
                           L_train=args.spectral_L_train,
                           L_max=args.spectral_L_max,
                           delta_star_max=(args.spectral_L_max if args.spectral_delta_star_max is None else args.spectral_delta_star_max),
                           ramp_lambda=args.spectral_ramp_lambda,
                           tau=args.spectral_tau,
                           gate_kind=args.spectral_gate_kind,
                           share_across_heads=args.spectral_share_across_heads,
                           use_slope=args.spectral_use_slope,
                           detach_q=args.spectral_detach_q,
                           subtract_b0=args.spectral_subtract_b0,
                           width_min=args.spectral_width_min,
                           width_max=args.spectral_width_max,
                           use_pointer_mask=args.spectral_use_pointer_mask,
                           pointer_local_blocks=args.spectral_pointer_local_blocks,
                           pointer_half_blocks=args.spectral_pointer_half_blocks,
                           pointer_global_blocks=args.spectral_pointer_global_blocks,
                           pointer_qblock_rep=args.spectral_pointer_qblock_rep,
                           lambda_omega=args.spectral_lambda_omega,
                           lambda_zero_mean=args.spectral_lambda_zero_mean,
                           lambda_entropy=args.spectral_lambda_entropy,
                           debug_stats=master_process,
                       )).cuda()
for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

spectral_bias_modules: list[SpectralBias] = []
if args.spectral_bias:
    for m in model.modules():
        sb = getattr(m, "spectral_bias", None)
        if sb is not None:
            spectral_bias_modules.append(sb)
if master_process and spectral_bias_modules:
    sb0 = spectral_bias_modules[0]
    print0(
        json.dumps(
            dict(
                tag="SCOPE_BINS",
                delta_tokens_edges=sb0._dbg_token_edges.tolist(),
                delta_blocks_edges=sb0._dbg_block_edges.tolist(),
                ptr_center_blocks_edges=sb0._dbg_block_edges.tolist(),
                kv_unique_blocks_max=int(sb0._dbg_kv_unique_hist.numel() - 1),
            )
        ),
        console=True,
    )
pointer_mask_state = dict(enabled=None, half_blocks=None)
def maybe_update_pointer_mask(step: int):
    if not spectral_bias_modules:
        return
    if not args.spectral_use_pointer_mask:
        enabled = False
        half_blocks = 0
    elif not args.spectral_pointer_schedule:
        enabled = True
        half_blocks = int(args.spectral_pointer_half_blocks)
    elif step < args.spectral_pointer_schedule_disable_steps:
        enabled = False
        half_blocks = 0
    elif step < args.spectral_pointer_schedule_mid_steps:
        enabled = True
        half_blocks = min(int(args.spectral_pointer_schedule_half_blocks_mid), int(args.spectral_pointer_half_blocks))
    else:
        enabled = True
        half_blocks = int(args.spectral_pointer_half_blocks)
    if pointer_mask_state["enabled"] == enabled and pointer_mask_state["half_blocks"] == half_blocks:
        return
    for sb in spectral_bias_modules:
        sb.set_pointer_mask_state(enabled=enabled, half_blocks=half_blocks)
    pointer_mask_state["enabled"] = enabled
    pointer_mask_state["half_blocks"] = half_blocks

def _hist_percentile(hist: Tensor, q: float) -> int | None:
    total = int(hist.sum().item())
    if total <= 0:
        return None
    cdf = hist.cumsum(0)
    target = q * total
    idx = int((cdf >= target).nonzero(as_tuple=False)[0].item())
    return idx

def log_scope_stats(*, step: int, train_loss: Tensor | None = None, sliding_window_blocks: int | None = None):
    if not master_process:
        return
    if not spectral_bias_modules:
        return
    if args.scope_log_every <= 0:
        return

    sbs = spectral_bias_modules
    sb0 = sbs[0]

    kv_hist = sum((sb._dbg_kv_unique_hist for sb in sbs), start=torch.zeros_like(sb0._dbg_kv_unique_hist))
    kv_total = int(kv_hist.sum().item())
    if kv_total > 0:
        ks = torch.arange(kv_hist.numel(), device=kv_hist.device, dtype=torch.float32)
        kv_mean = float((ks * kv_hist.float()).sum().item() / kv_total)
        kv_p50 = _hist_percentile(kv_hist, 0.50)
        kv_p90 = _hist_percentile(kv_hist, 0.90)
        kv_max = int(torch.nonzero(kv_hist, as_tuple=False).max().item())
    else:
        kv_mean, kv_p50, kv_p90, kv_max = None, None, None, None

    dt_hist = sum((sb._dbg_delta_tokens_hist for sb in sbs), start=torch.zeros_like(sb0._dbg_delta_tokens_hist))
    db_hist = sum((sb._dbg_delta_blocks_hist for sb in sbs), start=torch.zeros_like(sb0._dbg_delta_blocks_hist))
    pc_hist = sum((sb._dbg_ptr_center_hist for sb in sbs), start=torch.zeros_like(sb0._dbg_ptr_center_hist))

    dt_total = int(dt_hist.sum().item())
    db_total = int(db_hist.sum().item())
    pc_total = int(pc_hist.sum().item())

    def _to_frac_list(hist: Tensor, total: int) -> list[float]:
        if total <= 0:
            return []
        return (hist.float() / float(total)).tolist()

    outside_fracs = torch.stack([sb._dbg_ptr_outside_local_frac for sb in sbs]).float()
    block0_fracs = torch.stack([sb._dbg_ptr_block0_frac for sb in sbs]).float()
    pi_entropies = torch.stack([sb._dbg_pi_entropy_mean for sb in sbs]).float()

    reg_omega = float(torch.stack([sb._dbg_reg_omega for sb in sbs]).sum().item())
    reg_entropy = float(torch.stack([sb._dbg_reg_entropy for sb in sbs]).sum().item())
    reg_zero_mean = float(torch.stack([sb._dbg_reg_zero_mean for sb in sbs]).sum().item())
    reg_total = float(torch.stack([sb._dbg_reg_total for sb in sbs]).sum().item())

    payload = dict(
        tag="SCOPE_STATS",
        step=int(step),
        train_loss=(float(train_loss.item()) if train_loss is not None else None),
        sliding_window_blocks=int(sliding_window_blocks) if sliding_window_blocks is not None else None,
        pointer_mask=dict(enabled=bool(pointer_mask_state["enabled"]), half_blocks=int(pointer_mask_state["half_blocks"])),
        pointer_cfg=dict(
            local_blocks=int(sb0.pointer_local_blocks),
            half_blocks_max=int(sb0.pointer_half_blocks),
            global_blocks=int(sb0.pointer_global_blocks),
            qblock_rep=str(sb0.pointer_qblock_rep),
        ),
        kv_unique_blocks=dict(mean=kv_mean, p50=kv_p50, p90=kv_p90, max=kv_max),
        ptr_outside_local_frac=dict(mean=float(outside_fracs.mean().item()), min=float(outside_fracs.min().item()), max=float(outside_fracs.max().item())),
        ptr_center_block0_frac=dict(mean=float(block0_fracs.mean().item()), min=float(block0_fracs.min().item()), max=float(block0_fracs.max().item())),
        pi_entropy_mean=dict(mean=float(pi_entropies.mean().item()), min=float(pi_entropies.min().item()), max=float(pi_entropies.max().item())),
        reg=dict(omega=reg_omega, entropy=reg_entropy, zero_mean=reg_zero_mean, total=reg_total),
        delta_tokens_hist=_to_frac_list(dt_hist, dt_total),
        delta_blocks_hist=_to_frac_list(db_hist, db_total),
        ptr_center_blocks_hist=_to_frac_list(pc_hist, pc_total),
    )
    print0(json.dumps(payload), console=True)

# collect the parameters to optimize
spectral_params = [p for n, p in model.named_parameters() if "spectral_bias" in n]
hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n and "spectral_bias" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for n, p in model.named_parameters() if p.ndim < 2 and "spectral_bias" not in n]
head_params = [model.lm_head.weight]

# init the optimizer(s)
adam_params = [
    dict(params=head_params, lr=0.1/1024**0.5),
    dict(params=embed_params, lr=0.3),
    dict(params=scalar_params, lr=0.015),
]
if spectral_params:
    adam_params.append(dict(params=spectral_params, lr=0.015))
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
optimizer2 = Muon(hidden_matrix_params, lr=0.025, momentum=0.95, rank=rank, world_size=world_size)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: stable then decay
def get_lr(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        return (1 - x) / args.cooldown_frac

# attention window size schedule: linearly increase
@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
def get_window_size_blocks(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    window_size = next_multiple_of_n(1728 * x, n=128)
    return get_window_size_blocks_helper(window_size)

maybe_update_pointer_mask(0)
compile_enabled = bool(args.torch_compile)
if compile_enabled:
    try:
        import torch._dynamo as _dynamo  # type: ignore[import-not-found]

        _dynamo.config.verbose = bool(args.torch_dynamo_verbose)
        _dynamo.config.suppress_errors = bool(args.torch_dynamo_suppress_errors)
    except Exception as e:
        print_rank(json.dumps(dict(tag="DYNAMO_CONFIG_ERROR", error=str(e))), console=True)

    compile_mode = args.torch_compile_mode.strip()
    if compile_mode.lower() in {"", "none"}:
        compile_mode = None  # type: ignore[assignment]
    print0(
        json.dumps(
            dict(
                tag="TORCH_COMPILE_CFG",
                enabled=True,
                backend=args.torch_compile_backend,
                mode=compile_mode,
                fullgraph=bool(args.torch_compile_fullgraph),
                dynamo_verbose=bool(args.torch_dynamo_verbose),
                dynamo_suppress_errors=bool(args.torch_dynamo_suppress_errors),
                fallback_to_eager=bool(args.torch_compile_fallback_to_eager),
            )
        ),
        console=True,
    )
    model = torch.compile(
        model,
        dynamic=False,
        backend=args.torch_compile_backend,
        mode=compile_mode,
        fullgraph=bool(args.torch_compile_fullgraph),
    )
else:
    print0(json.dumps(dict(tag="TORCH_COMPILE_CFG", enabled=False)), console=True)

def _unwrap_compiled(m: nn.Module) -> nn.Module:
    return getattr(m, "_orig_mod", m)

def _is_compiler_error(e: BaseException) -> bool:
    try:
        from torch._dynamo.exc import BackendCompilerFailed  # type: ignore[import-not-found]
    except Exception:
        BackendCompilerFailed = ()  # type: ignore[assignment]
    try:
        from torch._inductor.exc import InductorError  # type: ignore[import-not-found]
    except Exception:
        InductorError = ()  # type: ignore[assignment]
    if isinstance(e, (BackendCompilerFailed, InductorError)):  # type: ignore[arg-type]
        return True
    msg = repr(e)
    return any(s in msg for s in ("torch._dynamo", "torch._inductor", "triton", "InductorError", "BackendCompilerFailed"))

def _log_exception(e: BaseException, *, where: str):
    import traceback

    payload = dict(
        tag="EXCEPTION",
        where=str(where),
        rank=int(rank),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=int(world_size),
        compile_enabled=bool(compile_enabled),
        pointer_mask=dict(enabled=pointer_mask_state.get("enabled"), half_blocks=pointer_mask_state.get("half_blocks")),
        error_type=type(e).__name__,
        error=str(e),
    )
    print_rank(json.dumps(payload), console=True)
    # Full traceback: rank-local file if enabled, otherwise best-effort to rank0.
    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    print_rank(tb, console=False)
    if master_process:
        print0(tb, console=True)
    if torch.cuda.is_available():
        mem = dict(
            allocated=int(torch.cuda.memory_allocated() // (1024**2)),
            reserved=int(torch.cuda.memory_reserved() // (1024**2)),
            max_allocated=int(torch.cuda.max_memory_allocated() // (1024**2)),
            max_reserved=int(torch.cuda.max_memory_reserved() // (1024**2)),
        )
        print_rank(json.dumps(dict(tag="CUDA_MEM_MIB", where=str(where), **mem)), console=True)

def run_with_compile_fallback(fn, *, where: str):
    global model, compile_enabled
    try:
        return fn()
    except Exception as e:
        _log_exception(e, where=where)
        if compile_enabled and args.torch_compile_fallback_to_eager and _is_compiler_error(e):
            model = _unwrap_compiled(model)
            compile_enabled = False
            print_rank(json.dumps(dict(tag="TORCH_COMPILE_FALLBACK", where=str(where))), console=True)
            return fn()
        raise

########################################
#            Warmup kernels            #
########################################

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 10
initial_state = dict(model=copy.deepcopy(_unwrap_compiled(model).state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
for _ in range(warmup_steps):
    inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device="cuda")
    def _warmup_fwd_bwd():
        model(inputs.to(torch.int32), targets, get_window_size_blocks(0)).backward()
    run_with_compile_fallback(_warmup_fwd_bwd, where="warmup_fwd_bwd")
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
_unwrap_compiled(model).load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del initial_state

########################################
#        Training and validation       #
########################################

train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    maybe_update_pointer_mask(step)
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        val_pointer_forced = False
        if spectral_bias_modules and args.spectral_use_pointer_mask and args.spectral_pointer_val_force:
            val_pointer_forced = True
            hb = int(args.spectral_pointer_val_half_blocks)
            if hb < 0:
                hb = int(args.spectral_pointer_half_blocks)
            for sb in spectral_bias_modules:
                sb.set_pointer_mask_state(enabled=True, half_blocks=hb)
            pointer_mask_state["enabled"] = True
            pointer_mask_state["half_blocks"] = hb
        model.eval()
        val_batch_size = world_size * args.val_seq_len
        assert args.val_tokens % val_batch_size == 0
        val_steps = args.val_tokens // val_batch_size
        val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = next(val_loader)
                def _val_fwd():
                    return model(inputs, targets, get_window_size_blocks(step))
                val_loss += run_with_compile_fallback(_val_fwd, where=f"val_fwd_step{step}")
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        model.train()
        if val_pointer_forced:
            maybe_update_pointer_mask(step)
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"{log_dir}/{run_id}", exist_ok=True)
            torch.save(log, f"{log_dir}/{run_id}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    window_blocks = get_window_size_blocks(step)
    def _train_fwd_bwd():
        loss = model(inputs, targets, window_blocks)
        loss.backward()
        return loss
    train_loss = run_with_compile_fallback(_train_fwd_bwd, where=f"train_fwd_bwd_step{step}")
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    # set optimization hyperparameters
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
    for group in optimizer2.param_groups:
        frac = min(step / 300, 1) # momentum warmup for muon
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers
    for opt in optimizers:
        opt.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)
    if args.scope_log_every > 0 and (step + 1) % args.scope_log_every == 0:
        log_scope_stats(step=step + 1, train_loss=train_loss.detach(), sliding_window_blocks=int(window_blocks.item()))

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()
