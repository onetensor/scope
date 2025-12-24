# Spec for:
# SCOPE: Spectral COntext Pointer Encoding for Long-Context Transformers

## 0) Goal

Train at 4k–8k context on 8×H100 (bf16). At inference/eval, run reliably at 128k–1M while **selectively** concentrating attention on a few query-chosen offsets and suppressing the rest—without hurting short-range quality.

---

## 1) Model & Training Skeleton

**Backbone**: GPT-2-medium equivalent

* layers: 24
* d_model: 1024
* n_heads: 16
* head_dim: 64
* ffn: 4×, SwiGLU, RMSNorm
* attn: FlashAttention-2, causal, with RoPE (or p-RoPE/YaRN as base; see toggles below)

**Sequence length (train)**: 4096 (8192 if comfy)

**Optimizer/schedule**

* AdamW(β1=0.9, β2=0.95, wd=0.1), grad_clip=1.0
* LR peak: 1.5e-4; cosine decay; warmup 2k steps
* Tokens: 1.2B–1.5B total (≈50–70k steps @ global batch ≈ 2–4M tokens/step)
* Mixed precision: bf16
* DDP/FSDP: sharded params + activation checkpointing

**Datasets**: SlimPajama/RefinedWeb subset or your internal mix. Shuffle by document; pack with EOD tokens.

---

## 2) Components (toggles)

* `rope_base`: `"rope"|"pi"|"yarn"|"p_rope"`
* `sia_envelope`: `true|false`  (scale-invariant attention pre-transform; optional but recommended)
* `spectral_bias`: `true|false`  (**this work**)

You can stack: `sia_envelope + spectral_bias` on top of the chosen RoPE variant.

---

## 3) Spectral Bias (this work)

### 3.1 Math

For a query at position (i) and a key at (j) (Δ = i − j ≥ 0), modify the logit:
[
\ell_{ij}^{\text{final}}
= \underbrace{\ell_{ij}^{\text{base}}}_{QK/\sqrt{d}}
\quad (+\ \text{SIA envelope if enabled})
\quad +\ \beta; b_q(\Delta).
]

We synthesize a **query-conditioned, band-limited, multi-peak** bias:
[
b_q(\Delta)=\sum_{m=1}^{M}\pi_m(q)\sum_{k=1}^{K} a_{m,k}(q)\cos!\Big(\omega_k(\Delta-\Delta^\star_m(q))\Big)
;+; c(q)\Delta
;-;\lambda_{\text{ramp}}\ \text{softplus}!\Big(\tfrac{|\Delta-\Delta^\star_{\text{main}}|-W(q)}{\tau}\Big).
]

* (M) = number of “pointers” (peaks), default **2**.
* (K) = number of spectral bands, default **6** (log-spaced).
* (\omega_k \in [2\pi/L_{\max},,2\pi/L_{\text{train}}]); with (L_{\max}=10^6).
* (\Delta^\star_m(q) \in [0, L_{\max}]): query-predicted offsets (linear phase ⇒ translation).
* (a_{m,k}(q)): non-neg. spectral weights (log-normal over (\log \omega)).
* (\pi_m(q)): mixture weights (softmax).
* (c(q)\Delta): optional global slope (small).
* The final softplus term is an optional gentle trough outside a main window (see §7).

**Stability knobs**

* Band-limit high (\omega) by construction.
* Penalize high-freq energy: (\lambda_\omega\sum_{m,k}\omega_k^2 a_{m,k}(q)^2).
* Zero-mean constraint: (\lambda_0\big(\sum_{\Delta=0}^{L} b_q(\Delta)\big)^2) (prevents logit temperature drift).
* Entropy on (\pi) to avoid trivial single-peak unless the data demand it.

### 3.2 Tensors & API

Create a module **`SpectralBias`** that returns a **bias matrix** shaped `[B, H, L, L]` (masked causal), or a **per-step vector** `[B, H, i+1]` for decoding.

**Constructor args**

```python
SpectralBias(
    head_dim: int,
    L_max: int = 1_000_000,
    K: int = 6,               # spectral bands
    M: int = 2,               # number of peaks
    beta: float = 0.5,        # bias scale
    ramp_lambda: float = 0.2, # optional trough scale
    tau: float = 64.0,        # ramp temperature in tokens
    w_min=None, w_max=None,   # override frequencies if desired
    share_across_heads: bool = False,   # share small MLP across heads
    use_slope: bool = True,
)
```

**Predictions per query token (per head or shared):**

* `Delta_star`: `[B, H, L, M]` (squashed to `[0, L_max]`)
* `mu, sigma`: `[B, H, L, M]` (for log-normal over log ω)
* `pi`: `[B, H, L, M]` (softmax)
* `c`: `[B, H, L, 1]` (tanh-squashed, tiny range)

Use a tiny **MLP** with input `q_i` (optionally stop-grad through `q_i` to stabilize early)
`q → Linear(d_head→4M+2M+M+1) → SiLU → Linear(…→ (Δ*, μ, σ, π, c))`.

**Frequency grid**

```python
w_min = 2*math.pi / L_max
w_max = 2*math.pi / L_train        # L_train = 4096 or 8192
omegas = torch.logspace(math.log10(w_min), math.log10(w_max), K)  # [K]
```

**Spectral weights**

```python
# log-normal over log(omega) per pointer m
a_mk ∝ exp(- (log(ω_k) - μ_m)^2 / (2 σ_m^2)); normalize over k
```

**Efficient construction of b_q(Δ)**
Use the identity `cos(ω(Δ-Δ*)) = cos(ωΔ)cos(ωΔ*) + sin(ωΔ)sin(ωΔ*)`.

Precompute once per sequence length:

```python
cos_wD = cos(ω_k * arange(0..L-1))  # [K, L]
sin_wD = sin(ω_k * arange(0..L-1))  # [K, L]
```

For each query `i` (vectorized over batch/heads):

1. compute `cos(ω Δ*_m(i))`, `sin(ω Δ*_m(i))` → `[B,H,M,K]`
2. mix with `a_mk(i)` and `π_m(i)` to get coefficients on `cos_wD` and `sin_wD`
3. form `b_i(Δ) = Σ_{m,k} π_m a_mk [cos_wD[k,Δ] * cos(ω_k Δ*_m) + sin_wD[k,Δ] * sin(ω_k Δ*_m)]`
4. add `c(i) * Δ` and the optional softplus trough around the **main** pointer (argmax π)

Place `b_i(Δ)` into the lower-triangular band for the i-th row of the bias matrix (Δ = i − j). Memory note: for training with L=4k you can materialize `[B,H,L,L]` in bf16; for decoding compute only the current row.

**Numerical hygiene**

* Clamp `σ ∈ [0.25, 2.0]` (over log-ω); init μ near mid-range, σ≈1.0.
* Squash `Δ*` by `Δ* = L_max * sigmoid(raw)`.
* Init final linear to near-zero so early training behaves like no bias.
* After adding bias, **don’t** change attention scaling (keep FlashAttn softmax stable).

---

## 4) Scale-Invariant Attention (optional but recommended)

Implement SIA as a **position-only** affine transform of logits before adding the spectral bias:
[
L_{ij}^{\text{SIA}} = a_{\Delta}, L_{ij}^{\text{base}} + m_{\Delta},
]
where (a_\Delta, m_\Delta) are functions of distance (e.g., piecewise-constant over **log-distance bins**). Learn them as small tables per layer/head or shared, regularized to produce flat entropy across bins. (Keep them lightweight; you just need the envelope.)

Order of ops inside attention:

```
L = (QK^T)/sqrt(d)
L = SIA(L)            # if enabled
L = L + beta * b_q(Δ) # spectral, query-conditioned
L = L + mask          # causal mask
P = softmax(L)
```

---

## 5) Regularizers & Loss Terms

Add to the main LM loss:

* `λ_ω * sum( (ω^2) * a^2 )`  (e.g., 1e-5)
* `λ_0 * mean_over_batch( mean_Δ b_q(Δ) )^2` (1e-4)
* `λ_H * mean( -Σ π log π )` to keep >1 peak early (1e-4)
* If you **don’t** hard-enforce linear phase via `Δ*`, add a small **phase linearity** penalty; here we already use linear phase, so N/A.

Curriculum:

* steps 0–2k: freeze module outputs to `Δ*=0`, `π=[1,0,…]`, `a` uniform, `c=0`, `λ_ramp=0`.
* steps 2k–10k: unfreeze `Δ*` & `π`.
* 10k+: unfreeze `(μ,σ)`; enable ramp with small λ (0.1–0.2).

---

## 6) Integration Points (this repo)

* **Add module** under `model/attn_bias.py` → `SpectralBias`.
* **Wire in** at the end of attention logits, before mask. Pass `Q` (projected per head) to produce parameters.
* **Caching for decode**: cache `cos_wD`/`sin_wD` once per max length; update per-step bias row from `Δ*` for the current query.
* **Config**: extend your YAML/args with the toggles/params above.

---

## 7) Optional “gating” (your ramp idea)

* Prefer **additive** gating (bias) over multiplicative scaling: it’s numerically safer (doesn’t break the Q/K variance).
* Your “ReLU + bias” version is fine as a *harder* trough. Implementation:
  [
  g(\Delta)= -\lambda_{\text{gate}}\ \max\Big(0,\ \frac{|\Delta-\Delta^\star|-W}{\tau}\Big).
  ]
  Start with small (\lambda_{\text{gate}}) (≤0.2) to keep gradients alive.
* Empirically, `softplus` works better than raw ReLU (no sharp kink), but keep both behind a flag:

  * `gate_kind = "none"|"softplus"|"relu"`

---

## 8) Evaluation (scripts you’ll want)

**A. PPL train-short/test-long**

* Sliding-window PPL at L = 8k, 32k, 128k, 512k, 1M on held-out.
* Compare `{rope, pi|yarn} (+/- sia) (+/- spectral_bias)`.

**B. Needle/pointer**

* Plant `(ANCHOR_i, VALUE_i)` at controlled offsets; ask later “value for ANCHOR_i?”
* Bin accuracy by log distance.

**C. Anchor-block retrieval**

* Concatenate 200–800 512-token passages; query names an anchor only one passage contains. Measure EM/F1.
* Track attention mass on relevant block vs others.

**D. Diagnostics (log to wandb/tensorboard)**

* Histograms of `Δ*` per head/layer over training.
* `b_q(Δ)` curves for random queries (expect severe asymmetry/multi-peaks).
* Attention-distance histograms per head/layer.
* If SIA on: entropy per log-distance bin (should be flat).

---

## 9) Default Hyperparams (good starting point)

```yaml
# spectral bias
K: 6
M: 2
beta: 0.5
ramp_lambda: 0.2
tau: 64.0
use_slope: true
lambda_omega: 1e-5
lambda_zero_mean: 1e-4
lambda_entropy: 1e-4
L_train: 4096
L_max: 1000000
gate_kind: "softplus"   # try "relu" in ablation
share_across_heads: true
sia_envelope: true
rope_base: "p_rope"     # or "yarn" if you prefer
```

---

## 10) Ablation checklist (short & decisive)

* **Peaks**: M=1 vs 2 vs 3.
* **Bands**: K=4/6/8.
* **Envelope**: SIA on/off.
* **Gate**: none/softplus/relu; λ sweep.
* **High-freq penalty**: on/off.
* **Slope**: on/off.
* **Share vs per-head** MLP.

Success bar (for 24×1024 @ 1.2B tokens):

* ≥3–5% rel. PPL gain over PI/YaRN at 128k+.
* +5–10 pts NIAH at 256k–512k; maintain short-context within 1 pt.

---

## 11) Pseudocode (core build, feel free to ignore, do whatever you believe is optimal)

```python
class SpectralBias(nn.Module):
    def __init__(self, d_head, K=6, M=2, L_train=4096, L_max=1_000_000,
                 beta=0.5, ramp_lambda=0.2, tau=64.0,
                 share_across_heads=True, use_slope=True, gate_kind="softplus"):
        super().__init__()
        self.K, self.M = K, M
        self.beta = beta
        self.ramp_lambda = ramp_lambda
        self.tau = tau
        self.use_slope = use_slope
        self.gate_kind = gate_kind

        w_min = 2*math.pi / L_max
        w_max = 2*math.pi / L_train
        self.register_buffer("omegas", torch.logspace(math.log10(w_min), math.log10(w_max), K))

        hidden = max(128, 2*d_head)
        out = M*(1 + 2 + 1) + (1 if use_slope else 0)  # Δ*, (μ,σ), logit π, optional slope
        self.mlp = nn.Sequential(nn.Linear(d_head, hidden), nn.SiLU(), nn.Linear(hidden, out))

    def forward(self, q, attn_mask=None):
        """
        q: [B, H, L, d_head]  -- use per-head q after W_q projection
        return bias: [B, H, L, L] (lower triangular)
        """
        B, H, L, Dh = q.shape
        K, M = self.K, self.M
        device = q.device

        # outputs
        y = self.mlp(q)  # [B,H,L,out]
        # unpack
        offs = 0
        Delta_raw = y[..., offs:offs+M]; offs += M
        mu = y[..., offs:offs+M]; offs += M
        sigma_raw = y[..., offs:offs+M]; offs += M
        pi_logits = y[..., offs:offs+M]; offs += M
        if self.use_slope:
            c_raw = y[..., offs:offs+1]; offs += 1

        Delta = (Delta_raw.sigmoid() * (L - 1)).unsqueeze(-1)     # [B,H,L,M,1] in tokens (clip to L at train)
        mu = mu.unsqueeze(-1)                                     # [B,H,L,M,1]
        sigma = 0.25 + 1.75 * sigma_raw.sigmoid()                 # [B,H,L,M,1]
        pi = pi_logits.softmax(dim=-1).unsqueeze(-1)              # [B,H,L,M,1]
        if self.use_slope:
            c = 0.01 * torch.tanh(c_raw)                          # small slope
        else:
            c = torch.zeros_like(Delta[..., :1, 0])

        # precompute cos/sin over Δ = 0..L-1
        D = torch.arange(L, device=device).float()                # [L]
        w = self.omegas.to(device)                                # [K]
        cos_wD = torch.cos(w[:, None] * D[None, :])               # [K,L]
        sin_wD = torch.sin(w[:, None] * D[None, :])               # [K,L]

        logw = torch.log(w)[None, None, None, None, :]            # [1,1,1,1,K]
        a_mk = torch.exp(- (logw - mu)**2 / (2*sigma**2))         # [B,H,L,M,K]
        a_mk = a_mk / (a_mk.sum(dim=-1, keepdim=True) + 1e-9)

        # linear phase translation
        cos_wDelta = torch.cos(w[None,None,None,None,:] * Delta)  # [B,H,L,M,K]
        sin_wDelta = torch.sin(w[None,None,None,None,:] * Delta)  # [B,H,L,M,K]

        # coefficients for cos_wD and sin_wD per query
        A = (pi * a_mk * cos_wDelta).sum(dim=3)  # [B,H,L,K]
        Bc = (pi * a_mk * sin_wDelta).sum(dim=3) # [B,H,L,K]

        # assemble b_i(Δ) = Σ_k [ A_k * cos(ω_k Δ) + B_k * sin(ω_k Δ) ]
        # -> einsum over K and Δ to get [B,H,L,LΔ]
        b = torch.einsum('bhlk,kl->bhll', A, cos_wD) + torch.einsum('bhlk,kl->bhll', Bc, sin_wD)  # [B,H,L,L]

        # add slope term
        b = b + c * D[None,None,None,:]

        # optional gate (trough) around main pointer
        # main pointer: argmax π
        main_idx = pi.squeeze(-1).argmax(dim=-1, keepdim=True)          # [B,H,L,1]
        Delta_main = torch.gather(Delta.squeeze(-1), dim=3, index=main_idx).squeeze(-1) # [B,H,L]
        W = (32.0 + 224.0 * torch.sigmoid(mu.squeeze(-1).mean(dim=-1))) # example width ~ 32..256
        # distance matrix Δ = i-j
        I = torch.arange(L, device=device)
        J = I[None, :]                                                   # [L]
        IJ = I[:, None] - J[None, :]                                     # [L,L]
        IJ = IJ.clamp(min=0).float()                                     # causal Δ
        dist = IJ[None, None, :, :] - Delta_main[..., None]              # [B,H,L,L]
        if self.gate_kind != "none" and self.ramp_lambda > 0:
            x = (dist.abs() - W[..., None]) / self.tau
            if self.gate_kind == "relu":
                trough = -self.ramp_lambda * torch.relu(x)
            else:
                trough = -self.ramp_lambda * torch.nn.functional.softplus(x)
            b = b + trough

        # mask strictly future positions (keep −inf for causal outside)
        if attn_mask is not None:
            b = b + attn_mask  # attn_mask should be 0 or -inf lower-triangular

        return self.beta * b
```

Wire this into attention:

```python
logits = (Q @ K.transpose(-2, -1)) * inv_sqrt_d
if cfg.sia_envelope:
    logits = apply_sia_envelope(logits, distances)   # per-Δ affine
if cfg.spectral_bias:
    logits = logits + spectral_bias(Q_per_head, attn_mask=causal_mask)
probs = softmax(logits)
```

---

## 12) “ReLU + bias” thoughts

Totally reasonable as a *harder* trough (additive). Keep it small (λ≤0.2) to avoid overriding content; temperature of softmax plus β already shift things a lot. In practice we’ve seen: `softplus` ramps train a bit smoother, `ReLU` ramps give crisper windows but can slow early convergence. Include both as an ablation.

---





---------
Extra notes:

1. **Name/positioning in code:** implement the spectral pointer as an **additive GRAPE-AP-style bias** (row-wise / endpoint-indexed). Practically, that just means: generate `b_t(Δ)` per query row and add to logits. This matches the telescoping/path-sum view and makes caching/decoding clean.

2. **Prefer “pointer-param” over free per-band phase:** in the spec, keep phase as **linear in ω** via (\phi_k=-\omega_k\Delta^\star). It’s both (a) more stable at 1M and (b) gives you a crisp theorem (shift theorem). If you later want extra flexibility, add a *small residual phase* term as an ablation, but don’t start there.

3. **Nonlinearity gating:** keep the gate **additive** and **smooth** by default (`softplus` or `logsigmoid`) rather than hard ReLU. The reason is practical: hard kinks tend to demand higher-frequency components (ringing) and can destabilize extrapolation; smooth gates give you almost the same suppression but stay “band-limited-friendly.” (Leave `relu` as an ablation flag.)

4. **Two pointers minimum (M=2):** since your goal is “library suppression + a couple boosted shelves,” you want at least two modes early. Keep the entropy regularizer on `π` for the first ~10k steps so it doesn’t collapse to M=1 immediately.

5. **Δ* range during training:** although you conceptually target (L_{\max}=1e6), during training at L=4k/8k it helps to squash (\Delta^\star) to **[0, L_train]** (or a small multiple like 2×L_train) until late in training, then relax toward (L_{\max}). Otherwise the model can waste capacity learning “pointers” it never sees. (This is a small curriculum tweak.)

Everything else in the spec is solid for an initial run.


---

What I meant by “GRAPE-AP-style” (and why it matters) is **how to think about / implement the bias** so it’s clean, causal, and cache-friendly. Here’s the detailed explanation without assuming you’ve read that paper.

## What “GRAPE-AP / path-integral additive bias” means in plain terms

In causal attention, for each query position `t`, you have a row of logits against all past keys `j ≤ t`. A **relative bias** is anything of the form:

[
\ell_{t,j} = \frac{q_t^\top k_j}{\sqrt d} ;+; b_t(\Delta), \quad \Delta = t-j \ge 0
]

Key point: **the bias can depend on the query row (t)** (because it’s query-conditioned), but within that row it’s only a function of distance (\Delta). This is exactly your setup: “for this query, here’s the curve over the whole context window.”

### The “path-integral” view

Instead of defining (b_t(\Delta)) directly, you can define **local increments** (“edge potentials”) along the timeline and sum them up:

* Define a per-step increment (\psi_t(\ell)) for each edge ((\ell-1 \rightarrow \ell)) up to time (t).
* Then define the bias to key (j) as the sum of increments along the path from (j) to (t):

[
b_t(t-j) = \sum_{\ell=j+1}^{t} \psi_t(\ell)
]

This is just “bias = accumulated cost as you walk from j to t.”

### Why this is useful

* It **guarantees causality/streaming friendliness**: you only need increments for steps up to `t`.
* It has a **composition law**: bias from `j→t` is bias from `j→m` + bias from `m→t` (because sums add).
* It gives you a clean way to add “forgetting” / suppression as a monotone accumulated term if you want it later.

### Any (b_t(\Delta)) can be written this way (telescoping trick)

Given any bias shape (b_t(\Delta)) you like, define:

[
\psi_t(\ell) := b_t(t-(\ell-1)) - b_t(t-\ell)
]

Then the sum from (j+1) to (t) telescopes and recovers (b_t(t-j)) (assuming (b_t(0)=0), which you can enforce by subtracting (b_t(0))).

So when I said “GRAPE-AP style,” I meant: **this method is naturally a row-wise, endpoint-indexed additive bias, and it automatically has a nice causal composition structure.**

## How this maps to your spectral pointer bias

Your actual implementation can stay exactly as in the spec: generate a row-wise function (b_t(\Delta)) using the Fourier mixture with linear-phase pointers, then add it to logits.

But now you can justify it in a “proofy” way:

* Your Fourier pointer produces a *desired row shape* (b_t(\Delta)) (multi-peak, asymmetric, huge period).
* Therefore, it corresponds to some path increments (\psi_t) (by the telescoping definition).
* So it inherits the “path-additive / compositional” properties automatically (good story, and good for streaming intuition).

## Code-level guidance: what you should implement

### Training (full attention matrix)

For each layer/head:

1. For each query position `t`, predict pointer params from `q_t`.
2. Build `b_t[Δ]` for `Δ=0..t` (vectorized over `t` ideally).
3. Convert to bias row over keys:

   * key index `j` corresponds to `Δ = t-j`, so `bias[t, j] = b_t[t-j]`.
4. Add `beta*bias` to attention scores before softmax/FlashAttention.

This is exactly what you already spec’d.

### Decoding (one row at a time)

At decode step `t`:

* compute pointer params from `q_t`
* compute bias vector `b_t[0..t]`
* add to the single attention row logits.

You don’t *need* the path/increment form, but it’s there if you want an alternative compute path later.

## Extra (optional) idea enabled by the path view: “forgetting” as monotone suppression (create a seperate file for this variant, everything else can be the same with this tacked on)

With the analogy of “library suppression” framing, you can add a **separate monotone term** that’s literally an accumulated negative cost:

[
b^{\text{forget}}*t(\Delta) = -\sum*{u=1}^{\Delta} \lambda ,\text{softplus}(s_t(u))
]

* This guarantees older tokens never get *more* mass solely from that term (it’s a smooth distance-dependent “tax”).
* Then your **spectral pointer term** provides non-monotone boosts (the bumps) on top:
  [
  b_t(\Delta)=b^{\text{pointer}}_t(\Delta) + b^{\text{forget}}_t(\Delta)
  ]
  This combo is a nice story: *default forget unless the pointer bumps resurrect a region.*

Not necessary as a baseline, however, is a interesting direction to explore.
