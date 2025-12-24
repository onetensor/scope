# Agent Notes (SCOPE)

This workspace contains a nested training repo in `modded-nanogpt/` (it also has its own `.git/`).
Unless a task explicitly says otherwise, treat `modded-nanogpt/` as the primary codebase for model
training + experiments.

## Project Goal

**SCOPE (Spectral COntext Pointer Encoding)**: preserve short-range LM quality while enabling
**selective long-context retrieval** at inference (128k–1M context).

Core ideas:

- A **query-conditioned, band-limited spectral bias** over relative distance `Δ = i - j` that
  reshapes attention without materializing dense `[T,T]` bias tensors.
- A **pointer-driven block-sparse KV mask** so far keys enter attention compute during training
  (union of local band + pointer windows + optional global anchors).

`modded-nanogpt/SPEC.md` is the source-of-truth spec.

## Source Of Truth / Do-Not-Edit

- `modded-nanogpt/SPEC.md`: full project specification.
- `modded-nanogpt/baseline.py`: frozen snapshot for ablations (do not modify).

## Repo Map (key files)

- `modded-nanogpt/scoped_medium.py`: end-to-end training script (model + data + DDP + training loop).
- `modded-nanogpt/model/attn_bias.py`: `SpectralBias` implementation + pointer `BlockMask` builder + telemetry buffers.
- `modded-nanogpt/run_ablations.sh`: A/B/C ablation harness.
- `modded-nanogpt/Dockerfile`: container build for Vast.ai.
- `modded-nanogpt/.github/workflows/ghcr.yml`: GHCR build/push workflow.

## SCOPE Controls (env vars)

Primary toggles used by `run_ablations.sh`:

- `SPECTRAL_BIAS=0|1`
- `SPECTRAL_IMPL=qk_aug|score_mod`
- `SPECTRAL_USE_POINTER_MASK=0|1`
- `SPECTRAL_POINTER_SCHEDULE=0|1`
- `SPECTRAL_POINTER_LOCAL_BLOCKS=<int>`
- `SPECTRAL_POINTER_HALF_BLOCKS=<int>`
- `SPECTRAL_POINTER_GLOBAL_BLOCKS=<int>`

### Validation pointer-mask state

Validation uses the same pointer-mask state as training at the current step (scheduled or fixed)
unless forced:

- `SPECTRAL_POINTER_VAL_FORCE=1`
- `SPECTRAL_POINTER_VAL_HALF_BLOCKS=<int>` (`-1` -> use configured max)

## Telemetry (rank0)

`modded-nanogpt/scoped_medium.py` emits JSON:

- `SCOPE_BINS`: bin edges at startup.
- `SCOPE_STATS`: every `SCOPE_LOG_EVERY` steps (KV unique-block stats, pointer behavior proxies,
  `pi` entropy, and regularizer magnitudes).

## Current Blocker / Debug Context

Some PyTorch nightlies exhibit an **Inductor lowering crash** compiling `flex_attention_backward`
when using nontrivial score modifications (assertion like `len(idx) == len(output_size)` during
kernel template generation). This often appears in ablation **B (bias-only)**.

Typical mitigations:

- Pin PyTorch to a different (newer/older) nightly where FlexAttention lowering works, or
- Change the spectral integration to avoid the problematic `score_mod` compilation path.

## Development Conventions

- Avoid dense `[T,T]` allocations anywhere in the long-context path.
- Keep `BlockMask` shapes as static as possible (compile stability; avoid variable-length KV lists).
- If changing distributed logic, ensure gradients are reduced consistently across ranks.
