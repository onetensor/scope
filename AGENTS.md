# Agent Notes (SCOPE)

This repo is the SCOPE research codebase (flattened; no nested repos).

## Project Goal

**SCOPE (Spectral COntext Pointer Encoding)**: preserve short-range LM quality while enabling
**selective long-context retrieval** at inference (128k–1M context).

Core ideas:

- A **query-conditioned, band-limited spectral bias** over relative distance `Δ = i - j` that
  reshapes attention without materializing dense `[T,T]` bias tensors.
- A **pointer-driven block-sparse KV mask** so far keys enter attention compute during training
  (union of local band + pointer windows + optional global anchors).

`SPEC.md` is the source-of-truth spec.

## Source Of Truth / Do-Not-Edit

- `SPEC.md`: full project specification.
- `baseline.py`: frozen snapshot for ablations (do not modify).

## Repo Map (key files)

- `scoped_medium.py`: end-to-end training script (model + data + DDP + training loop).
- `model/attn_bias.py`: `SpectralBias` implementation + pointer `BlockMask` builder + telemetry buffers.
- `run_ablations.sh`: A/B/C ablation harness.
- `Dockerfile`: container build for Vast.ai.
- `.github/workflows/ghcr.yml`: GHCR build/push workflow.

## SCOPE Controls (env vars)

Primary toggles used by `run_ablations.sh`:

- `SPECTRAL_BIAS=0|1`
- `SPECTRAL_IMPL=qk_aug|score_mod`
- `SPECTRAL_QK_AUG_ALIGN=16`
- `SPECTRAL_USE_POINTER_MASK=0|1`
- `SPECTRAL_POINTER_SCHEDULE=0|1`
- `SPECTRAL_POINTER_LOCAL_BLOCKS=<int>`
- `SPECTRAL_POINTER_HALF_BLOCKS=<int>`
- `SPECTRAL_POINTER_BUDGET_BLOCKS=<int>` (π allocates this extra block budget across peaks)
- `SPECTRAL_POINTER_RADIUS_MODE=fixed|pi`
- `SPECTRAL_POINTER_GLOBAL_BLOCKS=<int>`
- `SPECTRAL_POINTER_GLOBAL_BLOCKS_WARMUP=<int>`
- `SPECTRAL_POINTER_GLOBAL_BLOCKS_WARMUP_STEPS=<int>`
- `SPECTRAL_DELTA_STAR_MAX_SCHEDULE=0|1` (active cap on Δ*; reduces saturation to 0/max)
- `SPECTRAL_DELTA_STAR_MAX_SCHEDULE_MIN=<int>` (`-1` -> `SPECTRAL_L_TRAIN-1` default)
- `SPECTRAL_DELTA_STAR_MAX_SCHEDULE_MAX=<int>` (`-1` -> `max(TRAIN_SEQ_LEN,VAL_SEQ_LEN)-1` default)
- `SPECTRAL_DELTA_STAR_MAX_SCHEDULE_START_STEP=<int>`
- `SPECTRAL_DELTA_STAR_MAX_SCHEDULE_STEPS=<int>`
- `SPECTRAL_PI_ENTROPY_FLOOR_FRAC=<float>` (only penalize entropy below this fraction of `log(M)`)
- `SPECTRAL_LAMBDA_DELTA_EDGE=<float>` (Δ* edge-avoidance reg strength)
- `SPECTRAL_DELTA_EDGE_EPS=<float>` (normalized edge band, e.g. `0.05`)
- `SPECTRAL_DELTA_EDGE_SCHEDULE_START_STEP=<int>`
- `SPECTRAL_DELTA_EDGE_SCHEDULE_STEPS=<int>`

## Checkpointing

- `SAVE_CHECKPOINT=0|1` (default `1`) saves a `state_step*.pt` at the end of the run (rank0 only)
- `SAVE_OPTIMIZERS=0|1` (default `0`) additionally includes optimizer states (large; usually unnecessary)

## FlexAttention Controls (env vars)

- `FLEXATTN_DYNAMO_DISABLE=0|1` (graph-break around FlexAttention; use only as a workaround for compiler issues)
- `FLEXATTN_COMPILE=0|1` (when graph-broken, compile FlexAttention standalone to keep fused kernels)

### Validation pointer-mask state

Validation uses the same pointer-mask state as training at the current step (scheduled or fixed)
unless forced:

- `SPECTRAL_POINTER_VAL_FORCE=1`
- `SPECTRAL_POINTER_VAL_HALF_BLOCKS=<int>` (`-1` -> use configured max)

## Telemetry (rank0)

`scoped_medium.py` emits JSON:

- `SCOPE_BINS`: bin edges at startup.
- `SCOPE_STATS`: every `SCOPE_LOG_EVERY` steps (KV unique-block stats + pointer behavior proxies like
  `ptr_center_block0_frac_excl` / `ptr_center_docstart_frac`, plus `delta_star_max_active`, `pi` entropy, and regularizers).
- `NEEDLE_EVAL`: optional synthetic NIAH retrieval eval (by distance), gated by `NEEDLE_EVAL=1`.

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
