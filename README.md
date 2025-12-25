# SCOPE: Spectral COntext Pointer Encoding (Long-Context Selective Attention)

This repo is a research codebase for **SCOPE**: adding a *query-conditioned, band-limited spectral bias* to attention logits, plus a *pointer-driven block-sparse KV mask* so the model can learn long-range selection and still run efficiently at **128k–1M** context.

The goal is to preserve short-range quality while enabling “library retrieval”: for each query, concentrate attention on a few predicted offsets (and suppress most of the rest) without ever materializing dense `[T,T]` bias matrices.

## Repo Map

- `scoped_medium.py`: primary training script (model + data + optimizer + training loop).
- `baseline.py`: frozen snapshot of the original baseline (kept unmodified for ablations).
- `SPEC.md`: the full SCOPE project specification.
- `model/attn_bias.py`: `SpectralBias` module + pointer-driven `BlockMask` builder + SCOPE telemetry buffers.
- `run_ablations.sh`: runs the 3-run ablation set (A/B/C).
- `Dockerfile`: CUDA container used on Vast.ai; built/pushed via `.github/workflows/ghcr.yml`.

## What We Implemented So Far

- **Spectral bias via FlexAttention** without dense bias matrices.
- **Pointer-driven block-sparse KV selection** (union of):
  - local sliding window (stability)
  - pointer windows around predicted `Δ*` (learn long-range access)
  - optional global anchor blocks
- **Fixed-shape KV lists** using an “active radius via duplication” trick to avoid compile thrash.
- **Rank-0 telemetry** (`SCOPE_BINS` + `SCOPE_STATS`) to measure KV budget, pointer behavior, and regularizer magnitudes.

## Running

### Data

This code expects pre-tokenized `.bin` shards (same format as `data/cached_fineweb*.py` produces). Token id `50256` (`<|endoftext|>`) is treated as the document delimiter for doc-aware masking.

### Single run

From the repo root:

```bash
torchrun --standalone --nproc_per_node=8 scoped_medium.py
```

Useful overrides:

- `NUM_ITERATIONS=2000`
- `TRAIN_SEQ_LEN=65536` / `VAL_SEQ_LEN=65536`
- `VAL_LOSS_EVERY=0` (end-only) or a small interval
- `VAL_TOKENS=10485760`

### 3-run ablations (recommended)

Runs:

- **A**: baseline (spectral disabled)
- **B**: bias-only (spectral enabled, pointer mask OFF)
- **C**: bias + pointer mask (spectral enabled, pointer mask ON + schedule)

```bash
./run_ablations.sh
```

Controls:

- `NUM_ITERATIONS`, `VAL_TOKENS`, `VAL_LOSS_EVERY`, `SEED`
- `SCOPE_LOG_EVERY=200` (rank0-only SCOPE stats cadence)

## Needle-in-a-Haystack Eval (synthetic)

Emits a rank0 JSON blob with `tag="NEEDLE_EVAL"` (loss/ppl/token-acc/EM by distance):

- `NEEDLE_EVAL=1`
- `NEEDLE_SEQ_LEN=65536`
- `NEEDLE_DISTANCES=4096,8192,16384,32768`
- `NEEDLE_SAMPLES_PER_DISTANCE=8`
- `NEEDLE_ANCHOR_LEN=4`
- `NEEDLE_VALUE_LEN=8`

Evaluate a saved checkpoint without training:

```bash
LOAD_CHECKPOINT=/path/to/state_step001000.pt NUM_ITERATIONS=0 VAL_LOSS_EVERY=0 NEEDLE_EVAL=1 torchrun --standalone --nproc_per_node=8 scoped_medium.py
```

## SCOPE Controls (env vars)

- `SPECTRAL_BIAS=0|1`
- `SPECTRAL_USE_POINTER_MASK=0|1`
- `SPECTRAL_POINTER_SCHEDULE=0|1`
- `SPECTRAL_POINTER_LOCAL_BLOCKS=16`
- `SPECTRAL_POINTER_HALF_BLOCKS=4`
- `SPECTRAL_POINTER_GLOBAL_BLOCKS=0|1|2`
- `SPECTRAL_POINTER_GLOBAL_BLOCKS_WARMUP=2`
- `SPECTRAL_POINTER_GLOBAL_BLOCKS_WARMUP_STEPS=300`

### Validation pointer-mask state

By default, validation uses the same pointer-mask state as training at the current step (scheduled or fixed).

You can force pointer masking during validation:

- `SPECTRAL_POINTER_VAL_FORCE=1`
- `SPECTRAL_POINTER_VAL_HALF_BLOCKS=<N>` (or `-1` to use the configured max)

## Compiler / Debug Controls

- `TORCH_COMPILE=0|1`
- `TORCH_COMPILE_MODE=default|reduce-overhead|max-autotune`
- `TORCHDYNAMO_VERBOSE=0|1`
- `TORCHDYNAMO_SUPPRESS_ERRORS=0|1`
- `TORCH_COMPILE_FALLBACK_TO_EAGER=0|1`
- `FLEXATTN_DYNAMO_DISABLE=0|1` (graph-break around FlexAttention if needed)
- `FLEXATTN_COMPILE=0|1` (compile FlexAttention standalone when graph-broken; keeps fused kernel)
- `LOG_ALL_RANKS=0|1` (write one logfile per rank)

## Docker / Vast.ai

The GH Actions workflow in `.github/workflows/ghcr.yml` builds and pushes:

- `ghcr.io/<org>/<repo>:latest`
- `ghcr.io/<org>/<repo>:<sha>`

On Vast.ai, run the container and launch `torchrun` inside it.

## Known Issues

- **FlexAttention + Inductor**: some PyTorch nightlies fail compiling `flex_attention_backward` when using nontrivial score modifications (assertion in Inductor lowering). If you hit an error like `assert len(idx) == len(output_size)` during `flex_attention_backward`, it’s likely an upstream compiler issue. Pinning to a different nightly (or newer build) is the usual fix.

