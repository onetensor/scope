# SCOPE: Spectral COntext Pointer Encoding 

This repo is a research codebase for SCOPE: adding a query-conditioned, band-limited spectral bias to attention logits, plus a pointer-driven block-sparse KV mask so the model can learn long-range selection and still run efficiently at 128k–1M context. The goal is to preserve short-range quality while enabling “library retrieval”: for each query, concentrate attention on a few predicted offsets (and suppress most of the rest) without ever materializing dense `[T,T]` bias matrices. 

Through these experiments, I am trying to validate if a model learn its own block-sparse layout over a large context window, and you can give it a tiny “pointer budget” and still let it find the right few places to look.

## Moving from One Large Attention Matrix to Sliding Window + Learned Jumps

Standard long-context recipes usually look like, dense attention, which is unrealistic at large context sizes, or hand-designed sparse patterns. These methods all do not allow for variance in where attention is allowed to go. The model can only adjust how much it uses those edges.

This project is meant to explore if you allowed each query block to chooses a "small set of key blocks" to attend to, under a strict budget, and learn how to do so by itself.

Essentially:

* The sequence is split into "blocks" (e.g. 128–256 tokens).
* For each query block, we:

  * Always give it a local sliding window (cheap, safe).
  * Give it a tiny number of long-range “pointer” slots.
  * Force those pointers to pick a few key/value blocks under a budget constraint.
* Everything is tracked and logged so we can see:

  * How many unique blocks are actually active.
  * How often pointers leave the local window.
  * Whether they prefer document starts, middles, etc.
    
The intent is for the model to have a sliding window, and also a learned sparse graph of long-range jumps, and that graph itself is a learnable object with its own statistics and regularizers.


## Implementation 

### Blocked sparse layout (sliding window + pointers)

The attention pattern is built in block space:

* `sliding_window_blocks`: how wide the always-on local window is.
* `pointer_mask`:

  * `local_blocks`: local neighborhood always available.
  * `half_blocks`: a band that’s active only for some positions.
  * `global_blocks`: blocks that stay visible everywhere.
* A pointer “budget”:

  * `budget_blocks`, `budget_is_total`, `budget_temp`
    together define *how many* extra blocks a query is allowed to activate and how sharply that budget is enforced.

The result is that each head/layer gets a binary mask over blocks that controls what query block may attend to these key blocks (local + chosen pointers).

### Spectral pointer distribution over blocks

The spectral part is for turning the rough structure into a probability distribution over blocks:

* For each query block, we compute logits over candidate key blocks (in code this passes through the “spectral” scoring path).
* A radius prior controls how likely we are to point:

  * Near the query (local-ish radii),
  * Toward special locations like document start,
  * Or far into the sequence.
* We then optionally can run Gumbel top-k for discrete sampled block choices while still being differentiable enough to train.

Stats such as these are tracked to see how exactly the model decides to focus its attention:

* `ptr_radius_hist`, `ptr_radius0_frac`: how often pointers pick radius 0 vs farther jumps.
* `ptr_center_docstart_frac`: how often we point at doc start.
* `ptr_outside_local_frac`: how often the model uses true long-range jumps instead of staying local.
* `kv_unique_blocks`: how many distinct K/V blocks are actually active.

### Budget + regularization: making sparsity an actual pressure

Just giving a pointer mechanism is not enough, as by default, the model is likely to just attend to everything or collapse to its sliding window and choose nothing.

This repo includes several regularizers that try to make the layout behave:

* **Budget penalties**
  Encourage the model to stay within its `budget_blocks` per query block.
* **Entropy regularization on the pointer distribution π**
  * Can push pointers to be more peaked (more decisive) or more spread out, depending on what you want.
* **Delta / radius regularizers**

  * Histograms over offset choices for tokens and blocks
  * Extra terms like `reg.delta_edge` and `reg.zero_mean` to control which offsets are easy for the model to do (to prevent collapse).



### Teacher / student path (experimental)

There also is an experimental teacher mechanism:

* The student model runs with the constrained pointer mask.
* A teacher (a heavier / less constrained attention) which is supposed to generate “good” pointer distributions or targets.

Right now this path is implemented but still being debugged. The goal is to see how much signal we can distill from a teacher into the pointer budget.


### Needle-in-a-Haystack evaluation harness

Implemented a basic NIAH style eval, where sequences have an anchor and value hidden at various distances. The aim here is to see whether this pointer based method can prevent performance degradation on further needles due to selection and sparsity.

### Logging

These are the main metrics which are being logged:

* Core training metrics:

  * `step`, `train_loss`, `sliding_window_blocks`.
* Pointer mask introspection:

  * `pointer_mask.enabled`, `local_blocks`, `half_blocks`, `global_blocks`.
  * `pointer_reset` knobs, if you’re running any reset schedule.
* Coverage and use:

  * `delta_star_max_active`: a proxy for how many keys were actually “live”.
  * `kv_unique_blocks.mean/p50/p90`: do we really use lots of distinct blocks?
  * Fractions of pointers outside local windows, into doc start, etc.
* Regularization breakdown:

  * `reg.omega`, `reg.entropy`, `reg.delta_edge`, `reg.zero_mean`, `reg.total`.


## 3. How This Fits into the Larger Story

Conceptually, this project is in a similar category as:

* Sliding-window / dilated / block sparse attention
* Global tokens, memory tokens, retrieval, etc.

But the emphasis here is slightly different:

1. **Layout as a dynamic, learned object**
   The model decides which blocks to connect, under strict budget and regularization.

2. **Block-space reasoning instead of token-space micromanagement**
   All decisions are made over blocks, which keeps the control problem small and easy to monitor.

3. **Heavy instrumentation > pure optimization**
   The point is not just to get a slightly better validation loss, but:

   * See how many blocks are actually active.
   * See where pointers go.
   * See how all of that changes across ablations.

## 4. Current Status & Directions

Right now, the repo has:

*  A working block-pointer attention mask integrated into a transformer training loop.
*  Configurable local + half + global blocks, plus pointer budgets and radius priors.
*  Optional Gumbel top-k selection for discrete pointers.
*  Detailed logging via `SCOPE_STATS` and a basic NIAH eval.
*  An experimental teacher / NCE path (still being debugged).

I am planning to work on:
* Comparing different radius priors and budget regimes.
* Checking how pointer usage evolves with scale and dataset.
* Probing whether teacher-guided pointers actually help with extreme long-range recall vs. pure self-supervision.
* Visualizing pointer graphs over real documents (not just synthetic needles).
