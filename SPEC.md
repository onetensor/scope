## Spec: Budget allocation + Gumbel Top-k + negative sampling + “memory reset” training tricks

### 0) Design goals (so you don’t regress into the old failure modes)

**Hard requirements**

* **No tensor shape changes at runtime** in compiled paths (keep the same `[H,QB,max_kv]` / `[H,QB,M*(2p+1)]` layouts; “turn off” slots via **dup center** like you already do).
* **No materializing** anything like `[B,H,L,L]` or “extra K dim trig tensors”.
* **Doc-aware** everywhere: pointer centers and windows must clamp to `doc_start_blk` and causal bound.

**Soft goals**

* Avoid “edge collapse” (mass in first + last bins).
* Avoid π drifting to uniform (`~log(M)`) unless you *want* uniform.
* Keep `kv_unique_blocks` near your target budget, and `ptr_outside_local_frac` controlled (small but nonzero).

---

## 1) Budget allocation framework: stick-breaking + sigmoid

### What to implement

Add a **separate set of logits** for budget allocation (don’t reuse `pi_logits` unless you *want* the spectral mixture and the KV-budget mixture to be the same thing).

For each query-block (and optionally head), produce `u_m` logits for `m=0..M-1`.

Use **stick-breaking** to turn them into per-peak budget fractions. The RAMba/HSA paper uses a sigmoid stick-breaking recurrence (their notation allocates from a remaining “stick”):

* `α_t = sigmoid(π_t) * c_{t-1}`
* `c_t = (1 - sigmoid(π_t)) * c_{t-1}` ([ar5iv][1])

### Your version (practical)

* Let `B_total = spectral_pointer_budget_blocks` (in **blocks**).
* Reserve **mandatory** slots first:

  * local window contributes `local_blocks`
  * each peak contributes at least **1** block (the center), so `M` more
* Remaining budget:
  `B_rem = max(B_total - local_blocks - M, 0)`

Now allocate `B_rem` across peaks via stick-breaking:

* `rem = B_rem`
* for m in 0..M-2:

  * `v_m = sigmoid(u_m / temp_budget)`  (temp optional)
  * `b_m = v_m * rem`
  * `rem = rem - b_m`
* `b_{M-1} = rem`

Then convert floats to integer **extra radius slots**:

* `extra_m = round(b_m)` (or floor + remainder distribution)
* convert extra slots to a half-radius:

  * offsets added per peak = `2*extra_m` (because you’ll add ±1, ±2, … symmetrically)
  * so `radius_m = min(extra_m, pointer_half_blocks_max)`

**Key point:** you can keep **exactly** the same fixed `(2p+1)` offsets tensor and do:
`active = abs(offs_ptr) <= radius_m` (where `radius_m` is `[H,QB,M]` int), and `torch.where(active, ptr, center_dup)` to keep shape fixed.

### Telemetry to add

Per step:

* histogram of `radius_m`
* `sum(extra_m)` distribution
* fraction of peaks with radius 0 (center-only)

---

## 2) Gumbel Top-k (for discrete selection without hard collapse)

Where it belongs: **inside the budget allocator**, *not* inside spectral coefficients (unless you’re willing to take the variance hit).

### Two modes

**Mode A (recommended first): stochastic *exploration* only, no gradients needed**

* When choosing which **offset indices** to keep (or which peaks to give budget to), sample with Gumbel Top-k:

  * `score = base_score + gumbel_noise`
  * pick top-k indices ([ar5iv][2])
    Because BlockMask is non-differentiable anyway, this acts like structured dropout.

**Mode B: Straight-Through Gumbel (only if you really need it)**

* forward: hard top-k mask
* backward: pretend it was softmax at temperature `τ`
  This can help train *the allocator* even if selection is discrete, but it adds instability—use only after Mode A works.

### Implementation constraints (rules to give your agent)

* Generate gumbel noise with **fixed shape** `[H,QB,M,(2p+1)]` or `[H,QB,M]`.
* Do **not** use `nonzero`, `topk` that changes shapes mid-graph; you can use `topk` if its *output shapes are fixed* (they are), but make sure you don’t then slice variable-length lists.
* Keep it behind `SPECTRAL_GUMBEL_TOPK=1` and make it deterministic per-step if you need reproducibility (seed from global step).

---

## 3) Negative sampling (to prevent “edges” and teach useful pointers)

You don’t have supervision for “correct” pointer locations, so negative sampling should be used as an **auxiliary shaping loss**, not a main objective.

### Option 1: Teacher-from-short-context (best signal, cheap if sparse)

Every `N` steps:

* take a **small** batch and **small** subseq length (`L_teacher = 4096` or 8192)
* run a teacher signal:

  * either dense attention (if feasible at that length)
  * or use your existing attention weights before sparsification (if you have them)
* convert teacher attention to **block importance** (sum attention into each KV block)
* positives = top blocks, negatives = random blocks inside doc but outside top, plus “other-doc” blocks
  This aligns with retrieval-style training where negatives are sampled for contrastive selection ([ar5iv][2]).

Loss: sampled softmax / InfoNCE over blocks.

### Option 2: “Edge negatives” (very cheap, directly attacks your failure)

* Define an “edge set” = blocks near doc start and blocks near the farthest allowable distance.
* Sample a few edge blocks as negatives per query-block.
* Penalize allocating budget to edge blocks unless the model *really* prefers them (hinge/logistic margin).
  This directly fights the “first bin + last bin big” pathology.

**Recommendation:** start with Option 2 (fast), then add Option 1 once stable.

---

## 4) Training tricks: memory resets (adapted to your transformer setting)

In the RAMba paper, they reset the **recurrent memory state** every 4K tokens during training and report it improves length generalization (but can hurt in-domain) ([ar5iv][1]).

You don’t have an RNN state, but you can mimic the *effect*:

### “Pointer reset” (transformer-appropriate analogue)

During training, with probability `p_reset` (or on a schedule):

* treat every `reset_len` tokens as a **hard boundary for pointer selection**:

  * for BlockMask building, clamp `doc_start_blk` to `max(doc_start_blk, segment_start_blk)`
  * i.e., pointers cannot reach before the segment boundary, even if the doc is longer
    This forces the allocator to avoid learning degenerate “always jump to ancient anchors” shortcuts, similar to what memory reset is meant to prevent. ([ar5iv][1])

Suggested defaults to try:

* `reset_len = 4096` (or 8192 once stable)
* `p_reset = 0.25` starting at step 0 → decay to 0 by step ~2k–5k

---

## 5) What to run next: ablations that will actually tell you something

You already have A/B/C. Add these **small** targeted 1k–2k step ablations at 64k train / 128k eval:

### Budget allocator ablations

1. **Current baseline**: `radius_mode=pi`, deterministic offsets
2. **Stick-break** allocator, deterministic offsets (no Gumbel)
3. Stick-break + **Gumbel Top-k**, `τ=1.0` constant
4. Stick-break + Gumbel Top-k, **τ schedule** (1.0 → 0.3 over 1k steps)

### Negative sampling ablations

5. Stick-break deterministic + **edge-negative loss** (tiny λ, e.g. 1e-4)
6. Stick-break deterministic + teacher-from-short-context every 200 steps (λ 1e-3)

### Reset trick ablations

7. Stick-break deterministic + pointer reset (`reset_len=4096`, `p_reset=0.25`)
8. Same as 7 but `p_reset` decays to 0 by step 1000

**Readouts to compare (besides loss / needle):**

* `kv_unique_blocks` (mean + p90) should track your intended budget
* `ptr_radius_hist` should not collapse all mass to 0 or max
* `delta_*_hist` should not become “all first + last bin”
* `pi_entropy_mean` should land in a stable band (for M=2, if you want non-single-peak but not uniform, you usually want something like ~0.2–0.55 rather than ~0.69)

---

## RULES

1. **Never introduce runtime-dependent tensor shapes** (no variable-length lists, no slicing by data-dependent k); always keep max shapes and mask with `torch.where` + center-dup padding.
2. **All selection decisions must be representable as elementwise ops** on fixed tensors (`abs(offs)<=radius`, etc.).
3. **Separate concerns:** spectral mixture `pi` is for spectral bias; budget logits are for KV allocation (don’t entangle unless explicitly ablated).
4. **Doc boundaries are sacred:** every pointer center and every selected block must clamp to `doc_start_blk` (and segment start if resets enabled).
5. **Stochasticity must be gated + reproducible:** Gumbel only when enabled, seeded by global step; never let randomness change shapes.
6. **Any new regularizer must log its magnitude** and be scheduled (start small / late), otherwise you’ll “win” by dominating the loss.


[1]: https://ar5iv.org/abs/2504.16795 "[2504.16795] Random Long-Context Access for Mamba via Hardware-aligned Hierarchical Sparse Attention"
[2]: https://ar5iv.org/abs/2410.01651 "[2410.01651] Efficient Long-range Language Modeling with Self-supervised Causal Retrieval"