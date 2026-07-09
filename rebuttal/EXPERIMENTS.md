# EPO Rebuttal — Experiment Plan (runnable on another server)

Companion to `REBUTTAL.md`. Goal: produce the evidence the three reviewers asked for, in priority
order. Each experiment lists **why / reviewer-theme / commands / what to log / expected outcome / cost**.

All commands assume the repo at `epo_code/` and the launcher `examples/general_running_server.sh`.
Set these first:

```bash
cd epo_code
export SCI_MODEL="/path/to/Qwen2.5-7B-Instruct"   # ScienceWorld backbone
export ALF_MODEL="/path/to/Qwen2.5-3B-Instruct"   # ALFWorld backbone
export N_GPUS=8
```

---

## 0. Prerequisite: launcher fix (ALREADY APPLIED — verify before running)

`examples/general_running_server.sh` **PPO branch** now forwards the EPO flags
(`entropy_smooth`, `entropy_smooth_mask_mode`, `entropy_smooth_min_ratio`,
`entropy_smooth_max_ratio`, `entropy_smooth_out_range_penalty`, `enable_smooth_weights`,
`entropy_distribution_output_file`, `window_size`, `adaptive_start_epoch`). Before the fix, a
`PPO+EPO` run launched through this script silently ran as **plain PPO** (the corridor was never
applied). Sanity-check the fix:

```bash
grep -n "entropy_smooth_min_ratio" examples/general_running_server.sh   # expect a hit in BOTH branches
```

**Two known core-code issues — decide policy BEFORE running (we did NOT change them, to preserve
reproducibility of the paper's numbers):**
1. `verl/workers/actor/dp_actor.py:772` — `self.config.get("max_step", 50)` misreads the key
   `max_steps`; the smoothing phase-gate threshold is therefore always `25`. For ScienceWorld
   (`max_steps=30`) smoothing only applies to turns ≥25. If you want smoothing to apply from mid-episode
   as described in the paper, change `"max_step"` → `"max_steps"`. **Keep as-is to match the paper.**
2. `dp_actor.py:793` — `max_epochs = 150` is hardcoded for the `enable_smooth_weights` schedule;
   ScienceWorld runs `K=125`. Harmless unless you rely on the exact schedule tail.

> Decision to make once: for the ablations below, either (A) keep both issues as-is so results are
> directly comparable to the submitted numbers (recommended for the rebuttal), or (B) fix both and
> re-run **all** EPO cells (baseline included) so the comparison stays internally consistent. Do **not**
> mix. Recommendation: **(A)**.

---

## Experiment 1 — κ_l>0 corridor ablation ★ TOP PRIORITY

**Why / Theme A (3fUM-W1, uSAW-W1).** The #1 concern from two reviewers: reported runs use κ_l=0, so
the corridor floor is inactive. This experiment shows what the two-sided corridor does and whether the
floor ever binds.

**Design.** Vary only `--entropy_smooth_min_ratio` ∈ **{0, 0.5, 0.8}** (0 = current paper setting; 0.5,
0.8 = active floor). Keep everything else at the paper config. Run on **ScienceWorld** (where EPO
matters most) for **PPO+EPO and GRPO+EPO**, ≥2 seeds. Add ALFWorld if budget allows.

**Commands (ScienceWorld, PPO+EPO):**
```bash
for KL in 0 0.5 0.8; do
for SEED in 0 1; do
bash examples/general_running_server.sh \
  --environment sciworld --rl_algorithm ppo --seed $SEED \
  --lr 3e-6 --lr_warmup_steps_ratio 0.1 --min_lr_ratio 0.2 --warmup_style cosine \
  --total_epochs 125 \
  --entropy_smooth True --enable_smooth_weights True --entropy_smooth_mask_mode token \
  --entropy_smooth_min_ratio $KL --entropy_smooth_max_ratio 2.0 \
  --entropy_smooth_out_range_penalty 0.05 --entropy_coeff 0.001 \
  --ppo_mini_batch_size 64 --ppo_micro_batch_size_per_gpu 8 --log_prob_micro_batch_size_per_gpu 8 \
  --model_path "$SCI_MODEL" --model_load_method local --n_gpus $N_GPUS
done; done
```

**Commands (ScienceWorld, GRPO+EPO):** same but `--rl_algorithm grpo --lr 5e-6 --group_size 8
--ppo_mini_batch_size 128 --ppo_micro_batch_size_per_gpu 16` (no critic).

**Commands (ALFWorld, optional):** `--environment alfworld --total_epochs 150 --model_path "$ALF_MODEL"
--entropy_smooth_out_range_penalty 0.1` and ALFWorld batch sizes (mini 256, micro 32 for PPO).

**Log / report.**
- Final IID/OOD Succ.\* and $\overline{\text{Succ.}}$ per κ_l (a small table mirroring Table 1).
- The **entropy-vs-floor overlay** (see Exp 2c): does measured entropy ever fall below κ_l·H̄?

**Expected outcome (report whichever holds).** In ScienceWorld/ALFWorld entropy stays above the floor,
so κ_l∈{0,0.5} give ≈equal results → **κ_l=0 was empirically harmless here** (supports Theme A). If
κ_l=0.8 slightly helps/hurts, report it. Combined with Exp 5 (collapse regime), this fully answers the
concern.

**Cost.** 3 κ_l × 2 seeds × 2 algos = 12 ScienceWorld runs (~16–23 h each). Minimal decisive subset:
κ_l∈{0,0.8}, 1 seed, PPO+EPO only = 2 runs.

---

## Experiment 2 — Analyses from existing W&B logs (NO new training) ★ do first, ~free

**Why / Themes C, D (uSAW-W2/W3/#5).** Turns "correlation" into quantified evidence and supports the
importance-ratio-variance claim, using logs you already have (W&B project
`verl_agent_{sciworld,alfworld}_{ppo,grpo}`).

Write one script (`analysis/wandb_analysis.py`) using the `wandb` API. Logged keys to use (verify
names in your dashboard): per-turn entropy `step_entropy_{i}`, mean entropy `actor/entropy` (or
`entropy`), clip fraction `actor/pg_clipfrac`, reward `critic/score/mean`, validation success rate keys.

**2a. Oscillation ↔ performance correlation (Theme C).** For every run, compute oscillation =
`std(diff(entropy_over_steps))` (or std of `step_entropy_*` across turns); scatter vs final success;
report Pearson/Spearman r. Expected: strong **negative** correlation.

**2b. Importance-ratio variance / clip-fraction (Theme D).** Plot `actor/pg_clipfrac` (and, if logged,
ratio variance) over training for baseline vs +EPO. Expected: EPO lower/more stable → supports the
variance-control claim empirically (we drop the formal claim per Theme D).

**2c. Entropy-vs-corridor overlay (Theme A support).** Plot mean entropy trajectory with the band
[κ_l·H̄, κ_r·H̄] (H̄ = running mean). Show whether the κ_l=0 floor would ever bind.

**Cost.** Hours of scripting, no GPU. **Run this first** — it may already settle Themes C/D.

---

## Experiment 3 — Direct causal intervention (mid-run EPO toggle)

**Why / Theme C (uSAW-W2).** The strongest causal evidence: toggle smoothing mid-training and show
entropy stabilizes + reward recovers exactly at the toggle.

**Option A (no code change):** run plain PPO with checkpointing (`--` set `trainer.save_freq` to e.g.
10), then **resume** the diverging checkpoint (≈step 40) with `--entropy_smooth True`. Compare the
continued-plain-PPO vs resumed-with-EPO curves from the same checkpoint.

**Option B (tiny code hook, cleaner):** add an epoch-gated enable in `dp_actor.py`. Around line 770,
replace `if self.config.entropy_smooth:` with a gate that also checks a new config
`entropy_smooth_start_epoch` (default 0):
```python
_start = self.config.get("entropy_smooth_start_epoch", 0)
if self.config.entropy_smooth and trainer_epoch >= _start:
```
Then run one job with `entropy_smooth_start_epoch=40` (EPO turns on at step 40) and a control with a
huge value (never on). Add `--entropy_smooth_start_epoch` to the launcher like the other flags. This is
additive and safe (default 0 = current behavior).

**Also (confounder control, uSAW-W3):** run the plain PPO baseline at 2–3 learning rates (e.g. 3e-6,
5e-6, 1e-5) to show instability persists across LR (not an LR artifact); EPO stable across the same.

**Log / report.** Entropy + reward vs step, with the toggle marked. Expected: sharp stabilization and
reward recovery at the toggle.

**Cost.** 2–4 runs.

---

## Experiment 4 — Hyperparameter sensitivity

**Why / Theme E (91dS-Q6, uSAW-#4).** One-knob-at-a-time sweep around the default, ScienceWorld
GRPO+EPO (cheapest stable setting), 1 seed.

**Runnable knobs (exposed in code):**
- λ (`--entropy_coeff`) ∈ {0.0005, 0.001, 0.002}
- α / penalty (`--entropy_smooth_out_range_penalty`) ∈ {0.05, 0.1, 0.2}
- κ_r (`--entropy_smooth_max_ratio`) ∈ {1.5, 2.0, 2.5}
- κ_l (`--entropy_smooth_min_ratio`) — covered by Exp 1
- smoothing weight (`--entropy_smooth_coeff`) ∈ {0.5, 1.0, 2.0}

**Caveats to state in the rebuttal (honesty):**
- **Entropy-window size** (`--window_size`) is currently **not used** in the reference averaging (the
  code uses a *cumulative* mean over all steps). Either (i) report the cumulative design as intentional
  (adapts fast early ~1/k, stabilizes later), or (ii) implement a true sliding window before sweeping it.
- **γ (schedule decay)** in the paper's β_k formula is **not wired** to a tunable — the code schedule
  (`_calculate_epoch_based_entropy_weight`) hardcodes decay rates. To sweep γ, expose it first, or
  sweep `--entropy_smooth_coeff` (the penalty weight) as the practical proxy and say so.

**Command template (vary ONE knob):**
```bash
bash examples/general_running_server.sh --environment sciworld --rl_algorithm grpo --seed 0 \
  --lr 5e-6 --total_epochs 125 --group_size 8 \
  --entropy_smooth True --enable_smooth_weights True --entropy_smooth_mask_mode token \
  --entropy_smooth_min_ratio 0 --entropy_smooth_max_ratio 2.0 \
  --entropy_smooth_out_range_penalty 0.1 --entropy_coeff 0.001 --entropy_smooth_coeff 1.0 \
  --ppo_mini_batch_size 128 --ppo_micro_batch_size_per_gpu 16 \
  --model_path "$SCI_MODEL" --model_load_method local --n_gpus $N_GPUS
```

**Report.** A small sensitivity table/figure per knob (final IID/OOD success). Expected: broad plateau
of good values → method not brittle.

**Cost.** ~4 values × 3 knobs = 12 short-ish GRPO runs; subset acceptable.

---

## Experiment 5 — Collapse-regime demo for the floor (small, optional but high-value)

**Why / Theme A closure.** Directly demonstrates the corridor **floor** matters when entropy *does*
collapse — validating the theory's anti-collapse claim in the regime where it applies.

**Design.** Induce collapse (e.g. higher LR like 1e-5, or κ_r large / λ small) so baseline/EPO-with-κ_l=0
shows entropy → 0; then show **κ_l=0.8** arrests the collapse and preserves reward. ScienceWorld or
ALFWorld GRPO, 1–2 seeds.

**Report.** Entropy + success curves for {baseline, EPO κ_l=0, EPO κ_l=0.8} in the collapse regime.
Expected: κ_l>0 prevents collapse where κ_l=0 cannot → the floor is a genuine safety rail.

**Cost.** 2–4 runs.

---

## Suggested minimal set if compute/time is tight
1. **Exp 2** (free, from logs) — likely settles Themes C & D.
2. **Exp 1** with κ_l∈{0,0.8}, PPO+EPO, ScienceWorld, 1 seed (2 runs) — settles Theme A.
3. **Exp 3 Option A/B**, one intervention run — strongest causal evidence.

## Mapping to reviewers
| Experiment | Theme | Reviewers |
|---|---|---|
| 1 κ_l ablation | A | 3fUM-W1, uSAW-W1 |
| 2a oscillation↔perf | C | uSAW-W2/W3 |
| 2b ratio/clip variance | D | uSAW-#5, 3fUM |
| 2c entropy-vs-corridor | A | 3fUM-W1, uSAW-W1 |
| 3 intervention + LR control | C | uSAW-W2/W3 |
| 4 sensitivity | E | 91dS-Q6, uSAW-#4 |
| 5 collapse regime | A | 3fUM-W1, uSAW-W1 |
