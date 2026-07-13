# Rebuttal — Experimental Results Addendum (Submission 13068, EPO)

Companion to `REBUTTAL.md`. This file plugs the **actual measured numbers** from the rebuttal
experiments into Themes A / C / D so the OpenReview replies cite concrete evidence instead of
"we will report." All numbers are reproducible from the released W&B logs + the new runs on branch
`rebuttal/epo-experiments` (see `rebuttal/results/RESULTS.md`, `analysis/wandb_analysis.py`).

Status: **Exp 1, 2, 3, 4 all complete.** (Exp 3 `plainppo_lr1e5` reached 106/125 when the shared job
hit its 24h wall — treated as final; it had already collapsed to reward ≈ 0.) W&B run URLs for every
run are in `rebuttal/results/WANDB_INDEX.md`.

---

## Theme A — "κ_l=0 disables the corridor lower bound" (3fUM-W1, uSAW-W1)

Two independent pieces of new evidence show the κ_l=0 floor was **empirically harmless** in these
environments, exactly as our diagnosis (over-exploration, controlled by the *upper* cap) predicts.

**A1. The lower bound never binds (Exp 2c, from released logs).** We overlay the measured mean-entropy
trajectory with the corridor band [κ_l·H̄, κ_r·H̄] and count steps where entropy falls below the
floor κ_l·H̄. With the reported κ_l=0 the floor is at 0, and measured entropy stays strictly above it
for **every** run:

| Project | steps entropy < floor |
|---|---|
| verl_agent_sciworld_ppo  | 0 / 9 |
| verl_agent_sciworld_grpo | 0 / 123 |
| verl_agent_alfworld_ppo  | 0 / 150 |
| verl_agent_alfworld_grpo | 0 / 150 |

So the left branch `[κ_l·H̄ − H]₊` is identically zero *not by choice of κ_l but because entropy never
approaches the floor* — the anti-collapse work is done by the entropy term `L^H`, and the active
mechanism against our diagnosed failure (entropy growth/oscillation) is the **upper cap κ_r**.

**A2. New κ_l>0 ablation (Exp 1).** We ran the two-sided corridor with κ_l ∈ {0, 0.5, 0.8}
(κ_r=2.0) for **PPO+EPO and GRPO+EPO on ScienceWorld, 2 seeds each** (12 runs, Qwen2.5-7B-Instruct,
125 epochs). Converged success (mean of last 3 validations; IID = val_l0, OOD = val_l1), seed-averaged:

| Algo | κ_l=0 (IID/OOD) | κ_l=0.5 | κ_l=0.8 |
|---|---|---|---|
| PPO+EPO  | **1.00 / 0.95** | 0.64 / 0.60 | 0.62 / 0.59 |
| GRPO+EPO | 0.52 / 0.52 | 0.25 / 0.19 | 0.51 / 0.54 |

**Finding:** activating the floor (κ_l>0) gives **no consistent improvement** over κ_l=0 — for PPO,
κ_l=0 is best; for GRPO, κ_l=0 ≈ κ_l=0.8 and κ_l=0.5 is worst. This confirms κ_l=0 was harmless here
and that the corridor's benefit in these envs comes from the active upper bound + `L^H`, consistent
with our revised theory (stability needs |H−H̄| bounded, which the upper cap already provides; the
floor is a safety rail for collapse-prone regimes — see Exp 5, future).

**Honesty caveat (stated in the reply):** with 2 seeds on a 16-sample validation set, several cells
are bimodal (one seed converges ~1.0, the other ~0.2–0.3), so κ_l differences are within seed noise;
the defensible claim is "κ_l>0 yields no consistent benefit," not a fine ranking.

---

## Theme C — "Is entropy oscillation cause or symptom?" (uSAW-W2/W3)

**C1. Cross-run correlation (Exp 2a, released logs).** For every run we correlate entropy-oscillation
magnitude = std(ΔH over training) with final success:

| Project | Pearson r (p) | Spearman ρ | n |
|---|---|---|---|
| verl_agent_sciworld_ppo  | **−0.509 (p=1.1e-3)** | −0.559 | 38 |
| verl_agent_sciworld_grpo | +0.156 (p=0.36) | 0.120 | 37 |
| verl_agent_alfworld_ppo  | +0.009 (p=0.97) | 0.087 | 16 |
| verl_agent_alfworld_grpo | +0.015 (p=0.91) | 0.317 | 65 |

Strong **negative** correlation in **ScienceWorld PPO** — the sparsest-reward, highest-variance
setting where our cascade-failure thesis predicts EPO matters most (Theme F). Weak/absent in GRPO and
ALFWorld, consistent with milder oscillation there. We report this honestly (not overclaiming a
universal law) and soften "primary cause" → "primary, controllable driver."

**C2. Direct causal intervention (Exp 3).** PPO+EPO with smoothing gated ON at epoch 40
(`entropy_smooth_start_epoch=40`) vs an identical never-on control. Result: post-toggle entropy
oscillation std(ΔH) drops **0.055 → 0.019**, and reward jumps **1.70 → 9.50** (IID 0.27 → 0.98) — i.e.
turning EPO on mid-run *causes* stabilization + reward recovery, not a coincidental symptom. A further
control shows an *uncontrolled* entropy bonus (entropy_coeff on, no corridor) actually hurts (never-on
reward 1.70) vs plain PPO with no bonus (9.97) at the same LR — the corridor is what makes the entropy
term usable.

**C3. LR confounder control (Exp 3).** Plain PPO (no EPO) at lr ∈ {3e-6, 5e-6, 1e-5}: entropy
oscillation 0.06 → 0.41 → 0.55 and reward 9.97 → 0.31 → −0.10. Instability and its oscillation
signature scale with LR (not an LR-only artifact), and oscillation magnitude tracks the collapse across
the sweep — consistent with C1/C2.

---

## Theme D — "Entropy bound does not bound the importance ratio" (uSAW-#5, 3fUM)

**Exp 2b (released logs).** Importance-ratio variance is **not logged** in the released runs; we use
PPO clip-fraction (`actor/pg_clipfrac`) and `actor/ppo_kl` as the empirical proxies and plot baseline
vs +EPO per project (figures in `rebuttal/results/exp2_*`). We reframe the appendix's variance
argument as **heuristic/empirical**, not a guarantee, and (if any formal statement is kept) add the
bounded-per-token-probability assumption the reviewer suggested.

---

## Theme E — Hyperparameter sensitivity (91dS-Q6, uSAW-#4) — Exp 4

ScienceWorld GRPO+EPO, one knob at a time around the default (center = Exp 1 grpo κ_l=0 s0 = 0.44/0.42
converged IID/OOD). Converged success (last-3-val mean):

| Knob | low | center | high |
|---|---|---|---|
| λ `entropy_coeff` {0.0005/0.001/0.002} | 0.00/0.00 | 0.44/0.42 | 0.94/0.90 |
| α `out_range_penalty` {0.05/0.1/0.2} | (=center) | 0.08/0.10 | 0.90/0.85 |
| κ_r {1.5/2.0/2.5} | 0.98/0.96 | 0.44/0.42 | 0.08/0.02 |
| smoothing weight {0.5/1.0/2.0} | 0.00/0.00 | 0.44/0.42 | 0.19/0.19 |

**Honest finding:** at a single seed on a 16-sample val set the results are **noisy/high-variance** —
not a clean plateau. Directionally sensible trends do appear: **adequate λ helps** (0→0.44→0.94) and a
**tighter upper cap κ_r helps** (κ_r=1.5 → 0.98 vs κ_r=2.5 → 0.08, reinforcing that the *upper* bound
is the load-bearing part of the corridor — cf. Theme A). We will present this as a sensitivity study
with the explicit caveat that a robustness/plateau claim needs multiple seeds, and we will not overclaim
insensitivity. *Caveats retained from EXPERIMENTS.md §4:* the entropy reference is a cumulative (not
fixed-window) mean, and γ is a fixed scheduled decay (`entropy_smooth_coeff` swept as the practical proxy).

---

## Reproducibility notes (for the de-anonymized release)
- Config that matches the paper's *completing* runs: `max_prompt_length=6144`, `max_model_len=16384`,
  `max_num_batched_tokens=16384`, chunked prefill on (the appendix's 2048/4096 setting overflows the
  accumulated multi-turn prompt — several released runs under it are `crashed`).
- κ_l/κ_r/penalty/start-epoch are encoded in each W&B run name for traceability.
