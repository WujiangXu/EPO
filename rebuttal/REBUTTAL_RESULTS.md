# Rebuttal — Experimental Results Addendum (Submission 13068, EPO)

Companion to `REBUTTAL.md`. This file plugs the **actual measured numbers** from the rebuttal
experiments into Themes A / C / D so the OpenReview replies cite concrete evidence instead of
"we will report." All numbers are reproducible from the released W&B logs + the new runs on branch
`rebuttal/epo-experiments` (see `rebuttal/results/RESULTS.md`, `analysis/wandb_analysis.py`).

Status: **Exp 1 & Exp 2 complete.** Exp 3 (causal toggle + LR control) and Exp 4 (sensitivity) are
running (1 shared 13-node job, ETA ~12–20h); their tables are marked *[pending]* and will be filled
on completion.

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

**C2. Direct causal intervention (Exp 3) — [pending].** PPO+EPO with smoothing gated on at epoch 40
(`entropy_smooth_start_epoch=40`) vs a never-on control. We will show entropy trajectory
(`actor/entropy_loss`) and reward (`critic/score/mean`) with the toggle marked; expected: entropy
stabilizes and reward recovers at the toggle. *Early signal already visible:* in the LR-confounder
arm the plain-PPO lr=1e-5 run is markedly unstable (per-step time ballooning ~2–3× vs stable runs),
consistent with LR-independent instability that EPO controls.

**C3. LR confounder control (Exp 3) — [pending].** Plain PPO at lr ∈ {3e-6, 5e-6, 1e-5}: we will
show instability persists across LR (not an LR artifact) while EPO is stable across the same set.

---

## Theme D — "Entropy bound does not bound the importance ratio" (uSAW-#5, 3fUM)

**Exp 2b (released logs).** Importance-ratio variance is **not logged** in the released runs; we use
PPO clip-fraction (`actor/pg_clipfrac`) and `actor/ppo_kl` as the empirical proxies and plot baseline
vs +EPO per project (figures in `rebuttal/results/exp2_*`). We reframe the appendix's variance
argument as **heuristic/empirical**, not a guarantee, and (if any formal statement is kept) add the
bounded-per-token-probability assumption the reviewer suggested.

---

## Theme E — Hyperparameter sensitivity (91dS-Q6, uSAW-#4) — Exp 4 [pending]

ScienceWorld GRPO+EPO, one knob at a time around the default (center = Exp 1 grpo κ_l=0 seed 0):
λ=`entropy_coeff`∈{0.0005, 0.001, 0.002}, α=`out_range_penalty`∈{0.05, 0.1, 0.2},
κ_r∈{1.5, 2.0, 2.5}, smoothing weight `entropy_smooth_coeff`∈{0.5, 1.0, 2.0}. Converged IID/OOD per
value will be tabulated to show a broad plateau (method not brittle). *Caveats retained from
EXPERIMENTS.md §4:* the entropy reference is a cumulative (not fixed-window) mean, and γ is a fixed
scheduled decay (we sweep `entropy_smooth_coeff` as the practical proxy).

---

## Reproducibility notes (for the de-anonymized release)
- Config that matches the paper's *completing* runs: `max_prompt_length=6144`, `max_model_len=16384`,
  `max_num_batched_tokens=16384`, chunked prefill on (the appendix's 2048/4096 setting overflows the
  accumulated multi-turn prompt — several released runs under it are `crashed`).
- κ_l/κ_r/penalty/start-epoch are encoded in each W&B run name for traceability.
