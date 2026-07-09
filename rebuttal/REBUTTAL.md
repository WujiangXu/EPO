# Rebuttal — Submission 13068
### *Exploring the Entropy Mechanism in On-policy Optimization for LLM Agents (EPO)*

> **How to use this file.** OpenReview requires a separate Official Comment per reviewer.
> Post the **General Response** once (or paste the relevant Theme into each reply), then post the
> matching **Per-Reviewer** block under each review. Themes are labeled **A–G** so per-reviewer
> replies can point to them without repetition.

---

## General Response

We thank all three reviewers for the careful and constructive reviews. We are encouraged that the
reviewers find the problem **important and timely** (3fUM, 91dS, uSAW), the method **simple,
orthogonal to PPO/GRPO, and easy to adopt** (3fUM-S1, 91dS-S2), the **ablations comprehensive**
(3fUM-S2), and the **gains promising, especially on ScienceWorld** (91dS, uSAW).

The reviews converge on two central technical concerns and several presentation issues. We address
them head-on below with (1) a **corrected and honest theoretical framing**, (2) **new experiments**
(the corridor lower-bound ablation and a direct causal intervention), and (3) **analyses computed
from our already-released training logs**. We also confirm that **code and full W&B training logs are
released** (anonymized link in the abstract footnote), and we will de-anonymize with configs and
evaluation splits.

### Theme A — "κ_l=0 disables the corridor lower bound" *(the central concern; 3fUM-W1, uSAW-W1)*

The reviewers are **correct**: our reported runs use κ_l=0, κ_r=2.0, so the left penalty branch
`[κ_l·H̄−H]₊ = [−H]₊ = 0` and only the **upper** cap `H < 2·H̄` is active. We should have made this
explicit; we apologize for the mismatch with the appendix narrative. Three points reconcile this:

1. **It is consistent with — indeed dictated by — our diagnosis.** Our core empirical claim (Intro) is
   that the multi-turn failure mode is *not* entropy collapse but **uncontrolled entropy
   growth/oscillation (over-exploration)**. The mechanism that fights *that* failure is exactly the
   **active upper bound (κ_r)**. So the branch that is "on" in our runs is precisely the one our thesis
   predicts should matter.
2. **Collapse prevention does not rely on the corridor floor.** The entropy term `L^H` in the objective
   already supplies a **direct upward gradient on entropy**, which is what keeps entropy away from zero.
   The corridor's role is to *dampen oscillation/growth*, not to be the sole anti-collapse device. The
   appendix conflated these two; we will separate them explicitly (see Theme A revision below).
3. **We now add the missing ablation.** We run PPO+EPO and GRPO+EPO with **κ_l ∈ {0.5, 0.8}** (the
   two-sided corridor — this is literally the code default) and report the full table. We also add, from
   existing logs, an **entropy-vs-floor overlay** showing whether the lower bound would ever bind in
   these environments. Expected outcome (which we will report either way): in ScienceWorld/ALFWorld the
   measured entropy stays well above κ_l·H̄, so κ_l=0 was empirically harmless *here*; the floor becomes
   load-bearing only in collapse-prone regimes, which we demonstrate with a higher-learning-rate setting.

**Theory revision.** The stability result (Prop. B.2, O(T) vs O(T²)) only requires the deviation
`|H−H̄|` to be bounded, which the **active upper cap + `L^H`** already provide; the two-sided corridor
is the general form and the floor is a safety rail for collapse-prone regimes. We will restate the
appendix accordingly and remove the claim that the floor is essential in the reported runs.

### Theme B — "Smoothing acts over training steps, not turns" + "circular proof" *(3fUM-W2)*

**On step-vs-turn: this is a notation problem in the paper, not in the method.** The implemented
reference is **per-turn**, and we will correct the equations to show it:

- `step_entropy_list[t]` stores the mean entropy at **turn t** of the rollout.
- Each RL step appends one **per-turn vector**; the historical reference is the mean over past steps
  **taken along the step axis only, keeping the turn dimension** (`np.mean(history, axis=0)`).
- In the loss, each token is anchored to **its own turn's** historical band
  (`avg_entropy_per_sample = entropy_history[rollout_step]`).

So the reference is H̄^ref_t (one band per turn position), averaged over training history — it **does**
act on per-turn quantities. The paper collapsed this to a scalar `H̄^{W_k}` for brevity; we will rewrite
Eq. (8)/Alg. 1 with the turn index restored.

**On circularity:** we intend Prop. B.2 as a **conditional** statement — *if* entropy remains in the
corridor (the design goal, empirically verified by the entropy curves in §5), *then* cumulative
deviation grows as O(T) rather than O(T²). We are characterizing the *benefit of maintaining* the
corridor, not proving the optimizer attains it; we will state this explicitly and drop any implication
of a self-contained convergence theorem. The arithmetic core (penalty=0 iff inside corridor; O(T)
linear bound; O(T²) triangular drift) is **machine-checked in Lean 4** (`theory_check.lean`), which we
will cite.

### Theme C — "Is entropy oscillation the cause, or a symptom?" *(uSAW-W2, uSAW-W3)*

Fair — our current evidence is largely correlational. We strengthen it three ways:

1. **Correlation across all runs (from released logs, no new training):** across seeds/methods we
   correlate the entropy-oscillation magnitude (std of step-to-step ΔH) with final success rate; we
   report the coefficient and scatter.
2. **Direct intervention (new experiment):** starting from a *diverging* PPO run, we **switch on EPO
   smoothing mid-training** (and, symmetrically, inject a controlled entropy perturbation into a stable
   run). If entropy stabilizes and reward recovers precisely when smoothing is toggled, oscillation is
   causal, not incidental.
3. **Confounder control:** we show baseline instability persists across multiple learning rates (ruling
   out an LR-only explanation), while EPO is stable across the same set.

We will also soften the wording from "the primary cause" to "a primary, *controllable* driver."

### Theme D — "An entropy bound does not bound the importance ratio" *(uSAW-#5, 3fUM)*

Correct — an entropy lower bound does not *formally* bound `π_θ(a|s)/π_old(a|s)`. We will (a) reframe
the appendix's variance argument as a **heuristic/empirical** mechanism rather than a guarantee, and
(b) support it directly with a plot from our logs: **measured importance-ratio variance and PPO
clip-fraction, with vs. without EPO**. If we retain any formal statement, we will add the stronger
assumption the reviewer suggests (bounded per-token probability ⇒ bounded ratio).

### Theme E — Novelty & why EPO fits sparse reward *(91dS-W1/W2/W3, 91dS-Q5)*

Entropy regularization is indeed standard; our contribution is **what to regularize and how**, in the
multi-turn setting:
- **(i)** identifying and characterizing the *exploration–exploitation cascade failure* specific to
  multi-turn sparse-reward agents;
- **(ii)** **trajectory-level** entropy aggregation across turns (vs. per-step);
- **(iii)** a **stateful, history-anchored, per-turn** smoothing regularizer (vs. the *stateless
  per-update* bonuses of single-turn RLVR / advantage-shaping methods);
- **(iv)** scheduled phase-based weighting.

Reviewer 3fUM's own summary captures this distinction well. **Why this helps under sparse reward:** with
no per-turn reward signal, nothing cancels entropy drift, so it compounds across turns (the O(T²)
argument); the historical anchor supplies the stable reference *exactly when the reward signal is
absent*. We will add this intuition to §3.

### Theme F — ALFWorld gains weaker; abstract balance *(91dS-W4, uSAW-#2)*

**Mechanism.** ScienceWorld requires 30+ actions before any feedback (extreme sparsity → severe
oscillation → large EPO gains). ALFWorld has more structured, shorter-horizon feedback (milder
oscillation → EPO mainly speeds convergence and improves OOD robustness). This is our thesis, not an
inconsistency: EPO helps most exactly where cascade failure is worst.

**On the 95.8→85.4 case (honest accounting).** That number is the **peak** IID metric (Succ.\*) for
*PPO+EPO on ALFWorld*, against an already near-saturated baseline. On the same row, the
robustness-oriented metrics improve: converged IID 72.3→73.4 (+1.5%), peak OOD 87.5→91.7 (+4.8%),
converged OOD 70.9→74.3 (+4.8%). For **GRPO+EPO on ALFWorld all four metrics improve** (up to +19.8%).
We will **rebalance the abstract** to report the trade-off alongside the headline gains (not the gains
in isolation).

**PPO+EPO > GRPO/GRPO+EPO in OOD ScienceWorld (91dS-Q, uSAW):** PPO's higher-variance, critic-based
updates amplify entropy oscillation under sparse reward, so they benefit most from smoothing; GRPO's
group-relative advantage already provides oscillation resistance, leaving less headroom. We will add
this analysis.

### Theme G — Presentation, terminology, typos, artifacts

- **Fig. 1 (91dS-Q1):** environment is **ScienceWorld (PPO)**. Takeaway: (a,b) per-turn entropy
  weighting (**EPO-Decay**) cannot separate early- vs. late-turn entropy — the curves overlap the
  unweighted **EPO-Base**, showing shared parameters prevent per-turn control; (c) PPO oscillates with
  flat reward while EPO keeps entropy stable (1.2→0.3) and reward rising. We will rewrite the caption.
- **EPO-Decay (91dS-Q2):** an ablation variant that applies *higher* entropy weight to early turns and
  *lower* to late turns. We will define it at first mention (or defer it to §5) instead of using it
  before the method is introduced.
- **EPO-Base vs. EPO (91dS-Q3):** EPO-Base = EPO **without** the smoothing regularizer (trajectory-level
  entropy `L^H` only); EPO = full method (adds `L^smooth`). We will state this at first use.
- **"Adaptive" weighting (uSAW-#3):** correct — β is a fixed, monotonic **scheduled** decay, not
  entropy-feedback-driven. We will rename it "scheduled phase-based weighting" and note a truly
  entropy-adaptive variant as future work.
- **Reward curves >1 (uSAW-#1):** the terminal reward is a **×10-scaled binary outcome** (`10·won`;
  0 on intermediate turns), so training-reward curves lie on a 0–10 scale, not 0–1. The RL signal
  remains sparse and terminal-only; we will relabel the axis as "scaled episode return" and reconcile
  the `r_T∈{0,1}` notation with the ×10 implementation scaling.
- **EPO expansion (3fUM-C1):** we will use "**Entropy-regularized Policy Optimization**" consistently
  (the appendix's "Entropy-smoothed …" instances will be fixed).
- **"Appendix 5" refs (3fUM-C2):** a LaTeX cross-referencing bug from
  `\renewcommand{\sectionautorefname}{Appendix}`, which mislabels `\autoref` to the main Experiments
  section. We will fix the autoref configuration.
- **Software/Datasets scores (3fUM, 91dS):** code and **full W&B training logs are already released**
  (anonymized link in the abstract). We will de-anonymize and add configs + evaluation splits for full
  reproducibility.

### Committed experiments (priority order)

1. **κ_l>0 corridor ablation** *(Theme A)* — highest priority; directly resolves the concern shared by
   two reviewers. Includes the entropy-vs-floor overlay from existing logs.
2. **Log-based analyses** *(Themes C, D)* — oscillation-vs-success correlation; importance-ratio
   variance / clip-fraction with vs. without EPO. No new training.
3. **Direct causal intervention** *(Theme C)* — mid-run EPO toggle / entropy perturbation.
4. **Hyperparameter sensitivity** *(Theme E; 91dS-Q6, uSAW-#4)* — λ, γ, κ_r, entropy-window.

---

## Response to Reviewer 3fUM

We appreciate the precise, math-level reading. Both weaknesses are well taken and both are addressed.

- **W1 (κ_l=0 disables the lower bound):** You are right. Please see **Theme A**. Short version: the
  active upper bound (κ_r) is the mechanism aligned with our diagnosed failure (entropy *growth*, not
  collapse); anti-collapse comes from the entropy term `L^H`, not the floor; and we now add a κ_l∈{0.5,
  0.8} ablation plus an entropy-vs-floor overlay, and rewrite the appendix so the stability result no
  longer depends on the (inactive) floor.
- **W2 (step-vs-turn; circular proof):** Please see **Theme B**. The implemented reference is
  **per-turn** (H̄^ref_t), averaged over history while keeping the turn dimension — the scalar notation
  in the paper was an oversimplification we will fix. We reframe the proposition as a **conditional**
  stability statement (if-corridor-then-O(T)) and cite the Lean-checked arithmetic core.
- **C1 (EPO name) / C2 ("Appendix 5"):** both confirmed and fixed — see **Theme G**.
- **Soundness/Software:** with the theory reframed to match the runs and the code + W&B logs released,
  we hope W1/W2 are resolved and the soundness and software scores can be reconsidered.

## Response to Reviewer 91dS

Thank you for the detailed questions; we answer each directly.

- **W1 (novelty):** See **Theme E** — the novelty is trajectory-level, stateful, per-turn, history-
  anchored regularization for the multi-turn cascade failure, distinct from standard/stateless entropy
  bonuses.
- **W2/Q5 (why sparse-reward):** See **Theme E** — with no per-turn signal, entropy drift compounds
  across turns; the historical anchor supplies the missing reference exactly when reward is absent.
- **W3 (motivation for historical smoothing underdeveloped):** See **Themes B & C** — we clarify the
  per-turn, history-anchored mechanism and add causal evidence.
- **W4 (ALFWorld weaker/inconsistent):** See **Theme F** — mechanistic explanation + transparent
  accounting of the one metric that drops.
- **W5 & Q1–Q3 (presentation, Fig. 1, EPO-decay, EPO-Base vs EPO):** See **Theme G** for point-by-point
  answers; we will restructure the early exposition so EPO-Decay/EPO-Base are defined before use.
- **Q4 (ScienceWorld > ALFWorld):** See **Theme F** (sparsity severity) — EPO helps most where cascade
  failure is worst.
- **Q6 (sensitivity to window size & schedule):** we add a sensitivity study over the entropy-window,
  γ, λ, and κ_r (**Theme E / committed experiment 4**). Note: the reference uses a **cumulative**
  history (not a fixed window), which adapts fast early (weight ~1/k) and stabilizes later; we will make
  this design choice explicit.

## Response to Reviewer uSAW

Thank you — we agree these points sharpen the paper, and we are glad you see it as Findings-worthy.

- **W1 (κ_l=0 mismatch):** Agreed; see **Theme A** (new κ_l>0 ablation + reframed theory).
- **W2 (correlation vs. causation):** See **Theme C** — we add a **direct intervention** (mid-run EPO
  toggle) plus confounder controls, converting correlation into causal evidence.
- **W3 (overclaiming "cascade failure"):** See **Theme C** — we soften "primary cause" to "primary,
  controllable driver" and back it with the intervention and cross-run correlation.
- **#1 (reward curves > 1):** Clarified in **Theme G** — reward is a ×10-scaled binary terminal outcome
  (`10·won`); the RL signal is still sparse/terminal. We will relabel the axis.
- **#2 (abstract cherry-picking; 95.8→85.4):** See **Theme F** — honest accounting; we will rebalance
  the abstract.
- **#3 ("adaptive" is monotonic decay):** Correct; we rename to "scheduled phase-based weighting" (Theme
  G).
- **#4 (sensitivity for λ, γ, κ_l, κ_r):** added — committed experiment 4, plus the κ_l ablation.
- **#5 (entropy bound ⇏ ratio bound):** Agreed; see **Theme D** — reframed as heuristic + empirical
  ratio-variance / clip-fraction evidence, with the option to add a bounded-probability assumption.

---

## Summary of paper revisions we will make
1. State κ_l=0 in the reported runs; add the **κ_l∈{0.5,0.8} ablation** and entropy-vs-floor overlay;
   rewrite Appendix B so the stability result relies on the active upper bound + `L^H`, not the floor.
2. Rewrite Eq. (8)/Alg. 1 with the **per-turn** reference H̄^ref_t; restate the proposition as a
   **conditional** result; cite the **Lean-verified** arithmetic core.
3. Add **causal evidence** (mid-run intervention, cross-run oscillation↔success correlation, LR
   confounder control) and soften causal wording.
4. Add **importance-ratio variance / clip-fraction** plots; reframe the variance argument as heuristic.
5. Add a **hyperparameter sensitivity** study (λ, γ, κ_r, entropy-window).
6. **Rebalance the abstract**; add the ScienceWorld-vs-ALFWorld and PPO-vs-GRPO analyses.
7. Fix presentation: define EPO-Decay / EPO-Base before use; rewrite Fig. 1 caption; rename
   "adaptive"→"scheduled"; relabel reward axis; unify "Entropy-regularized Policy Optimization"; fix the
   broken `\autoref` ("Appendix 5") cross-references; de-anonymize code + logs with configs/splits.
