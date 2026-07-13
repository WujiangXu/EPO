# EPO Rebuttal — Results Log

Branch: `rebuttal/epo-experiments`. Tracking file for all rebuttal experiments.
Updated on every launch and completion. See `rebuttal/EXPERIMENTS.md` / `rebuttal/REBUTTAL.md`
for the plan and reviewer mapping.

## Environment
- Cluster: Slurm. Login node has NO GPU and CANNOT see `/ai4rl`. Submit via `--account=usr-sr`
  with QOS `h200_dev` / `h200_rl_shared` / `h200_usr-sr_high`. 8×H200 per node.
- **Storage: everything on `/ai4rl/fsx/impwxu/epo` (petabyte lustre, compute-node-only).** Login-node
  local disk is tiny tmpfs (512M/1G) — do NOT stage there. `$EPO_BASE=/ai4rl/fsx/impwxu/epo`.
- Networking: compute nodes reach the internet DIRECTLY; the inherited X2P proxy env is stale and
  must be unset in jobs (`env_common.sh` handles this).
- Env: micromamba env `epo` at `$EPO_BASE/micromamba` (python3.10; torch 2.6.0 cu124 + vllm 0.8.5 +
  verl + scienceworld). (`python3.10 -m venv` is broken here — no ensurepip; micromamba used instead.)
- Model: `SCI_MODEL=$EPO_BASE/models/Qwen2.5-7B-Instruct`.
- Shared job env: `rebuttal/results/logs/env_common.sh` (paths, CUDA, proxy unset, W&B key).
- W&B entity: `ruwujiang-rutgers-university`; projects `verl_agent_{sciworld,alfworld}_{ppo,grpo}`.
  Key stored at `$HOME/.wandb_key` (chmod 600, never committed).
- Config policy: keep the two `dp_actor.py` quirks AS-IS (max_step→25 threshold; max_epochs=150) to
  match the submitted numbers (EXPERIMENTS.md §0 option A).

## Provisioning status  — COMPLETE (provision job 191573 + finalize job 191587)
| Item | Status |
|---|---|
| micromamba env `epo` (torch2.6+cu124 / vllm0.8.5 / flash-attn / verl / scienceworld) | DONE (GPU=True) |
| Qwen2.5-7B-Instruct download | DONE (15G, 4 shards) at $EPO_BASE/models |
| setuptools pinned <81 (verl needs pkg_resources) | DONE |
| openjdk 17 + JAVA_HOME (ScienceWorld JVM) | DONE — SciWorld JVM OK, 30 tasks |
| W&B key + access | DONE — 42 runs in verl_agent_sciworld_ppo |
| Data preprocess ($HOME/data/verl-agent/text/{train,test}.parquet) | DONE |
| Smoke test (gate) | DONE — 4/4 PASS |

### Verified metric keys (from code, for Exp 2 analysis)
- mean entropy = `actor/entropy_loss` (no `actor/entropy`); clip frac = `actor/pg_clipfrac`;
  ratio variance NOT logged (proxy `actor/ppo_kl`); reward = `critic/score/mean`;
  success = `val_l0/success_rate` (IID) / `val_l1/success_rate` (OOD); per-turn = `step_entropy_<i>`.
### Runtime prereq (from code): ScienceWorld needs a JVM — openjdk installed into env, `JAVA_HOME` set in env_common.sh.

## Experiment 2 — log-based analyses (no training)  — DONE (figs/CSVs in rebuttal/results/exp2_*)
**2a oscillation↔success (Pearson r / Spearman ρ / n):**
| Project | Pearson r (p) | Spearman ρ | n |
|---|---|---|---|
| verl_agent_sciworld_ppo  | **−0.509 (p=1.1e-3)** | −0.559 | 38 |
| verl_agent_sciworld_grpo | +0.156 (p=0.36) | 0.120 | 37 |
| verl_agent_alfworld_ppo  | +0.009 (p=0.97) | 0.087 | 16 |
| verl_agent_alfworld_grpo | +0.015 (p=0.91) | 0.317 | 65 |

**2c entropy-vs-corridor floor-bind at κ_l=0:** floor binds **0 steps in ALL projects**
(sciworld_ppo 0/9, sciworld_grpo 0/123, alfworld_ppo 0/150, alfworld_grpo 0/150).

**2b variance:** importance-ratio variance NOT logged in the released runs; clip-fraction
(`actor/pg_clipfrac`) + `actor/ppo_kl` proxy plotted per project.

**Interpretation (for the rebuttal):**
- *Theme A (κ_l=0 harmless):* strongly supported — the κ_l=0 lower bound **never binds** in any
  released run, i.e. measured entropy always stays above the floor. Exp 1 (active κ_l) + Exp 5
  (collapse regime) will close this.
- *Theme C (oscillation↔success):* clear **negative** correlation in **ScienceWorld PPO**
  (r=−0.51, p≈1e-3) — exactly the sparsest, highest-variance setting where EPO helps most (Theme F).
  Weak/absent in GRPO and ALFWorld (consistent with milder cascade failure there); report honestly.
- Grouping-by-name is imperfect (runs use both `ours_*` and `ec0.001` tags), so 2b baseline-vs-EPO
  coloring is approximate; the 2a correlation uses ALL runs and is unaffected.

## Experiment 1 — κ_l ablation (ScienceWorld, 12 runs) — COMPLETE (jobs 192134-192145, h200_usr-sr_high)
W&B project: verl_agent_sciworld_{ppo,grpo}; run name `<algo>_s<seed>_..._ec0.001_es1.0_kl<κ_l>_kr2.0_sw`.
**Config:** `max_prompt_length=6144`, `max_model_len=16384`, `max_num_batched_tokens=16384`,
`enable_chunked_prefill=True`, per-run isolated data dir. (Original EXPERIMENTS.md 2048/4096 config
overflows the multi-turn prompt & crashes — the paper's own 2048/4096 runs are mostly `crashed`;
we adopted the paper's *completing* 6144-prompt config.) Metric protocol: converged = mean of last
3 validations; peak = max over training. Val sets are 16 samples (0.0625 granularity) → noisy.

**Per-run (converged IID / OOD | peak IID / OOD):**
| Algo | κ_l | seed | IID conv | OOD conv | IID peak | OOD peak |
|---|---|---|---|---|---|---|
| PPO+EPO | 0 | 0 | 1.000 | 0.917 | 1.000 | 1.000 |
| PPO+EPO | 0 | 1 | 1.000 | 0.979 | 1.000 | 1.000 |
| PPO+EPO | 0.5 | 0 | 1.000 | 0.979 | 1.000 | 1.000 |
| PPO+EPO | 0.5 | 1 | 0.271 | 0.229 | 0.438 | 0.438 |
| PPO+EPO | 0.8 | 0 | 0.229 | 0.229 | 0.562 | 0.375 |
| PPO+EPO | 0.8 | 1 | 1.000 | 0.958 | 1.000 | 1.000 |
| GRPO+EPO | 0 | 0 | 0.438 | 0.417 | 0.688 | 0.750 |
| GRPO+EPO | 0 | 1 | 0.604 | 0.625 | 1.000 | 1.000 |
| GRPO+EPO | 0.5 | 0 | 0.292 | 0.229 | 0.375 | 0.312 |
| GRPO+EPO | 0.5 | 1 | 0.208 | 0.146 | 0.312 | 0.312 |
| GRPO+EPO | 0.8 | 0 | 0.896 | 0.875 | 1.000 | 1.000 |
| GRPO+EPO | 0.8 | 1 | 0.125 | 0.208 | 0.312 | 0.562 |

**Seed-averaged per κ_l (converged IID / OOD | peak IID / OOD):**
| Algo | κ_l=0 | κ_l=0.5 | κ_l=0.8 |
|---|---|---|---|
| PPO+EPO | **1.000 / 0.948** \| 1.00/1.00 | 0.635 / 0.604 \| 0.72/0.72 | 0.615 / 0.594 \| 0.78/0.69 |
| GRPO+EPO | 0.521 / 0.521 \| 0.84/0.88 | 0.250 / 0.188 \| 0.34/0.31 | 0.510 / 0.542 \| 0.66/0.78 |

**κ_l comparison / Theme A conclusion:** activating the corridor floor (κ_l>0) does **not** improve
over κ_l=0. For PPO, κ_l=0 is clearly best (converged 1.00/0.95 vs ~0.62 for κ_l>0). For GRPO,
κ_l=0 ≈ κ_l=0.8 (~0.52) and κ_l=0.5 is worst. This **supports the rebuttal's Theme A claim** that
the κ_l=0 lower bound was *empirically harmless* here (consistent with Exp 2c: the floor never binds).
**Caveat:** strong seed variance — several cells are bimodal (one seed →~1.0, the other →~0.2), so
differences among κ_l are within 2-seed noise on the 16-sample val set; the safe claim is "κ_l>0 gives
no consistent benefit," not a precise ranking. The active upper cap (κ_r=2.0) + entropy term carry EPO.

## Experiments 3 + 4 — COMPLETE (one 13-node job 194742; 12 runs to 125, plainppo_lr1e5 to 106/125)
Single sbatch (`run_exp34_batch.sbatch`) requested 13 nodes, one 8×H200 step per node. The job hit its
24h wall; 12 runs finished 125/125, the intentionally-unstable `plainppo_lr1e5` reached 106/125
(treated as final — it had already collapsed). W&B URLs in `rebuttal/results/WANDB_INDEX.md`.

### Exp 3 — causal intervention + LR confounder control (ScienceWorld PPO, seed 0)
entOsc = std(ΔH); split at epoch 40 for the toggle/never pair. rew = final `critic/score/mean` (0–10 scale).
| Run | entOsc[<40] | entOsc[≥40] | reward | IID conv | OOD conv |
|---|---|---|---|---|---|
| toggle_se40 (EPO ON @40) | 0.055 | **0.019** | **9.50** | **0.98** | **0.98** |
| never_se9999 (EPO never) | 0.078 | 0.031 | 1.70 | 0.27 | 0.19 |
| plain PPO lr 3e-6 (no EPO) | 0.060 | 0.022 | 9.97 | 1.00 | 0.98 |
| plain PPO lr 5e-6 (no EPO) | 0.413 | 0.102 | 0.31 | 0.10 | 0.08 |
| plain PPO lr 1e-5 (no EPO)* | 0.548 | 0.129 | −0.10 | 0.00 | 0.00 |

*106/125. **Conclusions:**
- **Causal toggle:** switching EPO smoothing ON at epoch 40 drops post-toggle entropy oscillation
  (0.055→0.019) and lifts reward 1.70→9.50 (IID 0.27→0.98) vs the identical never-on control — direct
  evidence that controlling oscillation *causes* the gain (Theme C), not a symptom.
- **Uncontrolled entropy hurts:** never_se9999 (entropy bonus, no corridor) = 1.70; plain PPO (no bonus)
  at the same lr = 9.97 — an unbounded entropy term drives over-exploration; the corridor rescues it.
- **LR confounder:** plain-PPO instability scales with LR (entOsc 0.06→0.41→0.55; reward 9.97→0.31→−0.10
  for lr 3e-6/5e-6/1e-5) and entropy-oscillation tracks the collapse — oscillation↔failure is robust
  across LR (not an LR-only artifact).

### Exp 4 — one-knob sensitivity (ScienceWorld GRPO+EPO, seed 0; converged IID / OOD)
Center (defaults ec0.001, pen0.05, κ_r2.0, es_coeff1.0) = Exp1 grpo_kl0_s0 = **0.44 / 0.42**.
| Knob | low | center | high |
|---|---|---|---|
| λ entropy_coeff {0.0005, 0.001, 0.002} | 0.00 / 0.00 | 0.44 / 0.42 | 0.94 / 0.90 |
| α out_range_penalty {0.05, 0.1, 0.2} | — (=center) | 0.08 / 0.10 (0.1) | 0.90 / 0.85 (0.2) |
| κ_r {1.5, 2.0, 2.5} | 0.98 / 0.96 (1.5) | 0.44 / 0.42 | 0.08 / 0.02 (2.5) |
| es_coeff {0.5, 1.0, 2.0} | 0.00 / 0.00 (0.5) | 0.44 / 0.42 | 0.19 / 0.19 (2.0) |

**Conclusion (honest):** single-seed GRPO on the 16-sample val set is **noisy/high-variance** — not the
clean "broad plateau" hoped for. Two semi-clean trends: **higher λ helps** (0→0.44→0.94) and **smaller
κ_r helps** (0.98→0.44→0.08); penalty & es_coeff are non-monotonic. Report as: EPO is sensitive to
these knobs at 1 seed; the trends (tighter upper cap κ_r, adequate λ) are directionally sensible, but a
robustness claim needs more seeds. **Do not overclaim insensitivity.**
