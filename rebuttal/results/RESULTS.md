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

## Experiment 1 — κ_l ablation (ScienceWorld, full matrix 12 runs) — ALL (RE)SUBMITTED
W&B project: verl_agent_sciworld_{ppo,grpo}; run name `<algo>_s<seed>_..._ec0.001_es1.0_kl<κ_l>_kr2.0_sw`.
**Config note:** switched to `max_prompt_length=6144` + `max_model_len=32768` (matches the paper's
*completing* runs). The original EXPERIMENTS.md config (2048/4096) overflows the accumulated
multi-turn prompt and crashes mid-training — the paper's own 2048/4096 runs are mostly `crashed`.
Also fixed: launcher skips data-preprocess if parquet exists (concurrent-run race).
| Algo | κ_l | seed | Slurm job | IID Succ.* | OOD Succ.* | Status |
|---|---|---|---|---|---|---|
| PPO+EPO | 0 | 0 | 192022 | — | — | queued |
| PPO+EPO | 0 | 1 | 192023 | — | — | queued |
| PPO+EPO | 0.5 | 0 | 192024 | — | — | queued |
| PPO+EPO | 0.5 | 1 | 192025 | — | — | queued |
| PPO+EPO | 0.8 | 0 | 192026 | — | — | queued |
| PPO+EPO | 0.8 | 1 | 192027 | — | — | queued |
| GRPO+EPO | 0 | 0 | 192028 | — | — | queued |
| GRPO+EPO | 0 | 1 | 192029 | — | — | queued |
| GRPO+EPO | 0.5 | 0 | 192030 | — | — | queued |
| GRPO+EPO | 0.5 | 1 | 192031 | — | — | queued |
| GRPO+EPO | 0.8 | 0 | 192032 | — | — | queued |
| GRPO+EPO | 0.8 | 1 | 192033 | — | — | queued |

## Experiment 3 — causal intervention (planned)
| Config | W&B run name | Status |
|---|---|---|
| PPO+EPO start_epoch=40 | — | planned |
| PPO+EPO never-on (control) | — | planned |
| plain PPO lr 3e-6/5e-6/1e-5 | — | planned |

## Experiment 4 — sensitivity (planned)
| Knob | Values | Status |
|---|---|---|
| entropy_coeff | 0.0005/0.001/0.002 | planned |
| out_range_penalty | 0.05/0.1/0.2 | planned |
| max_ratio (κ_r) | 1.5/2.0/2.5 | planned |
| entropy_smooth_coeff | 0.5/1.0/2.0 | planned |
