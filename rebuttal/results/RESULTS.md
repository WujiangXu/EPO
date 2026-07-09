# EPO Rebuttal — Results Log

Branch: `rebuttal/epo-experiments`. Tracking file for all rebuttal experiments.
Updated on every launch and completion. See `rebuttal/EXPERIMENTS.md` / `rebuttal/REBUTTAL.md`
for the plan and reviewer mapping.

## Environment
- Cluster: Slurm login node, partition `h200`/`all`, 8×H200 per node, `cuda/12.4.1` module.
- Env: `$HOME/venvs/verl-agent-sciworld` (python3.10; torch 2.6.0 cu124 + vllm 0.8.5 + verl + scienceworld).
- Model: `SCI_MODEL=$HOME/models/Qwen2.5-7B-Instruct` (ScienceWorld backbone).
- W&B entity: `ruwujiang-rutgers-university`; projects `verl_agent_{sciworld,alfworld}_{ppo,grpo}`.
- Config policy: keep the two `dp_actor.py` quirks AS-IS (max_step→25 threshold; max_epochs=150) to
  match the submitted numbers (EXPERIMENTS.md §0 option A).

## Provisioning status
| Item | Status |
|---|---|
| SciWorld venv | in progress |
| Qwen2.5-7B-Instruct | pending |
| W&B login | pending (awaiting API key) |
| Data preprocess | pending |
| Smoke test (gate) | pending |

## Experiment 2 — log-based analyses (no training)
| Sub | Project | Metric | Value | Fig/CSV | Status |
|---|---|---|---|---|---|
| 2a oscillation↔success | — | Pearson/Spearman r | — | — | pending |
| 2b clip-frac / ratio var | — | EPO vs baseline | — | — | pending |
| 2c entropy-vs-corridor | — | floor-bind steps (κ_l=0) | — | — | pending |

## Experiment 1 — κ_l ablation (ScienceWorld, full matrix 12 runs)
| Algo | κ_l | seed | W&B run name | IID Succ.* | OOD Succ.* | Succ.̄ | Status |
|---|---|---|---|---|---|---|---|
| PPO+EPO | 0 | 0 | — | — | — | — | pending |
| PPO+EPO | 0 | 1 | — | — | — | — | pending |
| PPO+EPO | 0.5 | 0 | — | — | — | — | pending |
| PPO+EPO | 0.5 | 1 | — | — | — | — | pending |
| PPO+EPO | 0.8 | 0 | — | — | — | — | pending |
| PPO+EPO | 0.8 | 1 | — | — | — | — | pending |
| GRPO+EPO | 0 | 0 | — | — | — | — | pending |
| GRPO+EPO | 0 | 1 | — | — | — | — | pending |
| GRPO+EPO | 0.5 | 0 | — | — | — | — | pending |
| GRPO+EPO | 0.5 | 1 | — | — | — | — | pending |
| GRPO+EPO | 0.8 | 0 | — | — | — | — | pending |
| GRPO+EPO | 0.8 | 1 | — | — | — | — | pending |

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
