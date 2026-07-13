# EPO Rebuttal — W&B Run Index

Entity `ruwujiang-rutgers-university`. All new rebuttal runs on branch `rebuttal/epo-experiments`.
Open any URL to read training/val curves from any machine. Metric keys:
`actor/entropy_loss` (mean entropy), `actor/pg_clipfrac`, `actor/ppo_kl`, `critic/score/mean`
(reward, 0–10), `val_l0/success_rate` (IID), `val_l1/success_rate` (OOD), `step_entropy_<i>` (per-turn).

## Exp 2 — log-based analyses (no new runs; uses the released projects)
- Project (PPO, SciWorld): https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_ppo
- Project (GRPO, SciWorld): https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_grpo
- Project (PPO, ALFWorld): https://wandb.ai/ruwujiang-rutgers-university/verl_agent_alfworld_ppo
- Project (GRPO, ALFWorld): https://wandb.ai/ruwujiang-rutgers-university/verl_agent_alfworld_grpo
- Figures/CSVs: `rebuttal/results/exp2_*/`

## Exp 1 — κ_l ablation (ScienceWorld)
| run | W&B |
|---|---|
| ppo kl0 s0 | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_ppo/runs/yzf4pp0e |
| ppo kl0 s1 | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_ppo/runs/d9g3gy2h |
| ppo kl0.5 s0 | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_ppo/runs/er3rb8i2 |
| ppo kl0.5 s1 | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_ppo/runs/hn0066av |
| ppo kl0.8 s0 | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_ppo/runs/3szdx3wz |
| ppo kl0.8 s1 | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_ppo/runs/ni4chnvq |
| grpo kl0 s0 (also Exp4 center) | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_grpo/runs/xy8jfb4a |
| grpo kl0 s1 | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_grpo/runs/26jznylh |
| grpo kl0.5 s0 | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_grpo/runs/5m5z8xgn |
| grpo kl0.5 s1 | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_grpo/runs/a7w0j3du |
| grpo kl0.8 s0 | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_grpo/runs/q0laaxiq |
| grpo kl0.8 s1 | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_grpo/runs/08h8ckvd |

## Exp 3 — causal intervention + LR control (ScienceWorld PPO, seed 0)
| run | W&B |
|---|---|
| toggle_se40 (EPO on @ epoch 40) | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_ppo/runs/34snrr7o |
| never_se9999 (EPO never on) | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_ppo/runs/9o0cun41 |
| plain PPO lr 3e-6 | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_ppo/runs/vv8yc3tt |
| plain PPO lr 5e-6 | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_ppo/runs/spk0ao68 |
| plain PPO lr 1e-5 (106/125) | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_ppo/runs/zkejblvl |

## Exp 4 — one-knob sensitivity (ScienceWorld GRPO+EPO, seed 0)
| run | W&B |
|---|---|
| center ec0.001 (= Exp1 grpo kl0 s0) | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_grpo/runs/xy8jfb4a |
| ec0.0005 | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_grpo/runs/kekjxrv5 |
| ec0.002 | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_grpo/runs/2a9ytt9x |
| pen0.1 | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_grpo/runs/1kyy5ddy |
| pen0.2 | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_grpo/runs/3m8yrm6r |
| kr1.5 | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_grpo/runs/lou9bz48 |
| kr2.5 | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_grpo/runs/xnfx0tft |
| esc0.5 | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_grpo/runs/7juqirw2 |
| esc2.0 | https://wandb.ai/ruwujiang-rutgers-university/verl_agent_sciworld_grpo/runs/7415js66 |
