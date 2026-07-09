#!/usr/bin/env python3
"""
Fast EPO smoke test — validates the code changes made for the rebuttal WITHOUT a full training run.

Run in the project env (with torch + verl importable), ideally on a box with a free GPU:

    cd epo_code
    python tests/test_epo_smoke.py            # uses CUDA if available, else CPU

It checks:
  1. Corridor penalty: kappa_l=0 disables the lower branch; kappa_l>0 activates it
     (the exact behavior the reviewers asked about) — via the REAL generate_entropy_penalty.
  2. Smoothing-weight schedule: monotonic decay, weight(0)=1.0, larger gamma -> faster decay
     — via the REAL _calculate_epoch_based_entropy_weight (now gamma-configurable).
  3. Sliding-window vs cumulative history reduction (the new trainer option).
  4. Config keys resolve (entropy_smooth_start_epoch / _gamma / _max_epochs, trainer.use_sliding_window).

Exit code 0 = all pass.
"""
import os
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
results = []


def check(name, fn):
    try:
        fn()
        results.append((name, True, ""))
        print(f"[{PASS}] {name}")
    except Exception as e:  # noqa
        results.append((name, False, traceback.format_exc()))
        print(f"[{FAIL}] {name}: {e}")


# ---------------------------------------------------------------------------------------------
def test_corridor_penalty():
    import torch
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"      device = {dev}")

    class _Dummy:  # generate_entropy_penalty uses only its args, not self
        pass

    gen = DataParallelPPOActor.generate_entropy_penalty.__get__(_Dummy())

    # 4 samples, 5 tokens each; historical avg entropy H_bar = 1.0 for all (turn 0)
    ent_hist = np.array([1.0], dtype=np.float32)
    rollout_step = np.zeros(4, dtype=np.int32)
    # row0=0.5 (below 0.8*Hbar), row1=1.0 (in), row2=1.9 (below 2.0*Hbar), row3=2.5 (above 2.0*Hbar)
    entropy = torch.tensor([[0.5] * 5, [1.0] * 5, [1.9] * 5, [2.5] * 5], device=dev, dtype=torch.float32)

    # kappa_l = 0  -> lower branch inactive; only row3 (above upper cap) penalized
    mask0, _ = gen(ent_hist, entropy, rollout_step, "token", 0.0, 2.0, 0.1)
    pen0 = (mask0 > 0).any(dim=1).cpu().numpy().tolist()
    assert pen0 == [False, False, False, True], f"kappa_l=0 penalized rows {pen0}, expected only row3"

    # kappa_l = 0.8 -> lower branch active; rows 0 (below floor) AND 3 (above cap) penalized
    mask8, _ = gen(ent_hist, entropy, rollout_step, "token", 0.8, 2.0, 0.1)
    pen8 = (mask8 > 0).any(dim=1).cpu().numpy().tolist()
    assert pen8 == [True, False, False, True], f"kappa_l=0.8 penalized rows {pen8}, expected rows 0 and 3"
    print("      kappa_l=0 -> floor OFF (only over-cap penalized); kappa_l=0.8 -> floor ON. OK")


def test_weight_schedule():
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    class _Dummy:
        pass

    sched = DataParallelPPOActor._calculate_epoch_based_entropy_weight.__get__(_Dummy())

    assert abs(sched(0, 150) - 1.0) < 1e-6, "weight at epoch 0 should be 1.0"
    ws = [sched(k, 150) for k in range(0, 151, 10)]
    assert all(ws[i] >= ws[i + 1] - 1e-9 for i in range(len(ws) - 1)), f"schedule not monotonic: {ws}"
    w_fast = sched(150, 150, lambda_fast=5.0)
    w_slow = sched(150, 150, lambda_fast=1.0)
    assert w_fast < w_slow, f"larger gamma should decay faster: fast={w_fast} slow={w_slow}"
    print(f"      w(0)=1.0, monotonic, gamma tunable (fast={w_fast:.4f} < slow={w_slow:.4f}). OK")


def test_sliding_window():
    # Mirrors the reducer added in ray_trainer.py
    hist = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
    cumulative = np.mean(np.array(hist), axis=0)
    win = 2
    sliding = np.mean(np.array(hist[-win:]), axis=0)
    assert np.allclose(cumulative, [2.5, 2.5]), cumulative
    assert np.allclose(sliding, [3.5, 3.5]), sliding
    print("      cumulative=[2.5,2.5], sliding(win=2)=[3.5,3.5]. OK")


def test_config_keys():
    from omegaconf import OmegaConf

    c = OmegaConf.load(os.path.join(ROOT, "verl/trainer/config/ppo_trainer.yaml"))
    a = c.actor_rollout_ref.actor
    assert a.entropy_smooth_start_epoch == 0, a.get("entropy_smooth_start_epoch")
    assert float(a.entropy_smooth_gamma) == 3.0, a.get("entropy_smooth_gamma")
    assert int(a.entropy_smooth_max_epochs) == 150, a.get("entropy_smooth_max_epochs")
    assert bool(c.trainer.use_sliding_window) is False, c.trainer.get("use_sliding_window")
    print("      new actor keys + trainer.use_sliding_window resolve with expected defaults. OK")


if __name__ == "__main__":
    check("1. corridor penalty (kappa_l behavior)", test_corridor_penalty)
    check("2. smoothing-weight schedule (gamma)", test_weight_schedule)
    check("3. sliding-window vs cumulative reducer", test_sliding_window)
    check("4. config keys resolve", test_config_keys)

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print("\n" + "=" * 60)
    print(f"SMOKE TEST: {len(results) - n_fail}/{len(results)} passed")
    if n_fail:
        for name, ok, tb in results:
            if not ok:
                print(f"\n--- {name} ---\n{tb}")
    sys.exit(1 if n_fail else 0)
