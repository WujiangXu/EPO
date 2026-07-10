#!/usr/bin/env python3
"""
Rebuttal analyses from existing W&B logs — NO new training required.

Produces the three log-based figures referenced in EXPERIMENTS.md (Exp 2):
  2a. Entropy-oscillation  vs. final-success correlation        -> Theme C (causality)
  2b. Importance-ratio / clip-fraction, baseline vs +EPO        -> Theme D (ratio variance)
  2c. Entropy trajectory with the corridor band [kl*H, kr*H]    -> Theme A (does the floor bind?)

Usage
-----
  pip install wandb pandas numpy matplotlib scipy
  wandb login
  python analysis/wandb_analysis.py \
      --entity ruwujiang-rutgers-university \
      --project verl_agent_sciworld_ppo \
      --epo-substr es --baseline-substr ec0.001 \
      --kappa-l 0 --kappa-r 2.0 \
      --outdir analysis/figs

Notes
-----
* Metric key names differ across verl versions. Defaults below cover the common ones and the script
  auto-falls-back to any column matching the given regexes; pass --*-key to override explicitly.
* "EPO vs baseline" grouping is by substring on the run name (--epo-substr / --baseline-substr).
  Inspect your dashboard and adjust. With no baseline substring, 2a still runs over all runs.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- metric-key resolution ---------------------------------------------
# Ordered candidates; first column that exists (exact, then regex) is used.
# Verified against this codebase's logging (verl/trainer/ppo/ray_trainer.py, workers/actor/dp_actor.py):
#   mean entropy -> actor/entropy_loss  (there is no "actor/entropy" key)
#   clip frac    -> actor/pg_clipfrac
#   ratio var    -> NOT logged; actor/ppo_kl is the closest proxy
#   reward       -> critic/score/mean
#   success      -> val/success_rate, split-prefixed as val_l0 (IID) / val_l1 (OOD) or val_iid/val_ood
#   per-turn ent -> step_entropy_<i>  (top-level)
ENTROPY_KEYS   = ["actor/entropy_loss", "actor/entropy", "entropy", "actor/entropys"]
CLIPFRAC_KEYS  = ["actor/pg_clipfrac", "actor/clipfrac", "actor/pg_clipfrac_lower"]
RATIOVAR_KEYS  = ["actor/ratio_var", "actor/importance_ratio_var", "actor/ppo_ratio_var", "actor/ppo_kl"]
REWARD_KEYS    = ["critic/score/mean", "critic/rewards/mean", "episode_rewards_mean", "reward/mean"]
SUCCESS_KEYS   = ["val_l0/success_rate", "val_iid/success_rate", "val/success_rate",
                  "val_l1/success_rate", "val_ood/success_rate",
                  "val_iid_success_rate", "val_l0_success_rate",
                  "val/test_score/success_rate", "success_rate"]
STEP_ENT_RE    = re.compile(r"^step_entropy_\d+$")


def pick_key(df_cols, candidates, regexes=()):
    for c in candidates:
        if c in df_cols:
            return c
    for rx in regexes:
        rx = re.compile(rx) if isinstance(rx, str) else rx
        for c in df_cols:
            if rx.match(c):
                return c
    return None


def get_runs(entity, project, name_filter=None):
    import wandb
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    out = []
    for r in runs:
        if name_filter and name_filter not in r.name:
            continue
        out.append(r)
    return out


def run_history(run, keys):
    """Return a DataFrame with the requested keys (+ _step) using scan_history (full resolution)."""
    keep = [k for k in keys if k]
    try:
        df = run.history(keys=keep, pandas=True, samples=100000)
    except Exception:
        rows = list(run.scan_history(keys=keep + ["_step"]))
        df = pd.DataFrame(rows)
    return df


def is_epo(name, epo_substr, baseline_substr):
    if epo_substr and epo_substr in name:
        return True
    if baseline_substr and baseline_substr in name:
        return False
    return None  # unknown


# ----------------------------- 2a: oscillation vs success ----------------------------------------
def analysis_oscillation(runs, outdir, epo_substr, baseline_substr):
    rows = []
    for run in runs:
        df = run_history(run, ENTROPY_KEYS + SUCCESS_KEYS + [f"step_entropy_{i}" for i in range(60)])
        if df is None or df.empty:
            continue
        ent_key = pick_key(df.columns, ENTROPY_KEYS)
        suc_key = pick_key(df.columns, SUCCESS_KEYS, regexes=[r".*success.*"])
        step_ent_cols = [c for c in df.columns if STEP_ENT_RE.match(c)]

        # Oscillation metric: std of step-to-step change in the mean-entropy training series.
        osc = np.nan
        if ent_key and df[ent_key].notna().sum() > 3:
            series = df[ent_key].dropna().to_numpy(dtype=float)
            osc = float(np.nanstd(np.diff(series)))
        elif step_ent_cols:
            # Fallback: within-trajectory across-turn spread, averaged over training.
            osc = float(np.nanmean(df[step_ent_cols].std(axis=1)))

        succ = np.nan
        if suc_key and df[suc_key].notna().sum() > 0:
            succ = float(np.nanmax(df[suc_key].to_numpy(dtype=float)))

        if not (np.isnan(osc) or np.isnan(succ)):
            rows.append(dict(run=run.name, oscillation=osc, final_success=succ,
                             group={True: "EPO", False: "baseline"}.get(
                                 is_epo(run.name, epo_substr, baseline_substr), "other")))
    if not rows:
        print("[2a] no runs with both entropy and success metrics — check --*-key / substrings.")
        return
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(outdir, "oscillation_vs_success.csv"), index=False)

    from scipy.stats import pearsonr, spearmanr
    r_p, p_p = pearsonr(d.oscillation, d.final_success)
    r_s, p_s = spearmanr(d.oscillation, d.final_success)

    plt.figure(figsize=(5, 4))
    for g, sub in d.groupby("group"):
        plt.scatter(sub.oscillation, sub.final_success, label=g, alpha=0.8)
    plt.xlabel("entropy oscillation  (std of ΔH over training)")
    plt.ylabel("final success rate")
    plt.title(f"Oscillation vs. success\nPearson r={r_p:.2f} (p={p_p:.1e}), Spearman ρ={r_s:.2f}")
    plt.legend()
    plt.tight_layout()
    fp = os.path.join(outdir, "2a_oscillation_vs_success.png")
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f"[2a] n={len(d)}  Pearson r={r_p:.3f} (p={p_p:.2e})  Spearman ρ={r_s:.3f}  -> {fp}")


# ----------------------------- 2b: clip-fraction / ratio variance ---------------------------------
def _mean_curve(dfs, key):
    aligned = []
    for df in dfs:
        if key in df.columns:
            s = df[[key]].dropna().reset_index(drop=True)[key]
            aligned.append(s)
    if not aligned:
        return None, None
    L = min(len(s) for s in aligned)
    arr = np.stack([s.to_numpy(dtype=float)[:L] for s in aligned], axis=0)
    return arr.mean(0), arr.std(0)


def analysis_variance(runs, outdir, epo_substr, baseline_substr):
    groups = {"baseline": [], "EPO": []}
    for run in runs:
        g = is_epo(run.name, epo_substr, baseline_substr)
        if g is None:
            continue
        df = run_history(run, CLIPFRAC_KEYS + RATIOVAR_KEYS)
        if df is not None and not df.empty:
            groups["EPO" if g else "baseline"].append(df)
    if not groups["baseline"] and not groups["EPO"]:
        print("[2b] no grouped runs — set --epo-substr / --baseline-substr. Skipping.")
        return

    for metric_name, cands in [("clip_fraction", CLIPFRAC_KEYS), ("ratio_variance", RATIOVAR_KEYS)]:
        plotted = False
        plt.figure(figsize=(5, 4))
        for gname, dfs in groups.items():
            if not dfs:
                continue
            key = pick_key(set().union(*[set(df.columns) for df in dfs]), cands)
            if key is None:
                continue
            mean, std = _mean_curve(dfs, key)
            if mean is None:
                continue
            x = np.arange(len(mean))
            plt.plot(x, mean, label=f"{gname} (n={len(dfs)})")
            plt.fill_between(x, mean - std, mean + std, alpha=0.2)
            plotted = True
        if plotted:
            plt.xlabel("RL step")
            plt.ylabel(metric_name)
            plt.title(f"{metric_name}: baseline vs. +EPO")
            plt.legend()
            plt.tight_layout()
            fp = os.path.join(outdir, f"2b_{metric_name}.png")
            plt.savefig(fp, dpi=150)
            print(f"[2b] wrote {fp}")
        plt.close()
    if not any(RATIOVAR_KEYS[0] in df.columns for dfs in groups.values() for df in dfs):
        print("[2b] note: importance-ratio variance not logged; clip-fraction is the usable proxy.")


# ----------------------------- 2c: entropy vs corridor band ---------------------------------------
def analysis_corridor(runs, outdir, kappa_l, kappa_r, epo_substr):
    epo_runs = [r for r in runs if (not epo_substr) or (epo_substr in r.name)]
    if not epo_runs:
        print("[2c] no EPO runs matched; skipping.")
        return
    run = epo_runs[0]
    df = run_history(run, ENTROPY_KEYS)
    ent_key = pick_key(df.columns if df is not None else [], ENTROPY_KEYS)
    if ent_key is None or df.empty:
        print("[2c] entropy metric not found; skipping.")
        return
    ent = df[ent_key].dropna().to_numpy(dtype=float)
    hbar = np.array([ent[: i + 1].mean() for i in range(len(ent))])  # cumulative running mean H̄
    x = np.arange(len(ent))

    plt.figure(figsize=(6, 4))
    plt.plot(x, ent, label="mean token entropy H", color="tab:blue")
    plt.plot(x, hbar, label="running mean H̄", color="black", ls="--", lw=1)
    plt.fill_between(x, kappa_l * hbar, kappa_r * hbar, color="tab:green", alpha=0.15,
                     label=f"corridor [{kappa_l}·H̄, {kappa_r}·H̄]")
    below = ent < (kappa_l * hbar)
    plt.scatter(x[below], ent[below], color="red", s=12, zorder=5,
                label=f"below floor ({below.sum()} steps)")
    plt.xlabel("RL step")
    plt.ylabel("entropy")
    plt.title(f"Entropy vs. corridor — {run.name}\nfloor binds {below.sum()}/{len(ent)} steps")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fp = os.path.join(outdir, "2c_entropy_vs_corridor.png")
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f"[2c] floor binds at {below.sum()}/{len(ent)} steps  -> {fp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name-filter", default=None, help="only runs whose name contains this")
    ap.add_argument("--epo-substr", default="es", help="substring marking EPO runs")
    ap.add_argument("--baseline-substr", default="ec0", help="substring marking baseline runs")
    ap.add_argument("--kappa-l", type=float, default=0.0)
    ap.add_argument("--kappa-r", type=float, default=2.0)
    ap.add_argument("--outdir", default="analysis/figs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Fetching runs from {args.entity}/{args.project} ...")
    runs = get_runs(args.entity, args.project, args.name_filter)
    print(f"  {len(runs)} runs found.")
    if not runs:
        sys.exit("No runs — check --entity/--project and `wandb login`.")

    analysis_oscillation(runs, args.outdir, args.epo_substr, args.baseline_substr)
    analysis_variance(runs, args.outdir, args.epo_substr, args.baseline_substr)
    analysis_corridor(runs, args.outdir, args.kappa_l, args.kappa_r, args.epo_substr)
    print("Done. Figures in", args.outdir)


if __name__ == "__main__":
    main()
