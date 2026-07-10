#!/bin/bash
# Sourced by all EPO rebuttal jobs. Sets paths (on /ai4rl, compute-node-only lustre),
# CUDA, and clears the stale inherited X2P proxy (direct internet works on compute nodes).

# --- clear stale proxy (inherited port is per-host; direct works on compute) ---
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY X2P_PROXY_URL
export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"

# --- CUDA ---
export MODULEPATH=/public/modulefiles:$MODULEPATH
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.4.1 2>/dev/null || true
export CUDA_HOME=/public/apps/cuda/12.4.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# --- EPO storage base (petabyte lustre, visible on compute nodes only) ---
export EPO_BASE=/ai4rl/fsx/impwxu/epo
export MAMBA_ROOT_PREFIX=$EPO_BASE/micromamba
export EPO_ENV=epo
export SCI_MODEL=$EPO_BASE/models/Qwen2.5-7B-Instruct
# ScienceWorld is JVM-backed (py4j); conda openjdk lives in the env prefix.
export JAVA_HOME=$MAMBA_ROOT_PREFIX/envs/$EPO_ENV
export PATH=$JAVA_HOME/bin:$PATH
export TMPDIR=$EPO_BASE/tmp
export PIP_CACHE_DIR=$EPO_BASE/pip-cache
export HF_HOME=$EPO_BASE/hf-cache
export WANDB_DIR=$EPO_BASE/wandb

# --- repo path (FSx home; same absolute path on login + compute) ---
export EPO_REPO=/storage/home/impwxu/code/EPO

# --- W&B credentials (key stored outside the repo, never committed) ---
if [ -f /storage/home/impwxu/.wandb_key ]; then
  export WANDB_API_KEY=$(cat /storage/home/impwxu/.wandb_key)
fi
export WANDB_ENTITY=ruwujiang-rutgers-university

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME" "$WANDB_DIR" "$EPO_BASE/models" 2>/dev/null || true

# helper: run a command inside the epo micromamba env
epo_run() { micromamba run -n "$EPO_ENV" "$@"; }
