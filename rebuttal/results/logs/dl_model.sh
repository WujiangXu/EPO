#!/bin/bash
set -x
export MAMBA_ROOT_PREFIX=$HOME/micromamba
micromamba create -y -n hfget -c conda-forge python=3.10 pip || exit 10
micromamba run -n hfget pip install "huggingface_hub[cli]" || exit 20
micromamba run -n hfget huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir $HOME/models/Qwen2.5-7B-Instruct || exit 30
echo "===== [$(date)] MODEL DOWNLOAD COMPLETE ====="
