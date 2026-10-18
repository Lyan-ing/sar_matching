#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=1
conda run --no-capture-output -n match python -B code/keep_mechanism.py
conda run --no-capture-output -n match python -B code/keep_train.py
