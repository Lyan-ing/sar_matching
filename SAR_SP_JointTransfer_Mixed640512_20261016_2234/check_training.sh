#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
conda run --no-capture-output -n match python -B code/check_state.py
