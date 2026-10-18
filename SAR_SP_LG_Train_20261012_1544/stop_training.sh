#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
/home/yl/miniconda3/bin/conda run --no-capture-output -n match python -B "$root/code/control.py" stop "$@"
