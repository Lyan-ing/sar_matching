#!/bin/bash
set -euo pipefail
cd /home/yl/matching/sar_matching
conda run --no-capture-output -n match python -B SAR_SP_LG_Matchability_20261014_2031/code/control.py stop "${1:?Specify joint, control or frozen}"
