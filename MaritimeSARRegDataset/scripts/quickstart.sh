#!/usr/bin/env bash
set -euo pipefail
python -m sarreg demo --out data/demo --count 24
python -m sarreg prepare --dataset hrsid --root data/demo --out data/index/demo.jsonl
python -m sarreg generate --index data/index/demo.jsonl --split all --count 8 --config configs/easy.yaml --out output/demo
python -m sarreg validate --root output/demo
python -m sarreg visualize --root output/demo --out output/previews --count 8
printf '\nThis was a synthetic SOFTWARE demo. Replace data/demo with downloaded SAR data.\n'
