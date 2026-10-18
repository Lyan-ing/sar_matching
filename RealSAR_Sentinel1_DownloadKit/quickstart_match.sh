#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
conda run --no-capture-output -n match python -m pip install -r requirements.txt
conda run --no-capture-output -n match python -m pytest -q tests
conda run --no-capture-output -n match python download_s1_rtc.py --tile 14TPN --years 2020 --discover-only
conda run --no-capture-output -n match python download_s1_rtc.py --tile 14TPN --years 2020 --n 2 --size 512 --out output/sentinel1_14TPN_2020_two_acquisitions
