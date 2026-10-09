#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PIN=eb42fee2d71449efb0aa5c10549752b5d75384d8
if [ ! -d cache/LightGlue ]; then
  mkdir -p cache
  git clone https://github.com/cvg/LightGlue.git cache/LightGlue
  git -C cache/LightGlue checkout "$PIN"
fi
[ "$(git -C cache/LightGlue rev-parse HEAD)" = "$PIN" ] || { printf 'Existing LightGlue checkout differs from pinned commit.\n' >&2; exit 1; }
# Metadata only: official network implementation remains unchanged.
sed -i 's/^opencv-python$/opencv-python-headless/' cache/LightGlue/requirements.txt
conda run --no-capture-output -n match python -m pip install --no-deps -e cache/LightGlue
mkdir -p cache/torch/hub/checkpoints
fetch_weight() {
  local filename="$1" url="$2" expected="$3" target="cache/torch/hub/checkpoints/$1"
  if [ ! -f "$target" ]; then
    curl --fail --location --retry 2 "$url" -o "$target.partial.$$"
    printf '%s  %s\n' "$expected" "$target.partial.$$" | sha256sum --check
    mv "$target.partial.$$" "$target"
  fi
  printf '%s  %s\n' "$expected" "$target" | sha256sum --check
}
fetch_weight superpoint_v1.pth https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth 52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e
fetch_weight superpoint_lightglue_v0-1_arxiv.pth https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth 6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a
conda run --no-capture-output -n match python -m pip check
