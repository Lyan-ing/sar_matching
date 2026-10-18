#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
if [ -d runs ] || compgen -G 'formal_code_freeze_*.json' >/dev/null; then
  echo 'Refusing to overwrite an existing experiment. Reproduce in a NEW sibling experiment directory; resume only an explicitly stopped run.' >&2
  exit 2
fi
exec conda run --no-capture-output -n match python -B code/workflow.py
