#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
case "${1:-}" in t1|t2|t3|repeat) ;; *) echo 'usage: bash stop_training.sh {t1|t2|t3|repeat}'; exit 2;; esac
[ -d "runs/$1" ] || { echo 'Run not started'; exit 2; }
touch "runs/$1/STOP_REQUESTED"
echo 'Safe stop requested; current optimizer step completes and last checkpoint is saved. Workflow will not start following stages after an incomplete run.'
