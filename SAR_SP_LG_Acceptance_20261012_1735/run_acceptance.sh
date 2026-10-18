#!/usr/bin/env bash
# Safe completion guard: query first; never repeat a completed formal run.
set -euo pipefail
SAR_ACCEPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-status}" in
 status)
  for kind in formal validation real; do cat "$SAR_ACCEPT_DIR/${kind}_status.json"; done
  ;;
 formal|diagnostics)
  phase="$1"
  statefile="$SAR_ACCEPT_DIR/formal_status.json"
  physical=0
  program=run_formal.py
  if [[ "$phase" == diagnostics ]]; then statefile="$SAR_ACCEPT_DIR/real_status.json"; physical=2; program=run_diagnostics.py; fi
  if [[ -f "$statefile" ]] && conda run --no-capture-output -n match python -B -c 'import json,sys;sys.exit(0 if json.load(open(sys.argv[1])).get("state")=="completed" else 1)' "$statefile"; then
   printf '%s completed; protected results retained. No inference restarted.\n' "$phase"
   exit 0
  fi
  CUDA_VISIBLE_DEVICES="$physical" conda run --no-capture-output -n match python -B "$SAR_ACCEPT_DIR/code/$program" --gpu "$physical"
  ;;
 *) printf 'Usage: bash run_acceptance.sh [status|formal|diagnostics]\n' >&2; exit 2 ;;
esac
