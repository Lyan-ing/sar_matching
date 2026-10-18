#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
touch runs/formal/STOP_REQUESTED
echo 'Stop requested for this experiment only; training saves last state after the current step.'
