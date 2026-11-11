#!/usr/bin/env bash
set -euo pipefail

BASE="/tmp/drone-swarm-active-uavs-1000"

RUN="${1:-}"

if [[ -z "$RUN" ]]; then
    RUN="$(
        find "$BASE" -mindepth 1 -maxdepth 1 -type d -name 'run-*' \
            -printf '%T@ %p\n' 2>/dev/null |
        sort -nr |
        head -1 |
        cut -d' ' -f2-
    )"
fi

if [[ -z "$RUN" || ! -d "$RUN" ]]; then
    echo "ERROR: No simulation run directory found." >&2
    exit 1
fi

RUNTIME="$RUN/runtime"
HISTORY="$RUNTIME/coordinator_navigation_history.json"

printf 'RUN=%q\n' "$RUN"
printf 'RUNTIME=%q\n' "$RUNTIME"
printf 'NAVIGATION_HISTORY=%q\n' "$HISTORY"
