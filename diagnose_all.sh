#!/usr/bin/env bash
# diagnose_all.sh - AutoScaffold Full Subsystem & Connectivity Diagnostic Runner
set -e

# Resolve script directory and source shared environment
if [ -n "$AUTOSCAFFOLD_HOME" ] && [ -f "$AUTOSCAFFOLD_HOME/diagnose_all.sh" ]; then
    SCRIPT_DIR="$AUTOSCAFFOLD_HOME"
else
    SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
fi

if [ -f "$SCRIPT_DIR/scripts/env_setup.sh" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/scripts/env_setup.sh"
fi

# Locate best available Python interpreter for running diagnostics
DIAG_PY=""
if [ -n "${PYTHON_BIN:-}" ] && [ -x "$PYTHON_BIN" ]; then
    DIAG_PY="$PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1; then
    DIAG_PY="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    DIAG_PY="$(command -v python)"
else
    DIAG_PY="python"
fi

"$DIAG_PY" "$SCRIPT_DIR/scripts/diagnose_environment.py"
