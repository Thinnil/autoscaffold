#!/usr/bin/env bash
# Common environment loader and utility functions for autoscaffold

# Color definitions
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

# Resolve root script directory
if [ -z "$SCRIPT_DIR" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

# Load user configuration if present
if [ -f "$SCRIPT_DIR/config.env" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/config.env"
elif [ -f "$SCRIPT_DIR/config.env.example" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/config.env.example"
fi

# Apply fallbacks
RFDIFF_DIR="${RFDIFF_DIR:-$HOME/programs/RFdiffusion}"
MPNN_DIR="${MPNN_DIR:-$HOME/programs/ProteinMPNN}"
LOCAL_TOOLS_DIR="${LOCAL_TOOLS_DIR:-$SCRIPT_DIR/tools}"
LOCAL_ENVS_DIR="${LOCAL_ENVS_DIR:-$SCRIPT_DIR/envs}"
CONDA_ENV_RFDIFF="${CONDA_ENV_RFDIFF:-SE3nv}"
CONDA_ENV_MPNN="${CONDA_ENV_MPNN:-mlfold}"
CONDA_ENV_COLABFOLD="${CONDA_ENV_COLABFOLD:-colabfold}"
AUTOSCAFFOLD_REPO_URL="${AUTOSCAFFOLD_REPO_URL:-https://github.com/Thinnil/autoscaffold.git}"
AUTOSCAFFOLD_BRANCH="${AUTOSCAFFOLD_BRANCH:-main}"

# Check for local installations if default directories do not exist
if [ ! -d "$RFDIFF_DIR" ] && [ -d "$LOCAL_TOOLS_DIR/RFdiffusion" ]; then
    RFDIFF_DIR="$LOCAL_TOOLS_DIR/RFdiffusion"
fi

if [ ! -d "$MPNN_DIR" ] && [ -d "$LOCAL_TOOLS_DIR/ProteinMPNN" ]; then
    MPNN_DIR="$LOCAL_TOOLS_DIR/ProteinMPNN"
fi

# Hardware and Thread Optimization for DL inference (RFdiffusion, ProteinMPNN, ColabFold)
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [ -z "$OMP_NUM_THREADS" ]; then
    if command -v nproc >/dev/null 2>&1; then
        _CORES=$(nproc 2>/dev/null || echo 4)
        if [ "$_CORES" -gt 8 ]; then
            export OMP_NUM_THREADS=8
        else
            export OMP_NUM_THREADS="$_CORES"
        fi
    else
        export OMP_NUM_THREADS=4
    fi
fi
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$OMP_NUM_THREADS}"

# Function to detect conda / mamba / micromamba binary (cached)
_CACHED_CONDA_BIN=""
find_conda_bin() {
    if [ -n "$_CACHED_CONDA_BIN" ] && [ -x "$_CACHED_CONDA_BIN" ]; then
        echo "$_CACHED_CONDA_BIN"
        return 0
    fi

    if [ -x "$LOCAL_ENVS_DIR/bin/micromamba" ]; then
        _CACHED_CONDA_BIN="$LOCAL_ENVS_DIR/bin/micromamba"
    elif [ -x "$LOCAL_ENVS_DIR/micromamba" ]; then
        _CACHED_CONDA_BIN="$LOCAL_ENVS_DIR/micromamba"
    elif command -v micromamba >/dev/null 2>&1; then
        _CACHED_CONDA_BIN="$(command -v micromamba)"
    elif command -v mamba >/dev/null 2>&1; then
        _CACHED_CONDA_BIN="$(command -v mamba)"
    elif command -v conda >/dev/null 2>&1; then
        _CACHED_CONDA_BIN="$(command -v conda)"
    else
        _CACHED_CONDA_BIN=""
    fi
    echo "$_CACHED_CONDA_BIN"
}

# Function to activate a named environment or local prefix with session caching
_CONDA_SHELL_INITIALIZED=""
activate_env() {
    local env_name="$1"

    # Validate env_name format to prevent argument or command injection
    if [[ ! "$env_name" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
        echo -e "${RED}[ERROR] Invalid environment name: '$env_name'${NC}" >&2
        return 1
    fi

    # Environment activation caching: Check if already active in current shell
    if [ "$CONDA_DEFAULT_ENV" = "$env_name" ]; then
        return 0
    fi
    if [ -n "$CONDA_PREFIX" ]; then
        if [ "$CONDA_PREFIX" = "$LOCAL_ENVS_DIR/$env_name" ] || [ "${CONDA_PREFIX##*/}" = "$env_name" ]; then
            return 0
        fi
    fi

    local conda_bin
    conda_bin=$(find_conda_bin)

    # Initialize shell hook once per shell session
    if [ -z "$_CONDA_SHELL_INITIALIZED" ]; then
        if [ -n "$conda_bin" ] && [ -x "$conda_bin" ] && [[ "$conda_bin" == *"micromamba"* ]]; then
            eval "$("$conda_bin" shell hook -s bash)"
            _CONDA_SHELL_INITIALIZED=1
        else
            local conda_base
            conda_base=$("$conda_bin" info --base 2>/dev/null || conda info --base 2>/dev/null)
            if [ -n "$conda_base" ] && [ -f "$conda_base/etc/profile.d/conda.sh" ]; then
                # shellcheck disable=SC1091
                source "$conda_base/etc/profile.d/conda.sh"
                _CONDA_SHELL_INITIALIZED=1
            fi
        fi
    fi

    # 1. Check if local environment prefix exists in LOCAL_ENVS_DIR
    if [ -d "$LOCAL_ENVS_DIR/$env_name" ]; then
        if [ -n "$conda_bin" ] && [[ "$conda_bin" == *"micromamba"* ]]; then
            micromamba activate "$LOCAL_ENVS_DIR/$env_name"
            return $?
        elif command -v conda >/dev/null 2>&1; then
            conda activate "$LOCAL_ENVS_DIR/$env_name"
            return $?
        fi
    fi

    # 2. Check system / named conda environment
    if [ -n "$conda_bin" ] && [[ "$conda_bin" == *"micromamba"* ]]; then
        micromamba activate "$env_name"
        return $?
    elif command -v conda >/dev/null 2>&1; then
        conda activate "$env_name"
        return $?
    fi

    echo -e "${YELLOW}[WARN] Could not find conda environment hook. Attempting fallback 'conda activate $env_name'...${NC}"
    conda activate "$env_name" 2>/dev/null || true
}

# Function to run python code using environment or system python
run_in_env() {
    local env_name="$1"
    shift
    activate_env "$env_name"
    python "$@"
}
