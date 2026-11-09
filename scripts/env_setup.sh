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
if [ -n "$AUTOSCAFFOLD_HOME" ] && [ -f "$AUTOSCAFFOLD_HOME/scripts/env_setup.sh" ]; then
    SCRIPT_DIR="$AUTOSCAFFOLD_HOME"
elif [ -z "$SCRIPT_DIR" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
if [ -z "$SCRIPT_DIR" ] || [ "$SCRIPT_DIR" = "/" ] || [ "$SCRIPT_DIR" = "\\" ] || [[ "$SCRIPT_DIR" =~ ^[a-zA-Z]:[\\/]?$ ]]; then
    echo -e "\033[0;31m[ERROR] Dangerous or invalid SCRIPT_DIR in env_setup.sh: '$SCRIPT_DIR'\033[0m" >&2
    return 1 2>/dev/null || exit 1
fi

# Security sanitization helper to reject shell injection and directory traversal
_is_unsafe_path() {
    local p="$1"
    [ -z "$p" ] && return 0
    case "$p" in
        *[\;\&\|\`\>\<\$]*|*..*|*[$'\n'$'\r']*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Load user configuration if present
if [ -f "$SCRIPT_DIR/config.env" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/config.env"
elif [ -f "$SCRIPT_DIR/config.env.example" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/config.env.example"
fi

# Version Single Source of Truth
if [ -z "${AUTOSCAFFOLD_VERSION:-}" ]; then
    AUTOSCAFFOLD_VERSION=$(sed -n 's/^__version__ = ["'\''\([0-9.]*\)["'\''].*$/\1/p' "$SCRIPT_DIR/scripts/version.py" 2>/dev/null || true)
    [ -z "$AUTOSCAFFOLD_VERSION" ] && AUTOSCAFFOLD_VERSION=$(python3 -c "import sys, os; sys.path.insert(0, os.path.join(sys.argv[1], 'scripts')); from version import __version__; print(__version__)" "$SCRIPT_DIR" 2>/dev/null || python -c "import sys, os; sys.path.insert(0, os.path.join(sys.argv[1], 'scripts')); from version import __version__; print(__version__)" "$SCRIPT_DIR" 2>/dev/null || echo "2.2.0")
    export AUTOSCAFFOLD_VERSION
fi

# Discover and export general Python interpreter with capability-aware multi-tier resolution
if [ -z "${PYTHON_BIN:-}" ] || [ ! -x "${PYTHON_BIN:-}" ]; then
    _RESOLVED_PYTHON=""

    _probe_py() {
        local py="$1"
        local req_np="${2:-0}"
        if [ -z "$py" ] || [ ! -f "$py" ] || [ ! -x "$py" ]; then
            return 1
        fi
        # Strictly reject paths containing shell injection characters or directory traversal
        if _is_unsafe_path "$py"; then
            return 1
        fi
        # Enforce that candidate executable is a legitimate Python binary
        local base_py
        base_py="$(basename "$py")"
        if [[ ! "$base_py" =~ ^[pP][yY][tT][hH][oO][nN]([0-9]+(\.[0-9]+)?)?(\.[eE][xX][eE])?$ ]]; then
            return 1
        fi
        if [ "$req_np" -eq 1 ]; then
            "$py" -c "import sys, dataclasses, numpy; sys.exit(0 if sys.version_info >= (3, 8) else 1)" >/dev/null 2>&1
        else
            "$py" -c "import sys, dataclasses; sys.exit(0 if sys.version_info >= (3, 8) else 1)" >/dev/null 2>&1
        fi
    }

    # Tier 1: Check active virtual environment
    if [ -n "${CONDA_PREFIX:-}" ]; then
        for cand in "$CONDA_PREFIX/bin/python" "$CONDA_PREFIX/python.exe" "$CONDA_PREFIX/python"; do
            if _probe_py "$cand" 1; then
                _RESOLVED_PYTHON="$cand"
                break
            fi
        done
    fi

    # Tier 2: Check system python3 / python (>= 3.8 with dataclasses and numpy)
    if [ -z "$_RESOLVED_PYTHON" ]; then
        for sys_cmd in python3 python; do
            if command -v "$sys_cmd" >/dev/null 2>&1; then
                _SYS_CAND="$(command -v "$sys_cmd")"
                # Skip WindowsApps redirector stubs on Windows which lack scientific libraries
                if [[ "$_SYS_CAND" == *"WindowsApps"* ]]; then
                    continue
                fi
                if _probe_py "$_SYS_CAND" 1; then
                    _RESOLVED_PYTHON="$_SYS_CAND"
                    break
                fi
            fi
        done
    fi

    # Tier 2b: Fallback if neither has numpy, check if either has dataclasses and >= 3.8
    if [ -z "$_RESOLVED_PYTHON" ]; then
        for sys_cmd in python3 python; do
            if command -v "$sys_cmd" >/dev/null 2>&1; then
                _SYS_CAND="$(command -v "$sys_cmd")"
                if _probe_py "$_SYS_CAND" 0; then
                    _RESOLVED_PYTHON="$_SYS_CAND"
                    break
                fi
            fi
        done
    fi

    # Tier 3: Scan standard Conda/Micromamba root directories for Python >= 3.8 with dataclasses and numpy
    if [ -z "$_RESOLVED_PYTHON" ]; then
        _SCAN_ROOTS=(
            "${LOCAL_ENVS_DIR:-}"
            "$HOME/.micromamba/envs"
            "$HOME/micromamba/envs"
            "$HOME/.local/share/mamba/envs"
            "$HOME/miniconda3/envs"
            "$HOME/miniforge3/envs"
            "$HOME/mambaforge/envs"
            "$HOME/anaconda3/envs"
            "/opt/conda/envs"
            "/usr/local/miniconda3/envs"
        )
        for s_root in "${_SCAN_ROOTS[@]}"; do
            [ -z "$s_root" ] || [ ! -d "$s_root" ] && continue
            if _is_unsafe_path "$s_root"; then
                continue
            fi
            for env_dir in "$s_root"/*; do
                [ -z "$env_dir" ] || [ ! -d "$env_dir" ] && continue
                if _is_unsafe_path "$env_dir"; then
                    continue
                fi
                for cand in "$env_dir/bin/python" "$env_dir/python.exe" "$env_dir/python"; do
                    [ -z "$cand" ] || [ ! -f "$cand" ] || [ ! -x "$cand" ] && continue
                    if _probe_py "$cand" 1; then
                        _RESOLVED_PYTHON="$cand"
                        break 3
                    fi
                done
            done
        done
    fi

    # Tier 4: Tool-specific Python fallbacks
    if [ -z "$_RESOLVED_PYTHON" ]; then
        if [ -n "${PYTHON_RFDIFF:-}" ] && [ -x "$PYTHON_RFDIFF" ]; then
            _RESOLVED_PYTHON="$PYTHON_RFDIFF"
        elif [ -n "${PYTHON_MPNN:-}" ] && [ -x "$PYTHON_MPNN" ]; then
            _RESOLVED_PYTHON="$PYTHON_MPNN"
        fi
    fi

    # Tier 5: Host fallback (python3 or python)
    if [ -z "$_RESOLVED_PYTHON" ]; then
        if command -v python3 >/dev/null 2>&1; then
            _RESOLVED_PYTHON="$(command -v python3)"
        elif command -v python >/dev/null 2>&1; then
            _RESOLVED_PYTHON="$(command -v python)"
        else
            _RESOLVED_PYTHON="python3"
        fi
    fi

    PYTHON_BIN="$_RESOLVED_PYTHON"
fi
export PYTHON_BIN

check_git_sync_status() {
    if [ -d "$SCRIPT_DIR/.git" ] && command -v git >/dev/null 2>&1; then
        local local_sha upstream_sha
        local_sha=$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo "")
        upstream_sha=$(git -C "$SCRIPT_DIR" rev-parse origin/main 2>/dev/null || echo "")
        if [ -n "$local_sha" ] && [ -n "$upstream_sha" ]; then
            if [ "$local_sha" = "$upstream_sha" ]; then
                echo "Up-to-date with origin/main"
            else
                echo "Update Available (run Option [7] to update)"
            fi
            return 0
        fi
    fi
    echo "Local Installation (v$AUTOSCAFFOLD_VERSION)"
}

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

# Function to detect micromamba binary across system and user paths
find_micromamba_bin() {
    if command -v micromamba >/dev/null 2>&1; then
        command -v micromamba
        return 0
    fi
    for cand in \
        "$LOCAL_ENVS_DIR/bin/micromamba" \
        "$LOCAL_ENVS_DIR/micromamba" \
        "$HOME/.local/bin/micromamba" \
        "$HOME/bin/micromamba" \
        "$HOME/.micromamba/bin/micromamba" \
        "$HOME/micromamba/micromamba" \
        "/usr/local/bin/micromamba" \
        "/opt/conda/bin/micromamba"; do
        if [ -n "$cand" ] && [ -f "$cand" ] && [ -x "$cand" ]; then
            if ! _is_unsafe_path "$cand"; then
                echo "$cand"
                return 0
            fi
        fi
    done
    return 1
}

# Function to detect conda / mamba / micromamba binary (cached)
_CACHED_CONDA_BIN=""
find_conda_bin() {
    if [ -n "$_CACHED_CONDA_BIN" ] && [ -x "$_CACHED_CONDA_BIN" ]; then
        echo "$_CACHED_CONDA_BIN"
        return 0
    fi

    local mm_bin
    mm_bin="$(find_micromamba_bin 2>/dev/null || true)"
    if [ -n "$mm_bin" ] && [ -x "$mm_bin" ]; then
        _CACHED_CONDA_BIN="$mm_bin"
    elif command -v mamba >/dev/null 2>&1; then
        _CACHED_CONDA_BIN="$(command -v mamba)"
    elif command -v conda >/dev/null 2>&1; then
        _CACHED_CONDA_BIN="$(command -v conda)"
    else
        for cand in \
            "$CONDA_EXE" \
            "$CONDA_PREFIX/bin/conda" \
            "$HOME/miniconda3/bin/conda" \
            "$HOME/miniforge3/bin/conda" \
            "$HOME/mambaforge/bin/conda" \
            "$HOME/anaconda3/bin/conda" \
            "/opt/conda/bin/conda" \
            "/usr/local/miniconda3/bin/conda"; do
            if [ -n "$cand" ] && [ -f "$cand" ] && [ -x "$cand" ]; then
                if ! _is_unsafe_path "$cand"; then
                    _CACHED_CONDA_BIN="$cand"
                    break
                fi
            fi
        done
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
    "${PYTHON_BIN:-python}" "$@"
}

# Function to discover python executable for a target conda environment across standard roots
_CACHED_CONDA_ENVS_INFO=""

discover_conda_python() {
    local target_env="$1"
    local req_module="${2:-}"
    local enable_deep="${3:-0}"
    local candidate=""

    [ -z "$target_env" ] && return 1

    # Validate target_env format to prevent directory traversal and injection
    if [[ ! "$target_env" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
        return 1
    fi

    # Validate req_module identifier to prevent arbitrary Python code injection
    if [ -n "$req_module" ] && [[ ! "$req_module" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]]; then
        return 1
    fi

    # Session caching: return instantly if already resolved for this session
    local _cache_key="_CACHE_DISC_PY_${target_env}_${req_module}"
    _cache_key="${_cache_key//[^a-zA-Z0-9_]/_}"
    if [ -n "${!_cache_key:-}" ]; then
        if [ "${!_cache_key}" = "__NONE__" ]; then
            echo ""
            return 0
        fi
        echo "${!_cache_key}"
        return 0
    fi

    _verify_python() {
        local py="$1"
        if [ -z "$py" ] || [ ! -f "$py" ] || [ ! -x "$py" ]; then
            return 1
        fi
        if _is_unsafe_path "$py"; then
            return 1
        fi
        local base_py
        base_py="$(basename "$py")"
        if [[ ! "$base_py" =~ ^[pP][yY][tT][hH][oO][nN]([0-9]+(\.[0-9]+)?)?(\.[eE][xX][eE])?$ ]]; then
            return 1
        fi
        if [ -n "$req_module" ]; then
            "$py" -c "import sys; import $req_module" >/dev/null 2>&1
        else
            "$py" -c "import sys" >/dev/null 2>&1
        fi
    }

    # 1. Early exit: Check if an environment variable is already exported for this env/tool
    if [ "$target_env" = "SE3nv" ] || [ "$target_env" = "rfdiffusion" ]; then
        if [ -n "${PYTHON_RFDIFF:-}" ] && _verify_python "$PYTHON_RFDIFF"; then
            printf -v "$_cache_key" "%s" "$PYTHON_RFDIFF"
            echo "$PYTHON_RFDIFF"
            return 0
        fi
    elif [ "$target_env" = "mlfold" ]; then
        if [ -n "${PYTHON_MPNN:-}" ] && _verify_python "$PYTHON_MPNN"; then
            printf -v "$_cache_key" "%s" "$PYTHON_MPNN"
            echo "$PYTHON_MPNN"
            return 0
        fi
    fi

    local upper_env="${target_env^^}"
    upper_env="${upper_env//[^A-Z0-9]/_}"
    local env_var="PYTHON_${upper_env}"
    if [ -n "${!env_var:-}" ] && _verify_python "${!env_var}"; then
        printf -v "$_cache_key" "%s" "${!env_var}"
        echo "${!env_var}"
        return 0
    fi

    # 2. Early exit: Check if active conda environment matches
    if [ -n "$CONDA_PREFIX" ]; then
        if [ "$CONDA_DEFAULT_ENV" = "$target_env" ] || [ "${CONDA_PREFIX##*/}" = "$target_env" ]; then
            for cand in "$CONDA_PREFIX/bin/python" "$CONDA_PREFIX/python.exe" "$CONDA_PREFIX/python"; do
                if _verify_python "$cand"; then
                    printf -v "$_cache_key" "%s" "$cand"
                    echo "$cand"
                    return 0
                fi
            done
        fi
    fi

    # 3. Check local tools/envs directory
    for cand in "$LOCAL_ENVS_DIR/$target_env/bin/python" "$LOCAL_ENVS_DIR/$target_env/python.exe" "$LOCAL_ENVS_DIR/$target_env/python"; do
        if _verify_python "$cand"; then
            printf -v "$_cache_key" "%s" "$cand"
            echo "$cand"
            return 0
        fi
    done

    # 4. Check known base conda and micromamba installations
    for base_dir in \
        "$MAMBA_ROOT_PREFIX" \
        "$HOME/.micromamba" \
        "$HOME/micromamba" \
        "$HOME/.local/share/mamba" \
        "$CONDA_PREFIX/.." \
        "$HOME/miniconda3" \
        "$HOME/miniforge3" \
        "$HOME/mambaforge" \
        "$HOME/anaconda3" \
        "$HOME/.conda" \
        "/opt/conda" \
        "/usr/local/miniconda3" \
        "/usr/local/anaconda3"; do
        if [ -d "$base_dir/envs/$target_env" ]; then
            for cand in "$base_dir/envs/$target_env/bin/python" "$base_dir/envs/$target_env/python.exe" "$base_dir/envs/$target_env/python"; do
                if _verify_python "$cand"; then
                    printf -v "$_cache_key" "%s" "$cand"
                    echo "$cand"
                    return 0
                fi
            done
        fi
    done

    # 5. Fallback: Query conda info directly only if deep search is enabled or cache is already warm
    if [ "$enable_deep" -eq 1 ] || [ -n "$_CACHED_CONDA_ENVS_INFO" ]; then
        if [ -z "$_CACHED_CONDA_ENVS_INFO" ]; then
            local conda_bin
            conda_bin="$(find_conda_bin 2>/dev/null || true)"
            if [ -n "$conda_bin" ] && [ -x "$conda_bin" ]; then
                _CACHED_CONDA_ENVS_INFO="$("$conda_bin" info --envs 2>/dev/null || echo "")"
            else
                _CACHED_CONDA_ENVS_INFO="__NONE__"
            fi
        fi
        if [ "$_CACHED_CONDA_ENVS_INFO" != "__NONE__" ]; then
            local env_path
            env_path="$(echo "$_CACHED_CONDA_ENVS_INFO" | awk -v env="$target_env" '$1 == env {print $NF}' | head -n 1)"
            if [ -n "$env_path" ]; then
                for cand in "$env_path/bin/python" "$env_path/python.exe" "$env_path/python"; do
                    if _verify_python "$cand"; then
                        printf -v "$_cache_key" "%s" "$cand"
                        echo "$cand"
                        return 0
                    fi
                done
            fi
        fi
    fi

    printf -v "$_cache_key" "%s" "__NONE__"
    echo ""
    return 0
}

# Resolve and export neural model Python binaries if not explicitly set
if [ -z "$PYTHON_RFDIFF" ]; then
    _DISCOVERED_RFDIFF="$(discover_conda_python "${CONDA_ENV_RFDIFF:-SE3nv}" "torch" 2>/dev/null || true)"
    [ -z "$_DISCOVERED_RFDIFF" ] && _DISCOVERED_RFDIFF="$(discover_conda_python "rfdiffusion" "torch" 2>/dev/null || true)"
    if [ -n "$_DISCOVERED_RFDIFF" ]; then
        export PYTHON_RFDIFF="$_DISCOVERED_RFDIFF"
    fi
fi

if [ -z "$PYTHON_MPNN" ]; then
    _DISCOVERED_MPNN="$(discover_conda_python "${CONDA_ENV_MPNN:-mlfold}" "torch" 2>/dev/null || true)"
    [ -z "$_DISCOVERED_MPNN" ] && _DISCOVERED_MPNN="$(discover_conda_python "SE3nv" "torch" 2>/dev/null || true)"
    [ -z "$_DISCOVERED_MPNN" ] && [ -n "${PYTHON_RFDIFF:-}" ] && _DISCOVERED_MPNN="$PYTHON_RFDIFF"
    if [ -n "$_DISCOVERED_MPNN" ]; then
        export PYTHON_MPNN="$_DISCOVERED_MPNN"
    fi
fi
