#!/usr/bin/env bash
# diagnostics.sh - System, hardware, and environment diagnostics for autoscaffold

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/scripts/env_setup.sh"

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}     🔍 AUTOSCAFFOLD: SYSTEM & HARDWARE DIAGNOSTICS   ${NC}"
echo -e "${CYAN}======================================================${NC}"

# 1. Operating System
OS_NAME=$(uname -s 2>/dev/null || echo "Unknown")
echo -e "\n${BOLD}[1/5] Host System:${NC}"
echo -e "  OS: $OS_NAME"
if [ -f /etc/os-release ]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    echo -e "  Distro: $PRETTY_NAME"
fi
echo -e "  Kernel: $(uname -r 2>/dev/null || echo 'N/A')"
echo -e "  Arch:   $(uname -m 2>/dev/null || echo 'N/A')"

# 2. Hardware (CPU, RAM, GPU/CUDA)
echo -e "\n${BOLD}[2/5] Hardware & Accelerators:${NC}"
# CPU & RAM
if command -v nproc >/dev/null 2>&1; then
    echo -e "  CPU Cores: $(nproc)"
fi
if command -v free >/dev/null 2>&1; then
    RAM_TOTAL=$(free -h | awk '/^Mem:/ {print $2}')
    RAM_AVAIL=$(free -h | awk '/^Mem:/ {print $7}')
    echo -e "  System RAM: Total ${RAM_TOTAL}, Available ${RAM_AVAIL}"
fi

# NVIDIA GPU
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_DATA=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)
    if [ -n "$GPU_DATA" ]; then
        GPU_COUNT=$(echo "$GPU_DATA" | grep -c .)
        FIRST_LINE=$(echo "$GPU_DATA" | head -n 1)
        GPU_NAME=$(echo "$FIRST_LINE" | awk -F', ' '{print $1}')
        GPU_MEM=$(echo "$FIRST_LINE" | awk -F', ' '{print $2}')
        CUDA_VER=$(nvidia-smi 2>/dev/null | grep -o "CUDA Version: [0-9.]*" | head -n 1)
        echo -e "  ${GREEN}[✓] NVIDIA GPU Detected:${NC} $GPU_COUNT x $GPU_NAME ($GPU_MEM VRAM)"
        echo -e "  ${GREEN}[✓] Driver/CUDA:${NC} ${CUDA_VER:-N/A}"
    else
        echo -e "  ${YELLOW}[!] nvidia-smi returned no GPU devices.${NC}"
    fi
else
    echo -e "  ${YELLOW}[!] nvidia-smi not found. GPU acceleration might be unavailable or CPU mode used.${NC}"
fi

# 3. Core CLI Tools
echo -e "\n${BOLD}[3/5] Essential CLI Utilities:${NC}"
for tool in git python3 curl wget bash; do
    if command -v "$tool" >/dev/null 2>&1; then
        TOOL_PATH=$(command -v "$tool")
        echo -e "  ${GREEN}[✓] $tool:${NC} $TOOL_PATH"
    else
        echo -e "  ${YELLOW}[✗] $tool: NOT FOUND${NC}"
    fi
done

# 4. Conda / Mamba Environment Manager
echo -e "\n${BOLD}[4/5] Package & Environment Managers:${NC}"
CONDA_BIN=$(find_conda_bin)
if [ -n "$CONDA_BIN" ]; then
    echo -e "  ${GREEN}[✓] Package manager detected:${NC} $CONDA_BIN"
    if [[ "$CONDA_BIN" == *"micromamba"* ]]; then
        MM_VER=$("$CONDA_BIN" --version 2>&1 | head -n 1)
        echo -e "      Version: $MM_VER"
    else
        C_VER=$("$CONDA_BIN" --version 2>&1 || echo "")
        echo -e "      Version: $C_VER"
    fi
else
    echo -e "  ${YELLOW}[✗] No conda, mamba, or micromamba found in PATH or $LOCAL_ENVS_DIR${NC}"
    echo -e "      ${CYAN}Hint: Run the Automatic Dependency Installer (Option 5) to bootstrap micromamba.${NC}"
fi

# 5. Pipeline Environments & Tool Repositories
echo -e "\n${BOLD}[5/5] Pipeline Tools & Model Checkpoints:${NC}"

# RFdiffusion repo
if [ -d "$RFDIFF_DIR" ] && [ -f "$RFDIFF_DIR/scripts/run_inference.py" ]; then
    echo -e "  ${GREEN}[✓] RFdiffusion Repository:${NC} $RFDIFF_DIR"
    # Check weights
    if [ -d "$RFDIFF_DIR/models" ] && [ "$(ls -A "$RFDIFF_DIR/models" 2>/dev/null)" ]; then
        echo -e "      ${GREEN}[✓] Model weights present in $RFDIFF_DIR/models${NC}"
    else
        echo -e "      ${YELLOW}[!] Model weights missing in $RFDIFF_DIR/models${NC}"
    fi
else
    echo -e "  ${YELLOW}[✗] RFdiffusion Repository not found at:${NC} $RFDIFF_DIR"
fi

# ProteinMPNN repo
if [ -d "$MPNN_DIR" ] && [ -f "$MPNN_DIR/protein_mpnn_run.py" ]; then
    echo -e "  ${GREEN}[✓] ProteinMPNN Repository:${NC} $MPNN_DIR"
else
    echo -e "  ${YELLOW}[✗] ProteinMPNN Repository not found at:${NC} $MPNN_DIR"
fi

# Environments (cached conda env list query)
_CONDA_ENV_LIST_CACHE=""
check_env_exists() {
    local env_name="$1"
    if [ -d "$LOCAL_ENVS_DIR/$env_name" ]; then
        echo -e "  ${GREEN}[✓] Conda/Micromamba env '$env_name':${NC} Present (local prefix: $LOCAL_ENVS_DIR/$env_name)"
        return 0
    fi
    local conda_bin
    conda_bin=$(find_conda_bin)
    if [ -n "$conda_bin" ]; then
        if [ -z "$_CONDA_ENV_LIST_CACHE" ]; then
            _CONDA_ENV_LIST_CACHE=$("$conda_bin" env list 2>/dev/null || true)
        fi
        if echo "$_CONDA_ENV_LIST_CACHE" | grep -E "(^|\s)${env_name}(\s|$)" >/dev/null 2>&1; then
            echo -e "  ${GREEN}[✓] Conda env '$env_name':${NC} Present in environment registry"
            return 0
        fi
    fi
    echo -e "  ${YELLOW}[✗] Conda env '$env_name': NOT FOUND${NC}"
    return 1
}

check_env_exists "$CONDA_ENV_RFDIFF"
check_env_exists "$CONDA_ENV_MPNN"
check_env_exists "$CONDA_ENV_COLABFOLD"

echo -e "\n${CYAN}======================================================${NC}"
echo -e "${GREEN}Diagnostics check finished.${NC}\n"
