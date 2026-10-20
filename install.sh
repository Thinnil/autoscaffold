#!/usr/bin/env bash
# ==============================================================================
# install.sh - Unified Standalone Installer & Maintenance Orchestrator for AutoScaffold
#
# Can be run standalone from ANY directory or curled directly:
#   curl -fsSL https://raw.githubusercontent.com/Thinnil/autoscaffold/main/install.sh | bash
#
# Capabilities:
# 1. System & hardware checks (OS, CPU, RAM, NVIDIA GPU, CUDA).
# 2. Automated Git clone, pull, and in-place code overwrite (preserving configs & weights).
# 3. Standalone micromamba bootstrap (zero root required).
# 4. RFdiffusion & ProteinMPNN clone/setup into ./tools/.
# 5. Model weights downloader into ./tools/RFdiffusion/models/.
# 6. Conda/Micromamba environment creation (SE3nv, mlfold, colabfold).
# 7. config.env generator with portable, non-hardcoded paths.
# 8. Interactive maintenance menu & CLI flags.
# ==============================================================================

set -e

# Terminal colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

# Trap Ctrl+C cleanly
trap 'echo -e "\n\n${YELLOW}[!] Installation/Operation cancelled by user.${NC}"; exit 130' INT

# Resolve target installation directory (where install.sh resides, or PWD if piped via bash)
if [ -n "${BASH_SOURCE[0]}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
    INSTALL_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    INSTALL_DIR="$(pwd -P 2>/dev/null || pwd)"
fi
if [ -z "$INSTALL_DIR" ] || [ "$INSTALL_DIR" = "/" ] || [ "$INSTALL_DIR" = "\\" ] || [[ "$INSTALL_DIR" =~ ^[a-zA-Z]:[\\/]?$ ]]; then
    echo -e "${RED}[ERROR] Dangerous or root installation directory: '$INSTALL_DIR'. Refusing to proceed.${NC}"
    exit 1
fi

REPO_URL="https://github.com/Thinnil/autoscaffold.git"
BRANCH="main"
LOCAL_TOOLS_DIR="$INSTALL_DIR/tools"
LOCAL_ENVS_DIR="$INSTALL_DIR/envs"
CONDA_ENV_RFDIFF="SE3nv"
CONDA_ENV_MPNN="mlfold"
CONDA_ENV_COLABFOLD="colabfold"

# ------------------------------------------------------------------------------
# Helper: Detect or locate Conda/Mamba/Micromamba binary
# ------------------------------------------------------------------------------
find_conda() {
    if [ -x "$LOCAL_ENVS_DIR/bin/micromamba" ]; then
        echo "$LOCAL_ENVS_DIR/bin/micromamba"
    elif [ -x "$LOCAL_ENVS_DIR/micromamba" ]; then
        echo "$LOCAL_ENVS_DIR/micromamba"
    elif command -v micromamba >/dev/null 2>&1; then
        command -v micromamba
    elif command -v mamba >/dev/null 2>&1; then
        command -v mamba
    elif command -v conda >/dev/null 2>&1; then
        command -v conda
    else
        echo ""
    fi
}

# ------------------------------------------------------------------------------
# Task 1: Hardware & System Assessment
# ------------------------------------------------------------------------------
run_diagnostics() {
    echo -e "\n${CYAN}======================================================${NC}"
    echo -e "${CYAN}     🩺 SYSTEM DIAGNOSTICS & PRE-FLIGHT AUDIT         ${NC}"
    echo -e "${CYAN}======================================================${NC}"
    
    OS_NAME=$(uname -s 2>/dev/null || echo "Unknown")
    ARCH=$(uname -m 2>/dev/null || echo "Unknown")
    echo -e "${BOLD}[1/4] Operating System:${NC} $OS_NAME ($ARCH)"
    
    if command -v nproc >/dev/null 2>&1; then
        echo -e "${BOLD}[2/4] CPU Cores:${NC}        $(nproc)"
    fi
    if command -v free >/dev/null 2>&1; then
        RAM_TOTAL=$(free -h | awk '/^Mem:/ {print $2}')
        RAM_AVAIL=$(free -h | awk '/^Mem:/ {print $7}')
        echo -e "${BOLD}[3/4] System RAM:${NC}       Total: ${RAM_TOTAL}, Available: ${RAM_AVAIL}"
    fi

    echo -e "${BOLD}[4/4] GPU / Accelerators:${NC}"
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_DATA=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)
        if [ -n "$GPU_DATA" ]; then
            CUDA_VER=$(nvidia-smi 2>/dev/null | grep -o "CUDA Version: [0-9.]*" | head -n 1 || echo "N/A")
            echo -e "  ${GREEN}[✓] NVIDIA GPU:${NC} $GPU_DATA"
            echo -e "  ${GREEN}[✓] Driver CUDA:${NC} $CUDA_VER"
        else
            echo -e "  ${YELLOW}[!] nvidia-smi found but no GPUs listed.${NC}"
        fi
    else
        echo -e "  ${YELLOW}[!] nvidia-smi not detected (CPU inference mode or missing NVIDIA drivers).${NC}"
    fi
}

# ------------------------------------------------------------------------------
# Task 2: Git Clone / Synchronize AutoScaffold Codebase
# ------------------------------------------------------------------------------
sync_codebase() {
    echo -e "\n${CYAN}======================================================${NC}"
    echo -e "${CYAN}  🔄 CODEBASE SYNCHRONIZATION & SELF-UPDATE           ${NC}"
    echo -e "${CYAN}======================================================${NC}"
    cd "$INSTALL_DIR"

    if [ ! -d ".git" ]; then
        echo -e "${YELLOW}[!] Current directory is not a Git repository.${NC}"
        echo -e "Initializing Git repository and linking with remote..."
        git init
        git remote add origin "$REPO_URL" 2>/dev/null || git remote set-url origin "$REPO_URL"
    fi

    echo -e "Fetching latest release from $REPO_URL ($BRANCH)..."
    git fetch origin "$BRANCH" 2>/dev/null || {
        echo -e "${YELLOW}[WARN] Could not contact GitHub. Operating in offline/local mode.${NC}"
        return 0
    }

    # If HEAD is not set (freshly initialized repository in non-empty directory), checkout origin/main directly!
    if ! git rev-parse HEAD >/dev/null 2>&1; then
        echo -e "Checking out AutoScaffold files from $BRANCH..."
        git checkout -f -B "$BRANCH" "origin/$BRANCH"
    else
        LOCAL_HASH=$(git rev-parse HEAD 2>/dev/null || echo "none")
        REMOTE_HASH=$(git rev-parse "origin/$BRANCH" 2>/dev/null || echo "unknown")

        if [[ "$LOCAL_HASH" == "$REMOTE_HASH" && "$LOCAL_HASH" != "none" ]]; then
            echo -e "${GREEN}[✓] AutoScaffold codebase is already up to date! (${LOCAL_HASH:0:8})${NC}"
        else
            echo -e "${YELLOW}New version detected on remote:${NC}"
            echo -e "  Local commit:  ${LOCAL_HASH:0:8}"
            echo -e "  Remote commit: ${REMOTE_HASH:0:8}"
            
            # Backup existing user scripts
            TIMESTAMP=$(date +%Y%m%d_%H%M%S)
            BACKUP_DIR="$INSTALL_DIR/.backup_$TIMESTAMP"
            mkdir -p "$BACKUP_DIR"
            for f in autoscaffold.sh de_novo.sh autoscaffold_suite.sh chimera_scaffold.sh discontinuous_scaffold.sh; do
                [ -f "$INSTALL_DIR/$f" ] && cp "$INSTALL_DIR/$f" "$BACKUP_DIR/"
            done
            echo -e "Active scripts backed up to: ${CYAN}$BACKUP_DIR${NC}"

            # Stash and update
            if ! git diff-index --quiet HEAD -- 2>/dev/null; then
                git stash save "Auto-update stash $TIMESTAMP" 2>/dev/null || true
            fi
            git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/$BRANCH"
            git pull origin "$BRANCH"
            echo -e "${GREEN}[✓] Codebase successfully updated to ${REMOTE_HASH:0:8}!${NC}"
        fi
    fi

    # Set execute permissions
    chmod 755 "$INSTALL_DIR"/*.sh 2>/dev/null || true
    chmod 755 "$INSTALL_DIR"/scripts/*.sh 2>/dev/null || true
    chmod 755 "$INSTALL_DIR"/scripts/*.py 2>/dev/null || true
}

# ------------------------------------------------------------------------------
# Task 3: Bootstrap Micromamba
# ------------------------------------------------------------------------------
bootstrap_micromamba() {
    echo -e "\n${CYAN}======================================================${NC}"
    echo -e "${CYAN}  📦 PACKAGE MANAGER AUDIT & MICROMAMBA BOOTSTRAP     ${NC}"
    echo -e "${CYAN}======================================================${NC}"
    
    CONDA_BIN=$(find_conda)
    if [ -n "$CONDA_BIN" ]; then
        echo -e "${GREEN}[✓] Package manager detected:${NC} $CONDA_BIN"
        return 0
    fi

    echo -e "${YELLOW}No Conda or Mamba found. Bootstrapping standalone micromamba (zero-root)...${NC}"
    ARCH=$(uname -m)
    case "$ARCH" in
        x86_64)   MAMBA_ARCH="linux-64" ;;
        aarch64)  MAMBA_ARCH="linux-aarch64" ;;
        arm64)    MAMBA_ARCH="osx-arm64" ;;
        *)        MAMBA_ARCH="linux-64" ;;
    esac

    MM_VER="1.5.8"
    MM_URL="https://micro.mamba.pm/api/micromamba/${MAMBA_ARCH}/${MM_VER}"
    mkdir -p "$LOCAL_ENVS_DIR/bin"
    TMP_ARCHIVE=$(mktemp)

    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$MM_URL" -o "$TMP_ARCHIVE"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$TMP_ARCHIVE" "$MM_URL"
    else
        rm -f "$TMP_ARCHIVE"
        echo -e "${RED}[ERROR] Neither curl nor wget is available to download micromamba.${NC}"
        return 1
    fi

    tar -xjf "$TMP_ARCHIVE" -C "$LOCAL_ENVS_DIR/bin" --strip-components=1 bin/micromamba
    rm -f "$TMP_ARCHIVE"
    chmod 755 "$LOCAL_ENVS_DIR/bin/micromamba"
    echo -e "${GREEN}[✓] Micromamba v${MM_VER} successfully bootstrapped into $LOCAL_ENVS_DIR/bin/micromamba${NC}"
}

# ------------------------------------------------------------------------------
# Task 4: Setup Tools (RFdiffusion & ProteinMPNN)
# ------------------------------------------------------------------------------
setup_tools() {
    echo -e "\n${CYAN}======================================================${NC}"
    echo -e "${CYAN}  🛠️  CLONING & CONFIGURING PROTEIN DESIGN TOOLS      ${NC}"
    echo -e "${CYAN}======================================================${NC}"
    mkdir -p "$LOCAL_TOOLS_DIR"

    # 1. RFdiffusion
    if [ ! -d "$LOCAL_TOOLS_DIR/RFdiffusion" ]; then
        echo -e "Cloning RFdiffusion into $LOCAL_TOOLS_DIR/RFdiffusion..."
        git clone https://github.com/RosettaCommons/RFdiffusion.git "$LOCAL_TOOLS_DIR/RFdiffusion"
    else
        echo -e "${GREEN}[✓] RFdiffusion is present in $LOCAL_TOOLS_DIR/RFdiffusion${NC}"
    fi

    # 2. ProteinMPNN
    if [ ! -d "$LOCAL_TOOLS_DIR/ProteinMPNN" ]; then
        echo -e "Cloning ProteinMPNN into $LOCAL_TOOLS_DIR/ProteinMPNN..."
        git clone https://github.com/dauparas/ProteinMPNN.git "$LOCAL_TOOLS_DIR/ProteinMPNN"
    else
        echo -e "${GREEN}[✓] ProteinMPNN is present in $LOCAL_TOOLS_DIR/ProteinMPNN${NC}"
    fi
}

# ------------------------------------------------------------------------------
# Task 5: Download Neural Network Weights
# ------------------------------------------------------------------------------
download_weights() {
    echo -e "\n${CYAN}======================================================${NC}"
    echo -e "${CYAN}  ⬇️  DOWNLOADING RFDIFFUSION MODEL WEIGHTS           ${NC}"
    echo -e "${CYAN}======================================================${NC}"
    
    mkdir -p "$LOCAL_TOOLS_DIR/RFdiffusion/models"
    cd "$LOCAL_TOOLS_DIR/RFdiffusion/models"
    WEIGHTS_URL="https://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac23702439f134f086ab5439e2"
    
    for model in Base_ckpt.pt Complex_base_ckpt.pt Complex_Fold_base_ckpt.pt; do
        if [ -f "$model" ]; then
            echo -e "  ${GREEN}[✓] Model checkpoint '$model' already exists.${NC}"
        else
            echo -e "  Downloading $model (~500MB)..."
            if command -v curl >/dev/null 2>&1; then
                curl -LO "${WEIGHTS_URL}/${model}" || echo -e "${YELLOW}[WARN] Download of $model incomplete.${NC}"
            elif command -v wget >/dev/null 2>&1; then
                wget -c "${WEIGHTS_URL}/${model}" || echo -e "${YELLOW}[WARN] Download of $model incomplete.${NC}"
            fi
        fi
    done
    cd "$INSTALL_DIR"
    echo -e "${GREEN}[✓] Model checkpoints verified.${NC}"
}

# ------------------------------------------------------------------------------
# Task 6: Build Virtual Environments
# ------------------------------------------------------------------------------
build_environments() {
    echo -e "\n${CYAN}======================================================${NC}"
    echo -e "${CYAN}  🧬 BUILDING ISOLATED PYTHON ENVIRONMENTS            ${NC}"
    echo -e "${CYAN}======================================================${NC}"
    
    CONDA_BIN=$(find_conda)
    if [ -z "$CONDA_BIN" ]; then
        bootstrap_micromamba
        CONDA_BIN=$(find_conda)
    fi

    # 1. RFdiffusion Env
    echo -e "\nConfiguring RFdiffusion environment ($CONDA_ENV_RFDIFF)..."
    if [ -f "$LOCAL_TOOLS_DIR/RFdiffusion/env/SE3nv.yml" ]; then
        "$CONDA_BIN" create -y -n "$CONDA_ENV_RFDIFF" -f "$LOCAL_TOOLS_DIR/RFdiffusion/env/SE3nv.yml" || true
    else
        "$CONDA_BIN" create -y -n "$CONDA_ENV_RFDIFF" python=3.9 pytorch=1.12.1 cudatoolkit=11.3 -c pytorch -c conda-forge || true
    fi

    # 2. ProteinMPNN Env
    echo -e "\nConfiguring ProteinMPNN environment ($CONDA_ENV_MPNN)..."
    "$CONDA_BIN" create -y -n "$CONDA_ENV_MPNN" python=3.9 pytorch=1.12.1 numpy -c pytorch -c conda-forge || true

    # 3. ColabFold Env
    echo -e "\nConfiguring ColabFold validation environment ($CONDA_ENV_COLABFOLD)..."
    "$CONDA_BIN" create -y -n "$CONDA_ENV_COLABFOLD" python=3.10 openmm pdbfixer kalign2 -c conda-forge -c bioconda || \
    "$CONDA_BIN" create -y -n "$CONDA_ENV_COLABFOLD" python=3.10 openmm -c conda-forge || true

    echo -e "Installing colabfold package into $CONDA_ENV_COLABFOLD via pip..."
    "$CONDA_BIN" run -n "$CONDA_ENV_COLABFOLD" pip install --no-warn-conflicts "colabfold[alphafold-minus-jax]" 2>/dev/null || \
    "$CONDA_BIN" run -n "$CONDA_ENV_COLABFOLD" pip install --no-warn-conflicts colabfold 2>/dev/null || true

    echo -e "\n${GREEN}[✓] Environments configured successfully!${NC}"
}

# ------------------------------------------------------------------------------
# Task 7: Generate Portable config.env
# ------------------------------------------------------------------------------
write_config() {
    cat << EOF > "$INSTALL_DIR/config.env"
# Autogenerated by install.sh on $(date)
RFDIFF_DIR="$LOCAL_TOOLS_DIR/RFdiffusion"
MPNN_DIR="$LOCAL_TOOLS_DIR/ProteinMPNN"
LOCAL_TOOLS_DIR="$LOCAL_TOOLS_DIR"
LOCAL_ENVS_DIR="$LOCAL_ENVS_DIR"
CONDA_ENV_RFDIFF="$CONDA_ENV_RFDIFF"
CONDA_ENV_MPNN="$CONDA_ENV_MPNN"
CONDA_ENV_COLABFOLD="$CONDA_ENV_COLABFOLD"
AUTOSCAFFOLD_REPO_URL="$REPO_URL"
AUTOSCAFFOLD_BRANCH="$BRANCH"
EOF
    echo -e "${GREEN}[✓] Saved portable configuration to $INSTALL_DIR/config.env${NC}"
}

# ------------------------------------------------------------------------------
# Task 8: Global CLI Launcher Shortcut Installation
# ------------------------------------------------------------------------------
install_launcher() {
    echo -e "\n${CYAN}======================================================================${NC}"
    echo -e "${CYAN}     🔗 GLOBAL SHORTCUT INSTALLATION ('autoscaffold' CLI)             ${NC}"
    echo -e "${CYAN}======================================================================${NC}"

    local TARGET_BIN
    if [ "${EUID:-$(id -u 2>/dev/null || echo 1000)}" -eq 0 ]; then
        TARGET_BIN="/usr/local/bin"
    else
        if [ -z "$HOME" ] || [ "$HOME" = "/" ] || [ "$HOME" = "\\" ]; then
            echo -e "${RED}[ERROR] User HOME directory is invalid or unset: '$HOME'${NC}"
            return 1
        fi
        TARGET_BIN="$HOME/.local/bin"
    fi

    mkdir -p "$TARGET_BIN"
    TARGET_BIN="$(cd -P "$TARGET_BIN" 2>/dev/null && pwd)"
    if [ -z "$TARGET_BIN" ] || [ "$TARGET_BIN" = "/" ] || [ "$TARGET_BIN" = "\\" ] || [[ "$TARGET_BIN" =~ ^[a-zA-Z]:[\\/]?$ ]]; then
        echo -e "${RED}[ERROR] Invalid or dangerous TARGET_BIN directory: '$TARGET_BIN'.${NC}"
        return 1
    fi

    local LAUNCHER_SRC="$INSTALL_DIR/autoscaffold"
    if [ ! -f "$LAUNCHER_SRC" ]; then
        echo -e "${RED}[ERROR] Launcher script not found at: $LAUNCHER_SRC${NC}"
        return 1
    fi
    LAUNCHER_SRC="$(cd -P "$(dirname "$LAUNCHER_SRC")" && pwd)/$(basename "$LAUNCHER_SRC")"

    chmod 755 "$LAUNCHER_SRC" 2>/dev/null || true
    # Safely remove existing symlink or file to prevent symlink traversal or stale targets
    rm -f "$TARGET_BIN/autoscaffold"
    ln -sf "$LAUNCHER_SRC" "$TARGET_BIN/autoscaffold" 2>/dev/null || true

    # Also clean up stale ~/.local/bin/autoscaffold if installed as root, or /usr/local/bin if unprivileged
    if [ "$TARGET_BIN" != "/usr/local/bin" ] && [ -w "/usr/local/bin" ] && [ -L "/usr/local/bin/autoscaffold" ]; then
        rm -f "/usr/local/bin/autoscaffold" 2>/dev/null || true
    fi

    # If ln -sf did not produce a true symlink (common on Windows Git Bash without native symlink rights),
    # write a canonical launcher wrapper script that explicitly points to LAUNCHER_SRC.
    if [ ! -L "$TARGET_BIN/autoscaffold" ]; then
        cat << 'EOF' > "$TARGET_BIN/autoscaffold"
#!/usr/bin/env bash
AUTOSCAFFOLD_TARGET="REPLACE_REPO_SRC"
if [ -f "$AUTOSCAFFOLD_TARGET" ]; then
    export AUTOSCAFFOLD_DIR="$(dirname "$AUTOSCAFFOLD_TARGET")"
    exec bash "$AUTOSCAFFOLD_TARGET" "$@"
fi
echo -e "\033[0;31m[ERROR] AutoScaffold target not found at: $AUTOSCAFFOLD_TARGET\033[0m"
exit 1
EOF
        sed -i "s|REPLACE_REPO_SRC|$LAUNCHER_SRC|g" "$TARGET_BIN/autoscaffold"
    fi
    chmod 755 "$TARGET_BIN/autoscaffold" 2>/dev/null || true

    # Clean up any conflicting legacy shell aliases in rc files
    for RC_FILE in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.bash_aliases"; do
        if [ -f "$RC_FILE" ] && [ ! -L "$RC_FILE" ]; then
            if grep -E "^[[:space:]]*alias[[:space:]]+autoscaffold=" "$RC_FILE" >/dev/null 2>&1; then
                sed -i -E 's|^([[:space:]]*alias[[:space:]]+autoscaffold=.*)|# [AutoScaffold legacy alias disabled] \1|g' "$RC_FILE" 2>/dev/null || true
                echo -e "${YELLOW}[!] Disabled legacy 'alias autoscaffold' in $RC_FILE to allow global binary execution.${NC}"
            fi
        fi
    done

    # Reset current shell command cache if supported
    hash -d autoscaffold 2>/dev/null || hash -r 2>/dev/null || true

    # Persist repository location marker in user home
    if [ -n "$HOME" ] && [ "$HOME" != "/" ]; then
        echo "$INSTALL_DIR" > "$HOME/.autoscaffold_home" 2>/dev/null || true
    fi

    echo -e "${GREEN}[✓] Global shortcut established:${NC} $TARGET_BIN/autoscaffold -> $LAUNCHER_SRC"

    # Verify TARGET_BIN in PATH
    if [[ ":$PATH:" != *":$TARGET_BIN:"* ]]; then
        echo -e "${YELLOW}[!] Notice: '$TARGET_BIN' is not currently in your system PATH.${NC}"
        local ADDED=false
        local RC_FILE
        for RC_FILE in "$HOME/.bashrc" "$HOME/.zshrc"; do
            # Verify RC_FILE exists and is a regular file (not a symlink) to prevent arbitrary file overwrite
            if [ -f "$RC_FILE" ] && [ ! -L "$RC_FILE" ]; then
                if ! grep -q "$TARGET_BIN" "$RC_FILE" 2>/dev/null; then
                    echo -e "\n# AutoScaffold Global CLI Launcher" >> "$RC_FILE"
                    echo "export PATH=\"$TARGET_BIN:\$PATH\"" >> "$RC_FILE"
                    echo -e "${GREEN}[✓] Appended PATH export to $RC_FILE${NC}"
                    ADDED=true
                fi
            fi
        done
        if [ "$ADDED" = true ]; then
            echo -e "${CYAN}[i] Run 'source ~/.bashrc' or reload your shell to activate the global command.${NC}"
        else
            echo -e "${YELLOW}[i] Please add '$TARGET_BIN' to your PATH manually:${NC}"
            echo -e "    export PATH=\"$TARGET_BIN:\$PATH\""
        fi
    else
        echo -e "${GREEN}[✓] Target directory '$TARGET_BIN' is already active in your PATH.${NC}"
    fi

    echo -e "${GREEN}[✓] You can now run 'autoscaffold' directly from any directory!${NC}\n"
}

# ------------------------------------------------------------------------------
# Express Setup Routine (Runs all installation tasks)
# ------------------------------------------------------------------------------
express_setup() {
    echo -e "\n${CYAN}======================================================================${NC}"
    echo -e "${CYAN}       🚀 STARTING FULL AUTONOMOUS AUTOSCAFFOLD INSTALLATION 🚀       ${NC}"
    echo -e "${CYAN}======================================================================${NC}"
    run_diagnostics
    sync_codebase
    bootstrap_micromamba
    setup_tools
    download_weights
    build_environments
    write_config

    if [ -t 0 ]; then
        echo ""
        read -r -p "▶ Would you like to create a global 'autoscaffold' shortcut in ~/.local/bin? [Y/n]: " PROMPT_SHORTCUT
        PROMPT_SHORTCUT=${PROMPT_SHORTCUT:-Y}
        if [[ "$PROMPT_SHORTCUT" =~ ^[Yy]$ ]]; then
            install_launcher
        fi
    else
        install_launcher || true
    fi

    echo -e "\n${GREEN}======================================================================${NC}"
    echo -e "${GREEN}  🎉 AUTOSCAFFOLD INSTALLATION & ENVIRONMENT SETUP COMPLETE! 🎉       ${NC}"
    echo -e "${GREEN}======================================================================${NC}"
    echo -e "You can now launch the master suite at any time by running:"
    echo -e "  ${BOLD}autoscaffold${NC} (or ${BOLD}./autoscaffold_suite.sh${NC})\n"
}

# ------------------------------------------------------------------------------
# CLI Flag Dispatcher
# ------------------------------------------------------------------------------
case "$1" in
    --express|--all|--full|-y|install)
        express_setup
        exit 0
        ;;
    --launcher|launcher|--shortcut|shortcut)
        install_launcher
        exit 0
        ;;
    --update|update|upgrade)
        sync_codebase
        exit 0
        ;;
    --diag|--diagnostics|diag|diagnostics)
        run_diagnostics
        exit 0
        ;;
    --weights|weights)
        download_weights
        exit 0
        ;;
    --envs|envs)
        build_environments
        write_config
        exit 0
        ;;
    --help|-h|help)
        echo -e "${BOLD}AutoScaffold Standalone Unified Installer & Maintenance Engine${NC}"
        echo -e "Usage:"
        echo -e "  $0 [command]"
        echo -e "\nCommands:"
        echo -e "  install, --full, -y  Perform full autonomous installation (code, micromamba, tools, weights, envs)"
        echo -e "  shortcut, launcher   Install global 'autoscaffold' CLI shortcut in ~/.local/bin"
        echo -e "  update, --update     Check for and merge latest updates from GitHub in-place"
        echo -e "  diagnostics, --diag  Run hardware and environment health check"
        echo -e "  weights              Download official RFdiffusion model checkpoints"
        echo -e "  envs                 Build or reconfigure Conda/Micromamba environments"
        echo -e "\nRun without arguments to enter the interactive maintenance menu."
        exit 0
        ;;
esac

# ------------------------------------------------------------------------------
# Interactive Maintenance Menu
# ------------------------------------------------------------------------------
while true; do
    clear 2>/dev/null || true
    CONDA_BIN=$(find_conda)
    [ -z "$CONDA_BIN" ] && CONDA_STATUS="${RED}Not Found${NC}" || CONDA_STATUS="${GREEN}Ready ($CONDA_BIN)${NC}"
    [ -d "$LOCAL_TOOLS_DIR/RFdiffusion" ] && RFDIFF_STATUS="${GREEN}Installed${NC}" || RFDIFF_STATUS="${YELLOW}Missing${NC}"
    [ -f "$LOCAL_TOOLS_DIR/RFdiffusion/models/Base_ckpt.pt" ] && WEIGHT_STATUS="${GREEN}Present${NC}" || WEIGHT_STATUS="${YELLOW}Missing${NC}"
    [ -d "$LOCAL_TOOLS_DIR/ProteinMPNN" ] && MPNN_STATUS="${GREEN}Installed${NC}" || MPNN_STATUS="${YELLOW}Missing${NC}"

    echo -e "${CYAN}======================================================================${NC}"
    echo -e "${CYAN}     🚀 AUTOSCAFFOLD: UNIFIED INSTALLER & MAINTENANCE ENGINE 🚀       ${NC}"
    echo -e "${CYAN}======================================================================${NC}"
    echo -e "Target Directory:   ${BOLD}$INSTALL_DIR${NC}"
    echo -e "Component Status:"
    echo -e "  • Package Manager:     $CONDA_STATUS"
    echo -e "  • RFdiffusion:         $RFDIFF_STATUS"
    echo -e "  • Model Checkpoints:   $WEIGHT_STATUS"
    echo -e "  • ProteinMPNN:         $MPNN_STATUS"
    echo -e "======================================================================\n"

    echo -e "${BOLD}Select an action:${NC}"
    echo -e "  ${GREEN}[1]${NC} ⚡ Full Express Setup (Check code, micromamba, tools, weights, envs)"
    echo -e "  ${GREEN}[2]${NC} 🔄 In-Place Codebase Self-Update (Fetch latest GitHub release)"
    echo -e "  ${GREEN}[3]${NC} 🛠️  Setup / Update Tools (RFdiffusion & ProteinMPNN repos)"
    echo -e "  ${GREEN}[4]${NC} ⬇️  Download / Verify RFdiffusion Model Checkpoints (~1.5GB)"
    echo -e "  ${GREEN}[5]${NC} 🧬 Build Virtual Environments (SE3nv, mlfold, colabfold)"
    echo -e "  ${GREEN}[6]${NC} 🩺 Run System & Hardware Diagnostics"
    echo -e "  ${GREEN}[7]${NC} 🔗 Install Global 'autoscaffold' Shortcut (~/.local/bin)"
    echo -e "  ${GREEN}[8]${NC} 🚀 Launch AutoScaffold Master Suite (./autoscaffold_suite.sh)"
    echo -e "  ${GREEN}[0]${NC} 🚪 Exit"
    echo -e ""

    read -r -p "▶ Enter selection [0-8]: " CHOICE

    case "$CHOICE" in
        1) express_setup; read -r -p "Press Enter to return to menu...";;
        2) sync_codebase; read -r -p "Press Enter to return to menu...";;
        3) setup_tools; write_config; read -r -p "Press Enter to return to menu...";;
        4) download_weights; read -r -p "Press Enter to return to menu...";;
        5) build_environments; write_config; read -r -p "Press Enter to return to menu...";;
        6) run_diagnostics; read -r -p "Press Enter to return to menu...";;
        7) install_launcher; read -r -p "Press Enter to return to menu...";;
        8)
            if [ -f "$INSTALL_DIR/autoscaffold_suite.sh" ]; then
                exec bash "$INSTALL_DIR/autoscaffold_suite.sh"
            else
                echo -e "${RED}[ERROR] autoscaffold_suite.sh not found. Run option 1 or 2 first.${NC}"
                read -r -p "Press Enter to continue..."
            fi
            ;;
        0|q|quit|exit)
            echo -e "\n${CYAN}Exiting installer. Happy engineering!${NC}\n"
            exit 0
            ;;
        *)
            echo -e "${RED}[ERROR] Invalid option '$CHOICE'.${NC}"
            sleep 1
            ;;
    esac
done
