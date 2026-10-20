#!/usr/bin/env bash
# install_dependencies.sh - Automated local installer for RFdiffusion, ProteinMPNN, and Micromamba environments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/scripts/env_setup.sh"

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}  📦 AUTOSCAFFOLD: AUTOMATIC DEPENDENCY INSTALLER     ${NC}"
echo -e "${CYAN}======================================================${NC}"
echo -e "This wizard will set up local repositories and environments into:"
echo -e "  Tools directory:       ${BOLD}$LOCAL_TOOLS_DIR${NC}"
echo -e "  Environments directory: ${BOLD}$LOCAL_ENVS_DIR${NC}\n"

mkdir -p "$LOCAL_TOOLS_DIR"
mkdir -p "$LOCAL_ENVS_DIR"

# Step 1: Detect or bootstrap micromamba
MICROMAMBA_BIN=$(find_conda_bin)
if [ -z "$MICROMAMBA_BIN" ]; then
    echo -e "${YELLOW}[1/4] Conda/Mamba not found. Bootstrapping standalone micromamba...${NC}"
    ARCH=$(uname -m)
    case "$ARCH" in
        x86_64)   MAMBA_ARCH="linux-64" ;;
        aarch64)  MAMBA_ARCH="linux-aarch64" ;;
        arm64)    MAMBA_ARCH="osx-arm64" ;;
        *)        MAMBA_ARCH="linux-64" ;;
    esac

    MICROMAMBA_VERSION="1.5.8"
    MICROMAMBA_URL="https://micro.mamba.pm/api/micromamba/${MAMBA_ARCH}/${MICROMAMBA_VERSION}"
    mkdir -p "$LOCAL_ENVS_DIR/bin"
    echo -e "Downloading micromamba binary for ${MAMBA_ARCH} (v${MICROMAMBA_VERSION})..."
    TMP_ARCHIVE=$(mktemp)
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$MICROMAMBA_URL" -o "$TMP_ARCHIVE"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$TMP_ARCHIVE" "$MICROMAMBA_URL"
    else
        rm -f "$TMP_ARCHIVE"
        echo -e "${RED}[ERROR] Neither curl nor wget is available to download micromamba.${NC}"
        exit 1
    fi

    if [ ! -s "$TMP_ARCHIVE" ]; then
        rm -f "$TMP_ARCHIVE"
        echo -e "${RED}[ERROR] Downloaded micromamba archive is empty or failed to download.${NC}"
        exit 1
    fi

    tar -xjf "$TMP_ARCHIVE" -C "$LOCAL_ENVS_DIR/bin" --strip-components=1 bin/micromamba
    rm -f "$TMP_ARCHIVE"
    chmod 755 "$LOCAL_ENVS_DIR/bin/micromamba"
    MICROMAMBA_BIN="$LOCAL_ENVS_DIR/bin/micromamba"
    echo -e "${GREEN}[✓] Micromamba v${MICROMAMBA_VERSION} installed to $MICROMAMBA_BIN${NC}"
else
    echo -e "${GREEN}[1/4] Existing package manager found:${NC} $MICROMAMBA_BIN"
fi

# Step 2: Clone or update RFdiffusion
echo -e "\n${YELLOW}[2/4] Setting up RFdiffusion repository...${NC}"
if [ ! -d "$LOCAL_TOOLS_DIR/RFdiffusion" ]; then
    echo "Cloning RFdiffusion into $LOCAL_TOOLS_DIR/RFdiffusion..."
    git clone https://github.com/RosettaCommons/RFdiffusion.git "$LOCAL_TOOLS_DIR/RFdiffusion"
else
    echo -e "${GREEN}[✓] RFdiffusion already present in $LOCAL_TOOLS_DIR/RFdiffusion${NC}"
fi

# Download RFdiffusion weights if missing
mkdir -p "$LOCAL_TOOLS_DIR/RFdiffusion/models"
if [ ! -f "$LOCAL_TOOLS_DIR/RFdiffusion/models/Base_ckpt.pt" ] && [ ! -f "$LOCAL_TOOLS_DIR/RFdiffusion/models/Complex_base_ckpt.pt" ]; then
    echo "Downloading RFdiffusion model weights (~1.5GB total)..."
    read -p "Download model weights now? (y/N): " DL_WEIGHTS
    if [[ "$DL_WEIGHTS" =~ ^[Yy]$ ]]; then
        cd "$LOCAL_TOOLS_DIR/RFdiffusion/models"
        WEIGHTS_URL="https://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac23702439f134f086ab5439e2"
        for model in Base_ckpt.pt Complex_base_ckpt.pt Complex_Fold_base_ckpt.pt; do
            if [ ! -f "$model" ]; then
                echo "Fetching $model..."
                curl -LO "${WEIGHTS_URL}/${model}" || wget -c "${WEIGHTS_URL}/${model}" || echo -e "${YELLOW}[WARN] Could not download $model${NC}"
            fi
        done
        cd "$SCRIPT_DIR"
    fi
fi

# Step 3: Clone or update ProteinMPNN
echo -e "\n${YELLOW}[3/4] Setting up ProteinMPNN repository...${NC}"
if [ ! -d "$LOCAL_TOOLS_DIR/ProteinMPNN" ]; then
    echo "Cloning ProteinMPNN into $LOCAL_TOOLS_DIR/ProteinMPNN..."
    git clone https://github.com/dauparas/ProteinMPNN.git "$LOCAL_TOOLS_DIR/ProteinMPNN"
else
    echo -e "${GREEN}[✓] ProteinMPNN already present in $LOCAL_TOOLS_DIR/ProteinMPNN${NC}"
fi

# Step 4: Conda / Micromamba environment creation
echo -e "\n${YELLOW}[4/4] Conda / Micromamba Environment Setup${NC}"
echo -e "Would you like to build the Conda environments now?"
echo -e "  - ${BOLD}$CONDA_ENV_RFDIFF${NC} (PyTorch, SE(3)-Transformers, hydra-core)"
echo -e "  - ${BOLD}$CONDA_ENV_MPNN${NC} (PyTorch for ProteinMPNN)"
echo -e "  - ${BOLD}$CONDA_ENV_COLABFOLD${NC} (ColabFold batch validation)"
read -p "Build environments? (y/N): " BUILD_ENVS

if [[ "$BUILD_ENVS" =~ ^[Yy]$ ]]; then
    # Create or update RFdiffusion env
    echo -e "\nConfiguring RFdiffusion environment ($CONDA_ENV_RFDIFF)..."
    if [ -f "$LOCAL_TOOLS_DIR/RFdiffusion/env/SE3nv.yml" ]; then
        "$MICROMAMBA_BIN" create -y -n "$CONDA_ENV_RFDIFF" -f "$LOCAL_TOOLS_DIR/RFdiffusion/env/SE3nv.yml" || true
    else
        "$MICROMAMBA_BIN" create -y -n "$CONDA_ENV_RFDIFF" python=3.9 pytorch=1.12.1 cudatoolkit=11.3 -c pytorch -c conda-forge || true
    fi

    # Create ProteinMPNN env
    echo -e "\nConfiguring ProteinMPNN environment ($CONDA_ENV_MPNN)..."
    "$MICROMAMBA_BIN" create -y -n "$CONDA_ENV_MPNN" python=3.9 pytorch=1.12.1 -c pytorch -c conda-forge || true

    # Create ColabFold env
    echo -e "\nConfiguring ColabFold environment ($CONDA_ENV_COLABFOLD)..."
    "$MICROMAMBA_BIN" create -y -n "$CONDA_ENV_COLABFOLD" python=3.10 openmm pdbfixer kalign2 -c conda-forge -c bioconda || \
    "$MICROMAMBA_BIN" create -y -n "$CONDA_ENV_COLABFOLD" python=3.10 openmm -c conda-forge || true

    echo -e "Installing colabfold package into $CONDA_ENV_COLABFOLD via pip..."
    "$MICROMAMBA_BIN" run -n "$CONDA_ENV_COLABFOLD" pip install --no-warn-conflicts "colabfold[alphafold-minus-jax]" 2>/dev/null || \
    "$MICROMAMBA_BIN" run -n "$CONDA_ENV_COLABFOLD" pip install --no-warn-conflicts colabfold 2>/dev/null || true

    echo -e "\n${GREEN}[✓] Environments configured successfully!${NC}"
fi

# Update config.env with local paths
cat << EOF > "$SCRIPT_DIR/config.env"
# Autogenerated by install_dependencies.sh on $(date)
RFDIFF_DIR="$LOCAL_TOOLS_DIR/RFdiffusion"
MPNN_DIR="$LOCAL_TOOLS_DIR/ProteinMPNN"
LOCAL_TOOLS_DIR="$LOCAL_TOOLS_DIR"
LOCAL_ENVS_DIR="$LOCAL_ENVS_DIR"
CONDA_ENV_RFDIFF="$CONDA_ENV_RFDIFF"
CONDA_ENV_MPNN="$CONDA_ENV_MPNN"
CONDA_ENV_COLABFOLD="$CONDA_ENV_COLABFOLD"
AUTOSCAFFOLD_REPO_URL="$AUTOSCAFFOLD_REPO_URL"
AUTOSCAFFOLD_BRANCH="$AUTOSCAFFOLD_BRANCH"
EOF

echo -e "\n${GREEN}Saved customized paths to $SCRIPT_DIR/config.env${NC}"
echo -e "${GREEN}Installation process complete!${NC}\n"
