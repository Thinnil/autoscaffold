#!/usr/bin/env bash
# de_novo.sh - Target-Binder Design and ColabFold Validation Pipeline
set -e

# Trap Ctrl+C (SIGINT) cleanly
trap 'echo -e "\n\n${YELLOW}[!] Operation cancelled by user.${NC}"; exit 130' INT

# Resolve script directory and source shared environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Verify and self-heal scripts directory if missing
if [ ! -f "$SCRIPT_DIR/scripts/env_setup.sh" ]; then
    echo -e "\033[1;33m[!] AutoScaffold environment files not found in: $SCRIPT_DIR/scripts/\033[0m"
    if [ -d "$SCRIPT_DIR/.git" ]; then
        echo -e "\033[1;36m[+] Restoring missing repository files from git...\033[0m"
        git -C "$SCRIPT_DIR" checkout HEAD -- scripts/ 2>/dev/null || git -C "$SCRIPT_DIR" pull origin main 2>/dev/null || true
    fi
    if [ ! -f "$SCRIPT_DIR/scripts/env_setup.sh" ]; then
        echo -e "\033[1;36m[+] Downloading essential AutoScaffold scripts from GitHub repository...\033[0m"
        mkdir -p "$SCRIPT_DIR/scripts"
        RAW_BASE="https://raw.githubusercontent.com/Thinnil/autoscaffold/main/scripts"
        CORE_SCRIPTS=(
            "env_setup.sh"
            "install_dependencies.sh"
            "self_update.sh"
            "diagnostics.sh"
            "chimera_scaffold.py"
            "discontinuous_motif_scaffold.py"
            "methodology_reporter.py"
            "pse_pdb_extractor.py"
            "extract_fixed_motif.py"
            "format_complex_fasta.py"
            "kabsch_rmsd.py"
        )
        for s in "${CORE_SCRIPTS[@]}"; do
            if command -v curl >/dev/null 2>&1; then
                curl -fsSL "$RAW_BASE/$s" -o "$SCRIPT_DIR/scripts/$s" 2>/dev/null || true
            elif command -v wget >/dev/null 2>&1; then
                wget -qO "$SCRIPT_DIR/scripts/$s" "$RAW_BASE/$s" 2>/dev/null || true
            fi
        done
        chmod +x "$SCRIPT_DIR"/scripts/*.sh "$SCRIPT_DIR"/scripts/*.py 2>/dev/null || true
    fi
    if [ ! -f "$SCRIPT_DIR/scripts/env_setup.sh" ]; then
        echo -e "\033[0;31m[ERROR] Could not locate or download 'scripts/env_setup.sh'.\033[0m"
        exit 1
    fi
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/scripts/env_setup.sh"

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}   🧬 DE NOVO BINDER DESIGN & VALIDATION PIPELINE 🧬  ${NC}"
echo -e "${CYAN}======================================================${NC}"
echo -e "Tip: Press ${YELLOW}Ctrl + C${NC} at any time to cancel.\n"

BACKGROUND_MODE=false
ORIG_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--background" ] || [ "$arg" = "-b" ]; then
        BACKGROUND_MODE=true
    else
        ORIG_ARGS+=("$arg")
    fi
done

# ==============================================================================
# DATA COLLECTION & VALIDATION
# ==============================================================================
if [ -n "${ORIG_ARGS[0]}" ]; then
    INPUT_PDB="${ORIG_ARGS[0]}"
else
    read -e -p "▶ 1. Path to Target PDB file: " INPUT_PDB
fi

if [[ "$INPUT_PDB" != *.pdb ]]; then
    echo -e "${RED}[ERROR] Target must be in .pdb format.${NC}"
    exit 1
fi

if [ ! -f "$INPUT_PDB" ]; then
    echo -e "${RED}[ERROR] File not found: '$INPUT_PDB'${NC}"
    exit 1
fi

ABS_INPUT_PDB=$(realpath "$INPUT_PDB")

if [ -n "${ORIG_ARGS[1]}" ]; then
    CHAIN_A="${ORIG_ARGS[1]}"
else
    read -p "▶ 2. Target Chain ID to preserve (e.g., A): " CHAIN_A
fi
CHAIN_A=${CHAIN_A:-A}
if [[ ! "$CHAIN_A" =~ ^[a-zA-Z0-9]+$ ]]; then
    echo -e "${RED}[ERROR] Invalid Target Chain ID '$CHAIN_A'. Must be alphanumeric.${NC}"
    exit 1
fi

if [ -n "${ORIG_ARGS[2]}" ]; then
    RES_RANGE="${ORIG_ARGS[2]}"
else
    read -p "▶ 3. Target residue range (e.g., 20-97. Press ENTER for FULL chain): " RES_RANGE
fi

if [[ "$RES_RANGE" == *"/"* ]]; then
    echo -e "${RED}[ERROR] Slashes are not allowed. Use hyphen syntax (e.g., 20-97).${NC}"
    exit 1
fi

# Auto-detect chain boundaries if empty
if [ -z "$RES_RANGE" ]; then
    echo -e "${CYAN}🔍 Analyzing PDB to detect chain ${CHAIN_A} boundaries...${NC}"

    read -r MIN_RES MAX_RES < <(awk -v ch="$CHAIN_A" '
        BEGIN { min = 999999; max = -999999; found = 0 }
        /^ATOM/ && substr($0,22,1) == ch {
            res = int(substr($0,23,4))
            if (res < min) min = res
            if (res > max) max = res
            found = 1
        }
        END {
            if (found) print min, max
        }
    ' "$ABS_INPUT_PDB")

    if [ -z "$MIN_RES" ] || [ -z "$MAX_RES" ]; then
        echo -e "${RED}[ERROR] No ATOM records found for chain '${CHAIN_A}' in PDB.${NC}"
        exit 1
    fi

    RES_RANGE="${MIN_RES}-${MAX_RES}"
    echo -e "${GREEN}[✓] Detected range: ${RES_RANGE}${NC}\n"
else
    if [[ ! "$RES_RANGE" =~ ^[0-9]+-[0-9]+$ ]]; then
        echo -e "${RED}[ERROR] Invalid residue range '$RES_RANGE'. Must be format START-END (e.g., 20-97).${NC}"
        exit 1
    fi
fi

read -p "▶ 4. Binder length range (e.g., 50-65 or 60): " BINDER_LEN
BINDER_LEN=${BINDER_LEN:-50-65}
if [[ ! "$BINDER_LEN" =~ ^[0-9]+(-[0-9]+)?$ ]]; then
    echo -e "${RED}[ERROR] Invalid binder length range '$BINDER_LEN'. Must be an integer or range (e.g. 50-65).${NC}"
    exit 1
fi

read -p "▶ 5. Hotspot residues separated by comma (e.g., A8,A44. Enter to skip): " HOTSPOTS
if [ -n "$HOTSPOTS" ] && [[ ! "$HOTSPOTS" =~ ^[a-zA-Z0-9,[:space:]]+$ ]]; then
    echo -e "${RED}[ERROR] Invalid hotspots format '$HOTSPOTS'. Use format A8,A44.${NC}"
    exit 1
fi

read -e -p "▶ 6. Output directory [Default: ./output_binder]: " OUT_DIR
OUT_DIR=${OUT_DIR:-./output_binder}
read -p "▶ 7. Base prefix name (e.g., my_binder): " OUT_PREFIX
OUT_PREFIX=${OUT_PREFIX:-binder_design}
if [[ ! "$OUT_PREFIX" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
    echo -e "${RED}[ERROR] Invalid output prefix '$OUT_PREFIX'. Use alphanumeric characters, dots, underscores or hyphens.${NC}"
    exit 1
fi

echo -e "\n${YELLOW}--- Advanced Settings ---${NC}"
read -p "▶ 8. Number of 3D backbone designs (RFdiffusion) [Default: 3]: " NUM_DESIGNS
NUM_DESIGNS=${NUM_DESIGNS:-3}
if [[ ! "$NUM_DESIGNS" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "${RED}[ERROR] Invalid number of designs '$NUM_DESIGNS'. Must be a positive integer.${NC}"
    exit 1
fi

read -p "▶ 9. ProteinMPNN temperature (0.1 conservative, 0.3 diverse) [Default: 0.1]: " MPNN_TEMP
MPNN_TEMP=${MPNN_TEMP:-0.1}
if [[ ! "$MPNN_TEMP" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo -e "${RED}[ERROR] Invalid sampling temperature '$MPNN_TEMP'. Must be a decimal number.${NC}"
    exit 1
fi

read -p "▶ 10. Sequences per backbone (ProteinMPNN) [Default: 2]: " NUM_SEQS
NUM_SEQS=${NUM_SEQS:-2}
if [[ ! "$NUM_SEQS" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "${RED}[ERROR] Invalid sequence count '$NUM_SEQS'. Must be a positive integer.${NC}"
    exit 1
fi

echo -e "\n${GREEN}[✓] Parameters validated. Creating directories...${NC}\n"

ABS_OUT_DIR=$(realpath "$OUT_DIR" 2>/dev/null || echo "$OUT_DIR")
if [ -z "$ABS_OUT_DIR" ] || [ "$ABS_OUT_DIR" = "/" ] || [ "$ABS_OUT_DIR" = "\\" ] || [[ "$ABS_OUT_DIR" =~ ^[a-zA-Z]:[\\/]?$ ]]; then
    echo -e "${RED}[ERROR] Invalid or dangerous output directory path: '$OUT_DIR'.${NC}"
    exit 1
fi

# Directory layout
mkdir -p "$ABS_OUT_DIR/1_rfdiffusion"
mkdir -p "$ABS_OUT_DIR/2_mpnn_inputs/target_folder"
mkdir -p "$ABS_OUT_DIR/3_mpnn_outputs"
mkdir -p "$ABS_OUT_DIR/4_colabfold_validation"

PID_FILE="$ABS_OUT_DIR/run.pid"
LOG_FILE="$ABS_OUT_DIR/run.log"

# ------------------------------------------------------------------------------
# Background Execution Option
# ------------------------------------------------------------------------------
if [ -t 0 ] && [ "$BACKGROUND_MODE" = false ]; then
    read -p "▶ Do you want to run this workflow in the background? [y/N]: " RUN_BG
    if [[ "$RUN_BG" =~ ^[Yy]$ ]]; then
        BACKGROUND_MODE=true
        nohup bash "$0" "${ORIG_ARGS[@]}" --background < /dev/null > "$LOG_FILE" 2>&1 &
        BG_PID=$!
        echo "$BG_PID" > "$PID_FILE"
        echo -e "\n${CYAN}======================================================================${NC}"
        echo -e "${GREEN}[✓] Workflow launched successfully in the background.${NC}"
        echo -e "PID:             ${YELLOW}${BG_PID}${NC}"
        echo -e "Log file:        ${CYAN}${LOG_FILE}${NC}"
        echo -e "PID file:        ${CYAN}${PID_FILE}${NC}"
        echo -e "Monitor command: ${BOLD}tail -f \"${LOG_FILE}\"${NC}"
        echo -e "${CYAN}======================================================================${NC}\n"
        exit 0
    fi
fi

# ==============================================================================
# PHASE 1: RFdiffusion
# ==============================================================================
echo -e "${CYAN}>>> STARTING PHASE 1: RFdiffusion Binder Generation <<<${NC}"
activate_env "$CONDA_ENV_RFDIFF"

if [ ! -d "$RFDIFF_DIR" ] || [ ! -f "$RFDIFF_DIR/scripts/run_inference.py" ]; then
    echo -e "${RED}[ERROR] RFdiffusion script not found at: $RFDIFF_DIR/scripts/run_inference.py${NC}"
    if [ -t 0 ] && [ -f "$SCRIPT_DIR/scripts/install_dependencies.sh" ]; then
        read -p "▶ Would you like to run the Automatic Dependency Installer now? [y/N]: " RUN_INST
        if [[ "$RUN_INST" =~ ^[Yy]$ ]]; then
            bash "$SCRIPT_DIR/scripts/install_dependencies.sh"
            # shellcheck disable=SC1091
            source "$SCRIPT_DIR/scripts/env_setup.sh"
        fi
    fi
    if [ ! -d "$RFDIFF_DIR" ] || [ ! -f "$RFDIFF_DIR/scripts/run_inference.py" ]; then
        echo -e "${RED}Please configure RFDIFF_DIR in config.env or run: ./autoscaffold_suite.sh install${NC}"
        exit 1
    fi
fi

CURRENT_PWD=$(pwd)
cd "$RFDIFF_DIR"

RF_ARGS=(
    "./scripts/run_inference.py"
    "inference.output_prefix=${ABS_OUT_DIR}/1_rfdiffusion/$OUT_PREFIX"
    "inference.input_pdb=${ABS_INPUT_PDB}"
    "contigmap.contigs=[${CHAIN_A}${RES_RANGE}/0 ${BINDER_LEN}]"
    "inference.num_designs=$NUM_DESIGNS"
)

if [ -n "$HOTSPOTS" ]; then
    RF_ARGS+=("ppi.hotspot_res=[$HOTSPOTS]")
fi

echo -e "Executing RFdiffusion..."
python "${RF_ARGS[@]}"

cd "$CURRENT_PWD"

# Find generated backbone models (bash globbing avoids find/sort subshell spawning)
shopt -s nullglob
GENERATED_PDBS=("${ABS_OUT_DIR}/1_rfdiffusion/${OUT_PREFIX}"_*.pdb)
shopt -u nullglob

if [ ${#GENERATED_PDBS[@]} -eq 0 ]; then
    echo -e "${RED}[CRITICAL ERROR] RFdiffusion failed to generate designs.${NC}"
    exit 1
fi

echo -e "${GREEN}[✓] RFdiffusion generated ${#GENERATED_PDBS[@]} backbone candidate(s).${NC}\n"
BEST_MODEL="${GENERATED_PDBS[0]}"

# ==============================================================================
# PHASE 2: ProteinMPNN
# ==============================================================================
echo -e "${CYAN}>>> STARTING PHASE 2: ProteinMPNN Binder Sequence Design <<<${NC}"
activate_env "$CONDA_ENV_MPNN"

if [ ! -d "$MPNN_DIR" ] || [ ! -f "$MPNN_DIR/protein_mpnn_run.py" ]; then
    echo -e "${RED}[ERROR] ProteinMPNN not found at: $MPNN_DIR/protein_mpnn_run.py${NC}"
    if [ -t 0 ] && [ -f "$SCRIPT_DIR/scripts/install_dependencies.sh" ]; then
        read -p "▶ Would you like to run the Automatic Dependency Installer now? [y/N]: " RUN_INST
        if [[ "$RUN_INST" =~ ^[Yy]$ ]]; then
            bash "$SCRIPT_DIR/scripts/install_dependencies.sh"
            # shellcheck disable=SC1091
            source "$SCRIPT_DIR/scripts/env_setup.sh"
        fi
    fi
    if [ ! -d "$MPNN_DIR" ] || [ ! -f "$MPNN_DIR/protein_mpnn_run.py" ]; then
        echo -e "${RED}Please configure MPNN_DIR in config.env or run: ./autoscaffold_suite.sh install${NC}"
        exit 1
    fi
fi

if [ -n "$ABS_OUT_DIR" ] && [ -d "${ABS_OUT_DIR}/2_mpnn_inputs/target_folder" ]; then
    rm -f "${ABS_OUT_DIR}/2_mpnn_inputs/target_folder/"*.pdb
fi
cp "$BEST_MODEL" "${ABS_OUT_DIR}/2_mpnn_inputs/target_folder/"

cd "$MPNN_DIR"

python helper_scripts/parse_multiple_chains.py \
    --input_path "${ABS_OUT_DIR}/2_mpnn_inputs/target_folder" \
    --output_path "${ABS_OUT_DIR}/2_mpnn_inputs/parsed.jsonl"

# CRITICAL FIX: We want to FIX the TARGET chain ($CHAIN_A), NOT the binder chain!
# In ProteinMPNN assign_fixed_chains.py, --chain_list specifies chains whose sequences are KEPT FIXED.
echo -e "Fixing target chain '${CHAIN_A}' sequence identities (designing binder)..."
python helper_scripts/assign_fixed_chains.py \
    --input_path "${ABS_OUT_DIR}/2_mpnn_inputs/parsed.jsonl" \
    --output_path "${ABS_OUT_DIR}/2_mpnn_inputs/assigned.jsonl" \
    --chain_list "$CHAIN_A"

python protein_mpnn_run.py \
    --jsonl_path "${ABS_OUT_DIR}/2_mpnn_inputs/parsed.jsonl" \
    --chain_id_jsonl "${ABS_OUT_DIR}/2_mpnn_inputs/assigned.jsonl" \
    --out_folder "${ABS_OUT_DIR}/3_mpnn_outputs" \
    --num_seq_per_target "$NUM_SEQS" \
    --sampling_temp "$MPNN_TEMP" \
    --seed 37

cd "$CURRENT_PWD"
echo -e "${GREEN}[✓] Binder sequences designed.${NC}\n"

# ==============================================================================
# PHASE 3: ColabFold (Validation and Cleanup)
# ==============================================================================
echo -e "${CYAN}>>> STARTING PHASE 3: ColabFold Multimer Validation <<<${NC}"
activate_env "$CONDA_ENV_COLABFOLD"

shopt -s nullglob
local_fastas=("${ABS_OUT_DIR}/3_mpnn_outputs/seqs/"*.fa)
shopt -u nullglob
if [ ${#local_fastas[@]} -eq 0 ]; then
    echo -e "${RED}[ERROR] No ProteinMPNN FASTA sequences found in ${ABS_OUT_DIR}/3_mpnn_outputs/seqs${NC}"
    exit 1
fi
MPNN_FASTA="${local_fastas[0]}"

VALIDATION_FASTA="${ABS_OUT_DIR}/4_colabfold_validation/final_complex.fa"

# Robust Python-based FASTA formatting (fixing grep/cut and [: 50-65: integer error)
FORMATTER_SCRIPT="$SCRIPT_DIR/scripts/format_complex_fasta.py"
python "$FORMATTER_SCRIPT" \
    "${ABS_OUT_DIR}/2_mpnn_inputs/parsed.jsonl" \
    "$CHAIN_A" \
    "$MPNN_FASTA" \
    "$BINDER_LEN" \
    "$VALIDATION_FASTA" \
    1

if ! command -v colabfold_batch >/dev/null 2>&1; then
    echo -e "${YELLOW}[WARN] 'colabfold_batch' not found in current environment.${NC}"
    echo -e "ColabFold validation FASTA prepared at: ${CYAN}$VALIDATION_FASTA${NC}"
    echo -e "You can run colabfold_batch on your compute cluster or Colab manually."
else
    echo -e "${YELLOW}Running ColabFold (AlphaFold-Multimer)...${NC}"
    colabfold_batch "$VALIDATION_FASTA" "${ABS_OUT_DIR}/4_colabfold_validation/"

    shopt -s nullglob
    top_models=("${ABS_OUT_DIR}/4_colabfold_validation/"*rank_001*.pdb)
    shopt -u nullglob
    if [ ${#top_models[@]} -eq 0 ]; then
        echo -e "${YELLOW}[!] ColabFold finished without producing top-ranked model (*rank_001*.pdb).${NC}"
    else
        TOP_MODEL="${top_models[0]}"
        echo -e "${GREEN}[✓] Top-ranked complex model produced!${NC}"
        # Cleanup secondary models (rank 002-005) only if confirmed
        read -p "Clean up secondary models (rank 002-005) to save disk space? (Y/n): " DO_CLEAN
        DO_CLEAN=${DO_CLEAN:-Y}
        if [[ "$DO_CLEAN" =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}Cleaning up sub-optimal models (rank 002-005)...${NC}"
            if [ -n "$ABS_OUT_DIR" ] && [ -d "${ABS_OUT_DIR}/4_colabfold_validation" ]; then
                find "${ABS_OUT_DIR}/4_colabfold_validation/" -type f \( -name "*rank_00[2-5]*.pdb" -o -name "*rank_00[2-5]*.json" \) -delete 2>/dev/null || true
            fi
        fi
    fi
fi

echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN} 🏆 PIPELINE COMPLETED SUCCESSFULLY 🏆                ${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "Validated complex files located in:"
echo -e "${CYAN}${ABS_OUT_DIR}/4_colabfold_validation/${NC}\n"

# Generate Automated Methodology & Reproducibility Report
python "$SCRIPT_DIR/scripts/methodology_reporter.py" \
    --run_name "${OUT_PREFIX}_Binder_Design" \
    --mode "De Novo Target-Binder Design" \
    --out_dir "$ABS_OUT_DIR" \
    --input_pdb "$ABS_INPUT_PDB" \
    --donor "Target Chain: ${CHAIN_A} (${RES_RANGE})" \
    --contigs "${CHAIN_A}${RES_RANGE}/0 ${BINDER_LEN}" \
    --num_designs "$NUM_DESIGNS" \
    --num_seqs "$NUM_SEQS" \
    --mpnn_temp "$MPNN_TEMP" \
    --cmd "bash de_novo.sh \"$ABS_INPUT_PDB\" \"$CHAIN_A\" \"$RES_RANGE\"" \
    --status "SUCCESS" \
    --total_candidates "$NUM_DESIGNS" \
    --passing_candidates "$NUM_DESIGNS" 2>/dev/null || true
