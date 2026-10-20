#!/usr/bin/env bash
# discontinuous_scaffold.sh - Scaffold Discontinuous Binding Surfaces
# Powered by RFdiffusion, ProteinMPNN, and ColabFold Multimer
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

# Fast exit on help flag before pre-flight checks
for arg in "$@"; do
    if [ "$arg" = "--help" ] || [ "$arg" = "-h" ] || [ "$arg" = "help" ]; then
        echo -e "${BOLD}Discontinuous Motif Scaffolding Shell Wrapper${NC}"
        echo -e "Usage:"
        echo -e "  $0 [complex.pdb] [motif_chain] [motif_residues] [target_chain]"
        echo -e "  $0 --complex_pdb PDB --motif_chain CHAIN --motif_residues RESIDUES [options...]"
        echo -e "\nOptions:"
        echo -e "  --complex_pdb PDB       Path to complex PDB file"
        echo -e "  --motif_chain CHAIN     Chain ID containing discontinuous motif"
        echo -e "  --motif_residues RES    Residue intervals (e.g. '310-313,338-347,594-596,612-625')"
        echo -e "  --target_chain CHAIN    Target chain ID (optional, for target context mode)"
        echo -e "  --min_total_len N       Minimum total scaffold length [Default: 70]"
        echo -e "  --max_total_len N       Maximum total scaffold length [Default: 95]"
        echo -e "  --num_designs N         Number of RFdiffusion backbones [Default: 50]"
        echo -e "  --mpnn_seqs_per_bb N    MPNN sequences per backbone [Default: 4]"
        echo -e "  --mpnn_batch_size N     ProteinMPNN batch size for GPU parallelization [Default: 4]"
        echo -e "  --colabfold_batch_size N Batch chunk size for ColabFold inputs (0=consolidated) [Default: 0]"
        echo -e "  --colabfold_num_recycles N Number of prediction recycles in ColabFold [Default: 3]"
        echo -e "  --colabfold_num_models N Number of prediction models in ColabFold [Default: 1]"
        echo -e "  --out_dir DIR           Output directory [Default: outputs/discontinuous_scaffold]"
        echo -e "  --plddt_cutoff VAL      pLDDT filter cutoff [Default: 80.0]"
        echo -e "  --rmsd_cutoff VAL       Motif RMSD filter cutoff [Default: 1.2]"
        echo -e "  --iptm_cutoff VAL       ipTM filter cutoff [Default: 0.70]"
        echo -e "  --dry_run               Run pipeline without executing GPU inference"
        exit 0
    fi
done

echo -e "${CYAN}======================================================================${NC}"
echo -e "${CYAN} 🧬 AUTOSCAFFOLD: DISCONTINUOUS CONTACT MOTIF SCAFFOLDING 🧬          ${NC}"
echo -e "${CYAN} RFdiffusion (Noise=0.0)  •  ProteinMPNN  •  ColabFold (Kabsch RMSD)   ${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo -e "Tip: Press ${YELLOW}Ctrl + C${NC} at any time to cancel.\n"

# ------------------------------------------------------------------------------
# 1. Environment Verification
# ------------------------------------------------------------------------------
echo -e "${YELLOW}--- Environment Pre-Flight Checks ---${NC}"

# Check RFdiffusion Conda Env
if [ -n "$CONDA_ENV_RFDIFF" ]; then
    echo -e "  Checking RFdiffusion environment (${CONDA_ENV_RFDIFF})..."
    activate_env "$CONDA_ENV_RFDIFF" 2>/dev/null || echo -e "  ${YELLOW}[!] RFdiffusion env '$CONDA_ENV_RFDIFF' activation pending.${NC}"
fi

# Check ProteinMPNN Conda Env
if [ -n "$CONDA_ENV_MPNN" ]; then
    echo -e "  Checking ProteinMPNN environment (${CONDA_ENV_MPNN})..."
    activate_env "$CONDA_ENV_MPNN" 2>/dev/null || echo -e "  ${YELLOW}[!] ProteinMPNN env '$CONDA_ENV_MPNN' activation pending.${NC}"
fi

# Check ColabFold
if command -v colabfold_batch >/dev/null 2>&1; then
    echo -e "  ${GREEN}[✓] colabfold_batch detected on PATH.${NC}"
else
    echo -e "  ${YELLOW}[!] colabfold_batch not found on active PATH. Automated validation script will be generated.${NC}"
fi

echo -e "${GREEN}[✓] Environment checks passed.${NC}\n"

# ------------------------------------------------------------------------------
# 2. Argument Parsing & Interactive Prompts
# ------------------------------------------------------------------------------
COMPLEX_PDB=""
MOTIF_CHAIN=""
MOTIF_RESIDUES=""
TARGET_CHAIN=""
MIN_LEN="70"
MAX_LEN="95"
NUM_DESIGNS="50"
NUM_SEQS="4"
MPNN_BATCH_SIZE="4"
COLABFOLD_BATCH_SIZE="0"
COLABFOLD_NUM_RECYCLES="3"
COLABFOLD_NUM_MODELS="1"
OUT_DIR="outputs/discontinuous_scaffold"
PLDDT_CUTOFF="80.0"
RMSD_CUTOFF="1.2"
IPTM_CUTOFF="0.70"
DRY_RUN=""
BACKGROUND_MODE=false
ORIG_ARGS=("$@")

# Parse positional or named arguments if provided
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h|help)
            echo -e "${BOLD}Discontinuous Motif Scaffolding Shell Wrapper${NC}"
            echo -e "Usage:"
            echo -e "  $0 [complex.pdb] [motif_chain] [motif_residues] [target_chain]"
            echo -e "  $0 --complex_pdb PDB --motif_chain CHAIN --motif_residues RESIDUES [options...]"
            echo -e "\nOptions:"
            echo -e "  --complex_pdb PDB       Path to complex PDB file"
            echo -e "  --motif_chain CHAIN     Chain ID containing discontinuous motif"
            echo -e "  --motif_residues RES    Residue intervals (e.g. '310-313,338-347,594-596,612-625')"
            echo -e "  --target_chain CHAIN    Target chain ID (optional, for target context mode)"
            echo -e "  --min_total_len N       Minimum total scaffold length [Default: 70]"
            echo -e "  --max_total_len N       Maximum total scaffold length [Default: 95]"
            echo -e "  --num_designs N         Number of RFdiffusion backbones [Default: 50]"
            echo -e "  --mpnn_seqs_per_bb N    MPNN sequences per backbone [Default: 4]"
            echo -e "  --mpnn_batch_size N     ProteinMPNN batch size for GPU parallelization [Default: 4]"
            echo -e "  --colabfold_batch_size N Batch chunk size for ColabFold inputs (0=consolidated) [Default: 0]"
            echo -e "  --colabfold_num_recycles N Number of prediction recycles in ColabFold [Default: 3]"
            echo -e "  --colabfold_num_models N Number of prediction models in ColabFold [Default: 1]"
            echo -e "  --out_dir DIR           Output directory [Default: outputs/discontinuous_scaffold]"
            echo -e "  --plddt_cutoff VAL      pLDDT filter cutoff [Default: 80.0]"
            echo -e "  --rmsd_cutoff VAL       Motif RMSD filter cutoff [Default: 1.2]"
            echo -e "  --iptm_cutoff VAL       ipTM filter cutoff [Default: 0.70]"
            echo -e "  --dry_run               Run pipeline without executing GPU inference"
            echo -e "  --background, -b        Run pipeline detached in background via nohup"
            exit 0
            ;;
        --background|-b)
            BACKGROUND_MODE=true
            shift
            ;;
        --complex_pdb)
            COMPLEX_PDB="$2"
            shift 2
            ;;
        --motif_chain)
            MOTIF_CHAIN="$2"
            shift 2
            ;;
        --motif_residues)
            MOTIF_RESIDUES="$2"
            shift 2
            ;;
        --target_chain)
            TARGET_CHAIN="$2"
            shift 2
            ;;
        --min_total_len)
            MIN_LEN="$2"
            shift 2
            ;;
        --max_total_len)
            MAX_LEN="$2"
            shift 2
            ;;
        --num_designs)
            NUM_DESIGNS="$2"
            shift 2
            ;;
        --mpnn_seqs_per_bb)
            NUM_SEQS="$2"
            shift 2
            ;;
        --mpnn_batch_size)
            MPNN_BATCH_SIZE="$2"
            shift 2
            ;;
        --colabfold_batch_size)
            COLABFOLD_BATCH_SIZE="$2"
            shift 2
            ;;
        --colabfold_num_recycles)
            COLABFOLD_NUM_RECYCLES="$2"
            shift 2
            ;;
        --colabfold_num_models)
            COLABFOLD_NUM_MODELS="$2"
            shift 2
            ;;
        --out_dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --plddt_cutoff)
            PLDDT_CUTOFF="$2"
            shift 2
            ;;
        --rmsd_cutoff)
            RMSD_CUTOFF="$2"
            shift 2
            ;;
        --iptm_cutoff)
            IPTM_CUTOFF="$2"
            shift 2
            ;;
        --dry_run)
            DRY_RUN="--dry_run"
            shift
            ;;
        *)
            # Positional fallbacks: $1=PDB, $2=motif_chain, $3=motif_residues, $4=target_chain
            if [ -z "$COMPLEX_PDB" ]; then
                COMPLEX_PDB="$1"
            elif [ -z "$MOTIF_CHAIN" ]; then
                MOTIF_CHAIN="$1"
            elif [ -z "$MOTIF_RESIDUES" ]; then
                MOTIF_RESIDUES="$1"
            elif [ -z "$TARGET_CHAIN" ]; then
                TARGET_CHAIN="$1"
            fi
            shift
            ;;
    esac
done

# Interactive Prompts if arguments missing
IS_INTERACTIVE=false
if [ -z "$COMPLEX_PDB" ] && [ -t 0 ]; then
    IS_INTERACTIVE=true
fi

if [ -z "$COMPLEX_PDB" ]; then
    read -e -p "▶ 1. Path to Complex PDB: " COMPLEX_PDB
fi

if [[ "$COMPLEX_PDB" != *.pdb ]]; then
    echo -e "${RED}[ERROR] File must be in .pdb format: '$COMPLEX_PDB'${NC}"
    exit 1
fi
if [ ! -f "$COMPLEX_PDB" ]; then
    echo -e "${RED}[ERROR] File not found: '$COMPLEX_PDB'${NC}"
    exit 1
fi
ABS_COMPLEX_PDB=$(realpath "$COMPLEX_PDB" 2>/dev/null || echo "$COMPLEX_PDB")

if [ -z "$MOTIF_CHAIN" ]; then
    if [ -t 0 ]; then
        read -p "▶ 2. Motif Chain ID to scaffold (e.g. B): " MOTIF_CHAIN
    fi
fi
MOTIF_CHAIN=${MOTIF_CHAIN:-B}
if [[ ! "$MOTIF_CHAIN" =~ ^[a-zA-Z0-9]+$ ]]; then
    echo -e "${RED}[ERROR] Invalid Motif Chain ID '$MOTIF_CHAIN'. Must be alphanumeric.${NC}"
    exit 1
fi

if [ -z "$MOTIF_RESIDUES" ]; then
    if [ -t 0 ]; then
        read -p "▶ 3. Discontinuous Motif Residues (e.g. 310-313,338-347,594-596,612-625): " MOTIF_RESIDUES
    fi
fi
if [ -z "$MOTIF_RESIDUES" ]; then
    echo -e "${RED}[ERROR] Motif residues cannot be empty.${NC}"
    exit 1
fi
if [[ ! "$MOTIF_RESIDUES" =~ ^[0-9,\ -]+$ ]]; then
    echo -e "${RED}[ERROR] Invalid motif residues format '$MOTIF_RESIDUES'. Expected format like '310-313,338-347'.${NC}"
    exit 1
fi

if [ -z "$TARGET_CHAIN" ] && [ "$IS_INTERACTIVE" = true ]; then
    read -p "▶ 4. Target Chain ID [e.g. A, or leave empty for monomer scaffold]: " TARGET_CHAIN
fi
if [ -n "$TARGET_CHAIN" ] && [[ ! "$TARGET_CHAIN" =~ ^[a-zA-Z0-9]+$ ]]; then
    echo -e "${RED}[ERROR] Invalid Target Chain ID '$TARGET_CHAIN'. Must be alphanumeric.${NC}"
    exit 1
fi

if [ "$IS_INTERACTIVE" = true ]; then
    echo -e "\n${YELLOW}--- Scaffold Sizing & Budget ---${NC}"
    read -p "▶ 5. Minimum total scaffold length [Default: $MIN_LEN]: " INPUT_MIN_LEN
    MIN_LEN=${INPUT_MIN_LEN:-$MIN_LEN}

    read -p "▶ 6. Maximum total scaffold length [Default: $MAX_LEN]: " INPUT_MAX_LEN
    MAX_LEN=${INPUT_MAX_LEN:-$MAX_LEN}

    echo -e "\n${YELLOW}--- Sampling & Thresholds ---${NC}"
    read -p "▶ 7. Number of RFdiffusion backbones [Default: $NUM_DESIGNS]: " INPUT_NUM_DESIGNS
    NUM_DESIGNS=${INPUT_NUM_DESIGNS:-$NUM_DESIGNS}

    read -p "▶ 8. ProteinMPNN sequences per backbone [Default: $NUM_SEQS]: " INPUT_NUM_SEQS
    NUM_SEQS=${INPUT_NUM_SEQS:-$NUM_SEQS}

    read -e -p "▶ 9. Output Directory [Default: $OUT_DIR]: " INPUT_OUT_DIR
    OUT_DIR=${INPUT_OUT_DIR:-$OUT_DIR}

    read -p "▶ 10. Filter Cutoffs [pLDDT: $PLDDT_CUTOFF, Motif RMSD: $RMSD_CUTOFF, ipTM: $IPTM_CUTOFF] (press Enter to accept default): " CUSTOM_CUTOFFS
    if [ -n "$CUSTOM_CUTOFFS" ]; then
        read -p "      Minimum Scaffold pLDDT [$PLDDT_CUTOFF]: " INP_PLDDT
        read -p "      Maximum Motif RMSD (Å) [$RMSD_CUTOFF]: " INP_RMSD
        read -p "      Minimum ipTM [$IPTM_CUTOFF]: " INP_IPTM
        PLDDT_CUTOFF=${INP_PLDDT:-$PLDDT_CUTOFF}
        RMSD_CUTOFF=${INP_RMSD:-$RMSD_CUTOFF}
        IPTM_CUTOFF=${INP_IPTM:-$IPTM_CUTOFF}
    fi
fi

if [[ ! "$MIN_LEN" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "${RED}[ERROR] Invalid minimum length '$MIN_LEN'. Must be positive integer.${NC}"
    exit 1
fi
if [[ ! "$MAX_LEN" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "${RED}[ERROR] Invalid maximum length '$MAX_LEN'. Must be positive integer.${NC}"
    exit 1
fi
if [ "$MIN_LEN" -gt "$MAX_LEN" ]; then
    echo -e "${RED}[ERROR] Minimum length ($MIN_LEN) cannot exceed maximum length ($MAX_LEN).${NC}"
    exit 1
fi

if [[ ! "$NUM_DESIGNS" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "${RED}[ERROR] Invalid number of designs '$NUM_DESIGNS'. Must be positive integer.${NC}"
    exit 1
fi
if [[ ! "$NUM_SEQS" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "${RED}[ERROR] Invalid number of sequences per backbone '$NUM_SEQS'. Must be positive integer.${NC}"
    exit 1
fi
if [[ ! "$MPNN_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "${RED}[ERROR] Invalid MPNN batch size '$MPNN_BATCH_SIZE'. Must be positive integer.${NC}"
    exit 1
fi
if [[ ! "$COLABFOLD_BATCH_SIZE" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}[ERROR] Invalid ColabFold batch size '$COLABFOLD_BATCH_SIZE'. Must be non-negative integer.${NC}"
    exit 1
fi
if [[ ! "$COLABFOLD_NUM_RECYCLES" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "${RED}[ERROR] Invalid ColabFold recycles '$COLABFOLD_NUM_RECYCLES'. Must be positive integer.${NC}"
    exit 1
fi
if [[ ! "$COLABFOLD_NUM_MODELS" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "${RED}[ERROR] Invalid ColabFold models '$COLABFOLD_NUM_MODELS'. Must be positive integer.${NC}"
    exit 1
fi

for cutoff in "$PLDDT_CUTOFF" "$RMSD_CUTOFF" "$IPTM_CUTOFF"; do
    if [[ ! "$cutoff" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo -e "${RED}[ERROR] Invalid cutoff value '$cutoff'. Must be numeric.${NC}"
        exit 1
    fi
done

# ------------------------------------------------------------------------------
# 3. Execution
# ------------------------------------------------------------------------------
ABS_OUT_DIR=$(realpath "$OUT_DIR" 2>/dev/null || echo "$OUT_DIR")
if [ -z "$ABS_OUT_DIR" ] || [ "$ABS_OUT_DIR" = "/" ] || [ "$ABS_OUT_DIR" = "\\" ] || [[ "$ABS_OUT_DIR" =~ ^[a-zA-Z]:[\\/]?$ ]]; then
    echo -e "${RED}[ERROR] Invalid or dangerous output directory: '$OUT_DIR'${NC}"
    exit 1
fi
mkdir -p "$ABS_OUT_DIR"

PID_FILE="$ABS_OUT_DIR/run.pid"
LOG_FILE="$ABS_OUT_DIR/run.log"

# ------------------------------------------------------------------------------
# 2.5 Background Execution Option
# ------------------------------------------------------------------------------
if [ -t 0 ] && [ "$BACKGROUND_MODE" = false ]; then
    read -p "▶ Do you want to run this workflow in the background? [y/N]: " RUN_BG
    if [[ "$RUN_BG" =~ ^[Yy]$ ]]; then
        BACKGROUND_MODE=true
        if [ "$IS_INTERACTIVE" = true ]; then
            BG_ARGS=(
                "--complex_pdb" "$COMPLEX_PDB"
                "--motif_chain" "$MOTIF_CHAIN"
                "--motif_residues" "$MOTIF_RESIDUES"
                "--min_total_len" "$MIN_LEN"
                "--max_total_len" "$MAX_LEN"
                "--num_designs" "$NUM_DESIGNS"
                "--mpnn_seqs_per_bb" "$NUM_SEQS"
                "--mpnn_batch_size" "$MPNN_BATCH_SIZE"
                "--colabfold_batch_size" "$COLABFOLD_BATCH_SIZE"
                "--colabfold_num_recycles" "$COLABFOLD_NUM_RECYCLES"
                "--colabfold_num_models" "$COLABFOLD_NUM_MODELS"
                "--plddt_cutoff" "$PLDDT_CUTOFF"
                "--rmsd_cutoff" "$RMSD_CUTOFF"
                "--iptm_cutoff" "$IPTM_CUTOFF"
                "--out_dir" "$OUT_DIR"
            )
            [ -n "$TARGET_CHAIN" ] && BG_ARGS+=("--target_chain" "$TARGET_CHAIN")
            [ -n "$DRY_RUN" ] && BG_ARGS+=("$DRY_RUN")
        else
            BG_ARGS=("${ORIG_ARGS[@]}")
        fi
        nohup bash "$0" "${BG_ARGS[@]}" --background < /dev/null > "$LOG_FILE" 2>&1 &
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

PYTHON_ORCHESTRATOR="$SCRIPT_DIR/scripts/discontinuous_motif_scaffold.py"

CMD_ARGS=(
    "$PYTHON_ORCHESTRATOR"
    "--complex_pdb" "$ABS_COMPLEX_PDB"
    "--motif_chain" "$MOTIF_CHAIN"
    "--motif_residues" "$MOTIF_RESIDUES"
    "--min_total_len" "$MIN_LEN"
    "--max_total_len" "$MAX_LEN"
    "--num_designs" "$NUM_DESIGNS"
    "--mpnn_seqs_per_bb" "$NUM_SEQS"
    "--mpnn_batch_size" "$MPNN_BATCH_SIZE"
    "--colabfold_batch_size" "$COLABFOLD_BATCH_SIZE"
    "--colabfold_num_recycles" "$COLABFOLD_NUM_RECYCLES"
    "--colabfold_num_models" "$COLABFOLD_NUM_MODELS"
    "--out_dir" "$ABS_OUT_DIR"
    "--plddt_cutoff" "$PLDDT_CUTOFF"
    "--rmsd_cutoff" "$RMSD_CUTOFF"
    "--iptm_cutoff" "$IPTM_CUTOFF"
)

if [ -n "$TARGET_CHAIN" ]; then
    CMD_ARGS+=("--target_chain" "$TARGET_CHAIN")
fi

if [ -n "$DRY_RUN" ]; then
    CMD_ARGS+=("$DRY_RUN")
fi

echo -e "\n${CYAN}>>> Executing Discontinuous Motif Scaffolding Pipeline <<<${NC}\n"
python "${CMD_ARGS[@]}"

echo -e "\n${GREEN}[✓] Discontinuous scaffolding workflow finished successfully.${NC}"
echo -e "Results stored in: ${CYAN}${ABS_OUT_DIR}${NC}\n"
