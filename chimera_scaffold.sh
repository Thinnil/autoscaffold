#!/usr/bin/env bash
# chimera_scaffold.sh - Interactive & CLI Chimera Scaffolding Wizard
# Powered by RFdiffusion, ProteinMPNN, ColabFold, and PyMOL/Standalone extractor
set -e

# Trap Ctrl+C (SIGINT) cleanly
trap 'echo -e "\n\n${YELLOW}[!] Operation cancelled by user.${NC}"; exit 130' INT

# Resolve script directory and source shared environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/scripts/env_setup.sh"

echo -e "${CYAN}======================================================================${NC}"
echo -e "${CYAN} 🧬 AUTOSCAFFOLD: CHIMERA SCAFFOLDING WIZARD 🧬                         ${NC}"
echo -e "${CYAN} Graft & Ligate Active Motifs into Host Proteins (PDB or PyMOL .pse)  ${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo -e "Tip: Press ${YELLOW}Ctrl + C${NC} at any time to cancel.\n"

# ------------------------------------------------------------------------------
# 1. Parameter Parsing (CLI flags support)
# ------------------------------------------------------------------------------
INPUT_FILE=""
DONOR_ID=""
VESSEL_ID=""
MOTIF_RESIDUES=""
CHIMERA_MODE="replace"
VESSEL_REPLACE_RANGE=""
VESSEL_SPLIT_AFTER=""
LINKER_N="6-10"
LINKER_C="6-10"
NUM_DESIGNS="10"
NUM_SEQS="2"
MPNN_TEMP="0.1"
MPNN_BATCH_SIZE="1"
RMSD_THRESHOLD="1.0"
PLDDT_THRESHOLD="75.0"
OUT_DIR="outputs/chimera_scaffold"
DRY_RUN=""
LOCK_VESSEL_FLAG=""
BACKGROUND_MODE=false
ORIG_ARGS=("$@")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h|help)
            echo -e "${BOLD}Chimera Scaffolding Shell Wizard${NC}"
            echo -e "Usage:"
            echo -e "  $0 [options...]"
            echo -e "\nOptions:"
            echo -e "  --input FILE            Path to input file (.pdb or .pse)"
            echo -e "  --donor_id ID           Donor chain or object name (containing active motif)"
            echo -e "  --vessel_id ID          Vessel / recipient chain or object name"
            echo -e "  --motif_residues RES    Donor motif residues (e.g. '30-45' or '30-45,78-82')"
            echo -e "  --mode MODE             Chimera mode: replace, ligate_nterm, ligate_cterm, insert_loop [Default: replace]"
            echo -e "  --replace_range RANGE   Vessel residue range to excise (e.g. '55-70')"
            echo -e "  --split_after NUM       Vessel residue split point for insert_loop (e.g. 54)"
            echo -e "  --linker_n RANGE        N-terminal linker range [Default: 6-10]"
            echo -e "  --linker_c RANGE        C-terminal linker range [Default: 6-10]"
            echo -e "  --lock_vessel           Lock vessel framework residue positions in ProteinMPNN [Default: True]"
            echo -e "  --no_lock_vessel        Allow ProteinMPNN to redesign vessel framework residues"
            echo -e "  --num_designs N         Number of RFdiffusion backbones [Default: 10]"
            echo -e "  --num_seqs N            ProteinMPNN sequences per backbone [Default: 2]"
            echo -e "  --mpnn_temp TEMP        ProteinMPNN sampling temperature [Default: 0.1]"
            echo -e "  --mpnn_batch_size N     ProteinMPNN parallel GPU batch size [Default: 1]"
            echo -e "  --rmsd_threshold VAL    Kabsch RMSD cutoff in Å [Default: 1.0]"
            echo -e "  --plddt_threshold VAL   pLDDT quality cutoff [Default: 75.0]"
            echo -e "  --out_dir DIR           Output directory [Default: outputs/chimera_scaffold]"
            echo -e "  --dry_run               Execute pipeline without GPU inference"
            echo -e "  --background, -b        Run pipeline detached in background via nohup"
            exit 0
            ;;
        --background|-b)
            BACKGROUND_MODE=true
            shift
            ;;
        --input|-i)
            INPUT_FILE="$2"
            shift 2
            ;;
        --donor_id)
            DONOR_ID="$2"
            shift 2
            ;;
        --vessel_id)
            VESSEL_ID="$2"
            shift 2
            ;;
        --motif_residues|-m)
            MOTIF_RESIDUES="$2"
            shift 2
            ;;
        --mode)
            CHIMERA_MODE="$2"
            shift 2
            ;;
        --replace_range)
            VESSEL_REPLACE_RANGE="$2"
            shift 2
            ;;
        --split_after)
            VESSEL_SPLIT_AFTER="$2"
            shift 2
            ;;
        --linker_n)
            LINKER_N="$2"
            shift 2
            ;;
        --linker_c)
            LINKER_C="$2"
            shift 2
            ;;
        --num_designs)
            NUM_DESIGNS="$2"
            shift 2
            ;;
        --num_seqs)
            NUM_SEQS="$2"
            shift 2
            ;;
        --mpnn_temp)
            MPNN_TEMP="$2"
            shift 2
            ;;
        --mpnn_batch_size|--batch_size)
            MPNN_BATCH_SIZE="$2"
            shift 2
            ;;
        --rmsd_threshold)
            RMSD_THRESHOLD="$2"
            shift 2
            ;;
        --plddt_threshold)
            PLDDT_THRESHOLD="$2"
            shift 2
            ;;
        --out_dir|-o)
            OUT_DIR="$2"
            shift 2
            ;;
        --lock_vessel)
            LOCK_VESSEL_FLAG="--lock_vessel"
            shift
            ;;
        --no_lock_vessel)
            LOCK_VESSEL_FLAG="--no_lock_vessel"
            shift
            ;;
        --dry_run)
            DRY_RUN="--dry_run"
            shift
            ;;
        *)
            if [ -z "$INPUT_FILE" ]; then
                INPUT_FILE="$1"
            fi
            shift
            ;;
    esac
done

# ------------------------------------------------------------------------------
# 2. Interactive Input File Selection & Inspection
# ------------------------------------------------------------------------------
if [ -z "$INPUT_FILE" ]; then
    read -e -p "▶ 1. Path to input file (.pdb or PyMOL session .pse): " INPUT_FILE
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}[ERROR] Input file not found: '$INPUT_FILE'${NC}"
    exit 1
fi

ABS_INPUT_FILE=$(realpath "$INPUT_FILE" 2>/dev/null || echo "$INPUT_FILE")
echo -e "\n${CYAN}🔍 Inspecting structural units in '$INPUT_FILE'...${NC}"

if ! INSPECT_JSON=$(python "$SCRIPT_DIR/scripts/pse_pdb_extractor.py" inspect --input "$ABS_INPUT_FILE" 2>/dev/null); then
    echo -e "${RED}[ERROR] Failed to inspect structural units in '$ABS_INPUT_FILE'.${NC}"
    exit 1
fi

if ! echo "$INSPECT_JSON" | python -c "import sys, json; json.load(sys.stdin)" >/dev/null 2>&1; then
    echo -e "${RED}[ERROR] Received invalid JSON from structure inspector for '$ABS_INPUT_FILE'.${NC}"
    exit 1
fi

echo -e "\n${GREEN}Detected structural objects/chains:${NC}"
echo "$INSPECT_JSON" | python -c "
import sys, json
data = json.load(sys.stdin)
objs = data.get('objects', {})
idx = 1
for k, v in objs.items():
    name = v.get('name', k)
    length = v.get('length', 0)
    r_str = v.get('range_str', '')
    seq = v.get('sequence', '')
    seq_trunc = seq[:25] + '...' if len(seq) > 25 else seq
    print(f'  [{idx}] ID: {k:<8} Name: {name:<12} Length: {length:>4} aa  Range: {r_str:<10} Seq: {seq_trunc}')
    idx += 1
"

# ------------------------------------------------------------------------------
# 3. Interactive Structural Assignments
# ------------------------------------------------------------------------------
if [ -z "$DONOR_ID" ]; then
    echo -e ""
    read -p "▶ 2. Select Donor Object/Chain ID (contains desired active 3D motif): " DONOR_ID
fi
DONOR_ID=${DONOR_ID:-A}
if [[ ! "$DONOR_ID" =~ ^[a-zA-Z0-9_\-]+$ ]]; then
    echo -e "${RED}[ERROR] Invalid Donor ID '$DONOR_ID'. Must contain only alphanumeric characters, underscores, or hyphens.${NC}"
    exit 1
fi

if [ -z "$MOTIF_RESIDUES" ]; then
    echo -e "   Examples: '30-45', '30-45,78-82', 'A30-A45'"
    read -p "▶ 3. Specify Donor Residues to preserve in 3D: " MOTIF_RESIDUES
fi
if [ -z "$MOTIF_RESIDUES" ]; then
    echo -e "${RED}[ERROR] Motif residues cannot be empty.${NC}"
    exit 1
fi

if [ -z "$VESSEL_ID" ]; then
    read -p "▶ 4. Select Recipient / Vessel Object/Chain ID: " VESSEL_ID
fi
VESSEL_ID=${VESSEL_ID:-B}
if [[ ! "$VESSEL_ID" =~ ^[a-zA-Z0-9_\-]+$ ]]; then
    echo -e "${RED}[ERROR] Invalid Vessel ID '$VESSEL_ID'. Must contain only alphanumeric characters, underscores, or hyphens.${NC}"
    exit 1
fi

if [ -t 0 ] && [ "$CHIMERA_MODE" = "replace" ] && [ -z "$VESSEL_REPLACE_RANGE" ]; then
    echo -e "\n${YELLOW}--- Chimera Architecture Modes ---${NC}"
    echo -e "  [1] Graft / Replace: Replace a vessel loop/segment with donor motif"
    echo -e "  [2] Ligate to N-Terminus: Fuse donor motif to vessel N-terminus"
    echo -e "  [3] Ligate to C-Terminus: Fuse donor motif to vessel C-terminus"
    echo -e "  [4] Insert into Loop: Insert donor motif at an internal split point"
    read -p "▶ 5. Choose Mode [1/2/3/4, Default: 1]: " MODE_CHOICE
    MODE_CHOICE=${MODE_CHOICE:-1}

    case "$MODE_CHOICE" in
        1)
            CHIMERA_MODE="replace"
            read -p "     Vessel range to excise / replace (e.g. 55-70): " VESSEL_REPLACE_RANGE
            ;;
        2)
            CHIMERA_MODE="ligate_nterm"
            ;;
        3)
            CHIMERA_MODE="ligate_cterm"
            ;;
        4)
            CHIMERA_MODE="insert_loop"
            read -p "     Split vessel after residue number (e.g. 54): " VESSEL_SPLIT_AFTER
            ;;
    esac
fi

if [ -t 0 ]; then
    echo -e "\n${YELLOW}--- Linker Flexibility & Sampling Budget ---${NC}"
    read -p "▶ 6. N-terminal linker length [Default: $LINKER_N]: " INP_LN
    LINKER_N=${INP_LN:-$LINKER_N}
    read -p "▶ 7. C-terminal linker length [Default: $LINKER_C]: " INP_LC
    LINKER_C=${INP_LC:-$LINKER_C}
    read -p "▶ 8. Number of RFdiffusion backbones [Default: $NUM_DESIGNS]: " INP_ND
    NUM_DESIGNS=${INP_ND:-$NUM_DESIGNS}
    read -p "▶ 9. ProteinMPNN sequences per backbone [Default: $NUM_SEQS]: " INP_NS
    NUM_SEQS=${INP_NS:-$NUM_SEQS}
    read -p "▶ 10. ProteinMPNN parallel GPU batch size [Default: $MPNN_BATCH_SIZE]: " INP_BS
    MPNN_BATCH_SIZE=${INP_BS:-$MPNN_BATCH_SIZE}
    read -e -p "▶ 11. Output directory [Default: $OUT_DIR]: " INP_OD
    OUT_DIR=${INP_OD:-$OUT_DIR}
fi

ABS_OUT_DIR=$(realpath "$OUT_DIR" 2>/dev/null || echo "$OUT_DIR")
if [ -z "$ABS_OUT_DIR" ] || [ "$ABS_OUT_DIR" = "/" ] || [ "$ABS_OUT_DIR" = "\\" ] || [[ "$ABS_OUT_DIR" =~ ^[a-zA-Z]:[\\/]?$ ]]; then
    echo -e "${RED}[ERROR] Invalid or dangerous output directory specified: '$OUT_DIR'.${NC}"
    exit 1
fi
mkdir -p "$ABS_OUT_DIR"

PID_FILE="$ABS_OUT_DIR/run.pid"
LOG_FILE="$ABS_OUT_DIR/run.log"

# ------------------------------------------------------------------------------
# 3.5 Background Execution Option
# ------------------------------------------------------------------------------
if [ -t 0 ] && [ "$BACKGROUND_MODE" = false ]; then
    read -p "▶ Do you want to run this workflow in the background? [y/N]: " RUN_BG
    if [[ "$RUN_BG" =~ ^[Yy]$ ]]; then
        BACKGROUND_MODE=true
        if [ ${#ORIG_ARGS[@]} -eq 0 ]; then
            BG_ARGS=(
                "--input" "$ABS_INPUT_FILE"
                "--donor_id" "$DONOR_ID"
                "--vessel_id" "$VESSEL_ID"
                "--motif_residues" "$MOTIF_RESIDUES"
                "--mode" "$CHIMERA_MODE"
                "--linker_n" "$LINKER_N"
                "--linker_c" "$LINKER_C"
                "--num_designs" "$NUM_DESIGNS"
                "--num_seqs" "$NUM_SEQS"
                "--mpnn_temp" "$MPNN_TEMP"
                "--mpnn_batch_size" "$MPNN_BATCH_SIZE"
                "--rmsd_threshold" "$RMSD_THRESHOLD"
                "--plddt_threshold" "$PLDDT_THRESHOLD"
                "--out_dir" "$ABS_OUT_DIR"
            )
            [ -n "$VESSEL_REPLACE_RANGE" ] && BG_ARGS+=("--replace_range" "$VESSEL_REPLACE_RANGE")
            [ -n "$VESSEL_SPLIT_AFTER" ] && BG_ARGS+=("--split_after" "$VESSEL_SPLIT_AFTER")
            [ -n "$DRY_RUN" ] && BG_ARGS+=("$DRY_RUN")
            [ -n "$LOCK_VESSEL_FLAG" ] && BG_ARGS+=("$LOCK_VESSEL_FLAG")
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

# ------------------------------------------------------------------------------
# 4. Pipeline Execution
# ------------------------------------------------------------------------------
echo -e "\n${CYAN}>>> Executing Chimera Scaffolding Pipeline <<<${NC}\n"

CMD_ARGS=(
    "python" "$SCRIPT_DIR/scripts/chimera_scaffold.py"
    "--input_file" "$ABS_INPUT_FILE"
    "--donor_id" "$DONOR_ID"
    "--vessel_id" "$VESSEL_ID"
    "--motif_residues" "$MOTIF_RESIDUES"
    "--chimera_mode" "$CHIMERA_MODE"
    "--linker_n" "$LINKER_N"
    "--linker_c" "$LINKER_C"
    "--num_designs" "$NUM_DESIGNS"
    "--mpnn_seqs_per_bb" "$NUM_SEQS"
    "--mpnn_temp" "$MPNN_TEMP"
    "--mpnn_batch_size" "$MPNN_BATCH_SIZE"
    "--rmsd_threshold" "$RMSD_THRESHOLD"
    "--plddt_threshold" "$PLDDT_THRESHOLD"
    "--out_dir" "$ABS_OUT_DIR"
)

if [ -n "$VESSEL_REPLACE_RANGE" ]; then
    CMD_ARGS+=("--vessel_replace_range" "$VESSEL_REPLACE_RANGE")
fi

if [ -n "$VESSEL_SPLIT_AFTER" ]; then
    CMD_ARGS+=("--vessel_split_after" "$VESSEL_SPLIT_AFTER")
fi

if [ -n "$DRY_RUN" ]; then
    CMD_ARGS+=("$DRY_RUN")
fi

if [ -n "$LOCK_VESSEL_FLAG" ]; then
    CMD_ARGS+=("$LOCK_VESSEL_FLAG")
fi

# Activate environment if configured
if [ -n "$CONDA_ENV_RFDIFF" ]; then
    activate_env "$CONDA_ENV_RFDIFF" 2>/dev/null || true
fi

"${CMD_ARGS[@]}"

echo -e "\n${GREEN}[✓] Chimera Scaffolding completed successfully.${NC}"
echo -e "Results stored in: ${CYAN}${ABS_OUT_DIR}${NC}\n"
