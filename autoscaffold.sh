#!/usr/bin/env bash
# autoscaffold.sh - Motif Scaffolding Pipeline for De Novo Active Site & Epitope Design
# Powered by RFdiffusion and ProteinMPNN
set -e

# Trap Ctrl+C (SIGINT) cleanly
trap 'echo -e "\n\n${YELLOW}[!] Operation cancelled by user.${NC}"; exit 130' INT

# Resolve script directory and source shared environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/scripts/env_setup.sh"

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN} 🏗️  AUTOSCAFFOLD: DE NOVO MOTIF SCAFFOLDING 🏗️       ${NC}"
echo -e "${CYAN}======================================================${NC}"
echo -e "Tip: Press ${YELLOW}Ctrl + C${NC} at any time to cancel.\n"

# ------------------------------------------------------------------------------
# 1. Input PDB verification
# ------------------------------------------------------------------------------
if [ -n "$1" ]; then
    INPUT_PDB="$1"
else
    read -e -p "▶ 1. Path to source PDB (containing active site/motif): " INPUT_PDB
fi

if [[ "$INPUT_PDB" != *.pdb ]]; then
    echo -e "${RED}[ERROR] File must be in .pdb format!${NC}"
    exit 1
fi
if [ ! -f "$INPUT_PDB" ]; then
    echo -e "${RED}[ERROR] File not found: '$INPUT_PDB'${NC}"
    exit 1
fi
ABS_INPUT_PDB=$(realpath "$INPUT_PDB")

# ------------------------------------------------------------------------------
# 2. Topology Contig Assistant
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}--- Topology Assistant ---${NC}"
if [ -n "$2" ]; then
    CHAIN="$2"
else
    read -p "▶ 2. Chain ID of motif (e.g., A): " CHAIN
fi
CHAIN=${CHAIN:-A}
if [[ ! "$CHAIN" =~ ^[a-zA-Z0-9]+$ ]]; then
    echo -e "${RED}[ERROR] Invalid Chain ID '$CHAIN'. Must be alphanumeric.${NC}"
    exit 1
fi

if [ -n "$3" ]; then
    NUM_MOTIFS="$3"
else
    read -p "▶ 3. How many motif fragments will you preserve? (e.g., 3): " NUM_MOTIFS
fi
NUM_MOTIFS=${NUM_MOTIFS:-1}
if [[ ! "$NUM_MOTIFS" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "${RED}[ERROR] Invalid number of motifs '$NUM_MOTIFS'. Must be a positive integer.${NC}"
    exit 1
fi

CONTIGS=""
for (( i=1; i<=NUM_MOTIFS; i++ ))
do
    echo -e "${CYAN}  -> Fragment $i${NC}"
    read -p "     Length of scaffold to generate BEFORE this motif (e.g., 25-30 or 0): " PRE_INV
    read -p "     Residue range to keep from motif (e.g., 118-122): " MOTIF_RES

    PRE_INV=${PRE_INV:-0}
    if [[ ! "$PRE_INV" =~ ^[0-9]+(-[0-9]+)?$ ]]; then
        echo -e "${RED}[ERROR] Invalid scaffold length '$PRE_INV'. Must be an integer or range (e.g., 25-30 or 0).${NC}"
        exit 1
    fi
    if [[ ! "$MOTIF_RES" =~ ^[0-9]+-[0-9]+$ ]]; then
        echo -e "${RED}[ERROR] Invalid motif range '$MOTIF_RES'. Must be in format START-END (e.g., 118-122).${NC}"
        exit 1
    fi
    CONTIGS="${CONTIGS}${PRE_INV}/${CHAIN}${MOTIF_RES}/"
done

echo -e "${CYAN}  -> Scaffold Termination${NC}"
read -p "     Length of scaffold to generate at the END (e.g., 25-30 or 0): " POST_INV
POST_INV=${POST_INV:-0}
if [[ ! "$POST_INV" =~ ^[0-9]+(-[0-9]+)?$ ]]; then
    echo -e "${RED}[ERROR] Invalid scaffold termination length '$POST_INV'.${NC}"
    exit 1
fi
CONTIGS="${CONTIGS}${POST_INV}"

echo -e "${GREEN}  [✓] Generated contig formula: ${CONTIGS}${NC}"

# ------------------------------------------------------------------------------
# 3. General & Sampling Settings
# ------------------------------------------------------------------------------
read -p "▶ 4. Symmetry (e.g., C2 for dimer, press Enter for monomer) [Optional]: " SYMMETRY
if [ -n "$SYMMETRY" ] && [[ ! "$SYMMETRY" =~ ^[a-zA-Z0-9]+$ ]]; then
    echo -e "${RED}[ERROR] Invalid symmetry value '$SYMMETRY'. Must be alphanumeric.${NC}"
    exit 1
fi

echo -e "\n${YELLOW}--- Sequence & Expression Options ---${NC}"
read -p "▶ 5. Amino acids to omit (e.g., C) [Default: C]: " OMIT_AA
OMIT_AA=${OMIT_AA:-C}
if [[ ! "$OMIT_AA" =~ ^[a-zA-Z]*$ ]]; then
    echo -e "${RED}[ERROR] Invalid amino acid omission list '$OMIT_AA'.${NC}"
    exit 1
fi

read -e -p "▶ 6. Output directory [Default: ./output_motif]: " OUT_DIR
OUT_DIR=${OUT_DIR:-./output_motif}
read -p "▶ 7. Base prefix name (e.g., cutinase_novo): " OUT_PREFIX
OUT_PREFIX=${OUT_PREFIX:-scaffold_design}
if [[ ! "$OUT_PREFIX" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
    echo -e "${RED}[ERROR] Invalid output prefix '$OUT_PREFIX'. Use alphanumeric characters, dots, underscores or hyphens.${NC}"
    exit 1
fi

echo -e "\n${YELLOW}--- Sampling & Generation Settings ---${NC}"
read -p "▶ 8. Number of 3D scaffolds (RFdiffusion) [Default: 10]: " NUM_DESIGNS
NUM_DESIGNS=${NUM_DESIGNS:-10}
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

read -p "▶ 10. Sequences per scaffold (ProteinMPNN) [Default: 2]: " NUM_SEQS
NUM_SEQS=${NUM_SEQS:-2}
if [[ ! "$NUM_SEQS" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "${RED}[ERROR] Invalid sequence count '$NUM_SEQS'. Must be a positive integer.${NC}"
    exit 1
fi

echo -e "\n${GREEN}[✓] Configuration complete. Preparing workspace...${NC}\n"

mkdir -p "$OUT_DIR/1_rfdiffusion"
mkdir -p "$OUT_DIR/2_mpnn_inputs/target_folder"
mkdir -p "$OUT_DIR/3_mpnn_outputs"
ABS_OUT_DIR=$(realpath "$OUT_DIR")
if [ -z "$ABS_OUT_DIR" ] || [ "$ABS_OUT_DIR" = "/" ]; then
    echo -e "${RED}[ERROR] Invalid output directory path '$OUT_DIR'.${NC}"
    exit 1
fi

# ------------------------------------------------------------------------------
# PHASE 1: RFdiffusion Backbone Generation
# ------------------------------------------------------------------------------
echo -e "${CYAN}>>> STARTING PHASE 1: RFdiffusion Backbone Generation <<<${NC}"
activate_env "$CONDA_ENV_RFDIFF"

if [ ! -d "$RFDIFF_DIR" ] || [ ! -f "$RFDIFF_DIR/scripts/run_inference.py" ]; then
    echo -e "${RED}[ERROR] RFdiffusion script not found at: $RFDIFF_DIR/scripts/run_inference.py${NC}"
    echo -e "Please configure RFDIFF_DIR in config.env or run the dependency installer."
    exit 1
fi

CURRENT_PWD=$(pwd)
cd "$RFDIFF_DIR"

# Safe array-based argument expansion (avoiding eval)
RF_ARGS=(
    "./scripts/run_inference.py"
    "inference.output_prefix=${ABS_OUT_DIR}/1_rfdiffusion/$OUT_PREFIX"
    "inference.input_pdb=${ABS_INPUT_PDB}"
    "contigmap.contigs=[$CONTIGS]"
    "inference.num_designs=$NUM_DESIGNS"
)

if [ -n "$SYMMETRY" ]; then
    RF_ARGS+=("inference.symmetry=$SYMMETRY")
fi

echo -e "Executing RFdiffusion command..."
python "${RF_ARGS[@]}"

cd "$CURRENT_PWD"

# Find generated backbone models (bash globbing avoids find/sort subshell spawning)
shopt -s nullglob
GENERATED_PDBS=("${ABS_OUT_DIR}/1_rfdiffusion/${OUT_PREFIX}"_*.pdb)
shopt -u nullglob

if [ ${#GENERATED_PDBS[@]} -eq 0 ]; then
    echo -e "${RED}[CRITICAL ERROR] RFdiffusion failed to generate backbone models.${NC}"
    exit 1
fi

echo -e "${GREEN}[✓] RFdiffusion generated ${#GENERATED_PDBS[@]} backbone candidate(s).${NC}"
# Select best available model (starting with _0 or first)
BEST_MODEL="${GENERATED_PDBS[0]}"
MODEL_BASE="${BEST_MODEL##*/}"
MODEL_BASE="${MODEL_BASE%.pdb}"
TRB_FILE="${ABS_OUT_DIR}/1_rfdiffusion/${MODEL_BASE}.trb"

if [ ! -f "$TRB_FILE" ]; then
    echo -e "${YELLOW}[WARN] TRB file not found for ${MODEL_BASE}. Searching for any matching .trb...${NC}"
    shopt -s nullglob
    local_trbs=("${ABS_OUT_DIR}/1_rfdiffusion/"*.trb)
    shopt -u nullglob
    if [ ${#local_trbs[@]} -gt 0 ]; then
        TRB_FILE="${local_trbs[0]}"
    fi
fi

if [ -z "$TRB_FILE" ] || [ ! -f "$TRB_FILE" ]; then
    echo -e "${RED}[ERROR] No .trb metadata file found in ${ABS_OUT_DIR}/1_rfdiffusion.${NC}"
    exit 1
fi

# ------------------------------------------------------------------------------
# PHASE 1.5: Active Site Coordinate Extractor (Dynamic Chain ID)
# ------------------------------------------------------------------------------
echo -e "\n${CYAN}>>> Extracting active site coordinates (Chain ${CHAIN}) <<<${NC}"
EXTRACTOR_SCRIPT="$SCRIPT_DIR/scripts/extract_fixed_motif.py"
python "$EXTRACTOR_SCRIPT" "$TRB_FILE" "$MODEL_BASE" "$CHAIN" "${ABS_OUT_DIR}/2_mpnn_inputs/fixed.jsonl"

# ------------------------------------------------------------------------------
# PHASE 2: ProteinMPNN Sequence Design
# ------------------------------------------------------------------------------
echo -e "\n${CYAN}>>> STARTING PHASE 2: ProteinMPNN Sequence Design <<<${NC}"
activate_env "$CONDA_ENV_MPNN"

if [ ! -d "$MPNN_DIR" ] || [ ! -f "$MPNN_DIR/protein_mpnn_run.py" ]; then
    echo -e "${RED}[ERROR] ProteinMPNN not found at: $MPNN_DIR/protein_mpnn_run.py${NC}"
    echo -e "Please configure MPNN_DIR in config.env or run the dependency installer."
    exit 1
fi

# Populate target_folder with backbones to design
if [ -n "$ABS_OUT_DIR" ] && [ -d "${ABS_OUT_DIR}/2_mpnn_inputs/target_folder" ]; then
    rm -f "${ABS_OUT_DIR}/2_mpnn_inputs/target_folder/"*.pdb
fi
cp "$BEST_MODEL" "${ABS_OUT_DIR}/2_mpnn_inputs/target_folder/"

cd "$MPNN_DIR"

python helper_scripts/parse_multiple_chains.py \
    --input_path "${ABS_OUT_DIR}/2_mpnn_inputs/target_folder" \
    --output_path "${ABS_OUT_DIR}/2_mpnn_inputs/parsed.jsonl"

MPNN_ARGS=(
    "protein_mpnn_run.py"
    "--jsonl_path" "${ABS_OUT_DIR}/2_mpnn_inputs/parsed.jsonl"
    "--fixed_positions_jsonl" "${ABS_OUT_DIR}/2_mpnn_inputs/fixed.jsonl"
    "--out_folder" "${ABS_OUT_DIR}/3_mpnn_outputs"
    "--num_seq_per_target" "$NUM_SEQS"
    "--sampling_temp" "$MPNN_TEMP"
    "--omit_AAs" "$OMIT_AA"
    "--seed" "37"
)

python "${MPNN_ARGS[@]}"

cd "$CURRENT_PWD"

echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN} ✨ SCAFFOLD DESIGN COMPLETED SUCCESSFULLY ✨         ${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "Designed sequences:"
shopt -s nullglob
fa_files=("${ABS_OUT_DIR}/3_mpnn_outputs/seqs/"*.fa)
shopt -u nullglob
if [ ${#fa_files[@]} -gt 0 ]; then
    cat "${fa_files[@]}"
fi
echo -e "\nResults stored in: ${CYAN}${ABS_OUT_DIR}${NC}\n"
