#!/bin/bash
# de_novo.sh - Target-Binder Design and ColabFold Validation Pipeline
set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

RFDIFF_DIR="${RFDIFF_DIR:-$HOME/programs/RFdiffusion}"
MPNN_DIR="${MPNN_DIR:-$HOME/programs/ProteinMPNN}"
CONDA_BASE=$(conda info --base)

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}   🧬 DE NOVO BINDER DESIGN & VALIDATION PIPELINE 🧬  ${NC}"
echo -e "${CYAN}======================================================${NC}"
echo -e "Tip: Press ${YELLOW}Ctrl + C${NC} at any time to cancel.\n"

# ==============================================================================
# DATA COLLECTION & VALIDATION
# ==============================================================================
read -e -p "▶ 1. Path to Target PDB file: " INPUT_PDB

if [[ ! "$INPUT_PDB" == *.pdb ]]; then
    echo -e "${RED}[ERROR] Target must be in .pdb format.${NC}"
    exit 1
fi

if [ ! -f "$INPUT_PDB" ]; then
    echo -e "${RED}[ERROR] File not found: '$INPUT_PDB'${NC}"
    exit 1
fi

ABS_INPUT_PDB=$(realpath "$INPUT_PDB")

read -p "▶ 2. Target Chain ID to preserve (e.g., A): " CHAIN_A
read -p "▶ 3. Target residue range (e.g., 20-97. Press ENTER for FULL chain): " RES_RANGE

if [[ "$RES_RANGE" == *"/"* ]]; then
    echo -e "${RED}[ERROR] Slashes are not allowed. Use hyphen syntax (e.g., 20-97).${NC}"
    exit 1
fi

# Auto-detect chain boundaries if empty
if [ -z "$RES_RANGE" ]; then
    echo -e "${CYAN}🔍 Analyzing PDB to detect chain ${CHAIN_A} boundaries...${NC}"

    MIN_RES=$(awk -v ch="$CHAIN_A" '/^ATOM/ && substr($0,22,1)==ch {print int(substr($0,23,4))}' "$ABS_INPUT_PDB" | sort -n | head -n 1)
    MAX_RES=$(awk -v ch="$CHAIN_A" '/^ATOM/ && substr($0,22,1)==ch {print int(substr($0,23,4))}' "$ABS_INPUT_PDB" | sort -n | tail -n 1)

    if [ -z "$MIN_RES" ] || [ -z "$MAX_RES" ]; then
        echo -e "${RED}[ERROR] No ATOM records found for chain '${CHAIN_A}' in PDB.${NC}"
        exit 1
    fi

    RES_RANGE="${MIN_RES}-${MAX_RES}"
    echo -e "${GREEN}[✓] Detected range: ${RES_RANGE}${NC}\n"
fi

read -p "▶ 4. Binder length range (e.g., 50-65): " BINDER_LEN
read -p "▶ 5. Hotspot residues separated by comma (e.g., A8,A44. Enter to skip): " HOTSPOTS
read -e -p "▶ 6. Output directory: " OUT_DIR
read -p "▶ 7. Base prefix name (e.g., my_binder): " OUT_PREFIX

echo -e "\n${YELLOW}--- Advanced Settings ---${NC}"
read -p "▶ 8. Number of 3D backbone designs (RFdiffusion) [Default: 3]: " NUM_DESIGNS
NUM_DESIGNS=${NUM_DESIGNS:-3}
read -p "▶ 9. ProteinMPNN temperature (0.1 conservative, 0.3 diverse) [Default: 0.1]: " MPNN_TEMP
MPNN_TEMP=${MPNN_TEMP:-0.1}
read -p "▶ 10. Sequences per backbone (ProteinMPNN) [Default: 2]: " NUM_SEQS
NUM_SEQS=${NUM_SEQS:-2}

echo -e "\n${GREEN}[✓] Parameters validated. Creating directories...${NC}\n"

# Directory layout
mkdir -p "$OUT_DIR/1_rfdiffusion"
mkdir -p "$OUT_DIR/2_mpnn_inputs/target_folder"
mkdir -p "$OUT_DIR/3_mpnn_outputs"
mkdir -p "$OUT_DIR/4_colabfold_validation"

ABS_OUT_DIR=$(realpath "$OUT_DIR")

# ==============================================================================
# PHASE 1: RFdiffusion
# ==============================================================================
echo -e "${CYAN}>>> STARTING PHASE 1: RFdiffusion Binder Generation <<<${NC}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate SE3nv
cd "$RFDIFF_DIR"

CMD="./scripts/run_inference.py inference.output_prefix=${ABS_OUT_DIR}/1_rfdiffusion/$OUT_PREFIX inference.input_pdb=${ABS_INPUT_PDB} 'contigmap.contigs=[${CHAIN_A}${RES_RANGE}/0 ${BINDER_LEN}]' inference.num_designs=$NUM_DESIGNS"

if [ -n "$HOTSPOTS" ]; then 
    CMD="$CMD 'ppi.hotspot_res=[$HOTSPOTS]'"
fi

eval $CMD

BEST_MODEL="${ABS_OUT_DIR}/1_rfdiffusion/${OUT_PREFIX}_0.pdb"
if [ ! -f "$BEST_MODEL" ]; then
    echo -e "${RED}[CRITICAL ERROR] RFdiffusion failed to generate designs.${NC}"
    exit 1
fi
echo -e "${GREEN}[✓] 3D Backbones generated successfully.${NC}\n"

# ==============================================================================
# PHASE 2: ProteinMPNN
# ==============================================================================
echo -e "${CYAN}>>> STARTING PHASE 2: ProteinMPNN Binder Sequence Design <<<${NC}"
conda activate mlfold
cd "$MPNN_DIR"

cp "$BEST_MODEL" "${ABS_OUT_DIR}/2_mpnn_inputs/target_folder/"

python helper_scripts/parse_multiple_chains.py \
    --input_path "${ABS_OUT_DIR}/2_mpnn_inputs/target_folder" \
    --output_path "${ABS_OUT_DIR}/2_mpnn_inputs/parsed.jsonl"

python helper_scripts/assign_fixed_chains.py \
    --input_path "${ABS_OUT_DIR}/2_mpnn_inputs/parsed.jsonl" \
    --output_path "${ABS_OUT_DIR}/2_mpnn_inputs/assigned.jsonl" \
    --chain_list "B"

python protein_mpnn_run.py \
    --jsonl_path "${ABS_OUT_DIR}/2_mpnn_inputs/parsed.jsonl" \
    --chain_id_jsonl "${ABS_OUT_DIR}/2_mpnn_inputs/assigned.jsonl" \
    --out_folder "${ABS_OUT_DIR}/3_mpnn_outputs" \
    --num_seq_per_target $NUM_SEQS \
    --sampling_temp "$MPNN_TEMP" \
    --seed 37

echo -e "${GREEN}[✓] Binder sequences designed.${NC}\n"

# ==============================================================================
# PHASE 3: ColabFold (Validation and Cleanup)
# ==============================================================================
echo -e "${CYAN}>>> STARTING PHASE 3: ColabFold Multimer Validation <<<${NC}"
conda activate colabfold

MPNN_FASTA=$(ls "${ABS_OUT_DIR}/3_mpnn_outputs/seqs/"*.fa | head -n 1)
VALIDATION_FASTA="${ABS_OUT_DIR}/4_colabfold_validation/final_complex.fa"

RAW_SEQUENCE=$(grep -m 1 -A 1 "sample=1" "$MPNN_FASTA" | tail -n 1)

if [ ${#RAW_SEQUENCE} -lt 10 ]; then
    echo -e "${RED}[ERROR] Extracted sequence is empty or incomplete.${NC}"
    exit 1
fi

TARGET_SEQ=$(grep -o "\"seq_chain_${CHAIN_A}\": \"[^\"]*\"" "${ABS_OUT_DIR}/2_mpnn_inputs/parsed.jsonl" | head -n 1 | cut -d'"' -f4)

SEQ_BINDER=$RAW_SEQUENCE
if [ ${#RAW_SEQUENCE} -gt $BINDER_LEN ]; then
    LEN_TARGET=${#TARGET_SEQ}
    SEQ_BINDER=${RAW_SEQUENCE:$LEN_TARGET}
fi

MULTIMER_SEQUENCE="${TARGET_SEQ}:${SEQ_BINDER}"

echo ">Target_Binder_Complex" > "$VALIDATION_FASTA"
echo "$MULTIMER_SEQUENCE" >> "$VALIDATION_FASTA"

echo -e "${YELLOW}Running ColabFold (AlphaFold-Multimer)...${NC}"
colabfold_batch "$VALIDATION_FASTA" "${ABS_OUT_DIR}/4_colabfold_validation/"

if ! ls "${ABS_OUT_DIR}/4_colabfold_validation/"*rank_001*.pdb 1> /dev/null 2>&1; then
    echo -e "${RED}[CRITICAL ERROR] ColabFold failed to generate top-ranked PDB model.${NC}"
    exit 1
fi

# Cleanup secondary models to save disk space
echo -e "${YELLOW}Cleaning up sub-optimal models (rank 002-005)...${NC}"
find "${ABS_OUT_DIR}/4_colabfold_validation/" -type f \( -name "*rank_00[2-5]*.pdb" -o -name "*rank_00[2-5]*.json" \) -delete 2>/dev/null

echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN} 🏆 PIPELINE COMPLETED SUCCESSFULLY 🏆                ${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "Validated complex files located in:"
echo -e "${CYAN}${ABS_OUT_DIR}/4_colabfold_validation/${NC}\n"

cd - > /dev/null
