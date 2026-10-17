#!/bin/bash
# autoscaffold.sh - Motif Scaffolding Pipeline for De Novo Active Site & Epitope Design
# Powered by RFdiffusion and ProteinMPNN
set -e

# Terminal colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Default installation paths (adjust to your environment)
RFDIFF_DIR="${RFDIFF_DIR:-$HOME/programs/RFdiffusion}"
MPNN_DIR="${MPNN_DIR:-$HOME/programs/ProteinMPNN}"
CONDA_BASE=$(conda info --base)

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN} 🏗️  AUTOSCAFFOLD: DE NOVO MOTIF SCAFFOLDING 🏗️       ${NC}"
echo -e "${CYAN}======================================================${NC}"
echo -e "Tip: Press ${YELLOW}Ctrl + C${NC} at any time to cancel.\n"

# 1. Input PDB verification
read -e -p "▶ 1. Path to source PDB (containing active site/motif): " INPUT_PDB
if [[ ! "$INPUT_PDB" == *.pdb ]]; then
    echo -e "${RED}[ERROR] File must be in .pdb format!${NC}"
    exit 1
fi
if [ ! -f "$INPUT_PDB" ]; then
    echo -e "${RED}[ERROR] File not found: '$INPUT_PDB'${NC}"
    exit 1
fi
ABS_INPUT_PDB=$(realpath "$INPUT_PDB")

# ==============================================================================
# TOPOLOGY CONTIG ASSISTANT
# ==============================================================================
echo -e "\n${YELLOW}--- Topology Assistant ---${NC}"
read -p "▶ 2. Chain ID of motif (e.g., A): " CHAIN
read -p "▶ 3. How many motif fragments will you preserve? (e.g., 3): " NUM_MOTIFS

CONTIGS=""
for (( i=1; i<=$NUM_MOTIFS; i++ ))
do
    echo -e "${CYAN}  -> Fragment $i${NC}"
    read -p "     Length of scaffold to generate BEFORE this motif (e.g., 25-30 or 0): " PRE_INV
    read -p "     Residue range to keep from motif (e.g., 118-122): " MOTIF_RES

    PRE_INV=${PRE_INV:-0}
    CONTIGS="${CONTIGS}${PRE_INV}/${CHAIN}${MOTIF_RES}/"
done

echo -e "${CYAN}  -> Scaffold Termination${NC}"
read -p "     Length of scaffold to generate at the END (e.g., 25-30 or 0): " POST_INV
POST_INV=${POST_INV:-0}
CONTIGS="${CONTIGS}${POST_INV}"

echo -e "${GREEN}  [✓] Generated contig formula: ${CONTIGS}${NC}"

# ==============================================================================
# GENERAL PARAMETERS
# ==============================================================================
read -p "▶ 4. Symmetry (e.g., C2 for dimer, press Enter for monomer) [Optional]: " SYMMETRY

echo -e "\n${YELLOW}--- Sequence & Expression Options ---${NC}"
read -p "▶ 5. Amino acids to omit (e.g., C) [Default: C]: " OMIT_AA
OMIT_AA=${OMIT_AA:-C}

read -e -p "▶ 6. Output directory: " OUT_DIR
read -p "▶ 7. Base prefix name (e.g., cutinase_novo): " OUT_PREFIX

echo -e "\n${YELLOW}--- Sampling & Generation Settings ---${NC}"
read -p "▶ 8. Number of 3D scaffolds (RFdiffusion) [Default: 10]: " NUM_DESIGNS
NUM_DESIGNS=${NUM_DESIGNS:-10}
read -p "▶ 9. ProteinMPNN temperature (0.1 conservative, 0.3 diverse) [Default: 0.1]: " MPNN_TEMP
MPNN_TEMP=${MPNN_TEMP:-0.1}
read -p "▶ 10. Sequences per scaffold (ProteinMPNN) [Default: 2]: " NUM_SEQS
NUM_SEQS=${NUM_SEQS:-2}

echo -e "\n${GREEN}[✓] Configuration complete. Preparing workspace...${NC}\n"

mkdir -p "$OUT_DIR/1_rfdiffusion"
mkdir -p "$OUT_DIR/2_mpnn_inputs/target_folder"
mkdir -p "$OUT_DIR/3_mpnn_outputs"
ABS_OUT_DIR=$(realpath "$OUT_DIR")

# ==============================================================================
# PHASE 1: RFdiffusion
# ==============================================================================
echo -e "${CYAN}>>> STARTING PHASE 1: RFdiffusion Backbone Generation <<<${NC}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate SE3nv
cd "$RFDIFF_DIR"

CMD="./scripts/run_inference.py inference.output_prefix=${ABS_OUT_DIR}/1_rfdiffusion/$OUT_PREFIX inference.input_pdb=${ABS_INPUT_PDB} 'contigmap.contigs=[$CONTIGS]' inference.num_designs=$NUM_DESIGNS"

if [ -n "$SYMMETRY" ]; then 
    CMD="$CMD inference.symmetry=$SYMMETRY"
fi

eval $CMD

BEST_MODEL="${ABS_OUT_DIR}/1_rfdiffusion/${OUT_PREFIX}_0.pdb"
TRB_FILE="${ABS_OUT_DIR}/1_rfdiffusion/${OUT_PREFIX}_0.trb"
if [ ! -f "$BEST_MODEL" ]; then
    echo -e "${RED}[CRITICAL ERROR] RFdiffusion failed to generate backbone models.${NC}"
    exit 1
fi

# ==============================================================================
# PHASE 1.5: Active Site Coordinate Extractor
# ==============================================================================
cat << 'EOF' > "${ABS_OUT_DIR}/2_mpnn_inputs/extract_fixed.py"
import sys, json, pickle

trb_file = sys.argv[1]
pdb_name = sys.argv[2]
out_jsonl = sys.argv[3]

with open(trb_file, 'rb') as f:
    trb = pickle.load(f)

# Extract 1-based residue indices corresponding to original fixed motif
fixed_idx = [int(i) + 1 for i in trb['con_hal_idx0']]
data = {pdb_name: {"A": fixed_idx}}

with open(out_jsonl, 'w') as f:
    json.dump(data, f)
    f.write('\n')
EOF

echo -e "Extracting active site coordinates from .trb file..."
python "${ABS_OUT_DIR}/2_mpnn_inputs/extract_fixed.py" "$TRB_FILE" "${OUT_PREFIX}_0" "${ABS_OUT_DIR}/2_mpnn_inputs/fixed.jsonl"

# ==============================================================================
# PHASE 2: ProteinMPNN Sequence Design
# ==============================================================================
echo -e "\n${CYAN}>>> STARTING PHASE 2: ProteinMPNN Sequence Design <<<${NC}"
conda activate mlfold
cd "$MPNN_DIR"

rm -f "${ABS_OUT_DIR}/2_mpnn_inputs/target_folder/"*.pdb
cp "$BEST_MODEL" "${ABS_OUT_DIR}/2_mpnn_inputs/target_folder/"

python helper_scripts/parse_multiple_chains.py \
    --input_path "${ABS_OUT_DIR}/2_mpnn_inputs/target_folder" \
    --output_path "${ABS_OUT_DIR}/2_mpnn_inputs/parsed.jsonl"

python protein_mpnn_run.py \
    --jsonl_path "${ABS_OUT_DIR}/2_mpnn_inputs/parsed.jsonl" \
    --fixed_positions_jsonl "${ABS_OUT_DIR}/2_mpnn_inputs/fixed.jsonl" \
    --out_folder "${ABS_OUT_DIR}/3_mpnn_outputs" \
    --num_seq_per_target $NUM_SEQS \
    --sampling_temp "$MPNN_TEMP" \
    --omit_AAs "$OMIT_AA" \
    --seed 37

echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN} ✨ SCAFFOLD DESIGN COMPLETED SUCCESSFULLY ✨         ${NC}"
echo -e "${GREEN}======================================================${NC}"
cat "${ABS_OUT_DIR}/3_mpnn_outputs/seqs/"*.fa

cd - > /dev/null
