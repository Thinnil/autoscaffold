#!/usr/bin/env bash
# chimera_scaffold.sh - Interactive & CLI Chimera Scaffolding Wizard
# Powered by RFdiffusion, ProteinMPNN, ColabFold, and PyMOL/Standalone extractor
set -e

# Trap Ctrl+C (SIGINT) cleanly
trap 'echo -e "\n\n${YELLOW}[!] Operation cancelled by user.${NC}"; exit 130' INT

# Resolve script directory and source shared environment
if [ -n "$AUTOSCAFFOLD_HOME" ] && [ -f "$AUTOSCAFFOLD_HOME/chimera_scaffold.sh" ]; then
    SCRIPT_DIR="$AUTOSCAFFOLD_HOME"
else
    SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
fi
if [ -z "$SCRIPT_DIR" ] || [ "$SCRIPT_DIR" = "/" ] || [ "$SCRIPT_DIR" = "\\" ] || [[ "$SCRIPT_DIR" =~ ^[a-zA-Z]:[\\/]?$ ]]; then
    echo -e "\033[0;31m[ERROR] Dangerous or invalid SCRIPT_DIR: '$SCRIPT_DIR'\033[0m"
    exit 1
fi

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
        chmod 755 "$SCRIPT_DIR"/scripts/*.sh "$SCRIPT_DIR"/scripts/*.py 2>/dev/null || true
    fi
    if [ ! -f "$SCRIPT_DIR/scripts/env_setup.sh" ]; then
        echo -e "\033[0;31m[ERROR] Could not locate or download 'scripts/env_setup.sh'.\033[0m"
        exit 1
    fi
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/scripts/env_setup.sh"

# Fast canonical path resolution minimizing external process forks
fast_realpath() {
    local target="$1"
    if [ -z "$target" ]; then
        echo ""
        return
    fi
    if [[ "$target" =~ ^/ || "$target" =~ ^[a-zA-Z]:[\\/] ]]; then
        echo "$target"
        return
    fi
    if command -v realpath >/dev/null 2>&1; then
        realpath "$target" 2>/dev/null && return
    elif command -v readlink >/dev/null 2>&1; then
        readlink -f "$target" 2>/dev/null && return
    fi
    if [ -d "$target" ]; then
        (cd -P "$target" 2>/dev/null && echo "$PWD")
    elif [ -f "$target" ]; then
        local dir="${target%/*}"
        local file="${target##*/}"
        [ "$dir" = "$target" ] && dir="."
        local abs_dir
        abs_dir="$(cd -P "$dir" 2>/dev/null && echo "$PWD")"
        if [ -n "$abs_dir" ]; then
            echo "$abs_dir/$file"
        else
            echo "$target"
        fi
    else
        echo "$target"
    fi
}

# De-duplicate accidental nested 'outputs/outputs' in paths and anchor default relative output
canonicalize_out_dir() {
    local raw="$1"
    local repo_root="$2"
    local caller_dir="${AUTOSCAFFOLD_CALLER_PWD:-$PWD}"
    local resolved=""

    if [ -z "$raw" ]; then
        raw="outputs/chimera_scaffold"
    fi

    # If user passed default or relative path while already inside an 'outputs' directory
    if [[ "$caller_dir" =~ [/\\]outputs$ ]] && [[ "$raw" =~ ^outputs[/\\] ]]; then
        raw="${raw#outputs/}"
        raw="${raw#outputs\\}"
    fi

    # Anchor relative path strictly to caller's working directory ($AUTOSCAFFOLD_CALLER_PWD or $PWD)
    if [[ ! "$raw" =~ ^/ && ! "$raw" =~ ^[a-zA-Z]:[\\/] ]]; then
        resolved="$caller_dir/$raw"
    else
        resolved="$raw"
    fi

    # Collapse any accidental duplicated '/outputs/outputs/'
    while [[ "$resolved" =~ /outputs/outputs/ ]]; do
        resolved="${resolved//\/outputs\/outputs\//\/outputs\/}"
    done
    while [[ "$resolved" =~ \\outputs\\outputs\\ ]]; do
        resolved="${resolved//\\outputs\\outputs\\/\\outputs\\}"
    done
    while [[ "$resolved" =~ /outputs/outputs$ ]]; do
        resolved="${resolved%/outputs}"
    done
    while [[ "$resolved" =~ \\outputs\\outputs$ ]]; do
        resolved="${resolved%\\outputs}"
    done

    echo "$resolved"
}

echo -e "${CYAN}======================================================================${NC}"
echo -e "${CYAN} 🧬 AUTOSCAFFOLD: CHIMERA SCAFFOLDING WIZARD 🧬                         ${NC}"
echo -e "${CYAN} Graft & Ligate Active Motifs into Host Proteins (PDB or PyMOL .pse)  ${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo -e "Tip: Press ${YELLOW}Ctrl + C${NC} at any time to cancel.\n"

# ------------------------------------------------------------------------------
# 1. Parameter Parsing (CLI flags support)
# ------------------------------------------------------------------------------
INPUT_FILE=""
DONOR_PDB=""
VESSEL_PDB=""
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
PYTHON_RFDIFF="${PYTHON_RFDIFF:-}"
PYTHON_MPNN="${PYTHON_MPNN:-}"
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
            echo -e "  --input_file FILE       Alias for --input"
            echo -e "  --donor_pdb FILE        Path to standalone donor PDB"
            echo -e "  --vessel_pdb FILE       Path to standalone vessel PDB"
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
            echo -e "  --python_rfdiff PATH    Python interpreter for RFdiffusion"
            echo -e "  --python_mpnn PATH      Python interpreter for ProteinMPNN"
            echo -e "  --out_dir DIR           Output directory [Default: outputs/chimera_scaffold]"
            echo -e "  --dry_run               Execute pipeline without GPU inference"
            echo -e "  --background, -b        Run pipeline detached in background via nohup"
            exit 0
            ;;
        --background|-b)
            BACKGROUND_MODE=true
            shift
            ;;
        --input|--input_file|-i)
            INPUT_FILE="$2"
            shift 2
            ;;
        --donor_pdb)
            DONOR_PDB="$2"
            shift 2
            ;;
        --vessel_pdb)
            VESSEL_PDB="$2"
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
        --python_rfdiff)
            PYTHON_RFDIFF="$2"
            shift 2
            ;;
        --python_mpnn)
            PYTHON_MPNN="$2"
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
if [ -z "$INPUT_FILE" ] && ([ -z "$DONOR_PDB" ] || [ -z "$VESSEL_PDB" ]); then
    read -r -e -p "▶ 1. Path to input file (.pdb or PyMOL session .pse): " INPUT_FILE
fi

# Canonicalize standalone donor/vessel PDB files if provided using fast path resolution
if [ -n "$DONOR_PDB" ]; then
    if [ ! -f "$DONOR_PDB" ]; then
        echo -e "${RED}[ERROR] Donor PDB file not found: '$DONOR_PDB'${NC}"
        exit 1
    fi
    DONOR_PDB="$(fast_realpath "$DONOR_PDB")"
fi
if [ -n "$VESSEL_PDB" ]; then
    if [ ! -f "$VESSEL_PDB" ]; then
        echo -e "${RED}[ERROR] Vessel PDB file not found: '$VESSEL_PDB'${NC}"
        exit 1
    fi
    VESSEL_PDB="$(fast_realpath "$VESSEL_PDB")"
fi

if [ -n "$INPUT_FILE" ]; then
    if [ ! -f "$INPUT_FILE" ]; then
        echo -e "${RED}[ERROR] Input file not found: '$INPUT_FILE'${NC}"
        exit 1
    fi
    ABS_INPUT_FILE="$(fast_realpath "$INPUT_FILE")"
    echo -e "\n${CYAN}🔍 Inspecting structural units in '$INPUT_FILE'...${NC}"

    if ! INSPECT_JSON=$(python "$SCRIPT_DIR/scripts/pse_pdb_extractor.py" inspect --input "$ABS_INPUT_FILE" 2>/dev/null); then
        echo -e "${RED}[ERROR] Failed to inspect structural units in '$ABS_INPUT_FILE'.${NC}"
        exit 1
    fi

    echo -e "\n${GREEN}Detected structural objects/chains:${NC}"
    if ! echo "$INSPECT_JSON" | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(1)
objs = data.get('objects', {})
for idx, (k, v) in enumerate(objs.items(), 1):
    name = v.get('name', k)
    length = v.get('length', 0)
    r_str = v.get('range_str', '')
    seq = v.get('sequence', '')
    seq_trunc = seq[:25] + '...' if len(seq) > 25 else seq
    print(f'  [{idx}] ID: {k:<8} Name: {name:<12} Length: {length:>4} aa  Range: {r_str:<10} Seq: {seq_trunc}')
"; then
        echo -e "${RED}[ERROR] Received invalid JSON from structure inspector for '$ABS_INPUT_FILE'.${NC}"
        exit 1
    fi
fi

# ------------------------------------------------------------------------------
# 3. Interactive Structural Assignments
# ------------------------------------------------------------------------------
if [ -z "$DONOR_ID" ]; then
    echo -e ""
    read -r -p "▶ 2. Select Donor Object/Chain ID(s) [e.g. '6,7' or 'HSP_contact,contacts_objB']: " DONOR_ID
fi
DONOR_ID=${DONOR_ID:-A}
if [ -n "$INSPECT_JSON" ]; then
    RESOLVED_DONOR=$(echo "$INSPECT_JSON" | python -c "
import sys, json
raw_donor = sys.argv[1].strip()
try:
    data = json.load(sys.stdin)
    objs = list(data.get('objects', {}).keys())
    parts = [p.strip() for p in raw_donor.split(',') if p.strip()]
    resolved_parts = []
    for p in parts:
        if p.isdigit():
            idx = int(p) - 1
            if 0 <= idx < len(objs):
                resolved_parts.append(objs[idx])
            else:
                resolved_parts.append(p)
        else:
            resolved_parts.append(p)
    print(','.join(resolved_parts))
except Exception:
    print(raw_donor)
" "$DONOR_ID" 2>/dev/null || true)
    [ -n "$RESOLVED_DONOR" ] && DONOR_ID="$RESOLVED_DONOR"
fi

if [[ ! "$DONOR_ID" =~ ^[-a-zA-Z0-9_,\ ]+$ ]]; then
    echo -e "${RED}[ERROR] Invalid Donor ID '$DONOR_ID'. Must contain only alphanumeric characters, underscores, hyphens, or commas.${NC}"
    exit 1
fi
echo -e "  ${GREEN}[✓] Selected Donor:${NC} ${BOLD}$DONOR_ID${NC}"

if [ -z "$MOTIF_RESIDUES" ]; then
    echo -e "   Examples: 'all' (preserve entire donor structure), '30-45', '30-45,78-82'"
    read -r -p "▶ 3. Specify Donor Residues to preserve in 3D [Default: all]: " MOTIF_RESIDUES
fi
MOTIF_RESIDUES=${MOTIF_RESIDUES:-all}
if [ -z "$MOTIF_RESIDUES" ]; then
    echo -e "${RED}[ERROR] Motif residues cannot be empty.${NC}"
    exit 1
fi
if [[ ! "$MOTIF_RESIDUES" =~ ^[-a-zA-Z0-9,:\ \*%]+$ ]]; then
    echo -e "${RED}[ERROR] Invalid motif residues specification '$MOTIF_RESIDUES'.${NC}"
    exit 1
fi
echo -e "  ${GREEN}[✓] Preserved Motif:${NC} ${BOLD}$MOTIF_RESIDUES${NC}"

if [ -z "$VESSEL_ID" ]; then
    read -r -p "▶ 4. Select Recipient / Vessel Object/Chain ID [e.g. 1 or 7ZUB_chain_A]: " VESSEL_ID
fi
VESSEL_ID=${VESSEL_ID:-B}
if [ -n "$INSPECT_JSON" ] && [[ "$VESSEL_ID" =~ ^[0-9]+$ ]]; then
    RESOLVED_VESSEL=$(echo "$INSPECT_JSON" | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    objs = list(data.get('objects', {}).keys())
    idx = int(sys.argv[1]) - 1
    if 0 <= idx < len(objs):
        print(objs[idx])
    else:
        print(sys.argv[1])
except Exception:
    print(sys.argv[1])
" "$VESSEL_ID" 2>/dev/null || true)
    [ -n "$RESOLVED_VESSEL" ] && VESSEL_ID="$RESOLVED_VESSEL"
fi

if [[ ! "$VESSEL_ID" =~ ^[-a-zA-Z0-9_]+$ ]]; then
    echo -e "${RED}[ERROR] Invalid Vessel ID '$VESSEL_ID'. Must contain only alphanumeric characters, underscores, or hyphens.${NC}"
    exit 1
fi
echo -e "  ${GREEN}[✓] Selected Vessel:${NC} ${BOLD}$VESSEL_ID${NC}"

if [ -t 0 ] && [ "$CHIMERA_MODE" = "replace" ] && [ -z "$VESSEL_REPLACE_RANGE" ]; then
    echo -e "\n${YELLOW}--- Chimera Architecture Modes ---${NC}"
    echo -e "  [1] Graft / Replace: Replace a vessel loop/segment with donor motif"
    echo -e "  [2] Ligate to N-Terminus: Fuse donor motif to vessel N-terminus"
    echo -e "  [3] Ligate to C-Terminus: Fuse donor motif to vessel C-terminus"
    echo -e "  [4] Insert into Loop: Insert donor motif at an internal split point"
    read -r -p "▶ 5. Choose Mode [1/2/3/4, Default: 1]: " MODE_CHOICE
    MODE_CHOICE=${MODE_CHOICE:-1}

    case "$MODE_CHOICE" in
        1)
            CHIMERA_MODE="replace"
            read -r -p "     Vessel range to excise / replace (e.g. 55-70): " VESSEL_REPLACE_RANGE
            ;;
        2)
            CHIMERA_MODE="ligate_nterm"
            ;;
        3)
            CHIMERA_MODE="ligate_cterm"
            ;;
        4)
            CHIMERA_MODE="insert_loop"
            read -r -p "     Split vessel after residue number (e.g. 54): " VESSEL_SPLIT_AFTER
            ;;
    esac
fi

# Validate parameters against unexpected syntax or command injection characters
if [ -n "$VESSEL_REPLACE_RANGE" ] && [[ ! "$VESSEL_REPLACE_RANGE" =~ ^[-a-zA-Z0-9]+$ ]]; then
    echo -e "${RED}[ERROR] Invalid vessel_replace_range '$VESSEL_REPLACE_RANGE'. Expected 'START-END'.${NC}"
    exit 1
fi
if [ -n "$VESSEL_SPLIT_AFTER" ] && [[ ! "$VESSEL_SPLIT_AFTER" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}[ERROR] Invalid vessel_split_after '$VESSEL_SPLIT_AFTER'. Must be integer.${NC}"
    exit 1
fi

if [ -t 0 ]; then
    echo -e "\n${CYAN}======================================================${NC}"
    echo -e "${CYAN} 📏 MEASURING 3D JUNCTION GAPS & LINKER BUDGETING     ${NC}"
    echo -e "${CYAN}======================================================${NC}"

    PRELIM_OUT_DIR="$(canonicalize_out_dir "$OUT_DIR" "$SCRIPT_DIR")"
    MEASURE_CMD=(
        python "$SCRIPT_DIR/scripts/chimera_scaffold.py" measure_junctions
        "--donor_id" "$DONOR_ID"
        "--vessel_id" "$VESSEL_ID"
        "--motif_residues" "$MOTIF_RESIDUES"
        "--chimera_mode" "$CHIMERA_MODE"
        "--out_dir" "$PRELIM_OUT_DIR"
    )
    if [ -n "$ABS_INPUT_FILE" ]; then
        MEASURE_CMD+=("--input_file" "$ABS_INPUT_FILE")
    fi
    if [ -n "$DONOR_PDB" ]; then
        MEASURE_CMD+=("--donor_pdb" "$DONOR_PDB")
    fi
    if [ -n "$VESSEL_PDB" ]; then
        MEASURE_CMD+=("--vessel_pdb" "$VESSEL_PDB")
    fi
    if [ -n "$VESSEL_REPLACE_RANGE" ]; then
        MEASURE_CMD+=("--vessel_replace_range" "$VESSEL_REPLACE_RANGE")
    fi
    if [ -n "$VESSEL_SPLIT_AFTER" ]; then
        MEASURE_CMD+=("--vessel_split_after" "$VESSEL_SPLIT_AFTER")
    fi

    MEASURE_JSON=$("${MEASURE_CMD[@]}" 2>/dev/null || true)

    if [ -n "$MEASURE_JSON" ] && echo "$MEASURE_JSON" | grep -q "junctions"; then
        # Parse recommended linkers, fragments table, and display summary in Python
        PARSED_RECOMMENDATIONS=$(echo "$MEASURE_JSON" | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    donor_segs = data.get('donor_segments', [])
    int_gaps = data.get('internal_gaps', [])
    tot_donor = data.get('total_donor_residues', 0)

    if len(donor_segs) > 1 or len(int_gaps) > 0:
        print('=' * 80)
        print(' 🧬 MULTI-DONOR DISCONTINUOUS MOTIF BREAKDOWN & INTERNAL GAP ANALYSIS 🧬')
        print('=' * 80)
        print(f'[+] Preserving {tot_donor} donor residues partitioned into {len(donor_segs)} segments:')
        print('  ┌──────────┬──────────────┬──────────────┬────────────────────────────────────────┐')
        print('  │ Segment  │ Residue Span │ Length (aa)  │ Distance to Next (Å)                   │')
        print('  ├──────────┼──────────────┼──────────────┼────────────────────────────────────────┤')
        for s in donor_segs:
            seg_id = s.get('segment_id', '')
            r_str = s.get('range_str', '')
            l_str = f\"{s.get('length', '')} aa\"
            d_str = s.get('distance_to_next', '')
            print(f'  │ {seg_id:<8} │ {r_str:<12} │ {l_str:<12} │ {d_str:<38} │')
        print('  └──────────┴──────────────┴──────────────┴────────────────────────────────────────┘')

        if int_gaps:
            print('\n[✓] Automated Internal Linker Budget:')
            for g in int_gaps:
                b_idx = g.get('bridge_idx', 1)
                f_s = g.get('from_segment', 1)
                t_s = g.get('to_segment', 2)
                rec = g.get('recommended_range', '')
                dist = g.get('distance_angstrom', 0.0)
                print(f'  • Bridge {b_idx} (S{f_s} -> S{t_s}): {rec} aa (spanning {dist} Å void)')
        print('=' * 80 + '\n')

    j = data.get('junctions', {})
    rec_n = ''
    rec_c = ''
    print('[✓] Calculated 3D Outer Junction Distances:')
    for k, v in j.items():
        desc = v.get('description', k)
        d = v.get('distance_angstrom', 0.0)
        ext = v.get('ext_min', 0)
        glob = v.get('recommended_range', '')
        param = v.get('param', '')
        if param == 'linker_n' and not rec_n:
            rec_n = glob
        elif param == 'linker_c' and not rec_c:
            rec_c = glob
        print(f'  • {desc}: Measured 3D gap: {d} Å')
        print(f'    - Theoretical minimum: {ext} aa (fully extended @ 3.8 Å/aa)')
        print(f'    - Recommended globular span: {glob} aa')
    print(f'__REC_N__={rec_n}')
    print(f'__REC_C__={rec_c}')
except Exception:
    pass
" 2>/dev/null || true)
        echo -e "------------------------------------------------------------"

        while IFS= read -r rec_line; do
            if [[ "$rec_line" =~ ^__REC_N__=(.*) ]]; then
                REC_LINKER_N="${BASH_REMATCH[1]}"
            elif [[ "$rec_line" =~ ^__REC_C__=(.*) ]]; then
                REC_LINKER_C="${BASH_REMATCH[1]}"
            else
                echo "$rec_line"
            fi
        done <<< "$PARSED_RECOMMENDATIONS"

        [ -n "$REC_LINKER_N" ] && LINKER_N="$REC_LINKER_N"
        [ -n "$REC_LINKER_C" ] && LINKER_C="$REC_LINKER_C"
    fi

    echo -e "\n${YELLOW}--- Linker Flexibility & Sampling Budget ---${NC}"
    read -r -p "▶ 6. N-terminal linker length [Default: $LINKER_N]: " INP_LN
    LINKER_N=${INP_LN:-$LINKER_N}
    read -r -p "▶ 7. C-terminal linker length [Default: $LINKER_C]: " INP_LC
    LINKER_C=${INP_LC:-$LINKER_C}
    read -r -p "▶ 8. Number of RFdiffusion backbones [Default: $NUM_DESIGNS]: " INP_ND
    NUM_DESIGNS=${INP_ND:-$NUM_DESIGNS}
    read -r -p "▶ 9. ProteinMPNN sequences per backbone [Default: $NUM_SEQS]: " INP_NS
    NUM_SEQS=${INP_NS:-$NUM_SEQS}
    read -r -p "▶ 10. ProteinMPNN parallel GPU batch size [Default: $MPNN_BATCH_SIZE]: " INP_BS
    MPNN_BATCH_SIZE=${INP_BS:-$MPNN_BATCH_SIZE}
    read -r -e -p "▶ 11. Output directory [Default: $OUT_DIR]: " INP_OD
    OUT_DIR=${INP_OD:-$OUT_DIR}
fi

ABS_OUT_DIR="$(canonicalize_out_dir "$OUT_DIR" "$SCRIPT_DIR")"
if [ -z "$ABS_OUT_DIR" ] || [ "$ABS_OUT_DIR" = "/" ] || [ "$ABS_OUT_DIR" = "\\" ] || [[ "$ABS_OUT_DIR" =~ ^[a-zA-Z]:[\\/]?$ ]]; then
    echo -e "${RED}[ERROR] Invalid or dangerous output directory specified: '$OUT_DIR'.${NC}"
    exit 1
fi
LOWER_OUT="${ABS_OUT_DIR,,}"
for SYS_DIR in "/etc" "/usr" "/bin" "/sbin" "/boot" "/dev" "c:\\windows" "c:\\program files"; do
    if [ "$LOWER_OUT" = "$SYS_DIR" ] || [[ "$LOWER_OUT" == "$SYS_DIR"/* ]] || [[ "$LOWER_OUT" == "$SYS_DIR"\\* ]]; then
        echo -e "${RED}[ERROR] Output directory cannot be in a system directory: '$OUT_DIR'.${NC}"
        exit 1
    fi
done
mkdir -p "$ABS_OUT_DIR"

# Clean up preliminary measurement directory if user specified a custom output directory
if [ -n "${PRELIM_OUT_DIR:-}" ] && [ "$PRELIM_OUT_DIR" != "$ABS_OUT_DIR" ] && [ -d "$PRELIM_OUT_DIR" ]; then
    if [ -d "$PRELIM_OUT_DIR/0_prep" ] && [ ! -d "$ABS_OUT_DIR/0_prep" ]; then
        mv "$PRELIM_OUT_DIR/0_prep" "$ABS_OUT_DIR/0_prep" 2>/dev/null || true
    fi
    rmdir "$PRELIM_OUT_DIR" 2>/dev/null || true
    rmdir "$(dirname "$PRELIM_OUT_DIR")" 2>/dev/null || true
fi

PID_FILE="$ABS_OUT_DIR/run.pid"
LOG_FILE="$ABS_OUT_DIR/run.log"

# Resolve neural model Python binaries if not explicitly configured
if [ -z "$PYTHON_RFDIFF" ] || [ ! -x "$PYTHON_RFDIFF" ] || ! "$PYTHON_RFDIFF" -c "import torch" >/dev/null 2>&1; then
    PYTHON_RFDIFF="$(discover_conda_python "${CONDA_ENV_RFDIFF:-SE3nv}" "torch" 2>/dev/null || true)"
    [ -z "$PYTHON_RFDIFF" ] && PYTHON_RFDIFF="$(discover_conda_python "rfdiffusion" "torch" 2>/dev/null || true)"
fi

if [ -z "$PYTHON_MPNN" ] || [ ! -x "$PYTHON_MPNN" ] || ! "$PYTHON_MPNN" -c "import torch" >/dev/null 2>&1; then
    PYTHON_MPNN="$(discover_conda_python "${CONDA_ENV_MPNN:-mlfold}" "torch" 2>/dev/null || true)"
    [ -z "$PYTHON_MPNN" ] && PYTHON_MPNN="$(discover_conda_python "SE3nv" "torch" 2>/dev/null || true)"
    [ -z "$PYTHON_MPNN" ] && [ -n "$PYTHON_RFDIFF" ] && PYTHON_MPNN="$PYTHON_RFDIFF"
fi

# ------------------------------------------------------------------------------
# 3.5 Background Execution Option
# ------------------------------------------------------------------------------
if [ -t 0 ] && [ "$BACKGROUND_MODE" = false ]; then
    read -r -p "▶ Do you want to run this workflow in the background? [y/N]: " RUN_BG
    if [[ "$RUN_BG" =~ ^[Yy]$ ]]; then
        BACKGROUND_MODE=true
        if [ ${#ORIG_ARGS[@]} -eq 0 ]; then
            BG_ARGS=(
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
            [ -n "$ABS_INPUT_FILE" ] && BG_ARGS+=("--input" "$ABS_INPUT_FILE")
            [ -n "$DONOR_PDB" ] && BG_ARGS+=("--donor_pdb" "$DONOR_PDB")
            [ -n "$VESSEL_PDB" ] && BG_ARGS+=("--vessel_pdb" "$VESSEL_PDB")
            [ -n "$VESSEL_REPLACE_RANGE" ] && BG_ARGS+=("--replace_range" "$VESSEL_REPLACE_RANGE")
            [ -n "$VESSEL_SPLIT_AFTER" ] && BG_ARGS+=("--split_after" "$VESSEL_SPLIT_AFTER")
            [ -n "$DRY_RUN" ] && BG_ARGS+=("$DRY_RUN")
            [ -n "$LOCK_VESSEL_FLAG" ] && BG_ARGS+=("$LOCK_VESSEL_FLAG")
            [ -n "$PYTHON_RFDIFF" ] && BG_ARGS+=("--python_rfdiff" "$PYTHON_RFDIFF")
            [ -n "$PYTHON_MPNN" ] && BG_ARGS+=("--python_mpnn" "$PYTHON_MPNN")
        else
            BG_ARGS=("${ORIG_ARGS[@]}")
            if [[ ! " ${ORIG_ARGS[*]} " =~ " --python_rfdiff " ]] && [ -n "$PYTHON_RFDIFF" ]; then
                BG_ARGS+=("--python_rfdiff" "$PYTHON_RFDIFF")
            fi
            if [[ ! " ${ORIG_ARGS[*]} " =~ " --python_mpnn " ]] && [ -n "$PYTHON_MPNN" ]; then
                BG_ARGS+=("--python_mpnn" "$PYTHON_MPNN")
            fi
        fi
        nohup bash "$SCRIPT_DIR/chimera_scaffold.sh" "${BG_ARGS[@]}" --background < /dev/null > "$LOG_FILE" 2>&1 &
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

if [ -n "$PYTHON_RFDIFF" ]; then
    CMD_ARGS+=("--python_rfdiff" "$PYTHON_RFDIFF")
fi

if [ -n "$PYTHON_MPNN" ]; then
    CMD_ARGS+=("--python_mpnn" "$PYTHON_MPNN")
fi

if [ -n "$ABS_INPUT_FILE" ]; then
    CMD_ARGS+=("--input_file" "$ABS_INPUT_FILE")
fi

if [ -n "$DONOR_PDB" ]; then
    CMD_ARGS+=("--donor_pdb" "$DONOR_PDB")
fi

if [ -n "$VESSEL_PDB" ]; then
    CMD_ARGS+=("--vessel_pdb" "$VESSEL_PDB")
fi

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

if [ -n "$RFDIFF_DIR" ] && [ -f "$RFDIFF_DIR/scripts/run_inference.py" ]; then
    CMD_ARGS+=("--rfdiff_script" "$RFDIFF_DIR/scripts/run_inference.py")
elif [ -f "$HOME/programs/RFdiffusion/scripts/run_inference.py" ]; then
    CMD_ARGS+=("--rfdiff_script" "$HOME/programs/RFdiffusion/scripts/run_inference.py")
fi

if [ -n "$MPNN_DIR" ] && [ -f "$MPNN_DIR/protein_mpnn_run.py" ]; then
    CMD_ARGS+=("--mpnn_script" "$MPNN_DIR/protein_mpnn_run.py")
elif [ -f "$HOME/programs/ProteinMPNN/protein_mpnn_run.py" ]; then
    CMD_ARGS+=("--mpnn_script" "$HOME/programs/ProteinMPNN/protein_mpnn_run.py")
fi

# Activate environment if configured
if [ -n "$CONDA_ENV_RFDIFF" ]; then
    activate_env "$CONDA_ENV_RFDIFF" 2>/dev/null || true
fi

"${CMD_ARGS[@]}"

echo -e "\n${GREEN}[✓] Chimera Scaffolding completed successfully.${NC}"
echo -e "Results stored in: ${CYAN}${ABS_OUT_DIR}${NC}\n"
