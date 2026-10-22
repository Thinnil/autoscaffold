#!/usr/bin/env bash
# autoscaffold_suite.sh - Unified Master Suite for Autonomous Protein Engineering
# Integrates:
# [1] De Novo Motif Scaffolding
# [2] De Novo Target-Binder Design
# [3] Discontinuous Binding Surface Scaffolding
# [4] Chimera Scaffolding (PDB / PyMOL .pse)
# [5] Diagnostics & Quality Validation
set -e

# Trap Ctrl+C (SIGINT) cleanly
trap 'echo -e "\n\n${YELLOW}[!] Operation cancelled by user.${NC}"; exit 130' INT

# Resolve script directory and source shared environment
if [ -n "$AUTOSCAFFOLD_HOME" ] && [ -f "$AUTOSCAFFOLD_HOME/autoscaffold_suite.sh" ]; then
    SCRIPT_DIR="$AUTOSCAFFOLD_HOME"
else
    SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
fi
if [ -z "$SCRIPT_DIR" ] || [ "$SCRIPT_DIR" = "/" ] || [ "$SCRIPT_DIR" = "\\" ] || [[ "$SCRIPT_DIR" =~ ^[a-zA-Z]:[\\/]?$ ]]; then
    echo -e "\033[0;31m[ERROR] Invalid or dangerous SCRIPT_DIR root: '$SCRIPT_DIR'.\033[0m"
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
        echo -e "Please ensure you have cloned the full repository:"
        echo -e "  git clone https://github.com/Thinnil/autoscaffold.git"
        echo -e "  cd autoscaffold && ./autoscaffold_suite.sh"
        exit 1
    fi
    echo -e "\033[0;32m[✓] AutoScaffold scripts successfully synchronized.\033[0m\n"
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/scripts/env_setup.sh"

# ------------------------------------------------------------------------------
# CLI Subcommand Routing
# ------------------------------------------------------------------------------
BACKGROUND_FLAG=""
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--background" ] || [ "$arg" = "-b" ]; then
        BACKGROUND_FLAG="--background"
    else
        ARGS+=("$arg")
    fi
done

if [ ${#ARGS[@]} -gt 0 ]; then
    SUBCMD="${ARGS[0]}"
    CMD_REST=("${ARGS[@]:1}")
    [ -n "$BACKGROUND_FLAG" ] && CMD_REST+=("$BACKGROUND_FLAG")
    case "$SUBCMD" in
        motif|novo)
            [ -f "$SCRIPT_DIR/autoscaffold.sh" ] || { echo -e "${RED}[ERROR] Script not found: $SCRIPT_DIR/autoscaffold.sh${NC}"; exit 1; }
            exec bash "$SCRIPT_DIR/autoscaffold.sh" "${CMD_REST[@]}"
            ;;
        binder|target)
            [ -f "$SCRIPT_DIR/de_novo.sh" ] || { echo -e "${RED}[ERROR] Script not found: $SCRIPT_DIR/de_novo.sh${NC}"; exit 1; }
            exec bash "$SCRIPT_DIR/de_novo.sh" "${CMD_REST[@]}"
            ;;
        discontinuous|disc)
            [ -f "$SCRIPT_DIR/discontinuous_scaffold.sh" ] || { echo -e "${RED}[ERROR] Script not found: $SCRIPT_DIR/discontinuous_scaffold.sh${NC}"; exit 1; }
            exec bash "$SCRIPT_DIR/discontinuous_scaffold.sh" "${CMD_REST[@]}"
            ;;
        chimera|pse)
            [ -f "$SCRIPT_DIR/chimera_scaffold.sh" ] || { echo -e "${RED}[ERROR] Script not found: $SCRIPT_DIR/chimera_scaffold.sh${NC}"; exit 1; }
            exec bash "$SCRIPT_DIR/chimera_scaffold.sh" "${CMD_REST[@]}"
            ;;
        diagnostics|diag)
            [ -f "$SCRIPT_DIR/scripts/diagnostics.sh" ] || { echo -e "${RED}[ERROR] Script not found: $SCRIPT_DIR/scripts/diagnostics.sh${NC}"; exit 1; }
            exec bash "$SCRIPT_DIR/scripts/diagnostics.sh" "${CMD_REST[@]}"
            ;;
        install|setup|deps)
            if [ -f "$SCRIPT_DIR/install.sh" ]; then
                exec bash "$SCRIPT_DIR/install.sh" "${CMD_REST[@]}"
            else
                [ -f "$SCRIPT_DIR/scripts/install_dependencies.sh" ] || { echo -e "${RED}[ERROR] Script not found: $SCRIPT_DIR/scripts/install_dependencies.sh${NC}"; exit 1; }
                exec bash "$SCRIPT_DIR/scripts/install_dependencies.sh" "${CMD_REST[@]}"
            fi
            ;;
        shortcut|launcher)
            if [ -f "$SCRIPT_DIR/install.sh" ]; then
                exec bash "$SCRIPT_DIR/install.sh" launcher "${CMD_REST[@]}"
            else
                echo -e "${RED}[ERROR] Script not found: $SCRIPT_DIR/install.sh${NC}"; exit 1
            fi
            ;;
        update|upgrade)
            if [ -f "$SCRIPT_DIR/install.sh" ]; then
                exec bash "$SCRIPT_DIR/install.sh" update "${CMD_REST[@]}"
            else
                [ -f "$SCRIPT_DIR/scripts/self_update.sh" ] || { echo -e "${RED}[ERROR] Script not found: $SCRIPT_DIR/scripts/self_update.sh${NC}"; exit 1; }
                exec bash "$SCRIPT_DIR/scripts/self_update.sh" "${CMD_REST[@]}"
            fi
            ;;
        help|--help|-h)
            echo -e "${BOLD}AutoScaffold Master Suite CLI${NC}"
            echo -e "Usage:"
            echo -e "  $0 [subcommand] [arguments...] [--background]"
            echo -e "\nSubcommands:"
            echo -e "  motif             De Novo Active Site & Epitope Motif Scaffolding"
            echo -e "  binder            Target-Binder Design & ColabFold Validation"
            echo -e "  discontinuous     Discontinuous Contact Motif Scaffolding"
            echo -e "  chimera           Chimera Scaffolding: Graft Motif into Vessel Protein (PDB/PSE)"
            echo -e "  diagnostics       Run environment & dependency diagnostic checks"
            echo -e "  install           Run automated dependency, tool, and environment installer"
            echo -e "  launcher          Install global 'autoscaffold' shortcut into ~/.local/bin"
            echo -e "  update            Check and apply in-place updates from GitHub"
            echo -e "\nOptions:"
            echo -e "  --background, -b  Run selected workflow in background"
            echo -e "\nRun without arguments to launch the interactive TUI menu."
            exit 0
            ;;
        *)
            echo -e "${RED}[ERROR] Unknown subcommand: '$SUBCMD'${NC}"
            echo -e "Available subcommands: motif, binder, discontinuous, chimera, diagnostics, install, update, help"
            exit 1
            ;;
    esac
fi

# ------------------------------------------------------------------------------
# Interactive TUI Menu
# ------------------------------------------------------------------------------
if [ "${AUTOSCAFFOLD_CLEAR:-0}" = "1" ] && [ "${AUTOSCAFFOLD_NO_CLEAR:-1}" != "1" ]; then
    clear 2>/dev/null || true
else
    echo ""
    COLS=$(tput cols 2>/dev/null || echo 80)
    [ "$COLS" -gt 80 ] && COLS=80
    printf '%.0s─' $(seq 1 "$COLS") 2>/dev/null || true
    echo -e "\n"
fi
echo -e "${CYAN}======================================================================${NC}"
echo -e "${CYAN}        🧬 AUTOSCAFFOLD SUITE: AUTONOMOUS PROTEIN ENGINEERING 🧬        ${NC}"
echo -e "${CYAN}       RFdiffusion  •  ProteinMPNN  •  ColabFold  •  PyMOL / PSE       ${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo -e "Tip: Press ${YELLOW}Ctrl + C${NC} at any time to exit.\n"

echo -e "${BOLD}Select a workflow to launch:${NC}"
echo -e "  ${GREEN}[1]${NC} 🏗️  De Novo Motif Scaffolding (Active site & epitope contigs)"
echo -e "  ${GREEN}[2]${NC} 🎯 De Novo Target-Binder Design (Interface hotspot scaffolding)"
echo -e "  ${GREEN}[3]${NC} 🧩 Discontinuous Binding Surface Scaffolding"
echo -e "  ${GREEN}[4]${NC} 🧬 Chimera Scaffolding: Graft Motif into Vessel Protein (PDB/PSE)"
echo -e "  ${GREEN}[5]${NC} 🩺 System Diagnostics & Environment Pre-Flight Checks"
echo -e "  ${GREEN}[6]${NC} 📦 Automatic Dependency & Environment Installer"
echo -e "  ${GREEN}[7]${NC} 🔄 In-Place Self-Update & GitHub Upgrade"
echo -e "  ${GREEN}[8]${NC} 🚪 Exit"
echo -e ""

read -r -p "▶ Enter selection [1-8]: " WORKFLOW_CHOICE

case "$WORKFLOW_CHOICE" in
    1)
        [ -f "$SCRIPT_DIR/autoscaffold.sh" ] || { echo -e "${RED}[ERROR] Script not found: $SCRIPT_DIR/autoscaffold.sh${NC}"; exit 1; }
        exec bash "$SCRIPT_DIR/autoscaffold.sh"
        ;;
    2)
        [ -f "$SCRIPT_DIR/de_novo.sh" ] || { echo -e "${RED}[ERROR] Script not found: $SCRIPT_DIR/de_novo.sh${NC}"; exit 1; }
        exec bash "$SCRIPT_DIR/de_novo.sh"
        ;;
    3)
        [ -f "$SCRIPT_DIR/discontinuous_scaffold.sh" ] || { echo -e "${RED}[ERROR] Script not found: $SCRIPT_DIR/discontinuous_scaffold.sh${NC}"; exit 1; }
        exec bash "$SCRIPT_DIR/discontinuous_scaffold.sh"
        ;;
    4)
        [ -f "$SCRIPT_DIR/chimera_scaffold.sh" ] || { echo -e "${RED}[ERROR] Script not found: $SCRIPT_DIR/chimera_scaffold.sh${NC}"; exit 1; }
        exec bash "$SCRIPT_DIR/chimera_scaffold.sh"
        ;;
    5)
        [ -f "$SCRIPT_DIR/scripts/diagnostics.sh" ] || { echo -e "${RED}[ERROR] Script not found: $SCRIPT_DIR/scripts/diagnostics.sh${NC}"; exit 1; }
        exec bash "$SCRIPT_DIR/scripts/diagnostics.sh"
        ;;
    6)
        if [ -f "$SCRIPT_DIR/install.sh" ]; then
            exec bash "$SCRIPT_DIR/install.sh"
        else
            [ -f "$SCRIPT_DIR/scripts/install_dependencies.sh" ] || { echo -e "${RED}[ERROR] Script not found: $SCRIPT_DIR/scripts/install_dependencies.sh${NC}"; exit 1; }
            exec bash "$SCRIPT_DIR/scripts/install_dependencies.sh"
        fi
        ;;
    7)
        if [ -f "$SCRIPT_DIR/install.sh" ]; then
            exec bash "$SCRIPT_DIR/install.sh" update
        else
            [ -f "$SCRIPT_DIR/scripts/self_update.sh" ] || { echo -e "${RED}[ERROR] Script not found: $SCRIPT_DIR/scripts/self_update.sh${NC}"; exit 1; }
            exec bash "$SCRIPT_DIR/scripts/self_update.sh"
        fi
        ;;
    8|q|quit|exit)
        echo -e "\n${CYAN}Exiting AutoScaffold Suite. Happy designing!${NC}\n"
        exit 0
        ;;
    *)
        echo -e "${RED}[ERROR] Invalid selection '${WORKFLOW_CHOICE}'. Exiting.${NC}"
        exit 1
        ;;
esac
