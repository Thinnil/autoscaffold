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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
            [ -f "$SCRIPT_DIR/scripts/install_dependencies.sh" ] || { echo -e "${RED}[ERROR] Script not found: $SCRIPT_DIR/scripts/install_dependencies.sh${NC}"; exit 1; }
            exec bash "$SCRIPT_DIR/scripts/install_dependencies.sh" "${CMD_REST[@]}"
            ;;
        update|upgrade)
            [ -f "$SCRIPT_DIR/scripts/self_update.sh" ] || { echo -e "${RED}[ERROR] Script not found: $SCRIPT_DIR/scripts/self_update.sh${NC}"; exit 1; }
            exec bash "$SCRIPT_DIR/scripts/self_update.sh" "${CMD_REST[@]}"
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
clear 2>/dev/null || true
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

read -p "▶ Enter selection [1-8]: " WORKFLOW_CHOICE

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
        [ -f "$SCRIPT_DIR/scripts/install_dependencies.sh" ] || { echo -e "${RED}[ERROR] Script not found: $SCRIPT_DIR/scripts/install_dependencies.sh${NC}"; exit 1; }
        exec bash "$SCRIPT_DIR/scripts/install_dependencies.sh"
        ;;
    7)
        [ -f "$SCRIPT_DIR/scripts/self_update.sh" ] || { echo -e "${RED}[ERROR] Script not found: $SCRIPT_DIR/scripts/self_update.sh${NC}"; exit 1; }
        exec bash "$SCRIPT_DIR/scripts/self_update.sh"
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