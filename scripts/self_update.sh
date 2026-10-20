#!/usr/bin/env bash
# self_update.sh - Autonomous in-place upgrade and overwrite engine for autoscaffold

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/scripts/env_setup.sh"

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}  🔄 AUTOSCAFFOLD: IN-PLACE SELF-UPDATE & UPGRADE     ${NC}"
echo -e "${CYAN}======================================================${NC}"
echo -e "Repository location: ${BOLD}$SCRIPT_DIR${NC}"
echo -e "Remote target:       ${BOLD}$AUTOSCAFFOLD_REPO_URL ($AUTOSCAFFOLD_BRANCH)${NC}\n"

cd "$SCRIPT_DIR"

if [[ ! "$AUTOSCAFFOLD_BRANCH" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
    echo -e "${RED}[ERROR] Invalid git branch name '$AUTOSCAFFOLD_BRANCH'${NC}"
    exit 1
fi

if [ ! -d ".git" ]; then
    echo -e "${YELLOW}[!] Current directory is not a git repository.${NC}"
    echo -e "Initializing git repository to link with remote..."
    git init
    git remote add origin "$AUTOSCAFFOLD_REPO_URL" 2>/dev/null || git remote set-url origin "$AUTOSCAFFOLD_REPO_URL"
fi

# Fetch latest status
echo -e "Fetching latest changes from remote..."
git fetch origin "$AUTOSCAFFOLD_BRANCH" 2>/dev/null || {
    echo -e "${RED}[ERROR] Could not fetch from $AUTOSCAFFOLD_REPO_URL.${NC}"
    echo -e "Please check your network connection or repository URL in config.env."
    exit 1
}

LOCAL_HASH=$(git rev-parse HEAD 2>/dev/null || echo "none")
REMOTE_HASH=$(git rev-parse "origin/$AUTOSCAFFOLD_BRANCH" 2>/dev/null || echo "unknown")

if [[ "$LOCAL_HASH" == "$REMOTE_HASH" ]]; then
    echo -e "${GREEN}[✓] AutoScaffold is already up to date! (Commit: ${LOCAL_HASH:0:8})${NC}"
    exit 0
fi

echo -e "${YELLOW}An update is available!${NC}"
echo -e "  Current local commit:  ${LOCAL_HASH:0:8}"
echo -e "  Latest remote commit: ${REMOTE_HASH:0:8}"

echo -e "\nRecent changes on remote:"
git log --oneline -n 5 "origin/$AUTOSCAFFOLD_BRANCH" || true

echo ""
read -p "Proceed with in-place upgrade? Local user configs/weights will be preserved. (y/N): " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Update aborted by user."
    exit 0
fi

# Backup modified or critical shell scripts
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$SCRIPT_DIR/.backup_$TIMESTAMP"
mkdir -p "$BACKUP_DIR"

echo -e "\nBacking up active scripts to $BACKUP_DIR..."
for f in autoscaffold.sh de_novo.sh autoscaffold_suite.sh; do
    if [ -f "$SCRIPT_DIR/$f" ]; then
        cp "$SCRIPT_DIR/$f" "$BACKUP_DIR/"
    fi
done

# Perform safe pull / stash
echo "Merging updates..."
if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    echo "Stashing uncommitted changes..."
    git stash save "Autoscaffold auto-update stash $TIMESTAMP"
fi

git checkout "$AUTOSCAFFOLD_BRANCH" 2>/dev/null || git checkout -b "$AUTOSCAFFOLD_BRANCH" "origin/$AUTOSCAFFOLD_BRANCH"
git pull origin "$AUTOSCAFFOLD_BRANCH"

# Ensure all scripts have safe execute permissions
chmod 755 "$SCRIPT_DIR/autoscaffold" 2>/dev/null || true
chmod 755 "$SCRIPT_DIR"/*.sh 2>/dev/null || true
chmod 755 "$SCRIPT_DIR"/scripts/*.sh 2>/dev/null || true
chmod 755 "$SCRIPT_DIR"/scripts/*.py 2>/dev/null || true

# Refresh global launcher if install.sh is present
if [ -f "$SCRIPT_DIR/install.sh" ]; then
    bash "$SCRIPT_DIR/install.sh" shortcut >/dev/null 2>&1 || true
fi

echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN} [✓] AutoScaffold upgraded successfully to ${REMOTE_HASH:0:8}! ${NC}"
echo -e "${GREEN}======================================================${NC}\n"
