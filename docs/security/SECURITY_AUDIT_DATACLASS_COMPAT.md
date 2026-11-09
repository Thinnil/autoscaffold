# Security Audit Report: Dataclass Compatibility Layer & Chimera Pipeline Suite

**Audit Scope**: AutoScaffold v2.2.0 compatibility layer, multi-stage pipeline, and shell orchestration scripts:
- `scripts/dataclass_compat.py`
- `scripts/chimera_stages/stage1_parsing.py`
- `scripts/chimera_stages/stage4_runners.py`
- `scripts/run_chimera_pipeline.py`
- `scripts/chimera_stages/__init__.py`
- `scripts/chimera_scaffold.py`
- `scripts/env_setup.sh`
- `chimera_scaffold.sh`
- `tests/test_dataclass_compat.py`

**Date**: 2026-11-09  
**Auditor**: Security Auditor Subagent (Orchestai Pipeline)  
**Status**: PASSED (All vulnerabilities patched, 114/114 unit tests passing)

---

## 1. Executive Summary

An exhaustive security audit was conducted covering all files modified or created during the dataclass compatibility migration turn. All six mandatory checklist dimensions were rigorously scrutinized:
1. **SECRETS**: Verification of zero hardcoded API credentials, private tokens, passwords, or authentication keys.
2. **RCE**: Audit against arbitrary code execution vectors (`eval()`, `exec()`, `shell=True` in `subprocess`, `curl | bash`, unquoted variables).
3. **INJECTION**: Deep analysis of candidate interpreter resolution loops, path expansions, metacharacter handling, and module name evaluation in shell scripts.
4. **PATH TRAVERSAL**: Validation of path canonicalization, root directory escapes (`/`, `\`, drive roots), and directory traversal attempts (`..`).
5. **PERMISSIONS**: Enforcement of strict permission boundaries (no `chmod 777`, no unauthorized `sudo`).
6. **DEPENDENCIES**: Audit of the pure-Python zero-dependency fallback dataclass shim to guarantee no untrusted external modules or unpicklers are invoked.

---

## 2. Audit Findings by Category & File

### A) Vulnerabilities Found

#### 1. `scripts/env_setup.sh`
- **Vulnerability 1.1 (Shell & Command Injection in `_probe_py`)**:
  - **Location**: `scripts/env_setup.sh:39-47`
  - **Severity**: High
  - **Description**: `_probe_py()` only evaluated `[ ! -x "$py" ]`, allowing directory targets (`-x` on Unix directories) to trigger execution errors, and lacked sanitization against shell injection metacharacters (`;&|`><\n\r$`), directory traversal (`..`), or untrusted binary names.
- **Vulnerability 1.2 (Untrusted Path Traversal & Injection in Tier 3 Conda Filesystem Scan)**:
  - **Location**: `scripts/env_setup.sh:95-106`
  - **Severity**: High
  - **Description**: Filesystem candidate scan across Conda/Micromamba root directories iterated over `$s_root/*` without validating `s_root` or `env_dir` against directory traversal (`..`) or metacharacters, and passed candidates without confirming regular file status (`[ -f "$cand" ]`).
- **Vulnerability 1.3 (Arbitrary Python Code Injection in `discover_conda_python` Module Probe)**:
  - **Location**: `scripts/env_setup.sh:340-350`
  - **Severity**: High
  - **Description**: `req_module` parameter was interpolated directly into `"$py" -c "import sys; import $req_module"` without identifier syntax validation. If an untrusted module string containing separators or arbitrary Python code was passed, it would execute within the interpreter. Additionally, `target_env` lacked format validation, potentially allowing directory traversal in `$base_dir/envs/$target_env`.
- **Vulnerability 1.4 (Lack of Regular File Verification in Environment Manager Discovery)**:
  - **Location**: `scripts/env_setup.sh:213, 247`
  - **Severity**: Medium
  - **Description**: `find_micromamba_bin` and `find_conda_bin` tested candidates with `[ -x "$cand" ]` without `[ -f "$cand" ]` and lacked sanitization against metacharacters and directory traversal sequences.
- **Vulnerability 1.5 (Missing Root Escaping Guard on `SCRIPT_DIR`)**:
  - **Location**: `scripts/env_setup.sh:12-18`
  - **Severity**: Medium
  - **Description**: In the event `AUTOSCAFFOLD_HOME` or `dirname` resolved to root (`/`, `\`, or drive root), script execution was not immediately halted with a diagnostic error.

#### 2. `chimera_scaffold.sh`
- **Vulnerability 2.1 (Missing Regular File Check in Pre-Flight Interpreter Auto-Promotion)**:
  - **Location**: `chimera_scaffold.sh:819-830`
  - **Severity**: Low
  - **Description**: Auto-promotion checks verified candidate executable permission (`-x`) without explicitly confirming regular file status (`-f`).

#### 3. Other Inspected Files
- `scripts/dataclass_compat.py`: **None found** (Strictly adheres to standard library `collections`, `copy`, `typing`; no external dependencies; no `eval()`, `exec()`, or unpickling).
- `scripts/chimera_stages/stage1_parsing.py`: **None found** (Enforces strict regex validation `_RE_SHELL_INJECTION`, rejects directory traversal `..`, enforces `os.path.splitdrive` root guards).
- `scripts/chimera_stages/stage4_runners.py`: **None found** (Subprocess dispatch enforces `shell=False`, validates command lists with `_RE_SHELL_INJECTION`, isolates arguments, restricts paths).
- `scripts/run_chimera_pipeline.py`: **None found** (All subprocess executions enforce `shell=False`, input paths sanitized against root escapes and traversal).
- `scripts/chimera_stages/__init__.py`: **None found** (Pure symbol re-export, no I/O or execution).
- `scripts/chimera_scaffold.py`: **None found** (Preserves `shell=False` execution, sanitizes all parameters).
- `tests/test_dataclass_compat.py`: **None found** (Pure unit tests, mock isolation).

---

## 3. Patched Code with Fixes Applied

### Fix 1: Metacharacter & Traversal Sanitization Function (`scripts/env_setup.sh`)
```bash
# Security sanitization helper to reject shell injection and directory traversal
_is_unsafe_path() {
    local p="$1"
    [ -z "$p" ] && return 0
    case "$p" in
        *[\;\&\|\`\>\<\$]*|*..*|*[$'\n'$'\r']*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}
```
*One-line explanation*: Provides robust, cross-shell glob matching to reject shell injection metacharacters, control characters, and directory traversal sequences without syntax errors.

### Fix 2: Defensive Root Directory Check on `SCRIPT_DIR` (`scripts/env_setup.sh`)
```bash
if [ -z "$SCRIPT_DIR" ] || [ "$SCRIPT_DIR" = "/" ] || [ "$SCRIPT_DIR" = "\\" ] || [[ "$SCRIPT_DIR" =~ ^[a-zA-Z]:[\\/]?$ ]]; then
    echo -e "\033[0;31m[ERROR] Dangerous or invalid SCRIPT_DIR in env_setup.sh: '$SCRIPT_DIR'\033[0m" >&2
    return 1 2>/dev/null || exit 1
fi
```
*One-line explanation*: Halts execution immediately if the resolved script directory points to a dangerous root or filesystem boundary.

### Fix 3: Strict Interpreter Verification in `_probe_py` (`scripts/env_setup.sh`)
```bash
    _probe_py() {
        local py="$1"
        local req_np="${2:-0}"
        if [ -z "$py" ] || [ ! -f "$py" ] || [ ! -x "$py" ]; then
            return 1
        fi
        # Strictly reject paths containing shell injection characters or directory traversal
        if _is_unsafe_path "$py"; then
            return 1
        fi
        # Enforce that candidate executable is a legitimate Python binary
        local base_py
        base_py="$(basename "$py")"
        if [[ ! "$base_py" =~ ^[pP][yY][tT][hH][oO][nN]([0-9]+(\.[0-9]+)?)?(\.[eE][xX][eE])?$ ]]; then
            return 1
        fi
        if [ "$req_np" -eq 1 ]; then
            "$py" -c "import sys, dataclasses, numpy; sys.exit(0 if sys.version_info >= (3, 8) else 1)" >/dev/null 2>&1
        else
            "$py" -c "import sys, dataclasses; sys.exit(0 if sys.version_info >= (3, 8) else 1)" >/dev/null 2>&1
        fi
    }
```
*One-line explanation*: Ensures candidate Python binaries are regular executable files matching legitimate Python naming schemes and free from shell injection or traversal characters.

### Fix 4: Path Sanitization in Tier 3 Conda/Micromamba Scans (`scripts/env_setup.sh`)
```bash
        for s_root in "${_SCAN_ROOTS[@]}"; do
            [ -z "$s_root" ] || [ ! -d "$s_root" ] && continue
            if _is_unsafe_path "$s_root"; then
                continue
            fi
            for env_dir in "$s_root"/*; do
                [ -z "$env_dir" ] || [ ! -d "$env_dir" ] && continue
                if _is_unsafe_path "$env_dir"; then
                    continue
                fi
                for cand in "$env_dir/bin/python" "$env_dir/python.exe"; do
                    [ -z "$cand" ] || [ ! -f "$cand" ] || [ ! -x "$cand" ] && continue
                    if _probe_py "$cand" 1; then
                        _RESOLVED_PYTHON="$cand"
                        break 3
                    fi
                done
            done
        done
```
*One-line explanation*: Sanitizes all root directories and child environment folders before attempting execution of candidate binaries discovered during filesystem scans.

### Fix 5: Regular File and Sanitization in `find_micromamba_bin` & `find_conda_bin` (`scripts/env_setup.sh`)
```bash
        if [ -n "$cand" ] && [ -f "$cand" ] && [ -x "$cand" ]; then
            if ! _is_unsafe_path "$cand"; then
                echo "$cand"
                return 0
            fi
        fi
```
*One-line explanation*: Enforces regular file checks and metacharacter exclusion on detected environment manager binaries.

### Fix 6: Python Module & Target Environment Validation in `discover_conda_python` (`scripts/env_setup.sh`)
```bash
discover_conda_python() {
    local target_env="$1"
    local req_module="${2:-}"
    local candidate=""

    [ -z "$target_env" ] && return 1

    # Validate target_env format to prevent directory traversal and injection
    if [[ ! "$target_env" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
        return 1
    fi

    # Validate req_module identifier to prevent arbitrary Python code injection
    if [ -n "$req_module" ] && [[ ! "$req_module" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]]; then
        return 1
    fi

    _verify_python() {
        local py="$1"
        if [ -z "$py" ] || [ ! -f "$py" ] || [ ! -x "$py" ]; then
            return 1
        fi
        if _is_unsafe_path "$py"; then
            return 1
        fi
        local base_py
        base_py="$(basename "$py")"
        if [[ ! "$base_py" =~ ^[pP][yY][tT][hH][oO][nN]([0-9]+(\.[0-9]+)?)?(\.[eE][xX][eE])?$ ]]; then
            return 1
        fi
        if [ -n "$req_module" ]; then
            "$py" -c "import sys; import $req_module" >/dev/null 2>&1
        else
            "$py" -c "import sys" >/dev/null 2>&1
        fi
    }
...
```
*One-line explanation*: Strictly validates environment name format and Python identifier syntax for imported modules, eliminating arbitrary code injection during environment discovery.

### Fix 7: Pre-Flight Auto-Promotion Regular File Validation (`chimera_scaffold.sh`)
```bash
    if [ -n "$_DISCOVERED_PY" ] && [ -f "$_DISCOVERED_PY" ] && [ -x "$_DISCOVERED_PY" ]; then
        echo -e "${YELLOW}[!] Host Python is ${_PY_VER} (< 3.8). Auto-promoting PYTHON_BIN to discovered environment: ${_DISCOVERED_PY}${NC}"
        PYTHON_BIN="$_DISCOVERED_PY"
        CMD_ARGS[0]="$PYTHON_BIN"
    elif [ -n "${PYTHON_RFDIFF:-}" ] && [ -f "$PYTHON_RFDIFF" ] && [ -x "$PYTHON_RFDIFF" ]; then
        echo -e "${YELLOW}[!] Host Python is ${_PY_VER} (< 3.8). Auto-promoting PYTHON_BIN to PYTHON_RFDIFF: ${PYTHON_RFDIFF}${NC}"
        PYTHON_BIN="$PYTHON_RFDIFF"
        CMD_ARGS[0]="$PYTHON_BIN"
    elif [ -n "${PYTHON_MPNN:-}" ] && [ -f "$PYTHON_MPNN" ] && [ -x "$PYTHON_MPNN" ]; then
        echo -e "${YELLOW}[!] Host Python is ${_PY_VER} (< 3.8). Auto-promoting PYTHON_BIN to PYTHON_MPNN: ${PYTHON_MPNN}${NC}"
        PYTHON_BIN="$PYTHON_MPNN"
        CMD_ARGS[0]="$PYTHON_BIN"
```
*One-line explanation*: Guarantees that auto-promoted Python binaries are verified regular executable files before replacing the command invocation array.

---

## 4. Test Suite Verification

Following application of the defensive security patches, the entire unit test suite was executed:
```bash
python -m unittest discover -s tests -p "test_*.py"
```
**Result**:
```
Ran 114 tests in 24.289s

OK
```
All 114 unit tests, regression tests, CLI path resolution tests, and simulated bare Python 3.6 backward compatibility checks passed with zero errors or failures.
