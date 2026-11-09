# Technical Specification: Zero-Dependency Dataclass Compatibility & Intelligent Python Interpreter Prioritization

- **Target System**: AutoScaffold Suite (`scripts/dataclass_compat.py`, `scripts/chimera_stages/`, `scripts/env_setup.sh`, `chimera_scaffold.sh`, `scripts/run_chimera_pipeline.py`, `tests/test_dataclass_compat.py`)
- **Document ID**: `DATACLASS_AND_PYTHON_COMPAT_SPEC`
- **Architecture Pattern**: Zero-Dependency Python 3.6 Backport Fallback Shim, Intelligent Multi-Tier Conda/Micromamba Interpreter Discovery, and Conda Handshake Execution Safeguards
- **Author**: Prompt Polisher (Orchestai Pipeline)
- **Status**: Approved for Implementation

---

## 1. Executive Summary & OrchestAI Diagnostic Dossier Analysis

### 1.1 Problem Context & Failure Dossier
During runtime validation on legacy host platforms (e.g., Ubuntu 18.04 LTS Bionic workstations and institutional HPC compute clusters), AutoScaffold runs against the system-default Python interpreter `/usr/bin/python3` (Python 3.6.9). 

The preflight diagnostic tool (`scripts/diagnose_environment.py`) revealed two compounding systemic issues:
1. **Host Python Standard Library Limitation (`ModuleNotFoundError: No module named 'dataclasses'`)**:
   - The modular AutoScaffold architecture introduced in `chimera_stages` relies heavily on `@dataclass` and `field` (`stage1_parsing.py`, `stage4_runners.py`, and `run_chimera_pipeline.py`).
   - The `dataclasses` module was officially introduced into the Python standard library in **Python 3.7** (PEP 557). On Python 3.6.9 without an externally installed PyPI backport (`pip install dataclasses`), 8 core modules fail to import immediately:
     - `chimera_stages/stage1_parsing.py`
     - `chimera_stages/stage2_clash.py` (cascading dependency on `stage1_parsing`)
     - `chimera_stages/stage3_geometry.py` (cascading dependency on `stage1_parsing`)
     - `chimera_stages/stage4_runners.py`
     - `chimera_stages/stage5_export.py` (cascading dependency on `stage1_parsing`)
     - `chimera_stages/__init__.py`
     - `scripts/chimera_scaffold.py`
     - `scripts/run_chimera_pipeline.py`
2. **Flawed Interpreter Resolution in `scripts/env_setup.sh`**:
   - `env_setup.sh` evaluated interpreters using a basic `command -v python3` check.
   - If `/usr/bin/python3` existed, it was immediately exported as `PYTHON_BIN`, completely ignoring modern Conda/Micromamba environments located in standard user directories (such as `/home/manuel/micromamba/envs/conf_sel_analysis/bin/python`, `SE3nv`, or `mlfold`), which already possess Python >= 3.8 and complete scientific runtime stacks (`numpy`, `scipy`, `torch`).
3. **Silent Conda Activation Failure in `chimera_scaffold.sh`**:
   - In `chimera_scaffold.sh`, the execution command array `CMD_ARGS=("$PYTHON_BIN" "$SCRIPT_DIR/scripts/chimera_scaffold.py" ...)` was assembled **prior** to attempting environment activation.
   - Environment activation was subsequently invoked as `activate_env "$CONDA_ENV_RFDIFF" 2>/dev/null || true`. If `activate_env` failed (or ran inside a subshell where bash environment mutations did not affect the caller's variable bindings), the pipeline silently executed with the unaliased host Python 3.6.9, triggering an unhandled `ModuleNotFoundError`.

### 1.2 Primary Engineering Objectives
1. **Zero-Dependency Dataclass Compatibility Shim (`scripts/dataclass_compat.py`)**:
   - Implement a pure-Python fallback shim that transparently imports from `dataclasses` when available (Python >= 3.7 or when the backport is installed).
   - If `dataclasses` is unavailable (bare Python 3.6), provide a zero-dependency fallback providing `@dataclass`, `field()`, `asdict()`, and `is_dataclass()` supporting default values, `default_factory`, `__init__`, `__repr__`, `__eq__`, and `frozen=True`.
   - Update all `chimera_stages` and orchestrator scripts to import from `dataclass_compat`.
2. **Intelligent Multi-Tier Python Discovery in `scripts/env_setup.sh`**:
   - Prioritize Python interpreters: if the default `python3` is < 3.8 or lacks essential scientific libraries, scan active Conda prefixes, local environments, and standard Micromamba/Conda root paths for an installed Python >= 3.8 interpreter.
3. **Robust Conda Handshake Safeguards in `chimera_scaffold.sh`**:
   - Introspect the resolved interpreter before assembling `CMD_ARGS`.
   - If running with an interpreter < 3.8, proactively attempt to resolve the Conda environment interpreter (e.g. `SE3nv` or `mlfold`), switch `PYTHON_BIN` to it, and display clear, diagnostic CLI guidance if no modern interpreter can be located.
4. **100% Backward and Forward Compatibility**:
   - Maintain complete passing status across all 105+ existing unit tests with zero regressions.

---

## 2. Architecture & Component Interaction Flow

```mermaid
flowchart TD
    subgraph "Launcher & Environment Discovery (env_setup.sh)"
        Start["Shell Invocation<br/>(chimera_scaffold.sh / autoscaffold.sh)"] --> SourceEnv["source scripts/env_setup.sh"]
        SourceEnv --> ProbeHost["Probe host 'python3' version & capabilities"]
        ProbeHost --> CheckVer{"Host python3 >= 3.8<br/>and has numpy?"}
        CheckVer -->|Yes| SetHostPy["PYTHON_BIN = host python3"]
        CheckVer -->|No: Python 3.6 or Missing Libs| ScanConda["Scan Conda / Micromamba Environments<br/>(~/.micromamba, ~/micromamba, miniconda3, SE3nv, mlfold)"]
        ScanConda --> FoundModern{"Found Python >= 3.8<br/>with numpy/dataclasses?"}
        FoundModern -->|Yes| PreferModern["Promote Conda Python to PYTHON_BIN"]
        FoundModern -->|No| WarnHost["Retain host Python 3.6 with Compatibility Warning"]
    end

    subgraph "Execution Safeguards (chimera_scaffold.sh)"
        PreferModern --> Handshake["Pre-Flight Execution Handshake"]
        WarnHost --> Handshake
        SetHostPy --> Handshake
        Handshake --> ValidatePy{"Is resolved PYTHON_BIN >= 3.8?"}
        ValidatePy -->|No| CheckToolEnv["Try discovering tool env (SE3nv / mlfold)"]
        CheckToolEnv --> ToolEnvFound{"Tool Python Found?"}
        ToolEnvFound -->|Yes| UpgradePy["Upgrade CMD_ARGS[0] to Tool Python"]
        ToolEnvFound -->|No| FallbackShim["Execute with PYTHON_BIN + dataclass_compat.py"]
        ValidatePy -->|Yes| NativeExec["Execute directly with modern Python"]
    end

    subgraph "Python Runtime (scripts/dataclass_compat.py)"
        UpgradePy --> PythonExec["Execute scripts/chimera_scaffold.py"]
        FallbackShim --> PythonExec
        NativeExec --> PythonExec
        PythonExec --> ImportCompat["import dataclass_compat"]
        ImportCompat --> TryStdLib{"try import dataclasses"}
        TryStdLib -->|Success (Python >= 3.7)| UseStdLib["Export native dataclass, field, asdict, is_dataclass"]
        TryStdLib -->|ImportError (Python 3.6 bare)| FallbackClass["Zero-Dependency Fallback Decorator & Helpers<br/>(type hints, defaults, default_factory, frozen, asdict)"]
    end
```

---

## 3. Specification 1: Zero-Dependency `dataclass_compat.py`

### 3.1 Design Principles
The compatibility layer must reside at `scripts/dataclass_compat.py`. It must:
1. Require **zero external packages** beyond Python 3.6 standard library modules (`sys`, `inspect`, `copy`, `collections`).
2. Offer seamless drop-in API parity with Python's standard `dataclasses` module for all features utilized in AutoScaffold.
3. Allow graceful imports regardless of whether invoked as a package (`from scripts.dataclass_compat import ...` or `from .dataclass_compat import ...`) or top-level script (`from dataclass_compat import ...`).

### 3.2 Supported Dataclass Features
The shim must implement the following API primitives:
- `field(*, default=MISSING, default_factory=MISSING, init=True, repr=True, hash=None, compare=True, metadata=None)`
- `@dataclass(cls=None, *, init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)`
- `asdict(obj, *, dict_factory=dict)`
- `is_dataclass(obj_or_cls)`
- `FrozenInstanceError` (inheriting from `AttributeError`)

### 3.3 Core Mechanism of Fallback `@dataclass`
When `from dataclasses import ...` fails on Python 3.6:
1. **Sentinel Definition**:
   Define a singleton `_MISSING_TYPE` and `MISSING = _MISSING_TYPE()`.
2. **Field Descriptor / Container**:
   A `_Field` class storing:
   - `name`: string identifier
   - `type`: type annotation (from `cls.__annotations__` if present)
   - `default`: default literal value or `MISSING`
   - `default_factory`: callable returning default or `MISSING`
   - `init`, `repr`, `compare`, `frozen` flags
3. **Class Field Extraction**:
   Inspect `cls.__dict__` and `cls.__annotations__`:
   - Walk class MRO (excluding `object`) to inherit parent fields.
   - For every annotated attribute or class attribute holding a `_Field` instance, register a field in definition order.
   - If an attribute has a standard default value (e.g., `max_drift: float = 0.8`), convert it into a `_Field(default=0.8)`.
   - Validate that fields without defaults do not follow fields with defaults (raising `TypeError` matching standard library behavior).
4. **`__init__` Generation**:
   Construct an `__init__(self, *args, **kwargs)` function that:
   - Accepts positional arguments matching field order.
   - Accepts keyword arguments.
   - Evaluates `default_factory()` dynamically on instantiation when default factory is provided.
   - Assigns values to `self.__dict__` (or `object.__setattr__` when `frozen=True`).
5. **`__repr__` Generation**:
   Format string representation matching `<ClassName(field1=val1, field2=val2)>` for all fields where `repr=True`.
6. **`__eq__` Generation**:
   Compare tuples of comparison fields across two instances of the identical class.
7. **`frozen=True` Semantics**:
   - Override `__setattr__` and `__delattr__` on the decorated class to raise `FrozenInstanceError(f"cannot assign to field {name!r}")` if modifying an already initialized instance (`_initialized=True`).
8. **`asdict(obj)` Implementation**:
   - Recursively traverse dataclass instances, converting them to dictionaries.
   - Recursively convert lists, tuples, and dictionaries holding nested dataclasses.
9. **`is_dataclass(obj_or_cls)`**:
   - Return `True` if `obj_or_cls` has the dataclass marker attribute `_is_dataclass = True`.

### 3.4 Target Modules Requiring Import Migration
All modules referencing `from dataclasses import ...` must be refactored to use safe dual-root compatibility imports:
```python
try:
    from scripts.dataclass_compat import dataclass, field, asdict, is_dataclass
except ImportError:
    try:
        from .dataclass_compat import dataclass, field, asdict, is_dataclass
    except ImportError:
        from dataclass_compat import dataclass, field, asdict, is_dataclass
```
Affected modules:
- `scripts/chimera_stages/stage1_parsing.py` (defines `MotifAtom`, `MotifResidue`, `MotifSegment`, `MultiChainMotifManifest`)
- `scripts/chimera_stages/stage4_runners.py` (defines `RunnerSpec`)
- `scripts/run_chimera_pipeline.py` (defines `PipelineStageContext`, `PipelineResult`)
- `scripts/chimera_scaffold.py` (re-exports dataclasses)

---

## 4. Specification 2: Intelligent Python Prioritization in `scripts/env_setup.sh`

### 4.1 Root Cause in Current `env_setup.sh`
Lines 35–49 of `scripts/env_setup.sh` currently resolve `PYTHON_BIN` as follows:
```bash
if [ -z "${PYTHON_BIN:-}" ] || [ ! -x "${PYTHON_BIN:-}" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        ...
```
This naive lookup selects `/usr/bin/python3` (v3.6.9) regardless of its inability to run modern modules or the existence of fully populated virtual environments under user paths.

### 4.2 Multi-Tier Prioritization Algorithm
The new resolver logic in `scripts/env_setup.sh` must execute the following prioritized search:

1. **Tier 0: Explicit Override**:
   If `$PYTHON_BIN` is pre-defined in the environment or loaded from `config.env`, verify it exists and is executable. If valid, respect user configuration.
2. **Tier 1: Active Virtual Environment**:
   If `$CONDA_PREFIX` is active and `$CONDA_PREFIX/bin/python` exists:
   - Check if its version is >= 3.8 and has `numpy`. If yes, select it immediately.
3. **Tier 2: System `python3` Validation**:
   If `command -v python3` exists:
   - Query version: `py_ver=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "0.0")`
   - Check modules: `python3 -c 'import numpy, dataclasses' >/dev/null 2>&1`
   - If version >= 3.8 and modules are present, select `command -v python3`.
4. **Tier 3: User & Host Conda/Micromamba Environment Scan**:
   If host `python3` is < 3.8 or missing critical modules, scan candidate environments across known root prefixes:
   - `$LOCAL_ENVS_DIR/*/bin/python`
   - `$MAMBA_ROOT_PREFIX/envs/*/bin/python`
   - `$HOME/.micromamba/envs/*/bin/python`
   - `$HOME/micromamba/envs/*/bin/python`
   - `$HOME/.local/share/mamba/envs/*/bin/python`
   - `$HOME/miniconda3/envs/*/bin/python`
   - `$HOME/miniforge3/envs/*/bin/python`
   - `$HOME/mambaforge/envs/*/bin/python`
   - `$HOME/anaconda3/envs/*/bin/python`
   - `/opt/conda/envs/*/bin/python`
   - Target environment names: `SE3nv`, `mlfold`, `colabfold`, `conf_sel_analysis`, `openmm`, `rfdiffusion`.
   - For each candidate:
     ```bash
     if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
         # Valid modern interpreter found!
         PYTHON_BIN="$cand"
         break
     fi
     ```
5. **Tier 4: Tool-Specific Interpreter Fallback**:
   - Check `$PYTHON_RFDIFF` or `$PYTHON_MPNN` if already resolved.
6. **Tier 5: Graceful Host Fallback**:
   - If no Python >= 3.8 is found across any path, fall back to `command -v python3` or `python`, and output an informative diagnostic alert advising the user that AutoScaffold is running in compatibility mode.

---

## 5. Specification 3: Conda Activation & Execution Handshake in `chimera_scaffold.sh`

### 5.1 Root Cause in `chimera_scaffold.sh`
In `chimera_scaffold.sh`, `CMD_ARGS` is built at line 729 using `"$PYTHON_BIN"` before line 796 calls `activate_env "$CONDA_ENV_RFDIFF" 2>/dev/null || true`.
Because:
- `CMD_ARGS[0]` is bound at line 729 to the old `PYTHON_BIN`.
- `activate_env` suppresses errors (`2>/dev/null || true`).
- Conda activation inside shell scripts does not always mutate the existing binary paths for sub-processes.
The command executes using the original, un-upgraded Python binary.

### 5.2 Handshake Safeguard Requirements
1. **Dynamic Binary Binding**:
   Before dispatching `"${CMD_ARGS[@]}"`, `chimera_scaffold.sh` must perform a pre-flight execution handshake:
   ```bash
   # Check if active PYTHON_BIN satisfies version requirements
   local py_ver
   py_ver=$("$PYTHON_BIN" -c "import sys; print(sys.version_info[0]*10 + sys.version_info[1])" 2>/dev/null || echo "36")

   if [ "$py_ver" -lt 38 ]; then
       # Attempt to locate modern environment python (e.g. SE3nv or mlfold)
       local target_env="${CONDA_ENV_RFDIFF:-SE3nv}"
       local discovered_py
       discovered_py=$(discover_conda_python "$target_env" "dataclasses" 2>/dev/null || true)
       [ -z "$discovered_py" ] && discovered_py=$(discover_conda_python "${CONDA_ENV_MPNN:-mlfold}" "dataclasses" 2>/dev/null || true)
       
       if [ -n "$discovered_py" ] && [ -x "$discovered_py" ]; then
           echo -e "${YELLOW}[!] Host Python is $py_ver (< 3.8). Auto-promoting PYTHON_BIN to discovered environment: $discovered_py${NC}"
           PYTHON_BIN="$discovered_py"
           CMD_ARGS[0]="$PYTHON_BIN"
       else
           echo -e "${YELLOW}[WARN] Executing on Python < 3.8 host interpreter with dataclass_compat shim.${NC}"
       fi
   fi
   ```
2. **Explicit Conda Activation Logging**:
   When `activate_env` is invoked:
   - Log activation status cleanly.
   - Do not suppress fatal errors if activation fails; report whether the shell environment was successfully loaded or whether direct binary dispatch is utilized.
3. **Subprocess Dispatch Reliability**:
   Ensure `CMD_ARGS` uses the resolved, validated `PYTHON_BIN`.

---

## 6. Specification 4: Test Suite & Verification Matrix

### 6.1 Test Strategy
To prevent regressions across the 105+ unit tests, verification will follow a 3-tier testing matrix:

1. **Unit Test Suite for Compatibility Shim (`tests/test_dataclass_compat.py`)**:
   - Test standard class decoration `@dataclass`.
   - Test `@dataclass(frozen=True)` mutation prevention (raising `FrozenInstanceError` or `AttributeError`).
   - Test `field(default=...)` and `field(default_factory=list)`.
   - Test `asdict()` recursive serialization with primitive and nested dataclasses.
   - Test `is_dataclass()` on classes and instances.
   - Test inheritance and field ordering rules.
   - Test simulated Python 3.6 fallback mode: verify that the fallback decorator passes 100% of the assertions even on systems where native `dataclasses` is present by directly executing `_FallbackDataclass`.
2. **Integration Import Test (`scripts/diagnose_environment.py`)**:
   - Run `diagnose_environment.py` with both Python 3.6 simulation and host Python to verify that all 14 AutoScaffold modules report `[OK]`.
3. **Full Regression Test**:
   - Run `python -m unittest discover -s tests` and verify that all 105 tests pass without failures.

---

## 7. Implementation Checklist & File Modification Plan

| Step | Target File | Action | Description |
|:---|:---|:---|:---|
| **1** | `scripts/dataclass_compat.py` | **Create** | Zero-dependency compatibility shim with native import + robust fallback decorator, `field`, `asdict`, `is_dataclass`. |
| **2** | `tests/test_dataclass_compat.py` | **Create** | Exhaustive unit tests for `dataclass_compat.py` (both native and fallback modes). |
| **3** | `scripts/chimera_stages/stage1_parsing.py` | **Modify** | Replace `from dataclasses import dataclass, field` with `dataclass_compat` import. |
| **4** | `scripts/chimera_stages/stage4_runners.py` | **Modify** | Replace `from dataclasses import dataclass, field` with `dataclass_compat` import. |
| **5** | `scripts/run_chimera_pipeline.py` | **Modify** | Replace `from dataclasses import dataclass, field` with `dataclass_compat` import. |
| **6** | `scripts/chimera_scaffold.py` | **Modify** | Replace `from dataclasses import ...` if imported, and verify re-exports. |
| **7** | `scripts/env_setup.sh` | **Modify** | Implement multi-tier interpreter priority discovery (prioritize Python >= 3.8 with numpy/dataclasses over bare Python 3.6). |
| **8** | `chimera_scaffold.sh` | **Modify** | Add Conda activation handshake and dynamic `CMD_ARGS[0]` upgrade safeguard before pipeline execution. |
| **9** | Workspace Test Suite | **Verify** | Execute `python -m unittest discover -s tests` to ensure all 105+ unit tests pass. |

---

## 8. Conclusion
This technical specification resolves the root cause of the Python 3.6 `ModuleNotFoundError: No module named 'dataclasses'` failure through a dual-pronged approach:
1. Providing an in-tree, zero-dependency compatibility shim allowing AutoScaffold to execute reliably on legacy Python 3.6 hosts.
2. Introducing intelligent environment discovery in `env_setup.sh` and handshake safeguards in `chimera_scaffold.sh` to automatically detect and utilize modern Conda/Micromamba Python >= 3.8 environments whenever they are present.
