# Spotter Problem Report: Dataclass Compatibility & Python Interpreter Prioritization

- **Target System**: AutoScaffold Suite
- **Document ID**: `SPOTTER_PROBLEM_REPORT_DATACLASS_COMPAT`
- **Auditor**: Spotter (Orchestai Pipeline)
- **Reference Specification**: `docs/specifications/DATACLASS_AND_PYTHON_COMPAT_SPEC.md`
- **Status**: Complete & Verified

---

## 1. Executive Summary

This audit assesses the root causes of runtime failures when AutoScaffold is executed on legacy host environments (such as Ubuntu 18.04 LTS Bionic with `/usr/bin/python3` at Python 3.6.9). 

The audit identified three major structural defect vectors:
1. **ModuleNotFoundError on `dataclasses`**: The modular `chimera_stages` architecture directly imports `from dataclasses import dataclass, field` across core pipeline stages (`stage1_parsing.py`, `stage4_runners.py`, `run_chimera_pipeline.py`). Because `dataclasses` was only introduced in Python 3.7 (PEP 557), any execution under Python 3.6 without an externally installed PyPI backport immediately crashes.
2. **Naive Interpreter Resolution in `scripts/env_setup.sh`**: Lines 33–47 resolve `PYTHON_BIN` using `command -v python3` without inspecting interpreter versions or capability matrices (`dataclasses`, `numpy`). Consequently, `/usr/bin/python3` is locked as `PYTHON_BIN`, ignoring fully functional Conda/Micromamba virtual environments containing Python >= 3.8 and scientific libraries installed under user homes.
3. **Decoupled Conda Handshake & Frozen Command Dispatch in Shell Launchers**: In `chimera_scaffold.sh` (lines 728–799), `CMD_ARGS=("$PYTHON_BIN" ...)` binds the Python interpreter at line 729 *before* attempting `activate_env` at line 796. Furthermore, `activate_env "$CONDA_ENV_RFDIFF" 2>/dev/null || true` swallows failures silently, ensuring that the legacy, incompatible host interpreter is dispatched.

---

## 2. Complete Audit of `dataclass` Usage across Codebase

A complete search across `scripts/chimera_stages/`, `scripts/`, and `tests/` identifies all direct definitions, imports, and field features:

### 2.1 File-by-File Inventory

| File | Direct Dataclass Import | Line(s) | Dataclasses Defined / Re-exported | Features Used |
|:---|:---|:---|:---|:---|
| `scripts/chimera_stages/stage1_parsing.py` | `from dataclasses import dataclass, field` | 16 | `MotifAtom`, `MotifResidue`, `MotifSegment`, `MultiChainMotifManifest` | `frozen=True`, default literals, `default_factory=dict`, class `@property` |
| `scripts/chimera_stages/stage2_clash.py` | None (Indirect import) | 44 | Re-exports / uses Stage 1 data | Consumes `MotifSegment`, `MotifResidue`, `MotifAtom` |
| `scripts/chimera_stages/stage3_geometry.py` | None (Indirect import) | N/A | Interacts with Stage 1 data | Consumes `MotifSegment` coordinates |
| `scripts/chimera_stages/stage4_runners.py` | `from dataclasses import dataclass, field` | 21 | `RunnerSpec` | Default literals (`None`, `False`), `default_factory=list`, instance methods |
| `scripts/chimera_stages/stage5_export.py` | None (Indirect import) | N/A | Interacts with Stage 1 & 4 data | Consumes `MultiChainMotifManifest` & `RunnerSpec` |
| `scripts/chimera_stages/__init__.py` | Indirect re-exports | 24–27, 64 | Re-exports all Stage 1 dataclasses and `RunnerSpec` | Cascades import failure if Stage 1 or 4 fails |
| `scripts/chimera_scaffold.py` | Facade re-exports | 65–68, 114 | Re-exports all Stage 1 dataclasses and `RunnerSpec` | Facade interface for CLI and legacy tests |
| `scripts/run_chimera_pipeline.py` | `from dataclasses import dataclass, field` | 15 | `PipelineContext`, `PipelineResult` | 40+ scalar defaults, multiple `default_factory=dict` and `default_factory=list`, nested dataclasses |
| `tests/test_chimera_scaffold.py` | Consumes `RunnerSpec` | 48, 1356–1365, 1528–1559, 1711, 1782–1792 | `RunnerSpec` instantiation & testing | Attribute access, mutation, method execution |
| `tests/test_multichain_scaffolding.py` | Consumes Stage 1 dataclasses | 122, 146, 596, 645 | Instantiation of `MotifSegment`, `MotifResidue`, `MotifAtom` | Property access, structural checks |

---

### 2.2 Detailed Dataclass Definitions & Feature Matrix

#### 1. `MotifAtom` (`scripts/chimera_stages/stage1_parsing.py`, Lines 126–132)
```python
@dataclass(frozen=True)
class MotifAtom:
    atom_name: str
    element: str
    coords: np.ndarray  # shape: (3,), dtype: float64
    occupancy: float = 1.0
    b_factor: float = 0.0
```
- **Features Used**:
  - `@dataclass(frozen=True)`: Instances must be immutable. Any modification after initialization (`atom.occupancy = 0.5`) must raise `FrozenInstanceError` (or `AttributeError`).
  - Default literal parameters: `occupancy: float = 1.0`, `b_factor: float = 0.0`.
  - Non-scalar type annotation: `np.ndarray`.
  - Positional construction order: `atom_name`, `element`, `coords`, with optional trailing keyword/positional arguments for `occupancy` and `b_factor`.

#### 2. `MotifResidue` (`scripts/chimera_stages/stage1_parsing.py`, Lines 135–151)
```python
@dataclass
class MotifResidue:
    source_file: str
    source_chain: str
    orig_resi: int
    resn: str
    insertion_code: str = ""
    atoms: Dict[str, MotifAtom] = field(default_factory=dict)

    @property
    def ca_coords(self) -> Optional[np.ndarray]:
        ...
```
- **Features Used**:
  - `field(default_factory=dict)`: Dynamic factory instantiation on every object creation to avoid mutable default dictionary leakage across instances.
  - Default literal: `insertion_code: str = ""`.
  - Class `@property`: `ca_coords`. The fallback decorator must not confuse `@property` descriptors with dataclass fields.

#### 3. `MotifSegment` (`scripts/chimera_stages/stage1_parsing.py`, Lines 154–198)
```python
@dataclass
class MotifSegment:
    segment_id: str
    source_file: str
    source_chain: str
    residues: List[MotifResidue]
```
- **Features Used**:
  - Non-default fields exclusively.
  - Multiple helper `@property` methods: `start_resi`, `end_resi`, `length`, `n_term_ca`, `c_term_ca`, `n_term_n`, `c_term_c`.
  - Must allow normal in-memory mutation.

#### 4. `MultiChainMotifManifest` (`scripts/chimera_stages/stage1_parsing.py`, Lines 200–206)
```python
@dataclass
class MultiChainMotifManifest:
    source_files: List[str]
    segments: List[MotifSegment]
    total_residues: int
    provenance_map: Dict[int, Dict[str, Any]]
```
- **Features Used**:
  - Pure non-default fields storing nested dataclasses (`MotifSegment`).

#### 5. `RunnerSpec` (`scripts/chimera_stages/stage4_runners.py`, Lines 114–132)
```python
@dataclass
class RunnerSpec:
    runner_type: str
    command_prefix: List[str]
    env_name: Optional[str] = None
    env_prefix: Optional[str] = None
    python_binary: Optional[str] = None
    is_valid: bool = False
    cuda_available: bool = False
    torch_version: Optional[str] = None
    cuda_device_name: Optional[str] = None
    missing_modules: List[str] = field(default_factory=list)

    def build_command(self, script_or_args: Union[str, List[str]], *extra_args: str) -> List[str]:
        ...
```
- **Features Used**:
  - Mix of non-default positional arguments (`runner_type`, `command_prefix`).
  - Optional scalar defaults (`None`, `False`).
  - `field(default_factory=list)` for `missing_modules`.
  - Class method: `build_command`.
  - Mutability: Modified in-place during environment probing (`probe_python_environment`).

#### 6. `PipelineContext` (`scripts/run_chimera_pipeline.py`, Lines 99–158)
```python
@dataclass
class PipelineContext:
    input_file: Optional[str] = None
    donor_pdb: Optional[str] = None
    ...
    donor_metadata: Dict[str, Any] = field(default_factory=dict)
    vessel_metadata: Dict[str, Any] = field(default_factory=dict)
    seed_provenance: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    preserved_donor_segments: List[Tuple[int, int]] = field(default_factory=list)
    preserved_vessel_segments: List[Tuple[int, int]] = field(default_factory=list)
    last_donor_fixed: List[int] = field(default_factory=list)
```
- **Features Used**:
  - Massive dataclass with ~45 fields.
  - Every single field has either a default literal or a `default_factory` (`dict` or `list`).
  - Extensively mutated across stages 1–5 as intermediate pipeline state.

#### 7. `PipelineResult` (`scripts/run_chimera_pipeline.py`, Lines 160–170)
```python
@dataclass
class PipelineResult:
    status: str
    stage_failed: Optional[str] = None
    error_message: Optional[str] = None
    error_remediation: Optional[str] = None
    summary_data: Dict[str, Any] = field(default_factory=dict)
    passing_models: List[str] = field(default_factory=list)
    metrics_summary_csv: Optional[str] = None
    execution_time_seconds: float = 0.0
```
- **Features Used**:
  - Mandatory initial positional field (`status: str`) followed by default-valued fields.
  - `default_factory=dict` and `default_factory=list`.

---

## 3. Interpreter Resolution Audit in `scripts/env_setup.sh`

### 3.1 Line-by-Line Breakdown of Current Resolution
Lines 32–49 of `scripts/env_setup.sh` currently state:
```bash
32: # Discover and export general Python interpreter (prioritizing python3 over unaliased python)
33: if [ -z "${PYTHON_BIN:-}" ] || [ ! -x "${PYTHON_BIN:-}" ]; then
34:     if command -v python3 >/dev/null 2>&1; then
35:         PYTHON_BIN="$(command -v python3)"
36:     elif command -v python >/dev/null 2>&1; then
37:         PYTHON_BIN="$(command -v python)"
38:     elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
39:         PYTHON_BIN="$CONDA_PREFIX/bin/python"
40:     elif [ -n "${PYTHON_RFDIFF:-}" ] && [ -x "$PYTHON_RFDIFF" ]; then
41:         PYTHON_BIN="$PYTHON_RFDIFF"
42:     elif [ -n "${PYTHON_MPNN:-}" ] && [ -x "$PYTHON_MPNN" ]; then
43:         PYTHON_BIN="$PYTHON_MPNN"
44:     else
45:         PYTHON_BIN="python"
46:     fi
47: fi
48: export PYTHON_BIN
```

### 3.2 Root Cause of `/usr/bin/python3` (Python 3.6.9) Selection
1. **Blind `command -v python3` Check**: On standard Ubuntu 18.04 LTS installations, `/usr/bin/python3` exists and satisfies `command -v python3`.
2. **Zero Version & Capability Verification**: Line 34 does not evaluate the Python version (e.g. `sys.version_info >= (3, 8)`) or whether essential modules (`dataclasses`, `numpy`) can be imported.
3. **Premature Evaluation Ordering**: The check for `command -v python3` occurs *before* checking `$CONDA_PREFIX`, `$PYTHON_RFDIFF`, or any Conda paths.
4. **Delayed Model Python Discovery**: Discovery of `PYTHON_RFDIFF` and `PYTHON_MPNN` via `discover_conda_python` occurs at lines 345–360, *after* `PYTHON_BIN` has already been assigned and exported as `/usr/bin/python3`.

### 3.3 Audit of `discover_conda_python` (Lines 239–342)
- **Signature**: `discover_conda_python(target_env, req_module)`
- **Defects & Limitations**:
  1. **Strict Exact-Name Matching**: It only looks for a specific named environment `$target_env` (e.g. `$base_dir/envs/$target_env`). If a user created an environment named `conf_sel_analysis` or `scaffold_env`, `discover_conda_python "SE3nv"` will completely overlook it.
  2. **Insufficient Version Probe**: Its internal helper `_verify_python` executes `"$py" -c "import sys; import $req_module"` or `"$py" -c "import sys"`. It does not check if `sys.version_info >= (3, 8)`. Under Python 3.6, `import sys` succeeds, falsely verifying an obsolete binary if `req_module` is omitted.
  3. **No Wildcard Scanning**: It does not iterate over `$base_dir/envs/*/bin/python` to locate any available modern Python interpreter when standard named environments are missing.

### 3.4 Generalization Strategy for `scripts/env_setup.sh`
To resolve this:
- Reorder interpreter discovery into a 5-tier algorithm:
  - **Tier 0**: Pre-set valid `$PYTHON_BIN`.
  - **Tier 1**: Active virtual environment (`$CONDA_PREFIX/bin/python`) if >= 3.8.
  - **Tier 2**: System `python3` only if `python3 -c "import sys, dataclasses, numpy; sys.exit(0 if sys.version_info >= (3, 8) else 1)"` succeeds.
  - **Tier 3**: Intelligent multi-root environment scan: Scan `$LOCAL_ENVS_DIR/*/bin/python`, `$HOME/.micromamba/envs/*/bin/python`, `$HOME/miniconda3/envs/*/bin/python`, etc., for target environments (`SE3nv`, `mlfold`, `colabfold`, `conf_sel_analysis`) and any environment having Python >= 3.8 and `dataclasses`.
  - **Tier 4**: Tool-specific Python fallbacks (`$PYTHON_RFDIFF`, `$PYTHON_MPNN`).
  - **Tier 5**: Legacy fallback to host `python3` with a clear compatibility mode notice.

---

## 4. Conda Environment Handshake & Invocation Audit

### 4.1 Audit of `chimera_scaffold.sh` (Lines 720–802)
Lines 728–799 present a critical execution handshake flaw:
```bash
728: CMD_ARGS=(
729:     "$PYTHON_BIN" "$SCRIPT_DIR/scripts/chimera_scaffold.py"
730:     "--donor_id" "$DONOR_ID"
...
744: )
...
794: # Activate environment if configured
795: if [ -n "$CONDA_ENV_RFDIFF" ]; then
796:     activate_env "$CONDA_ENV_RFDIFF" 2>/dev/null || true
797: fi
798: 
799: "${CMD_ARGS[@]}"
```

#### Defect Analysis:
1. **Early Array Binding**: `CMD_ARGS[0]` is explicitly bound to `"$PYTHON_BIN"` at line 729. Even if `activate_env` at line 796 were to change the shell's `PATH` or define a new `python`, `"${CMD_ARGS[@]}"` executes the binary that was bound at line 729 (`/usr/bin/python3`).
2. **Silent Failure (`2>/dev/null || true`)**: If `activate_env` encounters an error, the error output is completely suppressed and return status masked by `|| true`. The user is never warned that the environment failed to activate.
3. **Subshell Environment Isolation**: In non-interactive bash execution without properly exported Conda shell hooks, `activate_env` cannot reliably propagate environment variables to subsequent subcommands.

### 4.2 Audit of Companion Shell Launchers
- **`discontinuous_scaffold.sh` (Lines 447–477)**:
  - Builds `CMD_ARGS` and calls `"$PYTHON_BIN" "${CMD_ARGS[@]}"` without calling `activate_env` at all. It depends 100% on whatever `PYTHON_BIN` was exported by `scripts/env_setup.sh`.
- **`autoscaffold.sh` (Lines 240, 320, 326, 387)**:
  - Calls `activate_env "$CONDA_ENV_RFDIFF"` at line 240, but later explicitly invokes `"$PYTHON_BIN" "$EXTRACTOR_SCRIPT" ...` at line 320. If `PYTHON_BIN` points to `/usr/bin/python3`, the script bypasses the active environment!
- **`de_novo.sh` (Lines 236, 295, 351, 366, 409)**:
  - Similar pattern: `activate_env` is run for tool steps, but intermediate Python helper scripts (`$FORMATTER_SCRIPT`, `methodology_reporter.py`) are dispatched using `"$PYTHON_BIN"`, bypassing the environment.

---

## 5. Edge Cases & Technical Risks in Zero-Dependency Fallback (`dataclass_compat.py`)

Implementing a zero-dependency `@dataclass` fallback requires addressing several subtle edge cases:

### 5.1 Handling `field(default=...)` and `field(default_factory=...)`
- **Default Factory Evaluation Timing**:
  - `atoms: Dict[str, MotifAtom] = field(default_factory=dict)`
  - `missing_modules: List[str] = field(default_factory=list)`
  - **Risk**: In Python, evaluating `default_factory()` at class definition time results in a single shared dict/list instance across all objects.
  - **Requirement**: `default_factory` must be stored as a callable in `_Field` and invoked inside `__init__` on each new instance: `val = field_spec.default_factory()`.
- **Parameter Ordering Validation**:
  - Dataclass semantics require all fields without defaults to precede fields with defaults.
  - If a user defines `class A: x: int = 1; y: int`, Python's `dataclasses` raises `TypeError: non-default argument 'y' follows default argument`.
  - The fallback must enforce this constraint during class decoration.

### 5.2 Handling `frozen=True` and `FrozenInstanceError`
- `MotifAtom` uses `@dataclass(frozen=True)`.
- In native Python `dataclasses`, mutating an instance raises `dataclasses.FrozenInstanceError`, which inherits from `AttributeError`.
- **Fallback Implementation Detail**:
  - Define `class FrozenInstanceError(AttributeError): pass`.
  - When `frozen=True`, `__setattr__` and `__delattr__` on the decorated class must be overridden.
  - During `__init__`, attributes must be assigned via `object.__setattr__(self, name, value)`.
  - After `__init__` sets `_initialized = True`, subsequent `__setattr__` calls must raise `FrozenInstanceError`.

### 5.3 Preserving Methods and Properties
- `MotifResidue` has `@property def ca_coords(self)`.
- `MotifSegment` has 7 `@property` methods (`start_resi`, `end_resi`, etc.).
- `RunnerSpec` has `def build_command(self, ...)`.
- **Risk**: A naive fallback iterating over `cls.__dict__` might treat methods, properties, or functions as fields.
- **Requirement**: The fallback must only consider names present in `cls.__annotations__` or explicit class attributes holding `_Field` instances, ignoring functions, descriptors, and dunder methods.

### 5.4 Recursive `asdict()` Behavior
- The pipeline converts metadata to dictionaries (e.g. `json.dumps(result.summary_data)`).
- **Requirement**: `asdict(obj)` must:
  - Check `is_dataclass(obj)`.
  - Recursively convert fields.
  - Handle nested dataclasses (e.g. `MultiChainMotifManifest` -> `MotifSegment` -> `MotifResidue` -> `MotifAtom`).
  - Handle dataclasses inside lists, tuples, and dictionaries.
  - Leave non-dataclass objects (such as `np.ndarray`, strings, floats) untouched.

### 5.5 `is_dataclass(obj_or_cls)` Compatibility
- In native `dataclasses`, `is_dataclass` checks for the presence of `_FIELDS` or `__dataclass_fields__`.
- **Requirement**: The fallback must attach a `__dataclass_fields__` dictionary mapping field names to `_Field` descriptors on both the decorated class and its instances, ensuring full compatibility whether native or fallback is active.

### 5.6 Dual-Root and Script Execution Import Resolution
- AutoScaffold modules are imported in multiple contexts:
  - From CLI runner: `python scripts/chimera_scaffold.py` (where `scripts/` is on `sys.path`).
  - As package: `from chimera_stages.stage1_parsing import ...` (where repo root is on `sys.path`).
  - From unit tests: `tests/test_chimera_scaffold.py`.
- **Requirement**: Modules must use dual/triple import fallbacks:
  ```python
  try:
      from scripts.dataclass_compat import dataclass, field, asdict, is_dataclass, FrozenInstanceError
  except ImportError:
      try:
          from .dataclass_compat import dataclass, field, asdict, is_dataclass, FrozenInstanceError
      except ImportError:
          from dataclass_compat import dataclass, field, asdict, is_dataclass, FrozenInstanceError
  ```

---

## 6. Implementation Action Plan for Downstream Agents

Based on this audit, the downstream implementation steps must proceed as follows:

1. **Step 1: Implement `scripts/dataclass_compat.py`**
   - Provide seamless stdlib pass-through if Python >= 3.7.
   - Provide standalone `_FallbackDataclass`, `field`, `asdict`, `is_dataclass`, `FrozenInstanceError` for Python 3.6.
   - Export an explicit way to test fallback mode (e.g. `_dataclass_fallback`) even on Python 3.7+.

2. **Step 2: Create Comprehensive Test Suite `tests/test_dataclass_compat.py`**
   - Test both native stdlib mode and forced fallback mode.
   - Verify defaults, `default_factory`, `frozen=True`, `FrozenInstanceError`, properties, inheritance, `asdict()`, and `is_dataclass()`.

3. **Step 3: Update Dataclass Imports across Core Modules**
   - Refactor `stage1_parsing.py`, `stage4_runners.py`, `run_chimera_pipeline.py`, and `chimera_scaffold.py` to import from `dataclass_compat`.

4. **Step 4: Refactor `scripts/env_setup.sh`**
   - Implement the 5-tier multi-root interpreter discovery logic with `sys.version_info >= (3, 8)` and capability checks.

5. **Step 5: Harden Conda Handshake in `chimera_scaffold.sh`**
   - Add pre-flight interpreter upgrade logic: if `PYTHON_BIN` is < 3.8, discover Conda tool interpreter, auto-promote `PYTHON_BIN`, and update `CMD_ARGS[0]` dynamically.
   - Log activation status explicitly instead of silently suppressing failures.

6. **Step 6: Execute Full Regression Testing**
   - Ensure all 105 existing unit tests continue to pass.
   - Ensure `diagnose_environment.py` runs cleanly.
