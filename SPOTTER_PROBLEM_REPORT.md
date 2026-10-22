# SPOTTER PROBLEM REPORT: Neural Python Environment Resolution, Pre-Flight Verification & Path Canonicalization in AutoScaffold

**Date**: 2026-10-22  
**Target Repository**: `C:\Users\g_san\.gemini\antigravity\scratch\autoscaffold`  
**Reference Specification**: `RFDIFF_ENV_SPEC.md` (v1.0.0)  
**Role**: Spotter Agent (Orchestai Pipeline)  
**Status**: Comprehensive Problem Report Complete  

---

## 1. Executive Summary & Incident Root Cause Analysis

### 1.1 The Incident
During user execution on Linux (Ubuntu / Rocky Linux):
```
[+] Running RFdiffusion: /home/memo/miniconda3/bin/python /home/memo/programs/autoscaffold/tools/RFdiffusion/scripts/run_inference.py inference.output_prefix=/home/memo/projects/autoscaffolding/outputs/outputs/chimera_scaffold/1_rfdiffusion/chimera_design inference.input_pdb=/home/memo/projects/autoscaffolding/outputs/outputs/chimera_scaffold/0_prep/chimera_seed.pdb 'contigmap.contigs=[V285-427/10/D1-2/3-9/D3-7/11-17/D8-8/3-9/D9-11/6-12/D12-12/3-9/D13-15/6-12/D16-16]' inference.num_designs=10 denoiser.noise_scale_ca=0.0 denoiser.noise_scale_frame=0.0
Traceback (most recent call last):
  File "/home/memo/programs/autoscaffold/tools/RFdiffusion/scripts/run_inference.py", line 20, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
subprocess.CalledProcessError: Command '['/home/memo/miniconda3/bin/python', ...]' returned non-zero exit status 1.
```

Two major defects caused this fatal failure:
1. **Hardcoded Subprocess Python Interpreters (`sys.executable`)**: The Python orchestrator (`scripts/chimera_scaffold.py`) invokes external neural modeling tools (`run_inference.py` for RFdiffusion and `protein_mpnn_run.py` for ProteinMPNN) using `sys.executable`. When `chimera_scaffold.sh` is invoked from a base miniconda environment (`/home/memo/miniconda3/bin/python`), `sys.executable` points to base Python where PyTorch (`torch`) and Hydra (`hydra-core`) are not installed.
2. **Accidental Nested Path Duplication (`/outputs/outputs/`)**: The shell wrapper defaults `OUT_DIR` to the relative string `"outputs/chimera_scaffold"`. When executed inside the project's `outputs/` folder, `fast_realpath` resolves against `$PWD`, producing redundant paths like `/home/memo/projects/autoscaffolding/outputs/outputs/chimera_scaffold`. Neither the shell script nor the Python path validator (`validate_output_dir`) performs lexical de-duplication.

---

## 2. In-Depth Component Analysis & Code Audit

### 2.1 `scripts/chimera_scaffold.py`

#### A. Hardcoded `sys.executable` for Subprocesses
- **Lines 1282–1291 (`rfdiff_cmd` construction)**:
  ```python
  rfdiff_cmd = [
      sys.executable,
      self.rfdiff_script or "scripts/run_inference.py",
      f"inference.output_prefix={os.path.join(work_rfdiff, 'chimera_design')}",
      ...
  ]
  ```
  `sys.executable` is strictly hardcoded. It forces RFdiffusion to run under the same Python interpreter as `chimera_scaffold.py`.
- **Lines 1345–1355 (`mpnn_cmd` construction)**:
  ```python
  mpnn_cmd = [
      sys.executable,
      self.mpnn_script or "protein_mpnn_run.py",
      "--pdb_path", bb,
      ...
  ]
  ```
  ProteinMPNN is also hardcoded to `sys.executable`. In biological structural workflows, RFdiffusion often requires an `SE3nv` conda environment (PyTorch 1.12-2.0 + se3-transformer + Hydra), whereas ProteinMPNN may live in `mlfold` or `proteinmpnn`. Hardcoding `sys.executable` to both makes multi-environment coexistence impossible.

#### B. Missing CLI Arguments in `main()` and `ChimeraScaffolder`
- `main()` / `argparse` (lines 1580–1630):
  - Does **NOT** accept `--python_rfdiff`.
  - Does **NOT** accept `--python_mpnn`.
  - Does **NOT** accept `--python_colabfold`.
- `ChimeraScaffolder.__init__` (lines 223–250):
  - Accepts `rfdiff_script`, `mpnn_script`, `colabfold_cmd`, but has no parameters for interpreter binaries (`python_rfdiff`, `python_mpnn`, `python_colabfold`).
  - No 5-tier environment discovery logic exists (`resolve_neural_python` is missing entirely).

#### C. Absence of Pre-Flight Probes & Verification
- No lightweight probe exists to verify whether the target interpreter can successfully `import torch` and `import hydra` before launching heavy computations.
- No GPU / CUDA availability probe (`torch.cuda.is_available()`) or hardware warning.
- When an import fails during `run_inference.py`, Python raises `subprocess.CalledProcessError` with a raw stack trace, providing zero actionable remediation commands for the user.

#### D. Path Handling in Python
- `self.out_dir = validate_output_dir(out_dir)` (line 283): relies on `scripts/pse_pdb_extractor.py:validate_output_dir()`.

---

### 2.2 `chimera_scaffold.sh`

#### A. Missing Python Resolution and Flags
- CLI Parameter Parsing (lines 124–251):
  - Does **NOT** parse `--python_rfdiff`.
  - Does **NOT** parse `--python_mpnn`.
- Pipeline Execution & Command Assembly (lines 617–632):
  ```bash
  CMD_ARGS=(
      "python" "$SCRIPT_DIR/scripts/chimera_scaffold.py"
      ...
  )
  ```
  - It runs bare `"python"` rather than a resolved Python interpreter.
  - `--python_rfdiff` and `--python_mpnn` are completely absent from `CMD_ARGS` and `BG_ARGS` (lines 573–597).
  - Shell-level environment resolution function (`discover_conda_python`) is completely missing.
- Ineffective late environment activation (lines 675–677):
  ```bash
  if [ -n "$CONDA_ENV_RFDIFF" ]; then
      activate_env "$CONDA_ENV_RFDIFF" 2>/dev/null || true
  fi
  ```
  This is called right before `${CMD_ARGS[@]}`. However, in non-interactive bash subshells, `conda activate` often fails or is a no-op without profile hooks, and `CMD_ARGS` had already hardcoded `"python"`. If ProteinMPNN is needed later, activating RFdiffusion's environment does not provide ProteinMPNN's dependencies.

#### B. Output Directory Resolution & Nested Duplication (`outputs/outputs`)
- Line 118: `OUT_DIR="outputs/chimera_scaffold"`
- Line 548: `ABS_OUT_DIR="$(fast_realpath "$OUT_DIR")"`
- Root Cause of `/outputs/outputs/chimera_scaffold`:
  1. `OUT_DIR` is a relative path.
  2. If the user invokes `chimera_scaffold.sh` from within the `outputs` directory (or if `$PWD` ends in `outputs`), `fast_realpath "$OUT_DIR"` joins `$PWD` with `outputs/chimera_scaffold`, producing `.../outputs/outputs/chimera_scaffold`.
  3. Default `OUT_DIR` is never anchored to `$SCRIPT_DIR`.
  4. There is no `canonicalize_out_dir` function in `chimera_scaffold.sh` to collapse duplicated `/outputs/outputs/`.

---

### 2.3 `scripts/env_setup.sh` & `config.env.example`

#### A. `scripts/env_setup.sh`
- Contains variables: `CONDA_ENV_RFDIFF="${CONDA_ENV_RFDIFF:-SE3nv}"`, `CONDA_ENV_MPNN="${CONDA_ENV_MPNN:-mlfold}"`.
- Lacks `discover_conda_python()`:
  - Cannot inspect multiple candidate conda roots (`~/miniconda3`, `~/miniforge3`, `~/mambaforge`, `~/anaconda3`, `~/.conda`, `/opt/conda`, `/usr/local/miniconda3`).
  - Cannot query `conda info --envs` directly to resolve the exact binary path of an environment.
  - Does not export `PYTHON_RFDIFF` or `PYTHON_MPNN`.

#### B. `config.env.example`
- Currently contains only 5 generic lines:
  ```bash
  PDB_FILE=path/to/input.pdb
  PSE_FILE=path/to/input.pse
  FIXED_MOTIF=residues_to_preserve
  RECIPIENT_PROTEIN=recipient_protein
  OUTPUT_DIR=path/to/output
  ```
- Missing configuration variables:
  - `PYTHON_RFDIFF`, `PYTHON_MPNN`, `PYTHON_COLABFOLD`
  - `CONDA_ENV_RFDIFF`, `CONDA_ENV_MPNN`, `CONDA_ENV_COLABFOLD`
  - `RFDIFF_DIR`, `MPNN_DIR`

---

### 2.4 `scripts/pse_pdb_extractor.py`

#### A. `validate_output_dir` (lines 111–134)
```python
def validate_output_dir(out_dir: str) -> str:
    if not out_dir or not str(out_dir).strip():
        raise ValueError("Output directory cannot be empty.")
    raw_str = str(out_dir).strip()
    real_out = os.path.realpath(os.path.abspath(raw_str))
    ...
```
- It performs system directory and traversal checks, but performs **no lexical de-duplication**.
- Consecutive repeated `/outputs/outputs/` directory segments remain uncollapsed.

---

## 3. Requirements Traceability Gap Analysis

| Requirement ID | Description from `RFDIFF_ENV_SPEC.md` | Existing Code Status | Verdict / Gap |
| :--- | :--- | :--- | :--- |
| **REQ-ENV-1** | Explicit CLI arguments (`--python_rfdiff`, `--python_mpnn`, `--python_colabfold`) in `chimera_scaffold.py` | Absent in `parse_args()` and `ChimeraScaffolder.__init__` | **FAIL / MISSING** |
| **REQ-ENV-2** | 5-Tier Fallback Environment Discovery Engine (`resolve_neural_python`) in Python & `discover_conda_python` in Shell | Absent in both `chimera_scaffold.py` and `env_setup.sh` | **FAIL / MISSING** |
| **REQ-ENV-3** | Shell wrapper parameter forwarding (`--python_rfdiff` and `--python_mpnn` in `CMD_ARGS` and `BG_ARGS`) | Absent in `chimera_scaffold.sh` | **FAIL / MISSING** |
| **REQ-ENV-4** | Configuration file integration for Python paths in `config.env.example` | Absent in `config.env.example` | **FAIL / MISSING** |
| **REQ-PATH-1** | Path canonicalization anchoring default relative output directories to `$SCRIPT_DIR` | Relative path anchored to `$PWD` | **FAIL / DEFECTIVE** |
| **REQ-PATH-2** | Regex / token de-duplication eliminating repeated `/outputs/outputs/` segments across shell and Python | Absent in `chimera_scaffold.sh` and `validate_output_dir()` | **FAIL / DEFECTIVE** |
| **REQ-PROBE-1**| Pre-flight subprocess probe validating `import torch` and `import hydra` before launching RFdiffusion | Absent in `chimera_scaffold.py` | **FAIL / MISSING** |
| **REQ-PROBE-2**| Pre-flight subprocess probe validating `import torch` before launching ProteinMPNN | Absent in `chimera_scaffold.py` | **FAIL / MISSING** |
| **REQ-PROBE-3**| Hardware inspection (CUDA GPU probe) with non-fatal user warning on CPU execution | Absent in `chimera_scaffold.py` | **FAIL / MISSING** |
| **REQ-DIAG-1** | High-visibility terminal alert boxes with actionable remediation commands upon dependency failure | Absent; pipeline throws unhandled `CalledProcessError` | **FAIL / MISSING** |

---

## 4. Remediation Blueprint & Implementation Checklist

### 4.1 Modifications Required in `scripts/chimera_scaffold.py`
1. **Implement `probe_python_environment(python_bin, required_modules, check_cuda=True, timeout_sec=10.0)`**:
   - Executes Python one-liner via `subprocess.run([python_bin, "-c", ...], capture_output=True, timeout=timeout_sec)`.
   - Returns structured dict: `{"missing_modules": [...], "cuda_available": bool, "torch_version": str, "cuda_device_name": str}`.
2. **Implement `display_dependency_alert(tool_name, python_bin, missing_modules, suggested_env)`**:
   - Renders high-visibility terminal alert box with exact, actionable options (Option 1: conda activate, Option 2: CLI flag, Option 3: config.env, Option 4: autoscaffold install).
3. **Implement `resolve_neural_python(explicit_path, env_var_names, target_env_names, required_modules, tool_label)`**:
   - Implements the strict 5-tier fallback hierarchy:
     - Tier 1: CLI flag
     - Tier 2: Environment variables (`PYTHON_RFDIFF`, `RFDIFF_PYTHON`, `PYTHON_MPNN`, `MPNN_PYTHON`)
     - Tier 3: Active conda prefix (`$CONDA_PREFIX/bin/python`)
     - Tier 4: Filesystem scan across standard conda roots (`~/miniconda3`, `~/miniforge3`, `~/mambaforge`, `~/anaconda3`, `~/.conda`, `/opt/conda`, `/usr/local/miniconda3`, `$LOCAL_ENVS_DIR`, Windows equivalents)
     - Tier 5: Fallback to `sys.executable`
4. **Update `ChimeraScaffolder.__init__`**:
   - Add `python_rfdiff`, `python_mpnn`, `python_colabfold` parameters.
   - Resolve interpreters during init and store `self.python_rfdiff`, `self.python_mpnn`.
5. **Update Subprocess Command Lists**:
   - In `rfdiff_cmd`: replace `sys.executable` with `self.python_rfdiff`.
   - In `mpnn_cmd`: replace `sys.executable` with `self.python_mpnn`.
6. **Integrate Pre-Flight Probes in `run_pipeline()`**:
   - Before Stage 1 (Backbone generation): Probe `self.python_rfdiff` for `["torch", "hydra"]`. If missing, display alert and raise `RuntimeError`. If CUDA is missing, emit warning banner.
   - Before Stage 3 (ProteinMPNN): Probe `self.python_mpnn` for `["torch"]`. If missing, display alert and raise `RuntimeError`.
7. **Update `parse_args()` in `main()`**:
   - Add `--python_rfdiff`, `--python_mpnn`, `--python_colabfold` arguments and pass them to `ChimeraScaffolder`.

### 4.2 Modifications Required in `chimera_scaffold.sh`
1. **Add CLI parsing** for `--python_rfdiff` and `--python_mpnn`.
2. **Add `canonicalize_out_dir()`**:
   - Anchor default to `$SCRIPT_DIR/outputs/chimera_scaffold`.
   - Detect if `$PWD` ends in `outputs` and trim relative leading `outputs/`.
   - Collapse repeated `/outputs/outputs/` segments using bash parameter substitution loop.
3. **Resolve Neural Python Binaries**:
   - Call `discover_conda_python` for RFdiffusion (`SE3nv` / `rfdiffusion`) and ProteinMPNN (`mlfold` / `SE3nv`).
   - Forward `--python_rfdiff "$PYTHON_RFDIFF"` and `--python_mpnn "$PYTHON_MPNN"` in both `CMD_ARGS` and `BG_ARGS`.

### 4.3 Modifications Required in `scripts/env_setup.sh`
1. **Add `discover_conda_python()`**:
   - Checks active `$CONDA_PREFIX`.
   - Checks local `$LOCAL_ENVS_DIR`.
   - Checks standard conda installation roots.
   - Fallback query via `conda info --envs`.
   - Exports `PYTHON_RFDIFF` and `PYTHON_MPNN` if found.

### 4.4 Modifications Required in `scripts/pse_pdb_extractor.py`
1. **Update `validate_output_dir()`**:
   - Add `re.sub(r'([/\\]outputs)+([/\\]outputs)', r'\1', raw_str, flags=re.IGNORECASE)` to eliminate duplicate nested output folders across any OS platform.

### 4.5 Modifications Required in `config.env.example`
1. Add documented template entries for:
   - `PYTHON_RFDIFF`, `PYTHON_MPNN`, `PYTHON_COLABFOLD`
   - `CONDA_ENV_RFDIFF`, `CONDA_ENV_MPNN`, `CONDA_ENV_COLABFOLD`
   - `RFDIFF_DIR`, `MPNN_DIR`

### 4.6 Verification & Test Suite Plan (`tests/test_chimera_scaffold.py`)
1. Add unit test `test_resolve_neural_python_hierarchy` verifying Tier 1 through Tier 5 resolution.
2. Add unit test `test_probe_python_environment` verifying detection of installed modules vs missing modules.
3. Add unit test `test_path_deduplication_validate_output_dir` validating collapsing of repeated `/outputs/outputs/`.
4. Add unit test `test_preflight_alert_formatting` checking alert box text output.
5. Add unit test `test_dry_run_with_custom_interpreters` ensuring full pipeline dry-run passes when custom python paths are supplied.

---
**Report compiled and verified by Spotter Agent.**
