# SPOTTER PROBLEM REPORT: Ephemeral Conda/Micromamba Execution & Terminal Scrollback Preservation

- **Document Version**: 1.0.0
- **Author**: Orchestai Spotter Agent
- **Target Subsystem**: AutoScaffold Runtime & Launcher Infrastructure
- **Reference Specification**: `CONDA_RUN_AND_TERMINAL_SPEC.md`
- **Impacted Files**:
  - `scripts/chimera_scaffold.py`
  - `scripts/env_setup.sh`
  - `chimera_scaffold.sh`
  - `autoscaffold_suite.sh`
  - `install.sh`
  - `autoscaffold`
  - `tests/test_chimera_scaffold.py`

---

## 1. Executive Summary & Problem Formulation

During production runs of AutoScaffold (specifically `chimera_scaffold.py` and the main `autoscaffold` CLI launcher), two severe usability and execution defects occur:

1. **Incident 1: Direct Binary Invocation Failure on Unactivated / Incomplete Conda Environments**
   ```text
   Traceback (most recent call last):
     File "/home/memo/programs/autoscaffold/scripts/chimera_scaffold.py", line 2140, in <module>
       main()
     File "/home/memo/programs/autoscaffold/scripts/chimera_scaffold.py", line 2135, in main
       res = scaffolder.run_pipeline()
     File "/home/memo/programs/autoscaffold/scripts/chimera_scaffold.py", line 1720, in run_pipeline
       probe_res = probe_python_environment(...)
     File "/home/memo/programs/autoscaffold/scripts/chimera_scaffold.py", line 409, in probe_python_environment
       raise EnvironmentError(...)
   OSError: RFdiffusion Python interpreter '/home/memo/miniconda3/envs/SE3nv/bin/python3.9' lacks required modules: torch, hydra
   ```
   **Core Defect**:
   AutoScaffold locates a raw interpreter path (`/home/memo/miniconda3/envs/SE3nv/bin/python3.9`), assumes direct binary execution is sufficient, and locks into it without testing alternative environments (such as Micromamba environments in `~/.micromamba/envs/` where `torch` and `hydra` are present) or testing whether ephemeral activation via `conda run` / `micromamba run` resolves dynamic linker (`LD_LIBRARY_PATH`) or environment hook issues ("do and undo automatically").

2. **Incident 2: Terminal Scrollback Destruction on Every Launcher Invocation**
   Whenever a user runs `autoscaffold`, all previous terminal output, command history, and prior log outputs in the terminal emulator are erased.
   **Core Defect**:
   Unconditional invocations of `clear 2>/dev/null || true` in `autoscaffold_suite.sh` and `install.sh` send screen-wiping control sequences (`\033[2J\033[H\033[3J`) to the terminal emulator, flushing the scrollback buffer instead of printing non-destructive section dividers.

---

## 2. In-Depth Root Cause Analysis & Code Inspection

### 2.1 Inspection of `scripts/chimera_scaffold.py`

#### A. Probe Limitations in `probe_python_environment()` (Lines 301–415)
1. **String-Only Binary Assumption**:
   ```python
   def probe_python_environment(
       python_bin: str,
       ...
   ):
       real_bin = os.path.realpath(os.path.expanduser(os.path.expandvars(str(python_bin).strip())))
       if not os.path.isfile(real_bin) or not (os.access(real_bin, os.X_OK) or sys.platform == "win32"):
           ...
       proc = subprocess.run([real_bin, "-c", probe_code], ...)
   ```
   - The probe function requires `python_bin` to be a filesystem path string pointing to a local file.
   - It performs `os.path.isfile(real_bin)` and `os.access(real_bin, os.X_OK)`.
   - It executes `[real_bin, "-c", probe_code]`.
   - **Flaw**: It cannot probe command prefixes like `["micromamba", "run", "-n", "SE3nv", "python"]` or `["conda", "run", "-n", "SE3nv", "--no-capture-output", "python"]`. If passed a list or a command runner prefix, it crashes on `os.path.isfile()` or `os.path.realpath()`.

2. **No Ephemeral Activation Probe ("Do & Undo")**:
   - Neural tools such as RFdiffusion (PyTorch, SE3-Transformer, CUDA C++ extensions) require runtime shared libraries in `$CONDA_PREFIX/lib` (`libtorch_cuda.so`, `libcudart.so`) and activation environment variables (`activate.d/`). Direct binary invocation fails with dynamic linker errors even if the package is installed. `probe_python_environment()` has no fallback to test whether executing through `conda run` or `micromamba run` allows successful import.

#### B. Flaws in `resolve_python_interpreter()` & `resolve_neural_python()` (Lines 417–555)

1. **Greedy Tier 1 Lock-In on Explicit Arguments (Lines 454–461)**:
   ```python
   # Tier 1: Explicit CLI argument
   if explicit_binary and str(explicit_binary).strip():
       clean_exp = os.path.realpath(os.path.expanduser(os.path.expandvars(str(explicit_binary).strip())))
       seen_cands.add(clean_exp)
       if is_executable_binary(clean_exp):
           is_valid = probe_candidate(clean_exp)
           return (clean_exp, is_valid)
       return (clean_exp, False)
   ```
   - When `chimera_scaffold.sh` discovers an unverified interpreter via `discover_conda_python` and passes `--python_rfdiff /home/memo/miniconda3/envs/SE3nv/bin/python`, `resolve_python_interpreter()` treats it as an explicit CLI argument.
   - It probes `clean_exp` directly. When `probe_candidate(clean_exp)` evaluates to `False`, it **immediately returns `(clean_exp, False)`**.
   - It **does not** test if running via `conda run -p <prefix>` or `micromamba run -p <prefix>` fixes the import failure.
   - It **does not** continue to Tier 2, Tier 3, or Tier 4 to check whether another working environment exists.

2. **Greedy Tier 2 Lock-In on Environment Variables (Lines 463–475)**:
   - Identical logic: if `PYTHON_RFDIFF` is exported in the shell environment, it returns `(clean_val, is_valid)` on the first existing binary, even if `is_valid` is `False`.

3. **Tier 4 Filesystem Scan Blindspots (Lines 492–533)**:
   ```python
   conda_roots = [
       local_envs_dir,
       os.path.join(conda_prefix, "envs") if conda_prefix else "",
       os.path.join(conda_prefix, "..", "envs") if conda_prefix else "",
       os.path.join(home_dir, "miniconda3", "envs"),
       os.path.join(home_dir, "miniforge3", "envs"),
       os.path.join(home_dir, "mambaforge", "envs"),
       os.path.join(home_dir, "anaconda3", "envs"),
       os.path.join(home_dir, ".conda", "envs"),
       "/opt/conda/envs",
       "/usr/local/miniconda3/envs",
       "/usr/local/anaconda3/envs",
       ...
   ]
   ```
   - **Missing Micromamba Paths**: Micromamba stores environments in:
     - `~/.micromamba/envs`
     - `~/micromamba/envs`
     - `~/.local/share/mamba/envs`
     - `$MAMBA_ROOT_PREFIX/envs`
     None of these roots are present in `conda_roots`. If the working `SE3nv` environment is installed in Micromamba, Tier 4 will never look there.
   - **Raw Binary Return Without Runner Testing**:
     ```python
     if os.path.isfile(cand) and is_executable_binary(cand):
         is_valid = probe_candidate(cand)
         if is_valid or not req_mods:
             return (cand, is_valid)
     ```
     Tier 4 tests direct execution only. If direct execution fails, it does not check if an environment manager (`micromamba` or `conda`) can run it via `micromamba run -n <env> python` or `conda run -n <env> --no-capture-output python`.

4. **Primitive Return Type**:
   - `resolve_python_interpreter()` returns `Tuple[str, bool]`. A `str` cannot express an ephemeral runner command like `["micromamba", "run", "-n", "SE3nv", "python"]` without string escaping ambiguities.

#### C. Subprocess Invocation Bottleneck in `run_pipeline()` (Lines 1730–1760 & 1800–1830)

1. **RFdiffusion Execution (Lines 1731–1754)**:
   ```python
   rfdiff_cmd = [
       self.python_rfdiff,
       self.rfdiff_script or "scripts/run_inference.py",
       f"inference.output_prefix={os.path.join(work_rfdiff, 'chimera_design')}",
       f"inference.input_pdb={self.seed_pdb_path}",
       f"contigmap.contigs=[{self.contig_string}]",
       f"inference.num_designs={self.num_designs}",
       "denoiser.noise_scale_ca=0.0",
       "denoiser.noise_scale_frame=0.0"
   ]
   ...
   subprocess.run(rfdiff_cmd, check=True, shell=False, cwd=rfdiff_repo)
   ```
   - `self.python_rfdiff` is embedded as `rfdiff_cmd[0]`.
   - If `self.python_rfdiff` were a list (e.g. `["micromamba", "run", "-n", "SE3nv", "python"]`), `rfdiff_cmd` would become `[['micromamba', 'run', ...], 'scripts/run_inference.py', ...]`, which raises `TypeError: expected str, bytes or os.PathLike object, not list` in `subprocess.run()`.
   - Line 1720 also calls `probe_python_environment(self.python_rfdiff, required_modules=["torch", "hydra"], raise_on_error=True)`. If `self.python_rfdiff` failed direct probe earlier, line 1720 immediately crashes the pipeline with `OSError`.

2. **ProteinMPNN Execution (Lines 1805–1829)**:
   ```python
   mpnn_cmd = [
       self.python_mpnn,
       self.mpnn_script or "protein_mpnn_run.py",
       "--pdb_path", bb,
       ...
   ]
   ...
   subprocess.run(mpnn_cmd, check=True, shell=False, cwd=mpnn_repo)
   ```
   - Exactly identical limitation: assumes `self.python_mpnn` is a single executable string path. Cannot accept runner prefixes.

---

### 2.2 Inspection of `scripts/env_setup.sh` and `chimera_scaffold.sh`

#### A. Blind Binary Discovery in `scripts/env_setup.sh:discover_conda_python()` (Lines 164–255)

1. **Executable-Only Check Without Module or Import Validation**:
   ```bash
   for base_dir in \
       "$CONDA_PREFIX/.." \
       "$HOME/miniconda3" \
       "$HOME/miniforge3" \
       "$HOME/mambaforge" \
       "$HOME/anaconda3" \
       "$HOME/.conda" \
       "/opt/conda" \
       "/usr/local/miniconda3" \
       "/usr/local/anaconda3"; do
       if [ -d "$base_dir" ]; then
           candidate="$base_dir/envs/$target_env/bin/python"
           if [ -x "$candidate" ]; then
               echo "$candidate"
               return 0
           fi
       fi
   done
   ```
   - The shell script checks only `[ -x "$candidate" ]`.
   - If a stale or incomplete `SE3nv` environment exists in `~/miniconda3/envs/SE3nv/bin/python`, it immediately echoes that path and exits `0`.
   - It does not check if the interpreter can run, nor whether `torch` or `hydra` can be imported.

2. **Absence of Micromamba Environment Search Paths**:
   - `discover_conda_python` does not search:
     - `~/.micromamba/envs`
     - `~/micromamba/envs`
     - `$MAMBA_ROOT_PREFIX/envs`
     - `~/.local/share/mamba/envs`
   - In `find_conda_bin()` (lines 69–89), it checks `command -v micromamba`, but omits common user install locations:
     - `~/.local/bin/micromamba`
     - `~/bin/micromamba`
     - `~/.micromamba/bin/micromamba`
     - `/usr/local/bin/micromamba`

3. **No Support for Command Runners**:
   - `discover_conda_python()` only returns a file path string. It cannot return or set runner commands (e.g., `micromamba run -n SE3nv python` or `conda run -n SE3nv --no-capture-output python`).

#### B. Upstream Parameter Pollution in `chimera_scaffold.sh`

1. **Greedy CLI Injection (Lines 622–630 & 710–715)**:
   ```bash
   if [ -z "$PYTHON_RFDIFF" ] || [ ! -x "$PYTHON_RFDIFF" ]; then
       PYTHON_RFDIFF="$(discover_conda_python "${CONDA_ENV_RFDIFF:-SE3nv}" 2>/dev/null || true)"
       [ -z "$PYTHON_RFDIFF" ] && PYTHON_RFDIFF="$(discover_conda_python "rfdiffusion" 2>/dev/null || true)"
   fi
   ...
   if [ -n "$PYTHON_RFDIFF" ]; then
       CMD_ARGS+=("--python_rfdiff" "$PYTHON_RFDIFF")
   fi
   ```
   - `chimera_scaffold.sh` queries `discover_conda_python`, receives the broken `/home/memo/miniconda3/envs/SE3nv/bin/python`, and appends `--python_rfdiff <broken_path>` to the arguments.
   - When `chimera_scaffold.py` is invoked with this explicit argument, Tier 1 in Python takes over, skips all fallback search, and causes the `OSError` crash.

---

### 2.3 Terminal Clear Commands Across Shell Scripts

Static audit of `autoscaffold` shell scripts revealed the exact source of terminal scrollback erasure:

| File | Line Number | Code | Impact |
| :--- | :--- | :--- | :--- |
| `autoscaffold_suite.sh` | **Line 157** | `clear 2>/dev/null \|\| true` | Executed immediately before rendering the main TUI menu when `autoscaffold` or `autoscaffold_suite.sh` is run without subcommands. Sends screen-clear escape codes that flush the terminal emulator scrollback buffer. |
| `install.sh` | **Line 514** | `clear 2>/dev/null \|\| true` | Executed inside the interactive maintenance menu loop (`while true; do clear ...`). Clears screen on every menu iteration. |
| `autoscaffold` | Lines 54–55 | `exec bash "$AUTOSCAFFOLD_HOME/autoscaffold_suite.sh" "$@"` | Root wrapper executes `autoscaffold_suite.sh` without defining `AUTOSCAFFOLD_NO_CLEAR=1`, directly triggering Line 157 in `autoscaffold_suite.sh`. |
| `autoscaffold.sh` | None | N/A | No unshielded `clear` command present. |
| `chimera_scaffold.sh` | None | N/A | No unshielded `clear` command present. |
| `de_novo.sh` | None | N/A | No unshielded `clear` command present. |
| `discontinuous_scaffold.sh`| None | N/A | No unshielded `clear` command present. |

**UX Consequence**:
The user types `autoscaffold` or selects options in the installer, and their terminal history—including the command line they just typed, previous compiler errors, or prior run summaries—vanishes completely.

---

## 3. Requirements Traceability Matrix against `CONDA_RUN_AND_TERMINAL_SPEC.md`

| Requirement ID | Description in Spec | Current Code Status | Gap / Defect Identified |
| :--- | :--- | :--- | :--- |
| **REQ-RUN-1** | Ephemeral execution wrappers (`conda run`, `micromamba run`, `mamba run`) via command prefixes. | **Missing** | `scripts/chimera_scaffold.py` only handles raw file path strings; no runner prefix support. |
| **REQ-RUN-2** | Automatic "Do & Undo": Subprocess execution must leave caller shell state 100% unmutated. | **Missing** | Directly invokes binary or attempts shell-hook activation in `env_setup.sh`, mutating state or failing on dynamic linker dependencies. |
| **REQ-RUN-3** | Support prefix-based execution (`run -p <dir>`) when environment names are ambiguous or unlinked. | **Missing** | Only checks `-n <env_name>` in shell; Python core has no prefix-based runner logic. |
| **REQ-DISC-1** | Broaden manager discovery to locate `micromamba`, `mamba`, and `conda` across standard paths. | **Partial** | Shell script checks `command -v micromamba` but misses `~/.local/bin/micromamba`, `~/bin/micromamba`, `~/.micromamba/bin/micromamba`. Python core has no manager discovery at all. |
| **REQ-DISC-2** | Broaden environment directory discovery to search all micromamba roots (`~/.micromamba/envs`, `~/micromamba/envs`, `$MAMBA_ROOT_PREFIX/envs`). | **Missing** | Neither `scripts/env_setup.sh` nor `scripts/chimera_scaffold.py` inspects any micromamba environment paths. |
| **REQ-DISC-3** | Candidate probing loop must continue iterating over subsequent candidates if an earlier candidate fails module probe. | **Missing** | Tier 1 and Tier 2 immediately return `(clean_path, False)` upon encountering the first candidate, aborting candidate search. |
| **REQ-DISC-4** | If an explicit CLI path fails direct probe, attempt ephemeral runner probe on its parent prefix before rejecting. | **Missing** | Explicit CLI paths that fail direct probe are immediately marked invalid without attempting `conda run -p` or `micromamba run -p`. |
| **REQ-DISC-5** | Only raise `EnvironmentError` after ALL candidates across all tiers and managers have been evaluated. | **Missing** | Raises `EnvironmentError` in `run_pipeline()` because the resolver stopped at the first invalid candidate. |
| **REQ-TERM-1** | Eliminate unconditional `clear 2>/dev/null \|\| true` from `autoscaffold_suite.sh` and `install.sh`. | **Failed** | `clear` is called unconditionally on line 157 in `autoscaffold_suite.sh` and line 514 in `install.sh`. |
| **REQ-TERM-2** | Implement non-destructive horizontal rule section dividers replacing `clear`. | **Missing** | No section divider function exists; screen is cleared destructively. |
| **REQ-TERM-3** | Honor `AUTOSCAFFOLD_NO_CLEAR=1` environment variable and default to preserving scrollback. | **Missing** | Neither launcher nor shell scripts check `AUTOSCAFFOLD_NO_CLEAR`. |
| **REQ-TERM-4** | Document environment resolution and terminal preservation in `config.env.example` and `README.md`. | **Missing** | Not yet documented. |

---

## 4. Actionable Remediation Guidance

### 4.1 In `scripts/chimera_scaffold.py`

1. **Introduce `RunnerSpec` Dataclass**:
   - Fields: `runner_type` (`'direct'`, `'micromamba'`, `'conda'`, `'mamba'`), `command_prefix: List[str]`, `env_name: Optional[str]`, `env_prefix: Optional[str]`, `python_binary: Optional[str]`, `is_valid: bool`, `cuda_available: bool`, `missing_modules: List[str]`.
   - Method: `build_command(script_or_args, *extra_args) -> List[str]`.
2. **Update `probe_python_environment()`**:
   - Accept `Union[str, List[str], RunnerSpec]` as the first parameter.
   - If a list or `RunnerSpec` is passed, extract `command_prefix` and invoke `cmd_prefix + ["-c", probe_code]`.
   - Maintain caching based on `(tuple(cmd_prefix), tuple(sorted(required_modules)))`.
3. **Refactor `resolve_python_interpreter()` into `resolve_execution_runner()`**:
   - Discover available managers (`micromamba`, `mamba`, `conda`).
   - Discover candidate environments across both Conda roots and Micromamba roots (`~/.micromamba/envs`, `~/micromamba/envs`, `~/.local/share/mamba/envs`, `$MAMBA_ROOT_PREFIX/envs`).
   - Probing loop: For each candidate, probe directly `[cand_bin, "-c", code]`. If that fails, probe with ephemeral runner `[mgr, "run", "-n" / "-p", ... "python", "-c", code]`.
   - If candidate passes module probe, keep it; if not, do **not** abort—continue searching subsequent candidates.
   - Retain `resolve_python_interpreter()` and `resolve_neural_python()` as backward-compatible wrappers returning `(runner.python_binary or runner.command_prefix[0], runner.is_valid)`.
4. **Refactor Subprocess Invocations in `run_pipeline()`**:
   - Store `self.rfdiff_runner: RunnerSpec` and `self.mpnn_runner: RunnerSpec`.
   - Build commands with `self.rfdiff_runner.build_command(self.rfdiff_script, ...)` and `self.mpnn_runner.build_command(self.mpnn_script, ...)`.

### 4.2 In `scripts/env_setup.sh` and `chimera_scaffold.sh`

1. **Add `find_micromamba_bin()` and Enhance `find_conda_bin()`**:
   - Search `~/.local/bin/micromamba`, `~/bin/micromamba`, `~/.micromamba/bin/micromamba`, `/usr/local/bin/micromamba`.
2. **Expand `get_all_env_roots()`**:
   - Add all Micromamba environment paths.
3. **Fix Upstream Parameter Forwarding in `chimera_scaffold.sh`**:
   - If `discover_conda_python` discovers a binary that cannot run, do not pass `--python_rfdiff` on the CLI; allow Python's multi-candidate resolver to find the working environment.

### 4.3 In `autoscaffold_suite.sh`, `install.sh`, and `autoscaffold`

1. **Default to Scrollback Preservation**:
   - In `autoscaffold`: export `AUTOSCAFFOLD_NO_CLEAR="${AUTOSCAFFOLD_NO_CLEAR:-1}"`.
2. **Replace `clear 2>/dev/null || true`**:
   - In `autoscaffold_suite.sh:157` and `install.sh:514`, check:
     ```bash
     if [ "${AUTOSCAFFOLD_NO_CLEAR:-1}" != "1" ] && [ "${AUTOSCAFFOLD_CLEAR:-0}" = "1" ]; then
         clear 2>/dev/null || true
     else
         echo ""
         COLS=$(tput cols 2>/dev/null || echo 80)
         [ "$COLS" -gt 80 ] && COLS=80
         printf '%.0s─' $(seq 1 "$COLS")
         echo -e "\n"
     fi
     ```

---

## 5. Verification Plan

1. **Unit Tests in `tests/test_chimera_scaffold.py`**:
   - `test_runner_spec_command_building`: verify direct and ephemeral runner command lists.
   - `test_probe_python_environment_with_command_prefix`: verify probe executes via command prefix list.
   - `test_resolve_runner_iterates_past_invalid_candidates`: mock Candidate 1 lacking `torch`, Candidate 2 possessing `torch`; assert Candidate 2 is selected.
   - `test_terminal_scripts_contain_no_bare_clear`: regex scan asserting zero unshielded `\bclear\b` in shell scripts.
2. **End-to-End Regression**:
   - Run `python -m unittest discover -s tests -p "test_chimera_scaffold.py"` to verify 100% test pass rate across the full suite.

---

### 📋 Action Receipt
- **Execution Mode**: Antigravity Cloud / Hybrid
- **Files Created / Modified**:
  - `C:\Users\g_san\.gemini\antigravity\scratch\autoscaffold\SPOTTER_PROBLEM_REPORT_CONDA_RUN.md`
- **Commands Executed**: None (File Creation via Tool)
- **External Services / Remotes**: None
- **Status / Verdict**: Complete (Problem Report Generated & Documented)
