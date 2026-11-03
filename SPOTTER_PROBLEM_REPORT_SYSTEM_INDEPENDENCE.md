# SPOTTER PROBLEM REPORT: System Independence, Zero-CUDA Portability, Donor 3D Invariance, and Trial Advisor Engine

**Target System**: AutoScaffold Suite (`autoscaffold.sh`, `autoscaffold_suite.sh`, `de_novo.sh`, `chimera_scaffold.sh`, `scripts/chimera_scaffold.py`, `scripts/run_chimera_pipeline.py`, `scripts/chimera_stages/`, `scripts/relax_chimera.py`, `scripts/methodology_reporter.py`)  
**Workspace**: `C:\Users\g_san\.gemini\antigravity\scratch\autoscaffold`  
**Reference Specification**: `docs/specifications/SYSTEM_INDEPENDENCE_AND_TRIAL_ADVISOR_SPEC.md`  
**Author**: Spotter Agent (Orchestai Pipeline)  
**Date**: November 3, 2026  
**Status**: Comprehensive Code Audit & Architecture Defect Assessment Complete  

---

## 1. Executive Summary

This Spotter audit thoroughly examined the current codebase against the user requirements:
1. Complete machine independence (system inspection of what is missing, what exists, what needs updating on any host).
2. Execution capability without CUDA authorization (zero-privilege, CPU fallback without crashes).
3. Strict protection of the donor 3D structure (ensuring generative diffusion does not distort, mutate, or drift active sites).
4. Automated post-run optimization and trial advice (analyzing linker length strain, terminus shifts, and anchor positions).
5. Central semantic versioning (`v2.2.0`) in interactive menus, bash headers, and CLI tools, along with upstream GitHub synchronization.

### Key Finding Summary
While the codebase contains robust modular stages (`scripts/chimera_stages/stage1` through `stage5`) and 81 passing unit tests, **critical runtime gaps prevent true machine independence and graceful non-CUDA execution**:
1. **Fatal CUDA Dependency in RFdiffusion and ProteinMPNN**: When CUDA is unavailable or unprivileged, the pipeline warns the user but fails to pass `inference.device=cpu` to RFdiffusion or `--device cpu` to ProteinMPNN, triggering fatal unhandled PyTorch CUDA runtime errors.
2. **OpenMM Platform Failure Risk**: `scripts/relax_chimera.py` does not explicitly fallback to CPU/Reference platform when OpenMM attempts to initialize an unavailable CUDA driver.
3. **Missing System Doctor**: There is no Python-based `scripts/system_doctor.py` or `--doctor` CLI flag; the existing `diagnostics.sh` only performs superficial bash queries and lacks deep environment/torch introspection.
4. **Coordinate Frame Misalignment in Drift Auditing**: `stage2_clash.py` only performs Kabsch superposition if center-of-mass shift exceeds 5.0 Å. Pure rotations in SE(3) space are not aligned before computing unaligned Euclidean drift, causing false drift rejections.
5. **No Next-Trials Recommendation Engine**: `scripts/trial_advisor.py` does not exist. No biophysical reasoning on linker slack ratios ($\sigma$), terminus reach, or anchor shifts is performed at the end of runs.
6. **Fragmented Versioning**: Version `2.1.0` is hardcoded in `methodology_reporter.py`, absent from Python CLI banners, and absent from `autoscaffold_suite.sh`.

---

## 2. In-Depth Code Audit & Defect Breakdown

### 2.1 Area 1: Environment Preflight Diagnostics & Zero-CUDA Execution

#### Defect 1.1: Missing `inference.device=cpu` Injection in RFdiffusion Command
- **Affected Files**:
  - `scripts/chimera_stages/stage4_runners.py`: Lines 950–958 (`run_closed_loop_backbone_generation`)
  - `scripts/chimera_scaffold.py`: Lines 631–639 (`rfdiff_cmd` assembly)
- **Current Code**:
  ```python
  cmd = rfdiff_runner.build_command(
      rfdiff_script,
      f"inference.output_prefix={prefix}",
      f"inference.input_pdb={seed_pdb}",
      f"contigmap.contigs=[{contig_str}]",
      "inference.num_designs=1",
      "denoiser.noise_scale_ca=0.0",
      "denoiser.noise_scale_frame=0.0",
  )
  ```
- **Root Cause**: RFdiffusion's Hydra configuration (`config/inference/base.yaml`) sets `inference.device=cuda:0` by default. If `rfdiff_runner.cuda_available` is False, the runner does NOT inject `inference.device=cpu`.
- **Impact**: On any machine without CUDA drivers or without GPU permissions (e.g. unprivileged containers, CPU workstations, non-GPU cloud nodes), RFdiffusion immediately crashes with `RuntimeError: CUDA error: no kernel image is available` or `RuntimeError: Found no NVIDIA driver`.
- **Remediation**:
  If `not getattr(rfdiff_runner, "cuda_available", False)`, explicitly append `inference.device=cpu` to the Hydra CLI overrides in both `stage4_runners.py` and `chimera_scaffold.py`.

#### Defect 1.2: Missing `--device cpu` in ProteinMPNN Runner
- **Affected Files**:
  - `scripts/chimera_scaffold.py`: Lines 729–745
  - `scripts/run_chimera_pipeline.py`: Lines 526–542
- **Current Code**:
  ```python
  mpnn_cmd = ctx.mpnn_runner.build_command(
      ctx.mpnn_script or "protein_mpnn_run.py",
      "--pdb_path", bb,
      "--fixed_positions_jsonl", jsonl_path,
      "--out_folder", work_mpnn,
      "--num_seq_per_target", str(ctx.mpnn_seqs_per_bb),
      "--sampling_temp", str(ctx.mpnn_temp),
      "--batch_size", str(ctx.mpnn_batch_size),
      "--seed", "37",
  )
  ```
- **Root Cause**: `protein_mpnn_run.py` checks `torch.cuda.is_available()`. If CUDA libraries are installed in the conda environment but the driver cannot be accessed or device authorization fails, PyTorch raises an unhandled CUDA initialization exception unless `--device cpu` is passed explicitly.
- **Remediation**:
  Inspect `mpnn_runner.cuda_available`. If False, append `"--device", "cpu"`.

#### Defect 1.3: Unhandled OpenMM Platform Initialization in `scripts/relax_chimera.py`
- **Affected Files**:
  - `scripts/relax_chimera.py`: Lines 334 and 493
- **Current Code**:
  ```python
  simulation = app.Simulation(pdb.topology, system, integrator)
  ```
- **Root Cause**: Calling `Simulation` without specifying `platform` queries all registered OpenMM platforms in default priority (`CUDA` -> `OpenCL` -> `CPU` -> `Reference`). If an environment has OpenMM with CUDA support installed but runs on a machine without a functioning NVIDIA driver, `Simulation` raises `OpenMMException: Error initializing context: CUDA_ERROR_NO_DEVICE (100)`.
- **Remediation**:
  Implement platform discovery with fallback:
  ```python
  def get_openmm_platform():
      import openmm as mm
      for plat_name in ["CUDA", "OpenCL", "CPU", "Reference"]:
          try:
              platform = mm.Platform.getPlatformByName(plat_name)
              # Test context initialization
              return platform
          except Exception:
              continue
      return mm.Platform.getPlatformByName("Reference")
  ```
  Pass `platform=get_openmm_platform()` to `app.Simulation(...)`.

#### Defect 1.4: CPU Thread Oversaturation
- **Affected Files**:
  - `scripts/env_setup.sh`: Lines 53–67
  - `scripts/chimera_stages/stage4_runners.py`
- **Root Cause**: In CPU fallback mode, PyTorch and OpenMM default to spawning threads equal to all logical cores (which can exceed 64-128 threads on HPC nodes), causing severe context-switching overhead and CPU thrashing.
- **Remediation**:
  Enforce `torch.set_num_threads(min(8, os.cpu_count() or 4))` in CPU fallback mode and set `OMP_NUM_THREADS` and `MKL_NUM_THREADS`.

#### Defect 1.5: Lack of an Automated System Doctor (`scripts/system_doctor.py` and CLI `--doctor`)
- **Current State**:
  The only existing diagnostics script is `scripts/diagnostics.sh` (a bash wrapper). It has no Python module import verification, cannot probe environment manager binaries programmatically, does not verify model weights file integrity, and cannot output structured JSON.
- **Requirement**:
  Create `scripts/system_doctor.py` fulfilling the 5-point inspection specification:
  1. Host OS, architecture, kernel, and disk space.
  2. Compute resources: CPU cores, RAM, CUDA authorization probe.
  3. Package managers: `micromamba`, `conda`, `mamba`.
  4. Conda environments matrix: `SE3nv`, `mlfold`, `colabfold`.
  5. Model weights: RFdiffusion checkpoints (`ActiveSite_v01.pt`, `Base_ckpt.pt`), ProteinMPNN weights (`vanilla_model_weights/`).
  Expose `--doctor` in `scripts/run_chimera_pipeline.py`, `autoscaffold.sh`, and `autoscaffold_suite.sh`.

---

### 2.2 Area 2: Donor 3D Structural Invariance & Generative Verification

#### Defect 2.1: Threshold-Gated Kabsch Alignment in `stage2_clash.py`
- **Affected Files**:
  - `scripts/chimera_stages/stage2_clash.py`: Lines 287–306 (`audit_motif_cartesian_drift`)
- **Current Code**:
  ```python
  cand_center = np.mean(cand_arr, axis=0)
  seed_center = np.mean(seed_arr, axis=0)
  com_shift = float(np.linalg.norm(cand_center - seed_center))

  if com_shift > 5.0 and len(cand_arr) >= 3:
      try:
          R_opt, t_opt, kabsch_rmsd, cand_aligned, sq_dist = kabsch_fit(
              cand_arr, seed_arr, return_aligned=True
          )
          vec_drifts = np.sqrt(sq_dist)
          _transform_pdb_coordinates(real_cand, R_opt, t_opt)
      except Exception:
          vec_drifts = raw_vec_drifts
  else:
      vec_drifts = raw_vec_drifts
  ```
- **Root Cause**: If RFdiffusion rotates the molecule around its center of mass (e.g. `com_shift < 5.0` Å), Kabsch alignment is bypassed. The unaligned Euclidean distances `raw_vec_drifts` are evaluated directly, which can be 10–20 Å even when the internal donor geometry is 100% rigid and pristine! This causes valid candidate models to be wrongly rejected.
- **Remediation**:
  Always compute optimal Kabsch superposition on donor Cα atoms when evaluating internal structural drift:
  1. Compute Kabsch rotation $R$ and translation $t$ to map candidate donor Cα onto seed donor Cα.
  2. Transform the entire candidate structure into the crystallographic frame (`_transform_pdb_coordinates`).
  3. Measure post-superposition per-atom drift $\text{Drift}_i = \|\mathbf{x}_{\text{cand}, i} - \mathbf{x}_{\text{seed}, i}\|_2$.
  4. Enforce $D_{\text{max}} \le 0.80$ Å and $D_{\text{mean}} \le 0.50$ Å.

#### Defect 2.2: Missing Trial Diagnostic History in Closed-Loop Diffusion
- **Affected Files**:
  - `scripts/chimera_stages/stage4_runners.py`: Lines 908–998 (`run_closed_loop_backbone_generation`)
- **Root Cause**: When a candidate is rejected due to high drift, the script simply logs to stdout and deletes candidate files. It does not record attempt metrics, rejection reasons, or drift magnitudes in an object that can be passed to the Trial Advisor. If all attempts fail, it raises `RuntimeError` with no diagnostic data.
- **Remediation**:
  Accumulate `trial_history: List[Dict[str, Any]]` containing `attempt_number`, `max_drift`, `mean_drift`, `status` ("ACCEPTED" | "REJECTED_DRIFT"), and return this metadata (or attach it to the orchestrator context) so the Trial Advisor can inspect failure modes.

#### Defect 2.3: Missing Post-Minimization Donor Invariance Audit
- **Affected Files**:
  - `scripts/chimera_stages/stage5_export.py`: Lines 788–846
- **Root Cause**: While OpenMM applies harmonic restraints to motif residues, there is no post-relaxation drift audit to verify that minimization did not perturb active site heavy atoms.
- **Remediation**:
  After `relax_structure` completes, run `audit_motif_cartesian_drift` on `chimera_minimized.pdb` to confirm $D_{\text{max}} \le 0.50$ Å before finalizing output.

---

### 2.3 Area 3: Post-Run Optimization & Next-Trials Recommendation Engine

#### Defect 3.1: Complete Absence of `scripts/trial_advisor.py`
- **Current State**:
  The codebase has no Trial Advisor module. Neither `scripts/chimera_scaffold.py` nor `scripts/run_chimera_pipeline.py` provides advice on next design iterations.
- **Requirement**:
  Create `scripts/trial_advisor.py` implementing:
  1. **Biophysical Metrics Collection**:
     - Junction Euclidean gap distances $d_{\text{gap}}^N$, $d_{\text{gap}}^C$.
     - Linker slack ratios: $\sigma = (L \times 3.8\text{ \AA}) / d_{\text{gap}}$.
     - Cartesian drift and rejection rates from closed-loop diffusion.
     - Steric clash count ($d < 1.9$ Å).
  2. **Rule-Based Diagnostic Matrix**:
     - $\sigma < 1.2$ or $d > L \times 3.8$: Recommend lengthening linker (`[L_min + 4, L_max + 8]`).
     - $\sigma > 3.2$ with clashes > 5: Recommend shortening linker (`[max(3, L_min - 4), max(5, L_max - 4)]`).
     - Repeated drift rejections ($D_{\text{max}} > 0.8$ Å): Recommend shifting host junction anchor by $\pm 1$ or $\pm 2$ residues.
     - Asymmetric reach ($d_N \ll d_C$ or $d_C \ll d_N$): Recommend switching grafting mode (`ligate_nterm` or `ligate_cterm`).
     - Severe clashes at host boundary in `replace` mode: Recommend widening `vessel_replace_range`.
  3. **Artifact Generation**:
     - `next_trials_recommendations.md` written to run output directory.
     - Append recommendations to `run_methodology.txt`.
     - Embed recommendations inside `chimera_summary.json` under `"trial_recommendations"`.
     - Generate a ready-to-run bash command for the next trial.

#### Defect 3.2: Missing Pipeline Hook for Post-Run Analysis
- **Affected Files**:
  - `scripts/run_chimera_pipeline.py`: Lines 315–329
  - `scripts/chimera_scaffold.py`: Lines 760–775
- **Root Cause**:
  `ChimeraPipelineOrchestrator.run()` returns immediately after Stage 5 export without invoking any advisor. Even worse, if Stage 4 fails (all candidates rejected), `_handle_stage_failure` is called without running any biophysical geometric analysis on why linkers failed.
- **Remediation**:
  Hook `run_trial_advisor(...)` into both success and failure paths of the orchestrator.

---

### 2.4 Area 4: Versioning in CLI and Menus

#### Defect 4.1: Fragmented and Hardcoded Version Strings
- **Affected Files**:
  - `scripts/methodology_reporter.py`: Line 28 (`AUTOSCAFFOLD_VERSION = "2.1.0"`)
  - `scripts/chimera_scaffold.py`: Missing `__version__`
  - `scripts/run_chimera_pipeline.py`: Missing `--version` flag
  - `autoscaffold_suite.sh`: Missing version display in banner
  - `autoscaffold.sh`, `de_novo.sh`, `chimera_scaffold.sh`: Inconsistent headers
- **Root Cause**: There is no central version metadata module.
- **Remediation**:
  Create `scripts/version.py`:
  ```python
  __version__ = "2.2.0"
  __version_info__ = (2, 2, 0)
  __author__ = "Thinnil / AutoScaffold Team"
  __license__ = "MIT"
  ```
  Import `__version__` into `scripts/__init__.py`, `scripts/chimera_scaffold.py`, `scripts/run_chimera_pipeline.py`, `scripts/methodology_reporter.py`, and `scripts/system_doctor.py`.

#### Defect 4.2: Missing Git Upstream Sync Checker in Shell Menus
- **Affected Files**:
  - `autoscaffold_suite.sh`: Lines 170–175
  - `scripts/env_setup.sh`
- **Root Cause**: The user cannot tell from the interactive menu whether their installation is up to date or has un-pulled commits from GitHub `origin/main`.
- **Remediation**:
  Implement `check_git_sync_status()` in `scripts/env_setup.sh` and display:
  `[Version: v2.2.0 | Upstream: Up-to-date with origin/main]`
  in the `autoscaffold_suite.sh` menu banner.

---

## 3. Gap & Risk Matrix

| ID | Component | File & Line | Severity | Impact |
| :--- | :--- | :--- | :--- | :--- |
| **G-01** | RFdiffusion Runner | `stage4_runners.py:950`, `chimera_scaffold.py:631` | **CRITICAL** | Crashes immediately on CPU / non-CUDA hosts because Hydra defaults to `cuda:0`. |
| **G-02** | ProteinMPNN Runner | `chimera_scaffold.py:730`, `run_chimera_pipeline.py:527` | **HIGH** | Fails to pass `--device cpu` when CUDA is unavailable or unprivileged. |
| **G-03** | OpenMM Relaxation | `relax_chimera.py:334,493` | **HIGH** | Simulation instantiation fails if OpenMM attempts to initialize missing CUDA driver without fallback. |
| **G-04** | Cartesian Drift Audit | `stage2_clash.py:295` | **HIGH** | Skips Kabsch fit when center-of-mass shift $\le 5.0$ Å, falsely rejecting purely rotated backbones. |
| **G-05** | Preflight Diagnostics | Missing `scripts/system_doctor.py` | **MEDIUM** | No automated doctor to inspect missing dependencies, disk space, or torch status. |
| **G-06** | Next-Trials Advisor | Missing `scripts/trial_advisor.py` | **HIGH** | No guidance on linker lengths, junction reach, or anchor shifts after runs complete or fail. |
| **G-07** | Versioning & Sync | `methodology_reporter.py:28`, `autoscaffold_suite.sh:170` | **MEDIUM** | Hardcoded `2.1.0`, missing `--version` CLI flags, and no git upstream sync status in TUI menu. |

---

## 4. Concrete Recommendations & Architecture Blueprint for the Coder

### Blueprint 1: Implement Central Semantic Versioning (`scripts/version.py`)
1. Create `scripts/version.py` defining `__version__ = "2.2.0"`.
2. Update `scripts/methodology_reporter.py` to import `AUTOSCAFFOLD_VERSION` from `scripts.version`.
3. Add `--version` flag to `scripts/run_chimera_pipeline.py`.
4. Update `autoscaffold_suite.sh` and `scripts/env_setup.sh` to extract version from `scripts/version.py`.

### Blueprint 2: Implement Machine-Independent System Doctor (`scripts/system_doctor.py`)
1. Create `scripts/system_doctor.py` with CLI flags: `--doctor`, `--json`, `--version`.
2. Inspect:
   - Host platform, CPU cores, RAM, free disk space.
   - CUDA driver & PyTorch CUDA authorization (non-crashing subprocess probe).
   - Environment managers (`micromamba`, `conda`, `mamba`).
   - Conda environments (`SE3nv`, `mlfold`, `colabfold`, `openmm`).
   - Model weights existence and non-empty file size verification.
3. Wire `--doctor` flag into `scripts/run_chimera_pipeline.py`, `autoscaffold_suite.sh`, and `autoscaffold.sh`.

### Blueprint 3: Enforce Seamless Zero-CUDA Fallback Across All Runners
1. In `scripts/chimera_stages/stage4_runners.py`:
   - If `not rfdiff_runner.cuda_available`, append `"inference.device=cpu"` to RFdiffusion command list.
2. In `scripts/chimera_scaffold.py`:
   - Inject `"inference.device=cpu"` for RFdiffusion and `"--device", "cpu"` for ProteinMPNN when GPU is absent.
3. In `scripts/relax_chimera.py`:
   - Implement `get_openmm_platform()` with explicit try-except fallback chain (`CUDA` -> `OpenCL` -> `CPU` -> `Reference`).
   - Cap PyTorch CPU threads using `torch.set_num_threads(min(8, os.cpu_count() or 4))`.

### Blueprint 4: Rigorous Donor 3D Structural Invariance & Drift Auditing
1. In `scripts/chimera_stages/stage2_clash.py`:
   - Always perform Kabsch fit on donor Cα atoms to remove global rigid-body translation and rotation before calculating coordinate drift.
   - Re-superpose candidate coordinates onto reference seed crystallographic frame.
   - Strictly reject any candidate exceeding $D_{\text{max}} > 0.80$ Å or $D_{\text{mean}} > 0.50$ Å.
2. In `scripts/chimera_stages/stage4_runners.py`:
   - Track per-attempt drift statistics in `run_closed_loop_backbone_generation` and return detailed attempt history for Trial Advisor inspection.
3. In `scripts/chimera_stages/stage5_export.py`:
   - Add post-minimization drift audit on `chimera_minimized.pdb`.

### Blueprint 5: Implement Next-Trials Recommendation Engine (`scripts/trial_advisor.py`)
1. Create `scripts/trial_advisor.py` with function `generate_trial_advice(run_context, results_summary, output_dir)`.
2. Compute:
   - Gap Euclidean distances and linker slack ratios $\sigma_N, \sigma_C$.
   - Attempt rejection rates and drift values.
   - Pre- and post-relaxation steric clashes.
3. Apply heuristic decision matrix:
   - Suggest linker expansion, linker contraction, junction anchor shift, or grafting mode change.
   - Produce ready-to-run reproduction command for next trial.
4. Output:
   - `outputs/<run_name>/next_trials_recommendations.md`.
   - Append to `run_methodology.txt`.
   - Embed in `chimera_summary.json`.
5. Hook into `run_chimera_pipeline.py` and `chimera_scaffold.py` in both success and failure cases.

### Blueprint 6: Update Interactive TUI Menus & Git Sync Checker
1. In `scripts/env_setup.sh`, add `check_git_sync_status()`.
2. In `autoscaffold_suite.sh`, update header banner:
   ```
   ======================================================================
           🧪 AUTOSCAFFOLD SUITE v2.2.0: AUTONOMOUS PROTEIN ENGINEERING 🧪
          RFdiffusion  •  ProteinMPNN  •  ColabFold  •  PyMOL / PSE
   ======================================================================
   [Version: v2.2.0 | Upstream: Up-to-date with origin/main]
   ```
3. Add `doctor` subcommand routing to `scripts/system_doctor.py`.

### Blueprint 7: Verification & Testing Suite
1. Create new comprehensive unit tests in `tests/`:
   - `tests/test_system_doctor.py` (hardware introspection, mocked CPU fallback, JSON output).
   - `tests/test_donor_invariance.py` (Kabsch alignment, artificial drift rejection, SACT fidelity).
   - `tests/test_trial_advisor.py` (slack ratio math, recommendations generation, markdown output).
   - `tests/test_versioning.py` (`--version` consistency across modules).
2. Verify all 81 existing tests continue to pass 100%.

---

## 5. Conclusion & Handoff to Coder

The audit confirms that the architecture is solid, but the identified gaps—specifically CPU/no-CUDA fallback execution, threshold-less Kabsch superposition for donor invariance, the Trial Advisor engine, and unified versioning—must be implemented to satisfy the user's requirements.

The Coder can proceed directly with implementing the 7 blueprints outlined above.
