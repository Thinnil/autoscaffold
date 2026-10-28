"""
scripts/run_chimera_pipeline.py
Central Orchestrator for the Chimera Scaffolding Pipeline.

Executes Stages 1 through 5 sequentially with strict step validation,
diagnostic error reporting, and typed context and result dataclasses.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# Resolve scripts directory for imports
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from chimera_stages.stage1_parsing import (
    MultiChainMotifManifest,
    prepare_input_structures,
    validate_output_dir,
    _RE_SANITIZE_IDENT,
    _RE_SHELL_INJECTION,
    _RE_OUTPUTS_DEDUP_MID,
)
from chimera_stages.stage2_clash import (
    map_fixed_positions_for_mpnn,
)
from chimera_stages.stage3_geometry import (
    compute_contigs_and_segments,
    measure_junctions,
)
from chimera_stages.stage4_runners import (
    RunnerSpec,
    resolve_execution_runner,
    probe_python_environment,
    display_cuda_warning,
    run_closed_loop_backbone_generation,
)
from chimera_stages.stage5_export import (
    create_merged_seed_pdb,
    extract_mpnn_sequences,
    thread_sequence_onto_pdb,
    export_pipeline_outputs,
    safe_join_path,
)

try:
    from scripts.format_complex_fasta import make_fixed_positions_dict
except ImportError:
    from format_complex_fasta import make_fixed_positions_dict


class StageExecutionError(Exception):
    """Raised when an explicit failure occurs during a pipeline stage."""

    def __init__(self, stage_name: str, message: str, remediation_tips: Optional[List[str]] = None):
        super().__init__(f"[{stage_name}] {message}")
        self.stage_name = stage_name
        self.message = message
        self.remediation_tips = remediation_tips or []


@dataclass
class PipelineContext:
    # Inputs & Targets
    input_file: Optional[str] = None
    donor_pdb: Optional[str] = None
    vessel_pdb: Optional[str] = None
    donor_id: Any = "A"
    vessel_id: Any = "B"
    motif_residues: str = ""
    motifs: Optional[str] = None
    chimera_mode: str = "replace"
    vessel_replace_range: Optional[str] = None
    vessel_split_after: Optional[int] = None
    linker_n: str = "6-10"
    linker_c: str = "6-10"

    # Execution Parameters
    num_designs: int = 10
    mpnn_seqs_per_bb: int = 2
    mpnn_temp: float = 0.1
    mpnn_batch_size: int = 1
    rmsd_threshold: float = 1.0
    plddt_threshold: float = 75.0
    lock_vessel: bool = True
    vessel_redesign_mode: str = "linkers_only"
    clamp_linkers: bool = True
    out_dir: str = "outputs/chimera_scaffold"
    dry_run: bool = False
    strict_python: bool = False

    # Resolved Tool Scripts & Runners
    rfdiff_script: Optional[str] = None
    mpnn_script: Optional[str] = None
    colabfold_cmd: Optional[str] = None
    python_rfdiff: Optional[Union[str, List[str], RunnerSpec]] = None
    python_mpnn: Optional[Union[str, List[str], RunnerSpec]] = None
    python_colabfold: Optional[Union[str, List[str], RunnerSpec]] = None
    rfdiff_runner: Optional[RunnerSpec] = None
    mpnn_runner: Optional[RunnerSpec] = None
    colabfold_runner: Optional[RunnerSpec] = None

    # Intermediate Pipeline State
    working_donor_pdb: str = ""
    working_vessel_pdb: str = ""
    donor_chain_id: str = "A"
    vessel_chain_id: str = "B"
    donor_metadata: Dict[str, Any] = field(default_factory=dict)
    vessel_metadata: Dict[str, Any] = field(default_factory=dict)
    is_multichain: bool = False
    manifest: Optional[MultiChainMotifManifest] = None
    seed_provenance: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    seed_pdb_path: str = ""
    seed_pdb: str = ""
    contig_string: str = ""
    preserved_donor_segments: List[Tuple[int, int]] = field(default_factory=list)
    preserved_vessel_segments: List[Tuple[int, int]] = field(default_factory=list)
    seed_chain_id: str = "A"
    donor_seed_offset: int = 0
    last_donor_fixed: List[int] = field(default_factory=list)


@dataclass
class PipelineResult:
    status: str  # "SUCCESS", "FAILED", "INITIALIZED"
    stage_failed: Optional[str] = None
    error_message: Optional[str] = None
    error_remediation: Optional[str] = None
    summary_data: Dict[str, Any] = field(default_factory=dict)
    passing_models: List[str] = field(default_factory=list)
    metrics_summary_csv: Optional[str] = None
    execution_time_seconds: float = 0.0


class ChimeraPipelineOrchestrator:
    """
    Central orchestrator sequencing stages 1 to 5 with strict validation checkpoints
    and diagnostic error reporting.
    """

    def __init__(self, context: PipelineContext):
        self.ctx = context
        self._initialize_context()

    def _initialize_context(self):
        ctx = self.ctx
        if ctx.input_file:
            ctx.input_file = os.path.realpath(ctx.input_file)
        if ctx.donor_pdb:
            ctx.donor_pdb = os.path.realpath(ctx.donor_pdb)
        if ctx.vessel_pdb:
            ctx.vessel_pdb = os.path.realpath(ctx.vessel_pdb)

        for p_label, p_val in [
            ("input_file", ctx.input_file),
            ("donor_pdb", ctx.donor_pdb),
            ("vessel_pdb", ctx.vessel_pdb),
        ]:
            if p_val:
                p_drive, p_rest = os.path.splitdrive(p_val)
                if (
                    (not p_drive and p_val.rstrip("/\\") == "")
                    or (p_drive and p_rest in ("/", "\\", ""))
                    or p_val in ("/", "\\", "//", "\\\\")
                ):
                    raise ValueError(f"Dangerous or root file specified for {p_label}: '{p_val}'")

        for id_label, id_val in [("donor_id", ctx.donor_id), ("vessel_id", ctx.vessel_id)]:
            if isinstance(id_val, str):
                if ".." in id_val or "/" in id_val or "\\" in id_val:
                    raise ValueError(f"Path traversal detected in {id_label}: '{id_val}'")
            elif isinstance(id_val, (list, tuple)):
                for item in id_val:
                    if ".." in str(item) or "/" in str(item) or "\\" in str(item):
                        raise ValueError(f"Path traversal detected in {id_label}: '{item}'")

        raw_motif = ctx.motifs if ctx.motifs is not None else ctx.motif_residues
        ctx.motif_residues = raw_motif
        ctx.is_multichain = (
            bool(ctx.motifs)
            or (":" in str(raw_motif))
            or (isinstance(ctx.donor_id, (list, tuple)) and len(ctx.donor_id) > 1)
            or (isinstance(ctx.donor_id, str) and "," in ctx.donor_id)
        )
        ctx.chimera_mode = ctx.chimera_mode.lower()

        v_mode = str(ctx.vessel_redesign_mode).lower().strip()
        if v_mode in ("vessel_full", "full", "false") or not ctx.lock_vessel:
            ctx.vessel_redesign_mode = "vessel_full"
            ctx.lock_vessel = False
        elif v_mode in ("vessel_interface", "interface"):
            ctx.vessel_redesign_mode = "vessel_interface"
            ctx.lock_vessel = True
        else:
            ctx.vessel_redesign_mode = "linkers_only"
            ctx.lock_vessel = True

        ctx.out_dir = validate_output_dir(ctx.out_dir)
        if "outputs" in ctx.out_dir.lower():
            ctx.out_dir = _RE_OUTPUTS_DEDUP_MID.sub(r"\1", ctx.out_dir)

        # Resolve Neural Runners
        if not ctx.rfdiff_runner:
            ctx.rfdiff_runner = resolve_execution_runner(
                explicit_target=ctx.python_rfdiff,
                env_var_keys=["PYTHON_RFDIFF", "RFDIFF_PYTHON"],
                target_env_names=["SE3nv", "rfdiffusion", "rf_diffusion", "se3nv"],
                required_modules=["torch", "hydra"],
                tool_label="RFdiffusion",
                strict=ctx.strict_python,
            )

        if not ctx.mpnn_runner:
            ctx.mpnn_runner = resolve_execution_runner(
                explicit_target=ctx.python_mpnn,
                env_var_keys=["PYTHON_MPNN", "MPNN_PYTHON"],
                target_env_names=["mlfold", "SE3nv", "mpnn", "proteinmpnn"],
                required_modules=["torch"],
                tool_label="ProteinMPNN",
                strict=ctx.strict_python,
            )
            if not ctx.mpnn_runner.is_valid and not ctx.strict_python:
                if ctx.rfdiff_runner.is_valid:
                    rfdiff_probe = probe_python_environment(
                        ctx.rfdiff_runner, required_modules=["torch"], check_cuda=False, raise_on_error=False
                    )
                    if len(rfdiff_probe.get("missing_modules", [])) == 0:
                        ctx.mpnn_runner = ctx.rfdiff_runner
                if not ctx.mpnn_runner.is_valid:
                    mpnn_fallback = resolve_execution_runner(
                        explicit_target=None,
                        env_var_keys=["PYTHON_MPNN", "MPNN_PYTHON"],
                        target_env_names=["SE3nv", "mpnn", "proteinmpnn", "rfdiffusion"],
                        required_modules=["torch"],
                        tool_label="ProteinMPNN (Fallback)",
                        strict=False,
                    )
                    if mpnn_fallback.is_valid:
                        ctx.mpnn_runner = mpnn_fallback

        if not ctx.colabfold_runner:
            ctx.colabfold_runner = resolve_execution_runner(
                explicit_target=ctx.python_colabfold,
                env_var_keys=["PYTHON_COLABFOLD", "COLABFOLD_PYTHON"],
                target_env_names=["colabfold", "colabfold_batch", "af2"],
                required_modules=[],
                tool_label="ColabFold",
                strict=ctx.strict_python,
            )

        if not ctx.rfdiff_script:
            for d in [
                os.environ.get("RFDIFF_DIR", ""),
                os.path.expanduser("~/programs/RFdiffusion"),
                os.path.join(SCRIPTS_DIR, "..", "tools", "RFdiffusion"),
                os.path.join(SCRIPTS_DIR, "tools", "RFdiffusion"),
                os.path.join(SCRIPTS_DIR, "..", "RFdiffusion"),
            ]:
                cand = os.path.join(d, "scripts", "run_inference.py") if d else ""
                if cand and os.path.isfile(cand):
                    ctx.rfdiff_script = os.path.realpath(cand)
                    break

        if not ctx.mpnn_script:
            for d in [
                os.environ.get("MPNN_DIR", ""),
                os.path.expanduser("~/programs/ProteinMPNN"),
                os.path.join(SCRIPTS_DIR, "..", "tools", "ProteinMPNN"),
                os.path.join(SCRIPTS_DIR, "tools", "ProteinMPNN"),
                os.path.join(SCRIPTS_DIR, "..", "ProteinMPNN"),
            ]:
                cand = os.path.join(d, "protein_mpnn_run.py") if d else ""
                if cand and os.path.isfile(cand):
                    ctx.mpnn_script = os.path.realpath(cand)
                    break

        for s_label, s_val in [("rfdiff_script", ctx.rfdiff_script), ("mpnn_script", ctx.mpnn_script)]:
            if s_val:
                if _RE_SHELL_INJECTION.search(str(s_val)):
                    raise ValueError(f"Dangerous shell operator or metacharacter detected in {s_label}: '{s_val}'")
                p_drive, p_rest = os.path.splitdrive(str(s_val))
                if (
                    (not p_drive and str(s_val).rstrip("/\\") in ("", "/", "\\"))
                    or (p_drive and p_rest in ("/", "\\", ""))
                    or str(s_val) in ("/", "\\", "//", "\\\\")
                ):
                    raise ValueError(f"Dangerous or root file specified for {s_label}: '{s_val}'")

    def run(self) -> PipelineResult:
        start_time = time.time()
        try:
            # Stage 1: Preparation, parsing & input structure resolution
            self._execute_stage1()

            # Stage 3: Geometric bridge math, linker budgeting & contig construction
            self._execute_stage3()

            # Stage 5A: Seed PDB synthesis & coordinate mapping
            self._execute_stage5_seed()

            # Stage 4 & Stage 2: Neural network execution (RFdiffusion, ProteinMPNN) with closed-loop drift auditing
            mpnn_results = self._execute_stage4_and_stage2()

            # Stage 5B: All-atom threading, relaxation, PyMOL & reporting
            summary = self._execute_stage5_export(mpnn_results)

            elapsed = time.time() - start_time
            return PipelineResult(
                status="SUCCESS",
                summary_data=summary,
                passing_models=summary.get("passing_models", []),
                metrics_summary_csv=summary.get("metrics_summary_csv"),
                execution_time_seconds=elapsed,
            )
        except StageExecutionError as e:
            return self._handle_stage_failure(e, time.time() - start_time)
        except Exception as e:
            return self._handle_unexpected_failure(e, time.time() - start_time)

    def _execute_stage1(self):
        ctx = self.ctx
        try:
            res = prepare_input_structures(
                input_file=ctx.input_file,
                donor_pdb=ctx.donor_pdb,
                vessel_pdb=ctx.vessel_pdb,
                donor_id=ctx.donor_id,
                vessel_id=ctx.vessel_id,
                out_dir=ctx.out_dir,
                is_multichain=ctx.is_multichain,
            )
            ctx.working_donor_pdb = res["working_donor_pdb"]
            ctx.working_vessel_pdb = res["working_vessel_pdb"]
            ctx.donor_metadata = res["donor_metadata"]
            ctx.vessel_metadata = res["vessel_metadata"]
            ctx.donor_chain_id = res["donor_chain_id"]
            ctx.vessel_chain_id = res["vessel_chain_id"]
        except Exception as e:
            raise StageExecutionError(
                stage_name="Stage 1 (Parsing & Preparation)",
                message=str(e),
                remediation_tips=[
                    "Check that the input donor and vessel files exist and are valid PDB or PyMOL .pse files.",
                    "Verify donor_id and vessel_id correspond to existing chain identifiers or PyMOL object names.",
                    "Ensure file paths do not contain forbidden directory traversal sequences ('..') or root escapes.",
                ],
            ) from e

    def _execute_stage3(self):
        ctx = self.ctx
        try:
            (
                contig,
                donor_segs,
                vessel_segs,
                offset,
                clamped_ln,
                clamped_lc,
                manifest,
            ) = compute_contigs_and_segments(
                working_donor_pdb=ctx.working_donor_pdb,
                working_vessel_pdb=ctx.working_vessel_pdb,
                donor_chain_id=ctx.donor_chain_id,
                vessel_chain_id=ctx.vessel_chain_id,
                motif_residues=ctx.motif_residues,
                chimera_mode=ctx.chimera_mode,
                vessel_metadata=ctx.vessel_metadata,
                donor_metadata=ctx.donor_metadata,
                linker_n=ctx.linker_n,
                linker_c=ctx.linker_c,
                vessel_replace_range=ctx.vessel_replace_range,
                vessel_split_after=ctx.vessel_split_after,
                clamp_linkers=ctx.clamp_linkers,
                seed_chain_id=ctx.seed_chain_id,
                is_multichain=ctx.is_multichain,
            )
            ctx.contig_string = contig
            ctx.preserved_donor_segments = donor_segs
            ctx.preserved_vessel_segments = vessel_segs
            ctx.donor_seed_offset = offset
            ctx.linker_n = clamped_ln
            ctx.linker_c = clamped_lc
            if manifest:
                ctx.manifest = manifest
                ctx.seed_provenance = manifest.provenance_map
        except Exception as e:
            raise StageExecutionError(
                stage_name="Stage 3 (Geometry & Contigs)",
                message=str(e),
                remediation_tips=[
                    "Verify the chimera_mode is one of: replace, ligate_nterm, ligate_cterm, or insert_loop.",
                    "Check that vessel_replace_range or vessel_split_after falls strictly within the vessel sequence bounds.",
                    "Ensure motif_residues contains valid residue numbers present in the donor structure.",
                ],
            ) from e

    def _execute_stage5_seed(self):
        ctx = self.ctx
        try:
            seed_path, prov = create_merged_seed_pdb(
                out_dir=ctx.out_dir,
                working_donor_pdb=ctx.working_donor_pdb,
                working_vessel_pdb=ctx.working_vessel_pdb,
                donor_chain_id=ctx.donor_chain_id,
                vessel_chain_id=ctx.vessel_chain_id,
                preserved_donor_segments=ctx.preserved_donor_segments,
                preserved_vessel_segments=ctx.preserved_vessel_segments,
                is_multichain=ctx.is_multichain,
                manifest=ctx.manifest,
                seed_provenance=ctx.seed_provenance,
                seed_chain_id=ctx.seed_chain_id,
                donor_seed_offset=ctx.donor_seed_offset,
            )
            ctx.seed_pdb_path = seed_path
            ctx.seed_pdb = seed_path
            ctx.seed_provenance = prov
        except Exception as e:
            raise StageExecutionError(
                stage_name="Stage 5A (Seed Synthesis)",
                message=str(e),
                remediation_tips=[
                    "Verify read/write permissions in the output directory.",
                    "Check that preserved donor and vessel segments contain ATOM records in the source files.",
                ],
            ) from e

    def _execute_stage4_and_stage2(self) -> List[Dict[str, Any]]:
        ctx = self.ctx
        work_rfdiff = safe_join_path(ctx.out_dir, "1_rfdiffusion")
        work_mpnn = safe_join_path(ctx.out_dir, "2_mpnn")
        os.makedirs(work_rfdiff, exist_ok=True)
        os.makedirs(work_mpnn, exist_ok=True)

        try:
            if not ctx.dry_run:
                probe_res = probe_python_environment(
                    ctx.rfdiff_runner,
                    required_modules=["torch", "hydra"],
                    check_cuda=True,
                    raise_on_error=True,
                    tool_name="RFdiffusion",
                    suggested_env="SE3nv",
                )
                if not probe_res.get("cuda_available", False):
                    display_cuda_warning(ctx.rfdiff_runner.display_label(), tool_name="RFdiffusion")

                generated_backbones = run_closed_loop_backbone_generation(
                    seed_pdb=ctx.seed_pdb_path,
                    contig_str=ctx.contig_string,
                    seed_provenance=ctx.seed_provenance,
                    target_passing_designs=ctx.num_designs,
                    max_attempts=max(50, ctx.num_designs * 5),
                    max_drift_cutoff=0.8,
                    mean_drift_cutoff=0.5,
                    rfdiff_runner=ctx.rfdiff_runner,
                    rfdiff_script=ctx.rfdiff_script,
                    work_dir=work_rfdiff,
                )
            else:
                mock_bb = safe_join_path(work_rfdiff, "chimera_design_0.pdb")
                with open(ctx.seed_pdb_path, "r", encoding="utf-8", buffering=65536) as sf, open(
                    mock_bb, "w", encoding="utf-8", buffering=65536
                ) as mf:
                    curr_resi = 0
                    prev_orig_resi = None
                    prev_orig_ch = None
                    for line in sf:
                        if line.startswith("ATOM  ") or line.startswith("HETATM"):
                            ch = line[21]
                            try:
                                orig_resi = int(line[22:26])
                            except ValueError:
                                orig_resi = 0
                            if ch != prev_orig_ch or orig_resi != prev_orig_resi:
                                curr_resi += 1
                                prev_orig_ch = ch
                                prev_orig_resi = orig_resi
                            mf.write(f"{line[:21]}A{curr_resi:>4}{line[26:]}")
                        elif line.startswith("END"):
                            mf.write("TER\nEND\n")
                generated_backbones = [mock_bb]

            if not ctx.dry_run and ctx.mpnn_script and os.path.isfile(ctx.mpnn_script):
                probe_python_environment(
                    ctx.mpnn_runner,
                    required_modules=["torch"],
                    check_cuda=False,
                    raise_on_error=True,
                    tool_name="ProteinMPNN",
                    suggested_env="mlfold",
                )

            mpnn_results = []
            for bb in generated_backbones:
                bb_name = os.path.splitext(os.path.basename(bb))[0]
                safe_bb_name = _RE_SANITIZE_IDENT.sub("_", bb_name) or "backbone"
                fixed_map, donor_fixed_matched = map_fixed_positions_for_mpnn(
                    backbone_pdb=bb,
                    preserved_donor_segments=ctx.preserved_donor_segments,
                    preserved_vessel_segments=ctx.preserved_vessel_segments,
                    working_donor_pdb=ctx.working_donor_pdb,
                    working_vessel_pdb=ctx.working_vessel_pdb,
                    seed_pdb_path=ctx.seed_pdb_path,
                    donor_chain_id=ctx.donor_chain_id,
                    vessel_chain_id=ctx.vessel_chain_id,
                    lock_vessel=ctx.lock_vessel,
                    vessel_redesign_mode=ctx.vessel_redesign_mode,
                    is_multichain=ctx.is_multichain,
                    chimera_mode=ctx.chimera_mode,
                )
                ctx.last_donor_fixed = donor_fixed_matched

                jsonl_path = safe_join_path(work_mpnn, f"{safe_bb_name}_fixed_positions.jsonl")
                make_fixed_positions_dict(bb_name, fixed_map, out_jsonl=jsonl_path)

                mpnn_cmd = ctx.mpnn_runner.build_command(
                    ctx.mpnn_script or "protein_mpnn_run.py",
                    "--pdb_path",
                    bb,
                    "--fixed_positions_jsonl",
                    jsonl_path,
                    "--out_folder",
                    work_mpnn,
                    "--num_seq_per_target",
                    str(ctx.mpnn_seqs_per_bb),
                    "--sampling_temp",
                    str(ctx.mpnn_temp),
                    "--batch_size",
                    str(ctx.mpnn_batch_size),
                    "--seed",
                    "37",
                )

                cand_info = {
                    "backbone": bb,
                    "fixed_positions_jsonl": jsonl_path,
                    "fixed_residues": fixed_map,
                    "donor_fixed": donor_fixed_matched,
                    "mpnn_command": mpnn_cmd,
                }

                if not ctx.dry_run and ctx.mpnn_script and os.path.isfile(ctx.mpnn_script):
                    mpnn_repo = os.path.dirname(os.path.abspath(ctx.mpnn_script))
                    cwd_dir = mpnn_repo if os.path.isdir(mpnn_repo) else None
                    subprocess.run(mpnn_cmd, check=True, shell=False, cwd=cwd_dir)

                mpnn_results.append(cand_info)

            # Sequence Threading
            for cand in mpnn_results:
                bb = cand["backbone"]
                bb_name = os.path.splitext(os.path.basename(bb))[0]
                cand["threaded_pdbs"] = []
                fa_path = safe_join_path(work_mpnn, "seqs", f"{bb_name}.fa")
                des_seqs = extract_mpnn_sequences(fa_path)
                cand["designed_sequences"] = des_seqs
                if des_seqs:
                    for s_idx, (hdr, seq) in enumerate(des_seqs, start=1):
                        th_pdb = safe_join_path(work_mpnn, f"{bb_name}_seq{s_idx}.pdb")
                        d_fixed_list = cand.get("donor_fixed", [])
                        if thread_sequence_onto_pdb(
                            bb, seq, th_pdb, seed_pdb=ctx.seed_pdb_path, fixed_donor_positions=d_fixed_list
                        ):
                            cand["threaded_pdbs"].append(th_pdb)

            return mpnn_results
        except Exception as e:
            raise StageExecutionError(
                stage_name="Stage 4 (Neural Runners & Closed Loop)",
                message=str(e),
                remediation_tips=[
                    "Check if motif coordinates are physically compatible with the vessel geometry.",
                    "Increase linker length (--linker_n / --linker_c) to relieve conformational strain.",
                    "Increase --num_designs or relax drift thresholds (--rmsd_threshold).",
                    "Ensure RFdiffusion and ProteinMPNN Python environments have torch and required dependencies.",
                ],
            ) from e

    def _execute_stage5_export(self, mpnn_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        ctx = self.ctx
        try:
            summary = export_pipeline_outputs(
                out_dir=ctx.out_dir,
                mpnn_results=mpnn_results,
                working_donor_pdb=ctx.working_donor_pdb,
                working_vessel_pdb=ctx.working_vessel_pdb,
                donor_chain_id=ctx.donor_chain_id,
                vessel_chain_id=ctx.vessel_chain_id,
                motif_residues=ctx.motif_residues,
                chimera_mode=ctx.chimera_mode,
                contig_string=ctx.contig_string,
                seed_pdb_path=ctx.seed_pdb_path,
                preserved_donor_segments=ctx.preserved_donor_segments,
                preserved_vessel_segments=ctx.preserved_vessel_segments,
                lock_vessel=ctx.lock_vessel,
                num_designs=ctx.num_designs,
                mpnn_seqs_per_bb=ctx.mpnn_seqs_per_bb,
                mpnn_temp=ctx.mpnn_temp,
                mpnn_batch_size=ctx.mpnn_batch_size,
                rmsd_threshold=ctx.rmsd_threshold,
                plddt_threshold=ctx.plddt_threshold,
                colabfold_runner=ctx.colabfold_runner,
                dry_run=ctx.dry_run,
                vessel_replace_range=ctx.vessel_replace_range,
                vessel_split_after=ctx.vessel_split_after,
                linker_n=ctx.linker_n,
                linker_c=ctx.linker_c,
                is_multichain=ctx.is_multichain,
                seed_provenance=ctx.seed_provenance,
            )
            return summary
        except Exception as e:
            raise StageExecutionError(
                stage_name="Stage 5B (Output Export)",
                message=str(e),
                remediation_tips=[
                    "Check write permissions in the results/ directory.",
                    "Verify the Kabsch RMSD alignment parameters and reference coordinate files.",
                ],
            ) from e

    def _handle_stage_failure(self, error: StageExecutionError, elapsed: float) -> PipelineResult:
        tips_str = "\n".join(f" {i + 1}. {tip}" for i, tip in enumerate(error.remediation_tips))
        box = f"""
====================================================================================================
 🛑 [CHIMERA PIPELINE FAILURE]
====================================================================================================
 Failed Stage:       {error.stage_name}
 Exception:          {error.message}

 [ACTIONABLE REMEDIATION]
{tips_str}
====================================================================================================
"""
        sys.stderr.write(box + "\n")
        sys.stderr.flush()
        return PipelineResult(
            status="FAILED",
            stage_failed=error.stage_name,
            error_message=error.message,
            error_remediation="\n".join(error.remediation_tips),
            execution_time_seconds=elapsed,
        )

    def _handle_unexpected_failure(self, error: Exception, elapsed: float) -> PipelineResult:
        box = f"""
====================================================================================================
 🛑 [CHIMERA PIPELINE UNEXPECTED FAILURE]
====================================================================================================
 Exception:          {type(error).__name__}: {str(error)}

 [ACTIONABLE REMEDIATION]
 1. Review the stack trace above for the exact source line.
 2. Ensure all input PDB files exist and contain valid atomic coordinate records.
====================================================================================================
"""
        sys.stderr.write(box + "\n")
        sys.stderr.flush()
        return PipelineResult(
            status="FAILED",
            stage_failed="Unknown",
            error_message=str(error),
            execution_time_seconds=elapsed,
        )


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "measure_junctions":
        meas_parser = argparse.ArgumentParser(description="Measure 3D junction distances and recommend linker bounds.")
        meas_parser.add_argument("cmd", choices=["measure_junctions"], help="Subcommand")
        meas_parser.add_argument("--input_file", "--input", "-i", dest="input_file", help="Input PDB or PSE file")
        meas_parser.add_argument("--donor_pdb", help="Standalone donor PDB")
        meas_parser.add_argument("--vessel_pdb", help="Standalone vessel PDB")
        meas_parser.add_argument("--donor_id", default="A", help="Donor chain ID or object name [Default: A]")
        meas_parser.add_argument("--vessel_id", default="B", help="Vessel chain ID or object name [Default: B]")
        meas_parser.add_argument("--motif_residues", "-m", default="all", help="Donor motif residues [Default: all]")
        meas_parser.add_argument(
            "--chimera_mode",
            "--mode",
            dest="chimera_mode",
            default="replace",
            choices=["replace", "ligate_nterm", "ligate_cterm", "insert_loop"],
            help="Chimera mode",
        )
        meas_parser.add_argument(
            "--vessel_replace_range",
            "--replace_range",
            dest="vessel_replace_range",
            help="Residue range to excise on vessel",
        )
        meas_parser.add_argument(
            "--vessel_split_after",
            "--split_after",
            dest="vessel_split_after",
            type=int,
            help="Split residue for insert_loop",
        )
        meas_parser.add_argument("--out_dir", "-o", default="outputs/chimera_scaffold", help="Output directory")

        meas_args = meas_parser.parse_args()

        prep = prepare_input_structures(
            input_file=meas_args.input_file,
            donor_pdb=meas_args.donor_pdb,
            vessel_pdb=meas_args.vessel_pdb,
            donor_id=meas_args.donor_id,
            vessel_id=meas_args.vessel_id,
            out_dir=meas_args.out_dir,
        )
        res = measure_junctions(
            working_donor_pdb=prep["working_donor_pdb"],
            working_vessel_pdb=prep["working_vessel_pdb"],
            donor_chain_id=prep["donor_chain_id"],
            vessel_chain_id=prep["vessel_chain_id"],
            motif_residues=meas_args.motif_residues,
            chimera_mode=meas_args.chimera_mode,
            vessel_metadata=prep["vessel_metadata"],
            donor_metadata=prep["donor_metadata"],
            vessel_replace_range=meas_args.vessel_replace_range,
            vessel_split_after=meas_args.vessel_split_after,
        )
        print(json.dumps(res, indent=2))
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Run Chimera Scaffolding Pipeline: Modular multi-stage molecular engineering."
    )
    parser.add_argument(
        "--input_file", "--input", "-i", dest="input_file", help="Input multi-chain PDB or PyMOL session (.pse) file"
    )
    parser.add_argument("--donor_pdb", help="Path to standalone donor PDB")
    parser.add_argument("--vessel_pdb", help="Path to standalone vessel PDB")
    parser.add_argument("--donor_id", default="A", help="Donor chain ID or object name [Default: A]")
    parser.add_argument("--vessel_id", default="B", help="Vessel chain ID or object name [Default: B]")
    parser.add_argument("--donor_chain", help="Alias for donor chain ID(s)")
    parser.add_argument("--motif_residues", "-m", help="Donor motif residue range(s)")
    parser.add_argument("--motifs", help="Multi-chain or multi-PDB motif specification")
    parser.add_argument("--donor_range", help="Alias for motif residue range or multi-chain spec")
    parser.add_argument(
        "--chimera_mode",
        "--mode",
        dest="chimera_mode",
        choices=["replace", "ligate_nterm", "ligate_cterm", "insert_loop"],
        default="replace",
        help="Chimera architecture mode",
    )
    parser.add_argument(
        "--vessel_replace_range",
        "--replace_range",
        dest="vessel_replace_range",
        help="Residue range to excise on vessel",
    )
    parser.add_argument(
        "--vessel_split_after",
        "--split_after",
        dest="vessel_split_after",
        type=int,
        help="Residue after which to split vessel for loop insertion",
    )
    parser.add_argument("--linker_n", default="6-10", help="N-terminal linker length or range")
    parser.add_argument("--linker_c", default="6-10", help="C-terminal linker length or range")
    parser.add_argument(
        "--lock_vessel",
        action="store_true",
        default=True,
        help="Lock vessel framework residue positions in ProteinMPNN",
    )
    parser.add_argument(
        "--no_lock_vessel",
        dest="lock_vessel",
        action="store_false",
        help="Allow ProteinMPNN to redesign vessel framework residues",
    )
    parser.add_argument(
        "--vessel_redesign_mode",
        choices=["linkers_only", "vessel_full", "vessel_interface"],
        default="linkers_only",
        help="Vessel redesign flexibility mode for ProteinMPNN",
    )
    parser.add_argument(
        "--clamp_linkers",
        dest="clamp_linkers",
        action="store_true",
        default=True,
        help="Biophysically clamp impossible linkers",
    )
    parser.add_argument(
        "--no_clamp_linkers",
        dest="clamp_linkers",
        action="store_false",
        help="Disable biophysical clamping of linkers",
    )
    parser.add_argument("--num_designs", type=int, default=10, help="Number of RFdiffusion backbones")
    parser.add_argument(
        "--mpnn_seqs_per_bb", type=int, default=2, help="ProteinMPNN sequences per backbone"
    )
    parser.add_argument("--mpnn_temp", type=float, default=0.1, help="ProteinMPNN temperature")
    parser.add_argument(
        "--mpnn_batch_size", type=int, default=1, help="Parallel GPU batch size for ProteinMPNN"
    )
    parser.add_argument("--rmsd_threshold", type=float, default=1.0, help="Kabsch RMSD cutoff in Å")
    parser.add_argument("--plddt_threshold", type=float, default=75.0, help="pLDDT quality cutoff")
    parser.add_argument("--rfdiff_script", help="Path to RFdiffusion run_inference.py")
    parser.add_argument("--mpnn_script", help="Path to ProteinMPNN protein_mpnn_run.py")
    parser.add_argument("--python_rfdiff", help="Python interpreter binary for RFdiffusion")
    parser.add_argument("--python_mpnn", help="Python interpreter binary for ProteinMPNN")
    parser.add_argument("--python_colabfold", help="Python interpreter binary or CLI for ColabFold")
    parser.add_argument(
        "--strict_python",
        action="store_true",
        default=False,
        help="Enforce strict resolution of Python interpreters without searching alternative fallback environments",
    )
    parser.add_argument("--out_dir", "-o", default="outputs/chimera_scaffold", help="Output directory")
    parser.add_argument("--dry_run", action="store_true", help="Prepare contigs, seed, and masks without GPU inference")

    args = parser.parse_args()

    motif_val = args.motifs or args.donor_range or args.motif_residues
    if not motif_val:
        parser.error("One of --motifs, --motif_residues, or --donor_range is required.")
    donor_id_val = args.donor_chain or args.donor_id

    ctx = PipelineContext(
        input_file=args.input_file,
        donor_pdb=args.donor_pdb,
        vessel_pdb=args.vessel_pdb,
        donor_id=donor_id_val,
        vessel_id=args.vessel_id,
        motif_residues=motif_val,
        motifs=args.motifs,
        chimera_mode=args.chimera_mode,
        vessel_replace_range=args.vessel_replace_range,
        vessel_split_after=args.vessel_split_after,
        linker_n=args.linker_n,
        linker_c=args.linker_c,
        lock_vessel=args.lock_vessel,
        vessel_redesign_mode=args.vessel_redesign_mode,
        clamp_linkers=args.clamp_linkers,
        num_designs=args.num_designs,
        mpnn_seqs_per_bb=args.mpnn_seqs_per_bb,
        mpnn_temp=args.mpnn_temp,
        mpnn_batch_size=args.mpnn_batch_size,
        rmsd_threshold=args.rmsd_threshold,
        plddt_threshold=args.plddt_threshold,
        rfdiff_script=args.rfdiff_script,
        mpnn_script=args.mpnn_script,
        python_rfdiff=args.python_rfdiff,
        python_mpnn=args.python_mpnn,
        python_colabfold=args.python_colabfold,
        strict_python=args.strict_python,
        out_dir=args.out_dir,
        dry_run=args.dry_run,
    )

    orchestrator = ChimeraPipelineOrchestrator(ctx)
    result = orchestrator.run()
    if result.status == "SUCCESS":
        print(json.dumps(result.summary_data, indent=2))
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
