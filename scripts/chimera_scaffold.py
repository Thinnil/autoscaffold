#!/usr/bin/env python3
"""
scripts/chimera_scaffold.py
Modular backward-compatible facade for the Chimera Scaffolding workflow.

Preserves the exact tertiary 3D structure and amino acid sequence of a designated
active motif/epitope ("Donor") while grafting or ligating it into a host scaffold
protein ("Recipient / Vessel").

This facade module re-exports all legacy dataclasses, session caches, functions,
constants, regexes, and the ChimeraScaffolder class adapter while delegating to the
modular chimera_stages package.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Add script directory to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# 1. Stage 1 Imports & Re-exports
from chimera_stages.stage1_parsing import (
    ONE_TO_THREE,
    _RE_SANITIZE_IDENT,
    _RE_OUTPUTS_DEDUP_MID,
    _RE_RESIDUE_RANGE,
    _RE_NUM_OR_SPAN,
    _RE_HYPHEN_RANGE,
    _RE_LINKER_VAL,
    _RE_SHELL_INJECTION,
    MotifAtom,
    MotifResidue,
    MotifSegment,
    MultiChainMotifManifest,
    _CA_COORDS_CACHE,
    _FALLBACK_COORDS_CACHE,
    _RESI_COORDS_CACHE,
    _ANCHOR_COORDS_CACHE,
    _ANCHOR_FALLBACK_CACHE,
    _ANCHOR_RESI_CACHE,
    _ANCHOR_RESI_FALLBACK_CACHE,
    _SEED_CA_CACHE,
    _REALPATH_CACHE,
    clear_ca_coords_cache,
    parse_multi_chain_motif_spec,
    build_multi_chain_manifest,
    get_residue_anchor_coords,
    get_residue_ca_coords,
    prepare_input_structures as _stage1_prepare_input_structures,
)

# 2. Stage 2 Imports & Re-exports
from chimera_stages.stage2_clash import (
    calculate_anchor_distance,
    batch_anchor_distances,
    _transform_pdb_coordinates,
    audit_motif_cartesian_drift,
    map_fixed_positions_for_mpnn as _stage2_map_fixed_positions,
    _purge_candidate_artifacts,
)

# 3. Stage 3 Imports & Re-exports
from chimera_stages.stage3_geometry import (
    compute_gap_linker_length,
    compute_gap_linker_string,
    compute_recommended_linker,
    compute_extension_minimum,
    check_and_clamp_linker,
    format_segments_to_contig,
    parse_motif_ranges as _stage3_parse_motif_ranges,
    parse_vessel_intervals as _stage3_parse_vessel_intervals,
    slice_intervals,
    measure_junctions as _stage3_measure_junctions,
    compute_contigs_and_segments as _stage3_compute_contigs_and_segments,
)

# 4. Stage 4 Imports & Re-exports
from chimera_stages.stage4_runners import (
    RunnerSpec,
    clear_probe_cache,
    clear_environment_cache,
    find_environment_managers,
    get_all_environment_roots,
    build_ephemeral_runner_cmd,
    format_runner_display,
    display_dependency_alert,
    display_cuda_warning,
    probe_python_environment,
    resolve_execution_runner,
    resolve_python_interpreter,
    resolve_neural_python,
    run_closed_loop_backbone_generation,
    _PROBE_CACHE,
    _CACHED_MANAGERS,
    _CACHED_ROOTS,
)

# 5. Stage 5 Imports & Re-exports
from chimera_stages.stage5_export import (
    create_merged_seed_pdb as _stage5_create_merged_seed_pdb,
    place_atom_nerf,
    build_sidechain_atoms,
    compute_cb_coords,
    extract_mpnn_sequences,
    extract_native_motif_sequence,
    thread_sequence_onto_pdb,
    export_pipeline_outputs,
)

try:
    from scripts.pse_pdb_extractor import (
        validate_output_dir,
        safe_join_path,
    )
    from scripts.format_complex_fasta import make_fixed_positions_dict
except ImportError:
    from pse_pdb_extractor import (
        validate_output_dir,
        safe_join_path,
    )
    from format_complex_fasta import make_fixed_positions_dict


# Dynamic module attribute lookup for Trap 2 (_CACHED_MANAGERS, _CACHED_ROOTS)
def __getattr__(name: str) -> Any:
    if name in ("_CACHED_MANAGERS", "_CACHED_ROOTS"):
        import chimera_stages.stage4_runners as s4
        return getattr(s4, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


class ChimeraScaffolder:
    """
    Production orchestrator facade for Chimera Scaffolding.
    Maintains 100% backward compatibility with legacy attributes, state mutations,
    and method signatures.
    """

    def __init__(
        self,
        input_file: Optional[str] = None,
        donor_pdb: Optional[str] = None,
        vessel_pdb: Optional[str] = None,
        donor_id: Any = "A",
        vessel_id: Any = "B",
        motif_residues: str = "",
        motifs: Optional[str] = None,
        chimera_mode: str = "replace",
        vessel_replace_range: Optional[str] = None,
        vessel_split_after: Optional[int] = None,
        linker_n: str = "6-10",
        linker_c: str = "6-10",
        num_designs: int = 10,
        mpnn_seqs_per_bb: int = 2,
        mpnn_temp: float = 0.1,
        mpnn_batch_size: int = 1,
        rmsd_threshold: float = 1.0,
        plddt_threshold: float = 75.0,
        lock_vessel: bool = True,
        vessel_redesign_mode: str = "linkers_only",
        clamp_linkers: bool = True,
        out_dir: str = "outputs/chimera_scaffold",
        dry_run: bool = False,
        rfdiff_script: Optional[str] = None,
        mpnn_script: Optional[str] = None,
        colabfold_cmd: Optional[str] = None,
        python_rfdiff: Optional[Union[str, List[str], RunnerSpec]] = None,
        python_mpnn: Optional[Union[str, List[str], RunnerSpec]] = None,
        python_colabfold: Optional[Union[str, List[str], RunnerSpec]] = None,
        strict_python: bool = False,
    ):
        self.input_file = os.path.realpath(input_file) if input_file else None
        self.donor_pdb = os.path.realpath(donor_pdb) if donor_pdb else None
        self.vessel_pdb = os.path.realpath(vessel_pdb) if vessel_pdb else None
        path_checks = [
            ("input_file", self.input_file),
            ("donor_pdb", self.donor_pdb),
            ("vessel_pdb", self.vessel_pdb),
        ]
        for p_label, p_val in path_checks:
            if p_val:
                p_drive, p_rest = os.path.splitdrive(p_val)
                is_root_path = (
                    (not p_drive and p_val.rstrip("/\\") == "")
                    or (p_drive and p_rest in ("/", "\\", ""))
                    or p_val in ("/", "\\", "//", "\\\\")
                )
                if is_root_path:
                    raise ValueError(f"Dangerous or root file specified for {p_label}: '{p_val}'")

        for id_label, id_val in [("donor_id", donor_id), ("vessel_id", vessel_id)]:
            if isinstance(id_val, str):
                if ".." in id_val or "/" in id_val or "\\" in id_val:
                    raise ValueError(f"Path traversal detected in {id_label}: '{id_val}'")
            elif isinstance(id_val, (list, tuple)):
                for item in id_val:
                    if ".." in str(item) or "/" in str(item) or "\\" in str(item):
                        raise ValueError(f"Path traversal detected in {id_label}: '{item}'")

        self.donor_id = donor_id
        self.vessel_id = vessel_id
        self.motifs = motifs
        raw_motif = motifs if motifs is not None else motif_residues
        self.motif_residues = raw_motif
        self.is_multichain = (
            bool(motifs)
            or (":" in str(raw_motif))
            or (isinstance(donor_id, (list, tuple)) and len(donor_id) > 1)
            or (isinstance(donor_id, str) and "," in donor_id)
        )
        self.manifest: Optional[MultiChainMotifManifest] = None
        self.seed_provenance: Dict[int, Dict[str, Any]] = {}
        self.chimera_mode = chimera_mode.lower()
        self.vessel_replace_range = vessel_replace_range
        self.vessel_split_after = vessel_split_after
        self.linker_n = linker_n
        self.linker_c = linker_c
        self.num_designs = num_designs
        self.mpnn_seqs_per_bb = mpnn_seqs_per_bb
        self.mpnn_temp = mpnn_temp
        self.mpnn_batch_size = mpnn_batch_size
        self.rmsd_threshold = rmsd_threshold
        self.plddt_threshold = plddt_threshold
        v_mode = str(vessel_redesign_mode).lower().strip()
        if v_mode in ("vessel_full", "full", "false") or not lock_vessel:
            self.vessel_redesign_mode = "vessel_full"
            self.lock_vessel = False
        elif v_mode in ("vessel_interface", "interface"):
            self.vessel_redesign_mode = "vessel_interface"
            self.lock_vessel = True
        else:
            self.vessel_redesign_mode = "linkers_only"
            self.lock_vessel = True
        self.clamp_linkers = clamp_linkers
        self.out_dir = validate_output_dir(out_dir)
        if "outputs" in self.out_dir.lower():
            self.out_dir = _RE_OUTPUTS_DEDUP_MID.sub(r"\1", self.out_dir)
        self.dry_run = dry_run
        self.strict_python = strict_python

        # Dynamic runner resolution respecting mocked functions
        cs_mod = sys.modules.get("scripts.chimera_scaffold") or sys.modules.get("chimera_scaffold")
        resolve_fn = (
            getattr(cs_mod, "resolve_execution_runner", resolve_execution_runner)
            if cs_mod
            else resolve_execution_runner
        )

        self.rfdiff_runner = resolve_fn(
            explicit_target=python_rfdiff,
            env_var_keys=["PYTHON_RFDIFF", "RFDIFF_PYTHON"],
            target_env_names=["SE3nv", "rfdiffusion", "rf_diffusion", "se3nv"],
            required_modules=["torch", "hydra"],
            tool_label="RFdiffusion",
            strict=strict_python,
        )
        self.python_rfdiff = (
            self.rfdiff_runner.python_binary
            if self.rfdiff_runner.runner_type == "direct"
            else self.rfdiff_runner.command_prefix
        )
        self.rfdiff_env_valid = self.rfdiff_runner.is_valid

        self.mpnn_runner = resolve_fn(
            explicit_target=python_mpnn,
            env_var_keys=["PYTHON_MPNN", "MPNN_PYTHON"],
            target_env_names=["mlfold", "SE3nv", "mpnn", "proteinmpnn"],
            required_modules=["torch"],
            tool_label="ProteinMPNN",
            strict=strict_python,
        )
        if not self.mpnn_runner.is_valid and not strict_python:
            if self.rfdiff_runner.is_valid:
                probe_fn = (
                    getattr(cs_mod, "probe_python_environment", probe_python_environment)
                    if cs_mod
                    else probe_python_environment
                )
                rfdiff_probe = probe_fn(
                    self.rfdiff_runner, required_modules=["torch"], check_cuda=False, raise_on_error=False
                )
                if len(rfdiff_probe.get("missing_modules", [])) == 0:
                    self.mpnn_runner = self.rfdiff_runner
                    self.python_mpnn = (
                        self.mpnn_runner.python_binary
                        if self.mpnn_runner.runner_type == "direct"
                        else self.mpnn_runner.command_prefix
                    )
                    self.mpnn_env_valid = True
            if not self.mpnn_runner.is_valid:
                mpnn_fallback = resolve_fn(
                    explicit_target=None,
                    env_var_keys=["PYTHON_MPNN", "MPNN_PYTHON"],
                    target_env_names=["SE3nv", "mpnn", "proteinmpnn", "rfdiffusion"],
                    required_modules=["torch"],
                    tool_label="ProteinMPNN (Fallback)",
                    strict=False,
                )
                if mpnn_fallback.is_valid:
                    self.mpnn_runner = mpnn_fallback
                    self.python_mpnn = (
                        self.mpnn_runner.python_binary
                        if self.mpnn_runner.runner_type == "direct"
                        else self.mpnn_runner.command_prefix
                    )
                    self.mpnn_env_valid = True

        self.python_mpnn = (
            self.mpnn_runner.python_binary
            if self.mpnn_runner.runner_type == "direct"
            else self.mpnn_runner.command_prefix
        )
        self.mpnn_env_valid = self.mpnn_runner.is_valid

        self.colabfold_runner = resolve_fn(
            explicit_target=python_colabfold,
            env_var_keys=["PYTHON_COLABFOLD", "COLABFOLD_PYTHON"],
            target_env_names=["colabfold", "colabfold_batch", "af2"],
            required_modules=[],
            tool_label="ColabFold",
            strict=strict_python,
        )
        self.python_colabfold = (
            self.colabfold_runner.python_binary
            if self.colabfold_runner.runner_type == "direct"
            else self.colabfold_runner.command_prefix
        )
        self.colabfold_env_valid = self.colabfold_runner.is_valid

        self.rfdiff_script = os.path.realpath(rfdiff_script) if rfdiff_script else None
        if not self.rfdiff_script:
            for d in [
                os.environ.get("RFDIFF_DIR", ""),
                os.path.expanduser("~/programs/RFdiffusion"),
                os.path.join(SCRIPT_DIR, "..", "tools", "RFdiffusion"),
                os.path.join(SCRIPT_DIR, "tools", "RFdiffusion"),
                os.path.join(SCRIPT_DIR, "..", "RFdiffusion"),
            ]:
                cand = os.path.join(d, "scripts", "run_inference.py") if d else ""
                if cand and os.path.isfile(cand):
                    self.rfdiff_script = os.path.realpath(cand)
                    break

        self.mpnn_script = os.path.realpath(mpnn_script) if mpnn_script else None
        if not self.mpnn_script:
            for d in [
                os.environ.get("MPNN_DIR", ""),
                os.path.expanduser("~/programs/ProteinMPNN"),
                os.path.join(SCRIPT_DIR, "..", "tools", "ProteinMPNN"),
                os.path.join(SCRIPT_DIR, "tools", "ProteinMPNN"),
                os.path.join(SCRIPT_DIR, "..", "ProteinMPNN"),
            ]:
                cand = os.path.join(d, "protein_mpnn_run.py") if d else ""
                if cand and os.path.isfile(cand):
                    self.mpnn_script = os.path.realpath(cand)
                    break

        for s_label, s_val in [("rfdiff_script", self.rfdiff_script), ("mpnn_script", self.mpnn_script)]:
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

        self.colabfold_cmd = colabfold_cmd

        # Internal state
        self.working_donor_pdb = ""
        self.working_vessel_pdb = ""
        self.donor_chain_id = "A"
        self.vessel_chain_id = "B"
        self.donor_metadata: Dict[str, Any] = {}
        self.vessel_metadata: Dict[str, Any] = {}
        self.seed_pdb_path = ""
        self.seed_pdb = ""
        self.contig_string = ""
        self.preserved_donor_segments: List[Tuple[int, int]] = []
        self.preserved_vessel_segments: List[Tuple[int, int]] = []
        self.seed_chain_id = "A"
        self.donor_seed_offset = 0
        self.last_donor_fixed: List[int] = []

    def prepare_input_structures(self):
        """Extracts or validates donor and vessel PDB files."""
        res = _stage1_prepare_input_structures(
            input_file=self.input_file,
            donor_pdb=self.donor_pdb,
            vessel_pdb=self.vessel_pdb,
            donor_id=self.donor_id,
            vessel_id=self.vessel_id,
            out_dir=self.out_dir,
            is_multichain=getattr(self, "is_multichain", False),
        )
        self.working_donor_pdb = res["working_donor_pdb"]
        self.working_vessel_pdb = res["working_vessel_pdb"]
        self.donor_metadata = res["donor_metadata"]
        self.vessel_metadata = res["vessel_metadata"]
        self.donor_chain_id = res["donor_chain_id"]
        self.vessel_chain_id = res["vessel_chain_id"]

    def parse_motif_ranges(self) -> List[Tuple[int, int]]:
        """Parses motif residue intervals and returns sorted contiguous segments."""
        if not self.working_donor_pdb:
            self.prepare_input_structures()
        intervals = _stage3_parse_motif_ranges(
            motif_residues=self.motif_residues,
            donor_metadata=self.donor_metadata,
            working_donor_pdb=self.working_donor_pdb,
            donor_chain_id=self.donor_chain_id,
        )
        self.preserved_donor_segments = intervals
        return intervals

    def parse_vessel_intervals(self) -> List[Tuple[int, int]]:
        """Parses vessel residue intervals."""
        if not self.working_vessel_pdb:
            self.prepare_input_structures()
        return _stage3_parse_vessel_intervals(self.vessel_metadata)

    @staticmethod
    def slice_intervals(
        intervals: List[Tuple[int, int]], keep_min: Optional[int], keep_max: Optional[int]
    ) -> List[Tuple[int, int]]:
        return slice_intervals(intervals, keep_min, keep_max)

    @staticmethod
    def format_segments_to_contig(segments: List[Tuple[int, int]], chain_label: str) -> str:
        return format_segments_to_contig(segments, chain_label)

    @staticmethod
    def check_and_clamp_linker(
        user_val: str, param_name: str, distance_angstrom: Optional[float], junction_label: str
    ) -> str:
        return check_and_clamp_linker(user_val, param_name, distance_angstrom, junction_label)

    def measure_junctions(self) -> Dict[str, Any]:
        """Calculates exact 3D Euclidean distances and recommended linker lengths."""
        if not self.working_donor_pdb or not self.working_vessel_pdb:
            self.prepare_input_structures()
        return _stage3_measure_junctions(
            working_donor_pdb=self.working_donor_pdb,
            working_vessel_pdb=self.working_vessel_pdb,
            donor_chain_id=self.donor_chain_id,
            vessel_chain_id=self.vessel_chain_id,
            motif_residues=self.motif_residues,
            chimera_mode=self.chimera_mode,
            vessel_metadata=self.vessel_metadata,
            donor_metadata=self.donor_metadata,
            vessel_replace_range=self.vessel_replace_range,
            vessel_split_after=self.vessel_split_after,
        )

    measure_junction_distances = measure_junctions

    def compute_contigs_and_segments(self) -> Tuple[str, List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Computes RFdiffusion contigs and list of preserved donor & vessel segments."""
        if not self.working_donor_pdb or not self.working_vessel_pdb:
            self.prepare_input_structures()

        (
            contig,
            donor_segs,
            vessel_segs,
            offset,
            clamped_ln,
            clamped_lc,
            manifest,
        ) = _stage3_compute_contigs_and_segments(
            working_donor_pdb=self.working_donor_pdb,
            working_vessel_pdb=self.working_vessel_pdb,
            donor_chain_id=self.donor_chain_id,
            vessel_chain_id=self.vessel_chain_id,
            motif_residues=self.motif_residues,
            chimera_mode=self.chimera_mode,
            vessel_metadata=self.vessel_metadata,
            donor_metadata=self.donor_metadata,
            linker_n=self.linker_n,
            linker_c=self.linker_c,
            vessel_replace_range=self.vessel_replace_range,
            vessel_split_after=self.vessel_split_after,
            clamp_linkers=self.clamp_linkers,
            seed_chain_id=self.seed_chain_id,
            is_multichain=getattr(self, "is_multichain", False),
        )

        self.contig_string = contig
        self.preserved_donor_segments = donor_segs
        self.preserved_vessel_segments = vessel_segs
        self.donor_seed_offset = offset
        self.linker_n = clamped_ln
        self.linker_c = clamped_lc
        if manifest:
            self.manifest = manifest
            self.seed_provenance = manifest.provenance_map

        return self.contig_string, self.preserved_donor_segments, self.preserved_vessel_segments

    def create_merged_seed_pdb(self, buffer_size: int = 65536) -> str:
        """Merges preserved donor motif atoms and preserved vessel atoms into chimera_seed.pdb."""
        seed_path, prov = _stage5_create_merged_seed_pdb(
            out_dir=self.out_dir,
            working_donor_pdb=self.working_donor_pdb,
            working_vessel_pdb=self.working_vessel_pdb,
            donor_chain_id=self.donor_chain_id,
            vessel_chain_id=self.vessel_chain_id,
            preserved_donor_segments=self.preserved_donor_segments,
            preserved_vessel_segments=self.preserved_vessel_segments,
            is_multichain=getattr(self, "is_multichain", False),
            manifest=getattr(self, "manifest", None),
            seed_provenance=getattr(self, "seed_provenance", None),
            seed_chain_id=getattr(self, "seed_chain_id", "A"),
            donor_seed_offset=getattr(self, "donor_seed_offset", 0),
            buffer_size=buffer_size,
        )
        self.seed_pdb_path = seed_path
        self.seed_pdb = seed_path
        self.seed_provenance = prov
        return seed_path

    def map_fixed_positions_for_mpnn(self, backbone_pdb: str) -> Dict[str, List[int]]:
        """Parses generated backbone PDB and generates ProteinMPNN fixed positions mask."""
        fixed_dict, donor_fixed = _stage2_map_fixed_positions(
            backbone_pdb=backbone_pdb,
            preserved_donor_segments=self.preserved_donor_segments,
            preserved_vessel_segments=self.preserved_vessel_segments,
            working_donor_pdb=self.working_donor_pdb,
            working_vessel_pdb=self.working_vessel_pdb,
            seed_pdb_path=getattr(self, "seed_pdb_path", None) or getattr(self, "seed_pdb", None),
            donor_chain_id=self.donor_chain_id,
            vessel_chain_id=self.vessel_chain_id,
            lock_vessel=self.lock_vessel,
            vessel_redesign_mode=self.vessel_redesign_mode,
            is_multichain=getattr(self, "is_multichain", False),
            chimera_mode=self.chimera_mode,
        )
        self.last_donor_fixed = donor_fixed
        return fixed_dict

    def run_pipeline(self) -> Dict[str, Any]:
        """Executes the entire chimera pipeline, updating instance state in-place."""
        self.prepare_input_structures()
        self.compute_contigs_and_segments()
        self.create_merged_seed_pdb()

        work_rfdiff = safe_join_path(self.out_dir, "1_rfdiffusion")
        work_mpnn = safe_join_path(self.out_dir, "2_mpnn")
        work_results = safe_join_path(self.out_dir, "results")
        os.makedirs(work_rfdiff, exist_ok=True)
        os.makedirs(work_mpnn, exist_ok=True)
        os.makedirs(work_results, exist_ok=True)

        summary_data: Dict[str, Any] = {
            "status": "INITIALIZED",
            "donor_source": self.working_donor_pdb,
            "vessel_source": self.working_vessel_pdb,
            "donor_chain": self.donor_chain_id,
            "vessel_chain": self.vessel_chain_id,
            "motif_residues": self.motif_residues,
            "chimera_mode": self.chimera_mode,
            "contig_formula": self.contig_string,
            "seed_pdb": self.seed_pdb_path,
            "num_designs": self.num_designs,
            "dry_run": self.dry_run,
            "candidates": [],
        }

        work_prep = safe_join_path(self.out_dir, "0_prep")
        contig_plan_path = safe_join_path(work_prep, "contig_plan.txt")
        with open(contig_plan_path, "w", encoding="utf-8", buffering=65536) as cpf:
            cpf.write(f"Contig Formula: {self.contig_string}\n")
            cpf.write(f"Chimera Mode: {self.chimera_mode}\n")
            cpf.write(f"Linker N: {self.linker_n}\n")
            cpf.write(f"Linker C: {self.linker_c}\n")
            cpf.write(f"Preserved Donor Segments: {self.preserved_donor_segments}\n")
            cpf.write(f"Preserved Vessel Segments: {self.preserved_vessel_segments}\n")
        summary_data["contig_plan"] = contig_plan_path

        if isinstance(self.python_rfdiff, RunnerSpec):
            self.rfdiff_runner = self.python_rfdiff
        elif isinstance(self.python_rfdiff, (list, tuple)):
            self.rfdiff_runner = RunnerSpec(
                runner_type="custom",
                command_prefix=list(self.python_rfdiff),
                python_binary=self.python_rfdiff[0] if self.python_rfdiff else None,
                is_valid=True,
            )
        elif isinstance(self.python_rfdiff, str) and (
            not getattr(self, "rfdiff_runner", None) or self.python_rfdiff != self.rfdiff_runner.python_binary
        ):
            self.rfdiff_runner = RunnerSpec(
                runner_type="direct",
                command_prefix=[self.python_rfdiff],
                python_binary=self.python_rfdiff,
                is_valid=True,
            )

        if not self.dry_run:
            cs_mod = sys.modules.get("scripts.chimera_scaffold") or sys.modules.get("chimera_scaffold")
            probe_fn = (
                getattr(cs_mod, "probe_python_environment", probe_python_environment)
                if cs_mod
                else probe_python_environment
            )
            probe_res = probe_fn(
                self.rfdiff_runner,
                required_modules=["torch", "hydra"],
                check_cuda=True,
                raise_on_error=True,
                tool_name="RFdiffusion",
                suggested_env="SE3nv",
            )
            if not probe_res.get("cuda_available", False):
                display_cuda_warning(self.rfdiff_runner.display_label(), tool_name="RFdiffusion")

        rfdiff_cmd = self.rfdiff_runner.build_command(
            self.rfdiff_script or "scripts/run_inference.py",
            f"inference.output_prefix={os.path.join(work_rfdiff, 'chimera_design')}",
            f"inference.input_pdb={self.seed_pdb_path}",
            f"contigmap.contigs=[{self.contig_string}]",
            f"inference.num_designs={self.num_designs}",
            "denoiser.noise_scale_ca=0.0",
            "denoiser.noise_scale_frame=0.0",
        )
        summary_data["rfdiffusion_command"] = rfdiff_cmd

        generated_backbones = []
        if not self.dry_run:
            if not self.rfdiff_script or not os.path.isfile(self.rfdiff_script):
                raise FileNotFoundError(
                    f"RFdiffusion script not found. Looked in: '{self.rfdiff_script}'. "
                    "Please configure RFDIFF_DIR in config.env or run: autoscaffold install"
                )
            generated_backbones = run_closed_loop_backbone_generation(
                seed_pdb=self.seed_pdb_path,
                contig_str=self.contig_string,
                seed_provenance=self.seed_provenance,
                target_passing_designs=self.num_designs,
                max_attempts=max(50, self.num_designs * 5),
                max_drift_cutoff=0.8,
                mean_drift_cutoff=0.5,
                rfdiff_runner=self.rfdiff_runner,
                rfdiff_script=self.rfdiff_script,
                work_dir=work_rfdiff,
            )
        else:
            mock_bb = safe_join_path(work_rfdiff, "chimera_design_0.pdb")
            with open(self.seed_pdb_path, "r", encoding="utf-8", buffering=65536) as sf, open(
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
            generated_backbones.append(mock_bb)

        if isinstance(self.python_mpnn, RunnerSpec):
            self.mpnn_runner = self.python_mpnn
        elif isinstance(self.python_mpnn, (list, tuple)):
            self.mpnn_runner = RunnerSpec(
                runner_type="custom",
                command_prefix=list(self.python_mpnn),
                python_binary=self.python_mpnn[0] if self.python_mpnn else None,
                is_valid=True,
            )
        elif isinstance(self.python_mpnn, str) and (
            not getattr(self, "mpnn_runner", None) or self.python_mpnn != self.mpnn_runner.python_binary
        ):
            self.mpnn_runner = RunnerSpec(
                runner_type="direct",
                command_prefix=[self.python_mpnn],
                python_binary=self.python_mpnn,
                is_valid=True,
            )

        if not self.dry_run and self.mpnn_script and os.path.isfile(self.mpnn_script):
            cs_mod = sys.modules.get("scripts.chimera_scaffold") or sys.modules.get("chimera_scaffold")
            probe_fn = (
                getattr(cs_mod, "probe_python_environment", probe_python_environment)
                if cs_mod
                else probe_python_environment
            )
            probe_fn(
                self.mpnn_runner,
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
            fixed_map = self.map_fixed_positions_for_mpnn(bb)
            donor_fixed_matched = getattr(self, "last_donor_fixed", [])
            jsonl_path = safe_join_path(work_mpnn, f"{safe_bb_name}_fixed_positions.jsonl")
            make_fixed_positions_dict(bb_name, fixed_map, out_jsonl=jsonl_path)

            mpnn_cmd = self.mpnn_runner.build_command(
                self.mpnn_script or "protein_mpnn_run.py",
                "--pdb_path",
                bb,
                "--fixed_positions_jsonl",
                jsonl_path,
                "--out_folder",
                work_mpnn,
                "--num_seq_per_target",
                str(self.mpnn_seqs_per_bb),
                "--sampling_temp",
                str(self.mpnn_temp),
                "--batch_size",
                str(self.mpnn_batch_size),
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

            if not self.dry_run and self.mpnn_script and os.path.isfile(self.mpnn_script):
                mpnn_repo = os.path.dirname(os.path.abspath(self.mpnn_script))
                cwd_dir = mpnn_repo if os.path.isdir(mpnn_repo) else None
                subprocess.run(mpnn_cmd, check=True, shell=False, cwd=cwd_dir)

            mpnn_results.append(cand_info)

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
                    seed_source = getattr(self, "seed_pdb_path", None) or getattr(self, "seed_pdb", None)
                    d_fixed_list = cand.get("donor_fixed", [])
                    if thread_sequence_onto_pdb(
                        bb, seq, th_pdb, seed_pdb=seed_source, fixed_donor_positions=d_fixed_list
                    ):
                        cand["threaded_pdbs"].append(th_pdb)

        summary_data["candidates"] = mpnn_results

        export_summary = export_pipeline_outputs(
            out_dir=self.out_dir,
            mpnn_results=mpnn_results,
            working_donor_pdb=self.working_donor_pdb,
            working_vessel_pdb=self.working_vessel_pdb,
            donor_chain_id=self.donor_chain_id,
            vessel_chain_id=self.vessel_chain_id,
            motif_residues=self.motif_residues,
            chimera_mode=self.chimera_mode,
            contig_string=self.contig_string,
            seed_pdb_path=self.seed_pdb_path,
            preserved_donor_segments=self.preserved_donor_segments,
            preserved_vessel_segments=self.preserved_vessel_segments,
            lock_vessel=self.lock_vessel,
            num_designs=self.num_designs,
            mpnn_seqs_per_bb=self.mpnn_seqs_per_bb,
            mpnn_temp=self.mpnn_temp,
            mpnn_batch_size=self.mpnn_batch_size,
            rmsd_threshold=self.rmsd_threshold,
            plddt_threshold=self.plddt_threshold,
            colabfold_runner=self.colabfold_runner,
            dry_run=self.dry_run,
            vessel_replace_range=self.vessel_replace_range,
            vessel_split_after=self.vessel_split_after,
            linker_n=self.linker_n,
            linker_c=self.linker_c,
            is_multichain=getattr(self, "is_multichain", False),
            seed_provenance=getattr(self, "seed_provenance", None),
        )
        summary_data.update(export_summary)
        return summary_data


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
        scaffolder = ChimeraScaffolder(
            input_file=meas_args.input_file,
            donor_pdb=meas_args.donor_pdb,
            vessel_pdb=meas_args.vessel_pdb,
            donor_id=meas_args.donor_id,
            vessel_id=meas_args.vessel_id,
            motif_residues=meas_args.motif_residues,
            chimera_mode=meas_args.chimera_mode,
            vessel_replace_range=meas_args.vessel_replace_range,
            vessel_split_after=meas_args.vessel_split_after,
            out_dir=meas_args.out_dir,
            clamp_linkers=False,
            dry_run=True,
        )
        res = scaffolder.measure_junctions()
        print(json.dumps(res, indent=2))
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Chimera Scaffolding: Graft active motifs into host scaffold proteins."
    )
    parser.add_argument(
        "--input_file", "--input", "-i", dest="input_file", help="Input multi-chain PDB or PyMOL session (.pse) file"
    )
    parser.add_argument("--donor_pdb", help="Path to standalone donor PDB")
    parser.add_argument("--vessel_pdb", help="Path to standalone vessel PDB")
    parser.add_argument("--donor_id", default="A", help="Donor chain ID or object name [Default: A]")
    parser.add_argument("--vessel_id", default="B", help="Vessel chain ID or object name [Default: B]")
    parser.add_argument("--donor_chain", help="Alias for donor chain ID(s) (e.g. 'A' or 'A,B')")
    parser.add_argument(
        "--motif_residues", "-m", help="Donor motif residue range(s) (e.g. '30-45' or '30-45,78-82')"
    )
    parser.add_argument(
        "--motifs",
        help=(
            "Multi-chain or multi-PDB motif specification "
            "(e.g. 'A:312-313, B:191, B:200' or 'f1.pdb:A:1-10, f2.pdb:B:20-30')"
        ),
    )
    parser.add_argument(
        "--donor_range",
        help="Alias for motif residue range or multi-chain spec",
    )
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
        help="Residue range to excise on vessel (e.g. '55-70')",
    )
    parser.add_argument(
        "--vessel_split_after",
        "--split_after",
        dest="vessel_split_after",
        type=int,
        help="Residue after which to split vessel for loop insertion (e.g. 54)",
    )
    parser.add_argument("--linker_n", default="6-10", help="N-terminal linker length or range [Default: 6-10]")
    parser.add_argument("--linker_c", default="6-10", help="C-terminal linker length or range [Default: 6-10]")
    parser.add_argument(
        "--lock_vessel",
        action="store_true",
        default=True,
        help="Lock vessel framework residue positions in ProteinMPNN [Default: True]",
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
        help="Vessel redesign flexibility mode for ProteinMPNN [Default: linkers_only]",
    )
    parser.add_argument(
        "--clamp_linkers",
        dest="clamp_linkers",
        action="store_true",
        default=True,
        help="Biophysically clamp impossible linkers [Default: True]",
    )
    parser.add_argument(
        "--no_clamp_linkers",
        dest="clamp_linkers",
        action="store_false",
        help="Disable biophysical clamping of linkers",
    )
    parser.add_argument("--num_designs", type=int, default=10, help="Number of RFdiffusion backbones [Default: 10]")
    parser.add_argument(
        "--mpnn_seqs_per_bb", type=int, default=2, help="ProteinMPNN sequences per backbone [Default: 2]"
    )
    parser.add_argument("--mpnn_temp", type=float, default=0.1, help="ProteinMPNN temperature [Default: 0.1]")
    parser.add_argument(
        "--mpnn_batch_size", type=int, default=1, help="Parallel GPU batch size for ProteinMPNN [Default: 1]"
    )
    parser.add_argument("--rmsd_threshold", type=float, default=1.0, help="Kabsch RMSD cutoff in Å [Default: 1.0]")
    parser.add_argument("--plddt_threshold", type=float, default=75.0, help="pLDDT quality cutoff [Default: 75.0]")
    parser.add_argument("--rfdiff_script", help="Path to RFdiffusion run_inference.py")
    parser.add_argument("--mpnn_script", help="Path to ProteinMPNN protein_mpnn_run.py")
    parser.add_argument(
        "--python_rfdiff",
        help="Path to Python interpreter binary for RFdiffusion (e.g. ~/miniconda3/envs/SE3nv/bin/python)",
    )
    parser.add_argument(
        "--python_mpnn",
        help="Path to Python interpreter binary for ProteinMPNN (e.g. ~/miniconda3/envs/mlfold/bin/python)",
    )
    parser.add_argument("--python_colabfold", help="Path to Python interpreter binary or CLI for ColabFold")
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

    scaffolder = ChimeraScaffolder(
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

    res = scaffolder.run_pipeline()
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
