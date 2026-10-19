#!/usr/bin/env python3
"""
scripts/methodology_reporter.py
Automated Methodology & Reproducibility Report Generator for AutoScaffold.

Generates comprehensive markdown/text reports capturing:
- Header with Run Name, Timestamp, AutoScaffold version, OS, Hardware (CPU, RAM, GPU, CUDA driver, active conda env).
- Pipeline Mode selected.
- Input Structures & Residue Ranges.
- Parameters Selected (contigs, sampling budget, MPNN temp, sequence count, linkers, cutoffs).
- Step-by-Step Computational Methodology (RFdiffusion SE(3) diffusion, coordinate pinning,
  ProteinMPNN fixed-position locking, AlphaFold2/ColabFold validation, Kabsch C-alpha alignment).
- Output Summary & Passing Candidate list.
- Exact Reproducibility CLI Command.

Writes both `run_methodology.txt` and `methodology.txt` in the specified output directory.
"""

import os
import platform
import datetime
import subprocess
import argparse
import shlex
import shutil
from typing import Dict, List, Any, Optional

AUTOSCAFFOLD_VERSION = "2.1.0"

_SYSTEM_HARDWARE_CACHE: Optional[Dict[str, str]] = None


def get_system_hardware_info() -> Dict[str, str]:
    """
    Collects system hardware details: OS, CPU, RAM, GPU, CUDA driver, active conda env.
    Caches inspected system hardware details in memory to avoid redundant shell and subprocess queries.
    """
    global _SYSTEM_HARDWARE_CACHE
    if _SYSTEM_HARDWARE_CACHE is not None:
        return _SYSTEM_HARDWARE_CACHE.copy()

    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": platform.processor() or platform.machine(),
        "ram": "Unknown",
        "gpu": "None / Not Detected",
        "cuda_driver": "N/A",
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "base/none"),
        "python_version": platform.python_version()
    }

    # RAM detection via psutil if available
    try:
        import psutil
        total_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        info["ram"] = f"{total_ram_gb} GB"
    except Exception:
        pass

    # GPU / CUDA detection via nvidia-smi with strict 1.5s timeout to avoid stalls
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            timeout=1.5
        )
        if res.returncode == 0 and res.stdout.strip():
            lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
            gpu_list = []
            driver_v = ""
            for gpu_line in lines:
                parts = [p.strip() for p in gpu_line.split(",")]
                if len(parts) >= 3:
                    gpu_name, mem_total, driver_v = parts[0], parts[1], parts[2]
                    gpu_list.append(f"{gpu_name} ({mem_total} MiB)")
                elif len(parts) == 1:
                    gpu_list.append(parts[0])
            info["gpu"] = "; ".join(gpu_list) if gpu_list else "Unknown NVIDIA GPU"
            info["cuda_driver"] = driver_v or "Detected"
    except Exception:
        # Fallback via torch if available
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
                info["cuda_driver"] = f"PyTorch CUDA {torch.version.cuda}"
        except Exception:
            pass

    _SYSTEM_HARDWARE_CACHE = info
    return info.copy()


def generate_methodology_report(
    run_name: str,
    pipeline_mode: str,
    output_dir: str,
    input_structures: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    methodology_steps: Optional[List[str]] = None,
    candidates_summary: Optional[Dict[str, Any]] = None,
    reproducibility_cmd: Optional[str] = None,
    extra_notes: Optional[str] = None
) -> str:
    """
    Builds the methodology report string, writes it to run_methodology.txt and methodology.txt
    inside output_dir, and returns the formatted report text.
    """
    clean_output_dir = os.path.realpath(output_dir)
    if clean_output_dir in ("/", "\\") or (len(clean_output_dir) <= 3 and clean_output_dir.endswith(":\\")):
        raise ValueError(f"Invalid or root output directory specified: '{output_dir}'")
    os.makedirs(clean_output_dir, exist_ok=True)
    sys_info = get_system_hardware_info()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    input_structures = input_structures or {}
    parameters = parameters or {}
    candidates_summary = candidates_summary or {}

    # Default methodology steps per pipeline mode if not supplied
    if not methodology_steps:
        if "Chimera" in pipeline_mode:
            methodology_steps = [
                "1. Structural Extraction & Atom Parsing: Parsed donor motif coordinates and vessel host framework from source files (PDB/PSE).",
                "2. Coordinate Seeding & Merging: Merged preserved donor motif atoms and preserved vessel framework into a single SE(3) coordinate seed.",
                "3. Backbone Generation (RFdiffusion): Executed SE(3) equivariant diffusion with coordinate pinning (denoiser.noise_scale_ca=0.0, noise_scale_frame=0.0) to synthesize de novo linkers while preserving native donor motif and vessel coordinates.",
                "4. Sequence Design (ProteinMPNN): Redesigned de novo linkers and connecting junctions. Pinned donor motif residue identities and locked vessel framework residue positions in fixed_positions_dict.jsonl.",
                "5. Structural Validation: Evaluated generated chimeras using Kabsch C-alpha superposition against native donor coordinates (RMSD threshold) and assessed folding quality/pLDDT."
            ]
        elif "Discontinuous" in pipeline_mode:
            methodology_steps = [
                "1. Discontinuous Motif Extraction: Extracted non-contiguous catalytic / binding epitope segments and target context coordinates.",
                "2. Backbone Scaffolding (RFdiffusion): Diffused intervening loops and scaffold framework with SE(3) coordinate pinning (noise_scale=0.0) around fixed epitope residues.",
                "3. Sequence Optimization (ProteinMPNN): Locked native motif sequence identities; designed all scaffold and linker positions.",
                "4. Multimer Validation (ColabFold / AlphaFold2): Assessed complex interface prediction, ipTM, pLDDT, and calculated motif Kabsch RMSD."
            ]
        elif "Binder" in pipeline_mode or "Novo" in pipeline_mode:
            methodology_steps = [
                "1. Target Structural Ingestion: Extracted target chain coordinate frame and identified interaction hotspots.",
                "2. De Novo Diffusion (RFdiffusion): Generated target-binding backbones using SE(3) equivariant reverse diffusion guided by hotspot potentials.",
                "3. Sequence Design (ProteinMPNN): Fixed target chain sequence identities; sampled de novo sequence for the newly generated binder.",
                "4. Interface Validation (ColabFold / AlphaFold-Multimer): Predicted target-binder complex, filtering by interface pLDDT and ipTM."
            ]
        else:
            methodology_steps = [
                "1. Input Ingestion: Loaded and validated source structural models and motif segments.",
                "2. Backbone Scaffolding (RFdiffusion): Executed conditional generative diffusion to scaffold motifs.",
                "3. Sequence Design (ProteinMPNN): Designed sequences with fixed-position sequence identity masking.",
                "4. Structural Quality Assessment: Scored backbone and complex conformations."
            ]

    lines = []
    lines.append("================================================================================")
    lines.append("              🧬 AUTOSCAFFOLD: COMPUTATIONAL METHODOLOGY REPORT 🧬             ")
    lines.append("================================================================================")
    lines.append(f"Run Name:            {run_name}")
    lines.append(f"Timestamp:           {timestamp}")
    lines.append(f"AutoScaffold Version:{AUTOSCAFFOLD_VERSION}")
    lines.append(f"Pipeline Mode:       {pipeline_mode}")
    lines.append(f"Output Directory:    {os.path.abspath(output_dir)}")
    lines.append("")

    lines.append("--------------------------------------------------------------------------------")
    lines.append("1. EXECUTION ENVIRONMENT & HARDWARE ARCHITECTURE")
    lines.append("--------------------------------------------------------------------------------")
    lines.append(f"Operating System:    {sys_info['os']}")
    lines.append(f"CPU Architecture:    {sys_info['cpu']}")
    lines.append(f"Host Memory (RAM):   {sys_info['ram']}")
    lines.append(f"GPU Hardware:        {sys_info['gpu']}")
    lines.append(f"CUDA / Driver:       {sys_info['cuda_driver']}")
    lines.append(f"Active Conda Env:    {sys_info['conda_env']}")
    lines.append(f"Python Runtime:      {sys_info['python_version']}")
    lines.append("")

    lines.append("--------------------------------------------------------------------------------")
    lines.append("2. INPUT STRUCTURES & MOLECULAR DEFINITIONS")
    lines.append("--------------------------------------------------------------------------------")
    if input_structures:
        for k, v in input_structures.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- (No input structures specified)")
    lines.append("")

    lines.append("--------------------------------------------------------------------------------")
    lines.append("3. COMPUTATIONAL PARAMETERS")
    lines.append("--------------------------------------------------------------------------------")
    if parameters:
        for k, v in parameters.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- (Default parameters used)")
    lines.append("")

    lines.append("--------------------------------------------------------------------------------")
    lines.append("4. STEP-BY-STEP METHODOLOGY & ALGORITHMIC WORKFLOW")
    lines.append("--------------------------------------------------------------------------------")
    for step in methodology_steps:
        lines.append(step)
    lines.append("")

    lines.append("--------------------------------------------------------------------------------")
    lines.append("5. CANDIDATE DESIGN & VALIDATION SUMMARY")
    lines.append("--------------------------------------------------------------------------------")
    total_cand = candidates_summary.get("total_candidates", "N/A")
    pass_cand = candidates_summary.get("passing_candidates", "N/A")
    status = candidates_summary.get("status", "COMPLETED")
    lines.append(f"Execution Status:    {status}")
    lines.append(f"Total Candidates:    {total_cand}")
    lines.append(f"Passing Candidates:  {pass_cand}")

    passing_list = candidates_summary.get("passing_models", [])
    if passing_list:
        lines.append("Passing Structural Models:")
        for m in passing_list:
            lines.append(f"  • {m}")
    elif pass_cand == 0:
        lines.append("Passing Structural Models: None met the stringent filter criteria.")

    metrics_csv = candidates_summary.get("metrics_summary_csv")
    if metrics_csv:
        lines.append(f"Metrics CSV:         {metrics_csv}")
    lines.append("")

    if extra_notes:
        lines.append("--------------------------------------------------------------------------------")
        lines.append("6. ADDITIONAL METHODOLOGICAL NOTES")
        lines.append("--------------------------------------------------------------------------------")
        lines.append(extra_notes.strip())
        lines.append("")

    lines.append("--------------------------------------------------------------------------------")
    lines.append("7. EXACT REPRODUCIBILITY CLI COMMAND")
    lines.append("--------------------------------------------------------------------------------")
    if reproducibility_cmd:
        lines.append(reproducibility_cmd.strip())
    else:
        lines.append(f"python scripts/chimera_scaffold.py --out_dir {shlex.quote(clean_output_dir)}")
    lines.append("================================================================================")
    lines.append("")

    report_content = "\n".join(lines)

    # Write run_methodology.txt and methodology.txt strictly within clean_output_dir
    file1 = os.path.realpath(os.path.join(clean_output_dir, "run_methodology.txt"))
    file2 = os.path.realpath(os.path.join(clean_output_dir, "methodology.txt"))

    # Validate no path traversal
    for target_f in (file1, file2):
        if not (target_f.startswith(clean_output_dir + os.sep) or target_f.startswith(clean_output_dir + "/")):
            raise ValueError(f"Path traversal detected for output file: {target_f}")

    BUFFER_SIZE = 64 * 1024  # 64 KB buffer for efficient sequential I/O
    with open(file1, "w", buffering=BUFFER_SIZE, encoding="utf-8") as f:
        f.write(report_content)

    # Symlink or copy to methodology.txt to avoid duplicate encoding and I/O overhead
    try:
        if os.path.lexists(file2):
            os.remove(file2)
        os.symlink(os.path.basename(file1), file2)
    except (OSError, NotImplementedError, AttributeError):
        shutil.copyfile(file1, file2)

    return report_content


def main():
    parser = argparse.ArgumentParser(description="AutoScaffold Methodology & Reproducibility Reporter")
    parser.add_argument("--run_name", default="AutoScaffold_Run", help="Name of the execution run")
    parser.add_argument("--mode", default="Motif Scaffolding", help="Pipeline Mode")
    parser.add_argument("--out_dir", "-o", default=".", help="Directory to save report")
    parser.add_argument("--input_pdb", help="Input PDB file path")
    parser.add_argument("--donor", help="Donor structure/chain/residue spec")
    parser.add_argument("--vessel", help="Vessel structure/chain spec")
    parser.add_argument("--contigs", help="RFdiffusion contig formula")
    parser.add_argument("--num_designs", type=int, default=10, help="Number of designs")
    parser.add_argument("--num_seqs", type=int, default=2, help="ProteinMPNN sequences per backbone")
    parser.add_argument("--mpnn_temp", type=float, default=0.1, help="MPNN temperature")
    parser.add_argument("--cmd", help="Exact reproducibility CLI command")
    parser.add_argument("--status", default="COMPLETED", help="Run status")
    parser.add_argument("--total_candidates", type=int, default=0, help="Total candidates generated")
    parser.add_argument("--passing_candidates", type=int, default=0, help="Total passing candidates")

    args = parser.parse_args()

    clean_out_dir = os.path.realpath(args.out_dir)
    if clean_out_dir in ("/", "\\") or (len(clean_out_dir) <= 3 and clean_out_dir.endswith(":\\")):
        raise ValueError(f"Invalid or root output directory specified: '{args.out_dir}'")

    input_structs = {}
    if args.input_pdb:
        input_structs["Input PDB"] = args.input_pdb
    if args.donor:
        input_structs["Donor / Motif"] = args.donor
    if args.vessel:
        input_structs["Vessel / Framework"] = args.vessel

    params = {
        "Number of Designs": args.num_designs,
        "ProteinMPNN Sequences": args.num_seqs,
        "ProteinMPNN Temperature": args.mpnn_temp
    }
    if args.contigs:
        params["Contig Formulation"] = args.contigs

    cand_summary = {
        "status": args.status,
        "total_candidates": args.total_candidates,
        "passing_candidates": args.passing_candidates
    }

    generate_methodology_report(
        run_name=args.run_name,
        pipeline_mode=args.mode,
        output_dir=clean_out_dir,
        input_structures=input_structs,
        parameters=params,
        candidates_summary=cand_summary,
        reproducibility_cmd=args.cmd
    )
    print(f"[+] Methodology report generated at: {os.path.join(clean_out_dir, 'run_methodology.txt')}")


if __name__ == "__main__":
    main()
